"""
RailBlock AI — Shadow Block Scheduler

This module schedules compatible maintenance jobs simultaneously inside
shared possession windows called Shadow Blocks.

It does NOT modify optimizer.py.

Flow:
    maintenance jobs
          ↓
    find shadow candidates
          ↓
    convert candidates into scheduling units
          ↓
    CP-SAT schedules units around trains
          ↓
    expand shadow units back into original maintenance jobs
"""

from collections import defaultdict
from copy import deepcopy

from ortools.sat.python import cp_model

from shadow_block import find_shadow_block_candidates


HORIZON = 1440

PRIORITY_WEIGHT = {
    "CRITICAL": 100,
    "HIGH": 50,
    "MEDIUM": 20,
    "LOW": 5,
}


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def _hhmm_to_minutes(value: str) -> int:
    hour, minute = value.split(":")
    return int(hour) * 60 + int(minute)


def _minutes_to_hhmm(minutes: int) -> str:
    minutes = max(0, min(minutes, HORIZON))

    if minutes == HORIZON:
        return "24:00"

    return f"{minutes // 60:02d}:{minutes % 60:02d}"


# ---------------------------------------------------------------------------
# Conflict audit
# ---------------------------------------------------------------------------

def _audit_shadow_conflicts(
    scheduled_blocks: list,
    trains: list,
) -> int:
    """
    Count real conflicts after scheduling.

    Jobs belonging to the SAME Shadow Block are intentionally allowed
    to overlap each other.

    Everything else must remain conflict-free.
    """

    conflicts = 0

    # Maintenance vs train
    for block in scheduled_blocks:

        block_start = block["start_minutes"]
        block_end = block["end_minutes"]

        for train in trains:

            if train["section"] != block["section"]:
                continue

            train_start = _hhmm_to_minutes(train["start"])
            train_end = _hhmm_to_minutes(train["end"])

            if (
                block_start < train_end
                and train_start < block_end
            ):
                conflicts += 1

    # Maintenance vs maintenance
    for i, block1 in enumerate(scheduled_blocks):

        for block2 in scheduled_blocks[i + 1:]:

            if block1["section"] != block2["section"]:
                continue

            shadow1 = block1.get("shadow_block_id")
            shadow2 = block2.get("shadow_block_id")

            # Members of the SAME Shadow Block are supposed
            # to execute simultaneously.
            if (
                shadow1 is not None
                and shadow1 == shadow2
            ):
                continue

            if (
                block1["start_minutes"] < block2["end_minutes"]
                and block2["start_minutes"] < block1["end_minutes"]
            ):
                conflicts += 1

    return conflicts


# ---------------------------------------------------------------------------
# Build scheduling units
# ---------------------------------------------------------------------------

def _build_scheduling_units(data: dict):
    """
    Convert maintenance jobs into scheduling units.

    A unit is either:

    1. One normal maintenance job

    OR

    2. One Shadow Block containing 2–3 compatible jobs.

    Shadow Block duration = longest member duration.
    """

    candidate_result = find_shadow_block_candidates(data)

    candidates = candidate_result["shadow_block_candidates"]

    used_ids = set()

    units = []

    # ------------------------------------------------------------------
    # Shadow Block units
    # ------------------------------------------------------------------

    for candidate in candidates:

        members = deepcopy(candidate["member_jobs"])

        for member in members:
            used_ids.add(member["maintenance_id"])

        units.append({
            "unit_id": candidate["shadow_block_id"],
            "section": candidate["section"],
            "duration_minutes": candidate[
                "shadow_duration_minutes"
            ],
            "members": members,
            "is_shadow": True,
            "candidate": deepcopy(candidate),
        })

    # ------------------------------------------------------------------
    # Normal individual jobs
    # ------------------------------------------------------------------

    for job in data.get("maintenance_jobs", []):

        if job["maintenance_id"] in used_ids:
            continue

        units.append({
            "unit_id": job["maintenance_id"],
            "section": job["section"],
            "duration_minutes": int(
                job["duration_minutes"]
            ),
            "members": [deepcopy(job)],
            "is_shadow": False,
            "candidate": None,
        })

    return units, candidate_result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def schedule_with_shadow_blocks(data: dict) -> dict:
    """
    Schedule normal maintenance jobs and Shadow Blocks.

    Shadow Block members share the same start time.

    The shared possession duration is the duration of the longest
    maintenance job inside the block.

    Example:

        Engineering     60 min
        Signal          30 min
        Traction        45 min

    becomes:

        Shadow Block = 60 min

    instead of:

        60 + 30 + 45 = 135 min

    Returns:

    {
        "scheduled_blocks": [...],
        "unscheduled_jobs": [...],
        "shadow_blocks": [...],
        "shadow_block_candidates": [...],
        "metrics": {...}
    }
    """

    copied_data = deepcopy(data)

    trains = copied_data.get("trains", [])

    units, candidate_result = _build_scheduling_units(
        copied_data
    )

    if not units:

        return {
            "scheduled_blocks": [],
            "unscheduled_jobs": [],
            "shadow_blocks": [],
            "shadow_block_candidates": [],
            "metrics": {
                "total_jobs": 0,
                "scheduled": 0,
                "unscheduled": 0,
                "shadow_blocks_scheduled": 0,
                "conflicts": 0,
                "time_saved_minutes": 0,
            },
        }

    # ------------------------------------------------------------------
    # Train lookup
    # ------------------------------------------------------------------

    train_windows = defaultdict(list)

    for index, train in enumerate(trains):

        train_windows[train["section"]].append({
            "index": index,
            "train_id": train.get(
                "train_id",
                f"T{index}"
            ),
            "start": _hhmm_to_minutes(
                train["start"]
            ),
            "end": _hhmm_to_minutes(
                train["end"]
            ),
        })

    # ------------------------------------------------------------------
    # CP-SAT model
    # ------------------------------------------------------------------

    model = cp_model.CpModel()

    scheduled_var = []
    start_var = []
    end_var = []
    interval_var = []

    for i, unit in enumerate(units):

        duration = int(
            unit["duration_minutes"]
        )

        unit_id = unit["unit_id"]

        scheduled = model.NewBoolVar(
            f"scheduled_{unit_id}"
        )

        start = model.NewIntVar(
            0,
            HORIZON - duration,
            f"start_{unit_id}",
        )

        end = model.NewIntVar(
            duration,
            HORIZON,
            f"end_{unit_id}",
        )

        interval = model.NewOptionalIntervalVar(
            start,
            duration,
            end,
            scheduled,
            f"interval_{unit_id}",
        )

        model.Add(
            end == start + duration
        ).OnlyEnforceIf(
            scheduled
        )

        # Anchor unused variables.
        model.Add(
            start == 0
        ).OnlyEnforceIf(
            scheduled.Not()
        )

        model.Add(
            end == duration
        ).OnlyEnforceIf(
            scheduled.Not()
        )

        scheduled_var.append(scheduled)
        start_var.append(start)
        end_var.append(end)
        interval_var.append(interval)

    # ------------------------------------------------------------------
    # HARD CONSTRAINT 1
    # Maintenance / Shadow Blocks must avoid trains.
    #
    # Train observations are NEVER placed together in AddNoOverlap,
    # because overlapping trains are permitted observations.
    # ------------------------------------------------------------------

    for i, unit in enumerate(units):

        section = unit["section"]

        for train in train_windows.get(
            section,
            [],
        ):

            before_train = model.NewBoolVar(
                f"before_{unit['unit_id']}_{train['index']}"
            )

            after_train = model.NewBoolVar(
                f"after_{unit['unit_id']}_{train['index']}"
            )

            model.Add(
                end_var[i] <= train["start"]
            ).OnlyEnforceIf(
                before_train
            )

            model.Add(
                start_var[i] >= train["end"]
            ).OnlyEnforceIf(
                after_train
            )

            model.AddBoolOr([
                before_train,
                after_train,
                scheduled_var[i].Not(),
            ])

    # ------------------------------------------------------------------
    # HARD CONSTRAINT 2
    # Different scheduling units cannot overlap on the same section.
    #
    # Members INSIDE a Shadow Block are represented by one shared interval,
    # so they are intentionally simultaneous.
    # ------------------------------------------------------------------

    intervals_by_section = defaultdict(list)

    for i, unit in enumerate(units):

        intervals_by_section[
            unit["section"]
        ].append(
            interval_var[i]
        )

    for intervals in intervals_by_section.values():

        if len(intervals) > 1:
            model.AddNoOverlap(intervals)

    # ------------------------------------------------------------------
    # DEADLINE FLAGS
    #
    # Each member receives its OWN deadline check even when it belongs
    # to a Shadow Block.
    # ------------------------------------------------------------------

    deadline_flags = []

    for i, unit in enumerate(units):

        for member in unit["members"]:

            deadline = _hhmm_to_minutes(
                member.get(
                    "deadline",
                    "24:00",
                )
            )

            member_duration = int(
                member["duration_minutes"]
            )

            deadline_met = model.NewBoolVar(
                "deadline_"
                + unit["unit_id"]
                + "_"
                + member["maintenance_id"]
            )

            # Deadline credit only if unit is scheduled.
            model.Add(
                deadline_met <= scheduled_var[i]
            )

            # If deadline_met = 1:
            # member finishes before deadline.
            model.Add(
                start_var[i]
                + member_duration
                <= deadline
            ).OnlyEnforceIf(
                deadline_met
            )

            # If scheduled but deadline_met = 0,
            # it must genuinely be late.
            model.Add(
                start_var[i]
                + member_duration
                > deadline
            ).OnlyEnforceIf([
                scheduled_var[i],
                deadline_met.Not(),
            ])

            deadline_flags.append(
                deadline_met
            )

    # ==================================================================
    # LEXICOGRAPHIC OBJECTIVES
    # ==================================================================

    # ------------------------------------------------------------------
    # Stage 1
    # Number of CRITICAL member jobs scheduled.
    #
    # A Shadow Block containing 2 CRITICAL jobs contributes 2,
    # not merely 1.
    # ------------------------------------------------------------------

    critical_expr = sum(

        sum(
            1
            for member in unit["members"]
            if member.get("priority") == "CRITICAL"
        )
        * scheduled_var[i]

        for i, unit in enumerate(units)
    )

    # ------------------------------------------------------------------
    # Stage 2
    # Weighted maintenance work.
    #
    # Uses ORIGINAL member durations — not Shadow Block duration.
    # ------------------------------------------------------------------

    weighted_work_expr = sum(

        sum(
            PRIORITY_WEIGHT.get(
                member.get(
                    "priority",
                    "LOW",
                ),
                5,
            )
            * int(
                member["duration_minutes"]
            )

            for member in unit["members"]
        )
        * scheduled_var[i]

        for i, unit in enumerate(units)
    )

    # ------------------------------------------------------------------
    # Stage 3
    # Total ORIGINAL maintenance jobs scheduled.
    # ------------------------------------------------------------------

    total_jobs_expr = sum(

        len(unit["members"])
        * scheduled_var[i]

        for i, unit in enumerate(units)
    )

    # ------------------------------------------------------------------
    # Stage 4
    # AI / deadline urgency.
    #
    # Sum original member urgency scores.
    # ------------------------------------------------------------------

    urgency_expr = sum(

        sum(
            int(
                member.get(
                    "urgency_score",
                    0,
                )
            )

            for member in unit["members"]
        )
        * scheduled_var[i]

        for i, unit in enumerate(units)
    )

    # ------------------------------------------------------------------
    # Stage 5
    # Original jobs that actually meet their deadlines.
    # ------------------------------------------------------------------

    deadline_expr = sum(
        deadline_flags
    )

    # ------------------------------------------------------------------
    # Stage 6
    # Minimize total member completion time.
    #
    # Members inside a shadow block all start together but may have
    # different durations.
    # ------------------------------------------------------------------

    completion_expr = sum(

        (
            len(unit["members"])
            * start_var[i]
        )
        +
        (
            sum(
                int(member["duration_minutes"])
                for member in unit["members"]
            )
            * scheduled_var[i]
        )

        for i, unit in enumerate(units)
    )

    # ------------------------------------------------------------------
    # Solve Stages 1–5
    # ------------------------------------------------------------------

    solver = cp_model.CpSolver()

    solver.parameters.max_time_in_seconds = 30.0

    stage_results = {}

    stages = [
        ("critical", critical_expr),
        ("weighted_work", weighted_work_expr),
        ("total_jobs", total_jobs_expr),
        ("urgency", urgency_expr),
        ("deadline", deadline_expr),
    ]

    all_higher_stages_optimal = True

    final_status = None

    for stage_name, expression in stages:

        model.Maximize(expression)

        status = solver.Solve(model)

        final_status = status

        stage_results[stage_name] = (
            solver.StatusName(status)
        )

        # A lower priority stage must NEVER run unless
        # this higher priority objective is proven optimal.
        if status != cp_model.OPTIMAL:

            all_higher_stages_optimal = False

            break

        # Exact integer optimum.
        if isinstance(expression, int):
            optimum = expression
        else:
            optimum = solver.Value(
                expression
            )

        # Lock the proven optimum.
        model.Add(
            expression == optimum
        )

    # ------------------------------------------------------------------
    # Stage 6
    # Only run if Stages 1–5 were proven OPTIMAL.
    # ------------------------------------------------------------------

    if all_higher_stages_optimal:

        model.Minimize(
            completion_expr
        )

        final_status = solver.Solve(
            model
        )

        stage_results[
            "completion"
        ] = solver.StatusName(
            final_status
        )

    # ------------------------------------------------------------------
    # Extract results
    # ------------------------------------------------------------------

    scheduled_blocks = []

    unscheduled_jobs = []

    shadow_blocks = []

    scheduled_unit_indexes = set()

    if final_status in (
        cp_model.OPTIMAL,
        cp_model.FEASIBLE,
    ):

        for i, unit in enumerate(units):

            if solver.Value(
                scheduled_var[i]
            ) != 1:

                for member in unit["members"]:

                    unscheduled = deepcopy(
                        member
                    )

                    unscheduled[
                        "scheduled"
                    ] = False

                    if unit["is_shadow"]:
                        unscheduled[
                            "shadow_block_candidate_id"
                        ] = unit["unit_id"]

                    unscheduled_jobs.append(
                        unscheduled
                    )

                continue

            scheduled_unit_indexes.add(i)

            start_minutes = solver.Value(
                start_var[i]
            )

            unit_end_minutes = solver.Value(
                end_var[i]
            )

            # ----------------------------------------------------------
            # Shadow Block output
            # ----------------------------------------------------------

            if unit["is_shadow"]:

                candidate = unit[
                    "candidate"
                ]

                shadow_blocks.append({
                    "shadow_block_id":
                        unit["unit_id"],

                    "section":
                        unit["section"],

                    "start":
                        _minutes_to_hhmm(
                            start_minutes
                        ),

                    "end":
                        _minutes_to_hhmm(
                            unit_end_minutes
                        ),

                    "start_minutes":
                        start_minutes,

                    "end_minutes":
                        unit_end_minutes,

                    "duration_minutes":
                        unit[
                            "duration_minutes"
                        ],

                    "maintenance_ids": [
                        member[
                            "maintenance_id"
                        ]
                        for member
                        in unit["members"]
                    ],

                    "departments": [
                        member[
                            "department"
                        ]
                        for member
                        in unit["members"]
                    ],

                    "potential_time_saved_minutes":
                        candidate[
                            "potential_time_saved_minutes"
                        ],

                    "scheduled":
                        True,
                })

            # ----------------------------------------------------------
            # Expand the unit back into ORIGINAL maintenance jobs.
            # ----------------------------------------------------------

            for member in unit["members"]:

                member_duration = int(
                    member[
                        "duration_minutes"
                    ]
                )

                member_end = (
                    start_minutes
                    + member_duration
                )

                block = deepcopy(
                    member
                )

                block[
                    "start"
                ] = _minutes_to_hhmm(
                    start_minutes
                )

                block[
                    "end"
                ] = _minutes_to_hhmm(
                    member_end
                )

                block[
                    "start_minutes"
                ] = start_minutes

                block[
                    "end_minutes"
                ] = member_end

                block[
                    "scheduled"
                ] = True

                if unit["is_shadow"]:

                    block[
                        "shadow_block_id"
                    ] = unit["unit_id"]

                    block[
                        "shadowed"
                    ] = True

                else:

                    block[
                        "shadowed"
                    ] = False

                scheduled_blocks.append(
                    block
                )

    else:

        for unit in units:

            for member in unit["members"]:

                unscheduled = deepcopy(
                    member
                )

                unscheduled[
                    "scheduled"
                ] = False

                unscheduled_jobs.append(
                    unscheduled
                )

    # ------------------------------------------------------------------
    # Sort outputs
    # ------------------------------------------------------------------

    scheduled_blocks.sort(
        key=lambda block: (
            block["section"],
            block["start_minutes"],
            block["maintenance_id"],
        )
    )

    shadow_blocks.sort(
        key=lambda block: (
            block["section"],
            block["start_minutes"],
            block["shadow_block_id"],
        )
    )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    actual_conflicts = (
        _audit_shadow_conflicts(
            scheduled_blocks,
            trains,
        )
    )

    scheduled_shadow_ids = {
        block["shadow_block_id"]
        for block in shadow_blocks
    }

    time_saved = sum(
        candidate[
            "potential_time_saved_minutes"
        ]
        for candidate
        in candidate_result[
            "shadow_block_candidates"
        ]
        if candidate[
            "shadow_block_id"
        ]
        in scheduled_shadow_ids
    )

    possession_without_stacking = sum(
        int(
            block[
                "duration_minutes"
            ]
        )
        for block in scheduled_blocks
    )

    possession_with_stacking = 0

    for i in scheduled_unit_indexes:

        possession_with_stacking += int(
            units[i][
                "duration_minutes"
            ]
        )

    metrics = {
        "total_jobs":
            len(
                copied_data.get(
                    "maintenance_jobs",
                    [],
                )
            ),

        "scheduled":
            len(
                scheduled_blocks
            ),

        "unscheduled":
            len(
                unscheduled_jobs
            ),

        "critical_jobs_scheduled":
            sum(
                block.get(
                    "priority"
                )
                == "CRITICAL"
                for block
                in scheduled_blocks
            ),

        "shadow_block_candidates":
            candidate_result[
                "candidate_count"
            ],

        "shadow_blocks_scheduled":
            len(
                shadow_blocks
            ),

        "possession_minutes_without_stacking":
            possession_without_stacking,

        "possession_minutes_with_stacking":
            possession_with_stacking,

        "time_saved_minutes":
            time_saved,

        "conflicts":
            actual_conflicts,

        "solver_status":
            (
                solver.StatusName(
                    final_status
                )
                if final_status
                is not None
                else "UNKNOWN"
            ),

        "stage_statuses":
            stage_results,
    }

    return {
        "scheduled_blocks":
            scheduled_blocks,

        "unscheduled_jobs":
            unscheduled_jobs,

        "shadow_blocks":
            shadow_blocks,

        "shadow_block_candidates":
            candidate_result[
                "shadow_block_candidates"
            ],

        "metrics":
            metrics,
    }