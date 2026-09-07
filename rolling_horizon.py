"""
RailBlock AI — Rolling Horizon Recovery

Purpose
-------
Recover a maintenance plan when something changes after the original
schedule has already been produced.

Examples:
    - a train is delayed
    - a new train movement appears
    - an emergency maintenance request arrives
    - a temporary disruption changes section availability

IMPORTANT:
This is a planning/recovery simulation only.
It does not control trains, signals, KAVACH, or railway infrastructure.

Core idea
---------
Instead of recalculating the entire network:

    Previous Schedule
           ↓
    Detect affected section + time window
           ↓
    LOCK unaffected maintenance blocks
           ↓
    Re-optimize only affected jobs
           ↓
    Merge locked + recovered schedule
"""

from copy import deepcopy

from shadow_scheduler import schedule_with_shadow_blocks


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

HORIZON = 1440


def _hhmm_to_minutes(value: str) -> int:
    """Convert HH:MM to minutes since midnight."""

    hour, minute = value.split(":")
    return int(hour) * 60 + int(minute)


def _minutes_to_hhmm(minutes: int) -> str:
    """Convert minutes since midnight back to HH:MM."""

    minutes = max(0, min(minutes, HORIZON))

    if minutes == HORIZON:
        return "24:00"

    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _block_start(block: dict) -> int:
    """Read a block's start time."""

    if "start_minutes" in block:
        return int(block["start_minutes"])

    return _hhmm_to_minutes(block["start"])


def _block_end(block: dict) -> int:
    """Read a block's end time."""

    if "end_minutes" in block:
        return int(block["end_minutes"])

    return _hhmm_to_minutes(block["end"])


def _overlaps(
    start1: int,
    end1: int,
    start2: int,
    end2: int,
) -> bool:
    """
    Return True when two half-open intervals overlap.

    Example:

        10:00–11:00
        10:30–11:30

    overlap.

    But:

        10:00–11:00
        11:00–12:00

    do not overlap.
    """

    return start1 < end2 and start2 < end1


# ---------------------------------------------------------------------------
# Conflict audit
# ---------------------------------------------------------------------------

def _audit_recovered_schedule(
    scheduled_blocks: list,
    trains: list,
) -> int:
    """
    Audit the FINAL recovered schedule.

    Checks:
        1. maintenance vs train conflicts
        2. maintenance vs maintenance conflicts

    Members of the SAME Shadow Block are intentionally allowed to overlap.
    """

    conflicts = 0

    # ---------------------------------------------------------------
    # Maintenance vs trains
    # ---------------------------------------------------------------

    for block in scheduled_blocks:

        b_start = _block_start(block)
        b_end = _block_end(block)

        for train in trains:

            if train["section"] != block["section"]:
                continue

            t_start = _hhmm_to_minutes(train["start"])
            t_end = _hhmm_to_minutes(train["end"])

            if _overlaps(
                b_start,
                b_end,
                t_start,
                t_end,
            ):
                conflicts += 1

    # ---------------------------------------------------------------
    # Maintenance vs maintenance
    # ---------------------------------------------------------------

    for i, block1 in enumerate(scheduled_blocks):

        for block2 in scheduled_blocks[i + 1:]:

            if block1["section"] != block2["section"]:
                continue

            shadow1 = block1.get("shadow_block_id")
            shadow2 = block2.get("shadow_block_id")

            # Same Shadow Block = intentional simultaneous maintenance.
            if (
                shadow1 is not None
                and shadow1 == shadow2
            ):
                continue

            if _overlaps(
                _block_start(block1),
                _block_end(block1),
                _block_start(block2),
                _block_end(block2),
            ):
                conflicts += 1

    return conflicts


# ---------------------------------------------------------------------------
# Shadow Block ID helper
# ---------------------------------------------------------------------------

def _rename_recovered_shadow_blocks(
    recovery_result: dict,
    reserved_shadow_ids: set,
) -> dict:
    """
    Give newly recovered Shadow Blocks unique IDs.

    Why?
    ----
    find_shadow_block_candidates() starts IDs from SB001 every time.

    Suppose the locked part of the old schedule already contains SB001.
    A new recovery run might also create SB001.

    We therefore rename recovered blocks:

        SB001 -> RSB001
        SB002 -> RSB002

    while preserving the relationship between member jobs.
    """

    result = deepcopy(recovery_result)

    id_mapping = {}

    counter = 1

    for shadow in result.get("shadow_blocks", []):

        old_id = shadow["shadow_block_id"]

        while True:

            new_id = f"RSB{counter:03d}"
            counter += 1

            if (
                new_id not in reserved_shadow_ids
                and new_id not in id_mapping.values()
            ):
                break

        id_mapping[old_id] = new_id

        shadow["shadow_block_id"] = new_id

    # Update expanded maintenance blocks.
    for block in result.get("scheduled_blocks", []):

        old_id = block.get("shadow_block_id")

        if old_id in id_mapping:
            block["shadow_block_id"] = id_mapping[old_id]

    # Update candidate information too.
    for candidate in result.get(
        "shadow_block_candidates",
        [],
    ):

        old_id = candidate.get("shadow_block_id")

        if old_id in id_mapping:
            candidate["shadow_block_id"] = id_mapping[old_id]

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def recover_schedule(
    previous_result: dict,
    updated_data: dict,
    disruption: dict,
) -> dict:
    """
    Perform Rolling Horizon Recovery.

    Parameters
    ----------
    previous_result:
        Previous output from schedule_with_shadow_blocks().

    updated_data:
        Current railway planning input.

        It should contain the CURRENT trains and maintenance jobs after
        the disruption has occurred.

    disruption:
        Description of the affected planning area.

        Example:

        {
            "type": "TRAIN_DELAY",
            "section": "Delhi-Ghaziabad",
            "start": "11:00",
            "end": "13:00",
            "description": "T009 delayed into maintenance window"
        }

    Meaning:
        Only maintenance in the specified SECTION and TIME WINDOW
        is considered affected.

        Everything outside this area is preserved whenever possible.

    Returns
    -------
    {
        "recovered_schedule": [...],
        "locked_blocks": [...],
        "reoptimized_blocks": [...],
        "unscheduled_jobs": [...],
        "shadow_blocks": [...],
        "changes": {...},
        "recovery": {...},
        "metrics": {...}
    }
    """

    previous = deepcopy(previous_result)
    current_data = deepcopy(updated_data)
    disruption_copy = deepcopy(disruption)

    # ------------------------------------------------------------------
    # Validate disruption
    # ------------------------------------------------------------------

    for field in ("section", "start", "end"):

        if field not in disruption_copy:
            raise ValueError(
                f"disruption must contain '{field}'"
            )

    affected_section = disruption_copy["section"]

    window_start = _hhmm_to_minutes(
        disruption_copy["start"]
    )

    window_end = _hhmm_to_minutes(
        disruption_copy["end"]
    )

    if window_end <= window_start:
        raise ValueError(
            "disruption end must be after disruption start"
        )

    previous_blocks = previous.get(
        "scheduled_blocks",
        [],
    )

    previous_shadow_blocks = previous.get(
        "shadow_blocks",
        [],
    )

    # ------------------------------------------------------------------
    # STEP 1
    # Find Shadow Blocks touched by the disruption.
    #
    # If ANY member of a Shadow Block is affected,
    # the WHOLE Shadow Block must be recovered together.
    # ------------------------------------------------------------------

    affected_shadow_ids = set()

    for block in previous_blocks:

        if block.get("section") != affected_section:
            continue

        if _overlaps(
            _block_start(block),
            _block_end(block),
            window_start,
            window_end,
        ):

            shadow_id = block.get(
                "shadow_block_id"
            )

            if shadow_id is not None:
                affected_shadow_ids.add(
                    shadow_id
                )

    # ------------------------------------------------------------------
    # STEP 2
    # Separate LOCKED blocks from affected blocks.
    # ------------------------------------------------------------------

    locked_blocks = []

    affected_previous_ids = set()

    for block in previous_blocks:

        block_is_affected = False

        # Same-section time overlap.
        if block.get("section") == affected_section:

            if _overlaps(
                _block_start(block),
                _block_end(block),
                window_start,
                window_end,
            ):
                block_is_affected = True

        # Whole Shadow Block must move together.
        if (
            block.get("shadow_block_id")
            in affected_shadow_ids
        ):
            block_is_affected = True

        if block_is_affected:

            affected_previous_ids.add(
                block["maintenance_id"]
            )

        else:

            locked_blocks.append(
                deepcopy(block)
            )

    # ------------------------------------------------------------------
    # STEP 3
    # Select jobs that should be re-optimized.
    #
    # Include:
    #   A. jobs previously affected
    #   B. NEW maintenance jobs on affected section
    #
    # Jobs outside the affected area remain locked.
    # ------------------------------------------------------------------

    previous_scheduled_ids = {
        block["maintenance_id"]
        for block in previous_blocks
    }

    locked_ids = {
        block["maintenance_id"]
        for block in locked_blocks
    }

    recovery_jobs = []

    for job in current_data.get(
        "maintenance_jobs",
        [],
    ):

        maintenance_id = job[
            "maintenance_id"
        ]

        if job.get("section") != affected_section:
            continue

        is_new_job = (
            maintenance_id
            not in previous_scheduled_ids
        )

        was_affected = (
            maintenance_id
            in affected_previous_ids
        )

        if (
            was_affected
            or is_new_job
        ):

            # Never accidentally re-optimize a locked job.
            if maintenance_id not in locked_ids:

                recovery_jobs.append(
                    deepcopy(job)
                )

    # ------------------------------------------------------------------
    # STEP 4
    # Convert LOCKED maintenance blocks into fixed occupancy windows.
    #
    # schedule_with_shadow_blocks() already knows how to avoid fixed train
    # windows.
    #
    # We therefore represent locked maintenance as synthetic occupancy
    # observations.
    #
    # This does NOT represent a real train.
    # It is only an internal planning constraint.
    # ------------------------------------------------------------------

    recovery_trains = deepcopy(
        current_data.get(
            "trains",
            [],
        )
    )
        # ------------------------------------------------------------------
    # Rolling-horizon cutoff
    #
    # Recovery must never move affected work into the past.
    # Everything before the disruption window is treated as already
    # committed history and is unavailable for re-optimization.
    # ------------------------------------------------------------------

    if window_start > 0:

        recovery_trains.append({
            "train_id": "RECOVERY_TIME_CUTOFF",
            "section": affected_section,
            "start": "00:00",
            "end": _minutes_to_hhmm(window_start),
            "planning_only": True,
        })

    for block in locked_blocks:

        if block.get("section") != affected_section:
            continue

        recovery_trains.append({
            "train_id":
                "LOCKED_MAINT_"
                + block["maintenance_id"],

            "section":
                block["section"],

            "start":
                block["start"],

            "end":
                block["end"],

            "planning_only":
                True,
        })

    # ------------------------------------------------------------------
    # STEP 5
    # Re-optimize only the affected jobs.
    # ------------------------------------------------------------------

    if recovery_jobs:

        recovery_input = {
            "trains":
                recovery_trains,

            "maintenance_jobs":
                recovery_jobs,

            "sections":
                deepcopy(
                    current_data.get(
                        "sections",
                        [],
                    )
                ),
        }

        recovery_result = (
            schedule_with_shadow_blocks(
                recovery_input
            )
        )

    else:

        recovery_result = {
            "scheduled_blocks": [],
            "unscheduled_jobs": [],
            "shadow_blocks": [],
            "shadow_block_candidates": [],
            "metrics": {
                "scheduled": 0,
                "unscheduled": 0,
                "conflicts": 0,
            },
        }

    # ------------------------------------------------------------------
    # STEP 6
    # Prevent recovered Shadow Block ID collisions.
    # ------------------------------------------------------------------

    locked_shadow_ids = {
        shadow["shadow_block_id"]
        for shadow in previous_shadow_blocks
        if shadow.get("shadow_block_id")
        not in affected_shadow_ids
    }

    recovery_result = (
        _rename_recovered_shadow_blocks(
            recovery_result,
            locked_shadow_ids,
        )
    )

    # ------------------------------------------------------------------
    # STEP 7
    # Merge locked + recovered maintenance.
    # ------------------------------------------------------------------

    recovered_blocks = deepcopy(
        recovery_result.get(
            "scheduled_blocks",
            [],
        )
    )

    recovered_schedule = (
        deepcopy(locked_blocks)
        + recovered_blocks
    )

    recovered_schedule.sort(
        key=lambda block: (
            block.get("section", ""),
            _block_start(block),
            block["maintenance_id"],
        )
    )

    # ------------------------------------------------------------------
    # STEP 8
    # Merge Shadow Block information.
    # ------------------------------------------------------------------

    locked_shadow_blocks = []

    for shadow in previous_shadow_blocks:

        shadow_id = shadow.get(
            "shadow_block_id"
        )

        if shadow_id not in affected_shadow_ids:

            locked_shadow_blocks.append(
                deepcopy(shadow)
            )

    recovered_shadow_blocks = deepcopy(
        recovery_result.get(
            "shadow_blocks",
            [],
        )
    )

    combined_shadow_blocks = (
        locked_shadow_blocks
        + recovered_shadow_blocks
    )

    # ------------------------------------------------------------------
    # STEP 9
    # Compare OLD vs NEW schedule.
    # ------------------------------------------------------------------

    previous_by_id = {
        block["maintenance_id"]: block
        for block in previous_blocks
    }

    recovered_by_id = {
        block["maintenance_id"]: block
        for block in recovered_schedule
    }

    moved = []

    unchanged = []

    newly_scheduled = []

    now_unscheduled = []

    # --------------------------------------------------------------
    # Previously scheduled jobs
    # --------------------------------------------------------------

    for maintenance_id, old_block in previous_by_id.items():

        new_block = recovered_by_id.get(
            maintenance_id
        )

        if new_block is None:

            # Only count it as recovery-unscheduled if it was
            # part of the affected area.
            if (
                maintenance_id
                in affected_previous_ids
            ):
                now_unscheduled.append(
                    maintenance_id
                )

            continue

        old_start = _block_start(
            old_block
        )

        old_end = _block_end(
            old_block
        )

        new_start = _block_start(
            new_block
        )

        new_end = _block_end(
            new_block
        )

        if (
            old_start == new_start
            and old_end == new_end
        ):

            unchanged.append(
                maintenance_id
            )

        else:

            moved.append({
                "maintenance_id":
                    maintenance_id,

                "old_start":
                    _minutes_to_hhmm(
                        old_start
                    ),

                "old_end":
                    _minutes_to_hhmm(
                        old_end
                    ),

                "new_start":
                    _minutes_to_hhmm(
                        new_start
                    ),

                "new_end":
                    _minutes_to_hhmm(
                        new_end
                    ),
            })

    # --------------------------------------------------------------
    # Newly scheduled jobs
    # --------------------------------------------------------------

    for maintenance_id in recovered_by_id:

        if maintenance_id not in previous_by_id:

            newly_scheduled.append(
                maintenance_id
            )

    # ------------------------------------------------------------------
    # STEP 10
    # Final safety/conflict audit.
    # ------------------------------------------------------------------

    actual_conflicts = (
        _audit_recovered_schedule(
            recovered_schedule,
            current_data.get(
                "trains",
                [],
            ),
        )
    )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    metrics = {
        "previous_scheduled_jobs":
            len(previous_blocks),

        "recovered_scheduled_jobs":
            len(recovered_schedule),

        "locked_jobs":
            len(locked_blocks),

        "affected_previous_jobs":
            len(affected_previous_ids),

        "reoptimized_jobs":
            len(recovered_blocks),

        "jobs_moved":
            len(moved),

        "jobs_unchanged":
            len(unchanged),

        "jobs_newly_scheduled":
            len(newly_scheduled),

        "jobs_now_unscheduled":
            len(now_unscheduled),

        "recovery_conflicts":
            actual_conflicts,

        "shadow_blocks_after_recovery":
            len(combined_shadow_blocks),
    }

    # ------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------

    return {
        "recovered_schedule":
            recovered_schedule,

        "locked_blocks":
            locked_blocks,

        "reoptimized_blocks":
            recovered_blocks,

        "unscheduled_jobs":
            deepcopy(
                recovery_result.get(
                    "unscheduled_jobs",
                    [],
                )
            ),

        "shadow_blocks":
            combined_shadow_blocks,

        "recovered_shadow_blocks":
            recovered_shadow_blocks,

        "changes": {
            "moved":
                moved,

            "unchanged":
                unchanged,

            "newly_scheduled":
                newly_scheduled,

            "now_unscheduled":
                now_unscheduled,
        },

        "recovery": {
            "type":
                disruption_copy.get(
                    "type",
                    "UNKNOWN",
                ),

            "section":
                affected_section,

            "window_start":
                _minutes_to_hhmm(
                    window_start
                ),

            "window_end":
                _minutes_to_hhmm(
                    window_end
                ),

            "description":
                disruption_copy.get(
                    "description",
                    "",
                ),
        },

        "metrics":
            metrics,
    }