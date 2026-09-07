"""
optimizer.py
============
RailBlock AI — CP-SAT Block Schedule Optimizer

What this module does
─────────────────────
Given validated, AI-enriched, and urgency-scored maintenance jobs and
a list of fixed train movements, this module uses Google OR-Tools CP-SAT
to find a *conflict-free* maintenance schedule inside a 24-hour window.

CP-SAT = Constraint Programming - Satisfiability solver.  It is a
constraint solver: we describe WHAT must be true (constraints) rather
than HOW to search for a solution.  CP-SAT then finds the best solution
automatically.

Why STAGED (lexicographic) optimization?
────────────────────────────────────────
A simple single-objective like "maximize urgency score" can produce
surprising results — for example, favouring lots of LOW-priority jobs
with high AI risk over one CRITICAL job.

Instead we solve the problem in 6 stages, fixing the optimum at each
stage before moving to the next goal.  This guarantees a strict priority
hierarchy:

    STAGE 1 — Maximize CRITICAL jobs scheduled  (safety first)
    STAGE 2 — Maximize weighted work  (priority × duration)
    STAGE 3 — Maximize total jobs scheduled
    STAGE 4 — Maximize total urgency score  (AI influence enters here)
    STAGE 5 — Maximize jobs that finish before their deadline
    STAGE 6 — Minimize total completion time  (compact schedule)

Because AI risk only appears in Stage 4, it can NEVER override the
CRITICAL-first guarantee established in Stage 1.

Why overlapping train observations are allowed
──────────────────────────────────────────────
Train movements are *observations of reality*, not decisions.
Two trains can physically overlap on a section if they run on
different tracks (e.g., up-line vs. down-line) — the input records both.

If we forced trains to be non-overlapping among themselves we would
incorrectly reject valid input and make the model infeasible.

The rule is: maintenance must avoid EACH train separately.
We do NOT apply NoOverlap among trains themselves.

Why large objective coefficients are dangerous
──────────────────────────────────────────────
CP-SAT uses integer arithmetic internally.  A coefficient like
1_000_000 * 1440 minutes * 15 jobs = 2.16 × 10¹⁰ can overflow the
solver's 64-bit integer limit and produce wrong results or crashes.

Staged optimization avoids this by keeping each objective expression
small (max coefficient = duration_minutes ≈ 1440).

Planning horizon
────────────────
00:00 – 24:00, represented internally as 0 – 1440 minutes.
"""

import copy
from ortools.sat.python import cp_model

# ── Constants ─────────────────────────────────────────────────────────────────

HORIZON = 1440   # minutes in a day (00:00 to 24:00)

PRIORITY_WEIGHT = {
    "CRITICAL": 100,
    "HIGH": 50,
    "MEDIUM": 20,
    "LOW": 5,
}


# ── Time helpers ──────────────────────────────────────────────────────────────

def _hhmm_to_minutes(time_str: str) -> int:
    """Convert 'HH:MM' string to integer minutes since midnight."""
    h, m = time_str.split(":")
    return int(h) * 60 + int(m)


def _minutes_to_hhmm(minutes: int) -> str:
    """Convert integer minutes since midnight back to 'HH:MM' string."""
    minutes = max(0, min(minutes, HORIZON))
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


# ── Conflict audit ────────────────────────────────────────────────────────────

def _audit_conflicts(scheduled_blocks: list, trains: list) -> int:
    """
    Count actual conflicts in the produced schedule.

    A conflict is:
        (a) a scheduled maintenance block that overlaps a train on the same section, OR
        (b) two scheduled maintenance blocks that overlap on the same section.

    This function is called AFTER the solver produces a solution so that
    metrics["conflicts"] is a real audit result — never hardcoded.

    Two intervals [s1, e1) and [s2, e2) overlap when s1 < e2 and s2 < e1.
    """
    conflicts = 0

    # (a) Maintenance vs train conflicts
    for block in scheduled_blocks:
        b_start = _hhmm_to_minutes(block["start"])
        b_end = _hhmm_to_minutes(block["end"])
        for train in trains:
            if train["section"] != block["section"]:
                continue
            t_start = _hhmm_to_minutes(train["start"])
            t_end = _hhmm_to_minutes(train["end"])
            if b_start < t_end and t_start < b_end:
                conflicts += 1

    # (b) Maintenance vs maintenance conflicts on the same section
    for i, b1 in enumerate(scheduled_blocks):
        for b2 in scheduled_blocks[i + 1:]:
            if b1["section"] != b2["section"]:
                continue
            s1 = _hhmm_to_minutes(b1["start"])
            e1 = _hhmm_to_minutes(b1["end"])
            s2 = _hhmm_to_minutes(b2["start"])
            e2 = _hhmm_to_minutes(b2["end"])
            if s1 < e2 and s2 < e1:
                conflicts += 1

    return conflicts


# ── Public API ────────────────────────────────────────────────────────────────

def optimize_schedule(data: dict) -> dict:
    """
    Produce a conflict-free maintenance block schedule using CP-SAT.

    Parameters
    ----------
    data : dict
        Railway planning data containing at minimum:
            trains            – list of fixed train movement dicts
            maintenance_jobs  – list of maintenance job dicts
        Each maintenance job may optionally include 'urgency_score' and
        'predicted_risk_score'.  The optimizer works correctly without them
        (backward compatible — urgency_score defaults to 0).

    Returns
    -------
    dict with keys:
        scheduled_blocks   – list of successfully scheduled maintenance blocks
        unscheduled_jobs   – list of jobs that could not be scheduled
        metrics            – summary statistics including audited conflicts count
    """
    trains = data.get("trains", [])
    jobs = data.get("maintenance_jobs", [])

    if not jobs:
        return {
            "scheduled_blocks": [],
            "unscheduled_jobs": [],
            "metrics": {
                "total_jobs": 0,
                "scheduled": 0,
                "unscheduled": 0,
                "conflicts": 0,
            },
        }

    # ── Build train interval lookup by section ─────────────────────────────
    # Train movements are FIXED — they are not solver variables.
    # We store them as plain (start, end) tuples to block out time windows.
    train_intervals_by_section = {}
    for train in trains:
        section = train["section"]
        t_start = _hhmm_to_minutes(train["start"])
        t_end = _hhmm_to_minutes(train["end"])
        if section not in train_intervals_by_section:
            train_intervals_by_section[section] = []
        train_intervals_by_section[section].append((t_start, t_end))

    # ── CP-SAT model setup ─────────────────────────────────────────────────
    model = cp_model.CpModel()

    # For each maintenance job we create:
    #   is_scheduled[i]   – Boolean variable (1=scheduled, 0=not scheduled)
    #   start_var[i]      – integer start time in minutes
    #   end_var[i]        – integer end time in minutes
    #   interval_var[i]   – optional interval (only active when scheduled)
    is_scheduled = []
    start_var = []
    end_var = []
    interval_var = []

    for i, job in enumerate(jobs):
        duration = int(job["duration_minutes"])
        mid = job["maintenance_id"]

        # is_scheduled: 1 if this job gets a maintenance block.
        sched = model.NewBoolVar(f"sched_{mid}")

        # Start time: anywhere in [0, HORIZON - duration].
        # If the job is NOT scheduled, these variables are unconstrained but
        # the interval is inactive (optional interval).
        s = model.NewIntVar(0, HORIZON - duration, f"start_{mid}")
        e = model.NewIntVar(duration, HORIZON, f"end_{mid}")

        # Optional interval: active only when sched == 1.
        # end = start + duration is enforced automatically.
        iv = model.NewOptionalIntervalVar(s, duration, e, sched, f"iv_{mid}")

        # When the job IS scheduled, the end must equal start + duration.
        # (Optional intervals still need this explicit for end variable.)
        model.Add(e == s + duration).OnlyEnforceIf(sched)
        # When NOT scheduled, anchor start to 0 (avoids unbounded variables).
        model.Add(s == 0).OnlyEnforceIf(sched.Not())
        model.Add(e == duration).OnlyEnforceIf(sched.Not())

        is_scheduled.append(sched)
        start_var.append(s)
        end_var.append(e)
        interval_var.append(iv)

    # ── HARD CONSTRAINT 1: Maintenance must not overlap train movements ──────
    # For each scheduled maintenance interval, for every train on the same
    # section, we add a pair of constraints ensuring no overlap:
    #     either maintenance ends before the train starts,
    #     or maintenance starts after the train ends.
    #
    # Why NOT one global NoOverlap([maintenance, train])?
    # Because trains can legitimately overlap EACH OTHER (different tracks),
    # so mixing them into a shared NoOverlap would incorrectly forbid valid input.
    # We enforce maintenance-vs-train separation job by job.

    for i, job in enumerate(jobs):
        section = job["section"]
        train_windows = train_intervals_by_section.get(section, [])
        duration = int(job["duration_minutes"])

        for (t_start, t_end) in train_windows:
            # Boolean: does maintenance finish before this train starts?
            before_train = model.NewBoolVar(
                f"before_{job['maintenance_id']}_{t_start}"
            )
            # Boolean: does maintenance start after this train ends?
            after_train = model.NewBoolVar(
                f"after_{job['maintenance_id']}_{t_end}"
            )

            # If scheduled, maintenance must be either before OR after the train.
            model.Add(end_var[i] <= t_start).OnlyEnforceIf(before_train)
            model.Add(start_var[i] >= t_end).OnlyEnforceIf(after_train)

            # At least one of before_train or after_train must be true
            # when the job is scheduled.
            model.AddBoolOr([before_train, after_train, is_scheduled[i].Not()])

    # ── HARD CONSTRAINT 2: Same-section maintenance must not overlap ─────────
    # Group maintenance interval variables by section, then apply
    # AddNoOverlap to keep them mutually exclusive on each section.
    # (This is where CP-SAT's NoOverlap applies — only to maintenance, not trains.)
    section_to_intervals = {}
    for i, job in enumerate(jobs):
        section = job["section"]
        if section not in section_to_intervals:
            section_to_intervals[section] = []
        section_to_intervals[section].append(interval_var[i])

    for section, ivs in section_to_intervals.items():
        if len(ivs) > 1:
            model.AddNoOverlap(ivs)

    # ── HARD CONSTRAINT 3: All blocks must stay inside 24-hour horizon ───────
    # (Already enforced by variable domains: start ∈ [0, HORIZON - duration],
    #  end = start + duration ≤ HORIZON.)

    # ═══════════════════════════════════════════════════════════════════════
    #  STAGED / LEXICOGRAPHIC OPTIMIZATION
    #  ─────────────────────────────────────────────────────────────────────
    #  We solve 6 times, fixing the optimum at each stage before moving on.
    #  This avoids large coefficient overflow and enforces the correct
    #  CRITICAL-first safety hierarchy.
    # ═══════════════════════════════════════════════════════════════════════

    solver = cp_model.CpSolver()
    # Time limit per stage — keeps things snappy for real-time planning.
    solver.parameters.max_time_in_seconds = 30.0

    def _solve_maximize(objective_expr, label: str):
        """Maximize objective_expr and return (status, optimum_value).

        Returns the exact integer optimum only when the solver status is
        cp_model.OPTIMAL — a proven global optimum.  FEASIBLE is intentionally
        NOT treated as a proven optimum for an intermediate lexicographic
        stage: accepting a FEASIBLE (non-proven) value and locking it in as
        an equality constraint could make later stages infeasible if a better
        solution actually existed.

        The optimum is read via solver.Value() on the CP-SAT integer
        expression — NOT via solver.ObjectiveValue() which returns a float
        and can introduce rounding errors when converted to int.
        """
        model.Maximize(objective_expr)
        status = solver.Solve(model)
        if status == cp_model.OPTIMAL:
            # Use solver.Value() to obtain the exact proven integer optimum.
            # solver.ObjectiveValue() returns a float and is deliberately
            # avoided here to prevent silent rounding bugs.
            exact_int = solver.Value(objective_expr)
            return status, exact_int
        # FEASIBLE or any other non-OPTIMAL status: do not lock this stage.
        return status, None

    def _lock_stage(stage_expr, exact_optimum, label: str):
        """
        Pin a stage's objective to its proven optimum using equality (==).

        Using == rather than >= is intentional:
          * >= would allow later stages to inadvertently raise this value,
            which can cascade and over-constrain subsequent stages.
          * == precisely preserves the proven optimum while giving the next
            stage full freedom within that fixed constraint.

        Only called when exact_optimum is not None (i.e. status was OPTIMAL).
        """
        model.Add(stage_expr == exact_optimum)

    # Track the overall solver status. If an intermediate stage fails to prove
    # OPTIMAL (e.g. returns FEASIBLE on a timeout), we stop optimizing lower-priority
    # objectives to prevent them from sacrificing the unfinished higher-priority one.
    final_status = cp_model.UNKNOWN

    # ── Stage 1: Maximize CRITICAL jobs scheduled ──────────────────────────
    # Safety hierarchy: CRITICAL jobs get scheduled first, no matter what.
    critical_flags = [
        is_scheduled[i]
        for i, job in enumerate(jobs)
        if job.get("priority") == "CRITICAL"
    ]
    if critical_flags:
        stage1_expr = sum(critical_flags)
        final_status, opt1 = _solve_maximize(stage1_expr, "STAGE1-CRITICAL")
        if opt1 is not None:
            _lock_stage(stage1_expr, opt1, "STAGE1-CRITICAL")
    else:
        final_status = cp_model.OPTIMAL

    # ── Stage 2: Maximize weighted work (priority_weight × duration) ────────
    # Among solutions with the same CRITICAL count, prefer more high-value work.
    if final_status == cp_model.OPTIMAL:
        work_weights = []
        for i, job in enumerate(jobs):
            w = PRIORITY_WEIGHT.get(job.get("priority", "LOW"), 5)
            d = int(job["duration_minutes"])
            work_weights.append(w * d)

        stage2_expr = sum(int(w) * v for w, v in zip(work_weights, is_scheduled))
        final_status, opt2 = _solve_maximize(stage2_expr, "STAGE2-WEIGHTED-WORK")
        if opt2 is not None:
            _lock_stage(stage2_expr, opt2, "STAGE2-WEIGHTED-WORK")

    # ── Stage 3: Maximize total number of jobs scheduled ────────────────────
    if final_status == cp_model.OPTIMAL:
        stage3_expr = sum(is_scheduled)
        final_status, opt3 = _solve_maximize(stage3_expr, "STAGE3-TOTAL-JOBS")
        if opt3 is not None:
            _lock_stage(stage3_expr, opt3, "STAGE3-TOTAL-JOBS")

    # ── Stage 4: Maximize total urgency score ───────────────────────────────
    # This is where AI predicted risk finally influences scheduling.
    # It cannot override Stages 1-3 because those optima are already fixed.
    if final_status == cp_model.OPTIMAL:
        urgency_scores = []
        for job in jobs:
            # If urgency_score is absent, treat as 0 (backward compatibility).
            urgency_scores.append(int(job.get("urgency_score", 0)))

        stage4_expr = sum(
            int(u) * v for u, v in zip(urgency_scores, is_scheduled)
        )
        final_status, opt4 = _solve_maximize(stage4_expr, "STAGE4-URGENCY")
        if opt4 is not None:
            _lock_stage(stage4_expr, opt4, "STAGE4-URGENCY")

    # ── Stage 5: Maximize jobs that finish before their deadline ────────────
    # For each job create a Boolean 'before_deadline[i]' that is 1 only when:
    #     job IS scheduled  AND  end_var[i] <= deadline_in_minutes
    if final_status == cp_model.OPTIMAL:
        before_deadline = []
        for i, job in enumerate(jobs):
            deadline_str = job.get("deadline", "23:59")
            deadline_min = _hhmm_to_minutes(deadline_str)

            bd = model.NewBoolVar(f"bd_{job['maintenance_id']}")

            # bd can only be 1 when the job is scheduled.
            model.Add(bd <= is_scheduled[i])

            # If bd == 1, the job must finish by its deadline.
            model.Add(end_var[i] <= deadline_min).OnlyEnforceIf(bd)

            # If the job is scheduled AND ends on time, bd is forced to 1.
            # We use a helper Boolean for the "scheduled AND ends on time" check.
            on_time = model.NewBoolVar(f"ontime_{job['maintenance_id']}")
            # on_time = 1  iff  is_scheduled[i] = 1  AND  end_var[i] <= deadline_min
            model.Add(end_var[i] <= deadline_min).OnlyEnforceIf(on_time)
            model.Add(end_var[i] > deadline_min).OnlyEnforceIf(on_time.Not())
            # bd must be at least on_time if scheduled (i.e., if scheduled and on time, bd = 1)
            model.Add(bd >= on_time - (1 - is_scheduled[i]))

            before_deadline.append(bd)

        stage5_expr = sum(before_deadline)
        final_status, opt5 = _solve_maximize(stage5_expr, "STAGE5-DEADLINE")
        if opt5 is not None:
            _lock_stage(stage5_expr, opt5, "STAGE5-DEADLINE")

    # ── Stage 6: Minimize total completion time (compact/early schedule) ─────
    # Among all equally good solutions, prefer earlier maintenance blocks.
    if final_status == cp_model.OPTIMAL:
        stage6_expr = sum(end_var[i] for i in range(len(jobs)))
        model.Minimize(stage6_expr)
        final_status = solver.Solve(model)

    # ── Extract solution ───────────────────────────────────────────────────

    scheduled_blocks = []
    unscheduled_jobs = []

    if final_status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for i, job in enumerate(jobs):
            if solver.Value(is_scheduled[i]) == 1:
                s_min = solver.Value(start_var[i])
                e_min = solver.Value(end_var[i])
                block = dict(job)  # copy all job fields
                block["start"] = _minutes_to_hhmm(s_min)
                block["end"] = _minutes_to_hhmm(e_min)
                block["start_minutes"] = s_min
                block["end_minutes"] = e_min
                block["scheduled"] = True
                scheduled_blocks.append(block)
            else:
                unscheduled = dict(job)
                unscheduled["scheduled"] = False
                unscheduled_jobs.append(unscheduled)
    else:
        # Solver could not find any solution — all jobs go to unscheduled.
        for job in jobs:
            unscheduled = dict(job)
            unscheduled["scheduled"] = False
            unscheduled_jobs.append(unscheduled)

    # ── Conflict audit ─────────────────────────────────────────────────────
    # Count actual overlaps in the produced schedule.
    # A correct CP-SAT solution should return 0.
    # We never hardcode this — it's always audited.
    actual_conflicts = _audit_conflicts(scheduled_blocks, trains)

    status_name = solver.StatusName(final_status)

    metrics = {
        "total_jobs": len(jobs),
        "scheduled": len(scheduled_blocks),
        "unscheduled": len(unscheduled_jobs),
        "conflicts": actual_conflicts,
        "solver_status": status_name,
        "solver_wall_time_seconds": round(solver.WallTime(), 3),
    }

    return {
        "scheduled_blocks": scheduled_blocks,
        "unscheduled_jobs": unscheduled_jobs,
        "metrics": metrics,
    }
