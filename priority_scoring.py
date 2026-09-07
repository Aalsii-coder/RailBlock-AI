"""
priority_scoring.py
===================
RailBlock AI — AI-Aware Urgency Scoring

What this module does
─────────────────────
It assigns an *urgency score* to every maintenance job by combining:

    1.  Base priority score   — official Indian Railways maintenance priority
                                (CRITICAL > HIGH > MEDIUM > LOW)

    2.  Deadline bonus        — jobs approaching or past their deadline get
                                a time-sensitive bonus so they are not missed.

    3.  AI risk bonus         — jobs the Random Forest model predicts as
                                higher-risk assets get a small extra boost.

Why AI cannot override official priority
────────────────────────────────────────
The base priority (CRITICAL, HIGH, MEDIUM, LOW) is set by railway
officials based on safety regulations and operational rules.
The AI risk bonus is intentionally small (max +40 points) compared to
the gap between priority levels (CRITICAL=100 vs HIGH=50).

This means a HIGH-priority job can NEVER outscore a CRITICAL-priority
job purely on AI risk — the safety hierarchy is preserved.

(The CP-SAT optimizer enforces this even more strongly through
 its staged lexicographic objective — see optimizer.py.)

Urgency score formula
──────────────────────
    urgency_score = priority_score + deadline_bonus + ai_risk_bonus

    priority_score:
        CRITICAL = 100
        HIGH     =  50
        MEDIUM   =  20
        LOW      =   5

    deadline_bonus  (based on minutes_until_deadline at planning_time):
        deadline reached or passed → +60
        within  60 minutes        → +50
        within 180 minutes        → +30
        within 360 minutes        → +15
        otherwise                 → +0

    ai_risk_bonus:
        round(predicted_risk_score * 40)
        (0 if predicted_risk_score is absent)

Priority ranking tie-breaking order
────────────────────────────────────
    1. Higher urgency_score  (descending)
    2. Earlier deadline      (ascending  — HH:MM string, lexicographic OK since zero-padded)
    3. Priority level        (CRITICAL > HIGH > MEDIUM > LOW)
    4. maintenance_id        (alphabetically ascending, deterministic)
"""

import copy

# ── Priority base scores ──────────────────────────────────────────────────────

PRIORITY_SCORE = {
    "CRITICAL": 100,
    "HIGH": 50,
    "MEDIUM": 20,
    "LOW": 5,
}

# For sorting: lower number = higher priority
PRIORITY_ORDER = {
    "CRITICAL": 0,
    "HIGH": 1,
    "MEDIUM": 2,
    "LOW": 3,
}


# ── Time helpers ──────────────────────────────────────────────────────────────

def _time_to_minutes(time_str: str) -> int:
    """Convert 'HH:MM' to integer minutes since midnight."""
    h, m = time_str.split(":")
    return int(h) * 60 + int(m)


def _minutes_until_deadline(deadline_str: str, planning_time_str: str) -> int:
    """
    Calculate how many minutes remain until a deadline.

    Returns a negative number if the deadline has already passed.
    """
    deadline_minutes = _time_to_minutes(deadline_str)
    planning_minutes = _time_to_minutes(planning_time_str)
    return deadline_minutes - planning_minutes


def _deadline_bonus(minutes_remaining: int) -> int:
    """
    Return the deadline bonus based on how much time is left.

    Negative or zero minutes_remaining means the deadline is already
    passed — maximum urgency bonus applies.
    """
    if minutes_remaining <= 0:
        return 60   # deadline reached or passed — most urgent
    elif minutes_remaining <= 60:
        return 50
    elif minutes_remaining <= 180:
        return 30
    elif minutes_remaining <= 360:
        return 15
    else:
        return 0


# ── Public API ────────────────────────────────────────────────────────────────

def score_maintenance_jobs(data: dict, planning_time: str = "00:00") -> dict:
    """
    Compute urgency scores for all maintenance jobs and return a priority ranking.

    Parameters
    ----------
    data : dict
        Railway planning data (output of predict_maintenance_risk or
        validate_and_standardize_data).  Each maintenance job may
        optionally contain 'predicted_risk_score'.

    planning_time : str
        The current time as 'HH:MM' — used to calculate deadline urgency.
        Defaults to '00:00' (start of day).

    Returns
    -------
    dict
        Deep copy of data with each maintenance job enriched with:
            priority_score        – base score for this job's priority level
            deadline_bonus        – time-pressure bonus
            ai_risk_bonus         – bonus from AI predicted risk (0 if absent)
            urgency_score         – total: priority + deadline + ai bonus
            minutes_until_deadline – signed minutes remaining (negative = overdue)
            score_explanation     – human-readable breakdown string

        Also adds 'priority_ranking' at the top level: maintenance jobs
        sorted by (urgency desc, deadline asc, priority level, id asc).
    """
    result = copy.deepcopy(data)

    scored_jobs = []
    for job in result["maintenance_jobs"]:
        job = dict(job)  # shallow copy — values are scalars/strings

        priority = job.get("priority", "LOW")
        p_score = PRIORITY_SCORE.get(priority, 5)

        mins_left = _minutes_until_deadline(job["deadline"], planning_time)
        d_bonus = _deadline_bonus(mins_left)

        # AI risk bonus — safe default 0 if the field is not present.
        risk_score = job.get("predicted_risk_score", 0.0)
        ai_bonus = round(risk_score * 40)

        total = p_score + d_bonus + ai_bonus

        job["priority_score"] = p_score
        job["deadline_bonus"] = d_bonus
        job["ai_risk_bonus"] = ai_bonus
        job["urgency_score"] = total
        job["minutes_until_deadline"] = mins_left
        job["score_explanation"] = (
            f"priority_score({p_score}) + "
            f"deadline_bonus({d_bonus}) + "
            f"ai_risk_bonus({ai_bonus}) = {total}"
        )

        scored_jobs.append(job)

    result["maintenance_jobs"] = scored_jobs

    # ── Build priority ranking ─────────────────────────────────────────────
    # Sort by:
    #   1. urgency_score        DESCENDING  (higher is more urgent)
    #   2. deadline             ASCENDING   (earlier deadline first)
    #   3. priority level       CRITICAL > HIGH > MEDIUM > LOW
    #   4. maintenance_id       ASCENDING   (alphabetical tie-breaker)
    ranked = sorted(
        scored_jobs,
        key=lambda j: (
            -j["urgency_score"],
            j["deadline"],                         # 'HH:MM' lexicographic works because zero-padded
            PRIORITY_ORDER.get(j["priority"], 99),
            j["maintenance_id"],
        ),
    )

    result["priority_ranking"] = [
        {
            "rank": idx + 1,
            "maintenance_id": j["maintenance_id"],
            "section": j["section"],
            "priority": j["priority"],
            "urgency_score": j["urgency_score"],
            "score_explanation": j["score_explanation"],
            "deadline": j["deadline"],
            "minutes_until_deadline": j["minutes_until_deadline"],
        }
        for idx, j in enumerate(ranked)
    ]

    return result
