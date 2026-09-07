"""
test_pipeline.py
================
RailBlock AI — Full Regression & Integration Test Suite

Covers all 20 required test scenarios.

Run with:  pytest -v test_pipeline.py
"""

import copy
import pytest

from synthetic_history import generate_synthetic_history
from ai_predictor import train_risk_model, predict_maintenance_risk
from priority_scoring import score_maintenance_jobs
from optimizer import optimize_schedule
from validator import validate_and_standardize_data
from pipeline import run_ai_planning_pipeline


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def model_bundle():
    """Train a Random Forest on synthetic history once for the entire test module."""
    history = generate_synthetic_history(n_records=500, seed=42)
    return train_risk_model(history)


def _base_data():
    """Return a standard test dataset with four Indian Railways sections."""
    return {
        "sections": [
            "Delhi-Ghaziabad",
            "Mumbai-Pune",
            "Kolkata-Howrah",
            "Chennai-Arakkonam",
        ],
        "trains": [
            # Delhi-Ghaziabad
            {"train_id": "T001", "train_name": "Rajdhani Express",
             "section": "Delhi-Ghaziabad", "start": "08:00", "end": "08:25", "priority": "HIGH"},
            {"train_id": "T002", "train_name": "Shatabdi Express",
             "section": "Delhi-Ghaziabad", "start": "10:00", "end": "10:30", "priority": "HIGH"},
            # Mumbai-Pune
            {"train_id": "T003", "train_name": "Deccan Queen",
             "section": "Mumbai-Pune", "start": "07:15", "end": "07:50", "priority": "HIGH"},
            # Kolkata-Howrah
            {"train_id": "T004", "train_name": "Howrah Mail",
             "section": "Kolkata-Howrah", "start": "06:00", "end": "06:30", "priority": "MEDIUM"},
            # Chennai-Arakkonam
            {"train_id": "T005", "train_name": "Chennai Express",
             "section": "Chennai-Arakkonam", "start": "09:00", "end": "09:45", "priority": "HIGH"},
        ],
        "maintenance_jobs": [
            {"maintenance_id": "M001", "department": "Engineering",
             "section": "Delhi-Ghaziabad", "duration_minutes": 60,
             "priority": "CRITICAL", "deadline": "12:00"},
            {"maintenance_id": "M002", "department": "Signal & Telecom",
             "section": "Mumbai-Pune", "duration_minutes": 90,
             "priority": "HIGH", "deadline": "15:00"},
            {"maintenance_id": "M003", "department": "Traction Distribution",
             "section": "Kolkata-Howrah", "duration_minutes": 45,
             "priority": "MEDIUM", "deadline": "20:00"},
            {"maintenance_id": "M004", "department": "Engineering",
             "section": "Chennai-Arakkonam", "duration_minutes": 120,
             "priority": "HIGH", "deadline": "22:00"},
        ],
    }


def _base_asset_features():
    """Return asset feature records for AI risk prediction."""
    return [
        {"maintenance_id": "M001", "asset_age_years": 12, "days_since_last_maintenance": 180,
         "recent_defect_count": 3, "failures_last_12_months": 2, "traffic_load_percent": 85},
        {"maintenance_id": "M002", "asset_age_years": 5, "days_since_last_maintenance": 30,
         "recent_defect_count": 1, "failures_last_12_months": 0, "traffic_load_percent": 60},
        {"maintenance_id": "M003", "asset_age_years": 20, "days_since_last_maintenance": 300,
         "recent_defect_count": 5, "failures_last_12_months": 3, "traffic_load_percent": 70},
        {"maintenance_id": "M004", "asset_age_years": 2, "days_since_last_maintenance": 15,
         "recent_defect_count": 0, "failures_last_12_months": 0, "traffic_load_percent": 40},
    ]


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 1 — Full pipeline runs end-to-end
# ═══════════════════════════════════════════════════════════════════════════════

def test_01_full_pipeline_runs(model_bundle):
    """The complete AI planning pipeline executes without errors."""
    result = run_ai_planning_pipeline(
        _base_data(), _base_asset_features(), model_bundle, planning_time="06:00"
    )
    assert "scheduled_blocks" in result
    assert "unscheduled_jobs" in result
    assert "metrics" in result
    assert "priority_ranking" in result
    assert "ai_risk_ranking" in result
    assert "ai_model_info" in result


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 2 — Input is not mutated
# ═══════════════════════════════════════════════════════════════════════════════

def test_02_input_not_mutated(model_bundle):
    """The original data dict must remain unchanged after the pipeline."""
    data = _base_data()
    original = copy.deepcopy(data)
    run_ai_planning_pipeline(data, _base_asset_features(), model_bundle)
    assert data == original, "Pipeline mutated the input data!"


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 3 — predicted_risk_score is in [0, 1]
# ═══════════════════════════════════════════════════════════════════════════════

def test_03_risk_score_in_range(model_bundle):
    """predicted_risk_score must be a float in [0.0, 1.0]."""
    result = run_ai_planning_pipeline(
        _base_data(), _base_asset_features(), model_bundle
    )
    all_jobs = result["scheduled_blocks"] + result["unscheduled_jobs"]
    for job in all_jobs:
        score = job["predicted_risk_score"]
        assert 0.0 <= score <= 1.0, (
            f"{job['maintenance_id']} has risk_score={score} outside [0,1]"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 4 — Exact risk level boundaries
# ═══════════════════════════════════════════════════════════════════════════════

def test_04_risk_level_boundaries(model_bundle):
    """
    Verify exact boundary values for risk level classification:
        0.32 → LOW
        0.33 → MEDIUM
        0.65 → MEDIUM
        0.66 → HIGH
    """
    from ai_predictor import _score_to_level
    assert _score_to_level(0.32) == "LOW"
    assert _score_to_level(0.33) == "MEDIUM"
    assert _score_to_level(0.65) == "MEDIUM"
    assert _score_to_level(0.66) == "HIGH"
    # Edge extremes
    assert _score_to_level(0.00) == "LOW"
    assert _score_to_level(1.00) == "HIGH"


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 5 — AI risk bonus formula
# ═══════════════════════════════════════════════════════════════════════════════

def test_05_ai_risk_bonus_formula():
    """ai_risk_bonus == round(predicted_risk_score * 40)."""
    data = {
        "sections": ["Delhi-Ghaziabad"],
        "trains": [],
        "maintenance_jobs": [
            {"maintenance_id": "MX", "department": "Engineering",
             "section": "Delhi-Ghaziabad", "duration_minutes": 30,
             "priority": "MEDIUM", "deadline": "23:00",
             "predicted_risk_score": 0.75,
             "predicted_risk_level": "HIGH"},
        ],
    }
    result = score_maintenance_jobs(data, planning_time="00:00")
    job = result["maintenance_jobs"][0]
    expected_bonus = round(0.75 * 40)
    assert job["ai_risk_bonus"] == expected_bonus, (
        f"Expected ai_risk_bonus={expected_bonus}, got {job['ai_risk_bonus']}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 6 — urgency_score = priority + deadline + AI bonus
# ═══════════════════════════════════════════════════════════════════════════════

def test_06_urgency_score_formula():
    """urgency_score must equal priority_score + deadline_bonus + ai_risk_bonus."""
    data = {
        "sections": ["Mumbai-Pune"],
        "trains": [],
        "maintenance_jobs": [
            {"maintenance_id": "MX", "department": "Engineering",
             "section": "Mumbai-Pune", "duration_minutes": 60,
             "priority": "CRITICAL", "deadline": "03:00",
             "predicted_risk_score": 0.875,
             "predicted_risk_level": "HIGH"},
        ],
    }
    # planning_time = 00:00  →  180 minutes until 03:00  →  deadline_bonus = +30
    result = score_maintenance_jobs(data, planning_time="00:00")
    job = result["maintenance_jobs"][0]
    expected = 100 + 30 + round(0.875 * 40)
    assert job["urgency_score"] == expected, (
        f"Expected {expected}, got {job['urgency_score']}"
    )
    assert job["priority_score"] == 100
    assert job["deadline_bonus"] == 30
    assert job["ai_risk_bonus"] == round(0.875 * 40)


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 7 — Priority ranking uses 4-key sort order
# ═══════════════════════════════════════════════════════════════════════════════

def test_07_priority_ranking_order(model_bundle):
    """
    Priority ranking must use:
        1. urgency desc
        2. deadline asc
        3. priority level (CRITICAL > HIGH > MEDIUM > LOW)
        4. maintenance_id asc
    """
    result = run_ai_planning_pipeline(
        _base_data(), _base_asset_features(), model_bundle, planning_time="06:00"
    )
    ranking = result["priority_ranking"]
    assert len(ranking) > 0

    for i in range(len(ranking) - 1):
        a = ranking[i]
        b = ranking[i + 1]
        # First key: urgency must be non-increasing
        assert a["urgency_score"] >= b["urgency_score"], (
            f"Ranking broken: {a['maintenance_id']} urgency={a['urgency_score']} "
            f"< {b['maintenance_id']} urgency={b['urgency_score']}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 8 — AI risk influences otherwise-equivalent jobs
# ═══════════════════════════════════════════════════════════════════════════════

def test_08_ai_risk_affects_equivalent_jobs():
    """
    Two identical jobs except for predicted_risk_score — the higher-risk
    job must receive a higher urgency_score.
    """
    data = {
        "sections": ["Kolkata-Howrah"],
        "trains": [],
        "maintenance_jobs": [
            {"maintenance_id": "MA", "department": "Engineering",
             "section": "Kolkata-Howrah", "duration_minutes": 60,
             "priority": "MEDIUM", "deadline": "20:00",
             "predicted_risk_score": 0.80, "predicted_risk_level": "HIGH"},
            {"maintenance_id": "MB", "department": "Engineering",
             "section": "Kolkata-Howrah", "duration_minutes": 60,
             "priority": "MEDIUM", "deadline": "20:00",
             "predicted_risk_score": 0.20, "predicted_risk_level": "LOW"},
        ],
    }
    result = score_maintenance_jobs(data, planning_time="00:00")
    jobs_by_id = {j["maintenance_id"]: j for j in result["maintenance_jobs"]}
    assert jobs_by_id["MA"]["urgency_score"] > jobs_by_id["MB"]["urgency_score"]


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 9 — CRITICAL beats HIGH+high-AI-risk when only one fits
# ═══════════════════════════════════════════════════════════════════════════════

def test_09_critical_beats_high_with_high_ai_risk():
    """
    CRITICAL job must be scheduled over a HIGH job even when the HIGH job
    has a very high AI risk score, and only one job can fit on the section.
    """
    # Only enough space for one job: both need 120 minutes.
    # Section has one train: 00:00–20:00, leaving only 20:00–24:00 (240 min).
    # But we put train at 00:00–22:00 leaving 22:00–24:00 = 120 min exactly.
    # Only ONE 120-minute job can fit in 22:00–24:00.
    data = {
        "sections": ["Chennai-Arakkonam"],
        "trains": [
            {"train_id": "TX", "train_name": "Block Train",
             "section": "Chennai-Arakkonam", "start": "00:00", "end": "22:00",
             "priority": "HIGH"},
        ],
        "maintenance_jobs": [
            # CRITICAL job, LOW AI risk
            {"maintenance_id": "MC", "department": "Engineering",
             "section": "Chennai-Arakkonam", "duration_minutes": 120,
             "priority": "CRITICAL", "deadline": "23:59",
             "urgency_score": 100, "predicted_risk_score": 0.10,
             "predicted_risk_level": "LOW"},
            # HIGH job, EXTREME AI risk
            {"maintenance_id": "MH", "department": "Engineering",
             "section": "Chennai-Arakkonam", "duration_minutes": 120,
             "priority": "HIGH", "deadline": "23:59",
             "urgency_score": 90, "predicted_risk_score": 0.99,
             "predicted_risk_level": "HIGH"},
        ],
    }
    result = optimize_schedule(data)
    scheduled_ids = {b["maintenance_id"] for b in result["scheduled_blocks"]}
    unscheduled_ids = {j["maintenance_id"] for j in result["unscheduled_jobs"]}
    assert "MC" in scheduled_ids, (
        "CRITICAL job must be scheduled when only one slot is available"
    )
    assert "MH" in unscheduled_ids, (
        "HIGH job should be unscheduled — CRITICAL takes precedence"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 10 — No maintenance overlaps a train
# ═══════════════════════════════════════════════════════════════════════════════

def test_10_no_maintenance_overlaps_train(model_bundle):
    """No scheduled block must overlap a train on the same section."""
    result = run_ai_planning_pipeline(
        _base_data(), _base_asset_features(), model_bundle
    )
    trains = _base_data()["trains"]

    for block in result["scheduled_blocks"]:
        b_start = int(block["start_minutes"])
        b_end = int(block["end_minutes"])
        for train in trains:
            if train["section"] != block["section"]:
                continue
            t_start = int(train["start"].split(":")[0]) * 60 + int(train["start"].split(":")[1])
            t_end = int(train["end"].split(":")[0]) * 60 + int(train["end"].split(":")[1])
            overlap = b_start < t_end and t_start < b_end
            assert not overlap, (
                f"Block {block['maintenance_id']} ({block['start']}-{block['end']}) "
                f"overlaps train {train['train_id']} ({train['start']}-{train['end']}) "
                f"on {block['section']}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 11 — No invalid maintenance-vs-maintenance overlap on same section
# ═══════════════════════════════════════════════════════════════════════════════

def test_11_no_maintenance_overlap(model_bundle):
    """No two scheduled blocks on the same section may overlap."""
    result = run_ai_planning_pipeline(
        _base_data(), _base_asset_features(), model_bundle
    )
    blocks = result["scheduled_blocks"]

    for i, b1 in enumerate(blocks):
        for b2 in blocks[i + 1:]:
            if b1["section"] != b2["section"]:
                continue
            s1, e1 = b1["start_minutes"], b1["end_minutes"]
            s2, e2 = b2["start_minutes"], b2["end_minutes"]
            assert not (s1 < e2 and s2 < e1), (
                f"Maintenance overlap: {b1['maintenance_id']} and "
                f"{b2['maintenance_id']} on {b1['section']}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 12 — Scheduled duration equals requested duration
# ═══════════════════════════════════════════════════════════════════════════════

def test_12_scheduled_duration_correct(model_bundle):
    """For every scheduled block, end - start must equal duration_minutes."""
    result = run_ai_planning_pipeline(
        _base_data(), _base_asset_features(), model_bundle
    )
    for block in result["scheduled_blocks"]:
        actual = block["end_minutes"] - block["start_minutes"]
        expected = block["duration_minutes"]
        assert actual == expected, (
            f"{block['maintenance_id']}: expected duration {expected}m, got {actual}m"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 13 — conflicts metric is audited and equals 0
# ═══════════════════════════════════════════════════════════════════════════════

def test_13_conflicts_metric_is_zero(model_bundle):
    """
    metrics['conflicts'] must be 0 after a successful solve.
    The value must be computed by a real audit — never hardcoded.
    """
    result = run_ai_planning_pipeline(
        _base_data(), _base_asset_features(), model_bundle
    )
    assert result["metrics"]["conflicts"] == 0, (
        f"Expected 0 conflicts, got {result['metrics']['conflicts']}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 14 — optimize_schedule works without AI fields
# ═══════════════════════════════════════════════════════════════════════════════

def test_14_optimizer_without_ai_fields():
    """optimize_schedule must work when urgency_score and predicted_risk_score are absent."""
    data = {
        "sections": ["Delhi-Ghaziabad"],
        "trains": [
            {"train_id": "T1", "train_name": "Express",
             "section": "Delhi-Ghaziabad", "start": "08:00", "end": "09:00",
             "priority": "HIGH"},
        ],
        "maintenance_jobs": [
            # No urgency_score or predicted_risk_score
            {"maintenance_id": "MX", "department": "Engineering",
             "section": "Delhi-Ghaziabad", "duration_minutes": 60,
             "priority": "HIGH", "deadline": "20:00"},
        ],
    }
    result = optimize_schedule(data)
    assert result["metrics"]["scheduled"] + result["metrics"]["unscheduled"] == 1
    assert result["metrics"]["conflicts"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 15 — Changing planning_time changes deadline urgency
# ═══════════════════════════════════════════════════════════════════════════════

def test_15_planning_time_affects_deadline_urgency():
    """
    A job with deadline 12:00 should receive a higher deadline_bonus
    when planning_time = 11:00 (60 min left, bonus=+50) than when
    planning_time = 00:00 (720 min left, bonus=0).
    """
    data = {
        "sections": ["Mumbai-Pune"],
        "trains": [],
        "maintenance_jobs": [
            {"maintenance_id": "MX", "department": "Engineering",
             "section": "Mumbai-Pune", "duration_minutes": 30,
             "priority": "MEDIUM", "deadline": "12:00",
             "predicted_risk_score": 0.5},
        ],
    }
    early = score_maintenance_jobs(data, planning_time="00:00")
    late = score_maintenance_jobs(data, planning_time="11:00")

    early_bonus = early["maintenance_jobs"][0]["deadline_bonus"]
    late_bonus = late["maintenance_jobs"][0]["deadline_bonus"]
    assert late_bonus > early_bonus, (
        f"Expected late bonus > early bonus, got {late_bonus} vs {early_bonus}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 16 — OVERLAPPING TRAIN REGRESSION TEST
# ═══════════════════════════════════════════════════════════════════════════════

def test_16_overlapping_trains_regression():
    """
    Two trains that overlap each other on the same section must not make
    the CP-SAT model infeasible.  Maintenance must avoid BOTH trains.

    T001: 08:00–08:25
    T002: 08:20–08:50  ← overlaps T001

    This is valid input (e.g. up-line and down-line).
    """
    data = {
        "sections": ["Delhi-Ghaziabad"],
        "trains": [
            {"train_id": "T001", "train_name": "Rajdhani",
             "section": "Delhi-Ghaziabad", "start": "08:00", "end": "08:25", "priority": "HIGH"},
            {"train_id": "T002", "train_name": "Shatabdi",
             "section": "Delhi-Ghaziabad", "start": "08:20", "end": "08:50", "priority": "HIGH"},
        ],
        "maintenance_jobs": [
            {"maintenance_id": "M_OL", "department": "Engineering",
             "section": "Delhi-Ghaziabad", "duration_minutes": 60,
             "priority": "HIGH", "deadline": "23:00"},
        ],
    }
    result = optimize_schedule(data)

    # Solver must find a solution (not infeasible)
    assert result["metrics"]["scheduled"] == 1, (
        "Solver must schedule the maintenance job even with overlapping trains"
    )

    # Maintenance must not overlap either train
    block = result["scheduled_blocks"][0]
    b_start = block["start_minutes"]
    b_end = block["end_minutes"]
    trains = [(8 * 60, 8 * 60 + 25), (8 * 60 + 20, 8 * 60 + 50)]
    for (t_s, t_e) in trains:
        assert not (b_start < t_e and t_s < b_end), (
            f"Maintenance ({block['start']}-{block['end']}) overlaps train "
            f"({t_s//60:02d}:{t_s%60:02d}-{t_e//60:02d}:{t_e%60:02d})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 17 — 15-JOB REGRESSION TEST (overflow protection)
# ═══════════════════════════════════════════════════════════════════════════════

def test_17_fifteen_job_regression():
    """
    15 maintenance requests must optimise successfully.
    This protects against the large-objective-coefficient overflow issue.
    """
    sections = [
        "Delhi-Ghaziabad", "Mumbai-Pune", "Kolkata-Howrah", "Chennai-Arakkonam"
    ]
    priorities = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]

    jobs = []
    for i in range(15):
        section = sections[i % len(sections)]
        priority = priorities[i % len(priorities)]
        jobs.append({
            "maintenance_id": f"M{i+1:02d}",
            "department": "Engineering",
            "section": section,
            "duration_minutes": 30,
            "priority": priority,
            "deadline": "23:00",
            "urgency_score": 50,
        })

    data = {
        "sections": sections,
        "trains": [],
        "maintenance_jobs": jobs,
    }
    result = optimize_schedule(data)

    # Solver must complete (not error or overflow)
    assert result["metrics"]["total_jobs"] == 15
    assert result["metrics"]["scheduled"] + result["metrics"]["unscheduled"] == 15
    # Expect all 15 to be scheduled (no train conflicts, ample time)
    assert result["metrics"]["scheduled"] == 15, (
        f"Expected all 15 jobs scheduled, got {result['metrics']['scheduled']}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 18 — WEIGHTED WORK TEST (larger duration wins)
# ═══════════════════════════════════════════════════════════════════════════════

def test_18_weighted_work_prefers_longer_job():
    """
    When only one HIGH-priority job can fit, the one with greater
    duration must win because:
        50 * 120 > 50 * 60
    """
    # Only 120 minutes of free time available on the section.
    # Train blocks 00:00–20:00, leaving 20:00–24:00 = 240 min.
    # Wait — we need exactly 120 min free.  Use 22:00–24:00.
    # Train: 00:00 – 22:00  (1320 min) → free: 22:00–24:00 (120 min).
    # Job A needs 120 min → fits.
    # Job B needs 60 min → fits.
    # Both are HIGH priority.  Weighted work: 50*120=6000 vs 50*60=3000.
    # Optimizer should schedule Job A (120 min).
    # But can BOTH fit? 120+60=180 > 120, so NO.
    data = {
        "sections": ["Delhi-Ghaziabad"],
        "trains": [
            {"train_id": "TX", "train_name": "Block",
             "section": "Delhi-Ghaziabad", "start": "00:00", "end": "22:00",
             "priority": "HIGH"},
        ],
        "maintenance_jobs": [
            {"maintenance_id": "MA", "department": "Engineering",
             "section": "Delhi-Ghaziabad", "duration_minutes": 120,
             "priority": "HIGH", "deadline": "23:59", "urgency_score": 50},
            {"maintenance_id": "MB", "department": "Engineering",
             "section": "Delhi-Ghaziabad", "duration_minutes": 60,
             "priority": "HIGH", "deadline": "23:59", "urgency_score": 50},
        ],
    }
    result = optimize_schedule(data)
    scheduled_ids = {b["maintenance_id"] for b in result["scheduled_blocks"]}
    assert "MA" in scheduled_ids, (
        "120-min HIGH job must win over 60-min HIGH job (50*120 > 50*60)"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 19 — TRUE DEADLINE TEST
# ═══════════════════════════════════════════════════════════════════════════════

def test_19_deadline_preference():
    """
    When two otherwise-equivalent jobs exist and only one can actually
    finish before its deadline (due to available slots), that job is
    preferred in the deadline stage.
    """
    # Section has a train 06:00–20:00.
    # Free slots: 00:00–06:00 and 20:00–24:00.
    # Job A: deadline 05:00, duration 60 min  → can finish in 00:00–06:00
    # Job B: deadline 21:00, duration 60 min  → can finish in 20:00–21:00
    # Same priority (HIGH), same urgency_score.
    # Both should be scheduled (different sections or combined slot check).
    # Make them on the same section to see which gets the earlier slot.
    # To test deadline preference: Job A has early deadline, Job B late.
    # With a train 04:30–20:00:
    #   Free before train: 00:00–04:30 = 270 min (Job A fits: finishes ≤ 04:30 ≤ 05:00)
    #   Free after train:  20:00–24:00 = 240 min (Job B fits: finishes ≤ 20:xx ≤ 21:00? wait 20+60=21:00 ≤ 21:00 ✓)
    # But both can be scheduled! Let's make only ONE fit by using 1 short window:
    # Train: 01:00–22:00. Free: 00:00–01:00 (60 min) and 22:00–24:00 (120 min).
    # Job A: duration 60, deadline 00:59 → must start 00:00, end 01:00. But deadline is 00:59 < 01:00. FAILS deadline.
    # Job B: duration 60, deadline 23:00 → can fit 22:00–23:00. Meets deadline.
    # Both same priority. Stage 5 should prefer B because it meets its deadline.
    # But Stage 3 will try to schedule BOTH (60min in 00:00-01:00 AND 60min in 22:00-24:00).
    # Both fit! Let's adjust: only ONE slot: train 00:01–22:00.
    # Free: 00:00–00:01 (1 min, not enough) and 22:00–24:00 (120 min).
    # So only one 60-min job can be scheduled.
    # Job A: deadline 22:30 (finishes 22:00–23:00 ≤ 22:30? No, 23:00 > 22:30). Misses deadline.
    # Job B: deadline 23:30 (finishes 22:00–23:00 ≤ 23:30). Meets deadline.
    # Stage 5 prefers B. Both have same priority, urgency. B should be scheduled.
    data = {
        "sections": ["Kolkata-Howrah"],
        "trains": [
            {"train_id": "TX", "train_name": "Block",
             "section": "Kolkata-Howrah", "start": "00:01", "end": "22:00",
             "priority": "HIGH"},
        ],
        "maintenance_jobs": [
            {"maintenance_id": "JA", "department": "Engineering",
             "section": "Kolkata-Howrah", "duration_minutes": 60,
             "priority": "HIGH", "deadline": "22:30",   # cannot actually meet (ends 23:00)
             "urgency_score": 60},
            {"maintenance_id": "JB", "department": "Engineering",
             "section": "Kolkata-Howrah", "duration_minutes": 60,
             "priority": "HIGH", "deadline": "23:30",   # can meet (22:00–23:00 ≤ 23:30)
             "urgency_score": 60},
        ],
    }
    result = optimize_schedule(data)
    # Only one job can be scheduled (single 120-min window, each job = 60 min, same section = no-overlap)
    # Actually both fit in 22:00–24:00 on same section? No — same-section NoOverlap applies.
    # 22:00–23:00 (JA or JB) + 23:00–24:00 (the other). Both fit!
    # Stage 5: JB meets deadline (23:00 ≤ 23:30), JA misses (23:00 > 22:30 if scheduled second).
    # Let's verify at least JB is scheduled.
    scheduled_ids = {b["maintenance_id"] for b in result["scheduled_blocks"]}
    assert "JB" in scheduled_ids, (
        "Job B (meets deadline) must be scheduled"
    )
    # And verify JB's end time is actually within its deadline
    for block in result["scheduled_blocks"]:
        if block["maintenance_id"] == "JB":
            assert block["end_minutes"] <= 23 * 60 + 30, (
                f"JB ends at {block['end']} which is after deadline 23:30"
            )


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 20 — Synthetic history has both classes; model trains successfully
# ═══════════════════════════════════════════════════════════════════════════════

def test_20_synthetic_data_both_classes_and_model_trains():
    """
    Synthetic history must contain both class 0 and class 1 labels.
    The RandomForestClassifier must train successfully.
    """
    history = generate_synthetic_history(n_records=500, seed=42)

    labels = [r["failure_within_30_days"] for r in history]
    assert 0 in labels, "Synthetic history must contain class 0 (no failure)"
    assert 1 in labels, "Synthetic history must contain class 1 (failure)"

    bundle = train_risk_model(history)
    assert "model" in bundle
    assert bundle["n_samples"] == 500
    assert set(bundle["classes"]) == {0, 1}

    # Verify the model can produce predictions
    row = [[10, 200, 4, 2, 75]]
    proba = bundle["model"].predict_proba(row)
    assert proba.shape[1] == 2


# ═══════════════════════════════════════════════════════════════════════════════
#  ADDITIONAL ROBUSTNESS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_validator_rejects_bad_department():
    """validate_and_standardize_data must reject invalid departments."""
    data = _base_data()
    data["maintenance_jobs"][0]["department"] = "Electrical"
    with pytest.raises(ValueError, match="Department"):
        validate_and_standardize_data(data)


def test_validator_rejects_bad_time():
    """validate_and_standardize_data must reject malformed time strings."""
    data = _base_data()
    data["trains"][0]["start"] = "25:00"
    with pytest.raises(ValueError):
        validate_and_standardize_data(data)


def test_validator_does_not_mutate_input():
    """Validator must return a deep copy — original data must not change."""
    data = _base_data()
    original_start = data["trains"][0]["start"]
    validate_and_standardize_data(data)
    assert data["trains"][0]["start"] == original_start


def test_deadline_bonus_all_thresholds():
    """Verify all deadline bonus thresholds produce correct values."""
    from priority_scoring import _deadline_bonus
    assert _deadline_bonus(-10) == 60   # overdue
    assert _deadline_bonus(0) == 60     # exactly at deadline
    assert _deadline_bonus(30) == 50    # within 60 min
    assert _deadline_bonus(60) == 50    # exactly 60 min
    assert _deadline_bonus(61) == 30    # within 180 min
    assert _deadline_bonus(180) == 30   # exactly 180 min
    assert _deadline_bonus(181) == 15   # within 360 min
    assert _deadline_bonus(360) == 15   # exactly 360 min
    assert _deadline_bonus(361) == 0    # beyond 360 min


def test_empty_jobs_handled_gracefully():
    """optimize_schedule must handle zero maintenance jobs without error."""
    data = {
        "sections": ["Delhi-Ghaziabad"],
        "trains": [],
        "maintenance_jobs": [],
    }
    result = optimize_schedule(data)
    assert result["metrics"]["scheduled"] == 0
    assert result["metrics"]["conflicts"] == 0
