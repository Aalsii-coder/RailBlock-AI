from copy import deepcopy

from shadow_scheduler import schedule_with_shadow_blocks


def make_job(
    maintenance_id,
    department,
    section="Delhi-Ghaziabad",
    duration=60,
    priority="HIGH",
    deadline="18:00",
    compatible=True,
    urgency=100,
):
    """
    Helper function for quickly creating maintenance jobs in tests.
    """

    return {
        "maintenance_id": maintenance_id,
        "department": department,
        "section": section,
        "duration_minutes": duration,
        "priority": priority,
        "deadline": deadline,
        "urgency_score": urgency,
        "shadow_compatible": compatible,
    }


def make_train(
    train_id,
    start,
    end,
    section="Delhi-Ghaziabad",
):
    """
    Helper function for quickly creating train movements.
    """

    return {
        "train_id": train_id,
        "section": section,
        "start": start,
        "end": end,
    }


# ---------------------------------------------------------------------------
# TEST 1
# Basic two-department Shadow Block
# ---------------------------------------------------------------------------

def test_two_jobs_scheduled_inside_one_shadow_block():

    data = {
        "trains": [],
        "maintenance_jobs": [
            make_job(
                "M001",
                "Engineering",
                duration=60,
            ),
            make_job(
                "M002",
                "Signal & Telecom",
                duration=30,
            ),
        ],
    }

    result = schedule_with_shadow_blocks(data)

    assert result["metrics"]["scheduled"] == 2
    assert result["metrics"]["shadow_blocks_scheduled"] == 1

    assert len(result["shadow_blocks"]) == 1

    shadow = result["shadow_blocks"][0]

    assert shadow["shadow_block_id"] == "SB001"
    assert shadow["duration_minutes"] == 60

    blocks = result["scheduled_blocks"]

    assert len(blocks) == 2

    # Both jobs must begin together.
    assert blocks[0]["start_minutes"] == blocks[1]["start_minutes"]

    # Original durations must still be preserved.
    durations = sorted(
        block["end_minutes"] - block["start_minutes"]
        for block in blocks
    )

    assert durations == [30, 60]


# ---------------------------------------------------------------------------
# TEST 2
# Three departments can share one possession
# ---------------------------------------------------------------------------

def test_three_department_shadow_block_saves_time():

    data = {
        "trains": [],
        "maintenance_jobs": [
            make_job(
                "M001",
                "Engineering",
                duration=60,
            ),
            make_job(
                "M002",
                "Signal & Telecom",
                duration=30,
            ),
            make_job(
                "M003",
                "Traction Distribution",
                duration=45,
            ),
        ],
    }

    result = schedule_with_shadow_blocks(data)

    assert result["metrics"]["scheduled"] == 3

    assert result["metrics"]["shadow_blocks_scheduled"] == 1

    shadow = result["shadow_blocks"][0]

    assert len(shadow["maintenance_ids"]) == 3

    assert shadow["duration_minutes"] == 60

    # Without stacking:
    # 60 + 30 + 45 = 135 minutes
    assert (
        result["metrics"]["possession_minutes_without_stacking"]
        == 135
    )

    # With stacking:
    # max(60, 30, 45) = 60
    assert (
        result["metrics"]["possession_minutes_with_stacking"]
        == 60
    )

    assert result["metrics"]["time_saved_minutes"] == 75


# ---------------------------------------------------------------------------
# TEST 3
# Shadow Block must avoid train movement
# ---------------------------------------------------------------------------

def test_shadow_block_does_not_overlap_train():

    data = {
        "trains": [
            make_train(
                "T001",
                "00:00",
                "01:00",
            )
        ],
        "maintenance_jobs": [
            make_job(
                "M001",
                "Engineering",
                duration=60,
            ),
            make_job(
                "M002",
                "Signal & Telecom",
                duration=30,
            ),
        ],
    }

    result = schedule_with_shadow_blocks(data)

    assert result["metrics"]["scheduled"] == 2

    shadow = result["shadow_blocks"][0]

    # Train occupies minute 0 → 60.
    # Shadow Block must therefore begin at 60 or later.
    assert shadow["start_minutes"] >= 60

    assert result["metrics"]["conflicts"] == 0


# ---------------------------------------------------------------------------
# TEST 4
# Overlapping train observations must still be allowed
# ---------------------------------------------------------------------------

def test_overlapping_trains_do_not_make_model_infeasible():

    data = {
        "trains": [
            make_train(
                "T001",
                "00:00",
                "01:00",
            ),
            make_train(
                "T002",
                "00:30",
                "01:30",
            ),
        ],
        "maintenance_jobs": [
            make_job(
                "M001",
                "Engineering",
                duration=60,
            ),
            make_job(
                "M002",
                "Signal & Telecom",
                duration=30,
            ),
        ],
    }

    result = schedule_with_shadow_blocks(data)

    assert result["metrics"]["scheduled"] == 2

    assert result["metrics"]["shadow_blocks_scheduled"] == 1

    shadow = result["shadow_blocks"][0]

    # Latest overlapping train finishes at 01:30 = minute 90.
    assert shadow["start_minutes"] >= 90

    assert result["metrics"]["conflicts"] == 0


# ---------------------------------------------------------------------------
# TEST 5
# Explicitly incompatible jobs must remain separate
# ---------------------------------------------------------------------------

def test_incompatible_jobs_are_not_shadowed():

    data = {
        "trains": [],
        "maintenance_jobs": [
            make_job(
                "M001",
                "Engineering",
                compatible=True,
            ),
            make_job(
                "M002",
                "Signal & Telecom",
                compatible=False,
            ),
        ],
    }

    result = schedule_with_shadow_blocks(data)

    assert result["metrics"]["scheduled"] == 2

    assert result["metrics"]["shadow_blocks_scheduled"] == 0

    assert len(result["shadow_blocks"]) == 0

    for block in result["scheduled_blocks"]:
        assert block["shadowed"] is False

    assert result["metrics"]["conflicts"] == 0


# ---------------------------------------------------------------------------
# TEST 6
# Jobs on different sections must never form one Shadow Block
# ---------------------------------------------------------------------------

def test_different_sections_do_not_shadow_together():

    data = {
        "trains": [],
        "maintenance_jobs": [
            make_job(
                "M001",
                "Engineering",
                section="Delhi-Ghaziabad",
            ),
            make_job(
                "M002",
                "Signal & Telecom",
                section="Ghaziabad-Meerut",
            ),
        ],
    }

    result = schedule_with_shadow_blocks(data)

    assert result["metrics"]["scheduled"] == 2

    assert result["metrics"]["shadow_blocks_scheduled"] == 0

    assert result["metrics"]["conflicts"] == 0


# ---------------------------------------------------------------------------
# TEST 7
# CRITICAL member jobs must still count as real jobs
# ---------------------------------------------------------------------------

def test_critical_member_count_is_preserved():

    data = {
        "trains": [],
        "maintenance_jobs": [
            make_job(
                "M001",
                "Engineering",
                priority="CRITICAL",
            ),
            make_job(
                "M002",
                "Signal & Telecom",
                priority="HIGH",
            ),
        ],
    }

    result = schedule_with_shadow_blocks(data)

    assert result["metrics"]["scheduled"] == 2

    assert result["metrics"]["critical_jobs_scheduled"] == 1

    assert result["metrics"]["shadow_blocks_scheduled"] == 1


# ---------------------------------------------------------------------------
# TEST 8
# AI information must survive Shadow Block expansion
# ---------------------------------------------------------------------------

def test_ai_fields_are_preserved():

    engineering = make_job(
        "M001",
        "Engineering",
    )

    engineering["predicted_risk_score"] = 0.91
    engineering["predicted_risk_level"] = "HIGH"
    engineering["ai_risk_bonus"] = 36

    signal = make_job(
        "M002",
        "Signal & Telecom",
    )

    signal["predicted_risk_score"] = 0.25
    signal["predicted_risk_level"] = "LOW"
    signal["ai_risk_bonus"] = 10

    data = {
        "trains": [],
        "maintenance_jobs": [
            engineering,
            signal,
        ],
    }

    result = schedule_with_shadow_blocks(data)

    m001 = next(
        block
        for block in result["scheduled_blocks"]
        if block["maintenance_id"] == "M001"
    )

    assert m001["predicted_risk_score"] == 0.91
    assert m001["predicted_risk_level"] == "HIGH"
    assert m001["ai_risk_bonus"] == 36


# ---------------------------------------------------------------------------
# TEST 9
# Fully occupied section means Shadow Block cannot be scheduled
# ---------------------------------------------------------------------------

def test_shadow_block_unscheduled_when_section_has_no_free_window():

    data = {
        "trains": [
            make_train(
                "T001",
                "00:00",
                "24:00",
            )
        ],
        "maintenance_jobs": [
            make_job(
                "M001",
                "Engineering",
                duration=60,
            ),
            make_job(
                "M002",
                "Signal & Telecom",
                duration=30,
            ),
        ],
    }

    result = schedule_with_shadow_blocks(data)

    assert result["metrics"]["scheduled"] == 0

    assert result["metrics"]["unscheduled"] == 2

    assert result["metrics"]["shadow_blocks_scheduled"] == 0

    assert len(result["shadow_blocks"]) == 0

    assert result["metrics"]["conflicts"] == 0


# ---------------------------------------------------------------------------
# TEST 10
# Input data must never be mutated
# ---------------------------------------------------------------------------

def test_shadow_scheduler_does_not_modify_input():

    data = {
        "trains": [
            make_train(
                "T001",
                "08:00",
                "08:30",
            )
        ],
        "maintenance_jobs": [
            make_job(
                "M001",
                "Engineering",
            ),
            make_job(
                "M002",
                "Signal & Telecom",
            ),
        ],
    }

    original = deepcopy(data)

    schedule_with_shadow_blocks(data)

    assert data == original


# ---------------------------------------------------------------------------
# TEST 11
# Member jobs retain their own original durations
# ---------------------------------------------------------------------------

def test_member_jobs_keep_individual_end_times():

    data = {
        "trains": [],
        "maintenance_jobs": [
            make_job(
                "M001",
                "Engineering",
                duration=90,
            ),
            make_job(
                "M002",
                "Signal & Telecom",
                duration=30,
            ),
            make_job(
                "M003",
                "Traction Distribution",
                duration=45,
            ),
        ],
    }

    result = schedule_with_shadow_blocks(data)

    blocks = {
        block["maintenance_id"]: block
        for block in result["scheduled_blocks"]
    }

    assert (
        blocks["M001"]["end_minutes"]
        - blocks["M001"]["start_minutes"]
        == 90
    )

    assert (
        blocks["M002"]["end_minutes"]
        - blocks["M002"]["start_minutes"]
        == 30
    )

    assert (
        blocks["M003"]["end_minutes"]
        - blocks["M003"]["start_minutes"]
        == 45
    )

    # They must all start together.
    assert (
        blocks["M001"]["start_minutes"]
        == blocks["M002"]["start_minutes"]
        == blocks["M003"]["start_minutes"]
    )


# ---------------------------------------------------------------------------
# TEST 12
# Normal job and Shadow Block must not overlap each other
# ---------------------------------------------------------------------------

def test_normal_job_does_not_overlap_shadow_block():

    data = {
        "trains": [],
        "maintenance_jobs": [
            make_job(
                "M001",
                "Engineering",
                duration=60,
                compatible=True,
            ),
            make_job(
                "M002",
                "Signal & Telecom",
                duration=30,
                compatible=True,
            ),
            make_job(
                "M003",
                "Traction Distribution",
                duration=45,
                compatible=False,
            ),
        ],
    }

    result = schedule_with_shadow_blocks(data)

    assert result["metrics"]["scheduled"] == 3

    assert result["metrics"]["shadow_blocks_scheduled"] == 1

    shadow = result["shadow_blocks"][0]

    normal = next(
        block
        for block in result["scheduled_blocks"]
        if block["maintenance_id"] == "M003"
    )

    shadow_start = shadow["start_minutes"]
    shadow_end = shadow["end_minutes"]

    normal_start = normal["start_minutes"]
    normal_end = normal["end_minutes"]

    overlap = (
        shadow_start < normal_end
        and normal_start < shadow_end
    )

    assert overlap is False

    assert result["metrics"]["conflicts"] == 0