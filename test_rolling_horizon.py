from copy import deepcopy

import pytest

from rolling_horizon import recover_schedule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_job(
    maintenance_id,
    section="Delhi-Ghaziabad",
    department="Engineering",
    duration=60,
    priority="HIGH",
    deadline="18:00",
    compatible=False,
):
    return {
        "maintenance_id": maintenance_id,
        "section": section,
        "department": department,
        "duration_minutes": duration,
        "priority": priority,
        "deadline": deadline,
        "urgency_score": 100,
        "shadow_compatible": compatible,
    }


def make_block(
    maintenance_id,
    start,
    end,
    section="Delhi-Ghaziabad",
    department="Engineering",
    priority="HIGH",
    shadow_block_id=None,
):
    start_h, start_m = map(int, start.split(":"))
    end_h, end_m = map(int, end.split(":"))

    block = {
        "maintenance_id": maintenance_id,
        "section": section,
        "department": department,
        "priority": priority,
        "start": start,
        "end": end,
        "start_minutes": start_h * 60 + start_m,
        "end_minutes": end_h * 60 + end_m,
        "scheduled": True,
        "shadowed": shadow_block_id is not None,
    }

    if shadow_block_id is not None:
        block["shadow_block_id"] = shadow_block_id

    return block


def make_train(
    train_id,
    start,
    end,
    section="Delhi-Ghaziabad",
):
    return {
        "train_id": train_id,
        "section": section,
        "start": start,
        "end": end,
    }


# ---------------------------------------------------------------------------
# TEST 1
# Delayed train moves only affected maintenance
# ---------------------------------------------------------------------------

def test_delayed_train_moves_only_affected_job():

    previous = {
        "scheduled_blocks": [
            make_block("M001", "09:00", "10:00"),
            make_block("M002", "11:00", "12:00"),
            make_block("M003", "14:00", "15:00"),
        ],
        "shadow_blocks": [],
    }

    updated_data = {
        "trains": [
            make_train(
                "T009",
                "11:00",
                "13:00",
            )
        ],
        "maintenance_jobs": [
            make_job("M001"),
            make_job("M002"),
            make_job("M003"),
        ],
    }

    disruption = {
        "type": "TRAIN_DELAY",
        "section": "Delhi-Ghaziabad",
        "start": "11:00",
        "end": "13:00",
    }

    result = recover_schedule(
        previous,
        updated_data,
        disruption,
    )

    blocks = {
        block["maintenance_id"]: block
        for block in result["recovered_schedule"]
    }

    assert blocks["M001"]["start"] == "09:00"
    assert blocks["M001"]["end"] == "10:00"

    assert blocks["M003"]["start"] == "14:00"
    assert blocks["M003"]["end"] == "15:00"

    # M002 must move because the delayed train occupies 11:00–13:00.
    assert blocks["M002"]["start"] != "11:00"

    assert result["metrics"]["recovery_conflicts"] == 0


# ---------------------------------------------------------------------------
# TEST 2
# Recovery must not move maintenance backwards in time
# ---------------------------------------------------------------------------

def test_recovery_never_moves_job_before_disruption_start():

    previous = {
        "scheduled_blocks": [
            make_block(
                "M001",
                "11:00",
                "12:00",
            )
        ],
        "shadow_blocks": [],
    }

    updated_data = {
        "trains": [
            make_train(
                "T001",
                "11:00",
                "13:00",
            )
        ],
        "maintenance_jobs": [
            make_job(
                "M001",
                duration=60,
            )
        ],
    }

    disruption = {
        "type": "TRAIN_DELAY",
        "section": "Delhi-Ghaziabad",
        "start": "11:00",
        "end": "13:00",
    }

    result = recover_schedule(
        previous,
        updated_data,
        disruption,
    )

    recovered = result["recovered_schedule"][0]

    # Cannot time-travel to 00:00, 08:00, etc.
    assert recovered["start_minutes"] >= 11 * 60

    # Delayed train occupies until 13:00.
    assert recovered["start_minutes"] >= 13 * 60

    assert result["metrics"]["recovery_conflicts"] == 0


# ---------------------------------------------------------------------------
# TEST 3
# Different section is untouched
# ---------------------------------------------------------------------------

def test_other_section_remains_locked():

    previous = {
        "scheduled_blocks": [
            make_block(
                "M001",
                "11:00",
                "12:00",
            ),
            make_block(
                "M002",
                "11:00",
                "12:00",
                section="Ghaziabad-Meerut",
            ),
        ],
        "shadow_blocks": [],
    }

    updated_data = {
        "trains": [
            make_train(
                "T001",
                "11:00",
                "13:00",
            )
        ],
        "maintenance_jobs": [
            make_job("M001"),
            make_job(
                "M002",
                section="Ghaziabad-Meerut",
            ),
        ],
    }

    disruption = {
        "type": "TRAIN_DELAY",
        "section": "Delhi-Ghaziabad",
        "start": "11:00",
        "end": "13:00",
    }

    result = recover_schedule(
        previous,
        updated_data,
        disruption,
    )

    m002 = next(
        block
        for block in result["recovered_schedule"]
        if block["maintenance_id"] == "M002"
    )

    assert m002["start"] == "11:00"
    assert m002["end"] == "12:00"


# ---------------------------------------------------------------------------
# TEST 4
# New emergency maintenance can enter recovery
# ---------------------------------------------------------------------------

def test_new_emergency_job_can_be_scheduled():

    previous = {
        "scheduled_blocks": [
            make_block(
                "M001",
                "09:00",
                "10:00",
            )
        ],
        "shadow_blocks": [],
    }

    updated_data = {
        "trains": [
            make_train(
                "T001",
                "11:00",
                "12:00",
            )
        ],
        "maintenance_jobs": [
            make_job("M001"),
            make_job(
                "M999",
                duration=30,
                priority="CRITICAL",
            ),
        ],
    }

    disruption = {
        "type": "EMERGENCY_MAINTENANCE",
        "section": "Delhi-Ghaziabad",
        "start": "11:00",
        "end": "12:00",
    }

    result = recover_schedule(
        previous,
        updated_data,
        disruption,
    )

    ids = {
        block["maintenance_id"]
        for block in result["recovered_schedule"]
    }

    assert "M001" in ids
    assert "M999" in ids

    assert "M999" in result["changes"]["newly_scheduled"]

    assert result["metrics"]["recovery_conflicts"] == 0


# ---------------------------------------------------------------------------
# TEST 5
# Missing disruption field rejected
# ---------------------------------------------------------------------------

def test_missing_disruption_field_rejected():

    previous = {
        "scheduled_blocks": [],
        "shadow_blocks": [],
    }

    updated_data = {
        "trains": [],
        "maintenance_jobs": [],
    }

    disruption = {
        "section": "Delhi-Ghaziabad",
        "start": "11:00",
        # end intentionally missing
    }

    with pytest.raises(ValueError):
        recover_schedule(
            previous,
            updated_data,
            disruption,
        )


# ---------------------------------------------------------------------------
# TEST 6
# Invalid disruption time rejected
# ---------------------------------------------------------------------------

def test_invalid_disruption_window_rejected():

    previous = {
        "scheduled_blocks": [],
        "shadow_blocks": [],
    }

    updated_data = {
        "trains": [],
        "maintenance_jobs": [],
    }

    disruption = {
        "section": "Delhi-Ghaziabad",
        "start": "13:00",
        "end": "11:00",
    }

    with pytest.raises(ValueError):
        recover_schedule(
            previous,
            updated_data,
            disruption,
        )


# ---------------------------------------------------------------------------
# TEST 7
# Entire Shadow Block moves together
# ---------------------------------------------------------------------------

def test_affected_shadow_block_moves_as_one_group():

    previous = {
        "scheduled_blocks": [
            make_block(
                "M001",
                "10:00",
                "11:00",
                department="Engineering",
                shadow_block_id="SB001",
            ),
            make_block(
                "M002",
                "10:00",
                "10:30",
                department="Signal & Telecom",
                shadow_block_id="SB001",
            ),
        ],
        "shadow_blocks": [
            {
                "shadow_block_id": "SB001",
                "section": "Delhi-Ghaziabad",
                "start": "10:00",
                "end": "11:00",
                "start_minutes": 600,
                "end_minutes": 660,
                "maintenance_ids": [
                    "M001",
                    "M002",
                ],
            }
        ],
    }

    updated_data = {
        "trains": [
            make_train(
                "T100",
                "10:45",
                "12:00",
            )
        ],
        "maintenance_jobs": [
            make_job(
                "M001",
                department="Engineering",
                duration=60,
                compatible=True,
            ),
            make_job(
                "M002",
                department="Signal & Telecom",
                duration=30,
                compatible=True,
            ),
        ],
    }

    disruption = {
        "type": "TRAIN_DELAY",
        "section": "Delhi-Ghaziabad",
        "start": "10:45",
        "end": "12:00",
    }

    result = recover_schedule(
        previous,
        updated_data,
        disruption,
    )

    blocks = {
        block["maintenance_id"]: block
        for block in result["recovered_schedule"]
    }

    # Both original members must still exist.
    assert "M001" in blocks
    assert "M002" in blocks

    # Both must move together.
    assert (
        blocks["M001"]["start_minutes"]
        == blocks["M002"]["start_minutes"]
    )

    # New recovery shadow ID.
    assert (
        blocks["M001"]["shadow_block_id"]
        == blocks["M002"]["shadow_block_id"]
    )

    assert blocks["M001"]["shadow_block_id"].startswith(
        "RSB"
    )

    assert result["metrics"]["recovery_conflicts"] == 0


# ---------------------------------------------------------------------------
# TEST 8
# Completely occupied affected section can make job unscheduled
# ---------------------------------------------------------------------------

def test_job_becomes_unscheduled_if_no_future_window_exists():

    previous = {
        "scheduled_blocks": [
            make_block(
                "M001",
                "12:00",
                "13:00",
            )
        ],
        "shadow_blocks": [],
    }

    updated_data = {
        "trains": [
            make_train(
                "T001",
                "12:00",
                "24:00",
            )
        ],
        "maintenance_jobs": [
            make_job(
                "M001",
                duration=60,
            )
        ],
    }

    disruption = {
        "type": "SECTION_BLOCKAGE",
        "section": "Delhi-Ghaziabad",
        "start": "12:00",
        "end": "24:00",
    }

    result = recover_schedule(
        previous,
        updated_data,
        disruption,
    )

    assert "M001" in result["changes"]["now_unscheduled"]

    assert result["metrics"]["jobs_now_unscheduled"] == 1

    assert result["metrics"]["recovery_conflicts"] == 0


# ---------------------------------------------------------------------------
# TEST 9
# No affected jobs means schedule stays unchanged
# ---------------------------------------------------------------------------

def test_unrelated_disruption_keeps_schedule_unchanged():

    previous = {
        "scheduled_blocks": [
            make_block(
                "M001",
                "08:00",
                "09:00",
            ),
            make_block(
                "M002",
                "15:00",
                "16:00",
            ),
        ],
        "shadow_blocks": [],
    }

    updated_data = {
        "trains": [],
        "maintenance_jobs": [
            make_job("M001"),
            make_job("M002"),
        ],
    }

    disruption = {
        "type": "TRAIN_DELAY",
        "section": "Delhi-Ghaziabad",
        "start": "11:00",
        "end": "12:00",
    }

    result = recover_schedule(
        previous,
        updated_data,
        disruption,
    )

    blocks = {
        block["maintenance_id"]: block
        for block in result["recovered_schedule"]
    }

    assert blocks["M001"]["start"] == "08:00"
    assert blocks["M002"]["start"] == "15:00"

    assert result["metrics"]["jobs_moved"] == 0
    assert result["metrics"]["jobs_unchanged"] == 2


# ---------------------------------------------------------------------------
# TEST 10
# Input must not be mutated
# ---------------------------------------------------------------------------

def test_recovery_does_not_modify_inputs():

    previous = {
        "scheduled_blocks": [
            make_block(
                "M001",
                "11:00",
                "12:00",
            )
        ],
        "shadow_blocks": [],
    }

    updated_data = {
        "trains": [
            make_train(
                "T001",
                "11:00",
                "13:00",
            )
        ],
        "maintenance_jobs": [
            make_job("M001")
        ],
    }

    disruption = {
        "type": "TRAIN_DELAY",
        "section": "Delhi-Ghaziabad",
        "start": "11:00",
        "end": "13:00",
    }

    old_previous = deepcopy(previous)
    old_updated = deepcopy(updated_data)
    old_disruption = deepcopy(disruption)

    recover_schedule(
        previous,
        updated_data,
        disruption,
    )

    assert previous == old_previous
    assert updated_data == old_updated
    assert disruption == old_disruption


# ---------------------------------------------------------------------------
# TEST 11
# Change report contains old and new timing
# ---------------------------------------------------------------------------

def test_moved_job_reports_old_and_new_times():

    previous = {
        "scheduled_blocks": [
            make_block(
                "M001",
                "11:00",
                "12:00",
            )
        ],
        "shadow_blocks": [],
    }

    updated_data = {
        "trains": [
            make_train(
                "T001",
                "11:00",
                "13:00",
            )
        ],
        "maintenance_jobs": [
            make_job("M001")
        ],
    }

    disruption = {
        "type": "TRAIN_DELAY",
        "section": "Delhi-Ghaziabad",
        "start": "11:00",
        "end": "13:00",
    }

    result = recover_schedule(
        previous,
        updated_data,
        disruption,
    )

    assert len(result["changes"]["moved"]) == 1

    change = result["changes"]["moved"][0]

    assert change["maintenance_id"] == "M001"

    assert change["old_start"] == "11:00"
    assert change["old_end"] == "12:00"

    assert change["new_start"] != "11:00"


# ---------------------------------------------------------------------------
# TEST 12
# Final recovered plan must be conflict-free
# ---------------------------------------------------------------------------

def test_final_recovered_schedule_is_conflict_free():

    previous = {
        "scheduled_blocks": [
            make_block(
                "M001",
                "09:00",
                "10:00",
            ),
            make_block(
                "M002",
                "11:00",
                "12:00",
            ),
            make_block(
                "M003",
                "14:00",
                "15:00",
            ),
        ],
        "shadow_blocks": [],
    }

    updated_data = {
        "trains": [
            make_train(
                "T001",
                "10:30",
                "13:00",
            )
        ],
        "maintenance_jobs": [
            make_job("M001"),
            make_job("M002"),
            make_job("M003"),
        ],
    }

    disruption = {
        "type": "TRAIN_DELAY",
        "section": "Delhi-Ghaziabad",
        "start": "10:30",
        "end": "13:00",
    }

    result = recover_schedule(
        previous,
        updated_data,
        disruption,
    )

    assert result["metrics"]["recovery_conflicts"] == 0