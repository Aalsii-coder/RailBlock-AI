"""
RailBlock AI — Repository Layer Tests

These tests verify that repository.py can:

- save planning runs
- save scheduled blocks
- retrieve a run by ID
- retrieve the latest run
- handle NORMAL / SHADOW / RECOVERY planning types
- delete planning runs cleanly
"""

import uuid

from repository import (
    delete_planning_run,
    get_latest_planning_run,
    get_planning_run,
    save_planning_run,
    save_scheduled_blocks,
)


def make_block(
    maintenance_id,
    start="09:00",
    end="10:00",
    section="Delhi-Ghaziabad",
    department="Engineering",
    priority="HIGH",
):
    return {
        "maintenance_id": maintenance_id,
        "section": section,
        "department": department,
        "priority": priority,
        "start": start,
        "end": end,
        "start_minutes": 540,
        "end_minutes": 600,
        "predicted_risk_score": 0.82,
        "urgency_score": 140,
        "scheduled": True,
    }


# ---------------------------------------------------------------------------
# TEST 1
# Save a normal planning run
# ---------------------------------------------------------------------------

def test_save_normal_planning_run():

    result = {
        "scheduled_blocks": [
            make_block(
                "TEST-NORMAL-" + str(uuid.uuid4())[:8]
            )
        ],
        "metrics": {
            "scheduled": 1,
            "conflicts": 0,
        },
    }

    run_id = save_planning_run(
        result=result,
        planning_type="NORMAL",
        planning_time="08:00",
        input_snapshot={
            "source": "pytest"
        },
    )

    try:
        assert isinstance(run_id, str)
        assert len(run_id) > 0

        stored = get_planning_run(run_id)

        assert stored is not None
        assert stored["planning_type"] == "NORMAL"
        assert stored["planning_time"] == "08:00"
        assert stored["metrics"]["conflicts"] == 0
        assert len(stored["scheduled_blocks"]) == 1

    finally:
        delete_planning_run(run_id)


# ---------------------------------------------------------------------------
# TEST 2
# Scheduled block fields survive round-trip
# ---------------------------------------------------------------------------

def test_scheduled_block_round_trip():

    maintenance_id = (
        "TEST-BLOCK-"
        + str(uuid.uuid4())[:8]
    )

    result = {
        "scheduled_blocks": [
            make_block(
                maintenance_id,
                priority="CRITICAL",
            )
        ],
        "metrics": {},
    }

    run_id = save_planning_run(
        result=result,
        planning_type="NORMAL",
    )

    try:
        stored = get_planning_run(run_id)

        block = stored["scheduled_blocks"][0]

        assert block["maintenance_id"] == maintenance_id
        assert block["priority"] == "CRITICAL"
        assert block["start"] == "09:00"
        assert block["end"] == "10:00"
        assert block["predicted_risk_score"] == 0.82
        assert block["urgency_score"] == 140

    finally:
        delete_planning_run(run_id)


# ---------------------------------------------------------------------------
# TEST 3
# Raw data is preserved
# ---------------------------------------------------------------------------

def test_raw_block_data_is_preserved():

    maintenance_id = (
        "TEST-RAW-"
        + str(uuid.uuid4())[:8]
    )

    block = make_block(
        maintenance_id
    )

    block["shadowed"] = True
    block["shadow_block_id"] = "SB777"
    block["ai_risk_bonus"] = 33

    run_id = save_planning_run(
        result={
            "scheduled_blocks": [block],
            "metrics": {},
        },
        planning_type="SHADOW",
    )

    try:
        stored = get_planning_run(run_id)

        raw = stored[
            "scheduled_blocks"
        ][0]["raw_data"]

        assert raw["shadowed"] is True
        assert raw["shadow_block_id"] == "SB777"
        assert raw["ai_risk_bonus"] == 33

    finally:
        delete_planning_run(run_id)


# ---------------------------------------------------------------------------
# TEST 4
# Recovery schedule can be stored
# ---------------------------------------------------------------------------

def test_save_recovery_result():

    maintenance_id = (
        "TEST-RECOVERY-"
        + str(uuid.uuid4())[:8]
    )

    result = {
        "recovered_schedule": [
            make_block(
                maintenance_id,
                start="13:00",
                end="14:00",
            )
        ],
        "metrics": {
            "jobs_moved": 1,
            "recovery_conflicts": 0,
        },
    }

    result["recovered_schedule"][0][
        "start_minutes"
    ] = 780

    result["recovered_schedule"][0][
        "end_minutes"
    ] = 840

    run_id = save_planning_run(
        result=result,
        planning_type="RECOVERY",
        planning_time="11:00",
    )

    try:
        stored = get_planning_run(run_id)

        assert stored["planning_type"] == "RECOVERY"
        assert stored["metrics"]["jobs_moved"] == 1

        block = stored["scheduled_blocks"][0]

        assert block["start"] == "13:00"
        assert block["end"] == "14:00"

    finally:
        delete_planning_run(run_id)


# ---------------------------------------------------------------------------
# TEST 5
# Manual save_scheduled_blocks works
# ---------------------------------------------------------------------------

def test_save_scheduled_blocks_for_existing_run():

    run_id = save_planning_run(
        result={
            "scheduled_blocks": [],
            "metrics": {},
        },
        planning_type="TEST",
    )

    maintenance_id = (
        "TEST-EXTRA-"
        + str(uuid.uuid4())[:8]
    )

    try:
        count = save_scheduled_blocks(
            planning_run_id=run_id,
            blocks=[
                make_block(
                    maintenance_id
                )
            ],
        )

        assert count == 1

        stored = get_planning_run(run_id)

        ids = {
            block["maintenance_id"]
            for block in stored[
                "scheduled_blocks"
            ]
        }

        assert maintenance_id in ids

    finally:
        delete_planning_run(run_id)


# ---------------------------------------------------------------------------
# TEST 6
# Missing planning run returns None
# ---------------------------------------------------------------------------

def test_missing_planning_run_returns_none():

    missing_id = str(uuid.uuid4())

    result = get_planning_run(
        missing_id
    )

    assert result is None


# ---------------------------------------------------------------------------
# TEST 7
# Saving blocks for missing run fails
# ---------------------------------------------------------------------------

def test_save_blocks_for_missing_run_raises():

    missing_id = str(uuid.uuid4())

    try:
        save_scheduled_blocks(
            planning_run_id=missing_id,
            blocks=[
                make_block(
                    "TEST-MISSING"
                )
            ],
        )

        assert False, (
            "Expected ValueError"
        )

    except ValueError:
        pass


# ---------------------------------------------------------------------------
# TEST 8
# Latest run retrieval
# ---------------------------------------------------------------------------

def test_get_latest_planning_run():

    run_id = save_planning_run(
        result={
            "scheduled_blocks": [],
            "metrics": {
                "marker": "latest-test"
            },
        },
        planning_type="LATEST_TEST",
    )

    try:
        latest = get_latest_planning_run(
            planning_type="LATEST_TEST"
        )

        assert latest is not None
        assert latest["run_id"] == run_id
        assert latest["metrics"]["marker"] == "latest-test"

    finally:
        delete_planning_run(run_id)


# ---------------------------------------------------------------------------
# TEST 9
# Delete planning run
# ---------------------------------------------------------------------------

def test_delete_planning_run():

    run_id = save_planning_run(
        result={
            "scheduled_blocks": [],
            "metrics": {},
        },
        planning_type="DELETE_TEST",
    )

    deleted = delete_planning_run(
        run_id
    )

    assert deleted is True

    assert get_planning_run(
        run_id
    ) is None


# ---------------------------------------------------------------------------
# TEST 10
# Deleting missing run returns False
# ---------------------------------------------------------------------------

def test_delete_missing_run_returns_false():

    missing_id = str(uuid.uuid4())

    deleted = delete_planning_run(
        missing_id
    )

    assert deleted is False