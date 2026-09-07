"""
RailBlock AI — FastAPI Tests

Tests:
- health endpoint
- input validation
- Shadow Block API
- Rolling Horizon Recovery API
- CORS middleware
- PostgreSQL persistence
- retrieval of saved planning runs
"""

import uuid

from fastapi.testclient import TestClient

from api import app
from repository import delete_planning_run


# ---------------------------------------------------------------------------
# FastAPI test client
# ---------------------------------------------------------------------------

client = TestClient(app)


# ===========================================================================
# TEST 1 — Health endpoint
# ===========================================================================

def test_health_endpoint():

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ok"
    assert body["service"] == "RailBlock AI"
    assert body["ai_model"] == "RandomForestClassifier"


# ===========================================================================
# TEST 2 — Normal planning rejects missing data
# ===========================================================================

def test_planning_run_rejects_missing_data():

    response = client.post(
        "/planning/run",
        json={
            "asset_features": [],
            "planning_time": "08:00",
        },
    )

    assert response.status_code == 400

    assert "data" in response.json()[
        "detail"
    ]


# ===========================================================================
# TEST 3 — Shadow endpoint rejects missing data
# ===========================================================================

def test_shadow_endpoint_rejects_missing_data():

    response = client.post(
        "/planning/shadow",
        json={},
    )

    assert response.status_code == 400

    assert "data" in response.json()[
        "detail"
    ]


# ===========================================================================
# TEST 4 — Recovery endpoint rejects missing previous result
# ===========================================================================

def test_recover_endpoint_rejects_missing_previous_result():

    response = client.post(
        "/planning/recover",
        json={
            "updated_data": {
                "trains": [],
                "maintenance_jobs": [],
                "sections": [],
            },

            "disruption": {
                "type": "TRAIN_DELAY",
                "section": "Delhi-Ghaziabad",
                "start": "11:00",
                "end": "13:00",
            },
        },
    )

    assert response.status_code == 400

    assert "previous_result" in response.json()[
        "detail"
    ]


# ===========================================================================
# TEST 5 — Shadow endpoint runs
# ===========================================================================

def test_shadow_endpoint_runs():

    payload = {
        "data": {
            "trains": [],

            "maintenance_jobs": [
                {
                    "maintenance_id": "API-SHADOW-A",
                    "department": "Engineering",
                    "section": "Delhi-Ghaziabad",
                    "duration_minutes": 60,
                    "priority": "HIGH",
                    "deadline": "18:00",
                    "urgency_score": 100,
                    "shadow_compatible": True,
                },

                {
                    "maintenance_id": "API-SHADOW-B",
                    "department": "Signal & Telecom",
                    "section": "Delhi-Ghaziabad",
                    "duration_minutes": 30,
                    "priority": "MEDIUM",
                    "deadline": "18:00",
                    "urgency_score": 80,
                    "shadow_compatible": True,
                },
            ],

            "sections": [
                "Delhi-Ghaziabad"
            ],
        }
    }

    response = client.post(
        "/planning/shadow",
        json=payload,
    )

    assert response.status_code == 200

    body = response.json()

    assert "scheduled_blocks" in body
    assert "shadow_blocks" in body
    assert "metrics" in body
    assert "planning_run_id" in body

    run_id = body[
        "planning_run_id"
    ]

    try:

        assert body["metrics"][
            "scheduled"
        ] == 2

        assert body["metrics"][
            "shadow_blocks_scheduled"
        ] == 1

        assert body["metrics"][
            "conflicts"
        ] == 0

    finally:

        delete_planning_run(
            run_id
        )


# ===========================================================================
# TEST 6 — Recovery endpoint runs
# ===========================================================================

def test_recover_endpoint_runs():

    previous_result = {
        "scheduled_blocks": [
            {
                "maintenance_id": "API-REC-A",
                "section": "Delhi-Ghaziabad",
                "department": "Engineering",
                "priority": "HIGH",

                "start": "11:00",
                "end": "12:00",

                "start_minutes": 660,
                "end_minutes": 720,

                "scheduled": True,
                "shadowed": False,
            }
        ],

        "shadow_blocks": [],
    }

    updated_data = {
        "trains": [
            {
                "train_id": "API-TRAIN-A",
                "section": "Delhi-Ghaziabad",
                "start": "11:00",
                "end": "13:00",
            }
        ],

        "maintenance_jobs": [
            {
                "maintenance_id": "API-REC-A",
                "department": "Engineering",
                "section": "Delhi-Ghaziabad",

                "duration_minutes": 60,

                "priority": "HIGH",
                "deadline": "18:00",

                "urgency_score": 100,

                "shadow_compatible": False,
            }
        ],

        "sections": [
            "Delhi-Ghaziabad"
        ],
    }

    disruption = {
        "type": "TRAIN_DELAY",
        "section": "Delhi-Ghaziabad",

        "start": "11:00",
        "end": "13:00",

        "description":
            "Delayed train entered maintenance window",
    }

    response = client.post(
        "/planning/recover",
        json={
            "previous_result":
                previous_result,

            "updated_data":
                updated_data,

            "disruption":
                disruption,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert "recovered_schedule" in body
    assert "changes" in body
    assert "metrics" in body
    assert "planning_run_id" in body

    run_id = body[
        "planning_run_id"
    ]

    try:

        assert body["metrics"][
            "recovery_conflicts"
        ] == 0

    finally:

        delete_planning_run(
            run_id
        )


# ===========================================================================
# TEST 7 — CORS middleware
# ===========================================================================

def test_cors_middleware_present():

    assert len(
        app.user_middleware
    ) > 0


# ===========================================================================
# TEST 8 — Normal planning is saved and retrievable
# ===========================================================================

def test_planning_run_is_saved_and_retrievable():

    unique_id = str(
        uuid.uuid4()
    )[:8]

    maintenance_id = (
        "API-M-"
        + unique_id
    )

    payload = {
        "data": {
            "trains": [],

            "maintenance_jobs": [
                {
                    "maintenance_id":
                        maintenance_id,

                    "department":
                        "Engineering",

                    "section":
                        "Delhi-Ghaziabad",

                    "duration_minutes":
                        60,

                    "priority":
                        "CRITICAL",

                    "deadline":
                        "18:00",
                }
            ],

            "sections": [
                "Delhi-Ghaziabad"
            ],
        },

        "asset_features": [
            {
                "maintenance_id":
                    maintenance_id,

                "asset_age_years":
                    20,

                "days_since_last_maintenance":
                    250,

                "recent_defect_count":
                    4,

                "failures_last_12_months":
                    2,

                "traffic_load_percent":
                    90,
            }
        ],

        "planning_time":
            "08:00",
    }

    response = client.post(
        "/planning/run",
        json=payload,
    )

    assert response.status_code == 200

    body = response.json()

    assert "planning_run_id" in body

    run_id = body[
        "planning_run_id"
    ]

    try:

        stored_response = client.get(
            f"/planning/runs/{run_id}"
        )

        assert (
            stored_response.status_code
            == 200
        )

        stored = (
            stored_response.json()
        )

        assert stored[
            "run_id"
        ] == run_id

        assert stored[
            "planning_type"
        ] == "NORMAL"

        assert stored[
            "planning_time"
        ] == "08:00"

        ids = {
            block["maintenance_id"]
            for block
            in stored[
                "scheduled_blocks"
            ]
        }

        assert maintenance_id in ids

    finally:

        delete_planning_run(
            run_id
        )


# ===========================================================================
# TEST 9 — Latest endpoint returns saved Shadow planning run
# ===========================================================================

def test_latest_endpoint_returns_saved_run():

    unique_id = str(
        uuid.uuid4()
    )[:8]

    first_id = (
        "API-SHADOW-E-"
        + unique_id
    )

    second_id = (
        "API-SHADOW-S-"
        + unique_id
    )

    payload = {
        "data": {
            "trains": [],

            "maintenance_jobs": [
                {
                    "maintenance_id":
                        first_id,

                    "department":
                        "Engineering",

                    "section":
                        "Delhi-Ghaziabad",

                    "duration_minutes":
                        60,

                    "priority":
                        "HIGH",

                    "deadline":
                        "18:00",

                    "urgency_score":
                        120,

                    "shadow_compatible":
                        True,
                },

                {
                    "maintenance_id":
                        second_id,

                    "department":
                        "Signal & Telecom",

                    "section":
                        "Delhi-Ghaziabad",

                    "duration_minutes":
                        30,

                    "priority":
                        "MEDIUM",

                    "deadline":
                        "18:00",

                    "urgency_score":
                        90,

                    "shadow_compatible":
                        True,
                },
            ],

            "sections": [
                "Delhi-Ghaziabad"
            ],
        }
    }

    response = client.post(
        "/planning/shadow",
        json=payload,
    )

    assert response.status_code == 200

    body = response.json()

    assert "planning_run_id" in body

    run_id = body[
        "planning_run_id"
    ]

    try:

        latest_response = client.get(
            "/planning/latest",
            params={
                "planning_type":
                    "SHADOW"
            },
        )

        assert (
            latest_response.status_code
            == 200
        )

        latest = (
            latest_response.json()
        )

        assert latest[
            "run_id"
        ] == run_id

        assert latest[
            "planning_type"
        ] == "SHADOW"

    finally:

        delete_planning_run(
            run_id
        )


# ===========================================================================
# TEST 10 — Missing planning run returns 404
# ===========================================================================

def test_missing_run_returns_404():

    response = client.get(
        "/planning/runs/"
        "00000000-0000-0000-0000-000000000000"
    )

    assert response.status_code == 404


# ===========================================================================
# TEST 11 — Recovery run is persisted
# ===========================================================================

def test_recovery_run_is_saved():

    unique_id = str(
        uuid.uuid4()
    )[:8]

    maintenance_id = (
        "API-REC-PERSIST-"
        + unique_id
    )

    previous_result = {
        "scheduled_blocks": [
            {
                "maintenance_id":
                    maintenance_id,

                "section":
                    "Delhi-Ghaziabad",

                "department":
                    "Engineering",

                "priority":
                    "HIGH",

                "start":
                    "11:00",

                "end":
                    "12:00",

                "start_minutes":
                    660,

                "end_minutes":
                    720,

                "scheduled":
                    True,

                "shadowed":
                    False,
            }
        ],

        "shadow_blocks": [],
    }

    updated_data = {
        "trains": [
            {
                "train_id":
                    "API-T-PERSIST-"
                    + unique_id,

                "section":
                    "Delhi-Ghaziabad",

                "start":
                    "11:00",

                "end":
                    "13:00",
            }
        ],

        "maintenance_jobs": [
            {
                "maintenance_id":
                    maintenance_id,

                "department":
                    "Engineering",

                "section":
                    "Delhi-Ghaziabad",

                "duration_minutes":
                    60,

                "priority":
                    "HIGH",

                "deadline":
                    "18:00",

                "urgency_score":
                    100,

                "shadow_compatible":
                    False,
            }
        ],

        "sections": [
            "Delhi-Ghaziabad"
        ],
    }

    disruption = {
        "type":
            "TRAIN_DELAY",

        "section":
            "Delhi-Ghaziabad",

        "start":
            "11:00",

        "end":
            "13:00",
    }

    response = client.post(
        "/planning/recover",
        json={
            "previous_result":
                previous_result,

            "updated_data":
                updated_data,

            "disruption":
                disruption,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert "planning_run_id" in body

    run_id = body[
        "planning_run_id"
    ]

    try:

        stored_response = client.get(
            f"/planning/runs/{run_id}"
        )

        assert (
            stored_response.status_code
            == 200
        )

        stored = (
            stored_response.json()
        )

        assert stored[
            "planning_type"
        ] == "RECOVERY"

        assert stored[
            "metrics"
        ][
            "recovery_conflicts"
        ] == 0

    finally:

        delete_planning_run(
            run_id
        )