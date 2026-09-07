from fastapi.testclient import TestClient

from api import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ok"
    assert body["service"] == "RailBlock AI"
    assert body["ai_model"] == "RandomForestClassifier"


def test_planning_run_rejects_missing_data():
    response = client.post(
        "/planning/run",
        json={
            "asset_features": [],
            "planning_time": "08:00",
        },
    )

    assert response.status_code == 400
    assert "data" in response.json()["detail"]


def test_shadow_endpoint_rejects_missing_data():
    response = client.post(
        "/planning/shadow",
        json={},
    )

    assert response.status_code == 400
    assert "data" in response.json()["detail"]


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
    assert "previous_result" in response.json()["detail"]


def test_shadow_endpoint_runs():
    payload = {
        "data": {
            "trains": [],
            "maintenance_jobs": [
                {
                    "maintenance_id": "M001",
                    "department": "Engineering",
                    "section": "Delhi-Ghaziabad",
                    "duration_minutes": 60,
                    "priority": "HIGH",
                    "deadline": "18:00",
                    "urgency_score": 100,
                    "shadow_compatible": True,
                },
                {
                    "maintenance_id": "M002",
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

    assert body["metrics"]["scheduled"] == 2
    assert body["metrics"]["shadow_blocks_scheduled"] == 1
    assert body["metrics"]["conflicts"] == 0


def test_recover_endpoint_runs():
    previous_result = {
        "scheduled_blocks": [
            {
                "maintenance_id": "M001",
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
                "train_id": "T001",
                "section": "Delhi-Ghaziabad",
                "start": "11:00",
                "end": "13:00",
            }
        ],
        "maintenance_jobs": [
            {
                "maintenance_id": "M001",
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
        "description": "Delayed train entered maintenance window",
    }

    response = client.post(
        "/planning/recover",
        json={
            "previous_result": previous_result,
            "updated_data": updated_data,
            "disruption": disruption,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert "recovered_schedule" in body
    assert "changes" in body
    assert "metrics" in body

    assert body["metrics"]["recovery_conflicts"] == 0


def test_cors_middleware_present():
    # Simple sanity check that the FastAPI app has middleware configured.
    assert len(app.user_middleware) > 0