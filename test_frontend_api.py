import uuid

from fastapi.testclient import TestClient

from api import app
from database import SessionLocal
from models import MaintenanceRequest, Train


client = TestClient(app)


# =========================================================
# HELPERS
# =========================================================


def unique_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def cleanup(
    train_ids=None,
    maintenance_ids=None,
):
    train_ids = train_ids or []
    maintenance_ids = maintenance_ids or []

    db = SessionLocal()

    try:
        if train_ids:
            (
                db.query(Train)
                .filter(
                    Train.train_id.in_(
                        train_ids
                    )
                )
                .delete(
                    synchronize_session=False
                )
            )

        if maintenance_ids:
            (
                db.query(
                    MaintenanceRequest
                )
                .filter(
                    MaintenanceRequest
                    .maintenance_id
                    .in_(
                        maintenance_ids
                    )
                )
                .delete(
                    synchronize_session=False
                )
            )

        db.commit()

    finally:
        db.close()


# =========================================================
# DASHBOARD
# =========================================================


def test_dashboard_endpoint():
    response = client.get(
        "/dashboard"
    )

    assert response.status_code == 200

    data = response.json()

    assert "metrics" in data
    assert (
        "recentScheduledBlocks"
        in data
    )
    assert "highRiskJobs" in data


# =========================================================
# TRAINS
# =========================================================


def test_train_endpoints():
    train_id = unique_id("TAPI")

    try:
        create_response = client.post(
            "/trains",
            json={
                "train_id":
                    train_id,

                "train_name":
                    "API Test Express",

                "section":
                    "Delhi-Ghaziabad",

                "start":
                    "06:30",

                "end":
                    "07:00",

                "priority":
                    "HIGH",
            },
        )

        assert (
            create_response.status_code
            == 201
        )

        created = (
            create_response.json()
        )

        assert (
            created["train_id"]
            == train_id
        )

        list_response = client.get(
            "/trains"
        )

        assert (
            list_response.status_code
            == 200
        )

        trains = (
            list_response.json()
        )

        assert any(
            train["train_id"]
            == train_id
            for train in trains
        )

    finally:
        cleanup(
            train_ids=[
                train_id
            ]
        )


# =========================================================
# MAINTENANCE
# =========================================================


def test_maintenance_endpoints():
    maintenance_id = unique_id(
        "MAPI"
    )

    try:
        create_response = client.post(
            "/maintenance",
            json={
                "maintenance_id":
                    maintenance_id,

                "department":
                    "Engineering",

                "section":
                    "Mumbai-Pune",

                "duration_minutes":
                    45,

                "priority":
                    "CRITICAL",

                "deadline":
                    "18:00",

                "status":
                    "PENDING",

                "shadow_compatible":
                    True,

                "predicted_risk_score":
                    0.93,

                "predicted_risk_level":
                    "HIGH",

                "priority_score":
                    100,

                "deadline_bonus":
                    30,

                "ai_risk_bonus":
                    37,

                "urgency_score":
                    167,
            },
        )

        assert (
            create_response.status_code
            == 201
        )

        list_response = client.get(
            "/maintenance"
        )

        assert (
            list_response.status_code
            == 200
        )

        jobs = (
            list_response.json()
        )

        stored = next(
            (
                job
                for job in jobs
                if job[
                    "maintenance_id"
                ]
                == maintenance_id
            ),
            None,
        )

        assert stored is not None

        assert (
            stored[
                "predicted_risk_score"
            ]
            == 0.93
        )

        assert (
            stored["urgency_score"]
            == 167
        )

    finally:
        cleanup(
            maintenance_ids=[
                maintenance_id
            ]
        )


# =========================================================
# AI RISK
# =========================================================


def test_ai_risk_endpoint():
    response = client.get(
        "/ai/risk"
    )

    assert response.status_code == 200

    assert isinstance(
        response.json(),
        list,
    )


# =========================================================
# PRIORITY
# =========================================================


def test_priority_endpoint():
    response = client.get(
        "/priority"
    )

    assert response.status_code == 200

    assert isinstance(
        response.json(),
        list,
    )