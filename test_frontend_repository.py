import uuid

import pytest

from database import SessionLocal
from models import MaintenanceRequest, Train
from repository import (
    add_maintenance_record,
    add_train_record,
    get_ai_risk_analysis,
    get_all_maintenance_requests,
    get_all_trains,
    get_dashboard_summary,
    get_priority_analysis,
)


# =========================================================
# TEST HELPERS
# =========================================================


def unique_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def cleanup(train_ids=None, maintenance_ids=None):
    train_ids = train_ids or []
    maintenance_ids = maintenance_ids or []

    db = SessionLocal()

    try:
        if train_ids:
            (
                db.query(Train)
                .filter(Train.train_id.in_(train_ids))
                .delete(synchronize_session=False)
            )

        if maintenance_ids:
            (
                db.query(MaintenanceRequest)
                .filter(
                    MaintenanceRequest.maintenance_id.in_(
                        maintenance_ids
                    )
                )
                .delete(synchronize_session=False)
            )

        db.commit()

    finally:
        db.close()


# =========================================================
# TRAIN TEST
# =========================================================


def test_add_and_get_train():
    train_id = unique_id("TTEST")

    try:
        created = add_train_record({
            "train_id": train_id,
            "train_name": "Repository Test Express",
            "section": "Delhi-Ghaziabad",
            "start": "06:00",
            "end": "06:30",
            "priority": "HIGH",
        })

        assert created["train_id"] == train_id
        assert created["train_name"] == "Repository Test Express"
        assert created["start"] == "06:00"
        assert created["end"] == "06:30"

        trains = get_all_trains()

        stored = next(
            (
                train
                for train in trains
                if train["train_id"] == train_id
            ),
            None,
        )

        assert stored is not None
        assert stored["section"] == "Delhi-Ghaziabad"
        assert stored["priority"] == "HIGH"

    finally:
        cleanup(train_ids=[train_id])


# =========================================================
# MAINTENANCE TEST
# =========================================================


def test_add_and_get_maintenance():
    maintenance_id = unique_id("MTEST")

    try:
        created = add_maintenance_record({
            "maintenance_id": maintenance_id,
            "department": "Engineering",
            "section": "Delhi-Ghaziabad",
            "duration_minutes": 60,
            "priority": "CRITICAL",
            "deadline": "18:00",
            "status": "PENDING",
            "shadow_compatible": True,
            "predicted_risk_score": 0.91,
            "predicted_risk_level": "HIGH",
            "priority_score": 100,
            "deadline_bonus": 30,
            "ai_risk_bonus": 36,
            "urgency_score": 166,
        })

        assert created["maintenance_id"] == maintenance_id
        assert created["priority"] == "CRITICAL"

        jobs = get_all_maintenance_requests()

        stored = next(
            (
                job
                for job in jobs
                if job["maintenance_id"] == maintenance_id
            ),
            None,
        )

        assert stored is not None

        assert stored["department"] == "Engineering"

        assert stored["duration_minutes"] == 60

        assert stored["predicted_risk_score"] == 0.91

        assert stored["predicted_risk_level"] == "HIGH"

        assert stored["urgency_score"] == 166

    finally:
        cleanup(
            maintenance_ids=[maintenance_id]
        )


# =========================================================
# AI RISK ORDERING TEST
# =========================================================


def test_ai_risk_analysis_orders_highest_first():
    high_id = unique_id("MHIGH")
    low_id = unique_id("MLOW")

    try:
        add_maintenance_record({
            "maintenance_id": high_id,
            "department": "Engineering",
            "section": "Delhi-Ghaziabad",
            "duration_minutes": 30,
            "priority": "HIGH",
            "deadline": "15:00",
            "predicted_risk_score": 0.95,
            "predicted_risk_level": "HIGH",
            "urgency_score": 120,
        })

        add_maintenance_record({
            "maintenance_id": low_id,
            "department": "Signal & Telecom",
            "section": "Mumbai-Pune",
            "duration_minutes": 30,
            "priority": "LOW",
            "deadline": "22:00",
            "predicted_risk_score": 0.10,
            "predicted_risk_level": "LOW",
            "urgency_score": 10,
        })

        result = get_ai_risk_analysis()

        ids = [
            job["maintenance_id"]
            for job in result
        ]

        assert high_id in ids
        assert low_id in ids

        assert ids.index(high_id) < ids.index(low_id)

    finally:
        cleanup(
            maintenance_ids=[
                high_id,
                low_id,
            ]
        )


# =========================================================
# PRIORITY ORDERING TEST
# =========================================================


def test_priority_analysis_orders_highest_urgency_first():
    high_id = unique_id("MURGENT")
    low_id = unique_id("MNORMAL")

    try:
        add_maintenance_record({
            "maintenance_id": high_id,
            "department": "Engineering",
            "section": "Delhi-Ghaziabad",
            "duration_minutes": 45,
            "priority": "CRITICAL",
            "deadline": "14:00",
            "urgency_score": 190,
        })

        add_maintenance_record({
            "maintenance_id": low_id,
            "department": "Engineering",
            "section": "Delhi-Ghaziabad",
            "duration_minutes": 45,
            "priority": "LOW",
            "deadline": "23:00",
            "urgency_score": 5,
        })

        result = get_priority_analysis()

        ids = [
            job["maintenance_id"]
            for job in result
        ]

        assert high_id in ids
        assert low_id in ids

        assert ids.index(high_id) < ids.index(low_id)

    finally:
        cleanup(
            maintenance_ids=[
                high_id,
                low_id,
            ]
        )


# =========================================================
# DASHBOARD TEST
# =========================================================


def test_dashboard_summary_uses_database_data():
    train_id = unique_id("TDASH")
    maintenance_id = unique_id("MDASH")

    before = get_dashboard_summary()

    try:
        add_train_record({
            "train_id": train_id,
            "train_name": "Dashboard Test Train",
            "section": "Mumbai-Pune",
            "start": "05:00",
            "end": "05:30",
            "priority": "MEDIUM",
        })

        add_maintenance_record({
            "maintenance_id": maintenance_id,
            "department": "Engineering",
            "section": "Mumbai-Pune",
            "duration_minutes": 30,
            "priority": "CRITICAL",
            "deadline": "17:00",
            "predicted_risk_score": 0.90,
            "predicted_risk_level": "HIGH",
            "urgency_score": 150,
        })

        after = get_dashboard_summary()

        assert (
            after["metrics"]["live_train_movements"]
            ==
            before["metrics"]["live_train_movements"] + 1
        )

        assert (
            after["metrics"]["total_jobs"]
            ==
            before["metrics"]["total_jobs"] + 1
        )

        assert (
            after["metrics"]["critical_jobs"]
            ==
            before["metrics"]["critical_jobs"] + 1
        )

        assert (
            after["metrics"]["high_risk_assets"]
            ==
            before["metrics"]["high_risk_assets"] + 1
        )

        assert "recentScheduledBlocks" in after
        assert "highRiskJobs" in after

    finally:
        cleanup(
            train_ids=[train_id],
            maintenance_ids=[maintenance_id],
        )