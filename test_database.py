"""
RailBlock AI — PostgreSQL Database Tests

These tests verify that the Python backend can:
- connect to PostgreSQL
- see the expected tables
- insert and read records
- store planning runs and scheduled blocks
- preserve relationships correctly
"""

import uuid

from sqlalchemy import inspect

from database import (
    SessionLocal,
    engine,
    test_database_connection as check_database_connection,
)
from models import (
    Section,
    Train,
    MaintenanceRequest,
    AssetFeature,
    PlanningRun,
    ScheduledBlock,
)


# ---------------------------------------------------------------------------
# TEST 1
# Database connection
# ---------------------------------------------------------------------------

def test_postgresql_connection():
    assert check_database_connection() is True


# ---------------------------------------------------------------------------
# TEST 2
# Required tables exist
# ---------------------------------------------------------------------------

def test_required_tables_exist():

    inspector = inspect(engine)

    tables = set(
        inspector.get_table_names()
    )

    required = {
        "sections",
        "trains",
        "maintenance_requests",
        "asset_features",
        "planning_runs",
        "scheduled_blocks",
    }

    assert required.issubset(tables)


# ---------------------------------------------------------------------------
# TEST 3
# Section insert + read
# ---------------------------------------------------------------------------

def test_insert_and_read_section():

    db = SessionLocal()

    section_name = (
        "TEST-SECTION-"
        + str(uuid.uuid4())[:8]
    )

    try:

        section = Section(
            section_name=section_name
        )

        db.add(section)
        db.commit()
        db.refresh(section)

        stored = (
            db.query(Section)
            .filter(
                Section.section_name
                == section_name
            )
            .first()
        )

        assert stored is not None
        assert stored.section_name == section_name

    finally:

        db.query(Section).filter(
            Section.section_name
            == section_name
        ).delete()

        db.commit()
        db.close()


# ---------------------------------------------------------------------------
# TEST 4
# Train insert + read
# ---------------------------------------------------------------------------

def test_insert_and_read_train():

    db = SessionLocal()

    train_id = (
        "TEST-T-"
        + str(uuid.uuid4())[:8]
    )

    try:

        train = Train(
            train_id=train_id,
            train_name="Test Express",
            section="Delhi-Ghaziabad",
            start_time="08:00",
            end_time="08:30",
            priority="HIGH",
            raw_data={
                "source": "pytest"
            },
        )

        db.add(train)
        db.commit()
        db.refresh(train)

        stored = (
            db.query(Train)
            .filter(
                Train.train_id
                == train_id
            )
            .first()
        )

        assert stored is not None
        assert stored.train_name == "Test Express"
        assert stored.start_time == "08:00"
        assert stored.end_time == "08:30"

    finally:

        db.query(Train).filter(
            Train.train_id
            == train_id
        ).delete()

        db.commit()
        db.close()


# ---------------------------------------------------------------------------
# TEST 5
# Maintenance request insert + read
# ---------------------------------------------------------------------------

def test_insert_and_read_maintenance_request():

    db = SessionLocal()

    maintenance_id = (
        "TEST-M-"
        + str(uuid.uuid4())[:8]
    )

    try:

        request = MaintenanceRequest(
            maintenance_id=maintenance_id,
            department="Engineering",
            section="Delhi-Ghaziabad",
            duration_minutes=60,
            priority="CRITICAL",
            deadline="14:00",
            shadow_compatible=True,
            status="PENDING",
        )

        db.add(request)
        db.commit()
        db.refresh(request)

        stored = (
            db.query(MaintenanceRequest)
            .filter(
                MaintenanceRequest.maintenance_id
                == maintenance_id
            )
            .first()
        )

        assert stored is not None
        assert stored.priority == "CRITICAL"
        assert stored.duration_minutes == 60
        assert stored.shadow_compatible is True

    finally:

        db.query(MaintenanceRequest).filter(
            MaintenanceRequest.maintenance_id
            == maintenance_id
        ).delete()

        db.commit()
        db.close()


# ---------------------------------------------------------------------------
# TEST 6
# Asset features
# ---------------------------------------------------------------------------

def test_insert_and_read_asset_features():

    db = SessionLocal()

    maintenance_id = (
        "TEST-AI-"
        + str(uuid.uuid4())[:8]
    )

    try:

        features = AssetFeature(
            maintenance_id=maintenance_id,
            asset_age_years=20,
            days_since_last_maintenance=250,
            recent_defect_count=4,
            failures_last_12_months=2,
            traffic_load_percent=90,
            predicted_risk_score=0.88,
            predicted_risk_level="HIGH",
        )

        db.add(features)
        db.commit()
        db.refresh(features)

        stored = (
            db.query(AssetFeature)
            .filter(
                AssetFeature.maintenance_id
                == maintenance_id
            )
            .first()
        )

        assert stored is not None
        assert stored.predicted_risk_score == 0.88
        assert stored.predicted_risk_level == "HIGH"
        assert stored.recent_defect_count == 4

    finally:

        db.query(AssetFeature).filter(
            AssetFeature.maintenance_id
            == maintenance_id
        ).delete()

        db.commit()
        db.close()


# ---------------------------------------------------------------------------
# TEST 7
# Planning run
# ---------------------------------------------------------------------------

def test_insert_and_read_planning_run():

    db = SessionLocal()

    run_id = str(uuid.uuid4())

    try:

        run = PlanningRun(
            run_id=run_id,
            planning_type="TEST",
            planning_time="08:00",
            status="COMPLETED",
            input_snapshot={
                "test": True
            },
            metrics={
                "scheduled": 2,
                "conflicts": 0,
            },
        )

        db.add(run)
        db.commit()
        db.refresh(run)

        stored = (
            db.query(PlanningRun)
            .filter(
                PlanningRun.run_id
                == run_id
            )
            .first()
        )

        assert stored is not None
        assert stored.planning_type == "TEST"
        assert stored.metrics["conflicts"] == 0

    finally:

        db.query(PlanningRun).filter(
            PlanningRun.run_id
            == run_id
        ).delete()

        db.commit()
        db.close()


# ---------------------------------------------------------------------------
# TEST 8
# Planning run + scheduled block relationship
# ---------------------------------------------------------------------------

def test_planning_run_stores_scheduled_block():

    db = SessionLocal()

    run_id = str(uuid.uuid4())

    maintenance_id = (
        "TEST-BLOCK-"
        + str(uuid.uuid4())[:8]
    )

    try:

        run = PlanningRun(
            run_id=run_id,
            planning_type="TEST",
            planning_time="08:00",
            status="COMPLETED",
        )

        block = ScheduledBlock(
            planning_run_id=run_id,
            maintenance_id=maintenance_id,
            section="Delhi-Ghaziabad",
            department="Engineering",
            priority="CRITICAL",
            start_time="09:00",
            end_time="10:00",
            start_minutes=540,
            end_minutes=600,
            predicted_risk_score=0.91,
            urgency_score=155,
            scheduled=True,
        )

        db.add(run)
        db.add(block)

        db.commit()

        stored_run = (
            db.query(PlanningRun)
            .filter(
                PlanningRun.run_id
                == run_id
            )
            .first()
        )

        stored_block = (
            db.query(ScheduledBlock)
            .filter(
                ScheduledBlock.maintenance_id
                == maintenance_id
            )
            .first()
        )

        assert stored_run is not None
        assert stored_block is not None

        assert stored_block.planning_run_id == run_id
        assert stored_block.start_time == "09:00"
        assert stored_block.end_time == "10:00"
        assert stored_block.urgency_score == 155

        assert len(
            stored_run.scheduled_blocks
        ) == 1

    finally:

        db.query(ScheduledBlock).filter(
            ScheduledBlock.maintenance_id
            == maintenance_id
        ).delete()

        db.query(PlanningRun).filter(
            PlanningRun.run_id
            == run_id
        ).delete()

        db.commit()
        db.close()