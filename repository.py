"""
RailBlock AI — Database Repository Layer

This module keeps PostgreSQL persistence logic separate from FastAPI.

It can:
- save planning runs
- save scheduled maintenance blocks
- retrieve one planning run
- retrieve the latest planning run
- delete a planning run (mainly useful for tests)

No AI or optimization logic belongs here.
"""

from copy import deepcopy
from typing import Optional

from sqlalchemy import desc

from database import SessionLocal
from models import PlanningRun, ScheduledBlock
from models import (
    AssetFeature,
    MaintenanceRequest,
    PlanningRun,
    ScheduledBlock,
    Train,
)


# ---------------------------------------------------------------------------
# Time helper
# ---------------------------------------------------------------------------

def _hhmm_to_minutes(value: str) -> int:
    """
    Convert HH:MM to minutes since midnight.
    """

    hour, minute = value.split(":")

    return int(hour) * 60 + int(minute)


# ---------------------------------------------------------------------------
# Scheduled block conversion
# ---------------------------------------------------------------------------

def _scheduled_block_from_dict(
    planning_run_id: str,
    block: dict,
) -> ScheduledBlock:
    """
    Convert one RailBlock AI schedule dictionary into
    a ScheduledBlock SQLAlchemy model.
    """

    start_time = block.get(
        "start",
        "00:00",
    )

    end_time = block.get(
        "end",
        "00:00",
    )

    start_minutes = block.get(
        "start_minutes"
    )

    end_minutes = block.get(
        "end_minutes"
    )

    if start_minutes is None:
        start_minutes = _hhmm_to_minutes(
            start_time
        )

    if end_minutes is None:
        end_minutes = _hhmm_to_minutes(
            end_time
        )

    return ScheduledBlock(
        planning_run_id=planning_run_id,

        maintenance_id=block[
            "maintenance_id"
        ],

        shadow_block_id=block.get(
            "shadow_block_id"
        ),

        section=block[
            "section"
        ],

        department=block.get(
            "department",
            "UNKNOWN",
        ),

        priority=block.get(
            "priority",
            "LOW",
        ),

        start_time=start_time,
        end_time=end_time,

        start_minutes=int(
            start_minutes
        ),

        end_minutes=int(
            end_minutes
        ),

        predicted_risk_score=block.get(
            "predicted_risk_score"
        ),

        urgency_score=block.get(
            "urgency_score"
        ),

        scheduled=bool(
            block.get(
                "scheduled",
                True,
            )
        ),

        # Store the complete original block as JSON too.
        # This keeps extra AI / shadow / recovery fields available
        # even if they do not have dedicated DB columns.
        raw_data=deepcopy(
            block
        ),
    )


# ---------------------------------------------------------------------------
# Save scheduled blocks
# ---------------------------------------------------------------------------

def save_scheduled_blocks(
    planning_run_id: str,
    blocks: list,
) -> int:
    """
    Save scheduled blocks for an existing planning run.

    Returns the number of blocks stored.
    """

    db = SessionLocal()

    try:

        planning_run = (
            db.query(PlanningRun)
            .filter(
                PlanningRun.run_id
                == planning_run_id
            )
            .first()
        )

        if planning_run is None:
            raise ValueError(
                f"planning run '{planning_run_id}' does not exist"
            )

        stored_count = 0

        for block in blocks:

            scheduled_block = (
                _scheduled_block_from_dict(
                    planning_run_id,
                    block,
                )
            )

            db.add(
                scheduled_block
            )

            stored_count += 1

        db.commit()

        return stored_count

    except Exception:

        db.rollback()
        raise

    finally:

        db.close()


# ---------------------------------------------------------------------------
# Save complete planning run
# ---------------------------------------------------------------------------

def save_planning_run(
    result: dict,
    planning_type: str = "NORMAL",
    planning_time: Optional[str] = None,
    input_snapshot: Optional[dict] = None,
) -> str:
    """
    Save one complete RailBlock AI planning result.

    Supported result forms:

        Standard planning:
            result["scheduled_blocks"]

        Shadow Block planning:
            result["scheduled_blocks"]

        Rolling Horizon recovery:
            result["recovered_schedule"]

    Returns
    -------
    str
        The generated planning run UUID.
    """

    db = SessionLocal()

    try:

        # ---------------------------------------------------------------
        # Determine which schedule list exists.
        # ---------------------------------------------------------------

        if "scheduled_blocks" in result:

            blocks = result.get(
                "scheduled_blocks",
                [],
            )

        elif "recovered_schedule" in result:

            blocks = result.get(
                "recovered_schedule",
                [],
            )

        else:

            blocks = []

        # ---------------------------------------------------------------
        # Create PlanningRun
        # ---------------------------------------------------------------

        run = PlanningRun(
            planning_type=planning_type,
            planning_time=planning_time,
            status="COMPLETED",
            input_snapshot=deepcopy(
                input_snapshot
            ),
            metrics=deepcopy(
                result.get(
                    "metrics",
                    {},
                )
            ),
        )

        db.add(run)

        # Flush so SQLAlchemy generates run_id before commit.
        db.flush()

        run_id = run.run_id

        # ---------------------------------------------------------------
        # Store scheduled blocks
        # ---------------------------------------------------------------

        for block in blocks:

            db.add(
                _scheduled_block_from_dict(
                    run_id,
                    block,
                )
            )

        db.commit()

        return run_id

    except Exception:

        db.rollback()
        raise

    finally:

        db.close()


# ---------------------------------------------------------------------------
# Convert database objects back to dictionaries
# ---------------------------------------------------------------------------

def _block_to_dict(
    block: ScheduledBlock,
) -> dict:
    """
    Convert ScheduledBlock SQLAlchemy object to JSON-friendly dict.
    """

    return {
        "id":
            block.id,

        "maintenance_id":
            block.maintenance_id,

        "shadow_block_id":
            block.shadow_block_id,

        "section":
            block.section,

        "department":
            block.department,

        "priority":
            block.priority,

        "start":
            block.start_time,

        "end":
            block.end_time,

        "start_minutes":
            block.start_minutes,

        "end_minutes":
            block.end_minutes,

        "predicted_risk_score":
            block.predicted_risk_score,

        "urgency_score":
            block.urgency_score,

        "scheduled":
            block.scheduled,

        "raw_data":
            block.raw_data,
    }


def _planning_run_to_dict(
    run: PlanningRun,
) -> dict:
    """
    Convert PlanningRun SQLAlchemy object to JSON-friendly dict.
    """

    return {
        "run_id":
            run.run_id,

        "planning_type":
            run.planning_type,

        "planning_time":
            run.planning_time,

        "status":
            run.status,

        "input_snapshot":
            run.input_snapshot,

        "metrics":
            run.metrics,

        "created_at":
            (
                run.created_at.isoformat()
                if run.created_at
                else None
            ),

        "scheduled_blocks": [
            _block_to_dict(block)
            for block
            in run.scheduled_blocks
        ],
    }


# ---------------------------------------------------------------------------
# Get one planning run
# ---------------------------------------------------------------------------

def get_planning_run(
    run_id: str,
) -> Optional[dict]:
    """
    Retrieve a planning run and its scheduled blocks.

    Returns None if the run does not exist.
    """

    db = SessionLocal()

    try:

        run = (
            db.query(PlanningRun)
            .filter(
                PlanningRun.run_id
                == run_id
            )
            .first()
        )

        if run is None:
            return None

        # Convert while session is still open so relationship data loads.
        return _planning_run_to_dict(
            run
        )

    finally:

        db.close()


# ---------------------------------------------------------------------------
# Get latest planning run
# ---------------------------------------------------------------------------

def get_latest_planning_run(
    planning_type: Optional[str] = None,
) -> Optional[dict]:
    """
    Return the most recently created planning run.

    Optional:
        planning_type="NORMAL"
        planning_type="SHADOW"
        planning_type="RECOVERY"
    """

    db = SessionLocal()

    try:

        query = db.query(
            PlanningRun
        )

        if planning_type is not None:

            query = query.filter(
                PlanningRun.planning_type
                == planning_type
            )

        run = (
            query
            .order_by(
                desc(
                    PlanningRun.created_at
                )
            )
            .first()
        )

        if run is None:
            return None

        return _planning_run_to_dict(
            run
        )

    finally:

        db.close()


# ---------------------------------------------------------------------------
# Delete planning run
# ---------------------------------------------------------------------------

def delete_planning_run(
    run_id: str,
) -> bool:
    """
    Delete one planning run.

    Because scheduled_blocks has ON DELETE CASCADE / ORM cascade,
    its associated blocks are removed too.

    Mainly useful for automated tests and cleanup.

    Returns:
        True  -> record existed and was deleted
        False -> record did not exist
    """

    db = SessionLocal()

    try:

        run = (
            db.query(PlanningRun)
            .filter(
                PlanningRun.run_id
                == run_id
            )
            .first()
        )

        if run is None:
            return False

        db.delete(
            run
        )

        db.commit()

        return True

    except Exception:

        db.rollback()
        raise

    finally:

        db.close()




# =============================================================================
# FRONTEND DATA REPOSITORY
# =============================================================================


def get_all_trains() -> list[dict]:
    """
    Return all stored train movements in the exact shape
    expected by the React frontend.
    """

    db = SessionLocal()

    try:
        trains = (
            db.query(Train)
            .order_by(
                Train.section,
                Train.start_time,
                Train.train_id,
            )
            .all()
        )

        return [
            {
                "train_id": train.train_id,
                "train_name": train.train_name,
                "section": train.section,
                "start": train.start_time,
                "end": train.end_time,
                "priority": train.priority,
            }
            for train in trains
        ]

    finally:
        db.close()


def add_train_record(train_data: dict) -> dict:
    """
    Save one train movement to PostgreSQL.
    """

    required = [
        "train_id",
        "section",
        "start",
        "end",
    ]

    for field in required:
        if field not in train_data:
            raise ValueError(
                f"train must contain '{field}'"
            )

    db = SessionLocal()

    try:
        train = Train(
            train_id=train_data["train_id"],
            train_name=train_data.get(
                "train_name"
            ),
            section=train_data["section"],
            start_time=train_data["start"],
            end_time=train_data["end"],
            priority=train_data.get(
                "priority"
            ),
            raw_data=deepcopy(
                train_data
            ),
        )

        db.add(train)
        db.commit()
        db.refresh(train)

        return {
            "train_id": train.train_id,
            "train_name": train.train_name,
            "section": train.section,
            "start": train.start_time,
            "end": train.end_time,
            "priority": train.priority,
        }

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


def get_all_maintenance_requests() -> list[dict]:
    """
    Return maintenance requests in the shape expected by
    the React frontend.

    If AI risk information exists for a job, the newest
    AssetFeature row is merged into the maintenance record.
    """

    db = SessionLocal()

    try:
        requests = (
            db.query(MaintenanceRequest)
            .order_by(
                MaintenanceRequest.created_at,
                MaintenanceRequest.maintenance_id,
            )
            .all()
        )

        output = []

        for request_row in requests:
            result = {
                "maintenance_id":
                    request_row.maintenance_id,

                "department":
                    request_row.department,

                "section":
                    request_row.section,

                "duration_minutes":
                    request_row.duration_minutes,

                "priority":
                    request_row.priority,

                "deadline":
                    request_row.deadline,

                "status":
                    request_row.status,

                "shadow_compatible":
                    request_row.shadow_compatible,
            }

            # Preserve extra fields originally submitted.
            if isinstance(
                request_row.raw_data,
                dict,
            ):
                result.update(
                    request_row.raw_data
                )

            latest_ai = (
                db.query(AssetFeature)
                .filter(
                    AssetFeature.maintenance_id
                    == request_row.maintenance_id
                )
                .order_by(
                    desc(
                        AssetFeature.created_at
                    )
                )
                .first()
            )

            if latest_ai is not None:
                result.update({
                    "asset_age_years":
                        latest_ai.asset_age_years,

                    "days_since_last_maintenance":
                        latest_ai.days_since_last_maintenance,

                    "recent_defect_count":
                        latest_ai.recent_defect_count,

                    "failures_last_12_months":
                        latest_ai.failures_last_12_months,

                    "traffic_load_percent":
                        latest_ai.traffic_load_percent,

                    "predicted_risk_score":
                        latest_ai.predicted_risk_score,

                    "predicted_risk_level":
                        latest_ai.predicted_risk_level,
                })

            output.append(result)

        return output

    finally:
        db.close()


def add_maintenance_record(
    maintenance_data: dict,
) -> dict:
    """
    Store one maintenance request.

    Extra frontend fields are preserved inside raw_data.
    """

    required = [
        "maintenance_id",
        "department",
        "section",
        "duration_minutes",
        "priority",
        "deadline",
    ]

    for field in required:
        if field not in maintenance_data:
            raise ValueError(
                f"maintenance request must contain '{field}'"
            )

    db = SessionLocal()

    try:
        existing = (
            db.query(MaintenanceRequest)
            .filter(
                MaintenanceRequest.maintenance_id
                == maintenance_data[
                    "maintenance_id"
                ]
            )
            .first()
        )

        if existing is not None:
            raise ValueError(
                "maintenance_id already exists"
            )

        row = MaintenanceRequest(
            maintenance_id=
                maintenance_data[
                    "maintenance_id"
                ],

            department=
                maintenance_data[
                    "department"
                ],

            section=
                maintenance_data[
                    "section"
                ],

            duration_minutes=int(
                maintenance_data[
                    "duration_minutes"
                ]
            ),

            priority=
                maintenance_data[
                    "priority"
                ],

            deadline=
                maintenance_data[
                    "deadline"
                ],

            shadow_compatible=bool(
                maintenance_data.get(
                    "shadow_compatible",
                    False,
                )
            ),

            status=
                maintenance_data.get(
                    "status",
                    "PENDING",
                ),

            raw_data=deepcopy(
                maintenance_data
            ),
        )

        db.add(row)
        db.commit()
        db.refresh(row)

        return {
            "maintenance_id":
                row.maintenance_id,

            "department":
                row.department,

            "section":
                row.section,

            "duration_minutes":
                row.duration_minutes,

            "priority":
                row.priority,

            "deadline":
                row.deadline,

            "status":
                row.status,

            "shadow_compatible":
                row.shadow_compatible,
        }

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


def get_ai_risk_analysis() -> list[dict]:
    """
    Return maintenance jobs ordered from highest to lowest
    predicted AI risk.

    Only jobs that currently have a predicted risk score
    are included.
    """

    jobs = get_all_maintenance_requests()

    jobs = [
        job
        for job in jobs
        if job.get(
            "predicted_risk_score"
        ) is not None
    ]

    jobs.sort(
        key=lambda job:
            (
                -float(
                    job.get(
                        "predicted_risk_score",
                        0,
                    )
                ),
                job["maintenance_id"],
            )
    )

    return jobs


def get_priority_analysis() -> list[dict]:
    """
    Return maintenance jobs ordered by urgency score.

    Extra priority fields stored in raw_data such as:
      priority_score
      deadline_bonus
      ai_risk_bonus
      urgency_score

    are preserved automatically by
    get_all_maintenance_requests().
    """

    jobs = get_all_maintenance_requests()

    jobs.sort(
        key=lambda job:
            (
                -int(
                    job.get(
                        "urgency_score",
                        0,
                    )
                ),
                job.get(
                    "deadline",
                    "24:00",
                ),
                job["maintenance_id"],
            )
    )

    return jobs


def get_dashboard_summary() -> dict:
    """
    Build dashboard metrics using real PostgreSQL data.
    """

    trains = get_all_trains()
    maintenance = (
        get_all_maintenance_requests()
    )

    latest = (
        get_latest_planning_run()
    )

    scheduled_blocks = []

    latest_metrics = {}

    if latest is not None:
        scheduled_blocks = latest.get(
            "scheduled_blocks",
            [],
        )

        latest_metrics = latest.get(
            "metrics",
            {},
        ) or {}

    high_risk_jobs = [
        job
        for job in maintenance
        if job.get(
            "predicted_risk_level"
        ) == "HIGH"
    ]

    high_risk_jobs.sort(
        key=lambda job:
            -float(
                job.get(
                    "predicted_risk_score",
                    0,
                )
            )
    )

    scheduled_count = (
        latest_metrics.get(
            "scheduled"
        )
    )

    if scheduled_count is None:
        scheduled_count = len(
            scheduled_blocks
        )

    unscheduled_count = (
        latest_metrics.get(
            "unscheduled",
            0,
        )
    )

    shadow_count = (
        latest_metrics.get(
            "shadow_blocks_scheduled"
        )
    )

    if shadow_count is None:
        shadow_ids = {
            block.get(
                "shadow_block_id"
            )
            for block
            in scheduled_blocks
            if block.get(
                "shadow_block_id"
            )
        }

        shadow_count = len(
            shadow_ids
        )

    metrics = {
        "live_train_movements":
            len(trains),

        "total_jobs":
            len(maintenance),

        "critical_jobs":
            sum(
                job.get(
                    "priority"
                )
                == "CRITICAL"
                for job
                in maintenance
            ),

        "high_risk_assets":
            len(
                high_risk_jobs
            ),

        "scheduled":
            scheduled_count,

        "unscheduled":
            unscheduled_count,

        "shadow_blocks":
            shadow_count,

        "conflicts":
            latest_metrics.get(
                "conflicts",
                latest_metrics.get(
                    "recovery_conflicts",
                    0,
                ),
            ),
    }

    return {
        "metrics":
            metrics,

        "recentScheduledBlocks":
            scheduled_blocks[:3],

        "highRiskJobs":
            high_risk_jobs[:5],
    }