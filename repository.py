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