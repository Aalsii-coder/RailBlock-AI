"""
RailBlock AI — PostgreSQL Database Models

These tables store the main inputs and outputs used by the prototype:

    sections
    trains
    maintenance_requests
    asset_features
    planning_runs
    scheduled_blocks

The optimization/AI logic remains separate from the database layer.
"""

import uuid

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    func,
)

from sqlalchemy.orm import relationship

from database import Base, engine


# ---------------------------------------------------------------------------
# Section
# ---------------------------------------------------------------------------

class Section(Base):
    __tablename__ = "sections"

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    section_name = Column(
        String(150),
        unique=True,
        nullable=False,
        index=True,
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


# ---------------------------------------------------------------------------
# Train movements
# ---------------------------------------------------------------------------

class Train(Base):
    __tablename__ = "trains"

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    train_id = Column(
        String(50),
        nullable=False,
        index=True,
    )

    train_name = Column(
        String(150),
        nullable=True,
    )

    section = Column(
        String(150),
        nullable=False,
        index=True,
    )

    start_time = Column(
        String(5),
        nullable=False,
    )

    end_time = Column(
        String(5),
        nullable=False,
    )

    priority = Column(
        String(20),
        nullable=True,
    )

    raw_data = Column(
        JSON,
        nullable=True,
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


# ---------------------------------------------------------------------------
# Maintenance requests
# ---------------------------------------------------------------------------

class MaintenanceRequest(Base):
    __tablename__ = "maintenance_requests"

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    maintenance_id = Column(
        String(50),
        unique=True,
        nullable=False,
        index=True,
    )

    department = Column(
        String(100),
        nullable=False,
    )

    section = Column(
        String(150),
        nullable=False,
        index=True,
    )

    duration_minutes = Column(
        Integer,
        nullable=False,
    )

    priority = Column(
        String(20),
        nullable=False,
    )

    deadline = Column(
        String(5),
        nullable=False,
    )

    shadow_compatible = Column(
        Boolean,
        default=False,
        nullable=False,
    )

    status = Column(
        String(30),
        default="PENDING",
        nullable=False,
    )

    raw_data = Column(
        JSON,
        nullable=True,
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


# ---------------------------------------------------------------------------
# AI asset features
# ---------------------------------------------------------------------------

class AssetFeature(Base):
    __tablename__ = "asset_features"

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    maintenance_id = Column(
        String(50),
        nullable=False,
        index=True,
    )

    asset_age_years = Column(
        Float,
        nullable=False,
    )

    days_since_last_maintenance = Column(
        Float,
        nullable=False,
    )

    recent_defect_count = Column(
        Integer,
        nullable=False,
    )

    failures_last_12_months = Column(
        Integer,
        nullable=False,
    )

    traffic_load_percent = Column(
        Float,
        nullable=False,
    )

    predicted_risk_score = Column(
        Float,
        nullable=True,
    )

    predicted_risk_level = Column(
        String(20),
        nullable=True,
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


# ---------------------------------------------------------------------------
# Planning runs
# ---------------------------------------------------------------------------

class PlanningRun(Base):
    __tablename__ = "planning_runs"

    run_id = Column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    planning_type = Column(
        String(30),
        default="NORMAL",
        nullable=False,
    )

    planning_time = Column(
        String(5),
        nullable=True,
    )

    status = Column(
        String(30),
        default="COMPLETED",
        nullable=False,
    )

    input_snapshot = Column(
        JSON,
        nullable=True,
    )

    metrics = Column(
        JSON,
        nullable=True,
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    scheduled_blocks = relationship(
        "ScheduledBlock",
        back_populates="planning_run",
        cascade="all, delete-orphan",
    )


# ---------------------------------------------------------------------------
# Scheduled maintenance blocks
# ---------------------------------------------------------------------------

class ScheduledBlock(Base):
    __tablename__ = "scheduled_blocks"

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    planning_run_id = Column(
        String(36),
        ForeignKey(
            "planning_runs.run_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    maintenance_id = Column(
        String(50),
        nullable=False,
        index=True,
    )

    shadow_block_id = Column(
        String(50),
        nullable=True,
        index=True,
    )

    section = Column(
        String(150),
        nullable=False,
        index=True,
    )

    department = Column(
        String(100),
        nullable=False,
    )

    priority = Column(
        String(20),
        nullable=False,
    )

    start_time = Column(
        String(5),
        nullable=False,
    )

    end_time = Column(
        String(5),
        nullable=False,
    )

    start_minutes = Column(
        Integer,
        nullable=False,
    )

    end_minutes = Column(
        Integer,
        nullable=False,
    )

    predicted_risk_score = Column(
        Float,
        nullable=True,
    )

    urgency_score = Column(
        Integer,
        nullable=True,
    )

    scheduled = Column(
        Boolean,
        default=True,
        nullable=False,
    )

    raw_data = Column(
        JSON,
        nullable=True,
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    planning_run = relationship(
        "PlanningRun",
        back_populates="scheduled_blocks",
    )


# ---------------------------------------------------------------------------
# Database initialization
# ---------------------------------------------------------------------------

def create_tables():
    """
    Create all RailBlock AI tables if they do not already exist.

    SQLAlchemy's create_all() is safe to call multiple times.
    Existing tables will not be deleted.
    """

    Base.metadata.create_all(
        bind=engine
    )