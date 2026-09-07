"""
RailBlock AI — FastAPI Backend

This module exposes the tested RailBlock AI planning logic through HTTP APIs.

Prototype endpoints
-------------------
GET  /health
POST /planning/run
POST /planning/shadow
POST /planning/recover

The API does NOT control railway infrastructure.
It only returns planning/scheduling recommendations.
"""

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from synthetic_history import generate_synthetic_history
from ai_predictor import train_risk_model
from pipeline import run_ai_planning_pipeline
from shadow_scheduler import schedule_with_shadow_blocks
from rolling_horizon import recover_schedule


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RailBlock AI API",
    description=(
        "AI-powered railway maintenance block planning prototype "
        "for SIH26027."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# CORS
#
# Allows the React frontend running locally to call this backend.
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite
        "http://127.0.0.1:5173",
        "http://localhost:3000",   # common React dev port
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Prototype AI model
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_model_bundle():
    """
    Train and cache the prototype Random Forest model.

    IMPORTANT:
    The current prototype uses SYNTHETIC historical data because we do not
    have authorized row-level Indian Railways maintenance/failure records.

    In production this function would load a model trained on authorized
    historical railway asset data.
    """

    history = generate_synthetic_history(
        n_records=500,
        seed=42,
    )

    return train_risk_model(history)


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """
    Simple check used by the frontend to confirm that the API is running.
    """

    return {
        "status": "ok",
        "service": "RailBlock AI",
        "version": "1.0.0",
        "ai_model": "RandomForestClassifier",
        "ai_training_data": "synthetic prototype data",
    }


# ---------------------------------------------------------------------------
# Normal AI + CP-SAT planning
# ---------------------------------------------------------------------------

@app.post("/planning/run")
def planning_run(payload: dict):
    """
    Run the standard RailBlock AI planning pipeline.

    Expected JSON:

    {
        "data": {
            "trains": [...],
            "maintenance_jobs": [...],
            "sections": [...]
        },

        "asset_features": [
            {
                "maintenance_id": "M001",
                "asset_age_years": 12,
                "days_since_last_maintenance": 180,
                "recent_defect_count": 3,
                "failures_last_12_months": 1,
                "traffic_load_percent": 85
            }
        ],

        "planning_time": "08:00"
    }
    """

    try:

        data = payload.get("data")

        asset_features = payload.get(
            "asset_features",
            [],
        )

        planning_time = payload.get(
            "planning_time",
            "00:00",
        )

        if data is None:
            raise ValueError(
                "payload must contain 'data'"
            )

        model_bundle = get_model_bundle()

        result = run_ai_planning_pipeline(
            data=data,
            asset_features=asset_features,
            model_bundle=model_bundle,
            planning_time=planning_time,
        )

        return result

    except (ValueError, TypeError) as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error


# ---------------------------------------------------------------------------
# Shadow Block scheduling
# ---------------------------------------------------------------------------

@app.post("/planning/shadow")
def planning_shadow(payload: dict):
    """
    Run Shadow Block Stacking.

    The input should already contain any AI/urgency fields required by
    scheduling.

    Expected JSON:

    {
        "data": {
            "trains": [...],
            "maintenance_jobs": [...],
            "sections": [...]
        }
    }

    Compatible jobs may contain:

        "shadow_compatible": true
    """

    try:

        data = payload.get("data")

        if data is None:
            raise ValueError(
                "payload must contain 'data'"
            )

        result = schedule_with_shadow_blocks(
            data
        )

        return result

    except (ValueError, TypeError, KeyError) as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error


# ---------------------------------------------------------------------------
# Rolling Horizon Recovery
# ---------------------------------------------------------------------------

@app.post("/planning/recover")
def planning_recover(payload: dict):
    """
    Recover an existing plan after a disruption.

    Expected JSON:

    {
        "previous_result": {
            "scheduled_blocks": [...],
            "shadow_blocks": [...]
        },

        "updated_data": {
            "trains": [...],
            "maintenance_jobs": [...],
            "sections": [...]
        },

        "disruption": {
            "type": "TRAIN_DELAY",
            "section": "Delhi-Ghaziabad",
            "start": "11:00",
            "end": "13:00",
            "description": "Train delay entered maintenance window"
        }
    }
    """

    try:

        previous_result = payload.get(
            "previous_result"
        )

        updated_data = payload.get(
            "updated_data"
        )

        disruption = payload.get(
            "disruption"
        )

        if previous_result is None:
            raise ValueError(
                "payload must contain 'previous_result'"
            )

        if updated_data is None:
            raise ValueError(
                "payload must contain 'updated_data'"
            )

        if disruption is None:
            raise ValueError(
                "payload must contain 'disruption'"
            )

        result = recover_schedule(
            previous_result=previous_result,
            updated_data=updated_data,
            disruption=disruption,
        )

        return result

    except (ValueError, TypeError, KeyError) as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error