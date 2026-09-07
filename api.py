"""
RailBlock AI — FastAPI Backend

This module exposes the tested RailBlock AI planning logic through HTTP APIs
and stores planning results in PostgreSQL.

Prototype endpoints
-------------------
GET  /health

POST /planning/run
POST /planning/shadow
POST /planning/recover

GET  /planning/latest
GET  /planning/runs/{run_id}

IMPORTANT:
This prototype provides planning recommendations only.
It does not directly control railway infrastructure, signalling,
KAVACH, or train movements.
"""

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from synthetic_history import generate_synthetic_history
from ai_predictor import train_risk_model
from pipeline import run_ai_planning_pipeline
from shadow_scheduler import schedule_with_shadow_blocks
from rolling_horizon import recover_schedule

from repository import (
    get_latest_planning_run,
    get_planning_run,
    save_planning_run,
)


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="RailBlock AI API",
    description=(
        "AI-powered railway maintenance block planning prototype "
        "for SIH26027."
    ),
    version="1.0.0",
)


# =============================================================================
# CORS
# =============================================================================
#
# This allows the React frontend running locally to communicate with FastAPI.
#

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# PROTOTYPE AI MODEL
# =============================================================================

@lru_cache(maxsize=1)
def get_model_bundle():
    """
    Train the prototype Random Forest model once and cache it.

    The prototype currently uses synthetic historical records.

    In a production railway deployment, this would be replaced by
    a model trained on authorized real historical asset-maintenance data.
    """

    history = generate_synthetic_history(
        n_records=500,
        seed=42,
    )

    model_bundle = train_risk_model(
        history
    )

    return model_bundle


# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.get("/health")
def health():
    """
    Check whether the RailBlock AI backend is running.
    """

    return {
        "status": "ok",
        "service": "RailBlock AI",
        "version": "1.0.0",
        "ai_model": "RandomForestClassifier",
        "ai_training_data": "synthetic prototype data",
    }


# =============================================================================
# NORMAL AI + CP-SAT PLANNING
# =============================================================================

@app.post("/planning/run")
def planning_run(payload: dict):
    """
    Run the standard RailBlock AI planning pipeline.

    Flow:

        input
          ↓
        validation
          ↓
        AI risk prediction
          ↓
        priority / urgency scoring
          ↓
        CP-SAT optimization
          ↓
        PostgreSQL persistence
          ↓
        JSON response

    Expected payload:

    {
        "data": {
            "trains": [...],
            "maintenance_jobs": [...],
            "sections": [...]
        },

        "asset_features": [...],

        "planning_time": "08:00"
    }
    """

    try:

        data = payload.get(
            "data"
        )

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

        # -------------------------------------------------------------
        # Get cached Random Forest model
        # -------------------------------------------------------------

        model_bundle = get_model_bundle()

        # -------------------------------------------------------------
        # Run AI + scoring + CP-SAT
        # -------------------------------------------------------------

        result = run_ai_planning_pipeline(
            data=data,
            asset_features=asset_features,
            model_bundle=model_bundle,
            planning_time=planning_time,
        )

        # -------------------------------------------------------------
        # Save result to PostgreSQL
        # -------------------------------------------------------------

        run_id = save_planning_run(
            result=result,
            planning_type="NORMAL",
            planning_time=planning_time,
            input_snapshot={
                "data": data,
                "asset_features": asset_features,
                "planning_time": planning_time,
            },
        )

        # -------------------------------------------------------------
        # Return original result + DB run ID
        # -------------------------------------------------------------

        response = dict(
            result
        )

        response[
            "planning_run_id"
        ] = run_id

        return response

    except (
        ValueError,
        TypeError,
        KeyError,
    ) as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error


# =============================================================================
# SHADOW BLOCK PLANNING
# =============================================================================

@app.post("/planning/shadow")
def planning_shadow(payload: dict):
    """
    Run Shadow Block Stacking.

    Compatible maintenance jobs may execute simultaneously inside
    one shared possession window.

    Expected payload:

    {
        "data": {
            "trains": [...],
            "maintenance_jobs": [...],
            "sections": [...]
        }
    }

    Compatible maintenance jobs contain:

        "shadow_compatible": true
    """

    try:

        data = payload.get(
            "data"
        )

        if data is None:
            raise ValueError(
                "payload must contain 'data'"
            )

        # -------------------------------------------------------------
        # Run Shadow Block scheduler
        # -------------------------------------------------------------

        result = (
            schedule_with_shadow_blocks(
                data
            )
        )

        # -------------------------------------------------------------
        # Save result to PostgreSQL
        # -------------------------------------------------------------

        run_id = save_planning_run(
            result=result,
            planning_type="SHADOW",
            input_snapshot={
                "data": data
            },
        )

        # -------------------------------------------------------------
        # Return response
        # -------------------------------------------------------------

        response = dict(
            result
        )

        response[
            "planning_run_id"
        ] = run_id

        return response

    except (
        ValueError,
        TypeError,
        KeyError,
    ) as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error


# =============================================================================
# ROLLING HORIZON RECOVERY
# =============================================================================

@app.post("/planning/recover")
def planning_recover(payload: dict):
    """
    Recover an existing maintenance plan after a disruption.

    Example disruption:

    {
        "type": "TRAIN_DELAY",
        "section": "Delhi-Ghaziabad",
        "start": "11:00",
        "end": "13:00",
        "description": "Delayed train entered maintenance window"
    }

    Expected payload:

    {
        "previous_result": {...},

        "updated_data": {
            "trains": [...],
            "maintenance_jobs": [...],
            "sections": [...]
        },

        "disruption": {...}
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

        # -------------------------------------------------------------
        # Validate payload
        # -------------------------------------------------------------

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

        # -------------------------------------------------------------
        # Run Rolling Horizon Recovery
        # -------------------------------------------------------------

        result = recover_schedule(
            previous_result=previous_result,
            updated_data=updated_data,
            disruption=disruption,
        )

        # -------------------------------------------------------------
        # Save recovered plan to PostgreSQL
        # -------------------------------------------------------------

        run_id = save_planning_run(
            result=result,
            planning_type="RECOVERY",
            planning_time=disruption.get(
                "start"
            ),
            input_snapshot={
                "updated_data": updated_data,
                "disruption": disruption,
            },
        )

        # -------------------------------------------------------------
        # Return response
        # -------------------------------------------------------------

        response = dict(
            result
        )

        response[
            "planning_run_id"
        ] = run_id

        return response

    except (
        ValueError,
        TypeError,
        KeyError,
    ) as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error


# =============================================================================
# LATEST STORED PLANNING RUN
# =============================================================================

@app.get("/planning/latest")
def planning_latest(
    planning_type: str | None = None,
):
    """
    Return the most recently saved planning run from PostgreSQL.

    Examples:

        GET /planning/latest

        GET /planning/latest?planning_type=NORMAL

        GET /planning/latest?planning_type=SHADOW

        GET /planning/latest?planning_type=RECOVERY
    """

    result = get_latest_planning_run(
        planning_type=planning_type
    )

    if result is None:

        raise HTTPException(
            status_code=404,
            detail="No planning run found",
        )

    return result


# =============================================================================
# SPECIFIC STORED PLANNING RUN
# =============================================================================

@app.get("/planning/runs/{run_id}")
def planning_run_by_id(
    run_id: str,
):
    """
    Retrieve one saved planning run from PostgreSQL using its UUID.
    """

    result = get_planning_run(
        run_id
    )

    if result is None:

        raise HTTPException(
            status_code=404,
            detail="Planning run not found",
        )

    return result

from repository import delete_planning_run


def test_planning_run_is_saved_and_retrievable():

    payload = {
        "data": {
            "trains": [],
            "maintenance_jobs": [
                {
                    "maintenance_id": "API-M001",
                    "department": "Engineering",
                    "section": "Delhi-Ghaziabad",
                    "duration_minutes": 60,
                    "priority": "CRITICAL",
                    "deadline": "18:00",
                }
            ],
            "sections": [
                "Delhi-Ghaziabad"
            ],
        },

        "asset_features": [
            {
                "maintenance_id": "API-M001",
                "asset_age_years": 20,
                "days_since_last_maintenance": 250,
                "recent_defect_count": 4,
                "failures_last_12_months": 2,
                "traffic_load_percent": 90,
            }
        ],

        "planning_time": "08:00",
    }

    response = client.post(
        "/planning/run",
        json=payload,
    )

    assert response.status_code == 200

    body = response.json()

    assert "planning_run_id" in body

    run_id = body["planning_run_id"]

    try:
        stored_response = client.get(
            f"/planning/runs/{run_id}"
        )

        assert stored_response.status_code == 200

        stored = stored_response.json()

        assert stored["run_id"] == run_id
        assert stored["planning_type"] == "NORMAL"
        assert stored["planning_time"] == "08:00"

        ids = {
            block["maintenance_id"]
            for block in stored["scheduled_blocks"]
        }

        assert "API-M001" in ids

    finally:
        delete_planning_run(run_id)


def test_latest_endpoint_returns_saved_run():

    payload = {
        "data": {
            "trains": [],
            "maintenance_jobs": [
                {
                    "maintenance_id": "API-SHADOW-1",
                    "department": "Engineering",
                    "section": "Delhi-Ghaziabad",
                    "duration_minutes": 60,
                    "priority": "HIGH",
                    "deadline": "18:00",
                    "urgency_score": 120,
                    "shadow_compatible": True,
                },
                {
                    "maintenance_id": "API-SHADOW-2",
                    "department": "Signal & Telecom",
                    "section": "Delhi-Ghaziabad",
                    "duration_minutes": 30,
                    "priority": "MEDIUM",
                    "deadline": "18:00",
                    "urgency_score": 90,
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

    run_id = response.json()[
        "planning_run_id"
    ]

    try:
        latest_response = client.get(
            "/planning/latest",
            params={
                "planning_type": "SHADOW"
            },
        )

        assert latest_response.status_code == 200

        latest = latest_response.json()

        assert latest["run_id"] == run_id
        assert latest["planning_type"] == "SHADOW"

    finally:
        delete_planning_run(run_id)


def test_missing_run_returns_404():

    response = client.get(
        "/planning/runs/00000000-0000-0000-0000-000000000000"
    )

    assert response.status_code == 404


def test_recovery_run_is_saved():

    previous_result = {
        "scheduled_blocks": [
            {
                "maintenance_id": "API-REC-1",
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
                "train_id": "API-T001",
                "section": "Delhi-Ghaziabad",
                "start": "11:00",
                "end": "13:00",
            }
        ],
        "maintenance_jobs": [
            {
                "maintenance_id": "API-REC-1",
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

    run_id = response.json()[
        "planning_run_id"
    ]

    try:
        stored_response = client.get(
            f"/planning/runs/{run_id}"
        )

        assert stored_response.status_code == 200

        stored = stored_response.json()

        assert stored["planning_type"] == "RECOVERY"
        assert stored["metrics"]["recovery_conflicts"] == 0

    finally:
        delete_planning_run(run_id)