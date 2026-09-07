"""
pipeline.py
===========
RailBlock AI — End-to-End AI Planning Pipeline

This module wires together the four core steps of RailBlock AI into a
single function call:

    1. validate_and_standardize_data()   — input validation
    2. predict_maintenance_risk()        — Random Forest AI risk scoring
    3. score_maintenance_jobs()          — urgency scoring with AI bonus
    4. optimize_schedule()               — CP-SAT conflict-free scheduling

The returned dict is JSON-safe — no sklearn model objects are included.
"""

import copy

from validator import validate_and_standardize_data
from ai_predictor import predict_maintenance_risk
from priority_scoring import score_maintenance_jobs
from optimizer import optimize_schedule


def run_ai_planning_pipeline(
    data: dict,
    asset_features: list,
    model_bundle: dict,
    planning_time: str = "00:00",
) -> dict:
    """
    Run the full RailBlock AI planning pipeline.

    Data flow
    ─────────
    Raw data
        → validate_and_standardize_data()
        → predict_maintenance_risk()
        → score_maintenance_jobs()
        → optimize_schedule()
        → flat JSON-safe response

    Parameters
    ----------
    data : dict
        Raw railway planning data with 'trains', 'maintenance_jobs', 'sections'.
    asset_features : list[dict]
        Asset health features for AI risk prediction, keyed by maintenance_id.
    model_bundle : dict
        Trained model bundle from train_risk_model().  The sklearn model
        object inside is used for prediction but is NOT included in the
        returned response.
    planning_time : str
        Current planning time as 'HH:MM'.  Affects deadline urgency scoring.

    Returns
    -------
    dict
        {
            "scheduled_blocks"  : [...],   # conflict-free maintenance blocks
            "unscheduled_jobs"  : [...],   # jobs that could not be scheduled
            "metrics"           : {...},   # summary statistics
            "priority_ranking"  : [...],   # jobs ranked by urgency score
            "ai_risk_ranking"   : [...],   # jobs ranked by AI risk
            "ai_model_info"     : {...},   # model metadata (no sklearn object)
        }
    """
    # Step 1 — Validate and standardise input.
    clean_data = validate_and_standardize_data(data)

    # Step 2 — Predict maintenance risk using the trained Random Forest.
    risk_data = predict_maintenance_risk(clean_data, asset_features, model_bundle)

    # Step 3 — Compute urgency scores (base priority + deadline + AI bonus).
    scored_data = score_maintenance_jobs(risk_data, planning_time)

    # Step 4 — CP-SAT optimisation: find conflict-free schedule.
    schedule = optimize_schedule(scored_data)

    # ── Assemble flat, JSON-safe response ──────────────────────────────────
    # Pull ai_risk_ranking and ai_model_info from scored_data (they were
    # added by predict_maintenance_risk and carried through by score_maintenance_jobs).
    # The sklearn model object inside model_bundle is deliberately NOT
    # included in the response — it is not JSON-serialisable and the caller
    # already holds the model_bundle if they need it.

    response = {
        "scheduled_blocks": schedule["scheduled_blocks"],
        "unscheduled_jobs": schedule["unscheduled_jobs"],
        "metrics": schedule["metrics"],
        "priority_ranking": scored_data.get("priority_ranking", []),
        "ai_risk_ranking": scored_data.get("ai_risk_ranking", []),
        "ai_model_info": scored_data.get("ai_model_info", {}),
    }

    return response
