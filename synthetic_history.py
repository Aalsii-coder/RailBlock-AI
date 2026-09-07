"""
synthetic_history.py
====================
RailBlock AI — Synthetic Training Data Generator

PURPOSE
-------
Generate realistic-looking maintenance history records so the
Random Forest model can be trained and tested without needing
access to real Indian Railways operational databases.

⚠️  IMPORTANT DISCLAIMER ⚠️
─────────────────────────────────────────────────────────────────
This data is ENTIRELY SYNTHETIC.

* It is NOT real Indian Railways data.
* It was NOT sourced from TMS, SMMS, TDMS, or any official system.
* Accuracy figures from a model trained here do NOT represent
  real-world Indian Railways failure prediction accuracy.
* The same model pipeline is designed to be retrained using
  authorised historical data from TMS / SMMS / TDMS before
  any production deployment.
─────────────────────────────────────────────────────────────────

The synthetic records intentionally cover a wide range so the
RandomForestClassifier sees both class 0 (no failure) and
class 1 (failure within 30 days) in training.
"""

import random


def generate_synthetic_history(n_records: int = 500, seed: int = 42) -> list:
    """
    Generate n_records synthetic maintenance history dictionaries.

    Each record contains only the five ML features plus the target label:

        asset_age_years           – how old the asset is (years)
        days_since_last_maintenance – days since the last service
        recent_defect_count       – defects found in the last inspection
        failures_last_12_months   – how many failures in the past year
        traffic_load_percent      – how busy this track section is (0-100)

        failure_within_30_days    – 1 if the asset failed within 30 days,
                                    0 otherwise  (the ML target label)

    Why only these five features?
    ─────────────────────────────
    The AI must predict physical asset failure risk.
    Fields like priority, deadline, and department describe
    *planning decisions*, not asset health.  Mixing them into
    the ML model would leak planning logic into the predictor.

    Parameters
    ----------
    n_records : int
        Number of records to generate (default 500).
    seed : int
        Random seed for reproducibility (default 42).

    Returns
    -------
    list[dict]
        A list of synthetic history records ready for train_risk_model().
    """
    rng = random.Random(seed)
    records = []

    for _ in range(n_records):
        # Asset age: newer assets (0-5 years) are less likely to fail;
        # older ones (15-30 years) are more likely.
        asset_age = round(rng.uniform(0, 30), 1)

        # Days since last maintenance: recently serviced assets are safer.
        days_since = rng.randint(0, 365)

        # Recent defect count: more defects → higher failure risk.
        defect_count = rng.randint(0, 10)

        # Failures in the last 12 months.
        failures_12m = rng.randint(0, 5)

        # Traffic load: heavier traffic wears assets faster.
        traffic_load = round(rng.uniform(10, 100), 1)

        # Compute a rough failure probability to make the labels realistic
        # (not just random noise — we want a learnable pattern).
        failure_prob = (
            0.02 * asset_age
            + 0.001 * days_since
            + 0.06 * defect_count
            + 0.10 * failures_12m
            + 0.003 * traffic_load
        )
        # Clip to [0, 1] and draw the binary label.
        failure_prob = min(failure_prob, 1.0)
        label = 1 if rng.random() < failure_prob else 0

        records.append(
            {
                "asset_age_years": asset_age,
                "days_since_last_maintenance": days_since,
                "recent_defect_count": defect_count,
                "failures_last_12_months": failures_12m,
                "traffic_load_percent": traffic_load,
                "failure_within_30_days": label,
            }
        )

    return records
