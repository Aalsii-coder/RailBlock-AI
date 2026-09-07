"""Quick example run — prints AI risk, urgency scores, schedule, metrics."""
from synthetic_history import generate_synthetic_history
from ai_predictor import train_risk_model
from pipeline import run_ai_planning_pipeline

history = generate_synthetic_history(500, seed=42)
bundle = train_risk_model(history)

data = {
    "sections": ["Delhi-Ghaziabad", "Mumbai-Pune", "Kolkata-Howrah", "Chennai-Arakkonam"],
    "trains": [
        {"train_id": "T001", "train_name": "Rajdhani Express",  "section": "Delhi-Ghaziabad",    "start": "08:00", "end": "08:25", "priority": "HIGH"},
        {"train_id": "T002", "train_name": "Shatabdi Express",  "section": "Delhi-Ghaziabad",    "start": "10:00", "end": "10:30", "priority": "HIGH"},
        {"train_id": "T003", "train_name": "Deccan Queen",      "section": "Mumbai-Pune",         "start": "07:15", "end": "07:50", "priority": "HIGH"},
        {"train_id": "T004", "train_name": "Howrah Mail",       "section": "Kolkata-Howrah",      "start": "06:00", "end": "06:30", "priority": "MEDIUM"},
        {"train_id": "T005", "train_name": "Chennai Express",   "section": "Chennai-Arakkonam",   "start": "09:00", "end": "09:45", "priority": "HIGH"},
    ],
    "maintenance_jobs": [
        {"maintenance_id": "M001", "department": "Engineering",           "section": "Delhi-Ghaziabad",  "duration_minutes": 60,  "priority": "CRITICAL", "deadline": "12:00"},
        {"maintenance_id": "M002", "department": "Signal & Telecom",      "section": "Mumbai-Pune",      "duration_minutes": 90,  "priority": "HIGH",     "deadline": "15:00"},
        {"maintenance_id": "M003", "department": "Traction Distribution", "section": "Kolkata-Howrah",   "duration_minutes": 45,  "priority": "MEDIUM",   "deadline": "20:00"},
        {"maintenance_id": "M004", "department": "Engineering",           "section": "Chennai-Arakkonam","duration_minutes": 120, "priority": "HIGH",     "deadline": "22:00"},
    ],
}
asset_features = [
    {"maintenance_id": "M001", "asset_age_years": 12, "days_since_last_maintenance": 180, "recent_defect_count": 3, "failures_last_12_months": 2, "traffic_load_percent": 85},
    {"maintenance_id": "M002", "asset_age_years": 5,  "days_since_last_maintenance": 30,  "recent_defect_count": 1, "failures_last_12_months": 0, "traffic_load_percent": 60},
    {"maintenance_id": "M003", "asset_age_years": 20, "days_since_last_maintenance": 300, "recent_defect_count": 5, "failures_last_12_months": 3, "traffic_load_percent": 70},
    {"maintenance_id": "M004", "asset_age_years": 2,  "days_since_last_maintenance": 15,  "recent_defect_count": 0, "failures_last_12_months": 0, "traffic_load_percent": 40},
]

result = run_ai_planning_pipeline(data, asset_features, bundle, planning_time="06:00")

print("=== AI RISK RANKING ===")
for r in result["ai_risk_ranking"]:
    print(f"  {r['maintenance_id']}: score={r['predicted_risk_score']}, level={r['predicted_risk_level']}")

print("\n=== PRIORITY RANKING ===")
for r in result["priority_ranking"]:
    print(f"  Rank {r['rank']}: {r['maintenance_id']} | {r['score_explanation']} | deadline={r['deadline']}")

print("\n=== SCHEDULED BLOCKS ===")
for b in result["scheduled_blocks"]:
    print(f"  {b['maintenance_id']} | {b['section']} | {b['start']}-{b['end']} | duration={b['duration_minutes']}m | priority={b['priority']}")

print("\n=== UNSCHEDULED JOBS ===")
if result["unscheduled_jobs"]:
    for j in result["unscheduled_jobs"]:
        print(f"  {j['maintenance_id']} - unscheduled")
else:
    print("  (none — all jobs scheduled)")

print("\n=== METRICS ===")
for k, v in result["metrics"].items():
    print(f"  {k}: {v}")

print("\n=== AI MODEL INFO ===")
for k, v in result["ai_model_info"].items():
    if k != "features_used":
        print(f"  {k}: {v}")
