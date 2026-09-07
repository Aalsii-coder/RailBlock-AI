"""
validator.py
============
RailBlock AI — Input Validation & Standardisation

This module checks that railway planning data coming into the system
is well-formed before any AI or optimisation step touches it.

What it validates
-----------------
* trains        — each train has an id, name, section, start/end times,
                  and a priority from the allowed set.
* maintenance_jobs — each job has an id, department, section, duration,
                     priority, and deadline.
* sections      — the list of track sections.

Departments allowed : Engineering, Signal & Telecom, Traction Distribution
Priorities allowed  : CRITICAL, HIGH, MEDIUM, LOW

Important
---------
The original input dict is NEVER modified.
The function returns a deep-copied, validated, and normalised version.
"""

import copy
import re

# ── Allowed values ────────────────────────────────────────────────────────────

VALID_DEPARTMENTS = {"Engineering", "Signal & Telecom", "Traction Distribution"}
VALID_PRIORITIES = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_time(time_str: str, field_name: str) -> str:
    """
    Accept 'HH:MM' (or 'H:MM') and return it normalised to 'HH:MM'.
    Raises ValueError if the format is wrong.
    """
    if not isinstance(time_str, str):
        raise ValueError(f"'{field_name}' must be a string, got {type(time_str).__name__}")

    # Allow single-digit hour like "8:00"
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", time_str.strip())
    if not match:
        raise ValueError(
            f"'{field_name}' must be in HH:MM format, got '{time_str}'"
        )

    hour = int(match.group(1))
    minute = int(match.group(2))

    if not (0 <= hour <= 23):
        raise ValueError(f"'{field_name}' hour must be 0-23, got {hour}")
    if not (0 <= minute <= 59):
        raise ValueError(f"'{field_name}' minute must be 0-59, got {minute}")

    return f"{hour:02d}:{minute:02d}"


def _require_fields(obj: dict, fields: list, context: str):
    """Raise KeyError if any required field is missing from a dict."""
    for f in fields:
        if f not in obj:
            raise KeyError(f"Missing required field '{f}' in {context}")


def _require_string(value, field: str, context: str):
    """Raise TypeError if value is not a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"'{field}' in {context} must be a non-empty string")


# ── Train validation ──────────────────────────────────────────────────────────

def _validate_train(train: dict, index: int) -> dict:
    """Validate a single train record and return a clean copy."""
    context = f"train[{index}] (id={train.get('train_id', '?')})"

    _require_fields(
        train,
        ["train_id", "train_name", "section", "start", "end", "priority"],
        context,
    )

    _require_string(train["train_id"], "train_id", context)
    _require_string(train["train_name"], "train_name", context)
    _require_string(train["section"], "section", context)

    priority = train["priority"]
    if priority not in VALID_PRIORITIES:
        raise ValueError(
            f"Train priority '{priority}' in {context} is not one of {VALID_PRIORITIES}"
        )

    clean = dict(train)  # shallow copy — all values are scalars
    clean["start"] = _parse_time(train["start"], "start")
    clean["end"] = _parse_time(train["end"], "end")
    clean["priority"] = priority

    return clean


# ── Maintenance job validation ────────────────────────────────────────────────

def _validate_maintenance_job(job: dict, index: int) -> dict:
    """Validate a single maintenance job record and return a clean copy."""
    context = f"maintenance_jobs[{index}] (id={job.get('maintenance_id', '?')})"

    _require_fields(
        job,
        ["maintenance_id", "department", "section", "duration_minutes", "priority", "deadline"],
        context,
    )

    _require_string(job["maintenance_id"], "maintenance_id", context)
    _require_string(job["section"], "section", context)

    department = job["department"]
    if department not in VALID_DEPARTMENTS:
        raise ValueError(
            f"Department '{department}' in {context} is not one of {VALID_DEPARTMENTS}"
        )

    priority = job["priority"]
    if priority not in VALID_PRIORITIES:
        raise ValueError(
            f"Priority '{priority}' in {context} is not one of {VALID_PRIORITIES}"
        )

    duration = job["duration_minutes"]
    if not isinstance(duration, (int, float)) or duration <= 0:
        raise ValueError(
            f"'duration_minutes' in {context} must be a positive number, got {duration}"
        )

    clean = dict(job)
    clean["deadline"] = _parse_time(job["deadline"], "deadline")
    clean["duration_minutes"] = int(duration)
    clean["priority"] = priority
    clean["department"] = department

    return clean


# ── Public API ────────────────────────────────────────────────────────────────

def validate_and_standardize_data(data: dict) -> dict:
    """
    Validate and standardise raw railway planning data.

    Parameters
    ----------
    data : dict
        Raw input with keys 'trains', 'maintenance_jobs', 'sections'.

    Returns
    -------
    dict
        A deep-copied, validated, normalised version of the input.
        The original *data* dict is never modified.

    Raises
    ------
    KeyError   — a required field is missing.
    TypeError  — a field has the wrong type.
    ValueError — a field has an invalid value.
    """
    # Deep copy so we never mutate the caller's data.
    data = copy.deepcopy(data)

    # Top-level structure
    for key in ("trains", "maintenance_jobs", "sections"):
        if key not in data:
            raise KeyError(f"Missing top-level key '{key}' in data")
        if not isinstance(data[key], list):
            raise TypeError(f"'{key}' must be a list")

    # Validate each train
    validated_trains = [
        _validate_train(train, i) for i, train in enumerate(data["trains"])
    ]

    # Validate each maintenance job
    validated_jobs = [
        _validate_maintenance_job(job, i)
        for i, job in enumerate(data["maintenance_jobs"])
    ]

    # Sections — just ensure each is a non-empty string
    validated_sections = []
    for i, section in enumerate(data["sections"]):
        if not isinstance(section, str) or not section.strip():
            raise TypeError(f"sections[{i}] must be a non-empty string")
        validated_sections.append(section.strip())

    return {
        "trains": validated_trains,
        "maintenance_jobs": validated_jobs,
        "sections": validated_sections,
    }
