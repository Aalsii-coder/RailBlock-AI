"""
RailBlock AI — Shadow Block Candidate Generator

This module finds maintenance jobs that may be performed simultaneously
inside one shared railway maintenance possession window.

IMPORTANT:
This module does NOT schedule the jobs.
It only identifies compatible candidate groups.

The CP-SAT optimizer will later decide when these shadow blocks can actually
be placed safely around train movements.
"""

from collections import defaultdict
from copy import deepcopy


ALLOWED_DEPARTMENTS = {
    "Engineering",
    "Signal & Telecom",
    "Traction Distribution",
}


PRIORITY_RANK = {
    "CRITICAL": 4,
    "HIGH": 3,
    "MEDIUM": 2,
    "LOW": 1,
}


def _job_sort_key(job: dict):
    """
    Sort important jobs first when forming shadow block groups.
    """

    priority = PRIORITY_RANK.get(job.get("priority", "LOW"), 1)
    urgency = int(job.get("urgency_score", 0))
    deadline = job.get("deadline", "24:00")
    maintenance_id = job.get("maintenance_id", "")

    return (
        -priority,
        -urgency,
        deadline,
        maintenance_id,
    )


def find_shadow_block_candidates(data: dict) -> dict:
    """
    Find maintenance jobs that may potentially share one maintenance window.

    Rules for this prototype:

    1. Jobs must be on the SAME railway section.
    2. Jobs must belong to DIFFERENT departments.
    3. Every job must explicitly contain:

           "shadow_compatible": True

    4. Only these departments are currently supported:
       - Engineering
       - Signal & Telecom
       - Traction Distribution

    5. Maximum one job from each department can appear inside a candidate.

    IMPORTANT:
    These are candidate groups only.
    The optimizer still decides whether and when they can be scheduled.

    Returns
    -------
    {
        "shadow_block_candidates": [...],
        "candidate_count": int,
        "potential_time_saved_minutes": int
    }
    """

    copied_data = deepcopy(data)

    jobs = copied_data.get("maintenance_jobs", [])

    # Store eligible jobs by railway section.
    jobs_by_section = defaultdict(list)

    for job in jobs:

        if job.get("shadow_compatible") is not True:
            continue

        department = job.get("department")

        if department not in ALLOWED_DEPARTMENTS:
            continue

        section = job.get("section")

        if not section:
            continue

        jobs_by_section[section].append(job)

    shadow_candidates = []

    shadow_counter = 1

    # Process sections in deterministic order.
    for section in sorted(jobs_by_section.keys()):

        remaining_jobs = sorted(
            jobs_by_section[section],
            key=_job_sort_key,
        )

        while remaining_jobs:

            used_departments = set()
            members = []
            leftover_jobs = []

            for job in remaining_jobs:

                department = job["department"]

                # Only one job from each department
                # can be placed in the same candidate.
                if (
                    department not in used_departments
                    and len(members) < 3
                ):
                    members.append(job)
                    used_departments.add(department)

                else:
                    leftover_jobs.append(job)

            # A shadow block requires jobs from at least
            # two different departments.
            if len(members) < 2:
                break

            durations = [
                int(job["duration_minutes"])
                for job in members
            ]

            # If jobs run simultaneously, the shared possession
            # lasts as long as the longest member job.
            shadow_duration = max(durations)

            separate_duration = sum(durations)

            time_saved = separate_duration - shadow_duration

            # Highest priority among member jobs.
            highest_priority = max(
                members,
                key=lambda job: PRIORITY_RANK.get(
                    job.get("priority", "LOW"),
                    1,
                ),
            ).get("priority", "LOW")

            # Earliest deadline protects the most urgent member.
            earliest_deadline = min(
                job.get("deadline", "24:00")
                for job in members
            )

            # Highest urgency among member jobs.
            max_urgency = max(
                int(job.get("urgency_score", 0))
                for job in members
            )

            candidate = {
                "shadow_block_id": f"SB{shadow_counter:03d}",
                "section": section,

                "maintenance_ids": [
                    job["maintenance_id"]
                    for job in members
                ],

                "departments": [
                    job["department"]
                    for job in members
                ],

                "member_jobs": deepcopy(members),

                "shadow_duration_minutes": shadow_duration,
                "separate_duration_minutes": separate_duration,
                "potential_time_saved_minutes": time_saved,

                "highest_priority": highest_priority,
                "earliest_deadline": earliest_deadline,
                "max_urgency_score": max_urgency,
            }

            shadow_candidates.append(candidate)

            shadow_counter += 1

            remaining_jobs = leftover_jobs

    total_time_saved = sum(
        block["potential_time_saved_minutes"]
        for block in shadow_candidates
    )

    return {
        "shadow_block_candidates": shadow_candidates,
        "candidate_count": len(shadow_candidates),
        "potential_time_saved_minutes": total_time_saved,
    }