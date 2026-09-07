from copy import deepcopy

from shadow_block import find_shadow_block_candidates


def make_job(
    maintenance_id,
    department,
    section="Delhi-Ghaziabad",
    duration=60,
    priority="HIGH",
    deadline="15:00",
    compatible=True,
):
    return {
        "maintenance_id": maintenance_id,
        "department": department,
        "section": section,
        "duration_minutes": duration,
        "priority": priority,
        "deadline": deadline,
        "urgency_score": 100,
        "shadow_compatible": compatible,
    }


def test_two_departments_create_shadow_candidate():

    data = {
        "maintenance_jobs": [
            make_job(
                "M001",
                "Engineering",
                duration=60,
            ),
            make_job(
                "M002",
                "Signal & Telecom",
                duration=30,
            ),
        ]
    }

    result = find_shadow_block_candidates(data)

    assert result["candidate_count"] == 1

    block = result["shadow_block_candidates"][0]

    assert block["shadow_duration_minutes"] == 60
    assert block["separate_duration_minutes"] == 90
    assert block["potential_time_saved_minutes"] == 30


def test_same_department_not_stacked():

    data = {
        "maintenance_jobs": [
            make_job(
                "M001",
                "Engineering",
            ),
            make_job(
                "M002",
                "Engineering",
            ),
        ]
    }

    result = find_shadow_block_candidates(data)

    assert result["candidate_count"] == 0


def test_different_sections_not_stacked():

    data = {
        "maintenance_jobs": [
            make_job(
                "M001",
                "Engineering",
                section="Delhi-Ghaziabad",
            ),
            make_job(
                "M002",
                "Signal & Telecom",
                section="Ghaziabad-Meerut",
            ),
        ]
    }

    result = find_shadow_block_candidates(data)

    assert result["candidate_count"] == 0


def test_incompatible_job_not_stacked():

    data = {
        "maintenance_jobs": [
            make_job(
                "M001",
                "Engineering",
            ),
            make_job(
                "M002",
                "Signal & Telecom",
                compatible=False,
            ),
        ]
    }

    result = find_shadow_block_candidates(data)

    assert result["candidate_count"] == 0


def test_three_departments_stack_together():

    data = {
        "maintenance_jobs": [
            make_job(
                "M001",
                "Engineering",
                duration=60,
            ),
            make_job(
                "M002",
                "Signal & Telecom",
                duration=30,
            ),
            make_job(
                "M003",
                "Traction Distribution",
                duration=45,
            ),
        ]
    }

    result = find_shadow_block_candidates(data)

    assert result["candidate_count"] == 1

    block = result["shadow_block_candidates"][0]

    assert len(block["maintenance_ids"]) == 3

    assert block["shadow_duration_minutes"] == 60

    assert block["separate_duration_minutes"] == 135

    assert block["potential_time_saved_minutes"] == 75


def test_highest_priority_preserved():

    data = {
        "maintenance_jobs": [
            make_job(
                "M001",
                "Engineering",
                priority="MEDIUM",
            ),
            make_job(
                "M002",
                "Signal & Telecom",
                priority="CRITICAL",
            ),
        ]
    }

    result = find_shadow_block_candidates(data)

    block = result["shadow_block_candidates"][0]

    assert block["highest_priority"] == "CRITICAL"


def test_earliest_deadline_preserved():

    data = {
        "maintenance_jobs": [
            make_job(
                "M001",
                "Engineering",
                deadline="18:00",
            ),
            make_job(
                "M002",
                "Signal & Telecom",
                deadline="12:00",
            ),
        ]
    }

    result = find_shadow_block_candidates(data)

    block = result["shadow_block_candidates"][0]

    assert block["earliest_deadline"] == "12:00"


def test_input_is_not_modified():

    data = {
        "maintenance_jobs": [
            make_job(
                "M001",
                "Engineering",
            ),
            make_job(
                "M002",
                "Signal & Telecom",
            ),
        ]
    }

    original = deepcopy(data)

    find_shadow_block_candidates(data)

    assert data == original