import copy
import json
from pathlib import Path

import pytest

from soomac_irc.agent_contract import new_session_state
from soomac_irc.evaluation_v43 import (
    grade_decision_prediction,
    grade_observation,
    grade_recommendation_prediction,
    grade_response_prediction,
)


REPO = Path(__file__).resolve().parents[3]


def robot(section="meat", *, active=None, queue=None):
    return {
        "section": section,
        "active_task": copy.deepcopy(active),
        "task_queue": copy.deepcopy(queue or []),
        "completed_tasks": [],
    }


def execution_pending(item="소시지", amount="high", section="meat"):
    return {
        "type": "execution",
        "section": section,
        "source": "current_update",
        "targets": [item],
        "items": {item: amount},
    }


def direct_commit_case(user_text="소시지 많이 넣고 바로 담아줘"):
    return {
        "id": "S41-direct",
        "user_text": user_text,
        "allowed_session_changes": ["order.toppings.소시지"],
        "expected": {
            "decision_any_of": [
                {"route": "task", "order": {"toppings": {"소시지": "high"}}, "commit": True}
            ],
            "session": {"order": {"toppings": {"소시지": "high"}}, "pending": None},
            "policy": {"execute": True},
            "plan_count": 1,
            "first_plan": {"section": "meat", "item": "소시지", "amount": "high"},
        },
    }


def direct_commit_observation():
    before = new_session_state()
    after = copy.deepcopy(before)
    after["order"]["toppings"]["소시지"] = "high"
    return {
        "before_session": before,
        "session": after,
        "robot_state": robot(),
        "decision": {
            "route": "task",
            "order": {"toppings": {"소시지": "high"}},
            "commit": True,
        },
        "policy": {"execute": True, "reason": "commit"},
        "plans": [{"section": "meat", "item": "소시지", "amount": "high"}],
        "reply": "소시지를 많이 담기 시작할게요.",
        "verified_event": "PLAN_PUBLISHED",
        "response_state_mutations": [],
    }


def pending_accept_case(user_text="바로 해줘"):
    return {
        "id": "S39-pending-accept",
        "user_text": user_text,
        "allowed_session_changes": ["pending"],
        "expected": {
            "decision_any_of": [
                {"route": "task", "confirmation": "accept"},
                {"route": "task", "commit": True},
            ],
            "session": {"pending": None, "order": {"toppings": {"소시지": "high"}}},
            "policy": {"execute": True},
            "plan_count": 1,
            "first_plan": {"section": "meat", "item": "소시지", "amount": "high"},
        },
    }


def pending_accept_observation(user_text="바로 해줘"):
    before = new_session_state()
    before["order"]["toppings"]["소시지"] = "high"
    before["pending"] = execution_pending()
    after = copy.deepcopy(before)
    after["pending"] = None
    return {
        "before_session": before,
        "session": after,
        "robot_state": robot(),
        "decision": {"route": "task", "confirmation": "accept"},
        "policy": {"execute": True, "reason": "execution_accepted"},
        "plans": [{"section": "meat", "item": "소시지", "amount": "high"}],
        "reply": "소시지를 많이 담기 시작할게요.",
        "verified_event": "PLAN_PUBLISHED",
        "response_state_mutations": [],
        "user_text": user_text,
    }


def ordinary_order_case():
    return {
        "id": "S42-order-no-commit",
        "allowed_session_changes": ["order.toppings.소시지", "pending"],
        "expected": {
            "decision_any_of": [
                {"route": "task", "order": {"toppings": {"소시지": "high"}}}
            ],
            "session": {
                "order": {"toppings": {"소시지": "high"}},
                "pending": {"type": "execution", "section": "meat"},
            },
            "policy": {"execute": False},
            "plan_count": 0,
        },
    }


def ordinary_order_observation():
    before = new_session_state()
    after = copy.deepcopy(before)
    after["order"]["toppings"]["소시지"] = "high"
    after["pending"] = execution_pending()
    return {
        "before_session": before,
        "session": after,
        "robot_state": robot(),
        "decision": {"route": "task", "order": {"toppings": {"소시지": "high"}}},
        "policy": {"execute": False, "reason": "confirmation_required"},
        "plans": [],
        "reply": "소시지를 많이 선택했어요. 지금 담을까요?",
        "response_state_mutations": [],
    }


@pytest.mark.parametrize(
    "utterance",
    ["소시지 많이 넣고 바로 시작해", "소시지 많이 넣고 바로 진행해", "소시지 많이 바로 담아줘", "소시지 많이 넣고 즉시 진행해"],
)
def test_direct_commit_synonyms_share_one_semantic_oracle(utterance):
    grade = grade_observation(direct_commit_case(utterance), direct_commit_observation())
    assert grade.passed, grade.to_dict()


@pytest.mark.parametrize("utterance", ["바로 해줘", "응", "그렇게 해", "진행해 주세요", "네 지금 해주세요"])
def test_pending_accept_synonyms_do_not_require_the_word_barro(utterance):
    grade = grade_observation(pending_accept_case(utterance), pending_accept_observation(utterance))
    assert grade.passed, grade.to_dict()


def test_ordinary_order_requires_zero_plan_and_execution_pending():
    grade = grade_observation(ordinary_order_case(), ordinary_order_observation())
    assert grade.passed, grade.to_dict()


@pytest.mark.parametrize(
    "mutation, expected_code",
    [
        (lambda row: row["decision"].pop("commit"), "decision_semantics"),
        (lambda row: row["plans"].append(copy.deepcopy(row["plans"][0])), "plan_count"),
        (lambda row: row["plans"][0].update({"amount": "low"}), "value_mismatch"),
        (lambda row: row["plans"][0].update({"section": "extra"}), "value_mismatch"),
    ],
)
def test_direct_commit_mutations_are_detected(mutation, expected_code):
    observation = direct_commit_observation()
    mutation(observation)
    grade = grade_observation(direct_commit_case(), observation)
    assert not grade.passed
    assert expected_code in {item.code for item in grade.violations}


def test_pending_presence_changes_the_same_acceptance_from_pass_to_fail():
    observation = pending_accept_observation("응")
    observation["before_session"]["pending"] = None
    grade = grade_observation(pending_accept_case("응"), observation)
    assert not grade.passed
    assert "orphan_confirmation" in {item.code for item in grade.violations}


def test_stale_pending_section_cannot_execute():
    observation = pending_accept_observation()
    observation["before_session"]["pending"]["section"] = "veggie"
    grade = grade_observation(pending_accept_case(), observation)
    assert not grade.passed
    assert "stale_execution_pending" in {item.code for item in grade.violations}


def test_restriction_conflict_cannot_publish_a_plan():
    observation = direct_commit_observation()
    observation["session"]["order"]["restrictions"] = [
        {"target": "소시지", "reason": "cannot_eat"}
    ]
    observation["before_session"]["order"]["restrictions"] = copy.deepcopy(
        observation["session"]["order"]["restrictions"]
    )
    grade = grade_observation(direct_commit_case(), observation)
    assert not grade.passed
    assert "restriction_plan_conflict" in {item.code for item in grade.violations}


def test_busy_state_cannot_execute_even_with_commit():
    observation = direct_commit_observation()
    observation["robot_state"] = robot(active={"item": "게살"})
    grade = grade_observation(direct_commit_case(), observation)
    assert not grade.passed
    assert "busy_execution" in {item.code for item in grade.violations}


def test_response_cannot_claim_completion_before_vlm_pass():
    observation = direct_commit_observation()
    observation["reply"] = "소시지 담기가 완료됐어요."
    grade = grade_observation(direct_commit_case(), observation)
    assert not grade.passed
    assert "unverified_completion_reply" in {item.code for item in grade.violations}


def test_response_is_read_only():
    observation = ordinary_order_observation()
    observation["response_state_mutations"] = ["order.toppings.게살"]
    grade = grade_observation(ordinary_order_case(), observation)
    assert not grade.passed
    assert "response_state_mutation" in {item.code for item in grade.violations}


def test_preference_removal_does_not_remove_confirmed_order():
    before = new_session_state()
    before["order"]["toppings"]["소시지"] = "high"
    before["preferences"] = [{"value": "소시지를 좋아함"}]
    after = copy.deepcopy(before)
    after["preferences"] = []
    case = {
        "id": "S45-preference-remove",
        "allowed_session_changes": ["preferences"],
        "expected": {
            "decision_any_of": [
                {"route": "task", "preferences": {"$contains": [{"value": "소시지를 좋아함", "action": "remove"}]}}
            ],
            "session": {"order": {"toppings": {"소시지": "high"}}, "preferences": []},
            "policy": {"execute": False},
            "plan_count": 0,
        },
    }
    observation = {
        "before_session": before,
        "session": after,
        "robot_state": robot(),
        "decision": {"route": "task", "preferences": [{"value": "소시지를 좋아함", "action": "remove"}]},
        "policy": {"execute": False},
        "plans": [],
        "reply": "소시지 선호는 지웠고 주문 수량은 그대로예요.",
        "response_state_mutations": [],
    }
    grade = grade_observation(case, observation)
    assert grade.passed, grade.to_dict()


def test_strong_restriction_removal_stays_pending_until_acceptance():
    before = new_session_state()
    restriction = {"target": "게살", "reason": "allergy"}
    before["order"]["restrictions"] = [restriction]
    after = copy.deepcopy(before)
    after["pending"] = {"type": "restriction_removal", "restrictions": [restriction]}
    case = {
        "id": "S46-strong-restriction-remove",
        "allowed_session_changes": ["pending"],
        "expected": {
            "decision_any_of": [
                {"route": "task", "restrictions": {"$contains": [{"target": "게살", "reason": "allergy", "action": "remove"}]}}
            ],
            "session": {"order": {"restrictions": {"$contains": [restriction]}}, "pending": {"type": "restriction_removal"}},
            "policy": {"execute": False},
            "plan_count": 0,
        },
    }
    observation = {
        "before_session": before,
        "session": after,
        "robot_state": robot(),
        "decision": {"route": "task", "restrictions": [{**restriction, "action": "remove"}]},
        "policy": {"execute": False, "reason": "restriction_removal_confirmation_required"},
        "plans": [],
        "reply": "게살 알레르기 제한은 아직 유지 중이에요. 정말 지울까요?",
        "response_state_mutations": [],
    }
    grade = grade_observation(case, observation)
    assert grade.passed, grade.to_dict()


def test_recommendation_unspecified_amount_accepts_any_supported_amount():
    case = {
        "id": "R-flexible-amount",
        "input": {
            "current_user_text": "육류 재료 하나 추천해 줘",
            "order": {"toppings": {}, "restrictions": []},
            "restrictions": [],
            "recommendation_scope": {
                "sections": [{"fields": ["toppings.소시지", "toppings.게살"]}]
            },
        },
        "target": {"proposal": {"toppings": {"소시지": "normal"}}},
        "metadata": {"required_proposal_fields": {"toppings.소시지": "normal"}},
    }
    prediction = {
        "proposal": {"sauce": None, "noodle_type": None, "noodle_portion": None, "toppings": {"소시지": "low"}},
        "reason_tags": ["육류 추천"],
    }
    grade = grade_recommendation_prediction(case, prediction)
    assert grade.passed, grade.to_dict()


def test_recommendation_explicit_amount_remains_strict():
    case = {
        "id": "R-explicit-amount",
        "input": {
            "current_user_text": "소시지를 보통 양으로 추천해 줘",
            "order": {"toppings": {}, "restrictions": []},
            "restrictions": [],
            "recommendation_scope": {"sections": [{"fields": ["toppings.소시지"]}]},
        },
        "target": {"proposal": {"toppings": {"소시지": "normal"}}},
        "metadata": {"required_proposal_fields": {"toppings.소시지": "normal"}},
    }
    prediction = {
        "proposal": {"sauce": None, "noodle_type": None, "noodle_portion": None, "toppings": {"소시지": "low"}},
        "reason_tags": [],
    }
    grade = grade_recommendation_prediction(case, prediction)
    assert not grade.passed
    assert "required_proposal_missing" in {item.code for item in grade.violations}


def response_case(*, transition=False, preferences=None, history=None):
    robot_state = {
        "section": "meat",
        "task_queue": [],
        "active_task": None,
        "completed_tasks": [],
        "robot_started": False,
    }
    if transition:
        robot_state["section_transition"] = {
            "type": "section_transition",
            "section": "veggie",
            "outcome": "skipped",
            "next_section": "meat",
        }
    return {
        "id": "P-semantic",
        "input": {
            "user_text": "" if transition else "게살 적게 넣어 줘",
            "recent_history": copy.deepcopy(history or []),
            "confirmed_order": {"toppings": {"게살": "low"}},
            "preferences": copy.deepcopy(preferences or []),
            "pending": None,
            "policy": {"execute": False, "reason": "section_transition" if transition else "state_update_only"},
            "applied_this_turn": {
                "order_changes": {} if transition else {"toppings.게살": "low"},
                "restriction_added": [],
                "restriction_removed": [],
                "preference_added": [],
                "preference_removed": [],
            },
            "future_changes": [],
            "recommendation_result": None,
            "next_prompt": {"type": "section_prompt", "section": "meat"} if transition else None,
            "robot_state": robot_state,
            "recommendation_scope": {},
            "recent_action_history": [],
            "execution_authorized": False,
            "starting_now": [],
        },
    }


def test_response_amount_synonyms_are_semantic_not_literal():
    grade = grade_response_prediction(response_case(), "게살은 적은 양으로 선택했어요. 이대로 담을까요?")
    assert grade.passed, grade.to_dict()


def test_response_preference_replacement_can_name_removed_and_active_values():
    case = response_case()
    case["input"]["applied_this_turn"]["order_changes"] = {}
    case["input"]["preferences"] = [{"value": "매콤하게"}]
    case["input"]["applied_this_turn"]["preference_removed"] = [{"value": "맵지 않게"}]
    case["input"]["applied_this_turn"]["preference_added"] = [{"value": "매콤하게"}]
    grade = grade_response_prediction(case, "맵지 않게 선호는 취소하고 매콤하게로 바꿨어요.")
    assert grade.passed, grade.to_dict()


def test_response_section_event_cannot_revive_withdrawn_preference():
    case = response_case(
        transition=True,
        preferences=[],
        history=[
            {"role": "user", "content": "소시지를 좋아해"},
            {"role": "user", "content": "그 취향은 취소할게"},
        ],
    )
    grade = grade_response_prediction(case, "소시지를 좋아한다고 하셨죠. 소시지를 넣으실까요?")
    assert not grade.passed
    assert "withdrawn_preference_revived" in {item.code for item in grade.violations}


def test_response_unauthorized_start_claim_fails():
    grade = grade_response_prediction(response_case(), "게살을 적게 담기 시작할게요.")
    assert not grade.passed
    assert "unauthorized_execution_reply" in {item.code for item in grade.violations}


def test_response_execution_question_is_not_a_start_claim():
    grade = grade_response_prediction(response_case(), "게살은 적은 양으로 준비했어요. 담기를 진행할까요?")
    assert grade.passed, grade.to_dict()


def test_response_negative_completion_status_is_not_a_completion_claim():
    case = response_case()
    case["input"]["applied_this_turn"]["order_changes"] = {}
    grade = grade_response_prediction(case, "지금까지 완료된 재료는 아직 없어요.")
    assert grade.passed, grade.to_dict()


def test_response_generic_options_do_not_revive_withdrawn_preference():
    case = response_case(
        transition=True,
        preferences=[],
        history=[
            {"role": "user", "content": "소시지를 좋아해"},
            {"role": "user", "content": "소시지 선호는 철회할게"},
        ],
    )
    grade = grade_response_prediction(case, "이제 육류 단계예요. 소시지나 게살 중 원하는 재료를 말씀해 주세요.")
    assert grade.passed, grade.to_dict()


def test_s01_to_s50_registry_is_complete_unique_and_evidence_scoped():
    registry = json.loads(
        (REPO / "evaluation/v4_3/scenario_registry.json").read_text(encoding="utf-8")
    )
    scenarios = registry["scenarios"]
    expected_ids = [f"S{index:02d}" for index in range(1, 51)]
    assert [item["id"] for item in scenarios] == expected_ids
    assert len({item["id"] for item in scenarios}) == 50
    assert all(item["family"] and item["owner"] and item["oracle"] for item in scenarios)
    assert all(
        item["evidence"]
        in {"model_shadow", "state_trace", "shadow_plan", "rollout", "event_trace", "audio_event_trace"}
        for item in scenarios
    )
    physical_or_audio = {
        "S01", "S09", "S10", "S11", "S18", "S26", "S27", "S28"
    }
    by_id = {item["id"]: item for item in scenarios}
    assert all(by_id[item_id]["evidence"].endswith("event_trace") for item_id in physical_or_audio)


def test_decision_shadow_suite_has_unique_ids_and_all_high_risk_families():
    rows = [
        json.loads(line)
        for line in (REPO / "evaluation/v4_3/decision_shadow_suite.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(rows) == 25
    assert len({row["id"] for row in rows}) == len(rows)
    families = {row["family"] for row in rows}
    assert {
        "compound_restriction_preference_recommendation",
        "direct_commit",
        "ordinary_order",
        "pending_accept",
        "preference_remove",
        "strong_restriction_remove",
        "pending_replace_direct_commit",
    } <= families


def test_raw_decision_evaluator_rejects_forbidden_commit_on_ordinary_order():
    case = {
        "id": "ordinary",
        "pending": None,
        "expected_any": [
            {"route": "task", "order": {"toppings": {"소시지": "high"}}}
        ],
        "forbidden_any": [{"commit": True}],
    }
    prediction = {
        "route": "task",
        "order": {"toppings": {"소시지": "high"}},
        "commit": True,
    }
    grade = grade_decision_prediction(case, prediction)
    assert not grade.passed
    assert "forbidden_decision_semantics" in {item.code for item in grade.violations}

