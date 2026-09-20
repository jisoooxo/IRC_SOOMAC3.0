import copy

from soomac_irc.agent_v7 import (
    build_policy_reply,
    classify_confirmation_intent,
    commit_section_skip,
    validate_transaction,
)
from soomac_irc.call_model_v7 import parse_free_reply
from soomac_irc.order_v7 import new_order


def _execution(section="noodle"):
    return {
        "section": section,
        "selected": [],
        "task_queue": [],
        "active_task": None,
        "completed_tasks": [],
        "skipped_sections": {},
        "robot_started": False,
    }


def _state(
    action,
    changes=None,
    *,
    user_text="test",
    order=None,
    execution=None,
    recommendation=None,
):
    return {
        "user_text": user_text,
        "order": copy.deepcopy(order if order is not None else new_order()),
        "execution": copy.deepcopy(
            execution if execution is not None else _execution()
        ),
        "recommendation": copy.deepcopy(
            recommendation
            if recommendation is not None
            else {"phase": "idle", "confirmed_sections": []}
        ),
        "history": [],
        "action_history": [],
        "tool_call": {
            "call_id": f"test_{action}",
            "name": action,
            "changes": copy.deepcopy(changes or {}),
        },
    }


def test_set_order_changes_value_without_starting_work():
    result = validate_transaction(_state(
        "set_order",
        {"toppings": {"양파": "low"}},
        execution=_execution("veggie"),
    ))

    assert result["order"]["toppings"] == {"양파": "low"}
    assert result["transaction"]["accepted"] == ["toppings.양파"]
    assert result["transaction"]["confirm_validation"]["requested"] is False


def test_set_order_and_confirm_requires_complete_valid_section():
    result = validate_transaction(_state(
        "set_order_and_confirm",
        {
            "sauce": "토마토",
            "noodle_type": "얇은면",
            "noodle_portion": "normal",
        },
        user_text="이대로 진행해",
    ))

    assert result["transaction"]["confirm_validation"] == {
        "requested": True,
        "allowed": True,
        "missing": {},
        "reason": "ready",
    }


def test_refuse_section_dislike_is_staged_until_acceptance():
    order = new_order()
    order["toppings"] = {"양파": "low"}
    execution = _execution("veggie")
    execution["selected"] = ["toppings.양파"]
    before_order = copy.deepcopy(order)
    before_execution = copy.deepcopy(execution)
    result = validate_transaction(_state(
        "refuse_section",
        {"reason": "dislike", "items": ["양파", "버섯"]},
        order=order,
        execution=execution,
    ))
    skip = result["transaction"]["section_skip"]

    assert skip["needs_confirmation"] is True
    assert skip["applied"] is False
    assert result["order"] == before_order
    assert result["execution"] == before_execution

    committed = commit_section_skip(
        result["order"],
        result["execution"],
        result["recommendation"],
        skip,
    )

    assert committed["order"]["toppings"] == {}
    assert committed["order"]["restrictions"] == [
        {"target": "양파", "reason": "dislike"},
        {"target": "버섯", "reason": "dislike"},
    ]
    assert committed["execution"]["skipped_sections"] == {
        "veggie": "dislike"
    }


def test_recommend_order_creates_proposal_without_mutating_order():
    order = new_order()
    result = validate_transaction(_state(
        "recommend_order",
        {
            "scope": "current",
            "recommended_order": {
                "sauce": "토마토",
                "noodle_type": "얇은면",
                "noodle_portion": "normal",
            },
        },
        order=order,
    ))

    assert result["order"] == order
    assert result["recommendation"]["phase"] == "confirming"
    assert result["transaction"]["recommendation_validation"][
        "proposal_created"
    ] is True


def test_confirm_section_allows_complete_current_section():
    order = new_order()
    order["toppings"] = {"양파": "low"}
    result = validate_transaction(_state(
        "confirm_section",
        user_text="응 진행해",
        order=order,
        execution=_execution("veggie"),
    ))

    assert result["transaction"]["confirm_validation"]["requested"] is True
    assert result["transaction"]["confirm_validation"]["allowed"] is True


def test_confirmation_parser_prefers_rejection_and_reasks_on_ambiguity():
    assert classify_confirmation_intent(
        "네, 그런데 진행하지 마",
        "start",
    ) is False
    assert classify_confirmation_intent("음 생각해볼게", "start") is None
    assert classify_confirmation_intent("그대로 진행해", "start") is True


def test_set_order_and_confirm_without_explicit_start_only_updates_selection():
    result = validate_transaction(_state(
        "set_order_and_confirm",
        {
            "sauce": "토마토",
            "noodle_type": "얇은면",
            "noodle_portion": "normal",
        },
        user_text="토마토 얇은면 보통으로",
    ))

    assert result["order"]["sauce"] == "토마토"
    assert result["transaction"]["confirm_validation"] == {
        "requested": True,
        "allowed": False,
        "missing": {},
        "reason": "explicit_confirmation_required",
    }


def test_confirm_section_without_explicit_agreement_cannot_start():
    order = new_order()
    order["toppings"] = {"양파": "low"}
    result = validate_transaction(_state(
        "confirm_section",
        user_text="양파 적게 맞지?",
        order=order,
        execution=_execution("veggie"),
    ))

    assert result["transaction"]["confirm_validation"]["allowed"] is False
    assert result["transaction"]["confirm_validation"]["reason"] == (
        "explicit_confirmation_required"
    )


def test_recommendation_is_not_applied_without_explicit_agreement():
    order = new_order()
    recommendation = {
        "phase": "confirming",
        "request": {"scope": "current"},
        "recommended_changes": {
            "sauce": "토마토",
            "noodle_type": "얇은면",
            "noodle_portion": "normal",
        },
        "confirmed_sections": [],
    }
    result = validate_transaction(_state(
        "confirm_section",
        user_text="그 조합 맛이 어때?",
        order=order,
        recommendation=recommendation,
    ))

    assert result["order"] == order
    assert result["recommendation"] == recommendation
    assert result["transaction"]["recommendation_validation"][
        "proposal_confirmed"
    ] is False
    assert result["transaction"]["confirm_validation"]["allowed"] is False


def test_cancel_order_is_explicitly_blocked_without_pending_confirmation():
    order = new_order()
    order["sauce"] = "토마토"
    result = validate_transaction(_state("cancel_order", order=order))
    cancel = result["transaction"]["cancel_validation"]

    assert cancel["requested"] is True
    assert cancel["blocked"] is True
    assert cancel["needs_confirmation"] is False
    assert result["order"] == order


def test_respond_does_not_mutate_order_or_execution():
    state = _state("respond", execution=_execution("meat"))
    result = validate_transaction(state)

    assert result["order"] == state["order"]
    assert result["execution"] == state["execution"]
    assert result["transaction"]["action"] == "respond"


def test_describe_scene_does_not_mutate_order_or_execution():
    state = _state("describe_scene", execution=_execution("extra"))
    result = validate_transaction(state)

    assert result["order"] == state["order"]
    assert result["execution"] == state["execution"]
    assert result["transaction"]["action"] == "describe_scene"


def test_all_hard_restricted_options_skip_section_immediately():
    result = validate_transaction(_state(
        "set_order",
        {
            "restriction_changes": [
                {"target": "양파", "reason": "allergy", "enabled": True},
                {"target": "버섯", "reason": "cannot_eat", "enabled": True},
            ]
        },
        execution=_execution("veggie"),
    ))
    skip = result["transaction"]["section_skip"]

    assert skip["source"] == "all_options_restricted"
    assert skip["needs_confirmation"] is False
    assert skip["applied"] is True
    assert result["execution"]["skipped_sections"]["veggie"] == "allergy"


def test_all_disliked_options_create_skip_confirmation():
    result = validate_transaction(_state(
        "set_order",
        {
            "restriction_changes": [
                {"target": "양파", "reason": "dislike", "enabled": True},
                {"target": "버섯", "reason": "dislike", "enabled": True},
            ]
        },
        execution=_execution("veggie"),
    ))
    skip = result["transaction"]["section_skip"]

    assert skip["source"] == "all_options_restricted"
    assert skip["needs_confirmation"] is True
    assert skip["applied"] is False
    assert result["execution"]["skipped_sections"] == {}
    assert len(result["order"]["restrictions"]) == 2


def test_allergy_after_robot_start_warns_without_stopping_robot():
    order = new_order()
    order["toppings"] = {"양파": "normal"}
    execution = _execution("veggie")
    execution["active_task"] = {"class": "양파", "repeat_count": 1}
    execution["robot_started"] = True
    result = validate_transaction(_state(
        "set_order",
        {
            "restriction_changes": [
                {"target": "양파", "reason": "allergy", "enabled": True},
            ],
        },
        user_text="양파 알레르기가 있어",
        order=order,
        execution=execution,
    ))
    policy = build_policy_reply(result)["policy_reply"]

    assert result["order"]["toppings"]["양파"] == "normal"
    assert result["execution"]["active_task"] == execution["active_task"]
    assert result["execution"]["robot_started"] is True
    assert "섭취에 유의" in policy
    assert "stop" not in result["transaction"]


def test_free_reply_parser_accepts_only_explanation_field():
    assert parse_free_reply(
        '{"explanation":"토마토 소스는 산뜻한 맛이에요."}'
    ) == "토마토 소스는 산뜻한 맛이에요."
    assert parse_free_reply(
        '{"explanation":"설명", "next_step":"야채 담기"}'
    ) is None
    assert parse_free_reply("JSON이 아닌 문장") is None
