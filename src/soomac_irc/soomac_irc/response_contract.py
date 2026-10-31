"""Pure, model-independent input/output contracts; no natural-language inference."""
import copy

from jsonschema import Draft202012Validator, ValidationError

from soomac_irc.agent_contract import RECOMMENDATION_SCHEMA, RESPONSE_SCHEMA, empty_order_patch
from soomac_irc.domain import AMOUNTS, NOODLE_TYPES, SAUCES, TOPPINGS
from soomac_irc.llm_policy import (
    build_recommendation_scope,
    section_execution_items, validate_recommendation_proposal,
)


RESPONSE_HISTORY_TURNS = 10
# Exceptional model failure only; never substitutes for successful dialogue.
RESPONSE_FAILURE_REPLY = "응답을 생성하지 못했어요. 다시 말씀해 주세요."


def validate_response_output(parsed):
    Draft202012Validator(RESPONSE_SCHEMA).validate(parsed)
    reply = parsed["reply"].strip()
    if not reply:
        raise ValueError("Response reply is empty")
    return reply


def build_response_model_input(user_text, session, policy, applied_changes, future_changes,
                               recommendation_result, next_prompt, robot_state, route):
    if type(policy.get("execute")) is not bool:
        raise ValueError("Response policy.execute must be a JSON boolean")
    event = None
    if user_text is None:
        event = robot_state.get("section_transition")
        if not isinstance(event, dict) or event.get("type") != "section_transition":
            raise ValueError("Event input requires an actual section_transition")
        if policy["execute"]:
            raise ValueError("Conversational section event cannot authorize execution")
    elif not isinstance(user_text, str) or not user_text.strip():
        raise ValueError("User input requires nonempty text")
    execution_authorized = policy["execute"] is True
    return {
        "input_source": "event" if event is not None else "user",
        "event": copy.deepcopy(event),
        "user_text": user_text,
        "recent_history": copy.deepcopy(session["history"][-(RESPONSE_HISTORY_TURNS * 2):]),
        "confirmed_order": copy.deepcopy(session["order"]),
        "preferences": copy.deepcopy(session["preferences"]),
        "pending": copy.deepcopy(session.get("pending")),
        "policy": copy.deepcopy(policy),
        "applied_this_turn": copy.deepcopy(applied_changes),
        "future_changes": copy.deepcopy(future_changes),
        "recommendation_result": copy.deepcopy(recommendation_result),
        "next_prompt": copy.deepcopy(next_prompt),
        "robot_state": copy.deepcopy(robot_state),
        "recommendation_scope": build_recommendation_scope(session, robot_state),
        "recent_action_history": copy.deepcopy(session.get("action_history", [])[-20:]),
        "execution_authorized": execution_authorized,
        "starting_now": section_execution_items(session["order"], robot_state["section"])
        if execution_authorized else [],
    }


def build_recommendation_model_input(session, decision, robot_state, user_text):
    event = robot_state.get("section_transition") if user_text is None else None
    if user_text is None and (not isinstance(event, dict) or event.get("type") != "section_transition"):
        raise ValueError("Event recommendation requires an actual section_transition")
    if user_text is not None and (not isinstance(user_text, str) or not user_text.strip()):
        raise ValueError("Recommendation user text must be nonempty")
    scope = build_recommendation_scope(session, robot_state)
    # 단계 전환도 남은 코스 전체를 본다. 물리적으로 지난/보호된 항목과
    # 강한 제한은 공통 scope/validator가 계속 제외한다.
    return {
        "input_source": "event" if event is not None else "user", "event": copy.deepcopy(event),
        "current_user_text": user_text.strip() if user_text is not None else None,
        "recent_history": copy.deepcopy(session["history"][-20:]),
        "order": copy.deepcopy(session["order"]),
        "restrictions": copy.deepcopy(session["order"].get("restrictions", [])),
        "preferences": copy.deepcopy(session["preferences"]), "pending": copy.deepcopy(session.get("pending")),
        "recommendation_action": decision["recommendation"]["action"],
        "robot_state": copy.deepcopy(robot_state), "recommendation_scope": scope,
        "supported_domain": {"sauces": list(SAUCES), "noodle_types": list(NOODLE_TYPES),
                             "toppings": list(TOPPINGS), "amounts": list(AMOUNTS)},
    }


def validate_recommendation_result(session, raw, robot_state):
    """Validate JSON and physical/domain safety, NOT the semantics of reason prose."""
    try:
        Draft202012Validator(RECOMMENDATION_SCHEMA).validate(raw)
    except ValidationError:
        return {
            "proposal": empty_order_patch(), "reason_tags": [], "accepted_fields": [],
            "rejected_fields": [{"reason": "model_schema_error"}],
            "reason_validation": {"source": "model", "schema_valid": False, "semantic_status": "unverified"},
        }
    result = validate_recommendation_proposal(session, raw["proposal"], robot_state)
    result.update({
        "raw_proposal": copy.deepcopy(raw["proposal"]),
        "reason_tags": copy.deepcopy(raw["reason_tags"]),
        "reason_validation": {"source": "model", "schema_valid": True, "semantic_status": "unverified"},
    })
    return result
