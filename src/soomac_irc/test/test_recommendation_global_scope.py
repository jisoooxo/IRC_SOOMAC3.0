"""Whole remaining-course candidates; mock contracts, not LLM quality scores."""
import copy
from unittest.mock import Mock

import pytest

from soomac_irc.agent_contract import new_decision, new_session_state, new_turn_state, normalize_decision
from soomac_irc.agent_prompts import RECOMMENDATION_SYSTEM
from soomac_irc.llm_langgraph import build_graph, generate_section_event_response
from soomac_irc.llm_policy import section_execution_items
from soomac_irc.response_contract import build_recommendation_model_input, validate_recommendation_result


def robot():
    return dict(section="veggie", active_task=None, task_queue=[], completed_tasks=[],
                robot_started=False, section_transition=dict(
                    type="section_transition", section="noodle", next_section="veggie", outcome="completed"))


def candidate():
    return dict(proposal=dict(sauce="크림", noodle_type=None, noodle_portion=None,
                             toppings={"버섯": "low", "소시지": "high", "치즈": "normal"}),
                reason_tags=["남은 코스 조합"])


def test_event_and_user_share_remaining_scope_without_fake_user_or_mutation():
    session = new_session_state()
    session["history"] = [{"role": "user", "content": "매운 걸 못 먹어요."}]
    session["order"]["restrictions"] = [{"target": "매운맛", "reason": "cannot_eat"}]
    state, decision = robot(), new_decision()
    decision["recommendation"]["action"] = "request"
    before = copy.deepcopy((session, state))
    event = build_recommendation_model_input(session, decision, state, None)
    user = build_recommendation_model_input(session, decision, state, "전체적으로 추천해줘")
    assert event["recommendation_scope"] == user["recommendation_scope"]
    assert [x["section"] for x in event["recommendation_scope"]["sections"]] == ["veggie", "meat", "extra", "sauce"]
    assert event["input_source"] == "event" and event["current_user_text"] is None
    assert event["recent_history"] == session["history"]
    assert "페퍼론치노" not in next(x for x in event["recommendation_scope"]["sections"] if x["section"] == "extra")["options"]
    event["recent_history"].clear()
    assert (session, state) == before


def test_event_future_candidates_remain_pending_until_acceptance_no_execution():
    session, state = new_session_state(), robot()
    before = copy.deepcopy(session)
    result = generate_section_event_response(session, state, Mock(return_value=candidate()),
        Mock(return_value="남은 구성으로 이 후보들을 선택할까요?"), "기존 안내")
    assert result["recommendation_result"]["rejected_fields"] == []
    assert result["session"]["pending"]["candidate"] == candidate()["proposal"]
    assert result["session"]["order"] == before["order"]
    assert result["session"]["history"] == before["history"]
    assert result["policy"]["execute"] is False
    assert session == before
    accepted = build_graph(lambda *_: normalize_decision(dict(route="task", confirmation="accept")),
                           Mock(), Mock(return_value="후보를 선택했어요.")).invoke(
        new_turn_state(result["session"], "그 구성으로 골라줘", state))
    assert accepted["session"]["order"]["toppings"] == candidate()["proposal"]["toppings"]
    assert accepted["session"]["order"]["sauce"] == "크림"
    assert accepted["policy"]["execute"] is False
    assert section_execution_items(accepted["session"]["order"], "veggie") == [{"item": "버섯", "amount": "low"}]


@pytest.mark.parametrize("restriction,blocked", [
    ({"target": "유제품", "reason": "allergy"}, {"sauce", "toppings.치즈"}),
    ({"target": "소시지", "reason": "cannot_eat"}, {"toppings.소시지"}),
])
def test_global_candidates_still_reject_strong_restrictions(restriction, blocked):
    session = new_session_state()
    session["order"]["restrictions"] = [restriction]
    result = validate_recommendation_result(session, candidate(), robot())
    assert blocked <= {x["field"] for x in result["rejected_fields"]}
    assert not blocked.intersection(result["accepted_fields"])
    assert "toppings.버섯" in result["accepted_fields"]


def test_global_candidates_preserve_past_active_and_completed_protection():
    state = robot()
    state["active_task"] = {"class": "버섯", "repeat_count": 1}
    state["completed_tasks"] = [{"class": "양파", "repeat_count": 1}]
    raw = candidate()
    raw["proposal"]["noodle_type"] = "넓은면"
    raw["proposal"]["toppings"]["양파"] = "high"
    result = validate_recommendation_result(new_session_state(), raw, state)
    rejected = {x["field"] for x in result["rejected_fields"]}
    assert {"noodle_type", "toppings.버섯", "toppings.양파"} <= rejected
    assert {"sauce", "toppings.소시지", "toppings.치즈"} <= set(result["accepted_fields"])


def test_global_candidates_do_not_synthesize_sauce_or_allow_unsupported_menus():
    raw = dict(proposal=dict(sauce=None, noodle_type=None, noodle_portion=None,
                            toppings={"소시지": "normal", "햄": "high"}), reason_tags=[])
    result = validate_recommendation_result(new_session_state(), raw, robot())
    assert result["proposal"]["sauce"] is None  # no Python sauce/default inference
    assert result["proposal"]["toppings"] == {"소시지": "normal"}
    assert any(x["field"] == "toppings.햄" for x in result["rejected_fields"])


def test_prompt_explains_missing_sauce_and_keeps_exceptions_not_a_semantic_eval():
    assert "proposal.sauce에 구체적인 소스 후보를 포함" in RECOMMENDATION_SYSTEM
    assert "유제품 제한" in RECOMMENDATION_SYSTEM
    assert "소스를 나중에 고르겠다고 명시" in RECOMMENDATION_SYSTEM
    assert "event에서는 현재 단계의 선택지만" not in RECOMMENDATION_SYSTEM
