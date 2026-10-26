"""Prompt/schema/state contracts, NOT evidence of model semantic performance."""
import copy
import json
from unittest.mock import Mock

import pytest
from jsonschema import Draft202012Validator

from soomac_irc import agent_prompts as prompts
from soomac_irc.agent_contract import (
    DECISION_SCHEMA, RECOMMENDATION_SCHEMA, RESPONSE_SCHEMA,
    new_session_state, new_turn_state, normalize_decision,
)
from soomac_irc.decision_model import build_decision_model_input, DECISION_HISTORY_TURNS
from soomac_irc.llm_langgraph import build_graph
from soomac_irc.llm_langgraph import generate_section_event_response
from soomac_irc.prompt_examples import EXAMPLES, PROMPT_EXAMPLE_VERSION, render_examples
from soomac_irc.response_contract import (
    build_recommendation_model_input, build_response_model_input, validate_recommendation_result,
)

CASES = [(group, ex) for group, values in EXAMPLES.items() for ex in values]


def session_for(group, data):
    session = new_session_state()
    session.update(order=copy.deepcopy(data["order" if group in ("decision", "recommendation") else "confirmed_order"]),
                   preferences=copy.deepcopy(data["preferences"]), pending=copy.deepcopy(data["pending"]),
                   history=copy.deepcopy(data["recent_history"]),
                   action_history=copy.deepcopy(data.get("recent_action_history", [])))
    return session


@pytest.mark.parametrize("group,example", CASES, ids=[e["id"] for _, e in CASES])
def test_every_demo_has_real_serializer_shape_schema_and_no_live_state_mutation(group, example):
    data = copy.deepcopy(example["input"])
    session = session_for(group, data)
    before = copy.deepcopy(session)
    state = copy.deepcopy(data["robot_state"])
    if group == "decision":
        rebuilt = build_decision_model_input(session, data["message"], state)
        schema = DECISION_SCHEMA
    elif group == "recommendation":
        rebuilt = build_recommendation_model_input(session, {"recommendation": {"action": data["recommendation_action"]}},
                                                   state, data["current_user_text"])
        schema = RECOMMENDATION_SCHEMA
        result = validate_recommendation_result(session, example["output"], state)
        assert not result["rejected_fields"]
        assert result["accepted_fields"]
    else:
        rebuilt = build_response_model_input(data["user_text"], session, data["policy"], data["applied_this_turn"],
            data["future_changes"], data["recommendation_result"], data["next_prompt"], state, group)
        schema = RESPONSE_SCHEMA
        assert type(data["execution_authorized"]) is bool
        assert set(data["applied_this_turn"]) == {
            "order_changes", "restriction_added", "restriction_removed", "preference_added", "preference_removed"}
        if data["input_source"] == "event":
            assert data["user_text"] is None
            assert data["event"]["type"] == "section_transition"
            assert not data["execution_authorized"]
    Draft202012Validator(schema).validate(example["output"])
    assert rebuilt == data
    assert session == before and example["input"] == data


@pytest.mark.parametrize("example", EXAMPLES["decision"], ids=[e["id"] for e in EXAMPLES["decision"]])
def test_decision_demos_replay_expected_authority_and_full_order(example):
    data = example["input"]
    session = session_for("decision", data)
    before = copy.deepcopy(session)
    proposal = dict(proposal=dict(sauce=None, noodle_type=None, noodle_portion=None, toppings={}), reason_tags=[])
    result = build_graph(lambda *_: normalize_decision(copy.deepcopy(example["output"])),
                         lambda *_: proposal, lambda *_: "테스트 응답").invoke(
        new_turn_state(session, data["message"], data["robot_state"]))
    expected_order = copy.deepcopy(before["order"])
    if example["id"] == "D01":
        expected_order["toppings"] = {"양파": "high", "버섯": "low"}
    elif example["id"] == "D04":
        expected_order["toppings"].pop("버섯")
        expected_order["sauce"] = "토마토"
    elif example["id"] == "D09":
        expected_order["restrictions"].append({"target": "유제품", "reason": "cannot_eat"})
    elif example["id"] == "N03":
        expected_order["toppings"] = {}
    elif example["id"] == "N05":
        expected_order.update(noodle_type="넓은면", noodle_portion="low")
    assert result["session"]["order"] == expected_order
    assert result["policy"]["execute"] is (example["id"] in {"D02", "D04", "N02", "N03"})
    if example["id"] == "N03":
        # Existing policy uses execute for authorized stage advance too, not
        # ingredient execution. Actual node plan=0 is covered by GL05 fast-path.
        assert result["policy"]["reason"] == "section_skip_allowed"
        assert not result["session"]["order"]["toppings"]
    if example["id"] == "D10":
        assert result["session"]["preferences"] == []
        assert result["session"]["pending"]["type"] == "restriction_removal"
    else:
        assert result["session"]["preferences"] == before["preferences"]
    assert result["session"]["history"][:-2] == before["history"]
    assert session == before


def test_examples_are_static_independent_system_content_not_live_history():
    assert PROMPT_EXAMPLE_VERSION == prompts.PROMPT_CONTRACT_VERSION
    assert DECISION_HISTORY_TURNS == 10  # 20 messages, not 20 pairs.
    systems = dict(decision=prompts.DECISION_SYSTEM, recommendation=prompts.RECOMMENDATION_SYSTEM,
                   task=prompts.TASK_RESPONSE_SYSTEM, general=prompts.GENERAL_RESPONSE_SYSTEM,
                   mixed=prompts.MIXED_RESPONSE_SYSTEM)
    for group, system in systems.items():
        assert system.count("[독립 시연 — 현재 사용자 사실이 아님]") == 1
        assert system.endswith(render_examples(group))
        for ex in EXAMPLES[group]:
            assert json.dumps(ex, ensure_ascii=False, separators=(",", ":")) in system
    session = new_session_state()
    state = dict(section="noodle", task_queue=[], active_task=None, completed_tasks=[], robot_started=False)
    data = build_decision_model_input(session, "처음 왔어요", state)
    assert data["recent_history"] == [] and data["order"] == new_session_state()["order"]
    assert not data["order"]["restrictions"] and not data["preferences"]
    # Lock only the independently reviewed exemplars against accidental rebuild
    # from old PREP targets. This is NOT a runtime text detector or a judge of
    # arbitrary model wording; changing the examples requires a fresh review.
    completed_only = next(ex for ex in EXAMPLES["task"] if ex["id"] == "S04")
    assert completed_only["input"]["robot_state"]["section"] == "meat"
    assert "section_transition" not in completed_only["input"]["robot_state"]
    assert completed_only["output"] == {"reply": "소시지 투입을 성공으로 확인한 기록은 없어요. 영상 확인은 실패했고, 재시도 후 해당 작업을 종료한 기록은 있어요."}
    no_pending = next(ex for ex in EXAMPLES["mixed"] if ex["id"] == "S09")
    assert no_pending["input"]["pending"] is None and no_pending["input"]["next_prompt"] is None
    assert no_pending["output"] == {"reply": "넓은면을 조금으로 바꿨어요. 파스타는 이탈리아어로 반죽을 뜻해요."}


@pytest.mark.parametrize("has_transition", [False, True])
def test_task_completion_does_not_fabricate_stage_transition_in_response_input(has_transition):
    session = new_session_state()
    task = {"class": "sausage", "repeat_count": 3}
    session["order"]["toppings"] = {"소시지": "high"}
    session["action_history"] = [
        {"type": "vlm_result", "class": "sausage", "verdict": "fail", "policy_result": "policy_success"},
        {"type": "task_completed", "section": "meat", "task": task}]
    session["history"] = [{"role": "assistant", "content": "다음 단계로 이동했어요."}]
    state = dict(section="extra" if has_transition else "meat", task_queue=[], active_task=None,
                 completed_tasks=[task], robot_started=True)
    if has_transition:
        state["section_transition"] = dict(type="section_transition", section="meat", next_section="extra", outcome="completed")
    before = copy.deepcopy(session)
    data = build_response_model_input(None if has_transition else "어디까지 진행됐나요?", session,
        dict(status="pass", reason="section_transition" if has_transition else "state_update_only", execute=False),
        dict(order_changes={}, restriction_added=[], restriction_removed=[], preference_added=[], preference_removed=[]),
        [], None, None, state, "task")
    assert (data["event"] is not None) is has_transition
    assert data["input_source"] == ("event" if has_transition else "user")
    assert data["robot_state"]["section"] == state["section"]
    assert data["recent_action_history"] == before["action_history"]
    assert data["recent_history"] == before["history"]
    assert not data["execution_authorized"]
    assert session == before


@pytest.mark.parametrize("outcome", ["completed", "skipped"])
def test_real_section_event_preserves_outcome_without_order_or_execution(outcome):
    session = new_session_state()
    before = copy.deepcopy(session)
    state = dict(section="extra", task_queue=[], active_task=None, completed_tasks=[], robot_started=False,
                 section_transition=dict(type="section_transition", section="meat", next_section="extra", outcome=outcome))
    response = Mock(return_value="추가 재료는 어떻게 고르시겠어요?")
    empty = dict(proposal=dict(sauce=None, noodle_type=None, noodle_portion=None, toppings={}), reason_tags=[])
    result = generate_section_event_response(session, state, Mock(return_value=empty), response, "기존 안내")
    data = build_response_model_input(*response.call_args.args)
    assert data["event"]["outcome"] == outcome
    assert result["session"] == before
    assert not result["policy"]["execute"]
    assert data["starting_now"] == [] and data["user_text"] is None


@pytest.mark.parametrize("case", ["unasked", "execution", "clarification", "recommendation", "restriction", "ordinary", "direct", "recommendation_commit"])
def test_recurring_question_failures_do_not_change_valid_authority_contracts(case):
    """Oracle meaning labels test code/state; actual LLM understanding is separate."""
    session = new_session_state()
    session["order"]["toppings"] = {"소시지": "high"}
    session["history"] = [{"role": "user", "content": "소시지는 넉넉히 골라 둬요."},
                          {"role": "assistant", "content": "소시지를 지금 담을까요?"}]
    sparse = dict(route="task", clarify=True)
    text = "네"
    if case in ("execution", "clarification"):
        session["pending"] = dict(type="execution", source="current_update", section="meat", targets=["소시지"])
        if case == "clarification":
            session["pending"]["question_type"] = "clarification"
            session["history"] += [{"role": "user", "content": "그중 한 가지는 바꿀래요."},
                                   {"role": "assistant", "content": "어떤 재료를 어떻게 바꿀까요?"}]
        else:
            sparse = dict(route="task", confirmation="accept")
    elif case in ("recommendation", "recommendation_commit"):
        session["pending"] = dict(type="recommendation", candidate=dict(sauce=None, noodle_type=None,
            noodle_portion=None, toppings={"치즈": "low"}), reason_tags=["추가 재료 후보"])
        session["history"][-1]["content"] = "추가 재료로 치즈를 조금 골라 둘까요?"
        sparse = dict(route="task", confirmation="accept")
        if case == "recommendation_commit":
            sparse["commit"] = True
            text = "그 후보도 고르고 현재 고른 소시지는 지금 담아 주세요."
    elif case == "restriction":
        session["order"]["restrictions"] = [{"target": "게살", "reason": "allergy"}]
        session["pending"] = dict(type="restriction_removal", candidate={"target": "게살", "reason": "allergy"})
        session["history"][-1]["content"] = "게살 알레르기 제한을 정말 해제할까요?"
        sparse = dict(route="task", confirmation="accept")
    elif case == "ordinary":
        sparse, text = dict(route="task", order={"toppings": {"소시지": "low"}}), "소시지는 조금으로 바꿔 주세요."
    elif case == "direct":
        sparse, text = dict(route="task", commit=True), "고른 소시지를 지금 바로 담아 주세요."
    before = copy.deepcopy(session)
    response = Mock(return_value="계약 테스트 응답")
    result = build_graph(lambda *_: normalize_decision(copy.deepcopy(sparse)), Mock(), response).invoke(
        new_turn_state(session, text, dict(section="meat", task_queue=[], active_task=None, completed_tasks=[], robot_started=False)))
    assert result["policy"]["execute"] is (case in {"execution", "direct", "recommendation_commit"})
    expected_order = copy.deepcopy(before["order"])
    if case == "ordinary":
        expected_order["toppings"]["소시지"] = "low"
    if case in {"recommendation", "recommendation_commit"}:
        expected_order["toppings"]["치즈"] = "low"
    if case == "restriction":
        expected_order["restrictions"] = []
    assert result["session"]["order"] == expected_order
    assert result["session"]["preferences"] == before["preferences"]
    assert not any(event["type"] in {"task_completed", "vlm_result", "section_transition"}
                   for event in result["session"]["action_history"][len(before["action_history"]):])
    if case in {"unasked", "clarification"}:
        assert response.call_args.args[6]["type"] == "clarification"
    if case in {"execution", "direct", "recommendation_commit"}:
        response.assert_not_called()  # Actual-start baseline path, no repeat question.
    assert result["session"]["history"][:-2] == before["history"]
    assert session == before
