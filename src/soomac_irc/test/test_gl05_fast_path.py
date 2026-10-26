"""Oracle decisions -> actual Graph/node control, NOT proof of language parsing.

Publishers are mocks, quantities/plans come from actual node methods. Robot/STT
busy policy and history are kept. Phrasing variants also need separate GPU tests.
"""
import copy
import json
from types import MethodType
from unittest.mock import Mock

import pytest

from soomac_irc.agent_contract import new_session_state, normalize_decision
from soomac_irc.llm_langgraph import build_graph
from test_section_question_events import IsolatedNode, make_node


PHRASES = ("바로 시작해", "바로 진행해", "바로 담아줘", "지금 해줘", "즉시 진행해", "응, 그걸로 해")


def node_for(session, sparse):
    node = make_node("meat")
    node.graph_state = copy.deepcopy(session)
    node.graph = build_graph(lambda *_: normalize_decision(copy.deepcopy(sparse)),
                             node.call_recommendation, node.call_response)
    node._start_current_section = MethodType(IsolatedNode._start_current_section, node)
    node._log_ignored_turn = Mock()
    return node


@pytest.mark.parametrize("phrase", PHRASES)
@pytest.mark.parametrize("context", ["new_order", "execution_pending", "recommendation_and_commit"])
def test_fast_path_actual_plan_once_with_latest_high_and_busy_duplicate_guard(phrase, context):
    session = new_session_state()
    if context == "new_order":
        sparse = {"route": "task", "order": {"toppings": {"소시지": "high"}}, "commit": True}
    elif context == "execution_pending":
        session["order"]["toppings"] = {"소시지": "high"}
        session["pending"] = dict(type="execution", source="current_update", section="meat", targets=["소시지"])
        session["history"] = [{"role": "assistant", "content": "고른 소시지를 많이 담을까요?"}]
        sparse = {"route": "task", "confirmation": "accept"}
    else:
        session["pending"] = dict(type="recommendation", reason_tags=["육류를 넉넉하게"],
            candidate=dict(sauce=None, noodle_type=None, noodle_portion=None, toppings={"소시지": "high"}))
        session["history"] = [{"role": "assistant", "content": "소시지를 많이 추천해요. 선택할까요?"}]
        sparse = {"route": "task", "confirmation": "accept", "commit": True}
    before = copy.deepcopy(session)
    node = node_for(session, sparse)
    node._process_turn(("소시지를 많이 골라서 " if context == "new_order" else "") + phrase)
    assert node.graph_state["order"]["toppings"] == {"소시지": "high"}
    assert node.graph_state["pending"] is None
    assert node.graph_state["history"][:-2] == before["history"]
    assert json.loads(node.plan_pub.publish.call_args.args[0].data) == {"class": "sausage", "repeat_count": 3}
    node.plan_pub.publish.assert_called_once()
    assert node.active_task == {"class": "소시지", "repeat_count": 3}
    node._process_turn(phrase)
    node.plan_pub.publish.assert_called_once()  # Busy duplicate cannot publish again.
    node._set_stt_enabled.assert_called_with(False)
    assert session == before


@pytest.mark.parametrize("kind", ["ordinary_order", "recommendation_accept_only"])
def test_selection_without_immediate_execution_never_publishes_plan(kind):
    session = new_session_state()
    if kind == "ordinary_order":
        sparse = {"route": "task", "order": {"toppings": {"소시지": "low"}}}
    else:
        session["pending"] = dict(type="recommendation", reason_tags=[],
            candidate=dict(sauce=None, noodle_type=None, noodle_portion=None, toppings={"소시지": "low"}))
        sparse = {"route": "task", "confirmation": "accept"}
    node = node_for(session, sparse)
    node._process_turn("소시지는 조금으로 선택할게")
    node.plan_pub.publish.assert_not_called()
    assert node.graph_state["order"]["toppings"] == {"소시지": "low"}
    assert node.graph_state["pending"]["type"] == "execution"


@pytest.mark.parametrize("phrase", PHRASES)
def test_explicit_execution_does_not_override_allergy_or_create_empty_target(phrase):
    session = new_session_state()
    session["order"]["restrictions"] = [{"target": "게살", "reason": "allergy"}]
    node = node_for(session, {"route": "task", "order": {"toppings": {"게살": "high"}}, "commit": True})
    node._process_turn("게살을 많이 " + phrase)
    node.plan_pub.publish.assert_not_called()
    assert node.graph_state["order"]["restrictions"] == session["order"]["restrictions"]
    assert "게살" not in node.graph_state["order"]["toppings"]
    empty = node_for(new_session_state(), {"route": "task", "commit": True})
    empty._process_turn(phrase)
    empty.plan_pub.publish.assert_not_called()
    assert empty.section == "meat"  # Bare commit isn't optional-stage skip.


@pytest.mark.parametrize("selected", [False, True])
def test_explicit_skip_advances_one_optional_stage_without_ingredient_plan(selected):
    session = new_session_state()
    sparse = {"route": "task", "skip_current": True}
    if selected:
        session["order"]["toppings"] = {"소시지": "high"}
        sparse["order"] = {"toppings": {"소시지": "none"}}
    node = node_for(session, sparse)
    node._process_turn("이번 육류는 빼고 다음 단계로 넘어가")
    node.plan_pub.publish.assert_not_called()
    assert node.section == "extra"
    assert node.graph_state["order"]["toppings"] == {}
    assert node.graph_state["action_history"][-1]["next_section"] == "extra"
