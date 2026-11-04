"""CPU controls for optional model execution speech; no model-quality claim."""
import copy
import ast
from functools import partial
from pathlib import Path
from types import MethodType

import pytest

from soomac_irc.agent_contract import new_session_state, new_turn_state
from soomac_irc.llm_langgraph import build_graph
from test_graph_role_refactor import compiled
from test_natural_multiturn_contract import robot_state


def run(mode, *, reply="양파는 적게, 버섯은 보통 양으로 담기 시작할게요.", commit=True, empty=False):
    session = new_session_state()
    session["order"].update(sauce="크림", noodle_type="얇은면", noodle_portion="normal")
    session["order"]["toppings"] = {} if empty else {"양파": "low", "버섯": "normal"}
    builder = build_graph if mode is None else partial(build_graph, execution_reply_mode=mode)
    graph, calls = compiled(builder, {"route": "task", "commit": commit}, response=reply)
    turn = new_turn_state(session, "바로 담아.", robot_state("veggie"))
    original = copy.deepcopy(turn)
    result = graph.invoke(turn)
    assert turn == original
    return result, calls


def test_default_still_fixed_and_never_calls_response():
    result, calls = run(None)
    assert result["reply"] == "양파는 적게, 버섯은 보통 양으로 담을게요."
    assert result["policy"]["execute"] is True
    assert [c["agent"] for c in calls] == ["decision"]


def test_model_receives_authorized_items_and_cannot_change_order_or_execution():
    fixed, _ = run("fixed")
    model, calls = run("model")
    payload = next(c for c in calls if c["agent"] == "response")["args"]
    from soomac_irc.response_contract import build_response_model_input
    value = build_response_model_input(*payload)
    assert value["execution_authorized"] is True
    assert value["starting_now"] == [{"item": "양파", "amount": "low"}, {"item": "버섯", "amount": "normal"}]
    assert model["policy"] == fixed["policy"]
    for key in ("order", "preferences", "pending", "action_history"):
        assert model["session"][key] == fixed["session"][key]
    assert model["session"]["history"][-1]["content"] == model["reply"]
    assert [c["agent"] for c in calls] == ["decision", "response"]


@pytest.mark.parametrize("reply", ["", RuntimeError("response unavailable")])
def test_failed_execution_speech_returns_fixed_without_revoking_start(reply):
    result, _ = run("model", reply=reply)
    assert result["policy"]["execute"] is True
    assert "response_error" in result["policy"]
    assert result["reply"] == "양파는 적게, 버섯은 보통 양으로 담을게요."


def test_nonexecution_behavior_is_unchanged():
    fixed, fcalls = run("fixed", commit=False)
    model, mcalls = run("model", commit=False)
    assert fixed == model
    assert fcalls == mcalls


def test_empty_section_keeps_existing_transition_behavior():
    fixed, fcalls = run("fixed", empty=True)
    model, mcalls = run("model", empty=True)
    assert fixed == model
    assert fcalls == mcalls


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="execution_reply_mode"):
        build_graph(None, None, None, execution_reply_mode="typo")


def execution_node(mode, reply):
    import test_section_question_events as node_tests
    node = node_tests.make_node("meat")
    node.execution_reply_mode = mode
    node._start_current_section = MethodType(node_tests.IsolatedNode._start_current_section, node)
    node.graph, _ = compiled(partial(build_graph, execution_reply_mode=mode),
        {"route": "task", "order": {"toppings": {"소시지": "high"}}, "commit": True},
        response=reply)
    return node


@pytest.mark.parametrize("mode,expected", [
    ("model", "소시지를 많이 담을게요."),
    ("fixed", "소시지를 많이 담을게요."),
])
def test_actual_node_publishes_selected_speech_and_same_plan(mode, expected):
    import json
    node = execution_node(mode, "소시지를 많이 담을게요.")
    node._process_turn("소시지만 많이 바로 담아줘.")
    node._publish_reply.assert_called_once_with(expected)
    assert node.graph_state["history"][-1]["content"] == expected
    assert json.loads(node.plan_pub.publish.call_args.args[0].data) == {
        "class": "sausage", "repeat_count": 3}
    assert node.runtime_log.log_turn.call_args.args[0]["status"] == "execution_started"


def test_node_model_failure_uses_fixed_start_announcement():
    node = execution_node("model", RuntimeError("HTTP unavailable"))
    node._process_turn("소시지만 많이 바로 담아줘.")
    node._publish_reply.assert_called_once_with("소시지를 많이 담을게요.")
    node.plan_pub.publish.assert_called_once()


def test_node_does_not_speak_start_when_plan_publication_fails():
    node = execution_node("model", "소시지를 많이 담을게요.")
    node.plan_pub.publish.side_effect = RuntimeError("plan unavailable")
    with pytest.raises(RuntimeError, match="plan unavailable"):
        node._process_turn("소시지만 많이 바로 담아줘.")
    node._publish_reply.assert_not_called()
    assert node.active_task is None


def test_node_passes_selected_mode_and_uses_evaluated_backend():
    # Parse only: this test never imports/starts ROS or loads a model.
    source = Path(__file__).parents[1] / "soomac_irc/llm_langgraph_node.py"
    tree = ast.parse(source.read_text())
    values = {t.id: n.value for n in tree.body if isinstance(n, ast.Assign)
              for t in n.targets if isinstance(t, ast.Name)}
    assert ast.literal_eval(values["EXECUTION_REPLY_MODE"]) == "fixed"
    assert ast.literal_eval(values["LLM_BACKEND"]) == "vllm"
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "build_graph"]
    assert len(calls) == 1
    mode = next(k.value for k in calls[0].keywords if k.arg == "execution_reply_mode")
    assert ast.unparse(mode) == "self.execution_reply_mode"


def test_fixed_node_ignores_stale_high_speech_after_low_correction():
    # Observed P failure: after acknowledging low, the next execution reply said high.
    import json
    import test_section_question_events as node_tests
    node = node_tests.make_node("veggie")
    node.execution_reply_mode = "fixed"
    node.graph_state["order"].update(sauce="크림", noodle_type="얇은면", noodle_portion="normal")
    node.graph_state["order"]["toppings"] = {"양파": "low", "버섯": "low"}
    node.graph_state["history"] = [
        {"role": "user", "content": "양파는 적게, 버섯은 많이 골라줘."},
        {"role": "assistant", "content": "양파는 적게, 버섯은 많이 선택했어요. 이대로 야채를 담을까요?"},
        {"role": "user", "content": "아니, 버섯도 적게 바꿔줘."},
        {"role": "assistant", "content": "버섯도 적은 양으로 줄였어요. 양파는 적은 양 그대로예요. 이대로 야채를 담을까요?"},
    ]
    node._start_current_section = MethodType(node_tests.IsolatedNode._start_current_section, node)
    node.graph, calls = compiled(partial(build_graph, execution_reply_mode="fixed"),
        {"route": "task", "commit": True}, response="양파는 적게, 버섯은 많이 담을게요.")
    node._process_turn("그럼 그걸로 담아줘.")
    node._publish_reply.assert_called_once_with("양파와 버섯을 적은 양으로 담을게요.")
    assert [c["agent"] for c in calls] == ["decision"]
    assert json.loads(node.plan_pub.publish.call_args.args[0].data) == {"class": "onion", "repeat_count": 1}
    assert node.task_queue == [{"class": "버섯", "repeat_count": 1}]
