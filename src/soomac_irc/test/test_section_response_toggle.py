"""Fixed/model section announcements: real node methods, CPU doubles, no ROS/GPU."""
import ast
import copy
from pathlib import Path
from unittest.mock import Mock

import pytest

from soomac_irc.response_contract import RESPONSE_FAILURE_REPLY
from test_section_question_events import make_node


def transition_log(node):
    return next(call.args[1] for call in reversed(node.runtime_log.log_event.call_args_list)
                if call.args[0] == "section_transition_response")


def test_production_toggle_defaults_to_existing_fixed_announcements():
    path = Path(__file__).parents[1] / "soomac_irc/llm_langgraph_node.py"
    assignment = next(n for n in ast.parse(path.read_text()).body
                      if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "hard_cording_response"
                              for t in n.targets))
    assert ast.literal_eval(assignment.value) is True


@pytest.mark.parametrize("section,next_section", [("noodle", "veggie"), ("veggie", "meat"), ("meat", "extra")])
@pytest.mark.parametrize("outcome", ["completed", "skipped"])
def test_model_toggle_passes_real_event_and_preserves_state(section, next_section, outcome):
    node = make_node(section, hard_cording_response=False)
    node.graph_state["history"] = [
        {"role": "user", "content": "매운 것은 잘 못 먹어요."},
        {"role": "assistant", "content": "맵지 않은 구성으로 도와드릴게요."},
    ]
    node.graph_state["order"]["restrictions"] = [{"target": "매운맛", "reason": "cannot_eat"}]
    before = copy.deepcopy(node.graph_state)
    reply = node._advance_after_section(section, outcome=outcome)
    assert reply == node.call_response.return_value
    assert node.section == next_section
    node.call_response.assert_called_once()
    node.call_recommendation.assert_called_once()
    assert node.call_recommendation.call_args.args[3] is None
    args = node.call_response.call_args.args
    assert args[0] is None  # event, not a fabricated customer turn
    assert args[1]["history"] == before["history"]
    assert args[2]["execute"] is False
    assert args[7]["section_transition"] == dict(
        type="section_transition", section=section, next_section=next_section, outcome=outcome)
    assert node.graph_state["order"] == before["order"]
    assert node.graph_state["preferences"] == before["preferences"]
    assert node.graph_state["history"] == before["history"] + [{"role": "assistant", "content": reply}]
    node.plan_pub.publish.assert_not_called()
    node._set_stt_enabled.assert_called_once_with(True)
    node.rail_move_ahead_pub.publish.assert_called_once()
    assert transition_log(node)["hard_cording_response"] is False


def test_model_toggle_preselected_order_asks_execution_without_recommendation():
    node = make_node(hard_cording_response=False)
    node.graph_state["order"]["toppings"] = {"양파": "low", "치즈": "low"}
    # Only the supported, current-stage selection is an execution question target.
    before_order = copy.deepcopy(node.graph_state["order"])
    node._advance_after_section("meat")
    assert node.graph_state["order"] == before_order
    assert node.graph_state["pending"]["type"] == "execution"
    assert node.graph_state["pending"]["items"] == {"치즈": "low"}
    node.call_recommendation.assert_not_called()
    args = node.call_response.call_args.args
    assert args[6]["type"] == "execution" and args[6]["items"] == {"치즈": "low"}
    assert args[2]["execute"] is False
    node.plan_pub.publish.assert_not_called()


def test_model_toggle_valid_proposal_creates_candidate_not_confirmed_order():
    node = make_node(hard_cording_response=False)
    node.graph_state["order"]["restrictions"] = [{"target": "페퍼론치노", "reason": "cannot_eat"}]
    node.call_recommendation.return_value = {
        "proposal": {"sauce": None, "noodle_type": None, "noodle_portion": None,
                     "toppings": {"치즈": "normal", "페퍼론치노": "high"}},
        "reason_tags": ["매운맛 제한을 고려"],
    }
    node._advance_after_section("meat")
    assert node.graph_state["pending"]["candidate"]["toppings"] == {"치즈": "normal"}
    assert node.graph_state["order"]["toppings"] == {}
    args = node.call_response.call_args.args
    assert args[5]["proposal"]["toppings"] == {"치즈": "normal"}
    assert args[6]["type"] == "recommendation"
    node.plan_pub.publish.assert_not_called()


@pytest.mark.parametrize("failed_stage", ["recommendation", "response", "empty_response"])
def test_model_failure_uses_safe_fallback_and_does_not_log_stale_response(failed_stage):
    node = make_node(hard_cording_response=False)
    node.graph_state["order"]["restrictions"] = [{"target": "게살", "reason": "allergy"}]
    node.call_response.trace_events = [{"model_input": {"stale": True}, "raw": "과거 응답"}]
    if failed_stage == "recommendation":
        node.call_recommendation.side_effect = RuntimeError("recommendation failed")
    elif failed_stage == "response":
        node.call_response.side_effect = RuntimeError("response failed")
    else:
        node.call_response.return_value = ""
    before = copy.deepcopy(node.graph_state)
    reply = node._advance_after_section("meat")
    assert reply == RESPONSE_FAILURE_REPLY
    assert node.graph_state["order"] == before["order"]
    assert node.graph_state["history"][-1]["content"] == RESPONSE_FAILURE_REPLY
    data = transition_log(node)
    assert data["fallback_error"]
    assert data["response_trace"] is None
    assert data["response_model_input"] is None
    assert data["response_raw"] is None
    node.plan_pub.publish.assert_not_called()
    node._set_stt_enabled.assert_called_once_with(True)


def test_model_trace_logging_preserves_prior_traces_and_captures_only_new_response():
    node = make_node(hard_cording_response=False)
    old = {"model_input": {"prior": True}, "raw": "이전 응답"}
    new = {"model_input": {"input_source": "event", "user_text": None}, "raw": '{"reply":"단계 안내"}'}
    node.call_response.trace_events = [copy.deepcopy(old)]
    def respond(*_args):
        node.call_response.trace_events.append(copy.deepcopy(new))
        return "모델 단계 안내"
    node.call_response.side_effect = respond
    node._advance_after_section("meat")
    assert node.call_response.trace_events == [old, new]
    assert transition_log(node)["response_trace"] == new
    assert transition_log(node)["response_model_input"] == new["model_input"]
    node._reset_model_trace_buffers.assert_not_called()


def test_model_toggle_duplicate_transition_never_calls_models_or_appends_history_twice():
    node = make_node(hard_cording_response=False)
    node._advance_after_section("meat")
    before = copy.deepcopy(node.graph_state)
    assert node._advance_after_section("meat") is None
    assert node.graph_state == before
    node.call_response.assert_called_once()
    node.call_recommendation.assert_called_once()
    node.rail_move_ahead_pub.publish.assert_called_once()


@pytest.mark.parametrize("section,next_section", [("extra", "lid"), ("lid", "sauce")])
def test_model_toggle_does_not_change_automatic_lid_or_sauce_paths(section, next_section):
    node = make_node(section, hard_cording_response=False)
    node.graph_state["order"]["sauce"] = "크림"
    node._start_current_section.return_value = True
    node._advance_after_section(section)
    assert node.section == next_section
    node._start_current_section.assert_called_once()
    node.call_response.assert_not_called()
    node.call_recommendation.assert_not_called()
    node._set_stt_enabled.assert_not_called()


def test_model_toggle_process_skip_retains_transition_reply_once():
    node = make_node(hard_cording_response=False)
    session = copy.deepcopy(node.graph_state)
    session["history"] = [{"role": "user", "content": "육류는 넘어갈게요."},
                          {"role": "assistant", "content": ""}]
    node.graph = Mock()
    node.graph.invoke.return_value = {"session": session,
        "policy": {"execute": True, "reason": "section_skip_allowed"}, "reply": ""}
    node._process_turn("육류는 넘어갈게요.")
    assert node.section == "extra"
    node.call_response.assert_called_once()
    node._publish_reply.assert_called_once_with(node.call_response.return_value)
    assert node.graph_state["history"] == [{"role": "user", "content": "육류는 넘어갈게요."},
                                          {"role": "assistant", "content": node.call_response.return_value}]
    node.plan_pub.publish.assert_not_called()
