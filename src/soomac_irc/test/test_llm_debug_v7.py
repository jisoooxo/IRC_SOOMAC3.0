import copy

import pytest

from soomac_irc.agent_v7 import build_policy_reply, validate_transaction
from soomac_irc.llm_debug_v7 import DebugSession
from soomac_irc.llm_node_v7 import sanitize_free_reply


class _CallModel:
    last_trace = None


class _GenerateReply:
    last_trace = None

    def __init__(self, reply="설명입니다."):
        self.reply = reply
        self.calls = 0

    def __call__(self, _state, _user_text):
        self.calls += 1
        return self.reply


class _ScriptedGraph:
    def __init__(self, tool_calls, *, call_model=None, raw_outputs=None):
        self.tool_calls = list(tool_calls)
        self.call_model = call_model
        self.raw_outputs = list(raw_outputs or [])
        self.calls = 0

    def invoke(self, state):
        self.calls += 1
        result = copy.deepcopy(state)
        tool_call = copy.deepcopy(self.tool_calls.pop(0))
        result["parser_status"] = "ok" if tool_call is not None else "invalid"
        result["tool_call"] = tool_call

        if self.call_model is not None:
            raw_output = self.raw_outputs.pop(0) if self.raw_outputs else "{}"
            self.call_model.last_trace = {
                "raw_output": raw_output,
                "parsed_output": copy.deepcopy(tool_call),
                "prompt_tokens": 100,
                "generated_tokens": 10,
                "latency_ms": 12.5,
            }

        result.update(validate_transaction(result))
        result.update(build_policy_reply(result))
        return result


def _tool(name, changes=None):
    return {
        "call_id": f"test_{name}",
        "name": name,
        "changes": copy.deepcopy(changes or {}),
    }


def _selection_tool():
    return _tool(
        "set_order",
        {
            "sauce": "토마토",
            "noodle_type": "얇은면",
            "noodle_portion": "normal",
        },
    )


def test_auto_flow_completes_virtual_task_and_advances_section():
    graph = _ScriptedGraph([_selection_tool(), _tool("confirm_section")])
    session = DebugSession(graph, _CallModel(), _GenerateReply())

    selection = session.turn("토마토 얇은면 보통으로")
    confirmation = session.turn("진행해")

    assert selection["transaction"]["action"] == "set_order"
    assert confirmation["transaction"]["confirm_validation"]["allowed"] is True
    assert session.section == "veggie"
    assert session.active_task is None
    assert session.completed_tasks == [{"class": "얇은면", "repeat_count": 2}]
    assert [event["event"] for event in confirmation["flow_events"]] == [
        "task_started",
        "task_completed",
        "section_advanced",
    ]
    assert "다음은 야채" in confirmation["final_reply"]


def test_manual_flow_waits_for_next_without_calling_model_again():
    graph = _ScriptedGraph([_selection_tool(), _tool("confirm_section")])
    session = DebugSession(
        graph,
        _CallModel(),
        _GenerateReply(),
        manual_flow=True,
    )

    session.turn("토마토 얇은면 보통으로")
    confirmation = session.turn("진행해")
    blocked = session.turn("양파 적게")
    completed = session.next_task()

    assert confirmation["flow_events"] == [
        {
            "event": "task_started",
            "task": {"class": "얇은면", "repeat_count": 2},
        }
    ]
    assert blocked["source"] == "blocked_during_manual_flow"
    assert graph.calls == 2
    assert completed["source"] == "manual_flow"
    assert session.section == "veggie"
    assert session.active_task is None


def test_debug_record_keeps_policy_and_only_decision_state_diff():
    graph = _ScriptedGraph([_selection_tool()])
    session = DebugSession(graph, _CallModel(), _GenerateReply())

    record = session.turn("토마토 얇은면 보통으로")
    paths = {change["path"] for change in record["state_diff"]}

    assert record["policy_reply"] == "요청한 주문 변경을 반영했어요."
    assert "order.sauce" in paths
    assert all(not path.startswith("history") for path in paths)
    assert all(not path.startswith("action_history") for path in paths)


def test_human_label_is_linked_to_previous_debug_turn(tmp_path):
    log_path = tmp_path / "debug.jsonl"
    graph = _ScriptedGraph([_tool("respond")])
    session = DebugSession(
        graph,
        _CallModel(),
        _GenerateReply(),
        log_path=log_path,
    )

    turn = session.turn("이 조합 맛이 어때?")
    label = session.mark("bad expected_tool=respond note=말투수정")

    assert label["target_record_id"] == turn["record_id"]
    assert label["expected_tool"] == "respond"
    assert label["note"] == "말투수정"
    assert len(log_path.read_text(encoding="utf-8").splitlines()) == 2


@pytest.mark.parametrize("unsafe_reply", [
    "치즈를 반영했어요.",
    "면 선택을 확정했어요.",
    "양파 담기를 시작할게요.",
    "다음 단계는 소스예요.",
])
def test_free_reply_operational_claim_variants_are_removed(unsafe_reply):
    safe, removed = sanitize_free_reply(unsafe_reply)

    assert safe == ""
    assert removed == [unsafe_reply]


def test_debug_uses_production_free_reply_guard():
    call_model = _CallModel()
    generate_reply = _GenerateReply("치즈를 반영했어요.")
    graph = _ScriptedGraph(
        [_tool("respond")],
        call_model=call_model,
        raw_outputs=['{"name":"respond","changes":{}}'],
    )
    session = DebugSession(graph, call_model, generate_reply)

    record = session.turn("치즈 추가해")

    assert generate_reply.calls == 1
    assert "free_reply_sentence_removed" in record["risk_flags"]
    assert record["tool_trace"]["raw_output"].startswith("{")
    assert "반영했어요" not in record["final_reply"]


def test_failed_turn_is_logged_with_raw_trace(tmp_path):
    class _FailingGraph:
        def invoke(self, _state):
            call_model.last_trace = {
                "raw_output": "broken",
                "parsed_output": None,
                "latency_ms": 1.0,
            }
            raise RuntimeError("graph failed")

    call_model = _CallModel()
    log_path = tmp_path / "failed.jsonl"
    session = DebugSession(
        _FailingGraph(),
        call_model,
        _GenerateReply(),
        log_path=log_path,
        runtime_metadata={"tool_adapter_name": "test_adapter"},
    )

    with pytest.raises(RuntimeError, match="graph failed"):
        session.turn("테스트")

    row = log_path.read_text(encoding="utf-8").splitlines()[0]
    assert '"record_type": "llm_debug_turn"' in row
    assert '"raw_output": "broken"' in row
    assert '"tool_adapter_name": "test_adapter"' in row
    assert '"turn_error"' in row


def test_log_write_failure_does_not_break_completed_turn(tmp_path):
    not_a_directory = tmp_path / "file"
    not_a_directory.write_text("occupied", encoding="utf-8")
    graph = _ScriptedGraph([_selection_tool()])
    session = DebugSession(
        graph,
        _CallModel(),
        _GenerateReply(),
        log_path=not_a_directory / "debug.jsonl",
    )

    record = session.turn("토마토 얇은면 보통으로")

    assert record["transaction"]["action"] == "set_order"
    assert session.order["sauce"] == "토마토"
    assert session.log_errors


def test_ambiguous_pending_confirmation_keeps_production_turn_number():
    session = DebugSession(_ScriptedGraph([]), _CallModel(), _GenerateReply())
    session.section = "veggie"
    session.awaiting_confirm = {
        "type": "refuse_section",
        "section": "veggie",
        "reason": "dislike",
        "items": ["양파", "버섯"],
        "restriction_changes": [],
        "source": "explicit_refusal",
        "needs_confirmation": True,
        "applied": False,
    }

    record = session.turn("음 생각해볼게")

    assert session.turn_index == 0
    assert record["turn_index"] == 0
    assert session.awaiting_confirm is not None


def test_manual_sauce_requires_finish_and_records_control_diff():
    session = DebugSession(
        _ScriptedGraph([]),
        _CallModel(),
        _GenerateReply(),
        manual_flow=True,
        runtime_metadata={"model_path": "test_model"},
    )
    session.section = "sauce"
    session.active_task = {"class": "토마토", "repeat_count": 1}
    session.robot_started = True

    wrong_signal = session.next_task()
    finished = session.finish_task()

    assert wrong_signal["source"] == "manual_flow_wrong_signal"
    assert wrong_signal["tool_trace"] is None
    assert wrong_signal["reply_trace"] is None
    assert session.order_finished is True
    assert finished["runtime"]["model_path"] == "test_model"
    assert {
        "path": "control.order_finished",
        "before": False,
        "after": True,
    } in finished["state_diff"]
    assert finished["tool_trace"] is None
    assert finished["reply_trace"] is None
