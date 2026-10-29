"""역할 분리의 CPU 계약 검사. 고정 Decision은 실제 모델의 말귀 증거가 아니다.

SOOMAC_GRAPH_BASELINE에 변경 전 모듈 경로를 주면 같은 입력/호출/payload를
전필드 차등 대조한다. 없어도 독립 oracle/결선/복합 누적 검사는 실행한다.
"""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
from types import MethodType
from unittest.mock import Mock

import pytest
from langgraph.graph import END, START, StateGraph

from soomac_irc.agent_contract import (
    DuplicateDecisionKeyError, TurnState, empty_order_patch, new_session_state,
    new_turn_state, normalize_decision,
)
from soomac_irc.decision_model import build_decision_model_input
from soomac_irc.llm_langgraph import build_graph
from soomac_irc.response_contract import (
    build_recommendation_model_input, build_response_model_input,
)
from test_natural_multiturn_contract import robot_state
import test_section_question_events as node_tests


@pytest.fixture(scope="module")
def baseline():
    path = os.environ.get("SOOMAC_GRAPH_BASELINE")
    if not path:
        return None
    spec = importlib.util.spec_from_file_location("graph_refactor_baseline", Path(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_graph


def proposal(**fields):
    patch = empty_order_patch()
    patch.update(fields)
    return {"proposal": patch, "reason_tags": ["사용자 선호와 제약을 반영한 후보"]}


def compiled(builder, sparse, rec=None, response="확인했어요."):
    calls = []

    def record(name, args, payload):
        calls.append({"agent": name, "args": copy.deepcopy(args),
                      "json": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))})

    def decision(*args):
        record("decision", args, build_decision_model_input(*args))
        if isinstance(sparse, Exception):
            raise sparse
        return normalize_decision(copy.deepcopy(sparse))

    def recommendation(*args):
        record("recommendation", args, build_recommendation_model_input(*args))
        if isinstance(rec, Exception):
            raise rec
        return copy.deepcopy(rec if rec is not None else proposal())

    def reply(*args):
        record("response", args, build_response_model_input(*args))
        if isinstance(response, Exception):
            raise response
        return response

    return builder(decision, recommendation, reply), calls


def run_pair(baseline, session, sparse, robot=None, rec=None, response="확인했어요.", text="이번 주문을 확인해줘"):
    turn = new_turn_state(session, text, robot or robot_state("meat"))
    before = copy.deepcopy(turn)
    graph, calls = compiled(build_graph, sparse, rec, response)
    result = graph.invoke(turn)
    assert turn == before, "caller의 입력을 변경하면 안 된다"
    assert set(result) == set(before), "private 문맥이 공개 반환에 노출되면 안 된다"
    assert set(result["session"]) == set(session)
    if baseline:
        old_graph, old_calls = compiled(baseline, sparse, rec, response)
        old_result = old_graph.invoke(copy.deepcopy(before))
        assert result == old_result
        assert calls == old_calls, "모델 입력/순서/횟수/직렬화가 바뀌면 안 된다"
    return result, calls, graph


def test_private_context_input_output_projection():
    class Internal(TurnState, total=False):
        _task_context: dict | None

    seen = []
    graph = StateGraph(Internal, input_schema=TurnState, output_schema=TurnState)
    def prepare(state):
        assert "_task_context" not in state  # 외부 입력의 사칭 문맥도 받지 않는다.
        return {"_task_context": {"sentinel": "private"}}

    graph.add_node("prepare", prepare)

    def finish(state):
        seen.append(state["_task_context"])
        return {"reply": "ok"}

    graph.add_node("finish", finish)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "finish")
    graph.add_edge("finish", END)
    instance = graph.compile()
    state = new_turn_state(new_session_state(), "확인", robot_state())
    result = instance.invoke(dict(state, _task_context={"sentinel": "external"}))
    assert seen == [{"sentinel": "private"}]
    assert set(result) == set(state)
    assert graph.input_schema is TurnState
    assert graph.output_schema is TurnState
    assert set(graph.input_schema.__annotations__) == set(state)
    assert set(instance.output_channels) == set(state)


@pytest.mark.parametrize("section", ["noodle", "veggie", "meat", "extra"])
@pytest.mark.parametrize("commit", [False, True])
@pytest.mark.parametrize("kind", ["none", "request", "revise", "allergy", "invalid", "unsupported", "clarify", "busy"])
def test_pre_and_post_recommendation_full_fields(baseline, section, commit, kind):
    session = new_session_state()
    session["order"].update(sauce="크림", noodle_type="얇은면", noodle_portion="low")
    session["order"]["toppings"] = {"양파": "low", "소시지": "high", "치즈": "low"}
    sparse = {"route": "task", "commit": commit}
    if kind in ("request", "revise", "allergy", "invalid", "unsupported", "clarify"):
        sparse["recommendation"] = {"action": "revise" if kind == "revise" else "request"}
    if kind == "allergy":
        sparse["restrictions"] = [{"target": "게살", "reason": "allergy", "action": "add"}]
    if kind == "invalid":
        sparse["order"] = {"toppings": {"양파": "invalid"}}
    if kind == "unsupported":
        sparse["order"] = {"toppings": {"햄": "high"}}
    if kind == "clarify":
        sparse["clarify"] = True
    robot = robot_state(section, active={"class": "소시지"} if kind == "busy" else None)
    result, calls, _ = run_pair(baseline, session, sparse, robot, proposal(toppings={"게살": "low", "치즈": "high"}))
    names = [c["agent"] for c in calls]
    assert names.count("recommendation") <= 1
    if kind in ("invalid", "unsupported", "clarify"):
        assert "recommendation" not in names
        assert result["policy"]["execute"] is False
    if kind == "allergy":
        rec_input = next(c["args"][0] for c in calls if c["agent"] == "recommendation")
        assert {"target": "게살", "reason": "allergy"} in rec_input["order"]["restrictions"]
        assert "게살" not in result["recommendation_result"]["proposal"]["toppings"]


@pytest.mark.parametrize("candidate", [proposal(toppings={"치즈": "high"}), proposal(), {"bad": True}, RuntimeError("rec failure")])
@pytest.mark.parametrize("response", ["확인했어요.", "", None, RuntimeError("reply failure")])
@pytest.mark.parametrize("commit", [False, True])
def test_recommendation_failures_and_independent_current_commit(baseline, candidate, response, commit):
    session = new_session_state()
    session["order"]["toppings"] = {"소시지": "high"}
    sparse = {"route": "task", "recommendation": {"action": "request"}, "commit": commit}
    result, calls, _ = run_pair(baseline, session, sparse, rec=candidate, response=response)
    assert [c["agent"] for c in calls].count("recommendation") == 1
    if commit:
        assert result["policy"]["execute"] is True
        assert "response" not in [c["agent"] for c in calls]


@pytest.mark.parametrize("ptype", ["execution", "recommendation", "restriction_removal"])
@pytest.mark.parametrize("confirmation", ["accept", "reject", "none"])
@pytest.mark.parametrize("replacement", [False, True])
@pytest.mark.parametrize("commit", [False, True])
def test_pending_lifecycle_and_replacement(baseline, ptype, confirmation, replacement, commit):
    session = new_session_state()
    session["order"]["toppings"] = {"소시지": "high"}
    pending = {"type": ptype}
    if ptype == "execution":
        pending.update(source="current_update", section="meat", targets=["소시지"])
    elif ptype == "recommendation":
        pending.update(candidate=proposal(toppings={"소시지": "low"})["proposal"], reason_tags=["이전 추천"])
    else:
        session["order"]["restrictions"] = [{"target": "게살", "reason": "allergy"}, {"target": "페퍼론치노", "reason": "cannot_eat"}]
        pending.update(candidate={"target": "게살", "reason": "allergy"},
                       deferred=[{"target": "페퍼론치노", "reason": "cannot_eat"}])
    session["pending"] = pending
    session["history"] = [{"role": "assistant", "content": "앞서 고른 내용을 진행할까요?"}]
    sparse = {"route": "task", "confirmation": confirmation, "commit": commit}
    if replacement:
        sparse["order"] = {"toppings": {"소시지": "low"}}
    result, calls, _ = run_pair(baseline, session, sparse)
    assert "recommendation" not in [c["agent"] for c in calls]
    if confirmation == "reject" and not replacement:
        assert result["policy"]["execute"] is False


@pytest.mark.parametrize("section", ["noodle", "veggie", "meat", "extra"])
@pytest.mark.parametrize("busy", [False, True])
@pytest.mark.parametrize("selected", [False, True])
def test_skip_and_empty_commit(baseline, section, busy, selected):
    session = new_session_state()
    if selected:
        session["order"]["toppings"] = {"양파": "low", "소시지": "high", "치즈": "low"}
    for action in ("skip_current", "commit"):
        run_pair(baseline, session, {"route": "task", action: True},
                 robot_state(section, active={"class": "소시지"} if busy else None))


@pytest.mark.parametrize("route", ["general", "mixed", "task"])
@pytest.mark.parametrize("sparse", [{}, {"clarify": True}, {"confirmation": "accept"}, {"commit": True, "skip_current": True}])
def test_routes_and_structural_repair(baseline, route, sparse):
    run_pair(baseline, new_session_state(), dict(sparse, route=route))


def test_duplicate_key_safety(baseline):
    result, calls, _ = run_pair(baseline, new_session_state(), DuplicateDecisionKeyError("duplicate"))
    assert result["policy"]["execute"] is False
    assert [c["agent"] for c in calls] == ["decision", "response"]


@pytest.mark.parametrize("turns", [2, 3, 6, 8, 12])
def test_cumulative_compound_episode(baseline, turns):
    # 각 턴 결과를 다음 입력으로 넘긴다. 매 턴 정답 state로 재설정하지 않는다.
    episode = [
        ("noodle", "매운 거 못 먹고 스트레스 받아. 추천해줘", {"route": "mixed", "restrictions": [{"target": "페퍼론치노", "reason": "cannot_eat", "action": "add"}], "recommendation": {"action": "request"}}, proposal(sauce="크림")),
        ("noodle", "크림으로 할게", {"route": "task", "confirmation": "accept"}, None),
        ("noodle", "얇은면 조금만", {"route": "task", "order": {"noodle_type": "얇은면", "noodle_portion": "low"}}, None),
        ("veggie", "버섯 조금 양파 많이", {"route": "task", "order": {"toppings": {"버섯": "low", "양파": "high"}}}, None),
        ("veggie", "아 그냥 둘 다 조금만", {"route": "task", "order": {"toppings": {"버섯": "low", "양파": "low"}}}, None),
        ("veggie", "고마워", {"route": "general"}, None),
        ("meat", "게살 알레르기 있어. 나머지로 추천해줘", {"route": "task", "restrictions": [{"target": "게살", "reason": "allergy", "action": "add"}], "recommendation": {"action": "request"}}, proposal(toppings={"소시지": "high"})),
        ("meat", "응 그걸로 골라줘", {"route": "task", "confirmation": "accept"}, None),
        ("meat", "지금 뭐 골랐지?", {"route": "mixed"}, None),
        ("meat", "바로 담아줘", {"route": "task", "commit": True}, None),
        ("extra", "매운 건 못 먹으니까 치즈만", {"route": "task", "order": {"toppings": {"치즈": "normal"}}}, None),
        ("extra", "응 그걸로 바로 해", {"route": "task", "confirmation": "accept"}, None),
    ][:turns]
    sessions = [new_session_state(), new_session_state()]
    for section, text, sparse, rec in episode:
        snapshots = []
        for index, builder in enumerate((build_graph, baseline or build_graph)):
            before = copy.deepcopy(sessions[index])
            graph, calls = compiled(builder, sparse, rec)
            state = new_turn_state(sessions[index], text, robot_state(section))
            result = graph.invoke(state)
            assert sessions[index] == before
            sessions[index] = result["session"]
            assert len(sessions[index]["history"]) == len(before["history"]) + 2
            snapshots.append((result, calls))
        assert snapshots[0] == snapshots[1]
    if turns >= 6:
        assert sessions[0]["order"]["toppings"]["양파"] == "low"
        assert sessions[0]["order"]["toppings"]["버섯"] == "low"
    if turns >= 8:
        assert sessions[0]["order"]["toppings"]["소시지"] == "high"
        assert "게살" not in sessions[0]["order"]["toppings"]


def test_real_recommendation_node_and_conditional_visits():
    graph, calls = compiled(build_graph, {"route": "task", "recommendation": {"action": "request"}}, proposal(toppings={"소시지": "high"}))
    assert "run_recommendation" in graph.get_graph().nodes
    state = new_turn_state(new_session_state(), "추천해줘", robot_state("meat"))
    updates = list(graph.stream(state, stream_mode="updates"))
    assert [next(iter(update)) for update in updates] == ["interpret_decision", "prepare_task", "run_recommendation", "generate_response"]
    assert [c["agent"] for c in calls] == ["decision", "recommendation", "response"]


@pytest.mark.parametrize("kind", ["strong_remove", "preselected_reject", "stale", "completed", "queued", "current_pending", "future_pending"])
def test_safety_and_future_pending_boundaries(baseline, kind):
    session = new_session_state()
    session["order"]["toppings"] = {"소시지": "high"}
    sparse = {"route": "task", "recommendation": {"action": "request"}, "commit": True}
    robot = robot_state("meat")
    if kind == "strong_remove":
        session["order"]["restrictions"] = [{"target": "게살", "reason": "allergy"}]
        sparse["restrictions"] = [{"target": "게살", "reason": "allergy", "action": "remove"}]
    elif kind == "preselected_reject":
        session["pending"] = {"type": "execution", "source": "preselected", "section": "meat", "targets": ["소시지"], "candidate": proposal(toppings={"소시지": "high"})["proposal"]}
        sparse = {"route": "task", "confirmation": "reject"}
    elif kind == "stale":
        session["pending"] = {"type": "execution", "source": "current_update", "section": "veggie", "targets": ["양파"]}
        sparse = {"route": "task", "confirmation": "accept"}
    elif kind == "completed":
        robot["completed_tasks"] = [{"class": "소시지", "repeat_count": 3}]
        sparse["order"] = {"toppings": {"소시지": "low"}}
    elif kind == "queued":
        robot["task_queue"] = [{"class": "소시지", "repeat_count": 3}]
    else:
        topping = "치즈" if kind == "future_pending" else "소시지"
        session["pending"] = {"type": "recommendation", "candidate": proposal(toppings={topping: "low"})["proposal"], "reason_tags": ["이전 후보"]}
        sparse = {"route": "task", "order": {"toppings": {"소시지": "low"}}, "commit": True}
    result, calls, _ = run_pair(baseline, session, sparse, robot, proposal(toppings={"치즈": "high"}))
    if kind == "strong_remove":
        assert result["session"]["order"]["restrictions"] == session["order"]["restrictions"]
        assert "recommendation" not in [c["agent"] for c in calls]
    if kind == "future_pending":
        assert result["policy"]["execute"] is True
        assert result["session"]["pending"] == session["pending"]


def test_graph_reuse_does_not_reuse_previous_private_context():
    responses = iter([
        {"route": "task", "order": {"toppings": {"햄": "high"}}},
        {"route": "task", "order": {"toppings": {"소시지": "low"}}},
        {"route": "general"},
    ])
    graph = build_graph(lambda *_: normalize_decision(next(responses)), lambda *_: proposal(), lambda *_: "확인했어요.")
    for _ in range(3):
        state = new_turn_state(new_session_state(), "확인해줘", robot_state("meat"))
        result = graph.invoke(dict(state, _task_context={"malicious": True}))
        assert set(result) == set(state)
        assert len(result["session"]["history"]) == 2
    assert result["session"]["order"]["toppings"] == {}


@pytest.mark.parametrize("kind", ["ordinary", "start", "skip", "empty_commit", "future_commit", "allergy", "graph_error", "reply_error", "plan_error", "pending_blocked", "log_error", "graph_and_reply_error"])
def test_actual_node_publishing_stt_and_log_fields(baseline, kind):
    baseline_path = os.environ.get("SOOMAC_GRAPH_BASELINE")
    old_method = None
    if baseline_path:
        source = Path(baseline_path).with_name("baseline_llm_langgraph_node.py")
        tree = ast.parse(source.read_text())
        old_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LLMLangGraphNode")
        body = [n for n in old_class.body if isinstance(n, ast.FunctionDef) and n.name == "_process_turn"]
        module = ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name="BaselineNode", bases=[], keywords=[], body=body, decorator_list=[])], type_ignores=[]))
        namespace = dict(vars(node_tests))
        exec(compile(module, str(source), "exec"), namespace)
        old_method = namespace["BaselineNode"]._process_turn

    snapshots = []
    for builder, method in ((build_graph, None), (baseline or build_graph, old_method)):
        session = new_session_state()
        sparse = {"route": "task", "order": {"toppings": {"소시지": "high"}}}
        if kind in ("start", "plan_error", "future_commit"):
            sparse["commit"] = True
        if kind == "future_commit":
            sparse["recommendation"] = {"action": "request"}
        if kind == "skip":
            sparse = {"route": "task", "skip_current": True}
        if kind == "empty_commit":
            sparse = {"route": "task", "commit": True}
        if kind == "allergy":
            session["order"]["restrictions"] = [{"target": "게살", "reason": "allergy"}]
            sparse = {"route": "task", "order": {"toppings": {"게살": "high"}}, "commit": True}
        if kind in ("graph_error", "graph_and_reply_error"):
            sparse = RuntimeError("graph failure")
        node = node_tests.make_node("meat")
        node.graph_state = copy.deepcopy(session)
        node._start_current_section = MethodType(node_tests.IsolatedNode._start_current_section, node)
        if method is not None:
            node._process_turn = MethodType(method, node)
        node.graph, model_calls = compiled(builder, sparse, proposal(toppings={"치즈": "low"}))
        node.runtime_log.next_turn_id.return_value = 17
        if kind in ("reply_error", "graph_and_reply_error"):
            node._publish_reply.side_effect = RuntimeError("reply failure")
        if kind == "plan_error":
            node.plan_pub.publish.side_effect = RuntimeError("plan failure")
        if kind == "log_error":
            node.runtime_log.log_turn.side_effect = RuntimeError("log failure")
        if kind == "pending_blocked":
            session["pending"] = {"type": "execution", "section": "meat", "targets": ["소시지"]}
            node.graph = Mock()
            node.graph.invoke.return_value = {"session": session, "policy": {"execute": True}, "reply": "시작"}
        caught = None
        try:
            node._process_turn("  현재 선택대로 진행해줘  ")
        except RuntimeError as error:
            caught = (type(error).__name__, str(error))
        logs = [copy.deepcopy(call.args[0]) for call in node.runtime_log.log_turn.call_args_list]
        for entry in logs:
            entry["latency_ms"].pop("total")  # 경과 시간만 비결정 값이다.
        plans = [json.loads(call.args[0].data) for call in node.plan_pub.publish.call_args_list]
        snapshots.append({"session": node.graph_state, "robot": node._build_robot_state(),
                          "logs": logs, "plans": plans, "stt": node._set_stt_enabled.call_args_list,
                          "published": node._publish_reply.call_args_list, "models": model_calls,
                          "event_rec": node.call_recommendation.call_args_list,
                          "event_response": node.call_response.call_args_list,
                          "rail": node.rail_move_ahead_pub.publish.call_args_list, "error": caught})
        if kind == "graph_and_reply_error":
            # 기존 한계 보존: graph 오류 fallback 발행도 실패하면 log 전 예외가 전파된다.
            assert logs == []
            assert caught == ("RuntimeError", "reply failure")
            assert node.graph_state == session
        else:
            assert logs
        if kind in ("start", "future_commit"):
            assert plans == [{"class": "sausage", "repeat_count": 3}]
            assert logs[-1]["status"] == "execution_started"
        if kind == "graph_error":
            assert logs[-1]["status"] == "graph_error"
            assert logs[-1]["decision_final"] is None
        if kind in ("reply_error", "plan_error"):
            assert logs[-1]["status"] == "post_graph_error"
            assert caught is not None
        assert session == new_session_state() or kind in ("allergy", "pending_blocked")
    assert snapshots[0] == snapshots[1]
