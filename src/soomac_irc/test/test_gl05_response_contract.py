"""GL05 ownership/input regressions, not actual model semantic acceptance."""
import copy
from unittest.mock import Mock

import pytest

from soomac_irc.agent_contract import new_session_state, new_turn_state, normalize_decision
from soomac_irc.llm_langgraph import build_graph
from soomac_irc.response_contract import (
    RESPONSE_FAILURE_REPLY, build_response_model_input, validate_recommendation_result,
    validate_response_output,
)
from jsonschema import ValidationError


def robot(section="meat"):
    return dict(section=section, active_task=None, task_queue=[], completed_tasks=[], robot_started=False)


def proposal(**toppings):
    return dict(sauce=None, noodle_type=None, noodle_portion=None, toppings=toppings)


@pytest.mark.parametrize("kind", ["execution_reject", "restriction_reject", "propose", "accept"])
def test_four_restored_paths_are_model_owned_with_original_history(kind):
    session = new_session_state()
    session["history"] = [
        {"role": "user", "content": "게살은 알레르기가 있어"},
        {"role": "assistant", "content": "게살을 제외하고 고를게요."},
        {"role": "user", "content": "소시지는 조금이면 돼"},
        {"role": "assistant", "content": "소시지를 조금 선택했어요."},
    ]
    session["order"]["restrictions"] = [{"target": "게살", "reason": "allergy"}]
    decision = {"route": "task", "confirmation": "reject"}
    expected = "execution_confirmation_rejected"
    next_type = None
    if kind == "execution_reject":
        session["order"]["toppings"] = {"소시지": "low"}
        session["pending"] = dict(type="execution", source="current_update", section="meat", targets=["소시지"])
    elif kind == "restriction_reject":
        session["pending"] = dict(type="restriction_removal", candidate={"target": "게살", "reason": "allergy"})
        expected = "restriction_removal_rejected"
    elif kind == "propose":
        decision = {"route": "task", "recommendation": {"action": "request"}}
        expected, next_type = "recommendation_proposed", "recommendation"
    else:
        decision = {"route": "task", "confirmation": "accept"}
        session["pending"] = dict(type="recommendation", candidate=proposal(소시지="low"), reason_tags=["게살 제외"])
        expected, next_type = "recommendation_accepted", "execution"
    before = copy.deepcopy(session)
    response = Mock(return_value="상태와 문맥을 설명한 모델 응답")
    graph = build_graph(
        lambda *_: normalize_decision(decision),
        lambda *_: dict(proposal=proposal(소시지="low"), reason_tags=["게살 알레르기를 고려한 대안"]),
        response,
    )
    result = graph.invoke(new_turn_state(session, "이번에는 이렇게 해줘", robot()))
    assert result["policy"]["reason"] == expected
    assert result["policy"]["execute"] is False
    response.assert_called_once()
    args = response.call_args.args
    assert args[1]["history"] == before["history"]
    assert args[1]["order"]["restrictions"] == before["order"]["restrictions"]
    assert (args[6] or {}).get("type") == next_type
    assert result["reply"] == response.return_value
    assert result["session"]["history"][:-2] == before["history"]
    assert session == before
    if kind == "propose":
        assert result["session"]["order"]["toppings"] == {}
    elif kind in ("execution_reject", "accept"):
        assert result["session"]["order"]["toppings"] == {"소시지": "low"}


@pytest.mark.parametrize("execute", ["false", 0, 1, None, [], {}])
def test_response_authorization_does_not_coerce_nonboolean(execute):
    with pytest.raises(ValueError, match="JSON boolean"):
        build_response_model_input("현재 상태는?", new_session_state(), {"execute": execute},
                                   {}, [], None, None, robot(), "task")


def test_user_event_inputs_are_distinct_copies_and_keep_history():
    session = new_session_state()
    session["history"] = [{"role": "user", "content": "매운 음식은 좋아해"},
                          {"role": "assistant", "content": "기호에 맞게 골라볼게요."}]
    before = copy.deepcopy(session)
    state = robot("extra")
    state["section_transition"] = dict(type="section_transition", section="meat", next_section="extra", outcome="completed")
    event = build_response_model_input(None, session, {"execute": False}, {}, [], None, None, state, "task")
    assert event["input_source"] == "event" and event["user_text"] is None
    assert event["event"] == state["section_transition"]
    assert event["recent_history"] == before["history"]
    event["recent_history"].clear()
    assert session == before
    user = build_response_model_input("매콤한 쪽으로", session, {"execute": False}, {}, [], None, None, state, "task")
    assert user["input_source"] == "user" and user["event"] is None
    with pytest.raises(ValueError, match="actual section_transition"):
        build_response_model_input(None, session, {"execute": False}, {}, [], None, None, robot(), "task")
    with pytest.raises(ValueError, match="cannot authorize"):
        build_response_model_input(None, session, {"execute": True}, {}, [], None, None, state, "task")


@pytest.mark.parametrize("parsed", [{"reply": True}, {"reply": 7}, {"reply": "ok", "commit": True}, [], {}])
def test_response_exact_json_shape(parsed):
    with pytest.raises(ValidationError):
        validate_response_output(parsed)


def test_partial_candidate_drops_dangerous_field_not_model_reason_evidence():
    session = new_session_state()
    session["order"]["restrictions"] = [{"target": "게살", "reason": "allergy"}]
    raw = dict(proposal=proposal(게살="high", 소시지="low"), reason_tags=["게살과 소시지 조합"])
    validated = validate_recommendation_result(session, raw, robot())
    assert validated["proposal"]["toppings"] == {"소시지": "low"}
    assert validated["raw_proposal"] == raw["proposal"]
    assert validated["reason_tags"] == raw["reason_tags"]
    assert validated["reason_validation"]["semantic_status"] == "unverified"
    assert {entry["field"] for entry in validated["rejected_fields"]} == {"toppings.게살"}


@pytest.mark.parametrize("raw", [None, {}, {"proposal": proposal(소시지="low"), "reason_tags": [True]}])
def test_malformed_recommendation_cannot_create_candidate(raw):
    result = validate_recommendation_result(new_session_state(), raw, robot())
    assert result["accepted_fields"] == []
    assert result["proposal"] == proposal()
    assert result["reason_validation"]["schema_valid"] is False


def test_response_failure_keeps_new_strong_restriction_instead_of_rolling_it_back():
    session = new_session_state()
    graph = build_graph(
        lambda *_: normalize_decision({"route": "task", "restrictions": [
            {"target": "게살", "reason": "allergy", "action": "add"}]}),
        Mock(), Mock(side_effect=RuntimeError("generation failed")),
    )
    result = graph.invoke(new_turn_state(session, "게살 알레르기야", robot()))
    assert result["session"]["order"]["restrictions"] == [{"target": "게살", "reason": "allergy"}]
    assert result["reply"] == RESPONSE_FAILURE_REPLY
    assert result["policy"]["execute"] is False
    assert result["policy"]["response_error"]["type"] == "RuntimeError"


def test_full_valid_recommendation_keeps_model_reasons_not_current_section_tags():
    session = new_session_state()
    session["preferences"] = [{"value": "매콤하게"}]
    reasons = ["앞서 매콤한 맛을 원하셨음", "소스와 추가 재료를 함께 제안"]
    raw = dict(proposal=proposal(페퍼론치노="low"), reason_tags=reasons)
    raw["proposal"]["sauce"] = "토마토"
    response = Mock(return_value="원하신 매콤한 맛을 고려한 구성입니다. 이렇게 선택할까요?")
    result = build_graph(
        lambda *_: normalize_decision({"route": "task", "recommendation": {"action": "request"}}),
        lambda *_: copy.deepcopy(raw), response,
    ).invoke(new_turn_state(session, "먹을 구성 추천해줘", robot("noodle")))
    assert result["recommendation_result"]["reason_tags"] == reasons
    assert result["session"]["pending"]["reason_tags"] == reasons
    assert response.call_args.args[5]["proposal"] == raw["proposal"]
    assert result["session"]["order"]["sauce"] is None
    assert result["policy"]["execute"] is False


@pytest.mark.parametrize("kind", ["execution", "recommendation", "restriction_removal"])
def test_failed_unasked_question_cannot_be_accepted_until_redelivered(kind):
    session = new_session_state()
    session["order"]["toppings"] = {"소시지": "low"}
    if kind == "execution":
        session["pending"] = dict(type=kind, source="current_update", section="meat", targets=["소시지"])
    elif kind == "recommendation":
        session["pending"] = dict(type=kind, candidate=proposal(소시지="high"), reason_tags=["더 든든하게"])
    else:
        session["order"]["restrictions"] = [{"target": "게살", "reason": "allergy"}]
        session["pending"] = dict(type=kind, candidate={"target": "게살", "reason": "allergy"})
    failed = build_graph(lambda *_: normalize_decision({"route": "task"}), Mock(),
                         Mock(side_effect=RuntimeError("fail"))).invoke(new_turn_state(session, "상태 알려줘", robot()))
    assert failed["session"]["pending"]["question_type"] == "undelivered"
    accept = Mock(side_effect=lambda *_: normalize_decision({"route": "task", "confirmation": "accept"}))
    blocked = build_graph(accept, Mock(), Mock(return_value="어떤 확인을 원하시나요?"))
    result = blocked.invoke(new_turn_state(failed["session"], "응", robot()))
    assert accept.call_count == 2  # same Decision gets one structural repair, not Python NL inference
    assert result["policy"]["execute"] is False
    assert result["session"]["order"] == failed["session"]["order"]
    assert result["session"]["pending"] == failed["session"]["pending"]
    redeliver = Mock(return_value="현재 대기 중인 확인 질문을 다시 설명한 응답")
    recovered = build_graph(lambda *_: normalize_decision({"route": "task"}), Mock(), redeliver)
    recovered = recovered.invoke(new_turn_state(result["session"], "다시 설명해줘", robot()))
    assert redeliver.call_args.args[6]["type"] == kind
    assert "question_type" not in recovered["session"]["pending"]


def test_event_proposal_is_validated_before_response_and_never_becomes_order():
    from soomac_irc.llm_langgraph import generate_section_event_response
    session = new_session_state()
    session["preferences"] = [{"value": "매콤하게"}]
    session["order"]["restrictions"] = [{"target": "치즈", "reason": "allergy"}]
    session["history"] = [{"role": "user", "content": "매운 맛을 좋아해"},
                          {"role": "assistant", "content": "취향에 맞게 골라볼게요."}]
    before = copy.deepcopy(session)
    state = robot("extra")
    state["section_transition"] = dict(type="section_transition", section="meat", next_section="extra", outcome="completed")
    raw = dict(proposal=proposal(치즈="high", 페퍼론치노="low"), reason_tags=["매운맛 선호"])
    raw["proposal"]["sauce"] = "크림"  # safe future sauce is now allowed in stage events
    recommendation, response = Mock(return_value=raw), Mock(return_value="매운맛을 좋아하셔서 페퍼론치노를 추천해요. 이렇게 선택할까요?")
    result = generate_section_event_response(session, state, recommendation, response, "기존 안내")
    assert recommendation.call_args.args[3] is None
    assert response.call_args.args[0] is None
    expected = dict(proposal(페퍼론치노="low"), sauce="크림")
    assert response.call_args.args[5]["proposal"] == expected
    assert {item["reason"] for item in response.call_args.args[5]["rejected_fields"]} == {"restriction_conflict"}
    assert result["session"]["pending"]["candidate"] == expected
    assert result["session"]["order"] == before["order"]
    assert result["session"]["history"] == before["history"]
    assert result["policy"]["execute"] is False
    assert session == before


def test_exact_duplicate_section_event_does_not_repeat_models_history_or_rail():
    from test_section_question_events import make_node
    node = make_node()
    node._advance_after_section("meat", outcome="skipped")
    before = copy.deepcopy(node.graph_state)
    assert node._advance_after_section("meat", outcome="skipped") is None
    assert node.graph_state == before
    node.call_response.assert_not_called()
    node.call_recommendation.assert_not_called()
    node.rail_move_ahead_pub.publish.assert_called_once()
    node.plan_pub.publish.assert_not_called()


@pytest.mark.parametrize("kind", ["execution", "recommendation", "restriction_removal"])
def test_actual_reply_publisher_failure_marks_pending_unasked(kind):
    from test_section_question_events import IsolatedNode, make_node
    node = make_node()
    if kind == "execution":
        pending = dict(type=kind, source="current_update", section="meat", targets=["소시지"])
    elif kind == "recommendation":
        pending = dict(type=kind, candidate=proposal(소시지="low"), reason_tags=[])
    else:
        pending = dict(type=kind, candidate={"target": "게살", "reason": "allergy"})
    node.graph_state["pending"] = copy.deepcopy(pending)
    node.reply_pub = Mock()
    node.reply_pub.publish.side_effect = RuntimeError("transport failed")
    with pytest.raises(RuntimeError, match="transport failed"):
        IsolatedNode._publish_reply(node, "대기 중인 질문을 담은 모델 발화")
    assert node.graph_state["pending"] == {**pending, "question_type": "undelivered"}
    node.plan_pub.publish.assert_not_called()


def test_post_graph_reply_publication_failure_preserves_order_and_blocks_confirmation():
    from test_section_question_events import make_node
    node = make_node()
    session = new_session_state()
    session["order"]["toppings"] = {"소시지": "high"}
    session["pending"] = dict(type="execution", source="current_update", section="meat", targets=["소시지"])
    session["history"] = [{"role": "user", "content": "소시지는 많이"},
                          {"role": "assistant", "content": "많이 선택했어요. 지금 담을까요?"}]
    node.graph = Mock()
    node.graph.invoke.return_value = dict(session=copy.deepcopy(session),
        policy={"status": "pass", "reason": "state_update_only", "execute": False}, reply=session["history"][-1]["content"])
    node._publish_reply.side_effect = RuntimeError("reply topic failed")
    with pytest.raises(RuntimeError, match="reply topic failed"):
        node._process_turn("소시지는 많이")
    assert node.graph_state["order"] == session["order"]
    assert node.graph_state["pending"]["question_type"] == "undelivered"
    assert node.runtime_log.log_turn.call_args.args[0]["status"] == "post_graph_error"
    node._start_current_section.assert_not_called()


def test_stt_error_after_successful_publication_does_not_mark_question_undelivered():
    from test_section_question_events import make_node
    node = make_node()
    session = new_session_state()
    session["order"]["toppings"] = {"소시지": "low"}
    session["pending"] = dict(type="execution", source="current_update", section="meat", targets=["소시지"])
    session["history"] = [{"role": "user", "content": "소시지는 조금"},
                          {"role": "assistant", "content": "소시지를 조금 담을까요?"}]
    node.graph = Mock()
    node.graph.invoke.return_value = dict(session=copy.deepcopy(session),
        policy={"status": "pass", "reason": "state_update_only", "execute": False}, reply=session["history"][-1]["content"])
    node._set_stt_enabled.side_effect = lambda enabled: (_ for _ in ()).throw(RuntimeError("STT failed")) if enabled else None
    with pytest.raises(RuntimeError, match="STT failed"):
        node._process_turn("소시지는 조금")
    node._publish_reply.assert_called_once()
    assert node.graph_state["pending"] == session["pending"]
    assert node.graph_state["history"] == session["history"]


@pytest.mark.parametrize("current", [False, True])
def test_future_candidate_acceptance_links_existing_current_order_to_execution_question(current):
    session = new_session_state()
    if current:
        session["order"]["toppings"] = {"소시지": "high"}
    session["pending"] = dict(type="recommendation", candidate=proposal(치즈="low"), reason_tags=["추가 재료 후보"])
    response = Mock(return_value="현재 사실과 다음 확인 질문을 설명한 응답")
    result = build_graph(lambda *_: normalize_decision({"route": "task", "confirmation": "accept"}), Mock(), response
                         ).invoke(new_turn_state(session, "그 추가 재료로 골라둘게", robot()))
    assert result["policy"]["execute"] is False
    assert result["session"]["order"]["toppings"] == ({"소시지": "high", "치즈": "low"} if current else {"치즈": "low"})
    if current:
        assert result["session"]["pending"] == dict(type="execution", source="current_update", section="meat", targets=["소시지"])
        assert response.call_args.args[6]["targets"] == ["소시지"]
        assert response.call_args.args[1]["order"]["toppings"]["소시지"] == "high"
    else:
        assert result["session"]["pending"] is None
        assert response.call_args.args[6] is None


def test_revalidated_pending_event_candidate_is_not_labeled_original_raw_proposal():
    from soomac_irc.llm_langgraph import generate_section_event_response
    session = new_session_state()
    session["order"]["restrictions"] = [{"target": "게살", "reason": "allergy"}]
    raw = dict(proposal=proposal(게살="high", 치즈="low"), reason_tags=["안전한 잔여 후보 설명 필요"])
    # At meat stage the original unsafe crab is removed before the pending is stored.
    validated = validate_recommendation_result(session, raw, robot())
    session["pending"] = dict(type="recommendation", candidate=validated["proposal"], reason_tags=validated["reason_tags"])
    state = robot("extra")
    state["section_transition"] = dict(type="section_transition", section="meat", outcome="completed", next_section="extra")
    rec, response = Mock(), Mock(return_value="이전에 제안한 치즈를 조금 선택할까요?")
    result = generate_section_event_response(session, state, rec, response, "기존 안내")
    rec.assert_not_called()
    assert "raw_proposal" not in result["recommendation_result"]
    assert result["recommendation_result"]["accepted_candidate"] == proposal(치즈="low")
    assert result["recommendation_result"]["candidate_source"] == "revalidated_pending"
    assert result["recommendation_result"]["reason_validation"]["semantic_status"] == "unverified"
    assert result["session"]["order"] == session["order"]


def test_event_uses_same_five_applied_fields_as_user_response():
    from soomac_irc.llm_langgraph import generate_section_event_response
    from soomac_irc.llm_policy import build_applied_changes
    session = new_session_state()
    state = robot("extra")
    state["section_transition"] = dict(type="section_transition", section="meat", next_section="extra", outcome="completed")
    response = Mock(return_value="추가 재료를 골라볼까요?")
    result = generate_section_event_response(
        session, state, Mock(return_value=dict(proposal=proposal(), reason_tags=[])), response, "기존 안내")
    assert response.call_args.args[3] == build_applied_changes(session, result["session"])
    assert set(response.call_args.args[3]) == {
        "order_changes", "restriction_added", "restriction_removed", "preference_added", "preference_removed"}
    assert not any(response.call_args.args[3].values())


@pytest.mark.parametrize("failure", ["exception", "schema", "unsafe", "past_section"])
def test_event_recommendation_failure_is_deterministic_not_a_second_model_guess(failure):
    from soomac_irc.llm_langgraph import generate_section_event_response
    session = new_session_state()
    session["order"]["restrictions"] = [{"target": "치즈", "reason": "allergy"}]
    before = copy.deepcopy(session)
    state = robot("extra")
    state["section_transition"] = dict(type="section_transition", section="meat", next_section="extra", outcome="completed")
    raw = dict(proposal=proposal(), reason_tags=[])
    if failure == "schema":
        raw = {"proposal": proposal(), "reason_tags": [True]}
    elif failure == "unsafe":
        raw["proposal"] = proposal(치즈="high")
    elif failure == "past_section":
        raw["proposal"]["noodle_type"] = "얇은면"
    rec = Mock(side_effect=RuntimeError("failed")) if failure == "exception" else Mock(return_value=raw)
    response = Mock(return_value="검증되지 않은 재료를 추천하는 모델 응답")
    result = generate_section_event_response(session, state, rec, response, "치즈를 담을까요?")
    response.assert_not_called()
    assert result["reply"] == RESPONSE_FAILURE_REPLY
    assert result["policy"]["execute"] is False
    assert result["model_error"]["stage"] == "recommendation"
    assert result["session"] == before
    assert session == before


def test_event_valid_empty_recommendation_is_optional_decline_not_failure():
    from soomac_irc.llm_langgraph import generate_section_event_response
    session = new_session_state()
    state = robot("extra")
    state["section_transition"] = dict(type="section_transition", section="meat", next_section="extra", outcome="completed")
    response = Mock(return_value="추가 재료는 어떻게 하실까요?")
    result = generate_section_event_response(
        session, state, Mock(return_value=dict(proposal=proposal(), reason_tags=[])), response, "기존 안내")
    response.assert_called_once()
    assert result["model_error"] is None
    assert result["session"] == session


@pytest.mark.parametrize("recovery", ["confirmed_order", "fresh_candidate", "failure"])
def test_event_invalid_prior_candidate_is_checked_after_recovery(recovery):
    from soomac_irc.llm_langgraph import generate_section_event_response
    session = new_session_state()
    session["order"]["restrictions"] = [{"target": "치즈", "reason": "allergy"}]
    session["pending"] = dict(type="recommendation", candidate=proposal(치즈="high"), reason_tags=["이전 후보"])
    if recovery == "confirmed_order":
        session["order"]["toppings"] = {"페퍼론치노": "low"}
    state = robot("extra")
    state["section_transition"] = dict(type="section_transition", section="meat", next_section="extra", outcome="completed")
    rec = Mock(return_value=dict(proposal=proposal(페퍼론치노="low"), reason_tags=["매운맛 선호"]))
    if recovery == "failure":
        rec.side_effect = RuntimeError("failed")
    response = Mock(return_value="유효한 현재 질문")
    before = copy.deepcopy(session)
    result = generate_section_event_response(session, state, rec, response, "기존 안내")
    assert result["session"]["order"] == before["order"]
    assert result["session"]["history"] == before["history"]
    assert result["policy"]["execute"] is False
    if recovery == "failure":
        response.assert_not_called()
        assert result["reply"] == RESPONSE_FAILURE_REPLY
    else:
        response.assert_called_once()
        assert result["model_error"] is None
        assert result["session"]["pending"]["type"] == (
            "execution" if recovery == "confirmed_order" else "recommendation")
        if recovery == "confirmed_order":
            rec.assert_not_called()
            # Keep rejected old-candidate evidence in trace, not as live speech context.
            assert result["recommendation_result"]["rejected_fields"]
            assert response.call_args.args[5] is None
