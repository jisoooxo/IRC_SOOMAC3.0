"""정답 Decision으로 거절·주문 보존·독립 commit의 실제 그래프 연결을 검증한다.

모델의 자연어 이해 성적이나 ROS 발행 시험이 아니다. 필드가 올바를 때 Python이
그 의미를 잃지 않는지 확인하며, 대화별 결과 session은 다음 턴에 그대로 전달한다.
"""
import copy

import pytest

from soomac_irc.agent_contract import (
    empty_order_patch,
    new_session_state,
    new_turn_state,
    normalize_decision,
)
from soomac_irc.llm_langgraph import build_graph, build_preselected_section_confirmation


def run_turn(session, sparse, *, section="meat", text="검수 발화", robot=None):
    def decision(*_args):
        decision.last_external_output = copy.deepcopy(sparse)
        return normalize_decision(copy.deepcopy(sparse))

    def recommendation(*_args):
        raise AssertionError("추천을 요청하지 않은 턴에 Recommendation 호출")

    state = copy.deepcopy(robot) if robot is not None else {
        "section": section, "active_task": None, "task_queue": [],
        "completed_tasks": [], "robot_started": False,
    }
    original_session, original_robot = copy.deepcopy(session), copy.deepcopy(state)
    graph = build_graph(decision, recommendation, lambda *_args: "ok")
    result = graph.invoke(new_turn_state(session, text, state))
    assert session == original_session
    assert state == original_robot
    assert result["session"]["history"][:len(session["history"])] == session["history"]
    return result


def selected_session(*, amount="high", pending_type="recommendation", candidate_item="치즈"):
    session = new_session_state()
    session["order"].update(sauce="크림", noodle_type="얇은면", noodle_portion=amount)
    session["order"]["toppings"] = {"소시지": amount}
    session["history"] = [
        {"role": "user", "content": "얇은면과 소시지는 지금 고른 양 그대로 둘게요."},
        {"role": "assistant", "content": "치즈를 추천드릴까요?"},
    ]
    if pending_type == "recommendation":
        candidate = empty_order_patch()
        candidate["toppings"] = {candidate_item: "low"}
        session["pending"] = {"type": pending_type, "candidate": candidate, "reason_tags": []}
    elif pending_type == "execution":
        session["pending"] = build_preselected_section_confirmation(session, "meat")
    else:
        restriction = {"target": "게살", "reason": "allergy"}
        session["order"]["restrictions"] = [restriction]
        session["pending"] = {"type": pending_type, "candidate": copy.deepcopy(restriction)}
    return session


@pytest.mark.parametrize("amount", ["low", "normal", "high"])
@pytest.mark.parametrize("candidate_item", ["치즈", "소시지"])
def test_recommendation_reject_does_not_swallow_current_order_commit(amount, candidate_item):
    session = selected_session(amount=amount, candidate_item=candidate_item)
    result = run_turn(
        session, {"route": "task", "confirmation": "reject", "commit": True},
        text="추천은 받지 않을게요. 지금 선택된 소시지 양 그대로 바로 시작해주세요.",
    )
    assert result["policy"]["reason"] == "execution_allowed"
    assert result["policy"]["execute"] is True
    assert result["session"]["pending"] is None
    assert result["session"]["order"] == session["order"]


def test_future_recommendation_reject_preserves_selected_noodle_quantity_and_commits():
    session = selected_session(amount="low")
    result = run_turn(
        session, {"route": "task", "confirmation": "reject", "commit": True},
        section="noodle", text="추천 말고 지금 고른 면 양 그대로 바로 담아줘.",
    )
    assert result["policy"]["execute"] is True
    assert result["session"]["order"] == session["order"]
    assert result["session"]["pending"] is None


@pytest.mark.parametrize("amount", ["low", "normal", "high"])
def test_preselected_reject_is_execution_deferral_not_order_cancellation(amount):
    session = selected_session(amount=amount, pending_type="execution")
    result = run_turn(session, {"route": "task", "confirmation": "reject"}, text="아직 담지 마.")
    assert result["session"]["order"] == session["order"]
    assert result["session"]["pending"] is None
    assert result["policy"]["reason"] == "execution_confirmation_rejected"
    assert result["policy"]["execute"] is False


def test_preselected_deferral_allergy_and_later_commit_use_accumulated_session():
    session = selected_session(pending_type="execution")
    deferred = run_turn(session, {
        "route": "task", "confirmation": "reject",
        "restrictions": [{"target": "게살", "reason": "allergy", "action": "add"}],
    }, text="아직 담지는 말고 게살 알레르기도 반영해줘.")
    assert deferred["session"]["order"]["toppings"] == {"소시지": "high"}
    assert deferred["session"]["order"]["restrictions"] == [{"target": "게살", "reason": "allergy"}]
    assert deferred["session"]["pending"] is None
    assert deferred["policy"]["execute"] is False
    resumed = run_turn(deferred["session"], {"route": "task", "commit": True}, text="이제 바로 담아줘.")
    assert resumed["policy"]["execute"] is True
    assert resumed["session"]["order"] == deferred["session"]["order"]


def test_preselected_reject_with_explicit_none_really_cancels_order():
    session = selected_session(pending_type="execution")
    result = run_turn(session, {
        "route": "task", "confirmation": "reject", "order": {"toppings": {"소시지": "none"}},
    }, text="아니 소시지도 주문에서 빼줘.")
    assert result["session"]["order"]["toppings"] == {}
    assert result["session"]["pending"] is None
    assert result["policy"]["execute"] is False


def test_same_execution_reject_and_commit_without_replacement_never_executes():
    session = selected_session(pending_type="execution")
    result = run_turn(session, {"route": "task", "confirmation": "reject", "commit": True})
    assert result["policy"]["execute"] is False
    assert result["session"]["order"] == session["order"]


def test_restriction_removal_reject_allows_safe_independent_commit_without_removing_limit():
    session = selected_session(pending_type="restriction_removal")
    result = run_turn(session, {"route": "task", "confirmation": "reject", "commit": True},
                      text="알레르기는 유지하고 선택한 소시지만 바로 담아줘.")
    assert result["session"]["order"] == session["order"]
    assert result["session"]["pending"] is None
    assert result["policy"]["execute"] is True


def test_deferred_strong_restriction_confirmation_still_blocks_commit():
    session = selected_session(pending_type="restriction_removal")
    second = {"target": "페퍼론치노", "reason": "cannot_eat"}
    session["order"]["restrictions"].append(second)
    session["pending"]["deferred"] = [second]
    result = run_turn(session, {"route": "task", "confirmation": "reject", "commit": True})
    assert result["policy"]["execute"] is False
    assert result["policy"]["reason"] == "restriction_removal_confirmation_required"
    assert result["session"]["order"] == session["order"]
    assert result["session"]["pending"]["candidate"] == second


@pytest.mark.parametrize("pending_type", ["recommendation", "restriction_removal"])
def test_reject_without_commit_never_executes(pending_type):
    session = selected_session(pending_type=pending_type)
    result = run_turn(session, {"route": "task", "confirmation": "reject"})
    assert result["policy"]["reason"] == pending_type + "_rejected"
    assert result["policy"]["execute"] is False
    assert result["session"]["order"] == session["order"]


@pytest.mark.parametrize("boundary", ["missing", "empty", "active", "queued", "allergy", "protected", "unsupported"])
def test_reject_plus_commit_does_not_bypass_safety(boundary):
    session = selected_session()
    sparse = {"route": "task", "confirmation": "reject", "commit": True}
    robot = {"section": "meat", "active_task": None, "task_queue": [],
             "completed_tasks": [], "robot_started": False}
    if boundary == "missing":
        robot["section"] = "noodle"
        session["order"]["noodle_type"] = None
    elif boundary == "empty":
        session["order"]["toppings"] = {}
    elif boundary == "active":
        robot["active_task"] = {"class": "소시지", "repeat_count": 1}
    elif boundary == "queued":
        robot["task_queue"] = [{"class": "소시지", "repeat_count": 1}]
    elif boundary == "allergy":
        session["order"]["restrictions"] = [{"target": "소시지", "reason": "allergy"}]
    elif boundary == "protected":
        sparse["order"] = {"noodle_portion": "normal"}
    elif boundary == "unsupported":
        sparse["order"] = {"toppings": {"햄": "normal"}}
    result = run_turn(session, sparse, robot=robot)
    assert result["policy"]["execute"] is False
    if boundary in {"active", "queued"}:
        assert result["policy"]["reason"] == "robot_busy"
    elif boundary == "missing":
        assert result["policy"]["reason"] == "missing_order"
    elif boundary == "empty":
        assert result["policy"]["reason"] == "empty_section_commit"
    elif boundary == "allergy":
        # 새 재료 요청이 아닌 기존 안전 충돌은 enforce_restrictions에서 제거한다.
        # 남은 실행 재료가 없으므로 최종 사유는 restriction_conflict가 아니라 empty다.
        assert result["policy"]["reason"] == "empty_section_commit"
        assert result["session"]["order"]["toppings"] == {}
        assert result["session"]["order"]["restrictions"] == session["order"]["restrictions"]
    elif boundary == "protected":
        assert result["policy"]["reason"] == "physical_state"
    elif boundary == "unsupported":
        assert result["policy"]["reason"] == "unsupported"
