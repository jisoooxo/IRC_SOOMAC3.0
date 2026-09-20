import copy
import types

import pytest

from soomac_irc.agent_v7 import build_policy_reply, validate_transaction
from soomac_irc.llm_node_v7 import LLMNode
from soomac_irc.order_v7 import new_order


class _Logger:
    def __init__(self):
        self.warnings = []

    def warning(self, message):
        self.warnings.append(message)


class _ReplyHarness:
    def __init__(self, generated_reply):
        self.generated_reply = generated_reply
        self.generate_calls = []
        self.logger = _Logger()

    def generate_reply(self, state, user_text):
        self.generate_calls.append({
            "state": copy.deepcopy(state),
            "user_text": user_text,
        })
        return self.generated_reply

    def get_logger(self):
        return self.logger


class _PendingHarness:
    def __init__(self, order, execution, pending):
        self.order = copy.deepcopy(order)
        self.section = execution["section"]
        self.selected = copy.deepcopy(execution["selected"])
        self.task_queue = copy.deepcopy(execution["task_queue"])
        self.active_task = copy.deepcopy(execution["active_task"])
        self.completed_tasks = copy.deepcopy(execution["completed_tasks"])
        self.skipped_sections = copy.deepcopy(execution["skipped_sections"])
        self.robot_started = execution["robot_started"]
        self.recommendation_state = {"phase": "idle", "confirmed_sections": []}
        self.awaiting_confirm = copy.deepcopy(pending)
        self.runtime_turn_index = 0
        self.history = []
        self.action_history = []
        self.stt_states = []
        self.replies = []
        self.advanced = []

    def _set_stt_enabled(self, enabled):
        self.stt_states.append(enabled)

    def _publish_reply(self, reply):
        self.replies.append(reply)

    def _section_prompt(self):
        return "야채를 골라 주세요."

    def _build_agent_state(self, user_text):
        return LLMNode._build_agent_state(self, user_text)

    def _commit_agent_result(self, result):
        return LLMNode._commit_agent_result(self, result)

    def _validate_agent_result(self, result):
        return LLMNode._validate_agent_result(self, result)

    def _advance_after_section(self, section):
        self.advanced.append(section)
        self.section = "meat"
        return "야채는 제외했어요. 다음은 육류 선택이에요."


class _Graph:
    def __init__(self, result):
        self.result = copy.deepcopy(result)

    def invoke(self, state):
        return copy.deepcopy(self.result)


def _execution(section="noodle"):
    return {
        "section": section,
        "selected": [],
        "task_queue": [],
        "active_task": None,
        "completed_tasks": [],
        "skipped_sections": {},
        "robot_started": False,
    }


def _transaction(action):
    return {
        "action": action,
        "confirm_validation": {
            "requested": False,
            "allowed": False,
            "missing": {},
            "reason": None,
        },
        "recommendation_validation": {
            "clean": {},
            "dropped": [],
            "blocked": [],
            "proposal_created": False,
            "proposal_confirmed": False,
            "needs_scope": False,
        },
        "section_skip": None,
    }


def _state_and_result(action, policy_reply=None, section="noodle"):
    order_before = new_order()
    state_before = {
        "order": order_before,
        "execution": _execution(section),
        "recommendation": {"phase": "idle", "confirmed_sections": []},
        "history": [],
        "action_history": [],
    }
    result = {
        "order": copy.deepcopy(order_before),
        "execution": _execution(section),
        "recommendation": {"phase": "idle", "confirmed_sections": []},
        "transaction": _transaction(action),
        "policy_reply": policy_reply,
    }
    return state_before, result


def _turn_node(result):
    node = object.__new__(LLMNode)
    node.order = new_order()
    node.section = "noodle"
    node.selected = []
    node.task_queue = []
    node.active_task = None
    node.completed_tasks = []
    node.skipped_sections = {}
    node.robot_started = False
    node.recommendation_state = {"phase": "idle", "confirmed_sections": []}
    node.runtime_turn_index = 0
    node.history = []
    node.action_history = []
    node.awaiting_confirm = None
    node.conversation_started = True
    node.order_finished = False
    node.graph = _Graph(result)
    node._process_pending_confirmation = lambda _user_text: False
    node._set_stt_enabled = lambda _enabled: None
    node._record_tool_turn = lambda *_args: None
    return node


def _set_order_graph_result():
    state = {
        "user_text": "토마토로 바꿔줘",
        "order": new_order(),
        "execution": _execution(),
        "recommendation": {"phase": "idle", "confirmed_sections": []},
        "history": [],
        "action_history": [],
        "parser_status": "ok",
        "tool_call": {
            "call_id": "test_set_order",
            "name": "set_order",
            "changes": {"sauce": "토마토"},
        },
    }
    result = copy.deepcopy(state)
    result.update(validate_transaction(state))
    result.update(build_policy_reply(result))
    return result


def test_policy_reply_gets_python_next_step_without_free_generation():
    state_before, result = _state_and_result(
        "set_order",
        "요청한 주문 변경을 반영했어요.",
    )
    result["order"]["sauce"] = "토마토"
    harness = _ReplyHarness("호출되면 안 됩니다.")

    reply = LLMNode._build_turn_reply(
        harness,
        "토마토로 먹고 싶어",
        state_before,
        result,
    )

    assert reply == (
        "요청한 주문 변경을 반영했어요. "
        "다음으로 선택할 항목은 면 종류, 면 양예요. 원하는 내용을 말씀해 주세요."
    )
    assert harness.generate_calls == []


def test_unverified_free_reply_state_claim_is_replaced_with_verified_next_step():
    state_before, result = _state_and_result("respond", section="meat")
    result["order"]["toppings"] = {"소시지": "high", "게살": "low"}
    harness = _ReplyHarness("육류 선택을 확정했어요. 육류 담기를 시작할게요.")

    reply = LLMNode._build_turn_reply(
        harness,
        "또 뭐 골라야 되는데",
        state_before,
        result,
    )

    assert "확정했어요" not in reply
    assert "시작할게요" not in reply
    assert reply == "현재 육류 선택은 소시지 많이, 게살 적게예요. 이대로 육류 담기를 진행할까요?"
    assert harness.logger.warnings


def test_partial_restriction_only_offers_remaining_current_section_item():
    state_before, result = _state_and_result(
        "set_order",
        "요청한 식이 조건을 반영했어요.",
        section="veggie",
    )
    result["order"]["restrictions"] = [{
        "target": "버섯",
        "reason": "dislike",
    }]
    harness = _ReplyHarness("호출되면 안 됩니다.")

    reply = LLMNode._build_turn_reply(
        harness,
        "버섯이 싫어요",
        state_before,
        result,
    )

    assert reply == (
        "요청한 식이 조건을 반영했어요. "
        "야채로는 양파가 남아 있어요. 어느 정도 양으로 넣을까요?"
    )
    assert harness.generate_calls == []


def test_completed_current_selection_asks_for_confirmation_without_advancing():
    state_before, result = _state_and_result(
        "set_order",
        "요청한 주문 변경을 반영했어요.",
        section="veggie",
    )
    result["order"]["toppings"] = {"양파": "low"}
    result["order"]["restrictions"] = [{
        "target": "버섯",
        "reason": "dislike",
    }]
    harness = _ReplyHarness("호출되면 안 됩니다.")

    reply = LLMNode._build_turn_reply(
        harness,
        "양파 적게",
        state_before,
        result,
    )

    assert reply == (
        "요청한 주문 변경을 반영했어요. "
        "현재 야채 선택은 양파 적게예요. 이대로 야채 담기를 진행할까요?"
    )
    assert harness.generate_calls == []


def test_free_reply_without_next_action_gets_verified_fallback_appended():
    state_before, result = _state_and_result("respond")
    result["order"]["sauce"] = "토마토"
    harness = _ReplyHarness("토마토 소스는 산뜻한 맛이에요.")

    reply = LLMNode._build_turn_reply(
        harness,
        "토마토 맛이 어때",
        state_before,
        result,
    )

    assert reply == (
        "토마토 소스는 산뜻한 맛이에요. "
        "다음으로 선택할 항목은 면 종류, 면 양예요. 원하는 내용을 말씀해 주세요."
    )


def test_free_explanation_is_kept_but_model_generated_workflow_question_is_removed():
    state_before, result = _state_and_result("respond")
    result["order"]["sauce"] = "토마토"
    harness = _ReplyHarness(
        "기분이 좋지 않을 때는 산뜻한 맛이 편할 수 있어요. 어떤 면을 선택해 주시겠어요?"
    )

    reply = LLMNode._build_turn_reply(
        harness,
        "기분이 안 좋은데 뭐가 좋을까",
        state_before,
        result,
    )

    assert reply == (
        "기분이 좋지 않을 때는 산뜻한 맛이 편할 수 있어요. "
        "다음으로 선택할 항목은 면 종류, 면 양예요. 원하는 내용을 말씀해 주세요."
    )
    assert harness.logger.warnings


def test_policy_question_does_not_get_duplicate_continuation():
    state_before, result = _state_and_result(
        "recommend_order",
        "현재 단계만 추천할까요, 아니면 남은 주문 전체를 추천할까요?",
    )
    result["transaction"]["recommendation_validation"]["needs_scope"] = True
    harness = _ReplyHarness("호출되면 안 됩니다.")

    reply = LLMNode._build_turn_reply(
        harness,
        "추천해줘",
        state_before,
        result,
    )

    assert reply == "현재 단계만 추천할까요, 아니면 남은 주문 전체를 추천할까요?"
    assert harness.generate_calls == []


def test_confirmed_work_waits_for_start_reply_without_free_continuation():
    state_before, result = _state_and_result(
        "confirm_section",
        "추천한 내용을 주문에 반영했어요.",
    )
    result["transaction"]["confirm_validation"].update({
        "requested": True,
        "allowed": True,
        "reason": "ready",
    })
    harness = _ReplyHarness("호출되면 안 됩니다.")

    reply = LLMNode._build_turn_reply(
        harness,
        "그대로 진행해",
        state_before,
        result,
    )

    assert reply == "추천한 내용을 주문에 반영했어요."
    assert harness.generate_calls == []


def test_rejected_dislike_skip_keeps_original_order_and_restrictions():
    order = new_order()
    order["toppings"] = {"양파": "low"}
    execution = _execution("veggie")
    execution["selected"] = ["toppings.양파"]
    pending = {
        "type": "refuse_section",
        "section": "veggie",
        "reason": "dislike",
        "items": ["양파", "버섯"],
        "restriction_changes": [
            {"target": "양파", "reason": "dislike", "enabled": True},
            {"target": "버섯", "reason": "dislike", "enabled": True},
        ],
        "source": "explicit_refusal",
        "needs_confirmation": True,
        "applied": False,
    }
    harness = _PendingHarness(order, execution, pending)

    handled = LLMNode._process_pending_confirmation(harness, "아니")

    assert handled is True
    assert harness.order == order
    assert harness.selected == ["toppings.양파"]
    assert harness.skipped_sections == {}
    assert harness.awaiting_confirm is None
    assert harness.advanced == []


def test_accepted_dislike_skip_commits_proposal_before_advancing():
    order = new_order()
    order["toppings"] = {"양파": "low"}
    execution = _execution("veggie")
    execution["selected"] = ["toppings.양파"]
    pending = {
        "type": "refuse_section",
        "section": "veggie",
        "reason": "dislike",
        "items": ["양파", "버섯"],
        "restriction_changes": [
            {"target": "양파", "reason": "dislike", "enabled": True},
            {"target": "버섯", "reason": "dislike", "enabled": True},
        ],
        "source": "explicit_refusal",
        "needs_confirmation": True,
        "applied": False,
    }
    harness = _PendingHarness(order, execution, pending)

    handled = LLMNode._process_pending_confirmation(harness, "응")

    assert handled is True
    assert harness.order["toppings"] == {}
    assert len(harness.order["restrictions"]) == 2
    assert harness.skipped_sections == {"veggie": "dislike"}
    assert harness.advanced == ["veggie"]


def test_pending_skip_uses_negative_first_common_parser():
    order = new_order()
    execution = _execution("veggie")
    pending = {
        "type": "refuse_section",
        "section": "veggie",
        "reason": "dislike",
        "items": ["양파", "버섯"],
        "restriction_changes": [],
        "source": "explicit_refusal",
        "needs_confirmation": True,
        "applied": False,
    }
    harness = _PendingHarness(order, execution, pending)

    handled = LLMNode._process_pending_confirmation(
        harness,
        "네, 그래도 넘어가지 마",
    )

    assert handled is True
    assert harness.awaiting_confirm is None
    assert harness.skipped_sections == {}
    assert harness.advanced == []


def test_pending_skip_ambiguous_answer_reasks_without_consuming_proposal():
    order = new_order()
    execution = _execution("veggie")
    pending = {
        "type": "refuse_section",
        "section": "veggie",
        "reason": "dislike",
        "items": ["양파", "버섯"],
        "restriction_changes": [],
        "source": "explicit_refusal",
        "needs_confirmation": True,
        "applied": False,
    }
    harness = _PendingHarness(order, execution, pending)

    handled = LLMNode._process_pending_confirmation(harness, "조금 생각해볼게")

    assert handled is True
    assert harness.awaiting_confirm == pending
    assert harness.replies == ["네 또는 아니요로 말씀해 주세요."]
    assert harness.advanced == []


def test_turn_commits_only_after_reply_generation_then_publishes():
    result = _set_order_graph_result()
    node = _turn_node(result)
    events = []

    node._build_turn_reply = types.MethodType(
        lambda self, *_args: events.append("reply") or "응답",
        node,
    )
    original_commit = LLMNode._commit_agent_result

    def commit(self, agent_result):
        original_commit(self, agent_result)
        events.append("commit")

    node._commit_agent_result = types.MethodType(commit, node)
    node._publish_reply = lambda _reply: events.append("publish")

    LLMNode._process_turn(node, "토마토로 바꿔줘")

    assert events == ["reply", "commit", "publish"]
    assert node.order["sauce"] == "토마토"


def test_reply_generation_failure_leaves_state_uncommitted_and_unpublished():
    result = _set_order_graph_result()
    node = _turn_node(result)
    published = []
    node._publish_reply = lambda reply: published.append(reply)

    def fail_reply(self, *_args):
        raise RuntimeError("reply failed")

    node._build_turn_reply = types.MethodType(fail_reply, node)

    with pytest.raises(RuntimeError, match="reply failed"):
        LLMNode._process_turn(node, "토마토로 바꿔줘")

    assert node.order == new_order()
    assert node.runtime_turn_index == 0
    assert node.history == []
    assert node.action_history == []
    assert published == []
