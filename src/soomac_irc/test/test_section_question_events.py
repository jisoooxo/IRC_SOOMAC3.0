"""ROS/GPU 없이 운영 노드의 실제 메서드 본문을 실행한다.

AST로 메서드만 로드해 하드웨어 import와 생성자 부작용을 피한다.
publisher와 모델은 mock이며, ROS 전송/실제 생성 문장 검증은 아니다.
"""
import ast
import copy
import json
from pathlib import Path
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from soomac_irc.domain import SECTION_ORDER
from soomac_irc.agent_contract import new_session_state, new_turn_state
from soomac_irc.llm_langgraph import (
    build_execution_started_reply, build_preselected_confirmation_reply,
    build_preselected_section_confirmation,
)
from soomac_irc.llm_policy import build_applied_changes


def node_methods():
    source = Path(__file__).parents[1] / "soomac_irc" / "llm_langgraph_node.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LLMLangGraphNode")
    names = {"_section_prompt", "_advance_after_section", "_publish_next_task", "_process_turn"}
    selected = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    node_class = ast.ClassDef(name="IsolatedNode", bases=[], keywords=[], body=selected, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[node_class], type_ignores=[]))
    ns = dict(globals(), String=SimpleNamespace, ENABLE_VLM=False)
    topic_names = next(n for n in tree.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "TOPIC_CLASS_NAMES" for t in n.targets))
    ns["TOPIC_CLASS_NAMES"] = ast.literal_eval(topic_names.value)
    exec(compile(module, str(source), "exec"), ns)
    return ns["IsolatedNode"]


IsolatedNode = node_methods()


def make_node(section="meat"):
    node = IsolatedNode()
    node.section = section
    node.graph_state = new_session_state()
    node.active_task = None
    node.task_queue = []
    node.completed_tasks = []
    node.robot_started = False
    node.vlm_confirmed = False
    node.conversation_started = True
    node.order_finished = False
    node.call_response = Mock(return_value="모델이 만든 단계 전환 안내")
    node.rail_move_ahead_pub = Mock()
    node.plan_pub = Mock()
    node.runtime_log = Mock()
    node._set_stt_enabled = Mock()
    node._start_current_section = Mock(return_value=False)
    node._publish_reply = Mock()
    node._reset_model_trace_buffers = Mock()
    node._collect_model_traces = Mock(return_value={})
    node.get_logger = Mock(return_value=Mock())
    node._build_robot_state = lambda: {
        "section": node.section, "active_task": copy.deepcopy(node.active_task),
        "completed_tasks": copy.deepcopy(node.completed_tasks),
        "task_queue": copy.deepcopy(node.task_queue),
        "robot_started": node.robot_started,
    }
    return node


class TestSectionQuestionEvents(unittest.TestCase):
    def test_start_consumes_execution_pending(self):
        node = make_node()
        node.graph_state["pending"] = {
            "type": "execution", "source": "current_update",
            "section": "meat", "targets": ["소시지"],
        }
        node.task_queue = [{"class": "소시지", "repeat_count": 2}]
        self.assertTrue(node._publish_next_task())
        self.assertIsNone(node.graph_state["pending"])
        self.assertTrue(node.robot_started)
        self.assertEqual(json.loads(node.plan_pub.publish.call_args.args[0].data),
                         {"class": "sausage", "repeat_count": 2})

    def test_no_task_does_not_consume_pending(self):
        node = make_node()
        node.graph_state["pending"] = {
            "type": "execution", "source": "current_update",
            "section": "meat", "targets": ["소시지"],
        }
        before = copy.deepcopy(node.graph_state)
        self.assertFalse(node._publish_next_task())
        self.assertEqual(node.graph_state, before)

    def test_skipped_and_completed_use_deterministic_next_prompt(self):
        for outcome in ("skipped", "completed"):
            with self.subTest(outcome=outcome):
                node = make_node()
                self.assertEqual(node._advance_after_section("meat", outcome=outcome),
                                 "다음은 추가 재료 차례입니다. 치즈와 페퍼론치노 중 원하는 재료와 양을 말씀해 주세요. 원하지 않으면 다음 단계라고 말씀하셔도 돼요.")
                self.assertIsNone(node.graph_state["pending"])
                self.assertEqual(node.graph_state["action_history"][-1]["outcome"], outcome)
                self.assertEqual(node.completed_tasks, [])
                node.call_response.assert_not_called()
                node.rail_move_ahead_pub.publish.assert_called_once()

    def test_transition_does_not_call_failing_response_model(self):
        node = make_node()
        node.call_response.side_effect = RuntimeError("모의 모델 실패")
        reply = node._advance_after_section("meat", outcome="skipped")
        self.assertNotIn("다 담", reply)
        self.assertIn("추가 재료", reply)
        node.call_response.assert_not_called()
        node._set_stt_enabled.assert_called_with(True)

    def test_lid_completion_starts_sauce_with_deterministic_reply(self):
        node = make_node("lid")
        node.graph_state["order"]["sauce"] = "토마토"
        node._start_current_section.return_value = True

        reply = node._advance_after_section("lid", outcome="completed")

        self.assertEqual(node.section, "sauce")
        self.assertEqual(reply, "마지막으로 고르신 소스를 올릴게요.")
        self.assertIsNone(node.graph_state["pending"])
        node._start_current_section.assert_called_once()
        node.call_response.assert_not_called()
        node._set_stt_enabled.assert_not_called()

    def test_empty_response_uses_same_safe_fallback(self):
        node = make_node()
        node.call_response.return_value = ""
        reply = node._advance_after_section("meat", outcome="skipped")
        self.assertNotIn("다 담", reply)
        self.assertIn("추가 재료", reply)

    def test_preselected_next_stage_keeps_safety_confirmation(self):
        node = make_node()
        node.graph_state["order"]["toppings"]["치즈"] = "low"
        node._advance_after_section("meat", outcome="skipped")
        self.assertEqual(node.graph_state["pending"]["items"], {"치즈": "low"})
        self.assertEqual(node.graph_state["pending"]["type"], "execution")
        self.assertEqual(node.graph_state["history"][-1]["content"],
                         "미리 고른 치즈를 적게 담을까요?")
        node.call_response.assert_not_called()

    def test_section_transition_response_event_records_deterministic_reply(self):
        node = make_node()
        reply = node._advance_after_section("meat", outcome="completed")

        matching = [
            call.args[1]
            for call in node.runtime_log.log_event.call_args_list
            if call.args[0] == "section_transition_response"
        ]
        self.assertIn("추가 재료", reply)
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["transition"]["next_section"], "extra")
        self.assertEqual(matching[0]["robot_state"]["section"], "extra")
        self.assertIsNone(matching[0]["response_trace"])
        self.assertIsNone(matching[0]["response_model_input"])
        self.assertIsNone(matching[0]["response_raw"])
        self.assertEqual(matching[0]["reply"], reply)
        node.call_response.assert_not_called()

    def test_extra_skip_starts_lid_with_deterministic_reply(self):
        node = make_node("extra")
        node._start_current_section.return_value = True

        reply = node._advance_after_section("extra", outcome="skipped")

        self.assertEqual(node.section, "lid")
        self.assertEqual(reply, "이제 뚜껑을 닫을게요.")
        node._start_current_section.assert_called_once()
        node.call_response.assert_not_called()
        node._set_stt_enabled.assert_not_called()

    def test_invalid_outcome_does_not_move_stage(self):
        node = make_node()
        with self.assertRaises(ValueError):
            node._advance_after_section("meat", outcome="unknown")
        self.assertEqual(node.section, "meat")
        node.rail_move_ahead_pub.publish.assert_not_called()

    def test_empty_section_runtime_path_passes_skipped(self):
        node = make_node()
        session = new_session_state()
        session["history"] = [{"role": "user", "content": "진행"},
                              {"role": "assistant", "content": "이전 응답"}]
        node.graph = Mock()
        node.graph.invoke.return_value = {"session": session, "policy": {"execute": True},
                                          "reply": "이전 응답"}
        node._process_turn("바로 진행해")
        self.assertEqual(node.section, "extra")
        node.call_response.assert_not_called()
        self.assertIn("추가 재료", node._publish_reply.call_args.args[0])
        self.assertIn("추가 재료", node.graph_state["history"][-1]["content"])
        self.assertEqual(len(node.graph_state["history"]), 2)
        self.assertEqual(node.runtime_log.log_turn.call_args.args[0]["status"], "section_advanced")

    def test_started_task_replaces_model_reply_with_structured_message(self):
        node = make_node("veggie")
        session = new_session_state()
        session["order"]["toppings"] = {"양파": "low", "버섯": "high"}
        session["history"] = [
            {"role": "user", "content": "버섯을 바꿔서 바로 담아"},
            {"role": "assistant", "content": "실행 확인이 남아 있어요. 담을까요?"},
        ]
        node.graph = Mock()
        node.graph.invoke.return_value = {
            "session": session,
            "policy": {"execute": True, "reason": "execution_allowed"},
            "reply": "실행 확인이 남아 있어요. 담을까요?",
        }
        node._start_current_section.return_value = True

        node._process_turn("버섯을 바꿔서 바로 담아")

        expected = "양파 적게, 버섯 많이 담기를 시작해요."
        node._publish_reply.assert_called_once_with(expected)
        self.assertEqual(node.graph_state["history"][-1]["content"], expected)
        self.assertEqual(node.runtime_log.log_turn.call_args.args[0]["status"], "execution_started")

    def test_empty_optional_commit_repeats_current_section_prompt(self):
        node = make_node("meat")
        session = new_session_state()
        session["history"] = [
            {"role": "user", "content": "어 진행해"},
            {"role": "assistant", "content": "지금 담기를 진행해요."},
        ]
        node.graph = Mock()
        node.graph.invoke.return_value = {
            "session": session,
            "policy": {"execute": False, "reason": "empty_section_commit"},
            "reply": "지금 담기를 진행해요.",
        }

        node._process_turn("어 진행해")

        reply = node._publish_reply.call_args.args[0]
        self.assertIn("육류", reply)
        self.assertEqual(node.section, "meat")
        self.assertEqual(node.graph_state["history"][-1]["content"], reply)

    def test_unresolved_pending_blocks_node_execution_as_last_defense(self):
        node = make_node("veggie")
        session = new_session_state()
        session["pending"] = {
            "type": "recommendation",
            "candidate": {
                "sauce": None, "noodle_type": None, "noodle_portion": None,
                "toppings": {"버섯": "normal"},
            },
            "reason_tags": [],
        }
        session["history"] = [
            {"role": "user", "content": "바로 해줘"},
            {"role": "assistant", "content": "지금 시작해요."},
        ]
        node.graph = Mock()
        node.graph.invoke.return_value = {
            "session": session,
            "policy": {"execute": True, "reason": "execution_allowed"},
            "reply": "지금 시작해요.",
        }

        node._process_turn("바로 해줘")

        node._start_current_section.assert_not_called()
        self.assertIn("추천", node._publish_reply.call_args.args[0])
        self.assertEqual(node.runtime_log.log_turn.call_args.args[0]["status"], "pending_blocked")


if __name__ == "__main__":
    unittest.main()
