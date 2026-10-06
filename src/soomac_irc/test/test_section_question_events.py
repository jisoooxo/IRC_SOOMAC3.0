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
    build_preselected_confirmation_reply, build_preselected_section_confirmation,
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

    def test_skipped_and_completed_send_distinct_facts(self):
        for outcome in ("skipped", "completed"):
            with self.subTest(outcome=outcome):
                node = make_node()
                self.assertEqual(node._advance_after_section("meat", outcome=outcome),
                                 "모델이 만든 단계 전환 안내")
                args = node.call_response.call_args.args
                self.assertEqual(args[7]["section_transition"]["outcome"], outcome)
                self.assertEqual(args[7]["section_transition"]["section"], "meat")
                self.assertEqual(args[6]["type"], "section_prompt")
                self.assertEqual(args[6]["section"], "extra")
                self.assertIsNone(node.graph_state["pending"])
                self.assertEqual(node.graph_state["action_history"][-1]["outcome"], outcome)
                self.assertFalse(args[2]["execute"])
                self.assertEqual(node.completed_tasks, [])
                node.rail_move_ahead_pub.publish.assert_called_once()

    def test_failed_response_only_announces_next_step(self):
        node = make_node()
        node.call_response.side_effect = RuntimeError("모의 모델 실패")
        reply = node._advance_after_section("meat", outcome="skipped")
        self.assertNotIn("다 담", reply)
        self.assertIn("추가 재료", reply)
        node.get_logger().error.assert_called_once()
        node._set_stt_enabled.assert_called_with(True)

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
        self.assertEqual(node.call_response.call_args.args[6]["type"], "execution")

    def test_extra_skip_starts_lid_without_claiming_extra_completion(self):
        # 2026-10-05: extra 다음은 lid(뚜껑)이며 사용자 입력 없이 자동 시작한다.
        node = make_node("extra")
        node._start_current_section.return_value = True
        node._advance_after_section("extra", outcome="skipped")
        self.assertEqual(node.section, "lid")
        node._start_current_section.assert_called_once()
        self.assertIsNone(node.call_response.call_args.args[6])
        self.assertEqual(node.call_response.call_args.args[7]["section_transition"]["outcome"], "skipped")
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
        self.assertEqual(node.call_response.call_args.args[7]["section_transition"]["outcome"], "skipped")
        node._publish_reply.assert_called_once_with("모델이 만든 단계 전환 안내")
        self.assertEqual(node.graph_state["history"][-1]["content"], "모델이 만든 단계 전환 안내")
        self.assertEqual(node.runtime_log.log_turn.call_args.args[0]["status"], "section_advanced")


if __name__ == "__main__":
    unittest.main()
