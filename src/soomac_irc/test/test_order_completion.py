"""실제 메서드를 AST로 로드해 ROS·GPU·오디오 없이 완료 복귀 계약을 검사한다."""

import ast
import copy
import itertools
import json
import queue
import threading
import traceback
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from soomac_irc.order import SECTION_LABELS, SECTION_ORDER, new_order

SOURCE = Path(__file__).resolve().parents[1] / 'soomac_irc'


class Message:
    def __init__(self, data=None):
        self.data = data


class FakeNode:
    def __init__(self, *args):
        self.publishers = {}
        self.subscriptions = {}
        self.logger = Mock()

    def create_publisher(self, message_type, topic, qos):
        publisher = SimpleNamespace(messages=[])
        publisher.publish = lambda message: publisher.messages.append(message)
        self.publishers[topic] = publisher
        return publisher

    def create_subscription(self, message_type, topic, callback, qos):
        self.subscriptions[topic] = callback

    def get_logger(self):
        return self.logger


def load_source(filename, names, namespace):
    tree = ast.parse((SOURCE / filename).read_text())
    body = []
    for item in tree.body:
        if isinstance(item, (ast.FunctionDef, ast.ClassDef)) and item.name in names:
            item.decorator_list = []
            if isinstance(item, ast.ClassDef) and item.name != 'UiNode':
                item.body = [method for method in item.body if not isinstance(method, ast.FunctionDef) or method.name != '__init__']
            body.append(item)
        elif isinstance(item, ast.Assign) and any(isinstance(target, ast.Name) and target.id in names for target in item.targets):
            body.append(item)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE / filename), 'exec'), namespace)
    return namespace


def base_namespace():
    return dict(Node=FakeNode, String=Message, Bool=Message, Int16=Message, Image=Message, CompressedImage=Message, json=json, threading=threading, traceback=traceback, copy=copy, queue=queue)


def make_ui():
    namespace = base_namespace()
    namespace.update(socketio=Mock(), socket_emit_lock=threading.RLock(), ros_node_lock=threading.Lock(), ros_node=None, latest_mic_state='idle', latest_stt_enabled=False, latest_agent_status={}, latest_vlm_snapshot=None, ui_session_active=False, cached_dialogue=deque())
    load_source('ui_node.py', {'UiNode', 'PUBLISH_QUEUE_SIZE', 'SUBSCRIPTION_QUEUE_SIZE', 'UI_START_MESSAGE', 'UI_START_TOPIC', 'UI_RESET_TOPIC', 'AGENT_STATUS_TOPIC', 'VLM_UI_IMAGE_TOPIC', 'RESET_ACTIONS', 'is_ui_session_active', 'activate_ui_session', 'reset_ui_session', 'emit_agent_status', 'emit_dialogue', 'emit_mic_state', 'sync_mic_with_stt', 'handle_start', 'handle_reset_work'}, namespace)
    node = namespace['UiNode']()
    namespace['ros_node'] = node
    namespace['handle_start']()
    return node, namespace


def make_llm():
    namespace = base_namespace()
    namespace.update(new_order=new_order, SECTION_LABELS=SECTION_LABELS, SECTION_ORDER=SECTION_ORDER, ENABLE_VLM=False, ENABLE_RUNTIME_LOG=False)
    load_source('llm_node.py', {'LLMNode', 'ORDER_COMPLETE_REPLY'}, namespace)
    node = namespace['LLMNode'].__new__(namespace['LLMNode'])
    FakeNode.__init__(node)
    node.camera_lock = threading.Lock()
    node.vlm_camera_messages = deque()
    node._clear_state()
    node._record_runtime_event = Mock()
    for field in ('reply_pub', 'done_pub', 'stt_enable_pub', 'status_pub'):
        setattr(node, field, node.create_publisher(Message, field, 10))
    return node, namespace


class TestOrderCompletion(unittest.TestCase):
    def deliver(self, ui, name, reply='마지막 안내'):
        if name == 'main':
            ui.main_reset_callback(Message('reset'))
        elif name == 'tts':
            ui.utterance_done_callback(Message(json.dumps({'text': reply, 'status': 'finished'})))
        elif name == 'status':
            ui.agent_status_callback(Message(json.dumps({'work_state': 'completed', 'completion_reply': reply, 'reset_allowed': False})))

    def test_all_signal_orders_wait_for_llm_ack(self):
        for ordering in itertools.permutations(('main', 'tts', 'status')):
            with self.subTest(ordering=ordering):
                ui, namespace = make_ui()
                for event in ordering[:2]:
                    self.deliver(ui, event)
                    self.assertEqual(ui.reset_publisher.messages, [])
                self.deliver(ui, ordering[2])
                self.assertTrue(namespace['is_ui_session_active']())
                self.assertEqual([json.loads(m.data) for m in ui.reset_publisher.messages], [{'action': 'complete'}])
                for event in ordering:
                    self.deliver(ui, event)
                self.assertEqual(len(ui.reset_publisher.messages), 1)
                ui.agent_status_callback(Message('{"work_state":"idle","reset_allowed":false}'))
                self.assertTrue(namespace['is_ui_session_active']())
                ui.agent_status_callback(Message('{"work_state":"idle","reset_allowed":true}'))
                self.assertFalse(namespace['is_ui_session_active']())
                events = [call.args[0] for call in namespace['socketio'].emit.call_args_list]
                self.assertEqual(events.count('work_complete'), 1)
                self.assertEqual(namespace['latest_agent_status'], {})
                self.assertEqual(list(namespace['cached_dialogue']), [])

    def test_wrong_or_generic_tts_does_not_reset(self):
        ui, namespace = make_ui()
        self.deliver(ui, 'main')
        self.deliver(ui, 'status')
        ui.tts_done_callback(Message('finished'))
        self.deliver(ui, 'tts', '이전 안내')
        for payload in ('bad', '[]', '{}', '{"text":42,"status":"finished"}', '{"text":"마지막 안내","status":"speaking"}'):
            ui.utterance_done_callback(Message(payload))
        self.assertEqual(ui.reset_publisher.messages, [])
        self.assertTrue(namespace['is_ui_session_active']())

    def test_final_failure_is_terminal_not_success(self):
        ui, _ = make_ui()
        self.deliver(ui, 'main')
        self.deliver(ui, 'status')
        ui.utterance_done_callback(Message('{"text":"마지막 안내","status":"failed"}'))
        self.assertEqual(len(ui.reset_publisher.messages), 1)
        ui.logger.warning.assert_called()

    def test_browser_cannot_request_completion(self):
        ui, namespace = make_ui()
        namespace['handle_reset_work']({'action': 'complete'})
        self.assertEqual(ui.reset_publisher.messages, [])
        namespace['handle_start']()
        self.assertEqual(len(ui.start_publisher.messages), 1)
        self.assertIn('/reset', ui.subscriptions)
        self.assertNotIn('/ui/reset', ui.subscriptions)

    def test_concurrent_start_is_atomic(self):
        ui, namespace = make_ui()
        namespace['reset_ui_session']('complete')
        ui.start_publisher.messages.clear()
        original_check = namespace['is_ui_session_active']
        locked_checks = []
        def check_under_lock():
            locked_checks.append(namespace['ros_node_lock'].locked())
            return original_check()
        namespace['is_ui_session_active'] = check_under_lock
        start_barrier = threading.Barrier(8)
        def click():
            start_barrier.wait(timeout=2)
            namespace['handle_start']()
        workers = [threading.Thread(target=click) for _ in range(8)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=2)
            self.assertFalse(worker.is_alive())
        self.assertEqual(locked_checks, [True] * 8)
        self.assertEqual(len(ui.start_publisher.messages), 1)

    def test_two_orders_with_real_llm_finish_and_clear(self):
        ui, namespace = make_ui()
        llm, llm_namespace = make_llm()
        for index in range(2):
            llm.order.update({'sauce': '토마토', 'noodle_type': '넓은면', 'noodle_portion': 'normal'})
            llm.section = 'sauce'
            llm.active_task = {'class': '토마토', 'repeat_count': 1}
            llm._process_finish()
            llm._publish_status()
            status = json.loads(llm.status_pub.messages[-1].data)
            self.assertEqual(status['completion_reply'], llm_namespace['ORDER_COMPLETE_REPLY'])
            self.assertEqual(llm.reply_pub.messages[-1].data, status['completion_reply'])
            ui.agent_status_callback(llm.status_pub.messages[-1])
            self.deliver(ui, 'tts', status['completion_reply'])
            self.deliver(ui, 'main')
            self.assertTrue(namespace['is_ui_session_active']())
            llm._process_reset(ui.reset_publisher.messages[-1].data)
            llm._publish_status()
            ui.agent_status_callback(llm.status_pub.messages[-1])
            self.assertFalse(namespace['is_ui_session_active']())
            self.assertEqual(llm.order, new_order())
            self.assertFalse(llm.order_finished)
            self.assertEqual(llm.completed_tasks, [])
            # 비활성 세션에서 받은 이전 종료 신호는 다음 주문에 남기지 않는다.
            self.deliver(ui, 'main')
            self.deliver(ui, 'tts', status['completion_reply'])
            if index == 0:
                namespace['handle_start']()
                self.assertIsNone(ui.last_tts_text)
                self.assertFalse(ui.main_reset_received)
        self.assertEqual(len(ui.start_publisher.messages), 2)
        self.assertEqual(len(ui.reset_publisher.messages), 2)


class TestTtsCompletion(unittest.TestCase):
    def make_tts(self):
        namespace = base_namespace()
        load_source('tts_node.py', {'TTSNode'}, namespace)
        node = namespace['TTSNode'].__new__(namespace['TTSNode'])
        FakeNode.__init__(node)
        events = []
        node.speaker = SimpleNamespace(stop=lambda: events.append('drained'), start=lambda: events.append('started'))
        node.stt_retrigger_pub = SimpleNamespace(publish=lambda message: events.append(('legacy', message.data)))
        node.utterance_done_pub = SimpleNamespace(publish=lambda message: events.append(('detail', json.loads(message.data))))
        return node, events

    def test_completion_after_audio_drain_preserves_original_text(self):
        for result in ('finished', 'failed'):
            node, events = self.make_tts()
            node._speak_reply = Mock(return_value=result)
            node.llm_callback(Message('  면 3개  '))
            self.assertEqual(events, ['drained', 'started', ('legacy', result), ('detail', {'text': '면 3개', 'status': result})])

    def test_exception_still_reports_failure_once(self):
        node, events = self.make_tts()
        node._speak_reply = Mock(side_effect=RuntimeError('synthesis failure'))
        node.llm_callback(Message('마지막 안내'))
        self.assertEqual(events[-2:], [('legacy', 'failed'), ('detail', {'text': '마지막 안내', 'status': 'failed'})])
        self.assertEqual(len(events), 4)

    def test_audio_drain_failure_is_not_success(self):
        node, events = self.make_tts()
        node._speak_reply = Mock(return_value='finished')
        node.speaker.stop = Mock(side_effect=RuntimeError('device error'))
        node.llm_callback(Message('마지막 안내'))
        self.assertEqual(events[-1][1]['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
