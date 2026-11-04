"""Actual node method, isolated from ROS/cameras/models; speech-only controls."""
import ast
import copy
import json
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from soomac_irc.vlm import build_vlm_spoken_reply, decide_vlm_outcome, parse_vlm_verdict


def isolated_node(raw, *, enabled=True, failures=0, retake=False):
    source = Path(__file__).parents[1] / 'soomac_irc/llm_langgraph_node.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LLMLangGraphNode')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_process_vlm_confirm')
    events = []
    def speech(*args, **kwargs):
        events.append(('build_speech', kwargs['allow_reason']))
        return build_vlm_spoken_reply(*args, **kwargs)
    namespace = dict(copy=copy, json=json, String=SimpleNamespace, ENABLE_VLM=True,
                     VLM_JUDGE_REASON_SPEAK=enabled, ENABLE_UNCERTAIN_RETAKE=retake,
                     TOPIC_CLASS_NAMES={'버섯': 'mushroom'},
                     parse_vlm_verdict=parse_vlm_verdict, decide_vlm_outcome=decide_vlm_outcome,
                     build_vlm_spoken_reply=speech,
                     build_vlm_request=lambda *args, **kwargs: {'images': [], 'system_prompt': 'fixture', 'user_text': 'fixture'})
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(compile(module, str(source), 'exec'), namespace)
    node = SimpleNamespace(
        active_task={'class': '버섯'}, vlm_confirmed=False,
        _snapshot_camera_images=lambda: [Mock()], _get_vlm_reference=lambda expected: None,
        comparison_image=None, comparison_source=None, _publish_vlm_ui_snapshot=Mock(),
        call_vlm=Mock(side_effect=raw) if isinstance(raw, Exception) else Mock(return_value=raw),
        verification_history=[{'class': '버섯', 'result': 'fail'} for _ in range(failures)],
        graph_state={'action_history': []}, camera_lock=nullcontext(),
        latest_camera_message=object(), vlm_camera_messages=deque([object()]),
        vlm_result_pub=Mock(), _publish_reply=Mock(), get_logger=lambda: Mock(),
        runtime_log=Mock(), _build_robot_state=lambda: {'section': 'veggie'},
    )
    node.vlm_result_pub.publish.side_effect = lambda msg: events.append(('topic', msg.data))
    node._publish_reply.side_effect = lambda reply: events.append(('speech', reply))
    namespace['_process_vlm_confirm'](node, True)
    return node, events


def test_toggle_is_enabled_in_source():
    source = Path(__file__).parents[1] / 'soomac_irc/llm_langgraph_node.py'
    tree = ast.parse(source.read_text())
    setting = next(n for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'VLM_JUDGE_REASON_SPEAK' for t in n.targets))
    assert ast.literal_eval(setting.value) is True


def test_fail_topic_precedes_reason_formatting_and_speech():
    node, events = isolated_node('도시락 안에서 버섯이 확인되지 않아요.\n판정: FAIL')
    assert events[0] == ('topic', '{"result": "fail", "class": "mushroom"}')
    assert events[1] == ('build_speech', True)
    reply = node._publish_reply.call_args.args[0]
    assert reply == ('판단 근거를 말씀드리면, 도시락 안에서 버섯이 확인되지 않아요. '
                     '버섯 위치를 확인하지 못했어요. 해당 작업을 한 번 다시 시도할게요.')
    assert '판정:' not in reply
    assert node.vlm_confirmed is False
    node.call_vlm.assert_called_once()
    event = node.runtime_log.log_event.call_args.args[1]
    assert event['spoken_reply'] == reply
    assert event['spoken_reason_allowed'] is True


@pytest.mark.parametrize('raw,retake', [
    ('버섯이 보여요.\n판정: PASS', False),
    ('화면이 흐려요.\n판정: UNCERTAIN', False),
    ('화면이 흐려요.\n판정: UNCERTAIN', True),
    ('잘린 출력에는 확정된 판정이 없음', False),
    (RuntimeError('model unavailable'), False),
    ('판정: FAIL', False),
])
def test_nonfail_or_missing_reason_keeps_policy_only(raw, retake):
    node, events = isolated_node(raw, retake=retake)
    event = node.runtime_log.log_event.call_args.args[1]
    assert node._publish_reply.call_args.args[0] == event['outcome']['policy_reply']
    assert '판단 근거' not in node._publish_reply.call_args.args[0]
    assert 'model unavailable' not in node._publish_reply.call_args.args[0]


@pytest.mark.parametrize('failures', [0, 1])
def test_toggle_only_changes_speech_not_outcome_or_topics(failures):
    raw = '도시락 안에서 버섯이 확인되지 않아요.\n판정: FAIL'
    on, on_events = isolated_node(raw, enabled=True, failures=failures)
    off, off_events = isolated_node(raw, enabled=False, failures=failures)
    assert [e for e in on_events if e[0] == 'topic'] == [e for e in off_events if e[0] == 'topic']
    assert on.verification_history == off.verification_history
    assert on.graph_state == off.graph_state
    on_log = on.runtime_log.log_event.call_args.args[1]
    off_log = off.runtime_log.log_event.call_args.args[1]
    assert on_log['outcome'] == off_log['outcome']
    assert '판단 근거' in on_log['spoken_reply']
    assert off_log['spoken_reply'] == off_log['outcome']['policy_reply']
    if failures:
        assert on_events[0] == ('topic', 'success')
        assert on_log['outcome']['policy_result'] == 'policy_success'
        assert on_log['outcome']['trusted_pass'] is False
        assert on_log['spoken_reply'].endswith('버섯 재시도를 마쳤어요. 다음 단계로 진행할게요.')
