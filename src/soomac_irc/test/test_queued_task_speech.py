"""후속 작업 안내와 실행/VLM 선행 조건을 ROS 없이 검사한다."""
import ast
import copy
from pathlib import Path
from types import MethodType

import pytest

from soomac_irc.llm_langgraph import _with_particle
from test_section_question_events import make_node


def queued_node(section, previous, following):
    node = make_node(section)
    source = Path(__file__).parents[1] / "soomac_irc/llm_langgraph_node.py"
    cls = next(n for n in ast.parse(source.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == "LLMLangGraphNode")
    method = next(n for n in cls.body
                  if isinstance(n, ast.FunctionDef) and n.name == "_process_next")
    namespace = dict(copy=copy, ENABLE_VLM=True, BY_VLM_NUM=3,
                     _with_particle=_with_particle)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    node._process_next = MethodType(namespace["_process_next"], node)
    node.ui_started = False
    node.vlm_confirmed = True
    node.active_task = {"class": previous, "repeat_count": 1}
    node.task_queue = [{"class": following, "repeat_count": 2}]
    return node


@pytest.mark.parametrize("section,previous,following,reply", [
    ("extra", "치즈", "페퍼론치노", "이제 페퍼론치노를 담을게요."),
    ("veggie", "양파", "버섯", "이제 버섯을 담을게요."),
])
def test_next_task_is_announced_once_after_plan(section, previous, following, reply):
    node = queued_node(section, previous, following)
    observed = []
    node.plan_pub.publish.side_effect = lambda _: observed.append("plan")
    node._publish_reply.side_effect = lambda _: observed.append("reply")
    node._process_next(3)
    assert observed == ["plan", "reply"]
    node._publish_reply.assert_called_once_with(reply)
    assert node.active_task["class"] == following
    assert node.task_queue == []


def test_failed_plan_does_not_announce_start():
    node = queued_node("extra", "치즈", "페퍼론치노")
    node.plan_pub.publish.side_effect = RuntimeError("plan unavailable")
    with pytest.raises(RuntimeError, match="plan unavailable"):
        node._process_next(3)
    node._publish_reply.assert_not_called()
    assert node.task_queue == [{"class": "페퍼론치노", "repeat_count": 2}]


@pytest.mark.parametrize("confirmed,result_code", [(False, 3), (True, 0)])
def test_unverified_or_invalid_next_does_not_start_or_speak(confirmed, result_code):
    node = queued_node("extra", "치즈", "페퍼론치노")
    node.vlm_confirmed = confirmed
    node._process_next(result_code)
    node.plan_pub.publish.assert_not_called()
    node._publish_reply.assert_not_called()
    assert node.active_task["class"] == "치즈"
