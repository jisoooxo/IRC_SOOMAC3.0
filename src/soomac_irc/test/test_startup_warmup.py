"""Startup isolation and failure gates, without ROS or GPU."""
import ast
from collections import deque
from pathlib import Path
import queue
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from soomac_irc.agent_contract import new_session_state
from soomac_irc.startup_warmup import warmup_vllm_agents
from soomac_irc.vllm_client import VllmError, VllmModel


NAMES = {role: role for role in ("decision", "recommendation", "response")}


@pytest.mark.parametrize("failure", [None, "decision", "recommendation", "response", "missing_tokens", "timeout"])
def test_prefix_calls_discard_outputs_restore_proxy_and_abort(monkeypatch, failure):
    from soomac_irc import startup_warmup as sw, vllm_client as vc
    model = VllmModel(16384)
    saved_usage = {"previous": True}
    model.active_adapter, model.last_usage = "previous", saved_usage
    seen = []
    log = Mock()
    monkeypatch.setattr(sw, "_warmup_specs", lambda processor: (
        (role, [1, 2, 3], [9], "grammar") for role in NAMES))

    def complete(role, prompt, max_tokens, stop, grammar):
        assert prompt == [1, 2, 3] and max_tokens == 1
        assert stop == [9] and grammar == "grammar"
        seen.append(role)
        if failure == role:
            raise ValueError("transport error")
        if failure == "timeout":
            raise TimeoutError("request timeout")
        return dict(text="DISCARD", token_ids=[] if failure == "missing_tokens" else [7],
                    finish_reason="length", completion_tokens=1)
    monkeypatch.setattr(vc, "complete_tokens", complete)
    if failure:
        with pytest.raises(VllmError, match="시작 워밍업 실패"):
            warmup_vllm_agents(model, object(), NAMES, log)
        expected_count = list(NAMES).index(failure) + 1 if failure in NAMES else 1
        assert len(seen) == expected_count
        assert not any("워밍업 완료" in str(call) for call in log.info.call_args_list)
    else:
        result = warmup_vllm_agents(model, object(), NAMES, log)
        assert list(result["roles"]) == list(NAMES)
        assert "DISCARD" not in str(result)
    assert model.active_adapter == "previous" and model.last_usage is saved_usage
    assert "DISCARD" not in str(log.mock_calls)


def test_prefix_excludes_user_content_and_uses_production_template_options():
    import torch
    from soomac_irc.startup_warmup import _prefix_tokens
    seen = []
    def render(messages, **kwargs):
        seen.append((messages, kwargs))
        return {"input_ids": torch.tensor([[1, 2, 3, 4 if messages[-1]["content"] == "A" else 5, 6]])}
    assert _prefix_tokens(SimpleNamespace(apply_chat_template=render), "SYSTEM") == [1, 2, 3]
    assert all(messages[0] == {"role": "system", "content": "SYSTEM"} for messages, _ in seen)
    assert all(options == dict(add_generation_prompt=True, tokenize=True, return_dict=True,
                               return_tensors="pt", enable_thinking=False) for _, options in seen)


def test_rejects_hf_missing_adapters_and_excessive_input(monkeypatch):
    from soomac_irc import startup_warmup as sw
    with pytest.raises(TypeError):
        warmup_vllm_agents(object(), object(), NAMES)
    with pytest.raises(ValueError):
        warmup_vllm_agents(VllmModel(), object(), {"decision": "decision"})
    monkeypatch.setattr(sw, "_warmup_specs", lambda _: [("decision", [1, 2, 3], [9], "g")])
    with pytest.raises(VllmError, match="길이 한도"):
        warmup_vllm_agents(VllmModel(3), object(), NAMES)


def test_empty_user_probe_excludes_shared_synthetic_token():
    import torch
    from soomac_irc.startup_warmup import _prefix_tokens
    def render(messages, **kwargs):
        content = messages[-1]["content"]
        ids = [1, 2, 9] if not content else [1, 2, 7, 4 if content == "A" else 5]
        return {"input_ids": torch.tensor([ids])}
    assert _prefix_tokens(SimpleNamespace(apply_chat_template=render), "SYSTEM") == [1, 2]


@pytest.mark.parametrize("backend,enabled,injected,fail", [
    ("vllm", True, False, False), ("vllm", False, False, False),
    ("hf", True, False, False), ("vllm", True, True, False),
    ("vllm", True, False, True),
])
def test_actual_constructor_gates_publishers_subscribers_and_worker(backend, enabled, injected, fail):
    # Execute the actual constructor AST, with hardware dependencies replaced.
    source = Path(__file__).parents[1] / "soomac_irc/llm_langgraph_node.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LLMLangGraphNode")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    isolated = ast.ClassDef(name="LLMLangGraphNode", bases=[ast.Name(id="Node", ctx=ast.Load())],
                           keywords=[], body=[init], decorator_list=[])
    events = []
    log = SimpleNamespace(info=lambda message: events.append(message))

    class Node:
        def __init__(self, name):
            pass

        def get_logger(self):
            return log

        def create_publisher(self, *args):
            events.append("publisher")
            return Mock()

        def create_subscription(self, *args):
            events.append("subscription")
            return Mock()

        def _clear_state(self):
            events.append("live_state")
            self.session = new_session_state()

        def __getattr__(self, name):
            return Mock()

    def warm(*args):
        assert "live_state" not in events and "publisher" not in events
        events.append("warmup")
        if fail:
            raise VllmError("warmup failure")

    model = VllmModel()
    namespace = dict(Node=Node, LLM_BACKEND=backend, LLM_STARTUP_WARMUP=enabled,
                     ENABLE_VLM=False, ENABLE_VLM_UI_IMAGES=False, EXECUTION_REPLY_MODE="fixed",
                     MODEL_PATH="unused", VLM_MAX_TOKENS=1024, NUMBER_IMAGE_FOR_CONFIRM=5,
                     load_vllm_backend=lambda *args: (model, object()),
                     load_model=lambda *args: (object(), object()),
                     warmup_vllm_agents=warm, build_graph=Mock(), queue=queue, deque=deque,
                     LLMSessionJsonlLogger=Mock(), String=object, Bool=object, Int16=object,
                     threading=SimpleNamespace(Event=Mock(), Lock=Mock(),
                         Thread=lambda **kwargs: SimpleNamespace(start=lambda: events.append("worker"))))
    for role in NAMES:
        namespace[role.upper() + "_ADAPTER_NAME"] = role
        namespace[role.upper() + "_ADAPTER_PATH"] = "unused"
        namespace["make_call_" + role] = Mock(return_value=Mock())
    exec(compile(ast.fix_missing_locations(ast.Module(body=[isolated], type_ignores=[])),
                 str(source), "exec"), namespace)
    ctor = namespace["LLMLangGraphNode"]
    if fail:
        with pytest.raises(VllmError):
            ctor()
        assert not set(events) & {"publisher", "subscription", "worker", "live_state", "LLM 입력 준비 완료"}
    else:
        node = ctor(Mock(call_recommendation=Mock(), call_response=Mock()) if injected else None)
        should_warm = backend == "vllm" and enabled and not injected
        assert events.count("warmup") == int(should_warm)
        if should_warm:
            assert events.index("warmup") < events.index("live_state") < events.index("publisher")
        assert events.index("subscription") < events.index("worker") < events.index("LLM 입력 준비 완료")
        assert node.session == new_session_state()
