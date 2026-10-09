from pathlib import Path
from types import SimpleNamespace

import pytest

from soomac_irc.adapter_runtime import activate_adapter
import soomac_irc.model_runtime as model_runtime


class FakeRoutedModel:
    def __init__(self):
        self.selected = []

    def set_adapter(self, name):
        self.selected.append(name)


def test_activate_adapter_selects_requested_name():
    model = FakeRoutedModel()

    assert activate_adapter(model, "decision") is True
    assert activate_adapter(model, "response") is True
    assert model.selected == ["decision", "response"]


def test_activate_adapter_none_keeps_legacy_path():
    model = object()

    assert activate_adapter(model, None) is False


def test_activate_adapter_rejects_model_without_peft_switching():
    with pytest.raises(TypeError, match="response Adapter"):
        activate_adapter(object(), "response")


def test_runtime_adapter_artifacts_exist():
    for adapter_path in (
        model_runtime.DECISION_ADAPTER_PATH,
        model_runtime.RESPONSE_ADAPTER_PATH,
    ):
        root = Path(adapter_path)
        assert (root / "adapter_config.json").is_file()
        assert (root / "adapter_model.safetensors").is_file()


def test_load_model_registers_two_named_adapters(monkeypatch):
    calls = []

    class FakeBaseModel:
        def eval(self):
            calls.append(("base_eval",))

    class FakePeftModel:
        def __init__(self, base):
            self.base = base

        @classmethod
        def from_pretrained(
            cls,
            base,
            path,
            *,
            adapter_name,
            is_trainable,
        ):
            calls.append(("from_pretrained", path, adapter_name, is_trainable))
            return cls(base)

        def load_adapter(self, path, *, adapter_name, is_trainable):
            calls.append(("load_adapter", path, adapter_name, is_trainable))

        def set_adapter(self, adapter_name):
            calls.append(("set_adapter", adapter_name))

        def eval(self):
            calls.append(("peft_eval",))

    monkeypatch.setattr(
        model_runtime,
        "AutoProcessor",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: "processor"),
    )
    monkeypatch.setattr(
        model_runtime,
        "_AutoVLM",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakeBaseModel()),
    )
    monkeypatch.setattr(
        model_runtime,
        "BitsAndBytesConfig",
        lambda **kwargs: kwargs,
    )
    monkeypatch.setattr(model_runtime, "PeftModel", FakePeftModel)

    model, processor = model_runtime.load_model("decision-path", "response-path")

    assert isinstance(model, FakePeftModel)
    assert processor == "processor"
    assert ("from_pretrained", "decision-path", "decision", False) in calls
    assert ("load_adapter", "response-path", "response", False) in calls
    assert ("set_adapter", "decision") in calls
