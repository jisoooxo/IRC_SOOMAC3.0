import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from soomac_irc.adapter_runtime import activate_adapter
import soomac_irc.model_runtime as model_runtime


def _write_adapter(root: Path) -> str:
    root.mkdir(parents=True)
    (root / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (root / "adapter_model.safetensors").write_bytes(b"test")
    return str(root)


def _write_manifest(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _set_manifest_paths(monkeypatch, tmp_path: Path) -> Path:
    manifest = tmp_path / "active_stack.json"
    monkeypatch.setattr(model_runtime, "RUNTIME_ADOPTION_MANIFEST", manifest)
    monkeypatch.setattr(model_runtime, "ROLLBACK_DECISION_ADAPTER_PATH", "rollback-decision")
    monkeypatch.setattr(model_runtime, "ROLLBACK_RESPONSE_ADAPTER_PATH", "rollback-response")
    monkeypatch.setattr(model_runtime, "ROLLBACK_RECOMMENDATION_ADAPTER_PATH", "rollback-recommendation")
    return manifest


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


@pytest.mark.parametrize(
    ("active", "all_gates_pass"),
    ((False, True), (True, False), (False, False)),
)
def test_runtime_manifest_requires_active_and_all_gates(
    monkeypatch, tmp_path, active, all_gates_pass
):
    manifest = _set_manifest_paths(monkeypatch, tmp_path)
    selected = {
        name: _write_adapter(tmp_path / name)
        for name in ("decision", "recommendation", "response")
    }
    _write_manifest(manifest, {
        "active": active,
        "all_gates_pass": all_gates_pass,
        "selected": selected,
    })

    decision, response, recommendation, error = model_runtime._resolve_runtime_adapter_paths()

    assert (decision, response, recommendation) == (
        "rollback-decision", "rollback-response", "rollback-recommendation"
    )
    assert error is None


def test_runtime_manifest_uses_only_complete_gate_passed_stack(monkeypatch, tmp_path):
    manifest = _set_manifest_paths(monkeypatch, tmp_path)
    selected = {
        name: _write_adapter(tmp_path / name)
        for name in ("decision", "recommendation", "response")
    }
    _write_manifest(manifest, {
        "active": True,
        "all_gates_pass": True,
        "selected": selected,
    })

    decision, response, recommendation, error = model_runtime._resolve_runtime_adapter_paths()

    assert (decision, response, recommendation) == (
        str(Path(selected["decision"]).resolve()),
        str(Path(selected["response"]).resolve()),
        str(Path(selected["recommendation"]).resolve()),
    )
    assert error is None


def test_runtime_manifest_rolls_back_when_selected_adapter_is_incomplete(monkeypatch, tmp_path):
    manifest = _set_manifest_paths(monkeypatch, tmp_path)
    selected = {
        name: _write_adapter(tmp_path / name)
        for name in ("decision", "recommendation", "response")
    }
    Path(selected["response"], "adapter_model.safetensors").unlink()
    _write_manifest(manifest, {
        "active": True,
        "all_gates_pass": True,
        "selected": selected,
    })

    decision, response, recommendation, error = model_runtime._resolve_runtime_adapter_paths()

    assert (decision, response, recommendation) == (
        "rollback-decision", "rollback-response", "rollback-recommendation"
    )
    assert "selected adapter incomplete" in error


def test_load_model_registers_three_named_adapters(monkeypatch):
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

    model, processor = model_runtime.load_model(
        "decision-path",
        "response-path",
        "recommendation-path",
    )

    assert isinstance(model, FakePeftModel)
    assert processor == "processor"
    assert ("from_pretrained", "decision-path", "decision", False) in calls
    assert ("load_adapter", "response-path", "response", False) in calls
    assert ("load_adapter", "recommendation-path", "recommendation", False) in calls
    assert ("set_adapter", "decision") in calls


def test_adopted_recommendation_adapter_is_routed_from_manifest():
    assert model_runtime.RECOMMENDATION_ADAPTER_PATH is not None
    root = Path(model_runtime.RECOMMENDATION_ADAPTER_PATH)
    assert (root / "adapter_config.json").is_file()
    assert (root / "adapter_model.safetensors").is_file()
