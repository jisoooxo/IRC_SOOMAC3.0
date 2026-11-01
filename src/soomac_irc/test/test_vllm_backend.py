import json
from types import SimpleNamespace

import pytest
import torch

from soomac_irc import vllm_client
from soomac_irc.vllm_client import VllmError, VllmModel, check_served_adapters


def _fake_complete(calls, token_ids=(7, 8, 1)):
    def complete_tokens(model, prompt_token_ids, max_tokens, stop_token_ids, grammar=None):
        calls.append({"model": model, "prompt": prompt_token_ids, "max_tokens": max_tokens,
                      "stop": stop_token_ids, "grammar": grammar})
        return {"text": "x", "token_ids": list(token_ids), "finish_reason": "stop",
                "prompt_tokens": len(prompt_token_ids), "completion_tokens": len(token_ids), "cached_tokens": 0}
    return complete_tokens


def test_generate_sends_active_adapter_grammar_and_returns_hf_shaped_tensor(monkeypatch):
    calls = []
    monkeypatch.setattr(vllm_client, "complete_tokens", _fake_complete(calls))
    grammar_processor = SimpleNamespace(compiled_grammars=[SimpleNamespace(grammar="root ::= \"{}\"")])
    model = VllmModel()
    model.set_adapter("decision")

    output = model.generate(input_ids=torch.tensor([[2, 3, 4]]), attention_mask=torch.ones(1, 3),
                            max_new_tokens=1024, eos_token_id=[1, 106], do_sample=False,
                            use_cache=True, logits_processor=[grammar_processor])

    assert calls == [{"model": "decision", "prompt": [2, 3, 4], "max_tokens": 1024,
                      "stop": [1, 106], "grammar": "root ::= \"{}\""}]
    assert output.tolist() == [[2, 3, 4, 7, 8, 1]]
    assert model.last_usage["completion_tokens"] == 3


def test_disable_adapter_uses_base_model_and_restores(monkeypatch):
    calls = []
    monkeypatch.setattr(vllm_client, "complete_tokens", _fake_complete(calls))
    model = VllmModel()
    model.set_adapter("response")
    with model.disable_adapter():
        model.generate(input_ids=torch.tensor([[5]]), max_new_tokens=8, eos_token_id=1)
    assert calls[0]["model"] == vllm_client.VLLM_BASE_MODEL
    assert calls[0]["stop"] == [1]
    assert model.active_adapter == "response"


def test_sampling_is_rejected():
    with pytest.raises(VllmError):
        VllmModel().generate(input_ids=torch.tensor([[5]]), max_new_tokens=8, eos_token_id=1, do_sample=True)


def test_served_adapter_must_come_from_current_training_output(tmp_path):
    served = tmp_path / "served_decision"
    served.mkdir()
    (served / "CONVERSION.json").write_text(json.dumps({"source": str(tmp_path / "old")}))
    roots = {"decision": str(served)}

    with pytest.raises(VllmError, match="현재 Adapter와 다름"):
        check_served_adapters({"decision": str(tmp_path / "new")}, roots)
    check_served_adapters({"decision": str(tmp_path / "old")}, roots)
    with pytest.raises(VllmError, match="등록되지 않음"):
        check_served_adapters({"response": str(tmp_path / "new")}, roots)
    with pytest.raises(VllmError, match="변환 기록 없음"):
        check_served_adapters({"decision": str(tmp_path / "old")}, {"decision": str(tmp_path / "raw_adapter")})


def test_missing_token_ids_is_a_transport_error(monkeypatch):
    def complete_tokens(*_args, **_kwargs):
        return {"text": "", "token_ids": None, "finish_reason": "stop",
                "prompt_tokens": 1, "completion_tokens": 0, "cached_tokens": 0}
    monkeypatch.setattr(vllm_client, "complete_tokens", complete_tokens)
    with pytest.raises(VllmError, match="token_ids"):
        VllmModel().generate(input_ids=torch.tensor([[5]]), max_new_tokens=8, eos_token_id=1)


def test_output_budget_is_capped_to_server_length(monkeypatch):
    calls = []
    monkeypatch.setattr(vllm_client, "complete_tokens", _fake_complete(calls))
    model = VllmModel(max_model_len=10)
    model.generate(input_ids=torch.tensor([[1, 2, 3, 4, 5, 6]]), max_new_tokens=1024, eos_token_id=1)
    assert calls[0]["max_tokens"] == 4
    with pytest.raises(VllmError, match="최대 길이"):
        model.generate(input_ids=torch.tensor([list(range(10))]), max_new_tokens=8, eos_token_id=1)


@pytest.mark.parametrize("served_len", [None, "10240", 8192])
def test_backend_refuses_server_without_enough_length(monkeypatch, served_len):
    # 서버 길이를 모르거나 노드 입력 기준(10240)보다 짧으면 노드 시작을 막는다.
    monkeypatch.setattr(vllm_client, "wait_until_ready", lambda: None)
    monkeypatch.setattr(vllm_client, "served_models",
                        lambda: [{"id": vllm_client.VLLM_BASE_MODEL, "max_model_len": served_len}])
    with pytest.raises(VllmError):
        vllm_client.load_vllm_backend("unused", {})


def test_served_adapter_detects_retraining_into_same_path(tmp_path):
    # 같은 경로에 다시 학습해 파일만 바뀐 경우도 변환 기록의 해시로 잡는다.
    source = tmp_path / "adapter"
    source.mkdir()
    (source / "adapter_model.safetensors").write_bytes(b"old weights")
    served = tmp_path / "served"
    served.mkdir()
    (served / "CONVERSION.json").write_text(json.dumps(
        {"source": str(source), "source_sha256": vllm_client._sha256(source / "adapter_model.safetensors")}))
    check_served_adapters({"decision": str(source)}, {"decision": str(served)})
    (source / "adapter_model.safetensors").write_bytes(b"new weights")
    with pytest.raises(VllmError, match="변환 뒤에 바뀜"):
        check_served_adapters({"decision": str(source)}, {"decision": str(served)})
