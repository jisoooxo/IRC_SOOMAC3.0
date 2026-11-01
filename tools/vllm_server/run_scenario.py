"""17턴 시나리오 하네스(manual_scenario_stack_probe)를 하네스 수정 없이 세 방식으로 돌린다.

    hf            : 현재 코드 그대로(HF, 입력 길이 처리 C 적용)
    hf-no-budget  : C 이전 동작(Decision 상한 16384, Recommendation·Response 자르기 없음)
    vllm          : vLLM 서버로 추론(서버가 먼저 떠 있어야 한다)

쓰는 법: python tools/vllm_server/run_scenario.py <mode> --measure-all --out <새 폴더>
보통은 check_after_training.sh가 메모리 상한 안에서 부른다.
2026-11-01: docs/2026-11-01_accel_evidence/run_scenario_{cache,vllm}.py를 합쳐 옮겼다.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src/soomac_irc"), str(ROOT / "src/soomac_irc/test")]

MODES = ("hf", "hf-no-budget", "vllm")
mode = sys.argv.pop(1) if len(sys.argv) > 1 else ""
if mode not in MODES:
    raise SystemExit(f"사용법: run_scenario.py {{{'|'.join(MODES)}}} --measure-all --out <폴더>")

import torch  # noqa: E402

import soomac_irc.model_runtime as model_runtime  # noqa: E402
from soomac_irc import decision_model, recommendation_model, response_model, vllm_client  # noqa: E402
import manual_scenario_stack_probe as probe  # noqa: E402

if mode == "hf-no-budget":
    # 상한은 호출할 때 모듈 전역에서 읽으므로 여기서 바꾸면 C 이전과 같은 입력이 된다.
    decision_model.DECISION_INPUT_MAX_TOKENS = 16384
    recommendation_model.RECOMMENDATION_INPUT_MAX_TOKENS = 10**9
    response_model.RESPONSE_INPUT_MAX_TOKENS = 10**9

if mode == "vllm":
    class HarnessVllmModel(vllm_client.VllmModel):
        def parameters(self):  # 하네스는 가중치가 CUDA에 있는지 확인한다. 가중치는 서버가 들고 있다
            return iter([torch.empty(1, device="cuda")])

    def load_model(decision_path, response_path, recommendation_path):
        served, processor = vllm_client.load_vllm_backend(model_runtime.MODEL_PATH, {
            "decision": decision_path, "recommendation": recommendation_path, "response": response_path})
        return HarnessVllmModel(max_model_len=served.max_model_len), processor

    _check_output = subprocess.check_output

    def check_output(args, *rest, **kwargs):  # 하네스는 GPU가 비어 있길 요구한다. vLLM 서버가 GPU를 쓰는 것은 정상이다
        if args and args[0] == "nvidia-smi":
            return ""
        return _check_output(args, *rest, **kwargs)

    model_runtime.load_model = load_model
    probe.subprocess.check_output = check_output

print(f"run_scenario mode={mode}", flush=True)
probe.main()
