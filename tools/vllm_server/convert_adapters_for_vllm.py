"""학습한 LoRA 어댑터를 vLLM 서버용으로 변환한다(원본은 바꾸지 않는다).

왜 필요한가: Gemma 4의 글로벌 층(k_eq_v)은 v_proj가 없고 V = k_proj(x)(LoRA 포함)이다.
vLLM은 기본 K 가중치를 V 자리에도 복사하지만 LoRA는 q/k/v 자리별로 붙어 V에는 LoRA가 빠진다.
v_proj 이름으로 같은 A/B를 하나 더 두면 V = W_k x + B A x 가 되어 HF(PEFT) 결과와 같아진다.

쓰는 법(어댑터를 새로 학습하고 model_runtime.py 경로를 바꾼 뒤, vLLM 서버를 끈 상태에서):
    python tools/vllm_server/convert_adapters_for_vllm.py            # 3개 모두
    python tools/vllm_server/convert_adapters_for_vllm.py response   # 일부만
변환 후에는 반드시 vLLM 서버를 다시 시작한다(서버는 시작할 때만 어댑터를 읽는다).

2026-11-01: docs/2026-11-01_accel_evidence/convert_adapters_for_vllm.py(당시 실행 기록)를 도구로 옮겼다.
변환 내용은 같고, 원본 해시 기록·실제 rank 확인·서버 실행 중 거부·전부 만든 뒤 교체를 더했다.
"""
import hashlib
import json
import re
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/soomac_irc"))
from safetensors.torch import load_file, save_file  # noqa: E402

from soomac_irc.model_runtime import (  # noqa: E402
    DECISION_ADAPTER_PATH, MODEL_PATH, RECOMMENDATION_ADAPTER_PATH, RESPONSE_ADAPTER_PATH,
)
from soomac_irc.vllm_client import VLLM_BASE_URL  # noqa: E402

OUT = ROOT / "models/vllm_adapters"
SOURCES = {"decision": DECISION_ADAPTER_PATH, "recommendation": RECOMMENDATION_ADAPTER_PATH, "response": RESPONSE_ADAPTER_PATH}
SERVER_MAX_LORA_RANK = 16  # start_vllm_server.sh의 VLLM_MAX_LORA_RANK 기본값
K_PROJ = re.compile(r"layers\.(\d+)\.self_attn\.k_proj\.")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def server_running() -> bool:
    try:
        with urllib.request.urlopen(VLLM_BASE_URL + "/health", timeout=2):
            return True
    except OSError:
        return False


def build_staging(name: str, source: Path, global_layers: set[int]) -> Path:
    tensors = load_file(str(source / "adapter_model.safetensors"))
    added = 0
    for key in list(tensors):
        match = K_PROJ.search(key)
        if match and int(match.group(1)) in global_layers:
            v_key = key.replace(".self_attn.k_proj.", ".self_attn.v_proj.")
            if v_key in tensors:
                raise SystemExit(f"{name}: 이미 v_proj가 있음({v_key}). 학습 대상 모듈이 바뀌었는지 확인")
            tensors[v_key] = tensors[key].clone()
            added += 1
    if added == 0:
        raise SystemExit(f"{name}: 글로벌 층 k_proj LoRA가 없음. 학습 대상 모듈(target_modules)을 확인")
    # 설정의 r이 아니라 실제 텐서 크기로 rank를 본다(rank_pattern으로 층마다 다를 수 있다).
    rank = max(t.shape[0] for k, t in tensors.items() if "lora_A" in k)
    if rank > SERVER_MAX_LORA_RANK:
        raise SystemExit(f"{name}: rank {rank} > {SERVER_MAX_LORA_RANK}. start_vllm_server.sh의 VLLM_MAX_LORA_RANK와 "
                         f"이 파일의 SERVER_MAX_LORA_RANK를 {rank} 이상으로 올린 뒤 다시 변환")

    # 임시 폴더에 다 쓴다. 교체는 3개가 모두 성공한 뒤 main에서 한다.
    staging = OUT / f".{name}.tmp"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    save_file(tensors, str(staging / "adapter_model.safetensors"))
    shutil.copy2(source / "adapter_config.json", staging / "adapter_config.json")
    record = {
        "source": str(source),
        "source_sha256": sha256(source / "adapter_model.safetensors"),
        "added_v_proj_tensors": added,
        "global_layers": sorted(global_layers),
        "rank": rank,
    }
    (staging / "CONVERSION.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{name}: v_proj {added}개 추가, rank {rank}")
    return staging


def swap_in(name: str, staging: Path) -> None:
    # 기존 변환본은 새 것이 자리 잡을 때까지 지우지 않는다(중간에 끊겨도 둘 중 하나는 남는다).
    target = OUT / name
    previous = OUT / f".{name}.old"
    shutil.rmtree(previous, ignore_errors=True)
    if target.exists():
        target.rename(previous)
    staging.rename(target)
    shutil.rmtree(previous, ignore_errors=True)
    print(f"{name} -> {target}")


def main() -> None:
    names = sys.argv[1:] or list(SOURCES)
    unknown = [n for n in names if n not in SOURCES]
    if unknown:
        raise SystemExit(f"모르는 어댑터 이름: {unknown}. 가능: {list(SOURCES)}")
    if server_running():
        raise SystemExit("vLLM 서버가 켜져 있음. 서버를 끄고 변환한 뒤 다시 시작해야 새 어댑터가 반영된다")
    text_config = json.loads((Path(MODEL_PATH) / "config.json").read_text(encoding="utf-8"))["text_config"]
    global_layers = {i for i, kind in enumerate(text_config["layer_types"]) if kind == "full_attention"}
    for name in names:
        if not (Path(SOURCES[name]) / "adapter_model.safetensors").is_file():
            raise SystemExit(f"{name}: 어댑터 파일 없음: {SOURCES[name]}. model_runtime.py 경로 확인")
    # 전부 만든 뒤에 교체한다. 하나라도 실패하면 기존 변환본은 그대로 남는다.
    staged = {name: build_staging(name, Path(SOURCES[name]), global_layers) for name in names}
    for name, staging in staged.items():
        swap_in(name, staging)


if __name__ == "__main__":
    main()
