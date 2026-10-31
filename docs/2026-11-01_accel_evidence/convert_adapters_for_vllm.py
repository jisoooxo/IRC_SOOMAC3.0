"""Gemma4 k_eq_v 글로벌 층용 LoRA 변환: k_proj LoRA를 v_proj 이름으로 복제한 사본을 만든다(원본 불변).

HF: 글로벌 층은 v_proj가 없고 V = k_proj(x)(LoRA 포함). vLLM: 기본 K 가중치를 K·V 슬롯에 복사하지만
LoRA는 q/k/v 슬롯별로 붙어 V에는 delta가 빠진다. v_proj에 같은 A/B를 두면 V = W_k x + B A x 로 HF와 같아진다.
"""
import json, re, shutil, sys
from pathlib import Path
from safetensors.torch import load_file, save_file

sys.path.insert(0, "/home/roma/IRC_SOOMAC3.0/src/soomac_irc")
from soomac_irc.model_runtime import DECISION_ADAPTER_PATH, MODEL_PATH, RECOMMENDATION_ADAPTER_PATH, RESPONSE_ADAPTER_PATH

OUT = Path("/home/roma/IRC_SOOMAC3.0/models/vllm_adapters")
text_cfg = json.load(open(f"{MODEL_PATH}/config.json"))["text_config"]
glob_layers = {i for i, t in enumerate(text_cfg["layer_types"]) if t == "full_attention"}
pattern = re.compile(r"layers\.(\d+)\.self_attn\.k_proj\.")

for name, src in (("decision", DECISION_ADAPTER_PATH), ("recommendation", RECOMMENDATION_ADAPTER_PATH), ("response", RESPONSE_ADAPTER_PATH)):
    src = Path(src)
    tensors = load_file(str(src / "adapter_model.safetensors"))
    added = 0
    for key in list(tensors):
        m = pattern.search(key)
        if m and int(m.group(1)) in glob_layers:
            v_key = key.replace(".self_attn.k_proj.", ".self_attn.v_proj.")
            assert v_key not in tensors, f"이미 v_proj가 있음: {v_key}"
            tensors[v_key] = tensors[key].clone()
            added += 1
    dst = OUT / name
    dst.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(dst / "adapter_model.safetensors"))
    shutil.copy2(src / "adapter_config.json", dst / "adapter_config.json")
    json.dump({"source": str(src), "added_v_proj_tensors": added, "global_layers": sorted(glob_layers)},
              open(dst / "CONVERSION.json", "w"), ensure_ascii=False, indent=1)
    print(name, "v_proj 추가", added, "->", dst)
