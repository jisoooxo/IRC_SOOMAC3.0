"""B3: same 14 token-identical inputs on vLLM (BF16 base) with converted / original / no LoRA; compare to HF reference."""
import json, sys, time
import xgrammar as xgr
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.lora.request import LoRARequest
from vllm.sampling_params import StructuredOutputsParams

S = "/tmp/claude-1000/-home-roma-IRC-SOOMAC3-0/f10905d6-bf59-4085-a889-d658537908bd/scratchpad"
MODEL = "/home/roma/Desktop/sLLM/gemma-4-12B-it"
CONV = "/home/roma/IRC_SOOMAC3.0/models/vllm_adapters"
ORIG = {"decision": "/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/runs/remote_all_a519cf_20261031_r2/results/checkpoints/attempt-d8a4ee465c95443daa47e2caf7fa1ee4/decision/best_adapter",
        "recommendation": "/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/runs/local_v6_batch_20261028/recommendation-c492d6c7538b/attempt-8f34316e864e429ba8ebdbd8b60f8a8a/recommendation/best_adapter",
        "response": "/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/runs/local_v6_batch_20261028/response-b1acc16-resume-bed9d88cbc5a/full/attempt-2b74a8919fce4e5690767295942c27d5/response/best_adapter"}
MAX_NEW = {"decision": 1024, "recommendation": 1024, "response": 512}

def main():
    data = json.load(open(f"{S}/b3_inputs.json")); ref = json.load(open(f"{S}/b3_hf_ref.json"))
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    stop_ids = sorted({tok.eos_token_id, tok.convert_tokens_to_ids("<turn|>")})
    ebnf = {k: str(xgr.Grammar.from_json_schema(json.dumps(v, ensure_ascii=False), any_order=(k == "decision"))) for k, v in data["schemas"].items()}

    llm = LLM(model=MODEL, dtype="bfloat16", enable_lora=True, max_lora_rank=16, max_loras=3, max_model_len=10240,
              gpu_memory_utilization=0.94, enable_prefix_caching=True, limit_mm_per_prompt={"image": 0, "audio": 0},
              structured_outputs_config={"backend": "xgrammar"})
    ids = {"conv": 1, "orig": 10}
    def lora(variant, stage):
        if variant == "none": return None
        path = f"{CONV}/{stage}" if variant == "conv" else ORIG[stage]
        return LoRARequest(f"{variant}-{stage}", ids[variant] + ["decision", "recommendation", "response"].index(stage), path)

    def norm(text):
        try: return json.dumps(json.loads(text), sort_keys=True, ensure_ascii=False)
        except Exception: return "INVALID:" + text

    results = []
    plan = [("conv", None), ("conv", None), ("orig", None), ("none", "decision")]  # conv twice: pass1 cold, pass2 warm
    for p, (variant, only) in enumerate(plan):
        for ev, r in zip(data["events"], ref):
            stage = ev["stage"]
            if only and stage != only: continue
            sp = SamplingParams(temperature=0, max_tokens=MAX_NEW[stage], stop_token_ids=stop_ids,
                                structured_outputs=StructuredOutputsParams(grammar=ebnf[stage]))
            t = time.perf_counter()
            out = llm.generate(TokensPrompt(prompt_token_ids=r["input_ids"]), sp, lora_request=lora(variant, stage), use_tqdm=False)[0]
            dt = time.perf_counter() - t
            text = out.outputs[0].text.strip()
            row = {"pass": p, "variant": variant, "key": r["key"], "in": r["in"], "out": len(out.outputs[0].token_ids), "s": round(dt, 3),
                   "cached": out.num_cached_tokens, "exact_vs_hf_off": text == r["off_text"], "sem_vs_hf_off": norm(text) == norm(r["off_text"]),
                   "sem_vs_hf_on": norm(text) == norm(r["on_text"]), "text": text, "hf_off": r["off_text"]}
            results.append(row)
            print(json.dumps({k: v for k, v in row.items() if k not in ("text", "hf_off")}, ensure_ascii=False), flush=True)
    json.dump(results, open(f"{S}/b3_vllm_rows.json", "w"), ensure_ascii=False, indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
