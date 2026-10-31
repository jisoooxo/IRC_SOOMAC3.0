"""HF transformers with BF16 base (same 3 adapters), cache OFF, same 14 token-identical inputs."""
import json, sys, time
sys.path.insert(0, "/home/roma/IRC_SOOMAC3.0/src/soomac_irc")
import torch, xgrammar as xgr
from xgrammar.contrib.hf import LogitsProcessor
import soomac_irc.model_runtime as mr
mr.MODEL_QUANTIZATION = "bf16"
S = "/tmp/claude-1000/-home-roma-IRC-SOOMAC3-0/f10905d6-bf59-4085-a889-d658537908bd/scratchpad"
data = json.load(open(f"{S}/b3_inputs.json")); ref = json.load(open(f"{S}/b3_hf_ref.json"))
model, processor = mr.load_model(mr.DECISION_ADAPTER_PATH, mr.RESPONSE_ADAPTER_PATH, mr.RECOMMENDATION_ADAPTER_PATH)
print("alloc GiB", round(torch.cuda.memory_allocated() / 2**30, 2), flush=True)
tok = processor.tokenizer
stop = list(dict.fromkeys(i for i in [tok.eos_token_id, tok.convert_tokens_to_ids("<turn|>")] if isinstance(i, int) and i >= 0))
comp = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tok, vocab_size=len(tok), stop_token_ids=stop))
gram = {k: comp.compile_json_schema(v, any_order=(k == "decision")) for k, v in data["schemas"].items()}
ADP = {"decision": "decision", "recommendation": "recommendation", "response": "response"}
MAX = {"decision": 1024, "recommendation": 1024, "response": 512}
rows = []
for ev, r in zip(data["events"], ref):
    st = ev["stage"]; model.set_adapter(ADP[st])
    ids = torch.tensor([r["input_ids"]], device="cuda")
    t = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=MAX[st], do_sample=False, use_cache=True, eos_token_id=stop, logits_processor=[LogitsProcessor(gram[st])])
    torch.cuda.synchronize()
    text = processor.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
    rows.append({"key": r["key"], "in": r["in"], "s": round(time.perf_counter() - t, 3), "text": text})
    print(json.dumps(rows[-1], ensure_ascii=False)[:220], flush=True)
json.dump(rows, open(f"{S}/b3_hf_bf16.json", "w"), ensure_ascii=False, indent=1)
print("DONE")
