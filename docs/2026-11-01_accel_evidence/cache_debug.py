import sys, json, glob
sys.path.insert(0, "/home/roma/IRC_SOOMAC3.0/src/soomac_irc")
import torch
from soomac_irc.model_runtime import load_model, DECISION_ADAPTER_PATH, RESPONSE_ADAPTER_PATH, RECOMMENDATION_ADAPTER_PATH
from soomac_irc.prefix_cache import PrefixKVCache
model, processor = load_model(DECISION_ADAPTER_PATH, RESPONSE_ADAPTER_PATH, RECOMMENDATION_ADAPTER_PATH)
msgs = None
for f in sorted(glob.glob("/home/roma/IRC_SOOMAC3.0/logs/llm/2026-10-31/*/agent_turns.jsonl")):
    for line in open(f):
        t = json.loads(line)
        for e in (t.get("llm_trace") or {}).get("decision", []):
            if e.get("messages"): msgs = e["messages"]; break
        if msgs: break
    if msgs: break
model.set_adapter("decision")
inputs = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=False).to("cuda")
pc = PrefixKVCache(processor)
torch.cuda.synchronize(); m0 = torch.cuda.memory_allocated()
with torch.inference_mode():
    entry = pc._build(model, ("decision", "x"), msgs[0]["content"], inputs["input_ids"])
torch.cuda.synchronize(); m1 = torch.cuda.memory_allocated()
c = entry[1]
l0, l5 = c.layers[0], c.layers[5]
print("types", type(l0).__name__, type(l5).__name__)
for n, l in (("L0", l0), ("L5", l5)):
    k = l.keys
    print(n, "shape", tuple(k.shape), "base None?", k._base is None, "storage GiB", round(k.untyped_storage().nbytes() / 2**30, 3), "contig", k.is_contiguous())
print("alloc delta GiB", round((m1 - m0) / 2**30, 3))
import gc
tot = {}
for i, l in enumerate(c.layers):
    for k, v in vars(l).items():
        if isinstance(v, torch.Tensor):
            key = f"{type(l).__name__}.{k}"
            tot[key] = tot.get(key, 0) + v.untyped_storage().nbytes()
print("cache tensors GiB", {k: round(v / 2**30, 3) for k, v in tot.items()})
print("cache attrs", [k for k in vars(c).keys()])
big = [(o.untyped_storage().nbytes(), tuple(o.shape), o.dtype) for o in gc.get_objects() if isinstance(o, torch.Tensor) and o.is_cuda and o.untyped_storage().nbytes() > 50 * 2**20 and o.dim() == 4]
big.sort(reverse=True)
print("4D cuda tensors >50MB:", len(big), big[:5])
