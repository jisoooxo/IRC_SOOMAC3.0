"""Replay logged VLM requests (real prompts, image counts, output lengths) with synthetic camera-size images."""
import glob, json, statistics as st, sys, time
sys.path.insert(0, "/home/roma/IRC_SOOMAC3.0/src/soomac_irc")
import numpy as np, torch, transformers
from PIL import Image
from soomac_irc.model_runtime import load_model, DECISION_ADAPTER_PATH, RESPONSE_ADAPTER_PATH, RECOMMENDATION_ADAPTER_PATH

events = []
for f in sorted(glob.glob("/home/roma/IRC_SOOMAC3.0/logs/llm/2026-10-3*/*/runtime_events.jsonl")):
    for line in open(f):
        e = json.loads(line)
        if e.get("event_type") == "vlm_result" and e.get("raw") and e["request"].get("image_count"):
            events.append(e)
by_class = {}
for e in events:
    by_class.setdefault(e["class"], e)  # first per class
sample = list(by_class.values())[:10]
model, processor = load_model(DECISION_ADAPTER_PATH, RESPONSE_ADAPTER_PATH, RECOMMENDATION_ADAPTER_PATH)
tok = processor.tokenizer
rng = np.random.default_rng(0)
def img():
    return Image.fromarray(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8))
rows = []
for i, e in enumerate([sample[0]] + sample):  # first = warmup
    req = e["request"]
    n_img = req["image_count"]
    out_len = len(tok(e["raw"], add_special_tokens=False)["input_ids"])
    content = [{"type": "image", "image": img()} for _ in range(n_img)] + [{"type": "text", "text": req["user_text"]}]
    msgs = [{"role": "system", "content": req["system_prompt"]}, {"role": "user", "content": content}]
    torch.cuda.synchronize(); t0 = time.perf_counter()
    inputs = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=False).to(model.device)
    torch.cuda.synchronize(); t1 = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16), model.disable_adapter():
        model.generate(**inputs, max_new_tokens=1, min_new_tokens=1, do_sample=False, use_cache=True)
        torch.cuda.synchronize(); t2 = time.perf_counter()
        model.generate(**inputs, max_new_tokens=out_len, min_new_tokens=out_len, do_sample=False, use_cache=True)
    torch.cuda.synchronize(); t3 = time.perf_counter()
    if i == 0:
        continue
    r = {"class": e["class"], "images": n_img, "in_tokens": inputs["input_ids"].shape[1], "out_tokens": out_len,
         "preprocess_s": round(t1 - t0, 3), "first_token_s": round(t2 - t1, 3), "full_s": round(t3 - t2, 3)}
    rows.append(r); print(json.dumps(r, ensure_ascii=False), flush=True)
print("SUMMARY", transformers.__version__, json.dumps({k: round(st.mean(r[k] for r in rows), 3) for k in ("images", "in_tokens", "out_tokens", "preprocess_s", "first_token_s", "full_s")}), flush=True)
print("PEAK_GiB", round(torch.cuda.max_memory_allocated() / 2**30, 2))
