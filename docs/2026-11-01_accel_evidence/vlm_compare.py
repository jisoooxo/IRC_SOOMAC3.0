"""VLM: logged real prompts + real meal-kit camera images; backend 'hf' (model_runtime.make_call_vlm) or 'vllm' (server)."""
import glob, json, sys, time
sys.path.insert(0, "/home/roma/IRC_SOOMAC3.0/src/soomac_irc")
from PIL import Image
backend, out_path = sys.argv[1], sys.argv[2]
events, seen = [], set()
for f in sorted(glob.glob("/home/roma/IRC_SOOMAC3.0/logs/llm/2026-10-3*/*/runtime_events.jsonl")):
    for line in open(f):
        e = json.loads(line)
        if e.get("event_type") == "vlm_result" and e["request"].get("image_count") and e["class"] not in seen:
            seen.add(e["class"]); events.append(e)
images = sorted(glob.glob("/home/roma/ros2_ws/src/soomac_irc/vlm_data/mealkit_coco_v15/valid/*.jpg"))
if backend == "hf":
    from soomac_irc.model_runtime import load_model, make_call_vlm, DECISION_ADAPTER_PATH, RESPONSE_ADAPTER_PATH, RECOMMENDATION_ADAPTER_PATH
    model, processor = load_model(DECISION_ADAPTER_PATH, RESPONSE_ADAPTER_PATH, RECOMMENDATION_ADAPTER_PATH)
    call_vlm = make_call_vlm(model, processor)
else:
    from soomac_irc import vllm_client
    from soomac_irc.model_runtime import VLM_MAX_TOKENS
    vllm_client.wait_until_ready(900)
    call_vlm = vllm_client.make_call_vlm_remote(VLM_MAX_TOKENS)
rows = []
for i, e in enumerate([events[0]] + events):
    n = e["request"]["image_count"]
    pil = [Image.open(images[(i * 37 + j * 11) % len(images)]).convert("RGB") for j in range(n)]
    t = time.perf_counter()
    reply = call_vlm(pil, e["request"]["system_prompt"], e["request"]["user_text"])
    dt = time.perf_counter() - t
    if i == 0:
        continue  # warmup
    verdict = "pass" if "PASS" in reply.upper()[-40:] else ("fail" if "FAIL" in reply.upper()[-40:] else "other")
    rows.append({"class": e["class"], "images": n, "s": round(dt, 3), "verdict": verdict, "reply": reply})
    print(json.dumps(rows[-1], ensure_ascii=False)[:260], flush=True)
json.dump(rows, open(out_path, "w"), ensure_ascii=False, indent=1)
print("DONE")
