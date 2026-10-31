import json, sys, time, urllib.request
sys.path.insert(0, "/home/roma/IRC_SOOMAC3.0/src/soomac_irc")
import xgrammar as xgr
from soomac_irc import vllm_client
S = "/tmp/claude-1000/-home-roma-IRC-SOOMAC3-0/f10905d6-bf59-4085-a889-d658537908bd/scratchpad"
t0 = time.time(); vllm_client.wait_until_ready(900); print(f"ready after {time.time()-t0:.0f}s", flush=True)
print("models:", [m["id"] for m in json.load(urllib.request.urlopen(vllm_client.VLLM_BASE_URL + "/v1/models"))["data"]])
data = json.load(open(f"{S}/b3_inputs.json")); ref = json.load(open(f"{S}/b3_hf_ref.json"))
gram = {k: str(xgr.Grammar.from_json_schema(json.dumps(v, ensure_ascii=False), any_order=(k == "decision"))) for k, v in data["schemas"].items()}
MAX = {"decision": 1024, "recommendation": 1024, "response": 512}
def norm(t):
    try: return json.dumps(json.loads(t), sort_keys=True, ensure_ascii=False)
    except Exception: return "INVALID:" + t
ok = 0
for ev, r in zip(data["events"], ref):
    t = time.perf_counter()
    out = vllm_client.complete_tokens(ev["stage"], r["input_ids"], MAX[ev["stage"]], [1, 106], gram[ev["stage"]])
    dt = time.perf_counter() - t; same = norm(out["text"]) == norm(r["off_text"]); ok += same
    print(json.dumps({"key": r["key"], "in": r["in"], "s": round(dt, 3), "cached": out["cached_tokens"], "out": out["completion_tokens"], "sem_vs_hf": same}, ensure_ascii=False), flush=True)
print(f"SEM_MATCH {ok}/{len(ref)}")
try:
    urllib.request.urlopen("https://pypi.org", timeout=3); print("INTERNET: OPEN")
except Exception as e:
    print("INTERNET: BLOCKED", type(e).__name__)
