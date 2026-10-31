"""14 logged inputs via the running vLLM server (token-identical), 2 passes; save texts/latency/cache."""
import json, sys, time
sys.path.insert(0, "/home/roma/IRC_SOOMAC3.0/src/soomac_irc")
import xgrammar as xgr
from soomac_irc import vllm_client
S = "/tmp/claude-1000/-home-roma-IRC-SOOMAC3-0/f10905d6-bf59-4085-a889-d658537908bd/scratchpad"
out_path = sys.argv[1]
data = json.load(open(f"{S}/b3_inputs.json")); ref = json.load(open(f"{S}/b3_hf_ref.json"))
gram = {k: str(xgr.Grammar.from_json_schema(json.dumps(v, ensure_ascii=False), any_order=(k == "decision"))) for k, v in data["schemas"].items()}
MAX = {"decision": 1024, "recommendation": 1024, "response": 512}
def norm(t):
    try: return json.dumps(json.loads(t), sort_keys=True, ensure_ascii=False)
    except Exception: return "INVALID:" + t
rows = []
for p in (0, 1):
    for ev, r in zip(data["events"], ref):
        t = time.perf_counter()
        o = vllm_client.complete_tokens(ev["stage"], r["input_ids"], min(MAX[ev["stage"]], 10240 - r["in"]), [1, 106], gram[ev["stage"]])
        rows.append({"pass": p, "key": r["key"], "in": r["in"], "s": round(time.perf_counter() - t, 3), "cached": o["cached_tokens"],
                     "out": o["completion_tokens"], "token_ids": o["token_ids"], "text": o["text"].strip(),
                     "sem_vs_hf": norm(o["text"]) == norm(r["off_text"])})
json.dump(rows, open(out_path, "w"), ensure_ascii=False)
for p in (0, 1):
    g = [r for r in rows if r["pass"] == p]
    print(f"pass{p}: mean {sum(r['s'] for r in g)/len(g):.3f}s sem_vs_hf {sum(r['sem_vs_hf'] for r in g)}/14")
