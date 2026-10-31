"""Turn-by-turn comparison: previous run / cache OFF / cache ON."""
import json
import sys

S = "/tmp/claude-1000/-home-roma-IRC-SOOMAC3-0/f10905d6-bf59-4085-a889-d658537908bd/scratchpad"
RUNS = {"prev": "/tmp/soomac_closed_loop_all_20261101_v2/turns.jsonl", "off": f"{S}/scn_off/turns.jsonl", "on": f"{S}/scn_on/turns.jsonl"}


def load(path):
    rows = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        rows[(r["episode"], r["turn"])] = r
    return rows


def summary(r):
    d = r.get("decision_final") or {}
    pend = r["after"].get("pending") or {}
    o = r["after"]["order"]
    tops = ",".join(f"{k}:{v}" for k, v in sorted(o["toppings"].items()))
    rest = ",".join(f"{x.get('target')}/{x.get('reason')}" for x in o.get("restrictions") or [])
    lat = {k: [round(e.get("latency_ms", 0) / 1000, 2) for e in v] for k, v in r["traces"].items() if v}
    return {
        "order": f"{o.get('noodle_type')}/{o.get('noodle_portion')} 소스={o.get('sauce')} [{tops}] 제약=[{rest}]",
        "pending": f"{pend.get('type')}:{json.dumps(pend.get('candidate') or pend.get('targets'), ensure_ascii=False)}" if pend else "-",
        "commit": d.get("commit"), "confirm": d.get("confirmation"), "rec": (d.get("recommendation") or {}).get("action"),
        "execute": r["policy"]["execute"], "reason": r["policy"].get("reason"),
        "failed": [k for k, v in r["checks"].items() if not v], "reply": r["reply"],
        "D": [e.get("raw") for e in r["traces"].get("decision", [])],
        "lat_s": round(r["latency_ms"] / 1000, 2), "stage_s": lat,
        "cache": [e.get("prefix_cache", {}).get("status") for g in r["traces"].values() for e in g],
    }


runs = {k: load(v) for k, v in RUNS.items()}
keys = sorted(runs["off"].keys(), key=lambda k: (k[0] != "compound", k[1]))
for key in keys:
    rows = {name: runs[name].get(key) for name in runs}
    base = rows["off"]
    print(f"\n=== {key[0]} T{key[1]} [{base['tag']}] \"{base['user_text']}\"  기대: {json.dumps(base['expected_order']['toppings'], ensure_ascii=False)} execute={base['checks'] and base['policy']}"[:400] if False else
          f"\n=== {key[0]} T{key[1]} [{base['tag']}] \"{base['user_text']}\"")
    for name in ("prev", "off", "on"):
        r = rows[name]
        if r is None:
            print(f"  {name:4s}: (없음)")
            continue
        s = summary(r)
        print(f"  {name:4s}: {'PASS' if not s['failed'] else 'FAIL ' + ','.join(s['failed'])} | exec={s['execute']} commit={s['commit']} conf={s['confirm']} rec={s['rec']} | {s['order']} | pending={s['pending']} | {s['lat_s']}s {s['stage_s']} cache={s['cache']}")
        print(f"        D={s['D']}")
        print(f"        P=\"{s['reply']}\"")
