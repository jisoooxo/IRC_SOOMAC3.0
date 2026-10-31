import json, os, statistics as st, re
S = "/tmp/claude-1000/-home-roma-IRC-SOOMAC3-0/f10905d6-bf59-4085-a889-d658537908bd/scratchpad"
cfgs = [c for c in "ABCD" if os.path.exists(f"{S}/mtp_{c}/replay14.json")]
def load_scn(p): return {(r["episode"], r["turn"]): r for r in map(json.loads, open(p))}
field = lambda r: (json.dumps(r["after"]["order"], sort_keys=True, ensure_ascii=False), r["policy"]["execute"],
                   (r.get("decision_final") or {}).get("commit"), json.dumps(r["after"].get("pending"), sort_keys=True, ensure_ascii=False))
rep = {c: json.load(open(f"{S}/mtp_{c}/replay14.json")) for c in cfgs}
scn = {c: {i: load_scn(f"{S}/mtp_{c}/scn{i}/turns.jsonl") for i in (1, 2) if os.path.exists(f"{S}/mtp_{c}/scn{i}/turns.jsonl")} for c in cfgs}
for c in cfgs:
    m = open(f"{S}/mtp_{c}/metrics.txt").read()
    num = lambda k: float(re.search(rf"^vllm:{k}\S* (\S+)", m, re.M).group(1)) if re.search(rf"^vllm:{k}\S* (\S+)", m, re.M) else None
    acc, dr = num("spec_decode_num_accepted_tokens_total"), num("spec_decode_num_draft_tokens_total")
    r = rep[c]; p0 = [x for x in r if x["pass"] == 0]; p1 = [x for x in r if x["pass"] == 1]
    per = {k: round(st.mean(x["s"] for x in p1 if x["key"] == k), 3) for k in ("decision", "response/task", "recommendation", "response/general")}
    real = [x["s"] for x in p0 if x["cached"]]
    tokA = sum(x["token_ids"] == y["token_ids"] for x, y in zip(r, rep["A"])) if "A" in rep else None
    tot = {i: round(sum(t["latency_ms"] for t in s.values()) / 1000, 1) for i, s in scn[c].items()}
    same12 = sum(field(scn[c][1][k]) == field(scn[c][2][k]) for k in scn[c][1]) if len(scn[c]) == 2 else None
    vsA = sum(field(scn[c][1][k]) == field(scn["A"][1][k]) for k in scn[c][1]) if "A" in scn and 1 in scn[c] else None
    t11 = [e.get("raw") for e in scn[c][1][("split_control", 11)]["traces"]["decision"]] if 1 in scn[c] else None
    print(f"[{c}] accept={acc}/{dr}={'%.2f' % (acc/dr) if acc and dr else '-'} | replay pass0 {st.mean(x['s'] for x in p0):.3f}s (prefix-hit {st.mean(real):.3f}s n={len(real)}) pass1 {st.mean(x['s'] for x in p1):.3f}s per-agent warm {per}")
    print(f"     sem_vs_HF pass0 {sum(x['sem_vs_hf'] for x in p0)}/14 pass1 {sum(x['sem_vs_hf'] for x in p1)}/14 | token-identical to A: {tokA}/28 | 17turn totals {tot} | run1==run2 fields {same12}/17 | run1 vs A-run1 fields {vsA}/17 | T11 D {t11}")
