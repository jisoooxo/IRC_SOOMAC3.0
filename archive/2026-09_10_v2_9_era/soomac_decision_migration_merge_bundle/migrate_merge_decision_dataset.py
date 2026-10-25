#!/usr/bin/env python3
import argparse, copy, json, re, collections
from pathlib import Path
import jsonschema

SEMANTIC_KEYS = {
    "order","restrictions","preferences","recommendation",
    "commit","confirmation","queries","clarify"
}

# Current-message surface lexicon only.
# Keep conservative: omission is safer than hallucinated mentions.
SUPPORTED = [
    "오일","토마토","크림","얇은면","넓은면",
    "양파","버섯","소시지","게살","치즈","페퍼론치노",
    "유제품","갑각류","육류","비건",
]
SET_TERMS = ["야채","채소","추가 재료"]
UNSUPPORTED_KNOWN = [
    "햄","베이컨","살라미","초리조","스팸","새우","미트볼","마늘","브로콜리",
    "모짜렐라","할라피뇨","참치","파프리카","당근","파슬리","고추",
    "올리브","치킨","감자","피클","바질",
]
STT_SURFACES = [
    "소세지","계살","개살","패퍼런치노","페퍼런치노","페페론치노",
]
MENTION_TERMS = SUPPORTED + SET_TERMS + UNSUPPORTED_KNOWN + STT_SURFACES

REFERENCE_ONLY = {"그거","이거","아까 그거","둘 다","첫 번째 거","두 번째 거"}

def load_jsonl(path):
    rows=[]
    with open(path,encoding="utf-8") as f:
        for n,line in enumerate(f,1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception as e:
                    raise RuntimeError(f"{path}:{n}: invalid JSON: {e}")
    return rows

def dump_jsonl(path, rows):
    with open(path,"w",encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n")

def extract_mentions(message):
    # Longest-first at the same position prevents nested duplicate surfaces.
    hits=[]
    for term in MENTION_TERMS:
        start=0
        while True:
            idx=message.find(term,start)
            if idx < 0: break
            hits.append((idx,-len(term),term))
            start=idx+len(term)
    hits.sort()
    out=[]
    occupied=[]
    for idx,neglen,term in hits:
        end=idx+len(term)
        if any(not (end<=a or idx>=b) for a,b in occupied):
            continue
        occupied.append((idx,end))
        out.append((idx,term))
    out.sort()
    return [t for _,t in out if t not in REFERENCE_ONLY]

def migrate_row(row):
    out=copy.deepcopy(row)
    old_target=copy.deepcopy(row.get("target",{}))
    keys=set(old_target)
    if not (keys & SEMANTIC_KEYS):
        return None, "empty_or_nonsemantic_target"

    new_target={"route":"task"}
    mentions=extract_mentions(str(row.get("input",{}).get("message","")))
    if mentions:
        new_target["mentions"]=mentions

    for k,v in old_target.items():
        new_target[k]=v

    # Strong invariant: legacy semantics must be byte-equivalent as JSON values.
    preserved={k:v for k,v in new_target.items() if k not in {"route","mentions"}}
    if preserved != old_target:
        return None, "semantic_target_changed"

    out["target"]=new_target
    out["migration"]={
        "source_contract":"v2.2.5-semantic-rebalance",
        "target_contract":"flexible_llm-route-mentions",
        "route_added":"task",
        "mentions_added":bool(mentions),
    }
    return out, None

def schema_errors(validator,row):
    return [e.message for e in validator.iter_errors(row["target"])]

def strict_checks(row, old_target=None):
    errs=[]
    t=row["target"]
    msg=str(row.get("input",{}).get("message",""))

    if "route" not in t:
        errs.append("missing_route")
    for m in t.get("mentions",[]):
        if m not in msg:
            errs.append(f"mention_not_grounded:{m}")
    if old_target is not None:
        preserved={k:v for k,v in t.items() if k not in {"route","mentions"}}
        if preserved != old_target:
            errs.append("legacy_semantic_loss_or_change")
    return errs

def context_key(row):
    # Exact full training example identity. Same text under different state is intentionally distinct.
    return json.dumps(
        {"input":row.get("input"),"target":row.get("target")},
        ensure_ascii=False,sort_keys=True,separators=(",",":")
    )

def semantic_minpair_key(row):
    # Used only for reporting near duplicate surfaces, never automatic deletion.
    return (
        str(row.get("input",{}).get("message","")).strip(),
        json.dumps(row.get("target"),ensure_ascii=False,sort_keys=True,separators=(",",":"))
    )

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--legacy",required=True)
    ap.add_argument("--supplement",required=True)
    ap.add_argument("--schema",required=True)
    ap.add_argument("--out",required=True)
    args=ap.parse_args()

    outdir=Path(args.out); outdir.mkdir(parents=True,exist_ok=True)
    legacy=load_jsonl(args.legacy)
    supplement=load_jsonl(args.supplement)
    schema=json.load(open(args.schema,encoding="utf-8"))
    validator=jsonschema.Draft202012Validator(schema)

    migrated_pass=[]; review=[]; failed=[]
    route_counts=collections.Counter()
    mentions_rows=0

    for row in legacy:
        migrated, reason=migrate_row(row)
        if reason:
            x=copy.deepcopy(row); x["migration_review_reason"]=reason
            review.append(x); continue

        errs=schema_errors(validator,migrated)
        errs += strict_checks(migrated, old_target=row.get("target",{}))
        if errs:
            x=copy.deepcopy(migrated); x["migration_validator_errors"]=errs
            failed.append(x); continue

        migrated_pass.append(migrated)
        route_counts[migrated["target"]["route"]]+=1
        if migrated["target"].get("mentions"): mentions_rows+=1

    dump_jsonl(outdir/"migrated_legacy_pass.jsonl",migrated_pass)
    dump_jsonl(outdir/"migrated_legacy_review_required.jsonl",review)
    dump_jsonl(outdir/"migrated_legacy_fail.jsonl",failed)

    merged=migrated_pass + supplement
    dump_jsonl(outdir/"merged_raw.jsonl",merged)

    seen=set(); final=[]; removed=[]
    for row in merged:
        k=context_key(row)
        if k in seen:
            x=copy.deepcopy(row); x["dedup_reason"]="exact_full_context_duplicate"
            removed.append(x)
        else:
            seen.add(k); final.append(row)

    dump_jsonl(outdir/"dedup_removed.jsonl",removed)

    final_fail=[]; final_pass=[]
    for row in final:
        errs=schema_errors(validator,row)
        errs += strict_checks(row)
        if errs:
            x=copy.deepcopy(row); x["final_validator_errors"]=errs
            final_fail.append(x)
        else:
            final_pass.append(row)

    dump_jsonl(outdir/"final_validator_fail.jsonl",final_fail)
    dump_jsonl(outdir/"final_train.jsonl",final_pass)

    surface_counts=collections.Counter(
        str(r.get("input",{}).get("message","")).strip() for r in final_pass
    )
    repeated_surfaces=sum(1 for _,c in surface_counts.items() if c>1)

    manifest={
        "legacy_original":len(legacy),
        "migration_pass":len(migrated_pass),
        "migration_review_required":len(review),
        "migration_fail":len(failed),
        "supplement":len(supplement),
        "merged_raw":len(merged),
        "dedup_removed":len(removed),
        "validator_fail":len(final_fail),
        "final_train":len(final_pass),
        "route_counts":dict(route_counts),
        "mentions_added_rows":mentions_rows,
        "mentions_absent_rows":len(migrated_pass)-mentions_rows,
        "repeated_surface_groups_retained":repeated_surfaces,
        "dedup_policy":"exact full input+target only; state/context differences and minimal pairs retained",
        "legacy_overwritten":False,
    }
    json.dump(manifest,open(outdir/"manifest.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)

    report=f"""# Decision dataset migration + merge report

legacy original = {len(legacy)}
migration pass = {len(migrated_pass)}
migration review required = {len(review)}
migration fail = {len(failed)}

supplement = {len(supplement)}

merged raw = {len(merged)}
dedup removed = {len(removed)}
validator fail = {len(final_fail)}
final train = {len(final_pass)}

route task = {route_counts.get('task',0)}
route general = {route_counts.get('general',0)}
route mixed = {route_counts.get('mixed',0)}

mentions added rows = {mentions_rows}
mentions absent rows = {len(migrated_pass)-mentions_rows}

Notes:
- Legacy v2.2.5 is a task-only Decision dataset; route is migrated to task when a legacy semantic target exists.
- Existing semantic target fields are preserved exactly.
- mentions only use literal current-message surfaces from a conservative lexicon.
- Reference-only expressions are not emitted as mentions.
- Exact full-context duplicates are removed.
- Same surface with different state/context is preserved.
- Minimal pairs are preserved.
"""
    (outdir/"migration_report.md").write_text(report,encoding="utf-8")
    print(report)

if __name__=="__main__":
    main()
