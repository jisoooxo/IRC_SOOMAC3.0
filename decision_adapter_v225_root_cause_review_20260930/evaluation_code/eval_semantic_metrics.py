#!/usr/bin/env python3
import argparse,json
from collections import Counter,defaultdict

def load(path):
    return [json.loads(x) for x in open(path,encoding='utf-8') if x.strip()]

def atoms(x):
    out=set(); x=x or {}
    o=x.get('order',{})
    for k,v in o.items():
        if k=='toppings':
            for tk,tv in v.items(): out.add(('order.topping',tk,str(tv)))
        else: out.add(('order.'+k,'',str(v)))
    for r in x.get('restrictions',[]): out.add(('restriction',r.get('target',''),r.get('reason','')+'|'+r.get('action','')))
    for p in x.get('preferences',[]): out.add(('preference',p.get('value',''),p.get('action','')))
    rr=x.get('recommendation')
    if rr:
        out.add(('recommendation.action','',rr.get('action','')))
        if 'scope' in rr: out.add(('recommendation.scope','',str(rr['scope'])))
        if 'criteria' in rr: out.add(('recommendation.criteria','',str(rr['criteria'])))
    if 'commit' in x: out.add(('commit','',str(x['commit']).lower()))
    if 'confirmation' in x: out.add(('confirmation','',str(x['confirmation'])))
    for q in x.get('queries',[]): out.add(('query',q.get('type',''),str(q.get('target',''))))
    if 'clarify' in x: out.add(('clarify','',str(x['clarify']).lower()))
    return out

def canon(x): return json.dumps(x or {},ensure_ascii=False,sort_keys=True,separators=(',',':'))

ap=argparse.ArgumentParser(); ap.add_argument('predictions'); args=ap.parse_args()
rows=load(args.predictions)
exact=0; fam=defaultdict(lambda:[0,0]); tp=fp=fn=0; multi_n=multi_ok=0
clar_tp=clar_fp=clar_fn=0; unsupported_sub=0; expected_clar=0
conf_n=conf_ok=commit_n=commit_ok=restr_n=restr_ok=query_n=query_ok=0
for r in rows:
    e=r.get('expected',r.get('target',{})); p=r.get('predicted',{})
    ok=canon(e)==canon(p); exact+=ok
    f=r.get('family',r.get('scenario_family','unknown')); fam[f][0]+=ok; fam[f][1]+=1
    ea,pa=atoms(e),atoms(p); tp+=len(ea&pa); fp+=len(pa-ea); fn+=len(ea-pa)
    if len(ea)>=2: multi_n+=1; multi_ok+=ea<=pa
    ec=e.get('clarify') is True; pc=p.get('clarify') is True
    if ec: expected_clar+=1
    if ec and pc: clar_tp+=1
    elif (not ec) and pc: clar_fp+=1
    elif ec and (not pc): clar_fn+=1
    if ec and not pc and any(k in p for k in ('order','restrictions','preferences','recommendation','commit','confirmation')): unsupported_sub+=1
    if 'confirmation' in e: conf_n+=1; conf_ok += p.get('confirmation')==e.get('confirmation')
    if 'commit' in e: commit_n+=1; commit_ok += p.get('commit')==e.get('commit')
    if 'restrictions' in e: restr_n+=1; restr_ok += canon({'restrictions':p.get('restrictions',[])})==canon({'restrictions':e.get('restrictions',[])})
    if 'queries' in e: query_n+=1; query_ok += canon({'queries':p.get('queries',[])})==canon({'queries':e.get('queries',[])})
N=len(rows)
prec=tp/(tp+fp) if tp+fp else 1.0; rec=tp/(tp+fn) if tp+fn else 1.0
cp=clar_tp/(clar_tp+clar_fp) if clar_tp+clar_fp else 1.0; cr=clar_tp/(clar_tp+clar_fn) if clar_tp+clar_fn else 1.0
print(f'exact_accuracy={exact}/{N}={exact/N:.4f}')
print(f'atom_precision={prec:.4f} atom_recall={rec:.4f}')
print(f'multi_intent_completeness={multi_ok}/{multi_n}={multi_ok/multi_n if multi_n else 1:.4f}')
print(f'clarify_precision={cp:.4f} clarify_recall={cr:.4f}')
print(f'clarify_to_mutation_error={unsupported_sub}/{expected_clar}={unsupported_sub/expected_clar if expected_clar else 0:.4f}')
if conf_n: print(f'confirmation_accuracy={conf_ok}/{conf_n}={conf_ok/conf_n:.4f}')
if commit_n: print(f'commit_accuracy={commit_ok}/{commit_n}={commit_ok/commit_n:.4f}')
if restr_n: print(f'restriction_exact={restr_ok}/{restr_n}={restr_ok/restr_n:.4f}')
if query_n: print(f'query_exact={query_ok}/{query_n}={query_ok/query_n:.4f}')
print('\nfamily_exact:')
for k,(a,b) in sorted(fam.items()): print(f'{k}: {a}/{b}={a/b:.3f}')
