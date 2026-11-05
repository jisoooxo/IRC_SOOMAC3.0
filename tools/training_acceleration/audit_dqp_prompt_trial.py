"""Read-only raw-log audit and derived reports for the 16K prompt trial."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import statistics
from tokenizers import Tokenizer

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'training/learning_first_v43/final/goals/GL10/dqp_prompt_multiturn_20261105'
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def clean(x):return str(x).replace('|','\\|').replace('\n',' ')

def audit(run):
    cases={c['record']['id']:c['record'] for c in read(BASE/'inputs/CASES.json')['rows']}
    expected=[];individual=rows(run/'INDIVIDUAL.jsonl');turns=rows(run/'TURNS.jsonl')
    for row in individual:
        if row.get('reused_from'):continue
        traces=row['traces']
        assert len(traces)==1
        expected.append(dict(id=row['id'],agent='decision',model_input=cases[row['id']]['input'],trace=traces[0]))
    for row in turns:
        for role in ('decision','recommendation','response'):
            for trace in row['traces'][role]:
                expected.append(dict(id=f"{row['episode']}/{row['turn']}/{role}",agent=role,model_input=trace.get('model_input'),trace=trace))
    requests=rows(run/'REQUESTS.jsonl');assert len(expected)==len(requests),(run.name,len(expected),len(requests))
    tok=Tokenizer.from_file('/home/roma/Desktop/sLLM/gemma-4-12B-it/tokenizer.json')
    proof=[];budgets=[];trim=[]
    for n,(want,actual) in enumerate(zip(expected,requests),1):
        req=actual['request'];trace=want['trace'];assert req['model']==want['agent']
        decoded=tok.decode(req['prompt_token_ids'],skip_special_tokens=False)
        block=decoded.rsplit('<|turn>user\n',1)[1];value,end=json.JSONDecoder().raw_decode(block)
        verified=want['model_input'] is not None
        if verified:assert canonical(value)==canonical(want['model_input']),want['id']
        raw=trace.get('raw');raw_verified=raw is not None
        if raw_verified:assert raw==actual['response']['text'].strip(),want['id']
        budget=len(req['prompt_token_ids'])+req['max_tokens'];assert budget<=16384
        if want['agent']=='decision' and trace.get('messages'):
            prompt=trace['messages'][0]['content'];assert prompt==(BASE/'inputs'/f"{read(run/'PLAN.json')['variant']}.txt").read_text()
        if trace.get('history_trimmed_messages'):trim.append(dict(id=want['id'],dropped=trace['history_trimmed_messages']))
        proof.append(dict(line=n,id=want['id'],role=want['agent'],input_values_and_types_verified=verified,raw_matches_http=raw_verified))
        budgets.append(dict(role=want['agent'],input=len(req['prompt_token_ids']),output_reserved=req['max_tokens'],total=budget))
    # Decision doesn't expose its history-trim counter; compare its built input
    # against the actual before-state's existing 20-message window explicitly.
    for row in turns:
        for trace in row['traces']['decision']:
            if trace.get('model_input') and trace['model_input']['recent_history']!=row['before']['history'][-20:]:
                trim.append(dict(id=f"{row['episode']}/{row['turn']}/decision",reason='history differs from before-state window'))
    write(run/'TRANSPORT_TOKEN_AUDIT.json',dict(count=len(proof),rows=proof,input_verified=sum(x['input_values_and_types_verified'] for x in proof),
        raw_verified=sum(x['raw_matches_http'] for x in proof),additional_history_trim=trim,
        max_input_by_role={role:max(x['input'] for x in budgets if x['role']==role) for role in {x['role'] for x in budgets}},
        max_input_plus_output=max(x['total'] for x in budgets),all_requests_fit_16384=True,requests_sha256=sha(run/'REQUESTS.jsonl'),
        expected_input_source='frozen independent input or actual runtime before-state trace',reused_independent_excluded=True))
    grouped=collections.defaultdict(list)
    for row in turns:grouped[row['episode']].append(row)
    summary=dict(run=run.name,independent=read(run/'RESULT.json')['individual'],observed_turns=len(turns),planned_turns=sum(x['planned'] for x in read(run/'RESULT.json')['episodes']),
        state_plan_pass=sum(x['state_plan_passed'] for x in turns),contract_pass=sum(x['contract_passed'] for x in turns),
        episodes=[dict(name=name,observed=len(items),state_plan_pass=all(x['state_plan_passed'] for x in items),contract_pass=all(x['contract_passed'] for x in items),
            first_state_failure=next((x['turn'] for x in items if not x['state_plan_passed']),None),
            first_contract_failure=next((x['turn'] for x in items if not x['contract_passed']),None),
            later_state_matches=[x['turn'] for x in items if x['prior_state_failure'] and x['state_plan_passed']]) for name,items in grouped.items()],
        sources=dict(collections.Counter(x['reply_evidence']['source'] for x in turns)),
        failed_turns=[dict(episode=x['episode'],turn=x['turn'],user_text=x['user_text'],reply=x['reply'],
            checks={k:v for k,v in {**x['checks'],**x['extra_checks']}.items() if not v},decision=[t.get('raw',t.get('error')) for t in x['traces']['decision']],
            actual_plans=x['actual_plans'],expected_plans=x['expected_plans'],prior_state_failure=x['prior_state_failure']) for x in turns if not x['contract_passed']],
        token_audit='TRANSPORT_TOKEN_AUDIT.json',manual_language_review='PENDING')
    times=[x['seconds'] for x in turns];summary['turn_seconds']={'mean':statistics.mean(times),'median':statistics.median(times),'min':min(times),'max':max(times)}
    write(run/'SUMMARY.json',summary)
    out=[f'# {run.name} 실제 대화 원문','', '**D6/Q6/Pbest150 별도 LoRA. 실제 생성 history 누적, 실패 후에도 실제 상태로 계속. 실물 ROS 없음.**',
         f"관측 {len(turns)}/{summary['planned_turns']}턴; 상태/계획 {summary['state_plan_pass']}, route/후보/취향/clarify 포함 {summary['contract_pass']} 자동통과. 자연어 의미 검수는 별도다.",'']
    for name,items in grouped.items():
        out += [f'## {name}','', '| 턴 | 사용자 | 실제 답변 | 출처 | 상태/계약 |','|---|---|---|---|---|']
        for r in items:out.append('| '+' | '.join([str(r['turn']),clean(r['user_text']),clean(r['reply']),r['reply_evidence']['source'],f"{r['state_plan_passed']}/{r['contract_passed']}"])+' |')
        out+=['','### Decision·계획·누적 오류','']
        for r in items:
            out.append(f"- {r['turn']}: D `{clean(' / '.join(t.get('raw',str(t.get('error'))) for t in r['traces']['decision']))}`; 실제 계획 `{json.dumps(r['actual_plans'],ensure_ascii=False)}`; 기대 `{json.dumps(r['expected_plans'],ensure_ascii=False)}`; 앞선 상태 오류 {r['prior_state_failure']}")
        out.append('')
    (run/'OUTPUTS.md').write_text('\n'.join(out)+'\n')
    print(json.dumps({k:summary[k] for k in ('run','observed_turns','state_plan_pass','contract_pass','sources')},ensure_ascii=False))

def comparison():
    completed=[]
    for p in sorted((BASE/'runs').glob('*/RESULT.json')):
        if not (p.parent/'SUMMARY.json').exists():audit(p.parent)
        completed.append(read(p.parent/'SUMMARY.json'))
    write(BASE/'COMPARISON.json',dict(status='IN_PROGRESS_UNTIL_MANUAL_REVIEW',runs=completed))
    if (BASE/'runs/patched_fixed/INDIVIDUAL.jsonl').exists() and (BASE/'runs/patched_fixed/RESULT.json').exists():
        left={r['id']:r for r in rows(BASE/'runs/baseline_fixed/INDIVIDUAL.jsonl')};right={r['id']:r for r in rows(BASE/'runs/patched_fixed/INDIVIDUAL.jsonl')}
        cases={c['record']['id']:c['record'] for c in read(BASE/'inputs/CASES.json')['rows']}
        changes=[]
        for key,a in left.items():
            b=right[key]
            if a['prediction']!=b['prediction'] or key.startswith('PT-'):
                changes.append(dict(id=key,input=cases[key]['input'],target=cases[key]['target'],baseline=a['prediction'],patched=b['prediction'],baseline_pass=a['grade']['automated_pass'],patched_pass=b['grade']['automated_pass']))
        write(BASE/'INDEPENDENT_COMPARISON.json',dict(rows=changes,strict_baseline=sum(r['grade']['automated_pass'] for r in left.values()),strict_patched=sum(r['grade']['automated_pass'] for r in right.values()),count=len(left)))

if __name__=='__main__':comparison()
