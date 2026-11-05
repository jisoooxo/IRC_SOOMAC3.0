"""Add field/call/pending audit to immutable results; does not rerun inference."""
import json
import hashlib
from pathlib import Path

BASE=Path(__file__).resolve().parents[2]/'training/learning_first_v43/final/goals/GL10/dqp_adoption_20261105'


def run(folder):
    specs=json.loads((BASE/'inputs/EPISODES.json').read_text())['extended']
    expected={(s['name'],n):t['decision'] for s in specs for n,t in enumerate(s['turns'],1)}
    prior=[json.loads(s) for s in (BASE.parent/'fixed_history_vllm_20261104/TURNS.jsonl').read_text().splitlines()]
    for row in prior:
        if row['episode'] in ('compound','split_control'):
            expected[row['episode'],row['turn']]=row['traces']['decision'][0]['external']
    turns=[json.loads(s) for s in (folder/'TURNS.jsonl').read_text().splitlines()]
    rows=[]
    for turn in turns:
        oracle=expected[turn['episode'],turn['turn']]
        wanted=(oracle.get('recommendation') or {}).get('action','none')=='request'
        called=bool(turn['traces']['recommendation'])
        pending=turn['after'].get('pending') or {}
        normalized=turn['decision_final']
        checks=dict(requested_Q_was_called=(not wanted or called),no_unrequested_Q=(wanted or not called))
        if wanted:
            # All authored request turns explicitly seek creamy sauce, no execution.
            checks['recommendation_candidate_stored']=pending.get('type')=='recommendation'
            checks['cream_candidate']= (pending.get('candidate') or {}).get('sauce')=='크림'
            checks['candidate_not_committed']=turn['after']['order']['sauce'] is None
        if oracle.get('clarify'):
            checks['decision_preserves_ambiguity']=normalized['understanding']=='clarify'
            checks['old_execution_pending_not_active']=(pending.get('type')!='execution' or pending.get('question_type')=='clarification')
        rows.append(dict(episode=turn['episode'],turn=turn['turn'],user_text=turn['user_text'],
            original_state_plan_pass=turn['passed'],extra_checks=checks,
            field_complete_pass=turn['passed'] and all(checks.values()),
            route_expected=oracle['route'],route_actual=normalized['route'],route_matches=oracle['route']==normalized['route'],
            missing_or_wrong_fields=[k for k,v in checks.items() if not v]))
    summary=dict(rows=rows,observed=len(rows),original_state_plan_pass=sum(r['original_state_plan_pass'] for r in rows),
        field_complete_pass=sum(r['field_complete_pass'] for r in rows),
        added_failures=[r for r in rows if r['missing_or_wrong_fields']],
        route_mismatches=[r for r in rows if not r['route_matches']],
        turns_sha256=hashlib.sha256((folder/'TURNS.jsonl').read_bytes()).hexdigest(),
        methodology='Post-output supplemental audit. Original raw/checks unchanged; missing Q/pending/clarify checks were identified during manual review. No new model inference.',
        language_semantics='Separate manual review required; this checks intent fields and pending, not all response meaning.')
    with (folder/'FIELD_AUDIT.json').open('x') as f:json.dump(summary,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(run=folder.name,original_pass=summary['original_state_plan_pass'],field_pass=summary['field_complete_pass'],
        added_failures=[f"{r['episode']}/{r['turn']}" for r in summary['added_failures']]),ensure_ascii=False))


if __name__=='__main__':
    for p in sorted((BASE/'runs').glob('*/RESULT.json')):
        if not (p.parent/'FIELD_AUDIT.json').exists():run(p.parent)
