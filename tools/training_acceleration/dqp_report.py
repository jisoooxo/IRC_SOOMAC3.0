"""Render recorded results without re-running inference or rewriting raw logs."""
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'training/learning_first_v43/final/goals/GL10/dqp_adoption_20261105'


def readlines(path):return [json.loads(x) for x in path.read_text().splitlines()]
def clean(value):return str(value).replace('|','\\|').replace('\n',' ')

def render_run(run):
    result=json.loads((run/'RESULT.json').read_text())
    plan=json.loads((run/'PLAN.json').read_text())
    text=[f'# {run.name} — 실제 추론 기록','',
          '**각 LoRA는 독립적이며 모델 병합·운영 교체·실물 ROS 실행은 없다.**',
          '자동 상태/계획 점수와 자연어 의미 검수를 구분한다. 실패 뒤 미도달 턴은 통과가 아니다.','',
          '| 역할 | 후보 | 원본 SHA256 |','|---|---|---|']
    for role,entry in plan['selection'].items():
        c=entry['candidate'];text.append(f"| {role} | {c['name']} | `{c['weights_sha256']}` |")
    text+=['','## 독립 입력 평가','', '| 역할 | 자동 통과/측정 |','|---|---|']
    for role,value in result['individual']['by_agent'].items():text.append(f"| {role} | {value['automated_pass']}/{value['count']} |")
    text+=['','D 점수는 전체 의미 필드의 엄격 비교이며 유효한 동등 표현도 불일치일 수 있다. Q/P 자동 통과만으로 의미 전수 통과를 뜻하지 않는다.','',
           '## 실제 D/Q/P 연결','', '| 에피소드 | 관측/계획 | 상태·계획 |','|---|---|---|']
    for ep in result['episodes']:
        text.append(f"| {ep['episode']} | {ep.get('observed_turns',ep.get('observed'))}/{ep.get('planned_turns',ep.get('planned'))} | {'PASS' if ep['passed'] else 'FAIL'} |")
    text+=['','## 실제 대화 원문','']
    audit_path=run/'FIELD_AUDIT.json'
    if audit_path.exists():
        audit=json.loads(audit_path.read_text())
        text+=['**추가 필드 감사:** 아래 원래 PASS는 상태/계획 검사만의 결과이다. Q 호출·후보 저장·모호성 상태까지 확인한 판정은 FIELD_AUDIT.json을 따른다.',
               f"원래 상태/계획 통과 {audit['original_state_plan_pass']}/{audit['observed']}; 추가 필드 포함 {audit['field_complete_pass']}/{audit['observed']}.",'']
        for item in audit['added_failures']:
            text.append(f"- {item['episode']}/{item['turn']}: {', '.join(item['missing_or_wrong_fields'])}")
        text.append('')
    for row in readlines(run/'TURNS.jsonl'):
        text += [f"### {row['episode']} / {row['turn']}",f"사용자: {row['user_text']}",
                 f"답변 [{row['reply_evidence']['source']}]: {row['reply']}",
                 'Decision raw:', '```json', '\n'.join(t.get('raw',json.dumps(t.get('error'),ensure_ascii=False)) for t in row['traces']['decision']), '```',
                 f"실행 계획: `{json.dumps(row['actual_plans'],ensure_ascii=False)}`",f"기대 계획: `{json.dumps(row['expected_plans'],ensure_ascii=False)}`",
                 f"상태 검사: {'PASS' if row['passed'] else 'FAIL'}; 실패 항목: {', '.join(k for k,v in row['checks'].items() if not v) or '없음'}",'']
    text+=['## 독립 P/Q 원문','', '| ID | 실제 출력 | 자동 검사 |','|---|---|---|']
    individual=readlines(run/'INDIVIDUAL.jsonl')
    for row in individual:
        if row['agent']!='decision':
            text.append(f"| {row['id']} | {clean(json.dumps(row['prediction'],ensure_ascii=False))} | {'PASS' if row['grade']['automated_pass'] else 'FAIL'} |")
    text+=['','## Decision 단독 순차 시나리오','',
           '이 구간만 Q/P는 이전에 검수한 고정 fixture이다. Decision은 실제 GPU 추론이며 gold D를 주입하지 않았다. 아래 답변을 이번 P 모델 생성으로 읽지 않는다.','']
    for ep in result.get('decision_only_episodes',[]):
        text.append(f"- {ep['episode']}: {'PASS' if ep['passed'] else 'FAIL'}, 관측{ep['observed_turns']}/{ep['planned_turns']}턴")
    if (run/'D_ONLY_TURNS.jsonl').exists():
        text+=['','| 턴 | 사용자 | D raw | 상태/계획 |','|---|---|---|---|']
        for row in readlines(run/'D_ONLY_TURNS.jsonl'):
            raw=' / '.join(t.get('raw','') for t in row['traces']['decision'])
            text.append(f"| {row['episode']}/{row['turn']} | {clean(row['user_text'])} | {clean(raw)} | {'PASS' if row['passed'] else 'FAIL'} |")
    (run/'OUTPUTS.md').write_text('\n'.join(text)+'\n')


def summarize():
    records=[]
    for path in sorted((BASE/'runs').glob('*/RESULT.json')):
        run=path.parent;render_run(run)
        data=json.loads(path.read_text());plan=json.loads((run/'PLAN.json').read_text());turns=readlines(run/'TURNS.jsonl')
        ind=readlines(run/'INDIVIDUAL.jsonl');metrics={}
        for role in ('decision','recommendation','response'):
            times=[r['seconds'] for r in ind if r['agent']==role and not r.get('error')]
            if times:metrics[role]=dict(mean=statistics.mean(times),median=statistics.median(times),minimum=min(times),maximum=max(times),count=len(times))
        records.append(dict(name=run.name,selection={role:row['candidate']['name'] for role,row in plan['selection'].items()},
            independent=data['individual']['by_agent'],episodes=data['episodes'],decision_only=data.get('decision_only_episodes'),
            latency_seconds=metrics,observed_turns=len(turns),failed_turns=[dict(episode=r['episode'],turn=r['turn'],
            user_text=r['user_text'],reply=r['reply'],expected_plans=r['expected_plans'],actual_plans=r['actual_plans'],
            decision_raw=[t.get('raw') for t in r['traces']['decision']],checks=r['checks']) for r in turns if not r['passed']],
            route_mismatches=[dict(episode=r['episode'],turn=r['turn'],expected=r['route_expected'],actual=r['decision_final']['route']) for r in turns if r.get('route_matches') is False],
            transport_verified=(run/'TRANSPORT_PROOF.json').exists(),cleanup=json.loads((run/'CLEANUP.json').read_text()) if (run/'CLEANUP.json').exists() else None))
        if (run/'FIELD_AUDIT.json').exists():
            audit=json.loads((run/'FIELD_AUDIT.json').read_text())
            records[-1]['field_audit']={k:audit[k] for k in ('observed','field_complete_pass','added_failures','route_mismatches')}
    (BASE/'COMPARISON_PROGRESS.json').write_text(json.dumps(dict(status='IN_PROGRESS_NOT_FINAL_SELECTION',runs=records),ensure_ascii=False,indent=2))
    print(json.dumps(dict(rendered=[r['name'] for r in records]),ensure_ascii=False))


if __name__=='__main__':summarize()
