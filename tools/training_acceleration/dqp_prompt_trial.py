"""Evaluation-only prompt x execution-speech trial, using separate D6/Q6/P150 LoRAs."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
from unittest.mock import patch

import dqp_evaluation as e
import dqp_run as previous
from dqp_evaluation import b, serving, probe
from soomac_irc.agent_contract import new_turn_state, normalize_decision
from soomac_irc.decision_model import make_call_decision, build_decision_inputs, build_decision_model_input
from soomac_irc.agent_prompts import DECISION_SYSTEM
from soomac_irc.llm_langgraph import build_graph

BASE=e.BASE.parent/'dqp_prompt_multiturn_20261105'
MAX_LEN=16384
GPU_UTIL=0.58
# Restore the initial full addendum; no production prompt file is changed.
_initial_prompt=(BASE/'inputs_initial_over_budget/patched.txt').read_text()
assert _initial_prompt.startswith(DECISION_SYSTEM)
ADDENDUM=_initial_prompt[len(DECISION_SYSTEM):]


def client_objects():
    import soomac_irc.input_budget as budget
    import soomac_irc.decision_model as dm
    import soomac_irc.response_model as pm
    import soomac_irc.recommendation_model as qm
    budget.SERVING_MAX_MODEL_LEN=MAX_LEN
    dm.DECISION_INPUT_MAX_TOKENS=MAX_LEN-dm.DECISION_MAX_TOKENS
    qm.RECOMMENDATION_INPUT_MAX_TOKENS=MAX_LEN-qm.RECOMMENDATION_MAX_TOKENS
    pm.RESPONSE_INPUT_MAX_TOKENS=MAX_LEN-pm.RESPONSE_MAX_TOKENS
    vc,model,processor,calls=serving.client_objects()
    model.max_model_len=MAX_LEN
    return vc,model,processor,calls


def t(text, decision=None, changes=None, *, route='task', execute=False, review='', **checks):
    return dict(text=text,decision={'route':route,**(decision or {})},changes=changes or {},
                execute=execute,review=review,**checks)

def specs():
    rows=e.extended_specs()
    short=rows[0]
    short['name']='accepted_short_then_explicit_execution'
    short['turns'][1].update(decision={'route':'task','confirmation':'accept'},execute=False,expected_plans=[])
    short['turns'].insert(2,t('그럼 면은 지금 바로 담아줘.',{'commit':True},execute=True))
    first=copy.deepcopy(short['turns'][0])
    rows += [dict(name='explicit_user_phrase',section='noodle',group='new',turns=[copy.deepcopy(first),
                t('그걸로 하고 면은 바로 담아줘.',{'confirmation':'accept','commit':True},{'sauce':'크림'},execute=True)]),
             dict(name='explicit_execution_paraphrase',section='noodle',group='new',turns=[copy.deepcopy(first),
                t('추천한 소스로 정하고 지금 얇은면을 담기 시작해 줘.',{'confirmation':'accept','commit':True},{'sauce':'크림'},execute=True)])]
    rows.append(dict(name='mood_then_food_then_no_execution',section='noodle',group='new',turns=[
        t('오늘 회사 일이 많아서 너무 지쳤어.',route='general',preferences_unchanged=True),
        t('친구하고 다퉈서 기분도 별로야.',route='general',preferences_unchanged=True),
        t('얇은면 보통으로 고르고 느끼한 소스를 하나 추천해 줘.',
          {'order':{'noodle_type':'얇은면','noodle_portion':'normal'},'recommendation':{'action':'request'}},
          {'noodle_type':'얇은면','noodle_portion':'normal'}),
        t('소스는 그 후보로 정해 두고 면은 아직 담지 마.',{'confirmation':'accept'},{'sauce':'크림'}),
        t('지금 면이 이미 담긴 건 아니지?',review='아직 선택만 한 사실. 시작/완료 주장 금지'),
        t('이제 그 면을 보통 양으로 바로 담아줘.',{'commit':True},execute=True)]))
    rows.append(dict(name='spicy_limit_then_block',section='extra',group='new',turns=[
        t('매콤한 건 힘들어. 매운맛은 못 먹겠어.',
          {'restrictions':[{'action':'add','target':'매운맛','reason':'cannot_eat'}]},
          {'restrictions':[{'target':'매운맛','reason':'cannot_eat'}]}),
        t('치즈만 적게 선택해 줘.',{'order':{'toppings':{'치즈':'low'}}},{'toppings':{'치즈':'low'}}),
        t('페퍼론치노도 조금 추가해줘.',{'order':{'toppings':{'페퍼론치노':'low'}}},review='cannot_eat로 추가 차단'),
        t('그럼 치즈만 지금 바로 담아줘.',{'commit':True},execute=True)]))
    oldcases={c['record']['id']:c['record']['target'] for c in b.read(e.BASE/'inputs/CASES.json')['rows'] if c['agent']=='decision'}
    core=[]
    for compound in (True,False):
        name='compound' if compound else 'split_control';order=e.initial_state('noodle')[0]['order'];turns=[]
        for n,(section,tag,text,execute) in enumerate(probe.scenario(compound),1):
            after=probe.expected_after(order,tag);changes={}
            for key,value in after.items():
                if key=='toppings':
                    delta={k:after[key].get(k) for k in set(order[key])|set(after[key]) if order[key].get(k)!=after[key].get(k)}
                    if delta:changes[key]=delta
                elif value!=order[key]:changes[key]=copy.deepcopy(value)
            turns.append(t(text,copy.deepcopy(oldcases[f'PRIOR-{name}-{n}-D']),changes,execute=execute))
            order=after
        core.append(dict(name=name,section='noodle',group='core_regression',turns=turns))
    rows=core+rows
    for row in rows:
        for turn in row['turns']:
            turn['expected_q']=turn['decision'].get('recommendation',{}).get('action') in ('request','revise')
            if turn['expected_q']:turn['expected_candidate']='크림'
            if turn['decision'].get('clarify'):turn['expected_clarify']=True
    return rows

def new_cases():
    pairs=[
      ('greeting','안녕, 처음 와서 조금 어색하네.',{'route':'general'}),
      ('greeting','만나서 반가워. 오늘 잘 부탁해.',{'route':'general'}),
      ('role','넌 여기서 사람들한테 어떤 도움을 주는 로봇이야?',{'route':'general'}),
      ('mood','오늘 회사 일이 많아서 너무 지쳤어.',{'route':'general'}),
      ('mood','친구하고 다퉈서 기분도 별로야.',{'route':'general'}),
      ('mood','시험을 망쳐서 마음이 무거워.',{'route':'general'}),
      ('food_preference','오늘은 담백한 맛으로 먹고 싶어.',{'route':'task','preferences':[{'action':'add','value':'담백하게'}]}),
      ('food_preference','느끼한 소스를 좋아해.',{'route':'task','preferences':[{'action':'add','value':'느끼하게'}]}),
      ('soft_spice','오늘은 덜 맵게 먹고 싶어.',{'route':'task','preferences':[{'action':'add','value':'맵지 않게'}]}),
      ('soft_spice','매운 걸 못 먹는 건 아니야. 이번에는 담백하게 먹을래.',{'route':'task','preferences':[{'action':'add','value':'담백하게'}]}),
      ('hard_spice','매운 음식을 먹기 힘들어. 순한 소스로 추천해줘.',{'route':'task','restrictions':[{'action':'add','target':'매운맛','reason':'cannot_eat'}],'recommendation':{'action':'request'}}),
      ('hard_spice','매콤한 건 힘들어. 부담 없는 조합으로 추천해 주겠어?',{'route':'task','preferences':[{'action':'add','value':'맵지 않게'}],'restrictions':[{'action':'add','target':'매운맛','reason':'cannot_eat'}],'recommendation':{'action':'request'}}),
      ('hard_spice','나는 매운맛을 못 먹어.',{'route':'task','restrictions':[{'action':'add','target':'매운맛','reason':'cannot_eat'}]}),
      ('mixed','면은 얇은 걸로 적게 골라줘. 그런데 로봇도 잠을 자?',{'route':'mixed','order':{'noodle_type':'얇은면','noodle_portion':'low'}}),
      ('mixed','얇은면 보통으로 골라줘. 그리고 파스타는 어느 나라 요리야?',{'route':'mixed','order':{'noodle_type':'얇은면','noodle_portion':'normal'}}),
      ('state_question','지금까지 내가 고른 면이 있어?',{'route':'task'}),
      ('accept_only','그걸로 담아줘.',{'route':'task','confirmation':'accept'}),
      ('accept_execute','그걸로 하고 면은 바로 담아줘.',{'route':'task','confirmation':'accept','commit':True}),
      ('accept_wait','소스는 그걸로 하고, 면은 아직 담지 마.',{'route':'task','confirmation':'accept'}),
      ('execution_question','그걸로 정하면 면도 바로 담기 시작하는 거야?',{'route':'task'}),
    ]
    rows=[]
    for n,(group,text,target) in enumerate(pairs,1):
        session,robot=e.initial_state('noodle')
        if n>=17:
            session['order'].update(noodle_type='얇은면',noodle_portion='normal')
            session['pending']={'type':'recommendation','candidate':{'sauce':'크림','noodle_type':None,'noodle_portion':None,'toppings':{}},'reason_tags':['느끼한 소스 요청']}
            session['history'] += [{'role':'user','content':'면은 얇은면 보통으로 고르고 느끼한 소스를 추천해 줘.'},
                                   {'role':'assistant','content':'느끼한 소스로 크림을 추천드려요. 크림으로 고를까요?'}]
        rows.append(dict(agent='decision',group='new_'+group,record=dict(id=f'PT-{n:02}',input=build_decision_model_input(session,text,robot),target=target)))
    return rows

def prompt(kind):
    return (BASE/'inputs'/f'{kind}.txt').read_text()

def verify():
    f=b.read(BASE/'inputs/FREEZE.json')
    serving.check_pins(f['source_sha256']);serving.check_pins(f['processor_sha256'])
    for filename,sha in f['files'].items():assert b.sha(BASE/'inputs'/filename)==sha,filename
    return f

def cpu_episode(spec,mode):
    cursor=[0]
    def d(session,text,robot,repair=None):
        if repair:raise RuntimeError('Oracle repair: '+str(repair))
        case=spec['turns'][cursor[0]];cursor[0]+=1
        return normalize_decision(case['decision'])
    def q(*args):return {'proposal':{'sauce':'크림','noodle_type':None,'noodle_portion':None,'toppings':{}},'reason_tags':['느끼한 소스 요청']}
    def p(*args):return 'CPU fixture only'
    for f in (d,q,p):f.trace_events=[]
    import io
    return episode(spec,dict(decision=d,recommendation=q,response=p),io.StringIO(),mode,cpu=True)

def prepare():
    out=BASE/'inputs';out.mkdir()
    (out/'baseline.txt').write_text(DECISION_SYSTEM)
    (out/'patched.txt').write_text(DECISION_SYSTEM+ADDENDUM)
    old=[copy.deepcopy(c) for c in b.read(e.BASE/'inputs/CASES.json')['rows'] if c['agent']=='decision']
    for c in old:
        if c['record']['id']=='PRIOR-recommendation_and_short_execution-2-D':
            c['record']['target']={'route':'task','confirmation':'accept'}
            c['user_adjudication']='2026-11-05 accept sauce then ask noodle execution is valid'
    cases=old+new_cases();episodes=specs()
    b.write_new(out/'CASES.json',{'rows':cases});b.write_new(out/'EPISODES.json',{'extended':episodes,'core_required_turns':17})
    vc,model,processor,calls=client_objects()
    from soomac_irc.input_budget import render_inputs,check_static_budget
    tokens=[]
    for kind in ('baseline','patched'):
        system=prompt(kind);check_static_budget(processor,system,MAX_LEN-1024,'Decision/'+kind)
        static=render_inputs(processor,system,{})[1]['input_ids'].shape[1]
        for case in cases:
            session,text,robot=*e.args_for(case),
            messages,inputs,actual=build_decision_inputs(session,text,robot,processor,system_prompt=system)
            assert actual==case['record']['input'],case['record']['id']+' added truncation'
            tokens.append(dict(variant=kind,id=case['record']['id'],prompt=inputs['input_ids'].shape[1],reserve=1024,static=static,history_trimmed=0))
        session,robot=e.initial_state('noodle')
        session['history']=[{'role':'user' if i%2==0 else 'assistant','content':'지금 선택된 재료와 양을 확인해 주세요.' if i%2==0 else '현재 선택한 면과 소스, 재료의 양을 그대로 유지하고 있어요.'} for i in range(20)]
        repair={'previous_output':{'route':'task','confirmation':'accept'},'errors':['latest question is clarification; do not accept an unasked question']}
        for label,r in [('history20',None),('history20_repair',repair)]:
            messages,inputs,actual=build_decision_inputs(session,'지금 선택한 내용을 확인해 줘.',robot,processor,r,system_prompt=system)
            assert len(actual['recent_history'])==20,kind+' '+label
            tokens.append(dict(variant=kind,id=label,prompt=inputs['input_ids'].shape[1],reserve=1024,static=static,history_trimmed=0))
    import torch
    assert not torch.cuda.is_initialized()
    cpu=[cpu_episode(s,m) for s in episodes for m in ('fixed','model')]
    assert all(r['state_plan_passed'] for r in cpu),[r for r in cpu if not r['state_plan_passed']]
    b.write_new(out/'CPU_PREFLIGHT.json',dict(token_rows=tokens,cpu_episodes=cpu,cuda_initialized=False,max_model_len=MAX_LEN))
    oldf=b.read(e.BASE/'inputs/FREEZE.json')
    sources=serving.source_pins();sources[str(Path(__file__).resolve())]=b.sha(Path(__file__))
    sources[str(Path(e.__file__))]=b.sha(Path(e.__file__))
    b.write_new(out/'FREEZE.json',dict(files={p.name:b.sha(p) for p in out.iterdir()},source_sha256=sources,processor_sha256=oldf['processor_sha256'],
        created_before_trial_outputs=True,selection_exposed_not_pristine_holdout=True,operational_prompts_changed=False))
    serving.emit('prepared',cases=len(cases),episodes=len(episodes),planned_turns=sum(len(x['turns']) for x in episodes),
        tokens={k:max(x['prompt'] for x in tokens if x['variant']==k) for k in ('baseline','patched')})

def episode(spec,calls,stream,mode,cpu=False):
    session,robot=e.initial_state(spec['section']);expected=copy.deepcopy(session['order'])
    graph=build_graph(calls['decision'],calls['recommendation'],calls['response'],execution_reply_mode=mode)
    rows=[];expected_section=spec['section'];prior_state_failure=None;prior_contract_failure=None
    for n,case in enumerate(spec['turns'],1):
        before,rb=copy.deepcopy(session),copy.deepcopy(robot);e.apply_expected(expected,case['changes'])
        expected_plans=case.get('expected_plans')
        if expected_plans is None:expected_plans=e.plans_for(expected,expected_section) if case['execute'] else []
        for call in calls.values():call.trace_events.clear()
        started=time.monotonic();result=graph.invoke(new_turn_state(session,case['text'],robot));session=copy.deepcopy(result['session'])
        traces={role:copy.deepcopy(call.trace_events) for role,call in calls.items()}
        plans=e.plans_for(session['order'],robot['section']) if result['policy']['execute'] else []
        checks=dict(order_matches=session['order']==expected,section_matches=robot['section']==expected_section,plans_match=plans==expected_plans,execute_matches=result['policy']['execute']==case['execute'],
            history_prefix=session['history'][:len(before['history'])]==before['history'],
            history_pair=session['history'][-2:]==[{'role':'user','content':case['text']},{'role':'assistant','content':result['reply']}],
            no_model_errors=not any(x.get('error') for group in traces.values() for x in group))
        extra=dict(route_matches=result['decision']['route']==case['decision']['route'])
        if case.get('preferences_unchanged') or case['decision']['route']=='general':extra['preferences_preserved']=session['preferences']==before['preferences']
        if not cpu:
            extra['Q_called_as_requested']=bool(traces['recommendation'])==case['expected_q']
            if case['expected_q']:
                pending=session.get('pending') or {};extra['candidate_saved']=pending.get('type')=='recommendation' and (pending.get('candidate') or {}).get('sauce')==case['expected_candidate'] and session['order']['sauce'] is None
        if case.get('expected_clarify'):
            extra['clarify_resolved']=result['decision']['understanding']=='clarify'
            if (before.get('pending') or {}).get('type')=='execution':extra['clarify_pending']=(session.get('pending') or {}).get('question_type')=='clarification'
        row=dict(episode=spec['name'],turn=n,user_text=case['text'],before=before,after=copy.deepcopy(session),robot_before=rb,
            decision_final=result['decision'],policy=result['policy'],reply=result['reply'],traces=traces,checks=checks,extra_checks=extra,
            expected_order=copy.deepcopy(expected),expected_section=expected_section,prior_state_failure=prior_state_failure,prior_contract_failure=prior_contract_failure,expected_plans=expected_plans,actual_plans=plans,route_expected=case['decision']['route'],
            state_plan_passed=all(checks.values()),contract_passed=all(checks.values()) and all(extra.values()),
            seconds=time.monotonic()-started,execution_reply_mode=mode,actual_ros=False,physical_simulation_only=True,
            reply_evidence=probe.response_evidence(traces['response'],result['reply'],spec['name'],str(n)))
        stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush();rows.append(row)
        if not cpu:serving.emit('turn',mode=mode,episode=spec['name'],turn=n,state=row['state_plan_passed'],contract=row['contract_passed'],reply=row['reply'])
        if not row['state_plan_passed'] and prior_state_failure is None:prior_state_failure=n
        if not row['contract_passed'] and prior_contract_failure is None:prior_contract_failure=n
        # Keep actual wrong state and continue; never overwrite it with the oracle.
        if result['policy']['execute']:e.finish_fixture(session,robot,plans)
        if case['execute']:
            expected_section=dict(noodle='veggie',veggie='meat',meat='extra',extra='lid').get(expected_section,expected_section)
    return dict(episode=spec['name'],planned=len(spec['turns']),observed=len(rows),
        state_plan_passed=len(rows)==len(spec['turns']) and all(x['state_plan_passed'] for x in rows),
        contract_passed=len(rows)==len(spec['turns']) and all(x['contract_passed'] for x in rows))

def independent(calls,folder,variant):
    import functional_eval
    old={r['id']:r for r in (json.loads(x) for x in (e.BASE/'runs/epoch6/INDIVIDUAL.jsonl').read_text().splitlines()) if r['agent']=='decision'}
    rows=[]
    with (folder/'INDIVIDUAL.jsonl').open('x') as f:
        for c in b.read(BASE/'inputs/CASES.json')['rows']:
            ident=c['record']['id'];error=None
            if False:  # 16k server batch shape changed: remeasure baseline too, no old GPU reuse.
                original=old[ident];prediction=copy.deepcopy(original['prediction']);traces=copy.deepcopy(original['traces']);error=original.get('error');reused=True;seconds=None
                assert traces[-1]['model_input']==c['record']['input']
            else:
                call=calls['decision'];call.trace_events.clear();started=time.monotonic();prediction=None;reused=False
                try:
                    call(*e.args_for(c));prediction=copy.deepcopy(call.last_external_output)
                except Exception as exc:error=repr(exc)
                traces=copy.deepcopy(call.trace_events);seconds=time.monotonic()-started
            grade=functional_eval.grade(c,prediction) if error is None else dict(automated_pass=False,failures=[error])
            row=dict(id=ident,group=c['group'],agent='decision',prediction=prediction,traces=traces,error=error,grade=grade,seconds=seconds,
                reused_from='dqp_adoption_20261105/runs/epoch6' if reused else None)
            f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();rows.append(row)
    return dict(count=len(rows),strict_pass=sum(r['grade']['automated_pass'] for r in rows),reused=sum(bool(r['reused_from']) for r in rows))

def run():
    verify();previous.assert_eval_port_free()
    from eval_fixed_history import acquire_lock
    lock,baseline=acquire_lock();process=None
    unit=f'soomac-prompt-trial-{os.getpid()}'
    try:
        selected=b.read(e.BASE/'SELECTION.json')['recommended'];adapters=BASE/'adapters';adapters.mkdir(exist_ok=True)
        for role,row in selected.items():
            assert b.sha(Path(row['path'])/'adapter_model.safetensors')==row['weights_sha256']
            if (adapters/role).is_symlink():assert (adapters/role).resolve()==Path(row['path']).resolve()
            else:(adapters/role).symlink_to(row['path'],target_is_directory=True)
        captured=serving.launcher_contract(adapters);argv=captured['original_argv']
        for flag in ('--max-model-len','--max-num-batched-tokens'):argv[argv.index(flag)+1]=str(MAX_LEN)
        argv[argv.index('--gpu-memory-utilization')+1]=str(GPU_UTIL)
        b.write_new(BASE/'LAUNCH.json',dict(argv=argv,unit=unit,selected=selected,gpu_baseline=baseline,
            environment={k:captured['env'][k] for k in ('VLLM_BATCH_INVARIANT','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE')},merged=False))
        with (BASE/'SERVER.log').open('x') as log:
            process=subprocess.Popen(['systemd-run','--user','--scope',f'--unit={unit}','-p','MemoryMax=40G','-p','MemorySwapMax=0','--',*argv],
                cwd=serving.ROOT,env=captured['env'],stdout=log,stderr=subprocess.STDOUT)
            serving.emit('server_starting',unit=unit,pid=process.pid);deadline=time.monotonic()+600
            while True:
                if process.poll() is not None:raise RuntimeError('Server exited')
                try:
                    with urllib.request.urlopen(serving.ENDPOINT+'/health',timeout=2) as response:
                        if response.status==200:break
                except OSError:pass
                if time.monotonic()>deadline:raise TimeoutError('startup >600s')
                time.sleep(2)
            for variant,mode in [('baseline','fixed'),('patched','fixed'),('baseline','model'),('patched','model')]:
                folder=BASE/'runs'/f'{variant}_{mode}';folder.mkdir(parents=True)
                vc,model,processor,calls=client_objects()
                calls['decision']=make_call_decision(model,processor,adapter_name='decision',system_prompt=prompt(variant))
                models=vc.served_models();assert next(m for m in models if m['id']=='gemma4-base')['max_model_len']==MAX_LEN;roots={m['id']:m.get('root') for m in models}
                for role,row in selected.items():assert Path(roots[role]).resolve()==Path(row['path']).resolve()
                b.write_new(folder/'SERVED_MODELS.json',models)
                b.write_new(folder/'PLAN.json',dict(variant=variant,execution_reply_mode=mode,system_sha256=b.sha(BASE/'inputs'/f'{variant}.txt'),selection=selected))
                original=vc.complete_tokens;counts=dict.fromkeys(selected,0)
                with (folder/'REQUESTS.jsonl').open('x') as reqs,(folder/'TURNS.jsonl').open('x') as stream:
                    def checked(*args,**kwargs):
                        request=previous.request_record(*args,**kwargs)
                        assert len(request['prompt_token_ids'])+request['max_tokens']<=MAX_LEN
                        response=original(*args,**kwargs);reqs.write(json.dumps(dict(request=request,response=response),ensure_ascii=False)+'\n');reqs.flush();counts[request['model']]+=1
                        return response
                    vc.complete_tokens=checked
                    try:
                        ind=independent(calls,folder,variant) if mode=='fixed' else {'reused_from':f'{variant}_fixed'}
                        episodes=[episode(spec,calls,stream,mode) for spec in b.read(BASE/'inputs/EPISODES.json')['extended']]
                    finally:vc.complete_tokens=original
                verify();b.write_new(folder/'RESULT.json',dict(individual=ind,episodes=episodes,requests=counts,semantic_review='PENDING'))
                serving.emit('variant_done',variant=variant,mode=mode,requests=counts)
    except BaseException as exc:
        b.write_new(BASE/'FAILURE.json',dict(type=type(exc).__name__,message=str(exc)));raise
    finally:
        try:
            if process:
                stop=subprocess.run(['systemctl','--user','stop',unit+'.scope'],capture_output=True,text=True,timeout=20);process.wait(timeout=30)
                b.write_new(BASE/'CLEANUP.json',dict(unit=unit,stop_returncode=stop.returncode,process_returncode=process.returncode))
        finally:os.close(lock)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);args=p.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
    prepare() if args.action=='prepare' else run()
