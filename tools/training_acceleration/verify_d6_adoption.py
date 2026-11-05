"""Verify the adopted source configuration and production adapter paths, without ROS."""
import copy
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
import dqp_evaluation as e
import dqp_run as previous
from dqp_evaluation import b, serving, probe
from soomac_irc.agent_contract import new_turn_state
from soomac_irc.llm_langgraph import build_graph
from soomac_irc.agent_prompts import DECISION_SYSTEM
from soomac_irc.decision_model import make_call_decision
from soomac_irc.recommendation_model import make_call_recommendation
from soomac_irc.response_model import make_call_response
from soomac_irc import vllm_client as vc
from soomac_irc.model_runtime import MODEL_PATH, DECISION_ADAPTER_PATH, RECOMMENDATION_ADAPTER_PATH, RESPONSE_ADAPTER_PATH
from soomac_irc.input_budget import SERVING_MAX_MODEL_LEN

BASE=serving.ROOT/'training/learning_first_v43/final/goals/GL10/d6_prompt_adoption_20261105'
PRIOR=BASE.parent/'dqp_prompt_multiturn_20261105'

# Same continuation/checking function as the frozen prompt trial; original file unchanged.
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


def run():
    import torch
    torch.set_num_threads(2)
    assert DECISION_SYSTEM==(PRIOR/'inputs/patched.txt').read_text()
    assert SERVING_MAX_MODEL_LEN==16384
    prep=b.read(BASE/'PREPARED.json')
    serving.check_pins(prep['protected_files'])
    sources=serving.source_pins()
    sources[str(Path(__file__).resolve())]=b.sha(Path(__file__))
    b.write_new(BASE/'SOURCE_AFTER.json',sources)
    previous.assert_eval_port_free()
    from eval_fixed_history import acquire_lock
    lock,baseline=acquire_lock()
    process=None
    unit=f'soomac-d6-adoption-{os.getpid()}'
    try:
        captured=serving.launcher_contract(serving.ROOT/'models/vllm_adapters')
        argv=captured['original_argv']
        for flag in ('--max-model-len','--max-num-batched-tokens'):
            assert argv[argv.index(flag)+1]=='16384'
        assert argv[argv.index('--gpu-memory-utilization')+1]=='0.58'
        b.write_new(BASE/'LAUNCH.json',dict(argv=argv,unit=unit,gpu_baseline=baseline,
            environment={k:captured['env'][k] for k in ('VLLM_BATCH_INVARIANT','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE')}))
        with (BASE/'SERVER.log').open('x') as log:
            process=subprocess.Popen(['systemd-run','--user','--scope',f'--unit={unit}','-p','MemoryMax=40G','-p','MemorySwapMax=0','--',*argv],
                cwd=serving.ROOT,env=captured['env'],stdout=log,stderr=subprocess.STDOUT)
            deadline=time.monotonic()+600
            while True:
                if process.poll() is not None:
                    raise RuntimeError('Adoption validation server exited')
                try:
                    with urllib.request.urlopen(serving.ENDPOINT+'/health',timeout=2) as response:
                        if response.status==200:
                            break
                except OSError:
                    pass
                if time.monotonic()>deadline:
                    raise TimeoutError('Startup >600s')
                time.sleep(2)
            vc.VLLM_BASE_URL=serving.ENDPOINT
            paths=dict(decision=DECISION_ADAPTER_PATH,recommendation=RECOMMENDATION_ADAPTER_PATH,response=RESPONSE_ADAPTER_PATH)
            model,processor=vc.load_vllm_backend(MODEL_PATH,paths)
            assert model.max_model_len==16384
            calls=dict(decision=make_call_decision(model,processor,adapter_name='decision'),
                recommendation=make_call_recommendation(model,processor,adapter_name='recommendation'),
                response=make_call_response(model,processor,adapter_name='response'))
            # model_runtime imports initialize CUDA on this installation, even
            # though this HTTP client owns no torch GPU tensors or HF model.
            assert isinstance(model,vc.VllmModel) and model.device=='cpu'
            allocated=torch.cuda.memory_allocated() if torch.cuda.is_initialized() else 0
            assert allocated==0,allocated
            b.write_new(BASE/'CLIENT_MODEL.json',dict(type=type(model).__name__,device=model.device,
                cuda_initialized=torch.cuda.is_initialized(),torch_allocated_bytes=allocated))
            models=vc.served_models()
            for role in paths:
                actual=Path(next(m for m in models if m['id']==role)['root'])
                assert actual.resolve()==(serving.ROOT/'models/vllm_adapters'/role).resolve()
                wanted=prep['converted_weight_sha256'] if role=='decision' else prep['unchanged_qp'][role]['sha256']
                assert b.sha(actual/'adapter_model.safetensors')==wanted
            b.write_new(BASE/'SERVED_MODELS.json',models)
            b.write_new(BASE/'PLAN.json',dict(variant='patched',execution_reply_mode='fixed',production_adapter_paths=paths,
                exact_tested_prompt=True,source_budget_used_without_override=True,physical_simulation_only=True))
            original=vc.complete_tokens
            counts=dict.fromkeys(paths,0)
            with (BASE/'REQUESTS.jsonl').open('x') as reqs,(BASE/'TURNS.jsonl').open('x') as stream:
                def checked(*args,**kwargs):
                    request=previous.request_record(*args,**kwargs)
                    assert len(request['prompt_token_ids'])+request['max_tokens']<=SERVING_MAX_MODEL_LEN
                    response=original(*args,**kwargs)
                    reqs.write(json.dumps(dict(request=request,response=response),ensure_ascii=False)+'\n');reqs.flush()
                    counts[request['model']]+=1
                    return response
                vc.complete_tokens=checked
                try:
                    episodes=[episode(spec,calls,stream,'fixed') for spec in b.read(BASE/'inputs/EPISODES.json')['extended']]
                finally:
                    vc.complete_tokens=original
            serving.check_pins(sources);serving.check_pins(prep['protected_files'])
            b.write_new(BASE/'RESULT.json',dict(individual={'not_repeated':True},episodes=episodes,requests=counts,semantic_review='PENDING'))
    except BaseException as exc:
        b.write_new(BASE/'FAILURE.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
    finally:
        try:
            if process:
                stop=subprocess.run(['systemctl','--user','stop',unit+'.scope'],capture_output=True,text=True,timeout=20)
                process.wait(timeout=30)
                b.write_new(BASE/'CLEANUP.json',dict(unit=unit,stop_returncode=stop.returncode,process_returncode=process.returncode))
        finally:
            os.close(lock)


if __name__=='__main__':
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    run()
