"""Exploratory actual-model follow-through of an observed restriction omission."""
import copy
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

import dqp_prompt_trial as trial
from soomac_irc.agent_contract import new_turn_state
from soomac_irc.decision_model import make_call_decision
from soomac_irc.llm_langgraph import build_graph

BASE = trial.BASE / 'failure_continuation'
MESSAGES = [
    '매콤한 건 힘들어. 부담 없는 조합으로 추천해 주겠어?',
    '추천은 아직 고르지 말고, 페퍼론치노만 적게 선택해줘.',
    '아직 담지 말고, 지금 매운맛 제한이 저장돼 있는지 확인해줘.',
    '매운맛은 아예 못 먹어. 페퍼론치노는 빼고 치즈만 적게 골라줘.',
    '페퍼론치노도 적게 추가해줘.',
    '치즈만 지금 바로 담아줘.',
]


def run():
    trial.verify()
    trial.previous.assert_eval_port_free()
    from eval_fixed_history import acquire_lock
    lock, baseline = acquire_lock()
    process = None
    unit = f'soomac-failure-continuation-{os.getpid()}'
    BASE.mkdir(exist_ok=True)
    try:
        failed_path = trial.BASE/'runs/baseline_model/TURNS.jsonl'
        failed = next(r for r in map(json.loads,failed_path.read_text().splitlines())
                      if r['episode']=='unresolved_reference_control' and r['turn']==6)
        assert not failed['policy']['execute']
        scenarios = [dict(name='spice_omission',messages=MESSAGES),
                     dict(name='missed_execution',messages=[failed['user_text'],
                         '아까 말한 대로 양파만 지금 바로 담아줘.',
                         '방금 버섯도 같이 담은 거야?'],
                         session=failed['before'],robot=failed['robot_before'])]
        # Freeze this exploratory sequence before seeing any of its outputs.
        trial.b.write_new(BASE / 'PLAN.json', dict(
            scenarios=scenarios, source_failed_turns_sha256=trial.b.sha(failed_path),
            prompts=['baseline', 'patched'],
            actual_ros=False, correct_state_injection=False, exploratory=True,
            purpose='Trace missing restriction through selection, explicit correction, re-challenge and execution.',
            source_sha256=trial.b.sha(Path(__file__)), parent_freeze=trial.b.sha(trial.BASE/'inputs/FREEZE.json')))
        captured = trial.serving.launcher_contract(trial.BASE / 'adapters')
        argv = captured['original_argv']
        for flag in ('--max-model-len', '--max-num-batched-tokens'):
            argv[argv.index(flag)+1] = str(trial.MAX_LEN)
        argv[argv.index('--gpu-memory-utilization')+1] = str(trial.GPU_UTIL)
        trial.b.write_new(BASE/'LAUNCH.json', dict(argv=argv, unit=unit, gpu_baseline=baseline))
        with (BASE/'SERVER.log').open('x') as log:
            process = subprocess.Popen(
                ['systemd-run','--user','--scope',f'--unit={unit}','-p','MemoryMax=40G','-p','MemorySwapMax=0','--',*argv],
                cwd=trial.serving.ROOT, env=captured['env'], stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic()+600
            while True:
                if process.poll() is not None:
                    raise RuntimeError('Diagnostic server exited')
                try:
                    with urllib.request.urlopen(trial.serving.ENDPOINT+'/health', timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                if time.monotonic()>deadline:
                    raise TimeoutError('Diagnostic server startup >600s')
                time.sleep(2)
            for variant in ('baseline','patched'):
                folder = BASE/variant
                folder.mkdir()
                vc, model, processor, calls = trial.client_objects()
                calls['decision'] = make_call_decision(model,processor,adapter_name='decision',system_prompt=trial.prompt(variant))
                models = vc.served_models()
                assert next(m for m in models if m['id']=='gemma4-base')['max_model_len']==trial.MAX_LEN
                selected=trial.b.read(trial.e.BASE/'SELECTION.json')['recommended']
                for role, row in selected.items():
                    assert Path(next(m for m in models if m['id']==role)['root']).resolve()==Path(row['path']).resolve()
                trial.b.write_new(folder/'SERVED_MODELS.json', models)
                original = vc.complete_tokens
                with (folder/'REQUESTS.jsonl').open('x') as requests, (folder/'TURNS.jsonl').open('x') as turns:
                    def checked(*args, **kwargs):
                        request = trial.previous.request_record(*args, **kwargs)
                        assert len(request['prompt_token_ids'])+request['max_tokens']<=trial.MAX_LEN
                        response = original(*args, **kwargs)
                        requests.write(json.dumps(dict(request=request,response=response),ensure_ascii=False)+'\n')
                        requests.flush()
                        return response
                    vc.complete_tokens = checked
                    try:
                        graph = build_graph(calls['decision'], calls['recommendation'], calls['response'], execution_reply_mode='fixed')
                        for scenario in scenarios:
                            session, robot = ((copy.deepcopy(scenario['session']),copy.deepcopy(scenario['robot']))
                                              if 'session' in scenario else trial.e.initial_state('extra'))
                            for n, message in enumerate(scenario['messages'], 1):
                                before = copy.deepcopy(session)
                                for call in calls.values():
                                    call.trace_events.clear()
                                result = graph.invoke(new_turn_state(session,message,robot))
                                session = copy.deepcopy(result['session'])
                                plans = trial.e.plans_for(session['order'],robot['section']) if result['policy']['execute'] else []
                                row = dict(episode=scenario['name'],turn=n,user_text=message,before=before,after=session,robot=copy.deepcopy(robot),
                                    decision=result['decision'],policy=result['policy'],reply=result['reply'],actual_plans=plans,
                                    traces={role:copy.deepcopy(call.trace_events) for role,call in calls.items()},actual_ros=False)
                                turns.write(json.dumps(row,ensure_ascii=False)+'\n')
                                turns.flush()
                                print(json.dumps(dict(variant=variant,episode=scenario['name'],turn=n,reply=result['reply'],order=session['order'],plans=plans),ensure_ascii=False),flush=True)
                                if result['policy']['execute']:
                                    trial.e.finish_fixture(session,robot,plans)
                    finally:
                        vc.complete_tokens = original
                trial.b.write_new(folder/'RESULT.json',dict(observed=sum(len(s['messages']) for s in scenarios),exploratory=True,manual_review='PENDING'))
        trial.verify()
    except BaseException as exc:
        trial.b.write_new(BASE/'FAILURE.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
    finally:
        try:
            if process:
                stop=subprocess.run(['systemctl','--user','stop',unit+'.scope'],capture_output=True,text=True,timeout=20)
                process.wait(timeout=30)
                trial.b.write_new(BASE/'CLEANUP.json',dict(unit=unit,stop_returncode=stop.returncode,process_returncode=process.returncode))
        finally:
            os.close(lock)


if __name__ == '__main__':
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    run()
