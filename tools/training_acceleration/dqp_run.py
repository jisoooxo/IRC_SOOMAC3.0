"""Stage independent LoRAs and evaluate via owned production-config vLLM server."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import socket
import time
import urllib.request
from unittest.mock import patch
import dqp_evaluation as e
from dqp_evaluation import b, serving, BASE, PREVIOUS


def verify():
    frozen=b.read(BASE/'inputs/FREEZE.json')
    for name,sha in frozen['files'].items():
        assert b.sha(BASE/'inputs'/name)==sha, name
    serving.check_pins(frozen['source_sha256']);serving.check_pins(frozen['processor_sha256'])
    stat=frozen['base_weight_stat'];actual=Path(stat['path']).stat()
    assert (actual.st_size,actual.st_mtime_ns)==(stat['size'],stat['mtime_ns'])
    return frozen


def stage():
    verify()
    import torch
    torch.set_num_threads(2)
    from safetensors.torch import load_file
    import convert_adapters_for_vllm as converter
    root=b.output_dir(BASE/'staged')
    archive=BASE/'archive/extracted/results/checkpoints'
    identity=b.read(archive/'INPUT_IDENTITY.json')
    for field,path in [('source_manifest_sha256',b.RELEASE/'dataset/MANIFEST.json'),
                       ('release_sha256',b.RELEASE/'RELEASE.json'),
                       ('model_config_sha256',b.BASE/'config.json'),
                       ('processor_tokenizer_sha256',b.BASE/'tokenizer.json')]:
        assert b.sha(path)==identity[field], field
    layers={n for n,kind in enumerate(b.read(b.BASE/'config.json')['text_config']['layer_types']) if kind=='full_attention'}
    source_root=archive/'attempt-20a5090493204fe58601d6016dd47a45/decision'
    rows=[]
    for epoch in range(1,7):
        source=source_root/f'checkpoint-{273*epoch}'
        commit=b.read(source/'CHECKPOINT_COMMIT.json')
        for name in ('adapter_config.json','adapter_model.safetensors','trainer_state.json'):
            assert b.sha(source/name)==commit['files'][name], str(source/name)
        destination=root/f'decision_epoch{epoch}';destination.mkdir()
        converter.OUT=destination
        temp=converter.build_staging('decision',source,layers)
        target=destination/'decision';temp.rename(target)
        before=load_file(str(source/'adapter_model.safetensors'));after=load_file(str(target/'adapter_model.safetensors'))
        assert len(before)==656 and len(after)==672
        for key,value in before.items():
            assert value.dtype==after[key].dtype and torch.equal(value,after[key]),key
        for key in after.keys()-before.keys():
            original=key.replace('.self_attn.v_proj.','.self_attn.k_proj.')
            assert original!=key and original in before and torch.equal(after[key],before[original]),key
        rows.append(dict(candidate=dict(agent='decision',name=f'epoch{epoch}',epoch=epoch,step=epoch*273,
            path=str(source),weights_sha256=commit['files']['adapter_model.safetensors'],config_sha256=commit['files']['adapter_config.json']),
            path=str(target),weights_sha256=b.sha(target/'adapter_model.safetensors'),original_tensors_exact=True,
            copied_v_tensors_exact=16,stored_dtype=sorted({str(v.dtype) for v in after.values()})))
        del before,after
    assert b.sha(source_root/'best_adapter/adapter_model.safetensors')==rows[1]['candidate']['weights_sha256']
    for row in b.read(PREVIOUS/'staged/IDENTITIES.json')['rows']:
        assert b.sha(Path(row['path'])/'adapter_model.safetensors')==row['weights_sha256']
        for name,key in [('adapter_model.safetensors','weights_sha256'),('adapter_config.json','config_sha256')]:
            assert b.sha(Path(row['candidate']['path'])/name)==row['candidate'][key]
        rows.append(row)
    b.write_new(root/'IDENTITIES.json',dict(rows=rows,decision_best_alias='epoch2',new_weights_merged=False))
    assert not torch.cuda.is_initialized()
    serving.emit('staged',count=len(rows),cuda_initialized=False)


def request_record(model,prompt_token_ids,max_tokens,stop_token_ids,grammar=None):
    assert model in ('decision','recommendation','response')
    return dict(model=model,prompt_token_ids=prompt_token_ids,max_tokens=max_tokens,stop_token_ids=stop_token_ids,grammar=grammar)


def assert_eval_port_free():
    # A stopped Uvicorn socket can remain in TIME_WAIT. REUSEADDR permits that
    # state, but still refuses an actual listener. Never terminate port owners.
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        sock.bind(('127.0.0.1',18001))


def individual(calls,output,roles):
    import functional_eval
    cases=[c for c in b.read(BASE/'inputs/CASES.json')['rows'] if c['agent'] in roles]
    summary=[]
    with (output/'INDIVIDUAL.jsonl').open('x') as f:
        for index,case in enumerate(cases,1):
            call=calls[case['agent']];call.trace_events.clear();start=time.monotonic()
            error=None;prediction=None;normalized=None
            try:
                value=call(*e.args_for(case))
                if case['agent']=='decision':
                    prediction=copy.deepcopy(call.last_external_output);normalized=value
                else:
                    prediction={'reply':value} if case['agent']=='response' else value
            except Exception as exc:
                error=f'{type(exc).__name__}: {exc}'
            traces=copy.deepcopy(call.trace_events)
            if error:
                grade=dict(automated_pass=False,failures=[dict(code='inference_error',detail=error)])
            else:
                grade=functional_eval.grade(case,prediction)
                if not traces or traces[-1].get('model_input')!=case['record']['input']:
                    raise RuntimeError('Frozen individual input drift: '+case['record']['id'])
            row=dict(id=case['record']['id'],agent=case['agent'],group=case['group'],prediction=prediction,
                normalized=normalized,error=error,traces=traces,grade=grade,seconds=time.monotonic()-start)
            f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
            summary.append({k:row[k] for k in ('id','agent','group','grade','seconds')})
            if index%10==0:
                serving.emit('individual_progress',done=index,total=len(cases))
    return dict(count=len(summary),by_agent={role:dict(count=sum(x['agent']==role for x in summary),
        automated_pass=sum(x['agent']==role and x['grade']['automated_pass'] for x in summary)) for role in roles},
        semantic_review='PENDING',strict_decision_matching_is_diagnostic=True)


def run(name,selection,roles=('decision','recommendation','response'),extended=True):
    verify()
    from eval_fixed_history import acquire_lock
    from soomac_irc.decision_model import make_call_decision
    identities=b.read(BASE/'staged/IDENTITIES.json')['rows']
    chosen={role:next(r for r in identities if r['candidate']['agent']==role and r['candidate']['name']==candidate)
            for role,candidate in selection.items()}
    for row in chosen.values():
        assert b.sha(Path(row['path'])/'adapter_model.safetensors')==row['weights_sha256']
    assert_eval_port_free()
    lock,baseline=acquire_lock()
    output=None;process=None;unit=f'soomac-dqp-eval-{os.getpid()}-{name}'
    try:
        output=b.output_dir(BASE/'runs'/name)
        adapters=output/'adapters';adapters.mkdir()
        for role,row in chosen.items():
            (adapters/role).symlink_to(row['path'],target_is_directory=True)
        captured=serving.launcher_contract(adapters)
        argv=captured['original_argv']  # Explicitly retain Decision, unlike old Q/P-only helper.
        assert sum(x.startswith('decision=') for x in argv)==1
        b.write_new(output/'PLAN.json',dict(selection=chosen,real_decision=True,real_recommendation=True,real_response=True,
            actual_ros=False,execution_reply_mode='fixed',merged_weights=False,baseline=baseline,
            roles_individual=list(roles),extended=extended,runner_sha256=b.sha(Path(__file__)),
            inputs_freeze_sha256=b.sha(BASE/'inputs/FREEZE.json')))
        b.write_new(output/'D_ONLY_PLAN.json',dict(fixture_sha256=b.sha(BASE/'inputs/D_ONLY_FIXTURE.json'),
            real_decision=True,real_recommendation=False,real_response=False,
            evaluation_source_sha256=b.sha(serving.ROOT/'tools/training_acceleration/dqp_decision_scenario.py')))
        b.write_new(output/'LAUNCH.json',dict(argv=argv,unit=unit,environment={k:captured['env'][k] for k in
            ['VLLM_BATCH_INVARIANT','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE']},differences=['evaluation port18001','candidate paths']))
        with (output/'SERVER.log').open('x') as log:
            process=subprocess.Popen(['systemd-run','--user','--scope',f'--unit={unit}',
                '-p','MemoryMax=40G','-p','MemorySwapMax=0','--',*argv],cwd=serving.ROOT,env=captured['env'],stdout=log,stderr=subprocess.STDOUT)
            serving.emit('server_starting',name=name,unit=unit,pid=process.pid)
            deadline=time.monotonic()+300
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f'Server exited {process.returncode}')
                try:
                    with urllib.request.urlopen(serving.ENDPOINT+'/health',timeout=2) as response:
                        if response.status==200:break
                except OSError:pass
                if time.monotonic()>deadline:raise TimeoutError('Server startup >300s')
                time.sleep(2)
            vc,model,processor,calls=serving.client_objects()
            calls['decision']=make_call_decision(model,processor,adapter_name='decision')
            models=vc.served_models();roots={m['id']:m.get('root') for m in models}
            assert set(roots)=={'gemma4-base','decision','recommendation','response'}
            vc.check_served_adapters({role:row['candidate']['path'] for role,row in chosen.items()},roots)
            for role,row in chosen.items():assert Path(roots[role]).resolve()==Path(row['path']).resolve()
            b.write_new(output/'SERVED_MODELS.json',models)
            original=vc.complete_tokens;counts=dict.fromkeys(chosen,0)
            try:
                with (output/'REQUESTS.jsonl').open('x') as requests,(output/'TURNS.jsonl').open('x') as turns:
                    def checked(*args,**kwargs):
                        request=request_record(*args,**kwargs)
                        answer=original(*args,**kwargs)
                        requests.write(json.dumps(dict(request=request,response=answer),ensure_ascii=False)+'\n');requests.flush()
                        counts[request['model']]+=1
                        return answer
                    vc.complete_tokens=checked
                    stats=individual(calls,output,roles)
                    # Reuse actual call objects; no gold Decision factory is installed.
                    with patch('soomac_irc.decision_model.make_call_decision',lambda *a,**k:calls['decision']), \
                         patch('soomac_irc.recommendation_model.make_call_recommendation',lambda *a,**k:calls['recommendation']), \
                         patch('soomac_irc.response_model.make_call_response',lambda *a,**k:calls['response']):
                        episodes=[e.probe.run_episode(compound,model,processor,turns,measure_all=False,execution_reply_mode='fixed') for compound in (True,False)]
                    if extended:
                        for spec in b.read(BASE/'inputs/EPISODES.json')['extended']:
                            episodes.append(e.extended_episode(spec,calls,turns))
                    import dqp_decision_scenario
                    with (output/'D_ONLY_TURNS.jsonl').open('x') as dstream:
                        d_only=dqp_decision_scenario.evaluate(model,processor,calls['decision'],dstream)
            finally:
                vc.complete_tokens=original
            verify()
            b.write_new(output/'RESULT.json',dict(individual=stats,episodes=episodes,decision_only_episodes=d_only,requests=counts,semantic_review='PENDING',actual_ros=False))
            serving.emit('run_complete',name=name,individual=stats,episodes=episodes,requests=counts)
    except BaseException as exc:
        if output is not None:b.write_new(output/'FAILURE.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
    finally:
        try:
            if process is not None:
                stop=subprocess.run(['systemctl','--user','stop',unit+'.scope'],capture_output=True,text=True,timeout=20)
                process.wait(timeout=30)
                b.write_new(output/'CLEANUP.json',dict(unit=unit,stop_returncode=stop.returncode,process_returncode=process.returncode))
        finally:os.close(lock)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['stage','epochs','run'])
    p.add_argument('--name');p.add_argument('--decision',default='epoch2');p.add_argument('--recommendation',default='epoch6');p.add_argument('--response',default='best150')
    p.add_argument('--roles',default='decision,recommendation,response')
    p.add_argument('--start-epoch',type=int,choices=range(1,7),default=1)
    a=p.parse_args();os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
    if a.action=='stage':stage()
    elif a.action=='epochs':
        for epoch in range(a.start_epoch,7):
            roles=['decision','recommendation','response']
            # Same frozen independent inputs already measured for D2 and Q6.
            prior=b.read(BASE/'runs/best2_current/RESULT.json')
            assert prior['individual']['count']==144
            if epoch==2:roles.remove('decision')
            if epoch==6:roles.remove('recommendation')
            run(f'epoch{epoch}',{role:f'epoch{epoch}' for role in ('decision','recommendation','response')},roles=tuple(roles))
    else:
        if not a.name:p.error('--name required')
        run(a.name,dict(decision=a.decision,recommendation=a.recommendation,response=a.response),roles=tuple(a.roles.split(',')))
