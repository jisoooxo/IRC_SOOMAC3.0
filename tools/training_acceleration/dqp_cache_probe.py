"""Bounded cache-state diagnosis of one byte-identical evaluation request."""
import copy
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

import dqp_prompt_trial as trial
from soomac_irc import vllm_client as vc

EXACT_BLOCKS = os.environ.get('DQP_EXACT_CACHE_BLOCKS') == '1'
BASE = trial.BASE/('cache_probe_exact_blocks' if EXACT_BLOCKS else 'cache_probe')


def run():
    trial.verify()
    trial.previous.assert_eval_port_free()
    from eval_fixed_history import acquire_lock
    lock, baseline = acquire_lock()
    process = None
    unit = f'soomac-cache-probe-{os.getpid()}'
    BASE.mkdir()
    try:
        run_path = trial.BASE/'runs/patched_model'
        audit = trial.b.read(run_path/'TRANSPORT_TOKEN_AUDIT.json')
        line = next(r['line'] for r in audit['rows'] if r['id']=='unresolved_reference_control/6/decision')
        original = json.loads((run_path/'REQUESTS.jsonl').read_text().splitlines()[line-1])['request']
        trial.b.write_new(BASE/'PLAN.json',dict(request=original,source_line=line,source_sha256=trial.b.sha(run_path/'REQUESTS.jsonl'),
            conditions=['cold','warm','prime8320','prime8576','prime8576_repeat','prime8320_repeat','cold_repeat'],
            maximum_completions=11,source_code_sha256=trial.b.sha(Path(__file__)),
            exact_cached_token_targets=EXACT_BLOCKS,prime_extra_tokens=64 if EXACT_BLOCKS else 0,
            only_extra_environment='VLLM_SERVER_DEV_MODE=1 on owned localhost server, to reset its cache',actual_ros=False))
        captured = trial.serving.launcher_contract(trial.BASE/'adapters')
        argv = captured['original_argv']
        for flag in ('--max-model-len','--max-num-batched-tokens'):
            argv[argv.index(flag)+1]=str(trial.MAX_LEN)
        argv[argv.index('--gpu-memory-utilization')+1]=str(trial.GPU_UTIL)
        captured['env']['VLLM_SERVER_DEV_MODE']='1'
        trial.b.write_new(BASE/'LAUNCH.json',dict(argv=argv,unit=unit,gpu_baseline=baseline,
            environment={k:captured['env'][k] for k in ('VLLM_BATCH_INVARIANT','VLLM_SERVER_DEV_MODE')}))
        with (BASE/'SERVER.log').open('x') as log:
            process=subprocess.Popen(['systemd-run','--user','--scope',f'--unit={unit}','-p','MemoryMax=40G','-p','MemorySwapMax=0','--',*argv],
                cwd=trial.serving.ROOT,env=captured['env'],stdout=log,stderr=subprocess.STDOUT)
            deadline=time.monotonic()+600
            while True:
                if process.poll() is not None:
                    raise RuntimeError('Cache probe server exited')
                try:
                    with urllib.request.urlopen(trial.serving.ENDPOINT+'/health',timeout=2) as response:
                        if response.status==200:
                            break
                except OSError:
                    pass
                if time.monotonic()>deadline:
                    raise TimeoutError('Startup >600s')
                time.sleep(2)
            vc.VLLM_BASE_URL=trial.serving.ENDPOINT
            models=vc.served_models()
            selected=trial.b.read(trial.e.BASE/'SELECTION.json')['recommended']
            assert next(m for m in models if m['id']=='gemma4-base')['max_model_len']==trial.MAX_LEN
            for role,row in selected.items():
                assert Path(next(m for m in models if m['id']==role)['root']).resolve()==Path(row['path']).resolve()
            trial.b.write_new(BASE/'SERVED_MODELS.json',models)
            with (BASE/'REQUESTS.jsonl').open('x') as stream:
                results=[]
                def call(label, request):
                    response=vc.complete_tokens(**request)
                    row=dict(label=label,request=request,response=response)
                    stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
                    print(json.dumps(dict(label=label,text=response['text'],cached=response['cached_tokens']),ensure_ascii=False),flush=True)
                    return row
                for label,prime in [('cold',0),('warm',None),('prime8320',8320),('prime8576',8576),
                                    ('prime8576_repeat',8576),('prime8320_repeat',8320),('cold_repeat',0)]:
                    if prime is not None:
                        reset=vc._post('/reset_prefix_cache',{})
                        assert reset['success'],reset
                    if prime:
                        prefix=copy.deepcopy(original)
                        prime_length=prime+(64 if EXACT_BLOCKS else 0)
                        prefix.update(prompt_token_ids=prefix['prompt_token_ids'][:prime_length],max_tokens=1,grammar=None)
                        call(label+'_priming_only',prefix)
                    result=call(label,original)
                    if EXACT_BLOCKS and prime:
                        assert result['response']['cached_tokens']==prime,result['response']['cached_tokens']
                    results.append(result)
                trial.b.write_new(BASE/'RESULT.json',dict(
                    all_target_requests_equal=all(r['request']==original for r in results),
                    target_calls=len(results),priming_calls=4,
                    responses=[dict(label=r['label'],text=r['response']['text'],cached_tokens=r['response']['cached_tokens']) for r in results],
                    unique_responses=len(set(r['response']['text'] for r in results))))
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


if __name__=='__main__':
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    run()
