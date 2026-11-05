"""One fresh trial stack per invocation; owned scopes only, monitored and cleaned."""
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import pynvml as nv

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'training/learning_first_v43/final/goals/GL10/tts_vllm_trial_20261105'
LABEL=sys.argv[1]
MODE=sys.argv[2]
SMOKE='--smoke' in sys.argv
POST_ONLY='--post-only' in sys.argv
OUT=BASE/LABEL
SOURCE=BASE/'baseline_16k' if POST_ONLY else OUT
UNITS=[]
WORKERS={}
STOP=threading.Event()
GUARD=threading.Event()
PHASE='preflight'
SAMPLES=[]


def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(path)


def phase(value):
    global PHASE
    PHASE=value;print(json.dumps(dict(phase=value,time=time.time())),flush=True)


def monitor(gpu):
    try:
        low=0
        with (OUT/'VRAM.jsonl').open('x') as f:
            while not STOP.is_set():
                now=time.monotonic();m=nv.nvmlDeviceGetMemoryInfo(gpu)
                ram=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
                row=dict(time=time.time(),phase=PHASE,used=m.used,free=m.free,total=m.total,ram_available=ram,
                         processes=[dict(pid=p.pid,used=p.usedGpuMemory) for p in nv.nvmlDeviceGetComputeRunningProcesses(gpu)])
                SAMPLES.append(row);f.write(json.dumps(row)+'\n');f.flush()
                low=low+1 if m.free<1024**3 or ram<6*1024**3 else 0
                if low>=4:raise RuntimeError('Guard: GPU free <1GiB or RAM <6GiB for four samples')
                STOP.wait(max(0,.05-(time.monotonic()-now)))
    except BaseException as error:
        save(OUT/'GUARD.json',dict(error=str(error),phase=PHASE));GUARD.set();(OUT/'STOP').touch()
        for unit in list(reversed(UNITS)):
            subprocess.run(['systemctl','--user','stop',unit],capture_output=True,timeout=30)


def check():
    if GUARD.is_set():raise RuntimeError('Resource guard stopped owned stack')
    for role,(p,unit) in WORKERS.items():
        if role=='llm' and (OUT/'COMPLETE.json').exists():continue
        if p.poll() is not None:raise RuntimeError(f'{role} exited: {p.returncode}')


def wait_result(path,timeout=180):
    deadline=time.monotonic()+timeout
    while not path.exists():
        check()
        if time.monotonic()>deadline:raise TimeoutError(str(path))
        time.sleep(.03)
    result=json.loads(path.read_text())
    if result.get('ok') is False:raise RuntimeError(str(result))
    return result


def command(role,name,value):
    path=OUT/'commands'/f'{role}_{name}.json';save(path,value)
    return OUT/'results'/path.name


def launch(role,argv,ros=False):
    unit=f'soomac-tts-trial-{os.getpid()}-{role}.scope'
    env=os.environ.copy();env.pop('PYTHONPATH',None)
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        VLLM_PORT='18001',VLLM_HOST='127.0.0.1',VLLM_MAX_LEN='16384',VLLM_MAX_BATCHED='16384',VLLM_GPU_MEM='.58',
        VLLM_MAX_SEQS='2',VLLM_BATCH_INVARIANT='1',VLLM_SPEC_MODEL='',VLLM_QUANT='bitsandbytes',
        MODEL_PATH='/home/roma/Desktop/sLLM/gemma-4-12B-it',ADAPTER_ROOT=str(ROOT/'models/vllm_adapters'),
        VLLM_WORKER_MULTIPROC_METHOD='spawn',HF_HUB_DISABLE_TELEMETRY='1',VLLM_NO_USAGE_STATS='1',
        ROS_DOMAIN_ID='217',ROS_LOCALHOST_ONLY='1',ROS_LOG_DIR=str(OUT/'ros_logs'),TTS_DEVICE='soomac_speaker',
        SOOMAC_TEST_RUN=str(OUT),SOOMAC_TRIAL_ROOT=str(BASE),TTS_TRIAL_MODE=MODE)
    if role=='tts':
        # TTS has its own engine/shape/precision; do not inherit Gemma invariance flags.
        env.pop('VLLM_BATCH_INVARIANT',None)
        env['LD_PRELOAD']='/home/roma/miniconda3/envs/cosyvoice3/lib/libstdc++.so.6'
    if ros:
        code='source /opt/ros/humble/setup.bash\nexport PYTHONPATH='+shlex.quote(str(ROOT/'src/soomac_irc'))+':${PYTHONPATH:-}\nexec '+shlex.join(argv)
        argv=['bash','--noprofile','--norc','-c',code]
    with (OUT/(role+'.log')).open('x') as log:
        p=subprocess.Popen(['systemd-run','--user','--scope','--unit='+unit,'-p','MemoryMax=38G','-p','MemorySwapMax=0','--',*argv],
                           cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    UNITS.append(unit);WORKERS[role]=(p,unit)
    save(OUT/'OWNED.json',dict(units=UNITS,pids={r:p.pid for r,(p,u) in WORKERS.items()}))
    return p


def cleanup():
    (OUT/'STOP').touch();records=[]
    for unit in list(reversed(UNITS)):
        s=subprocess.run(['systemctl','--user','stop',unit],capture_output=True,text=True,timeout=35)
        a=subprocess.run(['systemctl','--user','is-active',unit],capture_output=True,text=True,timeout=5)
        records.append(dict(unit=unit,stop_code=s.returncode,active=a.stdout.strip()))
    for p,u in WORKERS.values():
        try:p.wait(timeout=15)
        except subprocess.TimeoutExpired:pass
    save(OUT/'CLEANUP.json',records)
    assert all(r['active'] in ('inactive','unknown','failed') for r in records),records
    UNITS.clear();WORKERS.clear()


def stress_llm():
    rows=[json.loads(s) for s in (SOURCE/'REQUESTS.jsonl').read_text().splitlines()]
    original=next(r['request'] for r in rows if r['phase']=='scenario' and r['request']['model']=='decision')
    payload=dict(model=original['model'],prompt=original['prompt_token_ids'],max_tokens=original['max_tokens'],
                 temperature=0.0,stop_token_ids=original['stop_token_ids'],skip_special_tokens=True,return_token_ids=True,
                 structured_outputs={'grammar':original['grammar']})
    start=time.perf_counter()
    req=urllib.request.Request('http://127.0.0.1:18001/v1/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=120) as response: body=json.loads(response.read())
    assert body['choices'][0]['finish_reason']=='stop'
    return dict(start=start,end=time.perf_counter(),response=body)


def run(gpu):
    trial_python=ROOT/'tools/tts_vllm_trial/.venv/bin/python'
    tts_python='/home/roma/miniconda3/envs/cosyvoice3/bin/python' if MODE=='baseline' else str(trial_python)
    audio=str(ROOT/'tools/tts_vllm_trial/audio_node.py')
    if not SMOKE:
        phase('load_stt');launch('stt',['/home/roma/miniconda3/envs/gemma4_env/bin/python',audio,'stt'],True)
        wait_result(OUT/'stt_READY.json',240)
    phase('load_tts');launch('tts',[tts_python,audio,'tts'],True)
    wait_result(OUT/'tts_READY.json',600)
    if not SMOKE:
        phase('load_llm');launch('vllm',['bash',str(ROOT/'tools/vllm_server/start_vllm_server.sh')])
        deadline=time.monotonic()+600
        while True:
            check()
            try:
                with urllib.request.urlopen('http://127.0.0.1:18001/health',timeout=2) as r:
                    if r.status==200:break
            except OSError:pass
            if time.monotonic()>deadline:raise TimeoutError('LLM health >600s')
            time.sleep(.5)
        with urllib.request.urlopen('http://127.0.0.1:18001/v1/models',timeout=5) as r: models=json.loads(r.read())['data']
        assert next(m for m in models if m['id']=='gemma4-base')['max_model_len']==16384
        for role in ('decision','recommendation','response'):
            assert Path(next(m for m in models if m['id']==role)['root']).resolve()==(ROOT/'models/vllm_adapters'/role).resolve()
        save(OUT/'MODELS.json',models)
        phase('stt_probe');wait_result(command('stt','probe',dict(seconds=10)))
    texts=['얇은면을 보통 양으로 담을게요.','양파는 적게, 버섯은 많이 담을게요.',
           '게살 알레르기가 있으시군요. 게살은 제외하고 소시지만 많이 담을게요.']
    phase('fixed_text_probes')
    count=0 if POST_ONLY else 1 if SMOKE else 3
    for repeat in range(count):
        for i,text in enumerate(texts):
            wait_result(command('tts',f'probe_{repeat}_{i}',dict(text=text,seed=1234+repeat)),180)
    if SMOKE:
        save(OUT/'SMOKE_COMPLETE.json',dict(mode=MODE,probes=3));return
    if not POST_ONLY:
        phase('scenario')
        p=launch('llm',['/home/roma/miniconda3/envs/soomac_llm/bin/python',str(ROOT/'tools/tts_vllm_trial/llm_worker.py'),LABEL,'16384','warm'],True)
        wait_result(OUT/'READY.json',180)
        complete=wait_result(OUT/'COMPLETE.json',1200);p.wait(timeout=20);assert p.returncode==0
        WORKERS.pop('llm')
        assert complete['turns']==36
        # Preserve failed contracts as failures of qualification, while completing
        # the independent audio/resource diagnostics. No physical actions occur.
        save(OUT/'CONTRACT_STATUS.json',dict(passed=all(r['state_plan_passed'] and r['contract_passed'] for r in complete['episodes']),episodes=complete['episodes']))
    phase('concurrent_stress')
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(stress_llm)
        stt=command('stt','stress',dict(seconds=30))
        tts=command('tts','stress',dict(text=texts[-1],seed=1250))
        results=dict(llm=future.result(timeout=125),stt=wait_result(stt),tts=wait_result(tts))
    save(OUT/'STRESS.json',results)
    phase('asr_audio_checks')
    for i in range(3):
        wait_result(command('stt',f'audio_check_{i}',dict(file=str(SOURCE/'audio'/f'tts_probe_0_{i}.wav'))))
    save(OUT/'DONE.json',dict(mode=MODE,turns=0 if POST_ONLY else 36,probes=0 if POST_ONLY else 9,stress=True,post_only=POST_ONLY,time=time.time()))


def main():
    OUT.mkdir()
    for name in ('commands','results','audio','ros_logs'):(OUT/name).mkdir()
    lock=os.open('/tmp/soomac-r5-gpu.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    nv.nvmlInit();gpu=nv.nvmlDeviceGetHandleByIndex(0)
    baseline=nv.nvmlDeviceGetComputeRunningProcesses(gpu)
    assert not baseline, 'Other GPU compute active; will not stop it'
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        sock.bind(('127.0.0.1',18001))
    save(OUT/'PLAN.json',dict(mode=MODE,smoke=SMOKE,llm_max_len=16384,llm_gpu_mem=.58,port=18001,ros_domain=217,
                             gpu_guard_free_gib=1,ram_guard_free_gib=6,source_sha={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'tools/tts_vllm_trial').glob('*.py')}))
    thread=threading.Thread(target=monitor,args=(gpu,),daemon=True);thread.start()
    try:run(gpu)
    except BaseException as error:
        save(OUT/'FAILURE.json',dict(error=str(error),phase=PHASE,traceback=traceback.format_exc()));raise
    finally:
        phase('cleanup')
        try:cleanup()
        finally:
            STOP.set();thread.join(timeout=10)
            with socket.socket() as sock:port_busy=sock.connect_ex(('127.0.0.1',18001))==0
            save(OUT/'RESOURCES.json',dict(guard=GUARD.is_set(),compute_pids=[p.pid for p in nv.nvmlDeviceGetComputeRunningProcesses(gpu)],
                 owned_units=UNITS,port_busy=port_busy,samples=len(SAMPLES),peak_used=max((r['used'] for r in SAMPLES),default=0),
                 min_free=min((r['free'] for r in SAMPLES),default=0)))
            nv.nvmlShutdown();os.close(lock)


if __name__=='__main__':main()
