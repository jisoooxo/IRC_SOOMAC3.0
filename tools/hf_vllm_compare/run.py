"""Sequential owned GPU scopes. Never modify or terminate production services."""
import fcntl,json,os,socket,subprocess,sys,threading,time,traceback,urllib.request
from pathlib import Path
import pynvml as nv
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'training/learning_first_v43/final/goals/GL10/hf_vllm_compare_20261105'
ENVS=Path('/home/roma/miniconda3/envs')
CONFIGS=[dict(label='hf_nocache',backend='hf',prefix=False,max_len=16384,mem=.58,invariant=True),dict(label='hf_cache',backend='hf',prefix=True,max_len=16384,mem=.58,invariant=True),dict(label='vllm_16k_nocache',backend='vllm',prefix=False,max_len=16384,mem=.58,invariant=True),dict(label='vllm_16k_current',backend='vllm',prefix=True,max_len=16384,mem=.58,invariant=True),dict(label='vllm_16k_bi_off',backend='vllm',prefix=True,max_len=16384,mem=.58,invariant=False),dict(label='vllm_12k',backend='vllm',prefix=True,max_len=12288,mem=.50,invariant=True)]

CONFIGS.append(dict(label='hf_cache_lora_bf16',backend='hf',prefix=True,max_len=16384,mem=.58,invariant=True,worker_script='worker_lora_bf16.py',lora_dtype='bfloat16'))
CONFIGS.append(dict(label='vllm_16k_nocache_r2',backend='vllm',prefix=False,max_len=16384,mem=.58,invariant=True,offline_tts=True))
CONFIGS.append(dict(label='hf_cache_lora_bf16_r2',backend='hf',prefix=True,max_len=16384,mem=.58,invariant=True,worker_script='worker_lora_bf16.py',lora_dtype='bfloat16',offline_tts=True))

def save(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2));t.replace(p)

def one(cfg,gpu):
 out=BASE/cfg['label'];out.mkdir();save(out/'CONFIG.json',cfg)
 units=[];workers=[];stop=threading.Event();guard=threading.Event();samples=[];phase='load_audio'
 env=os.environ.copy();env.pop('PYTHONPATH',None)
 env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MAX_JOBS='2',VLLM_BATCH_INVARIANT='1' if cfg['invariant'] else '0',VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_NO_USAGE_STATS='1',DO_NOT_TRACK='1')
 def launch(role,argv):
  unit=f'soomac-hf-vllm-{os.getpid()}-{cfg["label"].replace("_","-")}-{role}.scope';units.append(unit)
  with (out/(role+'.log')).open('x') as log:
   p=subprocess.Popen(['systemd-run','--user','--scope','--quiet','--unit='+unit,'-p','MemoryMax=40G','-p','MemorySwapMax=0','--',*argv],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
  workers.append((role,p));save(out/'OWNED.json',dict(units=units,pids={r:p.pid for r,p in workers}));return p
 def check():
  if guard.is_set():raise RuntimeError('Resource guard stopped owned scopes')
  for role,p in workers:
   if role=='worker' and (out/'COMPLETE.json').exists() and p.poll() in (None,0):continue
   if p.poll() is not None:raise RuntimeError(f'{role} exited {p.returncode}')
 def monitor():
  low=0
  with (out/'VRAM.jsonl').open('x') as f:
   while not stop.is_set():
    t=time.monotonic();m=nv.nvmlDeviceGetMemoryInfo(gpu)
    ram=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
    row=dict(time=time.time(),phase=phase,used=m.used,free=m.free,total=m.total,ram_available=ram,processes=[dict(pid=p.pid,used=p.usedGpuMemory) for p in nv.nvmlDeviceGetComputeRunningProcesses(gpu)])
    samples.append(row);f.write(json.dumps(row)+'\n');f.flush()
    low=low+1 if m.free<1024**3 or ram<6*1024**3 else 0
    if low>=4:
     guard.set();save(out/'GUARD.json',row)
     for unit in list(reversed(units)):subprocess.run(['systemctl','--user','stop',unit],capture_output=True,timeout=35)
     break
    stop.wait(max(0,.05-(time.monotonic()-t)))
 thread=threading.Thread(target=monitor,daemon=True);thread.start();error=None;clean=[]
 try:
  for role,environment in [('stt','gemma4_env'),('tts','cosyvoice3')]:
   helper=ROOT/'tools/hf_vllm_compare/hold_tts_offline.py' if role=='tts' and cfg.get('offline_tts') else ROOT/f'tools/vllm_server/stack_hold_{role}.py'
   launch(role,[str(ENVS/environment/'bin/python'),'-I',str(helper)])
   deadline=time.monotonic()+300
   while 'HOLD' not in (out/(role+'.log')).read_text():
    check()
    if time.monotonic()>deadline:raise TimeoutError(role+' load')
    time.sleep(.5)
  if cfg['backend']=='vllm':
   phase='load_server'
   argv=[str(ENVS/'soomac_vllm_server/bin/vllm'),'serve','/home/roma/Desktop/sLLM/gemma-4-12B-it','--served-model-name','gemma4-base','--dtype','bfloat16','--quantization','bitsandbytes','--load-format','bitsandbytes','--enable-lora','--max-lora-rank','16','--max-loras','3','--lora-modules',*[r+'='+str(ROOT/'models/vllm_adapters'/r) for r in ['decision','recommendation','response']], '--enable-prefix-caching' if cfg['prefix'] else '--no-enable-prefix-caching','--enable-prompt-tokens-details','--max-model-len',str(cfg['max_len']),'--max-num-batched-tokens',str(cfg['max_len']),'--max-num-seqs','2','--gpu-memory-utilization',str(cfg['mem']),'--limit-mm-per-prompt','{"image":8,"audio":0}','--structured-outputs-config','{"backend":"xgrammar"}','--host','127.0.0.1','--port','18001']
   save(out/'LAUNCH.json',dict(argv=argv,environment={k:env[k] for k in ['VLLM_BATCH_INVARIANT','OMP_NUM_THREADS','VLLM_WORKER_MULTIPROC_METHOD']}))
   launch('server',argv);deadline=time.monotonic()+600
   while True:
    check()
    try:
     with urllib.request.urlopen('http://127.0.0.1:18001/health',timeout=2) as r:
      if r.status==200:break
    except OSError:pass
    if time.monotonic()>deadline:raise TimeoutError('server health')
    time.sleep(.5)
   with urllib.request.urlopen('http://127.0.0.1:18001/v1/models',timeout=5) as r:models=json.loads(r.read())['data']
   assert next(x for x in models if x['id']=='gemma4-base')['max_model_len']==cfg['max_len']
   for role in ['decision','recommendation','response']:assert Path(next(x for x in models if x['id']==role)['root']).resolve()==(ROOT/'models/vllm_adapters'/role).resolve()
   save(out/'MODELS.json',models)
  phase='worker';p=launch('worker',[str(ENVS/'soomac_llm/bin/python'),str(ROOT/'tools/hf_vllm_compare'/cfg.get('worker_script','worker.py')),cfg['label']]);deadline=time.monotonic()+2400
  while p.poll() is None:
   check()
   if (out/'PROGRESS.json').exists():
    state=json.loads((out/'PROGRESS.json').read_text());phase=state['phase'];save(BASE/'PROGRESS.json',dict(condition=cfg['label'],**state))
   if time.monotonic()>deadline:raise TimeoutError('worker >40min')
   time.sleep(.5)
  workers.remove(('worker',p))
  assert p.returncode==0,(cfg['label'],p.returncode)
  assert (out/'COMPLETE.json').exists()
 except BaseException as exc:
  error=dict(error=str(exc),traceback=traceback.format_exc());save(out/'DRIVER_FAILURE.json',error)
 finally:
  phase='cleanup'
  for unit in list(reversed(units)):
   r=subprocess.run(['systemctl','--user','stop',unit],capture_output=True,text=True,timeout=35)
   a=subprocess.run(['systemctl','--user','is-active',unit],capture_output=True,text=True,timeout=5)
   clean.append(dict(unit=unit,stop_code=r.returncode,active=a.stdout.strip()))
  for role,p in workers:
   try:p.wait(timeout=15)
   except subprocess.TimeoutExpired:pass
  stop.set();thread.join(5)
  with socket.socket() as sock:busy=sock.connect_ex(('127.0.0.1',18001))==0
  remaining=[]
  for p in nv.nvmlDeviceGetComputeRunningProcesses(gpu):
   path=Path(f'/proc/{p.pid}/cgroup');cg=path.read_text().strip() if path.exists() else 'UNKNOWN'
   remaining.append(dict(pid=p.pid,cgroup=cg,owned='soomac-hf-vllm-' in cg))
  save(out/'CLEANUP.json',dict(units=clean,port_busy=busy,gpu_remaining=remaining,guard=guard.is_set(),peak_bytes=max(r['used'] for r in samples),min_free_bytes=min(r['free'] for r in samples)))
  assert all(r['active'] in ['inactive','unknown','failed'] for r in clean) and not busy and not any(p['owned'] or p['cgroup']=='UNKNOWN' for p in remaining)
 return dict(condition=cfg['label'],completed=error is None,error=error)

def main():
 lock=os.open('/tmp/soomac-r5-gpu.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 nv.nvmlInit();gpu=nv.nvmlDeviceGetHandleByIndex(0)
 baseline=[]
 for p in nv.nvmlDeviceGetComputeRunningProcesses(gpu):
  cg=Path(f'/proc/{p.pid}/cgroup').read_text();assert '/rustdesk.service' in cg,'Other compute active'
  baseline.append(dict(pid=p.pid,cgroup=cg,used=p.usedGpuMemory))
 with socket.socket() as s:s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind(('127.0.0.1',18001))
 save(BASE/('GPU_BASELINE_'+('_'.join(sys.argv[1:]) or 'main')+'.json'),baseline)
 selected=CONFIGS if len(sys.argv)==1 else [c for c in CONFIGS if c['label'] in sys.argv[1:]]
 results=[]
 suffix='_'+('_'.join(sys.argv[1:])) if len(sys.argv)>1 else ''
 try:
  for cfg in selected:
   print(json.dumps(dict(start=cfg['label'],time=time.time())),flush=True)
   result=one(cfg,gpu);results.append(result);save(BASE/('MATRIX_PROGRESS'+suffix+'.json'),results)
   print(json.dumps(result),flush=True)
 finally:nv.nvmlShutdown();os.close(lock)
 save(BASE/('MATRIX_COMPLETE'+suffix+'.json'),results)
if __name__=='__main__':main()
