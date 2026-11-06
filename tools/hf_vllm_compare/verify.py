"""Verify pinned production assets and identify remaining GPU owners, read-only."""
import hashlib,importlib.metadata,json,socket,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'training/learning_first_v43/final/goals/GL10/hf_vllm_compare_20261105'
before=json.loads((BASE/'BEFORE.json').read_text())

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

checks={}
for name in ['source','tokenizer_config','input_sha']:
 checks[name]=dict(count=len(before[name]),changed=[p for p,h in before[name].items() if not Path(p).exists() or sha(p)!=h])
checks['adapters']={role:dict(original_target=str(Path(v['original']).resolve()),converted_target=str((ROOT/'models/vllm_adapters'/role).resolve()),target_unchanged=str((ROOT/'models/vllm_adapters'/role).resolve())==v['converted'],changed=[p for p,h in v['files'].items() if not Path(p).exists() or sha(p)!=h]) for role,v in before['adapters'].items()}
manifest_path=ROOT/'config/llm_active_stack.json'
manifest=json.loads(manifest_path.read_text())
checks['active_manifest']=dict(sha256=sha(manifest_path),active=manifest.get('active'),all_gates_pass=manifest.get('all_gates_pass'),selected_match={role:str((ROOT/manifest['selected'][role]).resolve())==v['original'] for role,v in before['adapters'].items()},baseline_manifest_sha_pinned=False,equals_git_head=subprocess.run(['git','show','HEAD:config/llm_active_stack.json'],cwd=ROOT,capture_output=True,check=True).stdout==manifest_path.read_bytes(),role='informational; current model_runtime uses direct paths, not resolver')
checks['loaded_adapter_paths']={str(p.relative_to(BASE)):{role:str(Path(value).resolve())==before['adapters'][role]['original'] for role,value in json.loads(p.read_text())['adapters'].items()} for p in BASE.glob('*/MODEL.json')}
checks['base_weight_stats_changed']=[p for p,v in before['base_files'].items() if dict(size=Path(p).stat().st_size,mtime_ns=Path(p).stat().st_mtime_ns)!=v]
checks['packages']={}
for env,prior in before['packages'].items():
 current={d.metadata['Name']:d.version for d in importlib.metadata.distributions(path=[str(p) for p in (Path('/home/roma/miniconda3/envs')/env/'lib').glob('python*/site-packages')])}
 checks['packages'][env]={n:[prior.get(n),current.get(n)] for n in set(prior)|set(current) if prior.get(n)!=current.get(n)}
ports={}
for port in [8000,18001]:
 with socket.socket() as s:s.settimeout(1);ports[port]=s.connect_ex(('127.0.0.1',port))==0
processes=[]
r=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True,check=True)
for line in r.stdout.splitlines():
 pid=int(line.split(',')[0]);p=Path(f'/proc/{pid}/cgroup');cg=p.read_text().strip() if p.exists() else 'UNKNOWN'
 processes.append(dict(pid=pid,cgroup=cg,usage=line,owned='soomac-hf-vllm-' in cg))
units=subprocess.run(['systemctl','--user','list-units','--all','--plain','--no-legend','soomac-hf-vllm-*'],capture_output=True,text=True,check=True).stdout.strip()
result=dict(time=time.time(),checks=checks,ports_busy=ports,gpu_processes=processes,trial_units=units)
(BASE/'AFTER.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))
assert not any(checks[n]['changed'] for n in ['source','tokenizer_config','input_sha'])
assert all(v['target_unchanged'] and not v['changed'] for v in checks['adapters'].values())
assert checks['active_manifest']['active'] and checks['active_manifest']['equals_git_head'] and all(checks['active_manifest']['selected_match'].values())
assert all(all(v.values()) for v in checks['loaded_adapter_paths'].values())
assert not checks['base_weight_stats_changed'] and not any(checks['packages'].values())
assert not units and not ports[18001] and all(not p['owned'] and p['cgroup']!='UNKNOWN' for p in processes)
