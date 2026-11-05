"""Read-only original-file/package checks; saves evidence only in trial artifacts."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import socket
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'training/learning_first_v43/final/goals/GL10/tts_vllm_trial_20261105'
before=json.loads((BASE/'BEFORE.json').read_text())

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

source={p:sha(Path(p)) for p in before['source']}
packages={}
for env in before['packages']:
    roots=list((Path('/home/roma/miniconda3/envs')/env/'lib').glob('python*/site-packages'))
    packages[env]={d.metadata['Name']:d.version for d in importlib.metadata.distributions(path=[str(p) for p in roots])}
adapters={}
for role,item in before['adapters'].items():
    root=(ROOT/'models/vllm_adapters'/role).resolve()
    adapters[role]=dict(root=str(root),files={name:sha(root/name) for name in item['files']})
ports={}
for port in (8000,18001):
    with socket.socket() as s:
        s.settimeout(1);ports[port]=s.connect_ex(('127.0.0.1',port))==0
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
gpu_owners=[]
for line in gpu.splitlines():
    pid=int(line.split(',')[0]);proc=Path('/proc')/str(pid)
    cgroup=(proc/'cgroup').read_text().strip() if (proc/'cgroup').exists() else 'UNKNOWN'
    gpu_owners.append(dict(pid=pid,cgroup=cgroup,trial_owned='soomac-tts-trial-' in cgroup,
        command=(proc/'comm').read_text().strip() if (proc/'comm').exists() else 'UNKNOWN'))
units=subprocess.run(['systemctl','--user','list-units','--all','--no-legend','--plain','soomac-tts-trial-*'],capture_output=True,text=True,check=True).stdout.strip()
result=dict(time=time.time(),source_count=len(source),source_unchanged=source==before['source'],
    source_changed=[p for p in source if source[p]!=before['source'][p]],
    packages_unchanged={e:packages[e]==before['packages'][e] for e in packages},
    package_changes={e:{n:[before['packages'][e].get(n),packages[e].get(n)] for n in set(before['packages'][e])|set(packages[e]) if before['packages'][e].get(n)!=packages[e].get(n)} for e in packages},
    adapters_unchanged=adapters==before['adapters'],ports_busy=ports,gpu_compute=gpu,gpu_owners=gpu_owners,trial_units=units,
    after=dict(source=source,packages=packages,adapters=adapters))
(BASE/'AFTER.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='after'},ensure_ascii=False,indent=2))
assert result['source_unchanged'] and all(result['packages_unchanged'].values()) and result['adapters_unchanged']
assert all(not p['trial_owned'] and p['cgroup']!='UNKNOWN' for p in gpu_owners) and not units and not ports[18001]
