import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'training/learning_first_v43/final/goals/GL10/tts_vllm_trial_20261105'
results=[]
conditions=[('hf_control_16k_r2','hf',[]),('vllm_16k','vllm',[]),('baseline_post_16k','baseline',['--post-only'])] if '--resume' in sys.argv else [('baseline_16k','baseline',[]),('hf_control_16k','hf',[]),('vllm_16k','vllm',[])]
suffix='_RESUMED' if '--resume' in sys.argv else ''
for label,mode,extra in conditions:
    print(json.dumps(dict(start=label,time=time.time())),flush=True)
    with (BASE/(label+'_driver.log')).open('x') as log:
        result=subprocess.run(['/home/roma/miniconda3/envs/cosyvoice3/bin/python',str(ROOT/'tools/tts_vllm_trial/measure.py'),label,mode,*extra],
                              cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    resources=json.loads((BASE/label/'RESOURCES.json').read_text()) if (BASE/label/'RESOURCES.json').exists() else None
    results.append(dict(label=label,returncode=result.returncode,resources=resources))
    (BASE/('MATRIX_PROGRESS'+suffix+'.json')).write_text(json.dumps(results,indent=2))
    print(json.dumps(dict(done=label,returncode=result.returncode,time=time.time())),flush=True)
    if resources is None or resources['compute_pids'] or resources['owned_units'] or resources['port_busy']:
        raise RuntimeError('Cleanup incomplete; refuse next condition')
    time.sleep(2)
(BASE/('MATRIX_COMPLETE'+suffix+'.json')).write_text(json.dumps(results,indent=2))
sys.exit(int(any(r['returncode'] for r in results)))
