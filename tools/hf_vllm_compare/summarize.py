import json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'training/learning_first_v43/final/goals/GL10/hf_vllm_compare_20261105'

def read(p):return json.loads(p.read_text())
def lines(p):
 if not p.exists():return []
 result=[]
 for l in p.read_text().splitlines():
  try:result.append(json.loads(l))
  except json.JSONDecodeError:continue
 return result

def stats(xs):
 xs=sorted(xs)
 if not xs:return None
 i=(len(xs)-1)*.95;lo=int(i);hi=min(lo+1,len(xs)-1)
 return dict(n=len(xs),mean=statistics.mean(xs),median=statistics.median(xs),p95=xs[lo]+(xs[hi]-xs[lo])*(i-lo),min=xs[0],max=xs[-1],sum=sum(xs))

def key(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'))
summary={}
for p in BASE.iterdir():
 if not p.is_dir() or not (p/'CONFIG.json').exists():continue
 replay=lines(p/'REPLAY.jsonl');turns=lines(p/'TURNS.jsonl');gpu=lines(p/'VRAM.jsonl')
 warm=[r for r in replay if r['sweep']>0]
 item=dict(config=read(p/'CONFIG.json'),complete=(p/'COMPLETE.json').exists(),failure=read(p/'DRIVER_FAILURE.json') if (p/'DRIVER_FAILURE.json').exists() else None,
  replay=dict(total=len(replay),valid_json=sum(r['json_valid'] for r in replay),stopped=sum(r['response']['finish_reason']=='stop' for r in replay),warm_all=stats([r['seconds'] for r in warm]),roles={role:{phase:dict(latency_s=stats([r['seconds'] for r in rs]),output_tokens=stats([r['response']['completion_tokens'] for r in rs]),cached_tokens=stats([r['response'].get('cached_tokens') or 0 for r in rs])) for phase in ['first_sweep','warm_repeated'] if (rs:=[r for r in replay if r['role']==role and ((r['sweep']==0) if phase=='first_sweep' else (r['sweep']>0))])} for role in ['decision','recommendation','response']},repeat_json_variation=[i for i in sorted({r['id'] for r in replay}) if len({key(r['parsed']) for r in replay if r['id']==i})>1]),
  scenario=dict(turns=len(turns),state_passed=sum(r['state_plan_passed'] for r in turns),contract_passed=sum(r['contract_passed'] for r in turns),extra_history_trim=sum(r.get('extra_history_trimmed_messages',0) for r in turns),turn_latency_s=stats([r['seconds'] for r in turns]),failed_turns=[dict(episode=r['episode'],turn=r['turn'],user=r['user_text'],reply=r['reply'],checks=r['checks'],extra_checks=r['extra_checks'],expected_plans=r['expected_plans'],actual_plans=r['actual_plans']) for r in turns if not r['contract_passed']]))
 if item['complete']:item['scenario']['wall_s']=read(p/'COMPLETE.json')['scenario_s']
 if gpu:
  item['gpu']=dict(peak_gib=max(r['used'] for r in gpu)/1024**3,min_free_gib=min(r['free'] for r in gpu)/1024**3,min_ram_available_gib=min(r['ram_available'] for r in gpu)/1024**3,max_sample_gap_s=max((b['time']-a['time'] for a,b in zip(gpu,gpu[1:])),default=0),phase_peaks={phase:max(r['used'] for r in gpu if r['phase']==phase)/1024**3 for phase in {r['phase'] for r in gpu}})
 if (p/'CLEANUP.json').exists():item['cleanup']=read(p/'CLEANUP.json')
 summary[p.name]=item
comparisons={}
for baseline in ['hf_nocache','hf_cache','hf_cache_lora_bf16_r2','vllm_16k_current']:
 a=lines(BASE/baseline/'REPLAY.jsonl')
 for label in summary:
  if label==baseline:continue
  b=lines(BASE/label/'REPLAY.jsonl');pairs=[]
  for x in a:
   if x['sweep']==0:continue
   y=next((r for r in b if r['id']==x['id'] and r['sweep']==x['sweep']),None)
   if y:
    assert x['request_sha256']==y['request_sha256']
    pairs.append(dict(id=x['id'],role=x['role'],sweep=x['sweep'],speed_ratio=x['seconds']/y['seconds'],baseline_s=x['seconds'],candidate_s=y['seconds'],json_equal=x['json_valid'] and y['json_valid'] and key(x['parsed'])==key(y['parsed']),baseline_output_tokens=x['response']['completion_tokens'],candidate_output_tokens=y['response']['completion_tokens']))
  if pairs:
   comparisons[baseline+'__'+label]=dict(n=len(pairs),json_equal=sum(r['json_equal'] for r in pairs),sum_speed_ratio=sum(r['baseline_s'] for r in pairs)/sum(r['candidate_s'] for r in pairs),paired_ratio_stats=stats([r['speed_ratio'] for r in pairs]),same_json_ratio_stats=stats([r['speed_ratio'] for r in pairs if r['json_equal']]),pairs=pairs)
result=dict(conditions=summary,comparisons=comparisons)
(BASE/'SUMMARY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
for label,r in summary.items():
 print(label, 'done',r['complete'],'replay',r['replay']['total'],'scenario',r['scenario']['turns'],'pass',r['scenario']['contract_passed'],'warm_sum',round(r['replay']['warm_all']['sum'],2) if r['replay']['warm_all'] else None,'peak',round(r.get('gpu',{}).get('peak_gib',0),2))
