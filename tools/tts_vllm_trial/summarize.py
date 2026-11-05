import json
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'training/learning_first_v43/final/goals/GL10/tts_vllm_trial_20261105'


def read(path):return json.loads(path.read_text())
def stats(xs):
    xs=sorted(xs)
    if not xs:return None
    def quantile(p):
        position=(len(xs)-1)*p;lo=int(position);hi=min(lo+1,len(xs)-1)
        return xs[lo]+(xs[hi]-xs[lo])*(position-lo)
    return dict(n=len(xs),min=xs[0],median=statistics.median(xs),p95=quantile(.95),max=xs[-1])


summary={}
for label in ['baseline_16k','hf_control_16k_r2','vllm_16k','baseline_post_16k']:
    p=BASE/label
    if not p.exists():continue
    item=dict(done=(p/'DONE.json').exists(),failure=read(p/'FAILURE.json') if (p/'FAILURE.json').exists() else None)
    probes=[read(x) for x in sorted((p/'results').glob('tts_probe_*.json'))]
    good=[r for r in probes if r.get('ok') and r.get('raw_audio_duration_s',0)>=.3]
    item['audio_rejection']=read(p/'AUDIO_REJECTION.json') if (p/'AUDIO_REJECTION.json').exists() else None
    item['probes']=dict(total=len(probes),passed=len(good),first_write_s=stats([r['first_write_delay_s'] for r in good]),
                        generation_rtf=stats([r['rtf'] for r in good]),generation_s=stats([r['generation_s'] for r in good]))
    item['probe_rows']=[dict(text=r['command']['text'],seed=r['command']['seed'],first_write_s=r['first_write_delay_s'],rtf=r['rtf'],wav=r['wav'],audio_s=r['raw_audio_duration_s']) for r in good]
    if (p/'COMPLETE.json').exists():
        completion=read(p/'COMPLETE.json')
        turns=[json.loads(x) for x in (p/'TURNS.jsonl').read_text().splitlines()]
        item['scenario']=dict(turns=len(turns),contract_passed=sum(r['contract_passed'] for r in turns),
            state_passed=sum(r['state_plan_passed'] for r in turns),wall_s=completion['scenario_wall_s'],
            llm_s=sum(r['timing']['llm_s'] for r in turns),first_audio_write_s=stats([r['timing']['text_to_first_audio_write_s'] for r in turns]),
            tts_first_write_s=stats([r['speech']['first_write_delay_s'] for r in turns]),tts_rtf=stats([r['speech']['rtf'] for r in turns]),
            episodes={name:dict(turns=len(rs),seconds=rs[-1]['speech']['callback_end']-rs[0]['timing']['graph_start']) for name in dict.fromkeys(r['episode'] for r in turns) if (rs:=[r for r in turns if r['episode']==name])})
    if not (p/'VRAM.jsonl').exists():continue
    rows=[json.loads(x) for x in (p/'VRAM.jsonl').read_text().splitlines()]
    if not rows:continue
    item['gpu']=dict(peak_gib=max(r['used'] for r in rows)/1024**3,min_free_gib=min(r['free'] for r in rows)/1024**3,
                    phases={name:dict(peak_gib=max(r['used'] for r in rs)/1024**3,min_free_gib=min(r['free'] for r in rs)/1024**3)
                            for name in dict.fromkeys(r['phase'] for r in rows) if (rs:=[r for r in rows if r['phase']==name])},
                    max_sample_gap_s=max((b['time']-a['time'] for a,b in zip(rows,rows[1:])),default=0))
    item['asr_checks']=[read(x) for x in sorted((p/'results').glob('stt_audio_check_*.json'))]
    if (p/'STRESS.json').exists():
        stress=read(p/'STRESS.json')
        # Intersect actual compute intervals (TTS producer, not merely playback).
        a=max(stress['llm']['start'],stress['stt']['start'],stress['tts']['synthesis']['start'])
        b=min(stress['llm']['end'],stress['stt']['end'],stress['tts']['synthesis']['end'])
        item['stress']=dict(three_way_overlap_s=max(0,b-a),tts_first_write_s=stress['tts']['first_write_delay_s'],
                           tts_rtf=stress['tts']['rtf'],stt_s=stress['stt']['inference_s'],llm_s=stress['llm']['end']-stress['llm']['start'])
    item['cleanup']=read(p/'RESOURCES.json') if (p/'RESOURCES.json').exists() else None
    summary[label]=item
if all(k in summary and summary[k]['probes']['passed']==9 and 'scenario' in summary[k] for k in ['baseline_16k','vllm_16k']):
    candidate=summary['vllm_16k'];control=summary['baseline_16k']
    median_gain=1-candidate['probes']['first_write_s']['median']/control['probes']['first_write_s']['median']
    summary['comparison']=dict(reference='Original HF baseline; same-environment HF invalid. Includes dependency, dtype and sampling changes.',first_write_median_gain_vs_control=median_gain,
        first_write_p95_ratio_vs_control=candidate['probes']['first_write_s']['p95']/control['probes']['first_write_s']['p95'],
        rtf_median_gain_vs_control=1-candidate['probes']['generation_rtf']['median']/control['probes']['generation_rtf']['median'],
        qualification=dict(median_gain_10pct=median_gain>=.1,
            p95_not_over_10pct_worse=candidate['probes']['first_write_s']['p95']<=1.1*control['probes']['first_write_s']['p95'],
            all_probes_rtf_under_1=candidate['probes']['generation_rtf']['max']<1,
            gpu_headroom_1_5gib=candidate['gpu']['min_free_gib']>=1.5 and 'stress' in candidate,
            scenario_contract=candidate['scenario']['contract_passed']==36,
            subjective_audio_quality='NOT_REVIEWED'))
(BASE/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps({k:dict(done=v['done'],probes=v['probes'],scenario=v.get('scenario'),gpu_peak_gib=v['gpu']['peak_gib']) for k,v in summary.items() if k!='comparison'},ensure_ascii=False,indent=2))
