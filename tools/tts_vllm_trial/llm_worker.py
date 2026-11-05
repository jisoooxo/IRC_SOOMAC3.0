"""Closed-loop real D/Q/P graph; speech for every reply; physical completion fixtures."""
import copy
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback

BASE=Path(os.environ['SOOMAC_TRIAL_ROOT'])
ROOT=Path('/home/roma/IRC_SOOMAC3.0')
LABEL=sys.argv[1]
MAXLEN=int(sys.argv[2])
WARM=sys.argv[3]=='warm'
OUT=BASE/LABEL
sys.path[:0]=[str(ROOT/'src/soomac_irc'),str(ROOT/'tools/training_acceleration')]


def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(path)


def main():
    import torch
    torch.set_num_threads(2)
    from soomac_irc import input_budget
    input_budget.SERVING_MAX_MODEL_LEN=MAXLEN
    import verify_d6_adoption as ev
    from soomac_irc import vllm_client as vc
    vc.VLLM_BASE_URL='http://127.0.0.1:18001'
    paths=dict(decision=ev.DECISION_ADAPTER_PATH,recommendation=ev.RECOMMENDATION_ADAPTER_PATH,response=ev.RESPONSE_ADAPTER_PATH)
    model,processor=vc.load_vllm_backend(ev.MODEL_PATH,paths)
    calls=dict(decision=ev.make_call_decision(model,processor,adapter_name='decision'),
               recommendation=ev.make_call_recommendation(model,processor,adapter_name='recommendation'),
               response=ev.make_call_response(model,processor,adapter_name='response'))
    original=vc.complete_tokens
    phase='warmup' if WARM else 'scenario'
    with (OUT/'REQUESTS.jsonl').open('x') as requests:
        def measured(*args,**kwargs):
            bound=inspect.signature(original).bind(*args,**kwargs);bound.apply_defaults();q=dict(bound.arguments)
            assert len(q['prompt_token_ids'])+q['max_tokens']<=MAXLEN
            start=time.perf_counter();response=original(*args,**kwargs);end=time.perf_counter()
            requests.write(json.dumps(dict(phase=phase,request=q,response=response,start=start,end=end),ensure_ascii=False)+'\n');requests.flush()
            return response
        vc.complete_tokens=measured
        warm_start=time.perf_counter()
        if WARM:
            from soomac_irc.startup_warmup import warmup_vllm_agents
            warmup_vllm_agents(model,processor,dict(decision='decision',recommendation='recommendation',response='response'))
        warm_s=time.perf_counter()-warm_start if WARM else 0.0
        phase='scenario'
        save(OUT/'READY.json',dict(time=time.time(),warmup_seconds=warm_s,maxlen=MAXLEN,
             input_budget_process_override=True,model_type=type(model).__name__,device=model.device,
             torch_allocated=torch.cuda.memory_allocated() if torch.cuda.is_initialized() else 0))
        original_build=ev.build_graph
        graph_timing={}
        def build(*args,**kwargs):
            graph=original_build(*args,**kwargs)
            class Proxy:
                def invoke(self,state):
                    graph_timing['start']=time.perf_counter()
                    answer=graph.invoke(state)
                    graph_timing['end']=time.perf_counter()
                    return answer
            return Proxy()
        ev.build_graph=build
        turns=[]
        with (OUT/'TURNS.jsonl').open('x') as stream:
            class SpeechStream:
                def write(self,line):
                    row=json.loads(line)
                    # Persist graph output before audio, so an audio failure cannot erase the turn.
                    with (OUT/'GRAPH_TURNS.jsonl').open('a') as raw:raw.write(line)
                    name=f'tts_{LABEL}_{len(turns):03d}.json'
                    command=dict(text=row['reply'],episode=row['episode'],turn=row['turn'])
                    save(OUT/'commands'/name,command)
                    deadline=time.monotonic()+120
                    path=OUT/'results'/name
                    while not path.exists():
                        if (OUT/'STOP').exists():raise RuntimeError('Resource guard or shutdown')
                        if time.monotonic()>deadline:raise TimeoutError('TTS >120s')
                        time.sleep(.02)
                    speech=json.loads(path.read_text());assert speech['ok'],speech
                    row['speech']=speech
                    row['timing']=dict(graph_start=graph_timing['start'],graph_end=graph_timing['end'],
                        llm_s=graph_timing['end']-graph_timing['start'],
                        text_to_first_audio_write_s=speech['first_write']-graph_timing['start'],
                        text_to_tts_callback_end_s=speech['callback_end']-graph_timing['start'])
                    stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush();turns.append(row)
                    save(BASE/'PROGRESS.json',dict(condition=LABEL,turns=len(turns),total=36,episode=row['episode'],turn=row['turn'],
                        llm_s=row['timing']['llm_s'],first_audio_s=row['timing']['text_to_first_audio_write_s'],
                        finished_audio_s=row['timing']['text_to_tts_callback_end_s'],state_pass=row['state_plan_passed'],contract_pass=row['contract_passed']))
                def flush(self):stream.flush()
            started=time.perf_counter()
            specs=json.loads((BASE/'EPISODES.json').read_text())
            episodes=[ev.episode(spec,calls,SpeechStream(),'fixed') for spec in specs]
            elapsed=time.perf_counter()-started
        assert len(turns)==36
        save(OUT/'COMPLETE.json',dict(episodes=episodes,turns=len(turns),scenario_wall_s=elapsed,
            warmup_s=warm_s,actual_ros_commands=False,physical_completion_fixtures=True,
            real_stt_transcripts_as_scenario_inputs=False,real_tts_each_turn=True))


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        save(OUT/'FAILURE.json',dict(error=str(exc),traceback=traceback.format_exc()));raise
