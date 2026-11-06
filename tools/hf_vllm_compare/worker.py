"""Same frozen inputs and closed-loop episodes; no ROS or production mutations."""
import gc,hashlib,json,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'training/learning_first_v43/final/goals/GL10/hf_vllm_compare_20261105'
OUT=BASE/sys.argv[1]
CFG=json.loads((OUT/'CONFIG.json').read_text())
sys.path[:0]=[str(ROOT/'src/soomac_irc'),str(ROOT/'tools/training_acceleration')]

def save(name,value):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,ensure_ascii=False,indent=2));t.replace(p)
def progress(**kw):save('PROGRESS.json',dict(time=time.time(),**kw))

def run():
 import torch
 torch.set_num_threads(2)
 from soomac_irc import input_budget,prefix_cache,model_runtime as mr,vllm_client as vc
 input_budget.SERVING_MAX_MODEL_LEN=CFG['max_len']
 prefix_cache.PREFIX_CACHE_ENABLED=CFG['prefix']
 from soomac_irc.decision_model import make_call_decision
 from soomac_irc.recommendation_model import make_call_recommendation
 from soomac_irc.response_model import make_call_response
 from soomac_irc.adapter_runtime import activate_adapter
 from soomac_irc.prefix_cache import PrefixKVCache
 import xgrammar as xgr
 from xgrammar.contrib.hf import LogitsProcessor
 import verify_d6_adoption as ev
 names=dict(decision=mr.DECISION_ADAPTER_PATH,recommendation=mr.RECOMMENDATION_ADAPTER_PATH,response=mr.RESPONSE_ADAPTER_PATH)
 vc.VLLM_BASE_URL='http://127.0.0.1:18001'
 progress(phase='model_load');t=time.perf_counter()
 if CFG['backend']=='hf':
  model,processor=mr.load_model(names['decision'],names['response'],names['recommendation'])
  assert set(model.peft_config)==set(names);assert model.is_loaded_in_4bit
  model.eval()
 else:model,processor=vc.load_vllm_backend(mr.MODEL_PATH,names)
 load_s=time.perf_counter()-t
 meta=dict(load_s=load_s,backend=CFG['backend'],dtype=str(getattr(model,'dtype',None)),adapters=names,torch_version=torch.__version__,processor_type=type(processor).__name__)
 if CFG['backend']=='hf':
  meta.update(quantization=model.config.quantization_config.to_dict(),lora_dtype_counts={d:sum('lora_' in n and str(p.dtype)==d for n,p in model.named_parameters()) for d in ['torch.float32','torch.bfloat16','torch.float16']})
 save('MODEL.json',meta)
 items=json.loads((BASE/'REPLAY.json').read_text())
 tok=processor.tokenizer
 # All grammar compilation and input rendering are outside timed generation.
 info=xgr.TokenizerInfo.from_huggingface(tok,vocab_size=len(tok),stop_token_ids=items[0]['request']['stop_token_ids'])
 compiler=xgr.GrammarCompiler(info);compiled={}
 for item in items:
  q=item['request'];ids=processor.apply_chat_template(item['messages'],add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt',enable_thinking=False)['input_ids'][0].tolist()
  assert ids==q['prompt_token_ids'];assert len(ids)+q['max_tokens']<=CFG['max_len']
  key=hashlib.sha256(q['grammar'].encode()).hexdigest()
  if key not in compiled:compiled[key]=compiler.compile_grammar(q['grammar'])
 cache=PrefixKVCache(processor,enabled=CFG['prefix']);all_replay=[]
 with (OUT/'REPLAY.jsonl').open('x') as stream:
  for sweep in range(3):
   for item in items:
    q=item['request'];role=item['role'];progress(phase='replay',sweep=sweep,id=item['id'],role=role)
    activate_adapter(model,role)
    if CFG['backend']=='hf':
     ids=torch.tensor([q['prompt_token_ids']],device=model.device)
     grammar=compiled[hashlib.sha256(q['grammar'].encode()).hexdigest()]
     args=dict(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=q['max_tokens'],do_sample=False,use_cache=True,eos_token_id=q['stop_token_ids'],logits_processor=[LogitsProcessor(grammar)])
     torch.cuda.synchronize();start=time.perf_counter()
     with torch.inference_mode():out=cache.generate(model,role,item['messages'][0]['content'],args)
     torch.cuda.synchronize();duration=time.perf_counter()-start
     generated=out[0,ids.shape[1]:].tolist();raw=processor.decode(generated,skip_special_tokens=True).strip()
     answer=dict(text=raw,token_ids=generated,finish_reason='stop' if generated and generated[-1] in q['stop_token_ids'] else 'length',prompt_tokens=ids.shape[1],completion_tokens=len(generated),cached_tokens=(cache.last_event or {}).get('cached_tokens',0),prefix_event=cache.last_event)
     del ids,out,args
    else:
     start=time.perf_counter();answer=vc.complete_tokens(**q);duration=time.perf_counter()-start
    try:parsed=json.loads(answer['text']);valid=True
    except (ValueError,TypeError):parsed=None;valid=False
    row=dict(id=item['id'],role=role,sweep=sweep,seconds=duration,response=answer,json_valid=valid,parsed=parsed,request_sha256=hashlib.sha256(json.dumps(q,sort_keys=True).encode()).hexdigest(),input_tokens=len(q['prompt_token_ids']))
    stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush();all_replay.append(row)
 cache.clear();del cache;gc.collect()
 if CFG['backend']=='hf':torch.cuda.empty_cache()
 prepare_start=time.perf_counter();warmup=None
 if CFG['backend']=='vllm' and CFG['prefix']:
  from soomac_irc.startup_warmup import warmup_vllm_agents
  warmup=warmup_vllm_agents(model,processor,{k:k for k in names})
 calls=dict(decision=make_call_decision(model,processor,adapter_name='decision'),recommendation=make_call_recommendation(model,processor,adapter_name='recommendation'),response=make_call_response(model,processor,adapter_name='response'))
 save('SCENARIO_PREPARATION.json',dict(seconds=time.perf_counter()-prepare_start,warmup=warmup,hf_fresh_callable_caches=True,vllm_cache_from_replay=CFG['backend']=='vllm' and CFG['prefix']))
 original=vc.complete_tokens
 with (OUT/'REQUESTS.jsonl').open('x') as http,(OUT/'TURNS.jsonl').open('x') as turns:
  def record(*args,**kwargs):
   import inspect
   bound=inspect.signature(original).bind(*args,**kwargs);bound.apply_defaults();q=dict(bound.arguments)
   start=time.perf_counter();answer=original(*args,**kwargs)
   http.write(json.dumps(dict(request=q,response=answer,seconds=time.perf_counter()-start),ensure_ascii=False)+'\n');http.flush();return answer
  vc.complete_tokens=record
  n=0
  class Stream:
   def write(self,line):
    nonlocal n
    row=json.loads(line)
    trims=sum(t.get('history_trimmed_messages',0) for group in row['traces'].values() for t in group)
    trims+=sum(max(0,len(row['before']['history'][-20:])-len(t.get('model_input',{}).get('recent_history',row['before']['history'][-20:]))) for t in row['traces']['decision'])
    row['extra_history_trimmed_messages']=trims
    if trims:row['contract_passed']=False
    turns.write(json.dumps(row,ensure_ascii=False)+'\n');turns.flush();n+=1
    progress(phase='scenario',turns=n,total=36,episode=row['episode'],turn=row['turn'],state=row['state_plan_passed'],contract=row['contract_passed'],seconds=row['seconds'])
   def flush(self):turns.flush()
  start=time.perf_counter()
  specs=json.loads((BASE/'EPISODES.json').read_text())
  episodes=[ev.episode(spec,calls,Stream(),'fixed') for spec in specs]
  elapsed=time.perf_counter()-start;vc.complete_tokens=original
 assert n==36
 save('COMPLETE.json',dict(replay=len(all_replay),scenario_turns=n,scenario_s=elapsed,episodes=episodes,actual_ros=False,tts_audio_playback=False,stt_tts_resident=True))
 progress(phase='done',turns=n)

if __name__=='__main__':
 try:run()
 except BaseException as e:
  save('FAILURE.json',dict(error=str(e),traceback=traceback.format_exc()));raise
