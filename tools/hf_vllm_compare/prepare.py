import hashlib,json,sys
from pathlib import Path
import importlib.metadata
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'training/learning_first_v43/final/goals/GL10/hf_vllm_compare_20261105'
sys.path[:0]=[str(ROOT/'src/soomac_irc')]
from soomac_irc import model_runtime as mr
from soomac_irc.agent_prompts import DECISION_SYSTEM,RECOMMENDATION_SYSTEM
from soomac_irc.response_model import RESPONSE_SYSTEM_BY_ROUTE
from transformers import AutoProcessor
from safetensors import safe_open
import torch

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 paths=dict(decision=mr.DECISION_ADAPTER_PATH,recommendation=mr.RECOMMENDATION_ADAPTER_PATH,response=mr.RESPONSE_ADAPTER_PATH)
 adapters={}
 config=json.loads((Path(mr.MODEL_PATH)/'config.json').read_text())
 global_layers={i for i,t in enumerate(config['text_config']['layer_types']) if t=='full_attention'}
 for role,source in paths.items():
  source=Path(source).resolve();dest=(ROOT/'models/vllm_adapters'/role).resolve()
  with safe_open(source/'adapter_model.safetensors',framework='pt',device='cpu') as a,safe_open(dest/'adapter_model.safetensors',framework='pt',device='cpu') as b:
   ak,bk=set(a.keys()),set(b.keys());assert ak<=bk
   assert all(torch.equal(a.get_tensor(k),b.get_tensor(k)) for k in ak)
   extras=bk-ak;assert len(extras)==2*len(global_layers)
   for k in extras:
    assert '.self_attn.v_proj.' in k and int(k.split('.layers.')[1].split('.')[0]) in global_layers
    assert torch.equal(b.get_tensor(k),a.get_tensor(k.replace('.v_proj.','.k_proj.')))
  adapters[role]=dict(original=str(source),converted=str(dest),same_original_tensors=len(ak),extra_kv_copies=len(extras),files={str(p):sha(p) for root in [source,dest] for p in [root/'adapter_model.safetensors',root/'adapter_config.json']})
 source=ROOT/'training/learning_first_v43/final/goals/GL10/tts_vllm_trial_20261105/vllm_16k'
 requests=[json.loads(l) for l in (source/'REQUESTS.jsonl').read_text().splitlines() if json.loads(l)['phase']=='scenario']
 turns=[json.loads(l) for l in (source/'TURNS.jsonl').read_text().splitlines()]
 traces=[(r['episode'],r['turn'],role,t) for r in turns for role in ['decision','recommendation','response'] for t in r['traces'][role]]
 assert len(traces)==len(requests)==62
 proc=AutoProcessor.from_pretrained(mr.MODEL_PATH,local_files_only=True)
 chosen={0,16,45,49,50,1,55,2,48,51,53,56};items=[]
 for i,(record,(episode,turn,role,trace)) in enumerate(zip(requests,traces)):
  if i not in chosen:continue
  q=record['request'];messages=trace['messages'];assert q['model']==role
  system=messages[0]['content']
  assert system in ([DECISION_SYSTEM] if role=='decision' else [RECOMMENDATION_SYSTEM] if role=='recommendation' else list(RESPONSE_SYSTEM_BY_ROUTE.values()))
  ids=proc.apply_chat_template(messages,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt',enable_thinking=False)['input_ids'][0].tolist()
  assert ids==q['prompt_token_ids'];assert len(ids)+q['max_tokens']<=12288
  items.append(dict(id=i,role=role,episode=episode,turn=turn,messages=messages,request=q,reference=record['response']))
 assert len(items)==12
 (BASE/'REPLAY.json').write_text(json.dumps(items,ensure_ascii=False,indent=2))
 episodes=source.parent/'EPISODES.json';(BASE/'EPISODES.json').write_bytes(episodes.read_bytes())
 files=list((ROOT/'src/soomac_irc/soomac_irc').glob('*.py'))+[ROOT/'tools/vllm_server/start_vllm_server.sh',ROOT/'tools/training_acceleration/verify_d6_adoption.py',ROOT/'tools/training_acceleration/dqp_evaluation.py',ROOT/'tools/vllm_server/stack_hold_stt.py',ROOT/'tools/vllm_server/stack_hold_tts.py']
 packages={e:{d.metadata['Name']:d.version for d in importlib.metadata.distributions(path=[str(p) for p in (Path('/home/roma/miniconda3/envs')/e/'lib').glob('python*/site-packages')])} for e in ['soomac_llm','soomac_vllm_server','gemma4_env','cosyvoice3']}
 record=dict(source={str(p):sha(p) for p in files},adapters=adapters,packages=packages,base_model=mr.MODEL_PATH,base_files={str(p):dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in Path(mr.MODEL_PATH).glob('*.safetensors')},tokenizer_config={str(p):sha(p) for p in Path(mr.MODEL_PATH).glob('*.json')},input_sha={str(p):sha(p) for p in [BASE/'REPLAY.json',BASE/'EPISODES.json']},replay_counts={r:sum(x['role']==r for x in items) for r in paths})
 (BASE/'BEFORE.json').write_text(json.dumps(record,ensure_ascii=False,indent=2));print(json.dumps(dict(adapters={k:{kk:v[kk] for kk in ['same_original_tensors','extra_kv_copies']} for k,v in adapters.items()},replay_counts=record['replay_counts'],source_files=len(files))))
if __name__=='__main__':main()
