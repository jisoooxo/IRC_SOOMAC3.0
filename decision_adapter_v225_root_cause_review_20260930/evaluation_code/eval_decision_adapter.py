#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy, json
from pathlib import Path
from collections import Counter
import torch
import xgrammar as xgr
from xgrammar.contrib.hf import LogitsProcessor as XGrammarLogitsProcessor
from transformers import AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
try:
    from transformers import AutoModelForMultimodalLM as AutoVLM
except ImportError:
    from transformers import AutoModelForImageTextToText as AutoVLM

def semantic_norm(obj, model_input):
    x=copy.deepcopy(obj)
    r=x.get('recommendation') if isinstance(x,dict) else None
    if isinstance(r,dict) and 'scope' not in r:
        if r.get('action')=='request': r['scope']='current'
        elif r.get('action')=='revise':
            prev=model_input.get('recommendation',{}).get('last_proposal')
            if prev and prev.get('scope'): r['scope']=prev['scope']
    return x

def main():
    ap=argparse.ArgumentParser()
    root=Path(__file__).resolve().parents[1]
    ap.add_argument('--model-path',default='/home/roma/Desktop/sLLM/gemma-4-12B-it')
    ap.add_argument('--adapter-path',required=True)
    ap.add_argument('--data',default=str(root/'latest_contract_eval/latest_contract_eval_sft.jsonl'))
    ap.add_argument('--schema',default=str(root/'contract/decision_schema.json'))
    ap.add_argument('--quantization',choices=['int8','nf4','bf16'],default='int8')
    ap.add_argument('--output',default='./decision_eval_predictions.jsonl')
    ap.add_argument('--limit',type=int,default=0)
    args=ap.parse_args()
    processor=AutoProcessor.from_pretrained(args.model_path,local_files_only=True)
    if args.quantization=='int8': q=BitsAndBytesConfig(load_in_8bit=True,llm_int8_threshold=6.0,llm_int8_has_fp16_weight=False)
    elif args.quantization=='nf4': q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_quant_storage=torch.bfloat16)
    else: q=None
    kw=dict(device_map={'':0},attn_implementation='sdpa',low_cpu_mem_usage=True,dtype=torch.bfloat16,local_files_only=True)
    if q is not None: kw['quantization_config']=q
    model=AutoVLM.from_pretrained(args.model_path,**kw)
    model=PeftModel.from_pretrained(model,args.adapter_path,is_trainable=False); model.eval()
    tok=processor.tokenizer; stop=[tok.eos_token_id,tok.convert_tokens_to_ids('<turn|>')]; stop=[x for x in dict.fromkeys(stop) if isinstance(x,int) and x>=0]
    schema=json.loads(Path(args.schema).read_text(encoding='utf-8'))
    info=xgr.TokenizerInfo.from_huggingface(tok,vocab_size=len(tok),stop_token_ids=stop)
    grammar=xgr.GrammarCompiler(info).compile_json_schema(schema)
    rows=[json.loads(x) for x in Path(args.data).open(encoding='utf-8') if x.strip()]
    if args.limit: rows=rows[:args.limit]
    correct=0; fam=Counter(); fam_ok=Counter(); out=[]
    with torch.inference_mode():
      for r in rows:
        msgs=r['messages'][:-1]; expected=json.loads(r['messages'][-1]['content']); model_input=json.loads(msgs[-1]['content'])
        inputs=processor.apply_chat_template(msgs,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt',enable_thinking=False).to(model.device)
        n=inputs['input_ids'].shape[1]
        gen=model.generate(**inputs,max_new_tokens=1024,do_sample=False,use_cache=True,eos_token_id=stop,logits_processor=[XGrammarLogitsProcessor(grammar)])
        raw=processor.decode(gen[0][n:],skip_special_tokens=True).strip()
        try: pred=json.loads(raw); ok=semantic_norm(pred,model_input)==semantic_norm(expected,model_input)
        except Exception: pred=None; ok=False
        f=r.get('scenario_family','unknown'); fam[f]+=1; fam_ok[f]+=int(ok); correct+=int(ok)
        out.append({'id':r['id'],'family':f,'expected':expected,'raw':raw,'predicted':pred,'ok':ok})
        print(('OK ' if ok else 'ERR'),r['id'],f)
    Path(args.output).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in out),encoding='utf-8')
    print(f'accuracy={correct}/{len(rows)}={correct/max(len(rows),1):.4f}')
    for f in sorted(fam): print(f'{f}: {fam_ok[f]}/{fam[f]}={fam_ok[f]/fam[f]:.3f}')
if __name__=='__main__': main()
