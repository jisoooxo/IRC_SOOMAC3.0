"""One trial-only precision control. Original HF loader/files remain intact."""
import runpy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/soomac_irc'))
from soomac_irc import model_runtime as mr
import torch
original=mr.load_model

def load(*args,**kwargs):
 model,processor=original(*args,**kwargs)
 count=0
 for name,param in model.named_parameters():
  if 'lora_' in name:
   param.data=param.data.to(dtype=torch.bfloat16);count+=1
 assert count==1968,count
 return model,processor
mr.load_model=load
runpy.run_path(str(ROOT/'tools/hf_vllm_compare/worker.py'),run_name='__main__')
