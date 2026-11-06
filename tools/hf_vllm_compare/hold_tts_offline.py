"""Trial-only wetext cache pin; original TTS helper/model stays unchanged."""
import runpy
from pathlib import Path
import wetext.wetext as wetext_module

root=Path(__file__).resolve().parents[2]
cache=Path('/home/roma/.cache/modelscope/hub/pengzhendong/wetext')
for language in ['en','zh']:
 for name in ['tagger','verbalizer']:
  assert (cache/language/'tn'/(name+'.fst')).is_file()

def local_snapshot(repo_id,*args,**kwargs):
 assert repo_id=='pengzhendong/wetext',repo_id
 return str(cache)

wetext_module.snapshot_download=local_snapshot
print('TRIAL wetext local cache only; helper uses text_frontend=False',flush=True)
runpy.run_path(str(root/'tools/vllm_server/stack_hold_tts.py'),run_name='__main__')
