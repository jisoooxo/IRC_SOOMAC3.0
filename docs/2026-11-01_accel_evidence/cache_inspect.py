import sys
if sys.argv[1] != "-":
    sys.path.insert(0, sys.argv[1])
sys.path.insert(0, "/home/roma/IRC_SOOMAC3.0/src/soomac_irc")
from collections import Counter
import torch, transformers
from soomac_irc.model_runtime import load_model, DECISION_ADAPTER_PATH, RESPONSE_ADAPTER_PATH, RECOMMENDATION_ADAPTER_PATH
model, processor = load_model(DECISION_ADAPTER_PATH, RESPONSE_ADAPTER_PATH, RECOMMENDATION_ADAPTER_PATH)
ids = torch.randint(1000, 20000, (1, 6040), device="cuda")
before = torch.cuda.memory_allocated()
with torch.inference_mode():
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True, logits_to_keep=1)
c = out.past_key_values
print("transformers", transformers.__version__, "cache", type(c).__name__, "layers", len(c.layers), dict(Counter(type(l).__name__ for l in c.layers)))
print("layer0 keys", tuple(c.layers[0].keys.shape), "layer5 keys", tuple(c.layers[5].keys.shape))
print("cache bytes GiB", round(sum(l.keys.numel()*l.keys.element_size()+l.values.numel()*l.values.element_size() for l in c.layers)/2**30, 3),
      "alloc delta GiB", round((torch.cuda.memory_allocated()-before)/2**30, 3))
for l in c.layers:
    l.keys, l.values = l.keys.clone(), l.values.clone()
torch.cuda.synchronize()
print("after clone: alloc delta GiB", round((torch.cuda.memory_allocated()-before)/2**30, 3))
