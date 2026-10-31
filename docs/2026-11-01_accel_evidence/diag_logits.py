"""Next-token logits: full pass vs prefix-cache continuation (sliding-trimmed cache vs untrimmed cache)."""
import copy
import json
import sys

import torch

sys.path.insert(0, "/home/roma/IRC_SOOMAC3.0/src/soomac_irc")
from transformers import DynamicCache

from soomac_irc.model_runtime import (
    DECISION_ADAPTER_NAME, DECISION_ADAPTER_PATH, RECOMMENDATION_ADAPTER_PATH, RESPONSE_ADAPTER_NAME,
    RESPONSE_ADAPTER_PATH, load_model,
)

rows = json.load(open(sys.argv[1]))  # list of {"adapter":..., "messages":..., "prefix": int}
model, processor = load_model(DECISION_ADAPTER_PATH, RESPONSE_ADAPTER_PATH, RECOMMENDATION_ADAPTER_PATH)


def report(name, ref, x):
    d = (ref.float() - x.float()).abs()
    top_ref = ref.float().topk(3)
    top_x = x.float().topk(3)
    print(f"  {name:28s} maxdiff={d.max():.4f} meandiff={d.mean():.5f} "
          f"top_ref={top_ref.indices.tolist()} {[round(v, 2) for v in top_ref.values.tolist()]} "
          f"top_x={top_x.indices.tolist()} {[round(v, 2) for v in top_x.values.tolist()]}", flush=True)


with torch.inference_mode():
    for row in rows:
        model.set_adapter(row["adapter"])
        ids = processor.apply_chat_template(row["messages"], add_generation_prompt=True, tokenize=True,
                                            return_dict=True, return_tensors="pt", enable_thinking=False)["input_ids"].cuda()
        L, n = ids.shape[1], row["prefix"]
        print(f"{row['adapter']} L={L} prefix={n} suffix={L - n}", flush=True)
        full = model(input_ids=ids, attention_mask=torch.ones_like(ids), logits_to_keep=1).logits[0, -1]
        full2 = model(input_ids=ids, attention_mask=torch.ones_like(ids), logits_to_keep=1).logits[0, -1]
        report("full vs full (repeat)", full, full2)

        pos = torch.arange(n, L, device="cuda")[None]
        for label, cache_factory in [("trimmed cache(config)", lambda: DynamicCache(config=model.config)),
                                     ("untrimmed cache(no config)", lambda: DynamicCache())]:
            c = cache_factory()
            model(input_ids=ids[:, :n], attention_mask=torch.ones_like(ids[:, :n]), past_key_values=c, use_cache=True, logits_to_keep=1)
            c2 = copy.deepcopy(c)
            out = model(input_ids=ids[:, n:], attention_mask=torch.ones_like(ids), past_key_values=c2,
                        position_ids=pos, use_cache=True, logits_to_keep=1).logits[0, -1]
            report(label, full, out)
            print(f"    cache layer0 len={c.layers[0].keys.shape[-2]} type={type(c.layers[0]).__name__}; "
                  f"layer5 len={c.layers[5].keys.shape[-2]} type={type(c.layers[5]).__name__}", flush=True)

        # chunked full prefill without any truncation at a different split (pure numerics baseline)
        for split in (n // 2, L - 1):
            c = DynamicCache()
            model(input_ids=ids[:, :split], attention_mask=torch.ones_like(ids[:, :split]), past_key_values=c, use_cache=True, logits_to_keep=1)
            out = model(input_ids=ids[:, split:], attention_mask=torch.ones_like(ids), past_key_values=c,
                        position_ids=torch.arange(split, L, device="cuda")[None], use_cache=True, logits_to_keep=1).logits[0, -1]
            report(f"untrimmed split@{split}", full, out)
