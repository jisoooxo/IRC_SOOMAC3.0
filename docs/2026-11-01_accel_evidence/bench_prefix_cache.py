"""Replay logged D/Q/P calls with prefix cache OFF vs ON; compare tokens, latency, VRAM; check VLM unaffected."""
import glob
import json
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, "/home/roma/IRC_SOOMAC3.0/src/soomac_irc")
import xgrammar as xgr
from xgrammar.contrib.hf import LogitsProcessor as XGrammarLogitsProcessor

from soomac_irc.agent_contract import DECISION_SCHEMA, RECOMMENDATION_SCHEMA, RESPONSE_SCHEMA
from soomac_irc.model_runtime import (
    DECISION_ADAPTER_NAME, DECISION_ADAPTER_PATH, RECOMMENDATION_ADAPTER_NAME, RECOMMENDATION_ADAPTER_PATH,
    RESPONSE_ADAPTER_NAME, RESPONSE_ADAPTER_PATH, load_model, make_call_vlm,
)
from soomac_irc.prefix_cache import PrefixKVCache

MAX_PER_KEY = {"decision": 12, "recommendation": 5, "response/task": 8, "response/general": 4, "response/mixed": 4}
MAX_NEW = {"decision": 1024, "recommendation": 1024, "response": 512}
ADAPTER = {"decision": DECISION_ADAPTER_NAME, "recommendation": RECOMMENDATION_ADAPTER_NAME, "response": RESPONSE_ADAPTER_NAME}


def collect_events():
    events, seen, counts = [], set(), {}
    for path in sorted(glob.glob("/home/roma/IRC_SOOMAC3.0/logs/llm/2026-1*/*/agent_turns.jsonl"), reverse=True):
        for line in open(path, encoding="utf-8"):
            try:
                turn = json.loads(line)
            except json.JSONDecodeError:
                continue
            for stage, evs in (turn.get("llm_trace") or {}).items():
                for ev in evs or []:
                    msgs = ev.get("messages") if isinstance(ev, dict) else None
                    if not msgs:
                        continue
                    key = stage if stage != "response" else f"response/{ev.get('route', 'task')}"
                    sig = json.dumps(msgs, ensure_ascii=False)
                    if key not in MAX_PER_KEY or sig in seen or counts.get(key, 0) >= MAX_PER_KEY[key]:
                        continue
                    seen.add(sig)
                    counts[key] = counts.get(key, 0) + 1
                    events.append((stage, key, msgs))
    print("events:", counts, flush=True)
    return events


def sync_time(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t


def main():
    t0 = time.perf_counter()
    model, processor = load_model(DECISION_ADAPTER_PATH, RESPONSE_ADAPTER_PATH, RECOMMENDATION_ADAPTER_PATH)
    print(f"load {time.perf_counter() - t0:.1f}s, alloc {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)
    tok = processor.tokenizer
    stop_ids = list(dict.fromkeys(i for i in [tok.eos_token_id, tok.convert_tokens_to_ids("<turn|>")] if isinstance(i, int) and i >= 0))
    info = xgr.TokenizerInfo.from_huggingface(tok, vocab_size=len(tok), stop_token_ids=stop_ids)
    comp = xgr.GrammarCompiler(info)
    grammars = {
        "decision": comp.compile_json_schema(DECISION_SCHEMA, any_order=True),
        "recommendation": comp.compile_json_schema(RECOMMENDATION_SCHEMA),
        "response": comp.compile_json_schema(RESPONSE_SCHEMA),
    }

    # VLM before any prefix cache exists
    call_vlm = make_call_vlm(model, processor)
    img = Image.new("RGB", (448, 448), (200, 60, 40))
    vlm_args = ([img], "너는 이미지 판정기다. 한 문장으로 답한다.", "이 이미지의 주된 색은?")
    vlm_before, vlm_t1 = sync_time(lambda: call_vlm(*vlm_args))

    cache = PrefixKVCache(processor)
    rows = []
    torch.cuda.reset_peak_memory_stats()
    for stage, key, msgs in collect_events():
        model.set_adapter(ADAPTER[stage])
        inputs = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                               return_tensors="pt", enable_thinking=False).to(model.device)
        n_in = inputs["input_ids"].shape[1]

        def args():
            return {**inputs, "max_new_tokens": MAX_NEW[stage], "do_sample": False, "use_cache": True,
                    "eos_token_id": stop_ids, "logits_processor": [XGrammarLogitsProcessor(grammars[stage])]}

        with torch.inference_mode():
            cache.enabled = False
            off, t_off = sync_time(lambda: cache.generate(model, ADAPTER[stage], msgs[0]["content"], args()))
            cache.enabled = True
            on, t_on = sync_time(lambda: cache.generate(model, ADAPTER[stage], msgs[0]["content"], args()))
        ev = dict(cache.last_event)
        same = torch.equal(off[0, n_in:], on[0, n_in:])
        row = {"key": key, "in": n_in, "out": int(off.shape[1] - n_in), "off_s": round(t_off, 3), "on_s": round(t_on, 3),
               "cache": ev, "same_tokens": same}
        if not same:
            row["off_text"] = processor.decode(off[0, n_in:], skip_special_tokens=True)
            row["on_text"] = processor.decode(on[0, n_in:], skip_special_tokens=True)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    print(f"peak alloc during replay {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, "
          f"after {torch.cuda.memory_allocated() / 2**30:.2f} GiB, cache entries {len(cache.entries)}", flush=True)

    vlm_after, vlm_t2 = sync_time(lambda: call_vlm(*vlm_args))
    print(json.dumps({"vlm_same": vlm_before == vlm_after, "vlm_before": vlm_before, "vlm_after": vlm_after,
                      "vlm_s": [round(vlm_t1, 3), round(vlm_t2, 3)]}, ensure_ascii=False), flush=True)

    hits = [r for r in rows if r["cache"].get("status") == "hit"]
    print("SUMMARY", json.dumps({
        "n": len(rows), "same_tokens": sum(r["same_tokens"] for r in rows),
        "hit_n": len(hits),
        "hit_off_mean_s": round(sum(r["off_s"] for r in hits) / max(len(hits), 1), 3),
        "hit_on_mean_s": round(sum(r["on_s"] for r in hits) / max(len(hits), 1), 3),
    }), flush=True)
    json.dump(rows, open("/tmp/claude-1000/-home-roma-IRC-SOOMAC3-0/f10905d6-bf59-4085-a889-d658537908bd/scratchpad/bench_rows.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
