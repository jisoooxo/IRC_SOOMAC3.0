"""Dump the 14 replay inputs + HF reference outputs (cache OFF/ON) and schemas for vLLM comparison."""
import importlib.util, json, sys, time
sys.path.insert(0, "/home/roma/IRC_SOOMAC3.0/src/soomac_irc")
import torch, xgrammar as xgr
from xgrammar.contrib.hf import LogitsProcessor
from soomac_irc.agent_contract import DECISION_SCHEMA, RECOMMENDATION_SCHEMA, RESPONSE_SCHEMA
from soomac_irc.model_runtime import load_model, DECISION_ADAPTER_PATH, RESPONSE_ADAPTER_PATH, RECOMMENDATION_ADAPTER_PATH
from soomac_irc.prefix_cache import PrefixKVCache
S = "/tmp/claude-1000/-home-roma-IRC-SOOMAC3-0/f10905d6-bf59-4085-a889-d658537908bd/scratchpad"
spec = importlib.util.spec_from_file_location("bench", f"{S}/bench_prefix_cache.py"); bench = importlib.util.module_from_spec(spec); spec.loader.exec_module(bench)
bench.MAX_PER_KEY = {"decision": 6, "recommendation": 2, "response/task": 4, "response/general": 2}
events = bench.collect_events()
json.dump({"events": [{"stage": s, "key": k, "messages": m} for s, k, m in events],
           "schemas": {"decision": DECISION_SCHEMA, "recommendation": RECOMMENDATION_SCHEMA, "response": RESPONSE_SCHEMA}},
          open(f"{S}/b3_inputs.json", "w"), ensure_ascii=False)
model, processor = load_model(DECISION_ADAPTER_PATH, RESPONSE_ADAPTER_PATH, RECOMMENDATION_ADAPTER_PATH)
tok = processor.tokenizer
stop = list(dict.fromkeys(i for i in [tok.eos_token_id, tok.convert_tokens_to_ids("<turn|>")] if isinstance(i, int) and i >= 0))
comp = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tok, vocab_size=len(tok), stop_token_ids=stop))
gram = {"decision": comp.compile_json_schema(DECISION_SCHEMA, any_order=True), "recommendation": comp.compile_json_schema(RECOMMENDATION_SCHEMA), "response": comp.compile_json_schema(RESPONSE_SCHEMA)}
cache = PrefixKVCache(processor); rows = []
for stage, key, msgs in events:
    model.set_adapter(bench.ADAPTER[stage])
    inp = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=False).to("cuda")
    n = inp["input_ids"].shape[1]; row = {"key": key, "in": n, "input_ids": inp["input_ids"][0].tolist()}
    for mode in ("off", "on"):
        cache.enabled = mode == "on"
        args = {**inp, "max_new_tokens": bench.MAX_NEW[stage], "do_sample": False, "use_cache": True, "eos_token_id": stop, "logits_processor": [LogitsProcessor(gram[stage])]}
        torch.cuda.synchronize(); t = time.perf_counter()
        with torch.inference_mode(): out = cache.generate(model, bench.ADAPTER[stage], msgs[0]["content"], args)
        torch.cuda.synchronize()
        row[f"{mode}_s"] = round(time.perf_counter() - t, 3); row[f"{mode}_text"] = processor.decode(out[0, n:], skip_special_tokens=True).strip(); row[f"{mode}_out"] = int(out.shape[1] - n)
    rows.append(row); print(json.dumps({k: v for k, v in row.items() if k != "input_ids"}, ensure_ascii=False), flush=True)
json.dump(rows, open(f"{S}/b3_hf_ref.json", "w"), ensure_ascii=False)
print("DONE")
