#!/usr/bin/env python3
"""Reproduce XGrammar JSON property-order behavior without loading model weights."""
import argparse
import importlib.metadata
import inspect
import json
from pathlib import Path

import xgrammar as xgr
from transformers import AutoProcessor

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", required=True, help="Local Gemma4 base-model directory; weights are not included in this bundle")
parser.add_argument("--schema", default=str(Path(__file__).resolve().parents[1] / "dataset_contract/decision_schema.json"))
args = parser.parse_args()

processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
tokenizer = processor.tokenizer
stop_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<turn|>")]
stop_ids = [value for value in dict.fromkeys(stop_ids) if isinstance(value, int) and value >= 0]
info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=len(tokenizer), stop_token_ids=stop_ids)
compiler = xgr.GrammarCompiler(info)
schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
sample = '{"confirmation":"reject","order":{"toppings":{"소시지":"high"}}}'

print("xgrammar_version=", importlib.metadata.version("xgrammar"))
print("compile_signature=", inspect.signature(xgr.GrammarCompiler.compile_json_schema))
print("sample=", sample)
for any_order in (False, True):
    grammar = compiler.compile_json_schema(schema, any_order=any_order)
    matcher = xgr.GrammarMatcher(grammar)
    accepted = matcher.accept_string(sample)
    print(f"any_order={any_order} accepted={accepted} completed={matcher.is_completed()}")
