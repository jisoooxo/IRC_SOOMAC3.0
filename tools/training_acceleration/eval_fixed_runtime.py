"""Fresh-process P/Q evaluation under one explicit inference precision policy.

The earlier mixed-setting run is excluded. No training, Decision model, ROS,
checkpoint write, package installation or deployment is performed here.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import benchmark as b
import correct_input_validation as review
from eval_correct_input_epochs import encode_validation, verify_freeze, ce_summary

ROOT = Path(__file__).resolve().parents[2]
POLICY = {
    "engine": "standard_HF_PEFT_runtime_inference_not_Unsloth_Trainer",
    "base_quantization": "NF4_double_quant",
    "base_dtype": "bfloat16", "bnb_compute_dtype": "bfloat16",
    "bnb_storage_dtype": "bfloat16", "adapter_storage_dtype": "float32",
    "autocast_enabled": False, "matmul_allow_tf32": False,
    "cudnn_allow_tf32": False, "deterministic_algorithms": True,
    "cublas_workspace_config": ":4096:8", "seed": 3407,
    "attention": "sdpa", "dropout": "eval",
    "do_sample": False, "prefix_cache": True,
    "new_process_per_checkpoint": True,
    "ce": "same runtime forward precision; fp32 CE; exact causal target shift",
    "ce_comparison": "Within this new run only; not equivalent to training-time BF16 autocast loss",
}


def score_ce(model, collator, item, torch):
    batch = collator([{k: item[k] for k in ("input_ids", "labels")}])
    batch = {k: v.to(model.device) for k, v in batch.items()}
    shifted = torch.nn.functional.pad(batch["labels"], (0, 1), value=-100)[..., 1:]
    positions = shifted.ne(-100).any(dim=0).nonzero(as_tuple=False).flatten()
    if not positions.numel():
        raise ValueError("No supervised targets")
    targets = shifted.index_select(1, positions)
    # CE and generation use the same disabled-autocast runtime policy.
    with torch.inference_mode(), torch.autocast("cuda", enabled=False):
        result = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                       logits_to_keep=positions, use_cache=False, return_dict=True)
        if result.logits.shape[:2] != targets.shape:
            raise RuntimeError("Causal logits/labels mismatch")
        ce = float(torch.nn.functional.cross_entropy(result.logits.float().flatten(0, 1),
                                                    targets.flatten(), ignore_index=-100))
    if not math.isfinite(ce):
        raise RuntimeError("Nonfinite CE")
    return {"ce": ce, "target_tokens": int(targets.ne(-100).sum()),
            "input_tokens": item["length"], "padded_tokens": batch["input_ids"].shape[1]}


def predict(case, call, torch):
    source = copy.deepcopy(case["record"]["input"])
    call.trace_events.clear()
    with torch.inference_mode(), torch.autocast("cuda", enabled=False):
        value = call(*review.rebuild(case["agent"], case["record"]))
    traces = copy.deepcopy(call.trace_events)
    if not traces or any(t.get("error") or t.get("model_input") != source or
                         t.get("history_trimmed_messages") for t in traces):
        raise RuntimeError("Frozen input/history/model-error gate failed")
    return {"id": case["record"]["id"], "origin": case["evaluation_origin"],
            "prediction": {"reply": value} if case["agent"] == "response" else value,
            "traces": traces}


def same_prediction(left, right):
    return (left["prediction"] == right["prediction"] and
            left["traces"][-1]["raw"] == right["traces"][-1]["raw"] and
            left["traces"][-1]["messages"] == right["traces"][-1]["messages"])


def worker(args):
    verify_freeze(args.frozen)
    plan = b.read(args.output / "PLAN.json")
    candidate = next(c for c in plan["candidates"] if f"{c['agent']}:{c['name']}" == args.worker)
    agent, path = candidate["agent"], Path(candidate["path"])
    for filename, expected in (("adapter_model.safetensors", candidate["weights_sha256"]),
                               ("adapter_config.json", candidate["config_sha256"])):
        if b.sha(path / filename) != expected:
            raise ValueError("Committed checkpoint hash mismatch")
    folder = args.output / f"{agent}_{candidate['name']}"
    folder.mkdir()
    lock = b.acquire_gpu_lock()
    budget = None
    try:
        sys.path.insert(0, str(ROOT / "src/soomac_irc"))
        sys.path.insert(0, str(b.RELEASE / "common"))
        import torch
        from budget import Budget
        from peft import get_peft_model_state_dict
        from safetensors.torch import load_file
        from transformers import AutoProcessor, DataCollatorForSeq2Seq
        from soomac_irc.model_runtime import load_model
        from soomac_irc.response_model import make_call_response
        from soomac_irc.recommendation_model import make_call_recommendation
        import functional_eval
        torch.manual_seed(POLICY["seed"])
        torch.set_default_dtype(torch.float32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        budget = Budget(b.PROCESS_BUDGET, b.RESERVE)
        kwargs = {"response_adapter_path" if agent == "response" else "recommendation_adapter_path": str(path)}
        model, processor = load_model(adapter_path=None, **kwargs)
        processor = AutoProcessor.from_pretrained(b.RELEASE / "assets/model-processor", local_files_only=True)
        model.requires_grad_(False)
        model.eval()
        if set(model.peft_config) != {agent} or not model.is_loaded_in_4bit:
            raise RuntimeError("Unexpected model/adapter topology")
        restored = get_peft_model_state_dict(model, adapter_name=agent)
        stored = load_file(str(path / "adapter_model.safetensors"), device="cpu")
        if len(restored) != 656 or set(stored) != set(restored):
            raise RuntimeError("Adapter tensor membership mismatch")
        for name, tensor in restored.items():
            if tensor.dtype != torch.float32 or not torch.equal(tensor.detach().cpu(), stored[name]):
                raise RuntimeError(f"Loaded adapter differs from original: {name}")
        del stored, restored
        quant = model.config.quantization_config
        for attr, expected in (("bnb_4bit_quant_type", "nf4"), ("bnb_4bit_use_double_quant", True),
                               ("bnb_4bit_compute_dtype", torch.bfloat16),
                               ("bnb_4bit_quant_storage", torch.bfloat16)):
            if getattr(quant, attr) != expected:
                raise RuntimeError(f"Quantization drift: {attr}")
        identity = {"candidate": candidate, "policy": POLICY, "pid": os.getpid(),
                    "python": sys.executable,
                    "parameter_dtypes": dict(Counter(str(p.dtype) for p in model.parameters())),
                    "loaded_adapter_all_656_tensors_bitwise_equal": True,
                    "adapter_parameter_dtype": "torch.float32", "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "autocast_enabled": torch.is_autocast_enabled("cuda"),
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "base_quantization": quant.to_dict(), "attention": model.config._attn_implementation}
        b.write_new(folder / "IDENTITY.json", identity)
        cases = [c for c in b.read(args.frozen / "FUNCTIONAL_CASES.json")["rows"] if c["agent"] == agent]
        probe_id = "QG-NEW-Q-06" if agent == "recommendation" else "QG-NEW-P-01"
        probe = next(c for c in cases if c["record"]["id"] == probe_id)
        call = (make_call_recommendation if agent == "recommendation" else make_call_response)(model, processor, adapter_name=agent)
        first, second = predict(probe, call, torch), predict(probe, call, torch)
        repeat = {"same_input_same_settings_exact_output": same_prediction(first, second), "before_ce": [first, second]}
        b.write_new(folder / "REPEAT_START.json", repeat)
        if not repeat["same_input_same_settings_exact_output"]:
            raise RuntimeError("Reproducibility gate failed; no full evaluation")
        print(json.dumps({"event": "repeat_gate_pass", "candidate": args.worker,
                          "prediction": first["prediction"]}, ensure_ascii=False), flush=True)
        predictions = []
        with (folder / "PREDICTIONS.jsonl").open("x") as stream:
            for case in cases:
                result = predict(case, call, torch)
                result["grade"] = functional_eval.grade(case, result["prediction"])
                stream.write(json.dumps(result, ensure_ascii=False) + "\n"); stream.flush()
                predictions.append(result)
                print(json.dumps({"event": "generation", "candidate": args.worker, "id": result["id"],
                    "prediction": result["prediction"], "pass": result["grade"]["automated_pass"]}, ensure_ascii=False), flush=True)
        encoded = encode_validation(processor, args.frozen)
        collator = DataCollatorForSeq2Seq(tokenizer=processor.tokenizer, padding=True,
                                         label_pad_token_id=-100, pad_to_multiple_of=8, return_tensors="pt")
        fixed = {r["id"]: (r, item) for r, item in encoded[agent, "REVIEWED"]}
        ce_rows = []
        with (folder / "CE_ROWS.jsonl").open("x") as stream:
            for index, (row, item) in enumerate(encoded[agent, "ORIGINAL"], 1):
                result = {"id": row["id"], "family": row.get("metadata", {}).get("scenario_family"),
                          "original": score_ce(model, collator, item, torch)}
                fixed_row, fixed_item = fixed[row["id"]]
                result["reviewed"] = result["original"] if fixed_row == row else score_ce(model, collator, fixed_item, torch)
                stream.write(json.dumps(result, ensure_ascii=False) + "\n"); stream.flush()
                ce_rows.append(result)
                if index % 10 == 0 or index == len(encoded[agent, "ORIGINAL"]):
                    budget.check(force=True)
                    print(json.dumps({"event": "ce_progress", "candidate": args.worker,
                                      "rows": index, "total": len(encoded[agent, "ORIGINAL"])}), flush=True)
        last = predict(probe, call, torch)
        b.write_new(folder / "REPEAT_END.json", {"after_ce_exact_output": same_prediction(first, last), "after_ce": last})
        if not same_prediction(first, last):
            raise RuntimeError("Post-CE reproducibility gate failed")
        if model.training or any(p.grad is not None for p in model.parameters()):
            raise RuntimeError("Unexpected training state")
        verify_freeze(args.frozen)
        b.write_new(folder / "SUMMARY.json", {"candidate": candidate,
            "original_validation": ce_summary(ce_rows, "original"),
            "reviewed_validation": ce_summary(ce_rows, "reviewed"),
            "functional_cases": len(predictions), "automated_pass": sum(r["grade"]["automated_pass"] for r in predictions),
            "repeat_gates": "PASS_before_and_after_CE", "budget": budget.summary(),
            "semantic_review": "PENDING", "decision_model_called": False})
    finally:
        if budget:
            budget.close()
        os.close(lock)


def run(args):
    verify_freeze(args.frozen)
    for agent in ("response", "recommendation"):
        b.source_identity(agent)
    output = b.output_dir(args.output)
    candidates = b.read(args.frozen / "CANDIDATES.json")["rows"]
    candidates.sort(key=lambda c: (0 if (c["agent"], c["name"]) == ("recommendation", "epoch6") else 1,
                                    c["agent"], c["step"]))
    paths = [Path(__file__), Path(review.__file__), ROOT / "tools/training_acceleration/eval_correct_input_epochs.py",
             ROOT / "tools/training_acceleration/functional_eval.py",
             *(ROOT / "src/soomac_irc/soomac_irc" / n for n in (
                 "model_runtime.py", "adapter_runtime.py", "response_model.py", "recommendation_model.py",
                 "response_contract.py", "agent_prompts.py", "evaluation_v43.py", "llm_policy.py", "prefix_cache.py"))]
    pins = {str(p): b.sha(p) for p in paths}
    b.write_new(output / "PLAN.json", {"policy": POLICY, "candidates": candidates,
        "frozen_sha256": b.sha(args.frozen / "FREEZE.json"), "source_sha256": pins,
        "old_gpu_scores_excluded": True, "decision_model_called": False, "training": False, "actual_ros": False})
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               PYTHONDONTWRITEBYTECODE="1", CUBLAS_WORKSPACE_CONFIG=POLICY["cublas_workspace_config"], PYTHONHASHSEED="3407")
    env.pop("PYTHONPATH", None)
    summaries = []
    for index, candidate in enumerate(candidates, 1):
        if any(b.sha(Path(p)) != expected for p, expected in pins.items()):
            raise RuntimeError("Evaluation code changed")
        key = f"{candidate['agent']}:{candidate['name']}"
        print(json.dumps({"event": "candidate_start", "index": index, "of": len(candidates), "candidate": key}), flush=True)
        with (output / f"{candidate['agent']}_{candidate['name']}.log").open("x") as stream:
            result = subprocess.run([sys.executable, "-u", "-B", str(Path(__file__).resolve()),
                "--frozen", str(args.frozen.resolve()), "--output", str(output.resolve()), "--worker", key],
                env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Worker failed; no automatic retry: {key} exit={result.returncode}")
        summary = b.read(output / f"{candidate['agent']}_{candidate['name']}" / "SUMMARY.json")
        summaries.append(summary)
        print(json.dumps({"event": "candidate_complete", "summary": summary}, ensure_ascii=False), flush=True)
    if any(b.sha(Path(p)) != expected for p, expected in pins.items()):
        raise RuntimeError("Evaluation code changed")
    verify_freeze(args.frozen)
    b.write_new(output / "RESULT.json", {"status": "INFERENCE_COMPLETE_SEMANTIC_REVIEW_PENDING",
        "summaries": summaries, "policy": POLICY, "old_gpu_scores_excluded": True,
        "decision_model_called": False, "training": False, "actual_ros": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker")
    args = parser.parse_args()
    worker(args) if args.worker else run(args)
