"""Read-only checkpoint inference: native CE and fixed-state P/Q generations.

Runs no Decision, optimizer, training, ROS, deployment or remote commands. All
outputs are new files under the requested result directory. GPU ownership uses
the existing shared fail-closed lock and budget.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
from pathlib import Path
import sys
import time

import benchmark
import correct_input_validation as reviewed

ROOT = Path(__file__).resolve().parents[2]


def verify_freeze(folder):
    receipt = benchmark.read(folder / "FREEZE.json")
    for name, expected in receipt["files"].items():
        if benchmark.sha(folder / name) != expected:
            raise ValueError(f"Frozen evaluation changed: {name}")
    for name, expected in receipt["protected_source_sha256"].items():
        if benchmark.sha(Path(name)) != expected:
            raise ValueError(f"Sealed source changed: {name}")
    return receipt


def encode_validation(processor, folder):
    sys.path.insert(0, str(benchmark.RELEASE / "common"))
    import core
    core.DATASET_ROOT = benchmark.RELEASE / "dataset"
    encoded = {}
    for agent in ("response", "recommendation"):
        core.MAX_LENGTH = 9216 if agent == "response" else 5120
        cfg = core.STAGE_CONFIGS[agent]
        cfg["schema_path"] = core.DATASET_ROOT / f"contract/{agent}_schema.json"
        cfg["systems_path" if agent == "response" else "system_path"] = (
            core.DATASET_ROOT / ("contract/response_system_by_route.json" if agent == "response"
                                 else "contract/recommendation_system.txt"))
        prompts, validator = core.load_contract(agent)
        for variant in ("ORIGINAL", "REVIEWED"):
            rows = [c["record"] for c in benchmark.read(folder / f"VALIDATION_{variant}.json")["rows"]
                    if c["agent"] == agent]
            encoded[(agent, variant)] = [(row, core.encode_record(agent, row, processor, prompts, validator))
                                        for row in rows]
    return encoded


def ce_one(model, collator, encoded, torch):
    # Identical causal target selection to sealed loss.scored_suffix, with an
    # independent fp32 CE reduction. No Trainer patches or implicit loss kwargs.
    batch = collator([{key: encoded[key] for key in ("input_ids", "labels")}])
    batch = {key: value.to(model.device) for key, value in batch.items()}
    shifted = torch.nn.functional.pad(batch["labels"], (0, 1), value=-100)[..., 1:]
    positions = shifted.ne(-100).any(dim=0).nonzero(as_tuple=False).flatten()
    if positions.numel() == 0:
        raise ValueError("No supervised targets")
    targets = shifted.index_select(1, positions)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                       logits_to_keep=positions, use_cache=False, return_dict=True)
        if output.logits.shape[:2] != targets.shape:
            raise RuntimeError("Selected logits/labels shape mismatch")
        loss = torch.nn.functional.cross_entropy(output.logits.float().flatten(0, 1), targets.flatten(),
                                                 ignore_index=-100, reduction="mean")
    value = float(loss)
    if not math.isfinite(value):
        raise RuntimeError("Nonfinite independent CE")
    return {"ce": value, "target_tokens": int(targets.ne(-100).sum()),
            "input_tokens": encoded["length"], "padded_tokens": int(batch["input_ids"].shape[1])}


def ce_summary(rows, key):
    return {"rows": len(rows), "row_mean_ce": sum(r[key]["ce"] for r in rows) / len(rows),
            "token_mean_ce": sum(r[key]["ce"] * r[key]["target_tokens"] for r in rows) /
                             sum(r[key]["target_tokens"] for r in rows)}


def grade_fixed(case, prediction):
    import functional_eval
    return functional_eval.grade(case, prediction)


def run(args):
    receipt = verify_freeze(args.frozen)
    candidates = benchmark.read(args.frozen / "CANDIDATES.json")["rows"]
    cases = benchmark.read(args.frozen / "FUNCTIONAL_CASES.json")["rows"]
    for case in cases:
        reviewed.rebuild(case["agent"], case["record"])
    if args.only:
        candidates = [c for c in candidates if f"{c['agent']}:{c['name']}" in args.only]
    if not candidates:
        raise ValueError("No selected candidates")
    output = benchmark.output_dir(args.output)
    protected_stats = {c["path"]: {name: (Path(c["path"]) / name).stat().st_mtime_ns
                                  for name in ("adapter_config.json", "adapter_model.safetensors")}
                       for c in candidates}
    pins = {str(p): benchmark.sha(p) for p in (
        Path(__file__), Path(reviewed.__file__), ROOT / "tools/training_acceleration/functional_eval.py",
        *(ROOT / "src/soomac_irc/soomac_irc" / n for n in (
            "model_runtime.py", "response_model.py", "recommendation_model.py", "response_contract.py",
            "agent_prompts.py", "evaluation_v43.py", "llm_policy.py", "prefix_cache.py")))}
    benchmark.write_new(output / "PLAN.json", {
        "freeze_sha256": benchmark.sha(args.frozen / "FREEZE.json"), "candidates": candidates,
        "engine": "standard_HF_PEFT_NF4_BF16_eval", "decision_model_called": False,
        "training": False, "actual_ros": False, "history_trimmed": False,
        "ce_policy": "selected native shifted targets; fp32 CE; row mean and token mean reported",
        "reference_caveat": "Same sealed encoding, independent HF/PEFT engine; not an assumed replay of Unsloth loss",
        "source_sha256": pins})
    lock = benchmark.acquire_gpu_lock()
    budget = None
    try:
        sys.path.insert(0, str(ROOT / "src/soomac_irc"))
        sys.path.insert(0, str(benchmark.RELEASE / "common"))
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                          TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1")
        import torch
        from budget import Budget
        from transformers import AutoProcessor, DataCollatorForSeq2Seq
        from soomac_irc.model_runtime import load_model
        from soomac_irc.response_model import make_call_response
        from soomac_irc.recommendation_model import make_call_recommendation
        from peft import get_peft_model_state_dict
        budget = Budget(benchmark.PROCESS_BUDGET, benchmark.RESERVE)
        torch.backends.cuda.matmul.allow_tf32 = True
        processor = AutoProcessor.from_pretrained(benchmark.RELEASE / "assets/model-processor",
                                                  local_files_only=True)
        encoded = encode_validation(processor, args.frozen)
        collator = DataCollatorForSeq2Seq(tokenizer=processor.tokenizer, padding=True,
                                         label_pad_token_id=-100, pad_to_multiple_of=8,
                                         return_tensors="pt")
        model = None
        previous_adapter = None
        summaries = []
        started = time.monotonic()
        for ci, candidate in enumerate(candidates, 1):
            agent, name, path = candidate["agent"], candidate["name"], Path(candidate["path"])
            for filename, expected in (("adapter_model.safetensors", candidate["weights_sha256"]),
                                       ("adapter_config.json", candidate["config_sha256"])):
                if benchmark.sha(path / filename) != expected:
                    raise ValueError(f"Checkpoint file differs from committed receipt: {path / filename}")
            adapter = f"{agent}_{name}"
            print(json.dumps({"event": "candidate_loading", "index": ci, "of": len(candidates),
                              "agent": agent, "name": name}), flush=True)
            if model is None:
                kwargs = {"response_adapter_path" if agent == "response" else "recommendation_adapter_path": str(path)}
                model, unused_processor = load_model(adapter_path=None, **kwargs)
                del unused_processor
                adapter = agent
            else:
                model.load_adapter(str(path), adapter_name=adapter, is_trainable=False, local_files_only=True)
                model.set_adapter(adapter)
                if previous_adapter:
                    model.delete_adapter(previous_adapter)
            previous_adapter = adapter
            model.requires_grad_(False)
            model.eval()
            model.config.use_cache = False
            if not getattr(model, "is_loaded_in_4bit", False):
                raise RuntimeError("Expected NF4 base")
            if any(parameter.requires_grad for parameter in model.parameters()):
                raise RuntimeError("Evaluation adapter unexpectedly trainable")
            restored = get_peft_model_state_dict(model, adapter_name=adapter)
            if len(restored) != 656 or any(not torch.isfinite(t).all() for t in restored.values()):
                raise RuntimeError("Loaded adapter tensors incomplete or nonfinite")
            del restored
            candidate_dir = output / f"{agent}_{name}"
            candidate_dir.mkdir()
            benchmark.write_new(candidate_dir / "IDENTITY.json", candidate)
            per_row = []
            original = encoded[(agent, "ORIGINAL")]
            fixed = {r["id"]: (r, e) for r, e in encoded[(agent, "REVIEWED")]}
            with (candidate_dir / "CE_ROWS.jsonl").open("x") as stream:
                for ri, (row, item) in enumerate(original, 1):
                    result = {"id": row["id"], "family": row.get("metadata", {}).get("scenario_family"),
                              "original": ce_one(model, collator, item, torch)}
                    fixed_row, fixed_item = fixed[row["id"]]
                    result["reviewed"] = (result["original"] if fixed_row == row
                                           else ce_one(model, collator, fixed_item, torch))
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                    stream.flush()
                    per_row.append(result)
                    if ri % 10 == 0 or ri == len(original):
                        budget.check(force=True)
                        print(json.dumps({"event": "ce_progress", "candidate": f"{agent}:{name}",
                                          "rows": ri, "total": len(original),
                                          "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
            call = (make_call_response if agent == "response" else make_call_recommendation)(
                model, processor, adapter_name=adapter)
            predictions, grades = [], []
            with (candidate_dir / "PREDICTIONS.jsonl").open("x") as stream:
                for case in [c for c in cases if c["agent"] == agent]:
                    args_for_call = reviewed.rebuild(agent, case["record"])
                    before = copy.deepcopy(case["record"]["input"])
                    begin = time.monotonic()
                    prediction, error = None, None
                    call.trace_events.clear()
                    try:
                        value = call(*args_for_call)
                        prediction = {"reply": value} if agent == "response" else value
                    except Exception as exc:
                        error = {"type": type(exc).__name__, "message": str(exc)}
                    traces = copy.deepcopy(call.trace_events)
                    if case["record"]["input"] != before or any(
                            t.get("model_input") != before or t.get("history_trimmed_messages") for t in traces):
                        raise RuntimeError("Frozen input/history changed during prediction")
                    entry = {"id": case["record"]["id"], "origin": case["evaluation_origin"],
                             "prediction": prediction, "error": error, "traces": traces,
                             "seconds": time.monotonic() - begin}
                    grade = (grade_fixed(case, prediction) if error is None else
                             {"automated_pass": False, "failures": [{"code": "inference_error"}]})
                    entry["grade"] = grade
                    stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    stream.flush()
                    predictions.append(entry)
                    grades.append(grade)
                    print(json.dumps({"event": "generation", "candidate": f"{agent}:{name}",
                                      "id": entry["id"], "prediction": prediction,
                                      "pass": grade["automated_pass"], "seconds": round(entry["seconds"], 2)},
                                     ensure_ascii=False), flush=True)
            summary = {"candidate": candidate, "original_validation": ce_summary(per_row, "original"),
                       "reviewed_validation": ce_summary(per_row, "reviewed"),
                       "functional_cases": len(grades),
                       "automated_pass": sum(g["automated_pass"] for g in grades),
                       "model_errors": sum(p["error"] is not None for p in predictions),
                       "human_semantic_review": "PENDING", "decision_model_called": False}
            benchmark.write_new(candidate_dir / "SUMMARY.json", summary)
            summaries.append(summary)
            del call
            gc.collect()
            torch.cuda.empty_cache()
            budget.check(force=True)
            print(json.dumps({"event": "candidate_complete", "summary": summary}, ensure_ascii=False), flush=True)
        verify_freeze(args.frozen)
        if any(benchmark.sha(Path(p)) != expected for p, expected in pins.items()):
            raise RuntimeError("Evaluation implementation changed during run")
        for p, names in protected_stats.items():
            if any((Path(p) / n).stat().st_mtime_ns != stamp for n, stamp in names.items()):
                raise RuntimeError("Original checkpoint changed during evaluation")
        benchmark.write_new(output / "RESULT.json", {"status": "INFERENCE_COMPLETE_SEMANTIC_REVIEW_PENDING",
            "summaries": summaries, "budget": budget.summary(), "seconds": time.monotonic() - started,
            "training": False, "decision_model_called": False, "protected_files_unchanged": True})
    finally:
        if budget is not None:
            budget.close()
        os.close(lock)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", action="append", help="Optional agent:name candidate selector")
    run(parser.parse_args())
