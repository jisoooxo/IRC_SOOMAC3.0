"""Evaluation-only reuse of the actual Unsloth production model/loss contract.

Never invokes production.execute(), train(), backward(), optimizer.step() or
save_model(). Older HF-only evaluations are excluded from every result here.
"""
from __future__ import annotations

import argparse
import ast
import copy
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import types

import production

b = production.benchmark
ROOT = Path(__file__).resolve().parents[2]
RUNS = Path("/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/runs/r5_unsloth_local_authorized_20261101")
GOLD_COMPILED = ROOT / "tools/unsloth_remote_golden/lock/golden_unsloth_compiled_module_gemma4_unified.py"
sys.path.insert(0, str(ROOT / "tools/training_acceleration"))
import correct_input_validation as reviewed


def verify_frozen(folder):
    receipt = b.read(folder / "FREEZE.json")
    for name, expected in receipt["files"].items():
        if b.sha(folder / name) != expected:
            raise RuntimeError(f"Frozen evaluation file changed: {name}")
    for name, expected in receipt["protected_source_sha256"].items():
        if b.sha(Path(name)) != expected:
            raise RuntimeError(f"Sealed data changed: {name}")
    return receipt


def contract():
    result = {}
    for agent in ("response", "recommendation"):
        b.source_identity(agent)
        source = RUNS / agent / "INPUT_IDENTITY.json"
        identity = b.read(source)
        if b.model_builder_fingerprint() != identity["model_builder_sha256"]:
            raise RuntimeError("Actual production model-builder changed")
        if production.versions() != identity["packages"]:
            raise RuntimeError("Actual production package versions changed")
        if b.sha(Path(production.__file__)) != identity["production_sha256"]:
            raise RuntimeError("Actual production script changed")
        if b.sha(GOLD_COMPILED) != identity["compiled_source_sha256"]:
            raise RuntimeError("Saved original compiled forward differs")
        result[agent] = {"source": str(source), "source_sha256": b.sha(source), "identity": identity}
    return result


def exact_loss_method():
    """Pin the copied eval loss method to the actual production method's AST."""
    prod = ast.parse(inspect.getsource(production.execute))
    original = next(n for c in ast.walk(prod) if isinstance(c, ast.ClassDef) and c.name == "Training"
                    for n in c.body if isinstance(n, ast.FunctionDef) and n.name == "compute_loss")
    ours = ast.parse(inspect.getsource(make_evaluator))
    current = next(n for c in ast.walk(ours) if isinstance(c, ast.ClassDef) and c.name == "Evaluation"
                   for n in c.body if isinstance(n, ast.FunctionDef) and n.name == "compute_loss")
    if ast.dump(original, include_attributes=False) != ast.dump(current, include_attributes=False):
        raise RuntimeError("Evaluation loss differs from actual production compute_loss")
    return True


def precision_snapshot(torch):
    # Unsloth uses PyTorch's new precision API. Reading the legacy allow_tf32
    # property can itself raise after that API has been selected. Never set flags
    # here: production's loader/TrainingArguments remain their sole owners.
    return {"matmul_fp32_precision": torch.backends.cuda.matmul.fp32_precision}


def make_evaluator(torch, budget):
    # This function is called only AFTER importing Unsloth and its patches.
    from loss import scored_suffix
    from verification import VerifiedTrainer

    class Evaluation(VerifiedTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            budget.check()
            if (self.model_accepts_loss_kwargs is not False
                    or self.label_smoother is not None or self.compute_loss_func is not None):
                raise ValueError("only native unsmoothed shifted loss is permitted")
            selected = scored_suffix(inputs)
            loss, outputs = super().compute_loss(model, selected, return_outputs=True,
                                                 num_items_in_batch=num_items_in_batch)
            logits = outputs.get("logits") if isinstance(outputs, dict) else getattr(outputs, "logits", None)
            if (not torch.is_tensor(logits) or logits.ndim != 3
                    or logits.shape[:2] != (inputs["input_ids"].shape[0],
                                            selected["logits_to_keep"].numel())):
                raise RuntimeError("native1 model did not return selected target logits")
            budget.check()
            return (loss, outputs) if return_outputs else loss

        def training_step(self, *args, **kwargs):
            raise RuntimeError("Training prohibited in this evaluator")

        def train(self, *args, **kwargs):
            raise RuntimeError("Training prohibited in this evaluator")

        def _save_checkpoint(self, *args, **kwargs):
            raise RuntimeError("Checkpoint saving prohibited in this evaluator")

        def prediction_step(self, model, inputs, *args, **kwargs):
            observed = {"target_tokens": int(inputs["labels"].ne(-100).sum()),
                        "padded_tokens": int(inputs["input_ids"].shape[-1])}
            result = super().prediction_step(model, inputs, *args, **kwargs)
            observed["ce"] = float(result[0])
            if not math.isfinite(observed["ce"]):
                raise RuntimeError("Nonfinite native CE")
            self.row_losses.append(observed)
            if len(self.row_losses) % 10 == 0:
                print(json.dumps({"event": "ce_progress", "rows": len(self.row_losses)}), flush=True)
            return result

    return Evaluation


def encoded_rows(agent, frozen, processor):
    import core
    core.DATASET_ROOT = production.DATASET
    core.MAX_LENGTH = production.EXPECTED[agent]["max_length"]
    cfg = core.STAGE_CONFIGS[agent]
    cfg["schema_path"] = production.DATASET / f"contract/{agent}_schema.json"
    cfg["systems_path" if agent == "response" else "system_path"] = production.DATASET / (
        "contract/response_system_by_route.json" if agent == "response" else "contract/recommendation_system.txt")
    prompts, validator = core.load_contract(agent)
    result = {}
    for kind in ("ORIGINAL", "REVIEWED"):
        rows = [c["record"] for c in b.read(frozen / f"VALIDATION_{kind}.json")["rows"] if c["agent"] == agent]
        result[kind] = [(r, core.encode_record(agent, r, processor, prompts, validator)) for r in rows]
    return result


def worker(args):
    verify_frozen(args.frozen)
    expected = contract()
    exact_loss_method()
    candidate = next(c for c in b.read(args.output / "PLAN.json")["candidates"]
                     if f"{c['agent']}:{c['name']}" == args.worker)
    agent, path = candidate["agent"], Path(candidate["path"])
    protected_stats = {name: (path / name).stat().st_mtime_ns
                       for name in ("adapter_model.safetensors", "adapter_config.json")}
    for name, key in (("adapter_model.safetensors", "weights_sha256"), ("adapter_config.json", "config_sha256")):
        if b.sha(path / name) != candidate[key]:
            raise RuntimeError("Committed adapter hash mismatch")
    folder = args.output / f"{agent}_{candidate['name']}"
    folder.mkdir()
    production.configure_environment(types.SimpleNamespace(execution_environment="local"))
    lock = b.acquire_gpu_lock()
    budget = None
    try:
        if any(name in sys.modules for name in ("torch", "transformers", "peft")):
            raise RuntimeError("Unsloth must be imported before Torch/Transformers/PEFT")
        import unsloth  # MUST precede torch, transformers, PEFT and verification.
        from unsloth import FastModel
        import torch
        from transformers import AutoProcessor, DataCollatorForSeq2Seq, TrainingArguments, set_seed
        from budget import Budget
        from peft import get_peft_model_state_dict
        from safetensors.torch import load_file
        import functional_eval
        from soomac_irc.recommendation_model import make_call_recommendation
        from soomac_irc.response_model import make_call_response
        budget = Budget(b.PROCESS_BUDGET, b.RESERVE)
        set_seed(42)
        model, unused_optimizer, parameters, names = b.model_and_optimizer("unsloth", agent, torch)
        # Original loader creates the original optimizer; it has no state/updates
        # before step(). Discard it before constructing an evaluation-only Trainer.
        if unused_optimizer.state:
            raise RuntimeError("Unexpected optimizer state before evaluation")
        del unused_optimizer, parameters, names
        compiled = Path(os.environ["UNSLOTH_COMPILE_LOCATION"]) / "unsloth_compiled_module_gemma4_unified.py"
        if b.sha(compiled) != expected[agent]["identity"]["compiled_source_sha256"]:
            raise RuntimeError("Generated native1 forward differs from production")
        processor = AutoProcessor.from_pretrained(production.PROCESSOR, local_files_only=True)
        encoded = encoded_rows(agent, args.frozen, processor)
        collator = DataCollatorForSeq2Seq(tokenizer=processor.tokenizer, padding=True,
            label_pad_token_id=-100, pad_to_multiple_of=8, return_tensors="pt")
        config = expected[agent]["identity"]
        train_args = TrainingArguments(output_dir=str(folder / "eval_only"), do_train=False, do_eval=True,
            per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=16,
            bf16=True, tf32=True, bf16_full_eval=False, gradient_checkpointing=False, use_cache=False,
            save_strategy="no", report_to="none", seed=42, data_seed=42,
            dataloader_num_workers=0, remove_unused_columns=False, average_tokens_across_devices=False)
        Path(train_args.output_dir).mkdir(parents=True, exist_ok=True)
        trainer = make_evaluator(torch, budget)(model=model, args=train_args, data_collator=collator)
        production.enforce_microbatch_mean(trainer, config)
        if trainer.accelerator.mixed_precision != "bf16" or trainer.args.bf16_full_eval:
            raise RuntimeError("Actual Trainer precision differs from production")
        # The same native checkpoint loader/656-tensor verification used by production.
        trainer._load_from_checkpoint(str(path))
        loaded, stored = get_peft_model_state_dict(model), load_file(str(path / "adapter_model.safetensors"))
        if set(loaded) != set(stored) or len(loaded) != 656 or any(
                loaded[k].dtype != stored[k].dtype or not torch.equal(loaded[k].detach().cpu(), stored[k]) for k in stored):
            raise RuntimeError("Native adapter restore dtype/value mismatch")
        del loaded, stored
        quantization = model.config.quantization_config
        if hasattr(quantization, "to_dict"):
            quantization = quantization.to_dict()
        b.write_new(folder / "IDENTITY.json", {"candidate": candidate,
            "actual_production": expected[agent], "compiled_source_sha256": b.sha(compiled),
            "compute_loss_ast_equal": True, "restored_656_tensors_dtype_and_values_equal": True,
            "parameter_dtypes": {dtype: sum(str(p.dtype) == dtype for p in model.parameters())
                                 for dtype in sorted({str(p.dtype) for p in model.parameters()})},
            "bf16": trainer.args.bf16, "tf32": trainer.args.tf32,
            "bf16_full_eval": trainer.args.bf16_full_eval,
            "accelerator_mixed_precision": trainer.accelerator.mixed_precision,
            "precision_backend": precision_snapshot(torch),
            "unsloth_return_logits": os.environ["UNSLOTH_RETURN_LOGITS"],
            "model_class": str(type(model)), "pid": os.getpid(), "training_steps": 0, "optimizer_steps": 0})
        b.write_new(folder / "QUANTIZATION.json", quantization)
        evaluations = {}
        all_losses = {}
        for kind in ("ORIGINAL", "REVIEWED"):
            if kind == "REVIEWED" and all(a[0] == z[0] for a,z in zip(encoded["ORIGINAL"], encoded["REVIEWED"])):
                evaluations[kind], all_losses[kind] = evaluations["ORIGINAL"], all_losses["ORIGINAL"]
                continue
            trainer.row_losses = []
            dataset = [{k: item[k] for k in ("input_ids", "labels")} for _, item in encoded[kind]]
            print(json.dumps({"event": "evaluation_start", "candidate": args.worker, "variant": kind}), flush=True)
            evaluations[kind] = trainer.evaluate(eval_dataset=dataset)
            if len(trainer.row_losses) != len(dataset):
                raise RuntimeError("Native CE row membership changed")
            all_losses[kind] = [{"id": row["id"], **loss} for (row,_), loss in zip(encoded[kind], trainer.row_losses)]
            if kind == "ORIGINAL" and args.worker in ("recommendation:epoch6", "response:best150"):
                reference = candidate["historical_eval_loss"]
                actual = evaluations[kind]["eval_loss"]
                gate = {"reference": reference, "actual": actual, "absolute_delta": abs(actual-reference),
                        "atol": 1e-5, "rtol": 1e-4,
                        "pass": math.isclose(actual, reference, abs_tol=1e-5, rel_tol=1e-4)}
                b.write_new(folder / "TRAINING_EVAL_REPLAY.json", gate)
                if not gate["pass"]:
                    raise RuntimeError("Original production validation loss replay failed; stop")
        b.write_new(folder / "NATIVE_CE.json", {"metrics": evaluations, "rows": all_losses})
        # Generation is an explicit Unsloth inference phase, AFTER native CE.
        FastModel.for_inference(model)
        cases = [c for c in b.read(args.frozen / "FUNCTIONAL_CASES.json")["rows"] if c["agent"] == agent]
        call = (make_call_recommendation if agent == "recommendation" else make_call_response)(model, processor, adapter_name="default")
        predictions = []
        with (folder / "PREDICTIONS.jsonl").open("x") as stream:
            for case in cases:
                call.trace_events.clear()
                # Keep the SAME BF16 Accelerator autocast policy, not a new dtype.
                with torch.inference_mode(), trainer.accelerator.autocast():
                    value = call(*reviewed.rebuild(agent, case["record"]))
                traces = copy.deepcopy(call.trace_events)
                if not traces or any(t.get("error") or t.get("model_input") != case["record"]["input"] or
                                     t.get("history_trimmed_messages") for t in traces):
                    raise RuntimeError("Frozen generation input/history changed")
                prediction = {"reply": value} if agent == "response" else value
                row = {"id": case["record"]["id"], "origin": case["evaluation_origin"], "prediction": prediction,
                       "traces": traces, "grade": functional_eval.grade(case, prediction)}
                predictions.append(row); stream.write(json.dumps(row, ensure_ascii=False)+"\n"); stream.flush()
                print(json.dumps({"event": "generation", "candidate": args.worker, "id": row["id"],
                    "prediction": prediction, "pass": row["grade"]["automated_pass"]}, ensure_ascii=False), flush=True)
                budget.check(force=True)
        verify_frozen(args.frozen)
        if any((path / name).stat().st_mtime_ns != stamp for name,stamp in protected_stats.items()):
            raise RuntimeError("Original checkpoint changed during evaluation")
        b.write_new(folder / "SUMMARY.json", {"candidate": candidate, "native_validation": evaluations,
            "functional_cases": len(predictions), "automated_pass": sum(p["grade"]["automated_pass"] for p in predictions),
            "semantic_review": "PENDING", "engine": "actual_Unsloth_production_contract",
            "old_scores_excluded": True, "training_steps": 0, "optimizer_steps": 0, "budget": budget.summary()})
    finally:
        if budget:
            budget.close()
        os.close(lock)


def run(args):
    verify_frozen(args.frozen)
    exact_loss_method()
    identities = contract()
    output = b.output_dir(args.output)
    candidates = b.read(args.frozen / "CANDIDATES.json")["rows"]
    for candidate in candidates:
        agent = candidate["agent"]
        state = b.read(Path(next(c["path"] for c in candidates if c["agent"] == agent and c["name"] == "epoch6")) / "trainer_state.json")
        values = {r["eval_loss"] for r in state["log_history"] if r.get("step") == candidate["step"] and "eval_loss" in r}
        if len(values) != 1:
            raise RuntimeError("Historical native loss reference ambiguous")
        candidate["historical_eval_loss"] = values.pop()
    candidates.sort(key=lambda c: (0 if (c["agent"],c["name"]) == ("recommendation","epoch6") else
        1 if (c["agent"],c["name"]) == ("response","best150") else 2, c["agent"], c["step"]))
    pins = {str(p): b.sha(p) for p in [Path(__file__), Path(production.__file__), Path(b.__file__),
            Path(reviewed.__file__), ROOT / "tools/training_acceleration/functional_eval.py",
            *(b.RELEASE / "common" / n for n in ("core.py", "loss.py", "verification.py", "budget.py")),
            *(ROOT / "src/soomac_irc/soomac_irc" / n for n in (
                "agent_prompts.py", "response_contract.py", "response_model.py", "recommendation_model.py",
                "adapter_runtime.py", "prefix_cache.py", "llm_policy.py", "evaluation_v43.py",
                "input_budget.py", "agent_contract.py"))]}
    b.write_new(output / "PLAN.json", {"actual_training_contract": identities, "candidates": candidates,
        "freeze_sha256": b.sha(args.frozen / "FREEZE.json"),
        "source_sha256": pins, "compute_loss_ast_equal": True, "old_scores_excluded": True,
        "original_eval_replay_gate": {"first": "recommendation:epoch6", "second": "response:best150", "atol": 1e-5, "rtol": 1e-4},
        "validation": "original production Trainer/native loss, bf16=True, tf32=True, bf16_full_eval=False",
        "generation": "same Unsloth model, FastModel.for_inference, BF16 Accelerator autocast, greedy runtime calls",
        "training_steps": 0, "optimizer_steps": 0, "actual_ros": False, "decision_model_called": False})
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    summaries = []
    for index, c in enumerate(candidates, 1):
        if any(b.sha(Path(p)) != expected for p,expected in pins.items()):
            raise RuntimeError("Pinned source changed")
        key = f"{c['agent']}:{c['name']}"
        print(json.dumps({"event":"candidate_start","index":index,"of":len(candidates),"candidate":key}),flush=True)
        with (output / f"{c['agent']}_{c['name']}.log").open("x") as stream:
            completed = subprocess.run([sys.executable,"-u","-B",str(Path(__file__).resolve()),
                "--frozen",str(args.frozen.resolve()),"--output",str(output.resolve()),"--worker",key],
                env=env,stdout=stream,stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f"Evaluation stopped; no automatic retry: {key} exit={completed.returncode}")
        summaries.append(b.read(output / f"{c['agent']}_{c['name']}" / "SUMMARY.json"))
        print(json.dumps({"event":"candidate_complete","candidate":key}),flush=True)
    verify_frozen(args.frozen)
    if any(b.sha(Path(p)) != expected for p,expected in pins.items()):
        raise RuntimeError("Pinned source changed")
    b.write_new(output / "RESULT.json", {"status":"NATIVE_EVAL_COMPLETE_SEMANTIC_REVIEW_PENDING",
        "summaries":summaries,"old_scores_excluded":True,"training_steps":0,"optimizer_steps":0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker")
    args = parser.parse_args()
    worker(args) if args.worker else run(args)
