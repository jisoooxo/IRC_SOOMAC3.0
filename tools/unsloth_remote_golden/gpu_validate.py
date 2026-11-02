#!/usr/bin/env python3
"""One-row Decision GPU validation for the golden-candidate smoke image. No training, no save.

Runs in the SAME container right after the resumed smoke finished (smoke_entry.sh).
Reuses production's own release checks, environment, encoder, budget and model builder,
then writes three bounded JSON files into <output>:
  LOSS_EVIDENCE.json      native1 selected-logit CE vs model loss, selected-label identity,
                          generated Unified forward SHA (compared to the local golden run)
  GRADIENT_EVIDENCE.json  one-row LoRA gradient census (finite, nonzero, per-kind counts)
  ENV_EVIDENCE.json       driver/CUDA/device, import graph facts, memory peaks
Exit 0 only if loss, labels and gradients pass and the GPU budget was not violated.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import sys
from types import SimpleNamespace

sys.path.insert(0, "/app/tools/unsloth_training")
import production  # noqa: E402  (also puts benchmark and sealed common on sys.path)
import benchmark  # noqa: E402

# Generated Unified forward of the running local golden Response job
# (runs/r5_unsloth_local_authorized_20261101/response/INPUT_IDENTITY.json and
# response_unsloth_native_loss3/RESULT.json). Model-architecture code, not data.
GOLDEN_COMPILED_SHA256 = "a8f33a5684d9b3eb3a6867a843e5b33c7a27900075e0683971d2892a8eb2d987"
MAX_EVIDENCE_BYTES = 65536


def driver_version():
    try:
        line = Path("/proc/driver/nvidia/version").read_text().splitlines()[0]
    except (OSError, IndexError):
        return None
    match = re.search(r"\b(\d{3}\.\d+(?:\.\d+)?)\b", line)
    return match.group(1) if match else None


def write(path, value):
    data = json.dumps(value, indent=1, sort_keys=True).encode()
    if len(data) > MAX_EVIDENCE_BYTES:
        raise ValueError(f"evidence too large: {path.name}")
    path.write_bytes(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    output.mkdir(exist_ok=False)

    source = SimpleNamespace(
        execution_environment="container", release=Path("/app/RELEASE.json"),
        model=Path(os.environ.get("MODEL_DIR", "/model")),
        dataset=Path(os.environ.get("DATASET_DIR", "/app/assets/dataset")),
        processor=Path("/app/assets/model-processor"))
    production.configure_source(source)
    production.configure_environment(source)
    unused_release, manifest, release_sha = production.source_preflight("decision")
    identity = {
        "agent": "decision", "release_sha256": release_sha,
        "source_manifest_sha256": production.SOURCE_SHA,
        "model_config_sha256": production.sha(production.BASE / "config.json"),
        "processor_tokenizer_sha256": production.sha(production.PROCESSOR / "tokenizer.json"),
        "model_builder_sha256": benchmark.model_builder_fingerprint(production.RELEASE),
        "production_sha256": production.sha(Path(production.__file__)),
        "packages": production.versions(),
    }
    lock = production.acquire_gpu_lock(shared_slot=os.environ.get("GPU_MODE") == "shared"
                                        and os.environ.get("GPU_MEMORY_BUDGET_BYTES", "").isdigit())

    # Same order as production.execute: Unsloth patches precede Torch/Transformers/PEFT.
    import unsloth  # noqa: F401
    import torch
    from transformers import DataCollatorForSeq2Seq, set_seed
    from budget import Budget
    from loss import scored_suffix

    requested = min(int(os.environ["SOOMAC_REMOTE_REQUESTED_BUDGET_BYTES"]),
                    int(os.environ["GPU_MEMORY_BUDGET_BYTES"]))
    budget = Budget(requested, int(os.environ["SOOMAC_NON_ALLOCATOR_RESERVE_BYTES"]))
    selection = production.smoke_selection("decision", manifest)
    processor, rows = production.encode_all("decision", manifest, selection)
    row = rows["validation"][0]
    set_seed(42)
    model, unused_optimizer, parameters, names = benchmark.model_and_optimizer(
        "unsloth", "decision", torch, model_dir=production.BASE,
        release_dir=production.RELEASE, max_seq_length=production.EXPECTED["decision"]["max_length"])
    budget.check(force=True)
    compiled = Path(os.environ["UNSLOTH_COMPILE_LOCATION"]) / "unsloth_compiled_module_gemma4_unified.py"
    compiled_sha = production.sha(compiled) if compiled.is_file() else None

    collator = DataCollatorForSeq2Seq(tokenizer=processor.tokenizer, padding=True,
                                      label_pad_token_id=-100, pad_to_multiple_of=8,
                                      return_tensors="pt")
    batch = {key: value.to("cuda") for key, value in collator([row]).items()}
    labels = row["labels"]
    expected_targets = [label for label in labels[1:] if label != -100]
    shift = scored_suffix(batch)["shift_labels"].reshape(-1).tolist()
    actual_targets = [label for label in shift if label != -100]
    selected_labels_equal = bool(expected_targets) and expected_targets == actual_targets

    loss_check = benchmark.check_native_loss(model, batch, torch)
    budget.check(force=True)

    model.train()
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(**scored_suffix(batch), use_cache=False)
    train_loss = out.loss.float()
    train_loss.backward()
    del out
    census = {"lora_A": [0, 0, 0], "lora_B": [0, 0, 0]}  # [count, with_grad, nonzero]
    finite = True
    for name, parameter in zip(names, parameters):
        kind = "lora_A" if "lora_A" in name else "lora_B" if "lora_B" in name else None
        if kind is None:
            raise RuntimeError(f"unexpected trainable tensor: {name}")
        census[kind][0] += 1
        if parameter.grad is not None:
            census[kind][1] += 1
            finite = finite and bool(torch.isfinite(parameter.grad).all())
            census[kind][2] += int(bool(torch.count_nonzero(parameter.grad)))
    budget.check(force=True)
    peak_reserved = torch.cuda.max_memory_reserved()
    budget.close()
    budget.check(force=True)
    summary = dict(budget.summary(), violation=budget.violation)
    os.close(lock)

    nonzero = census["lora_A"][2] + census["lora_B"][2]
    with_grad = census["lora_A"][1] + census["lora_B"][1]
    gradient = {
        "schema_version": 1, **identity, "row_id_index": 0, "row_split": "validation",
        "train_mode_loss": float(train_loss.item()), "loss_finite": bool(torch.isfinite(train_loss)),
        "trainable_tensors": len(names), "tensors_with_grad": with_grad,
        "census": {k: dict(zip(("count", "with_grad", "nonzero"), v)) for k, v in census.items()},
        "all_finite": finite, "nonzero_tensors": nonzero,
        "finite_nonzero": bool(finite and nonzero > 0 and with_grad == len(names)),
        "note": "fresh LoRA (B=0) => lora_A grads are expected zero; optimizer update proof is the smoke adapters",
    }
    loss = {
        "schema_version": 1, **identity, "status": loss_check.get("status"),
        "comparison": "same_model_eval_native1_selected_CE", "loss_check": loss_check,
        "selected_labels_equal": selected_labels_equal, "selected_targets": len(actual_targets),
        "row_length": len(row["input_ids"]), "unsloth_return_logits": os.environ.get("UNSLOTH_RETURN_LOGITS"),
        "fused0_accepted": False, "training_mode_dropout_equivalence_verified": False,
        "compiled_source_sha256": compiled_sha,
        "golden_compiled_source_sha256": GOLDEN_COMPILED_SHA256,
        "compiled_source_equals_golden": compiled_sha == GOLDEN_COMPILED_SHA256,
        "golden_numeric_equivalence": "UNKNOWN: no local golden Decision run on identical inputs",
    }
    modules = sorted(name for name in sys.modules if "." not in name)
    from unsloth.models import _utils as unsloth_utils
    env = {
        "schema_version": 1, "driver": driver_version(), "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "device_name": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "device_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "unsloth_xformers_is_none": getattr(unsloth_utils, "xformers", "absent") is None,
        "xformers_in_sys_modules": "xformers" in sys.modules,
        "sentence_transformers_in_sys_modules": "sentence_transformers" in sys.modules,
        "top_level_modules": modules,
        "top_level_modules_sha256": hashlib.sha256("\n".join(modules).encode()).hexdigest(),
        "torch_peak_reserved_bytes": peak_reserved,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "gpu_budget": summary,
        "gpu_memory_budget_env": os.environ.get("GPU_MEMORY_BUDGET_BYTES"),
    }
    write(output / "LOSS_EVIDENCE.json", loss)
    write(output / "GRADIENT_EVIDENCE.json", gradient)
    write(output / "ENV_EVIDENCE.json", env)
    ok = (loss["status"] == "PASS" and selected_labels_equal and gradient["finite_nonzero"]
          and summary["violation"] is None)
    print(json.dumps({"event": "SOOMAC_GPU_VALIDATION", "pass": ok,
                      "compiled_source_equals_golden": loss["compiled_source_equals_golden"],
                      "loss_check": loss_check, "nonzero_tensors": nonzero}), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
