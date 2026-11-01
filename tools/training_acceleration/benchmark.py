#!/usr/bin/env python3
"""Disposable, same-token baseline/Unsloth benchmark; never resumes production."""

import argparse
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time


RELEASE = Path("/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/releases/conversation_r5_20261101_build05_speed")
BASE = Path("/home/roma/Desktop/sLLM/gemma-4-12B-it")
MANIFEST_SHA = "7850948287237c5ace9d1b0f56931419760fbce3f0afa02cea40681acb6ceffa"
LOCK = Path("/tmp/soomac-r5-gpu.lock")
PROCESS_BUDGET = 32_000_000_000
RESERVE = 512 * 1024**2
EFFECTIVE_BATCH = 16
TRAINABLE_COUNT = 65_568_768


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def process_peak_rss_bytes():
    """Linux process high-water RSS; excludes other jobs and container overhead."""
    if not sys.platform.startswith("linux"):
        return None
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def output_dir(path):
    path = path.absolute()
    for parent in (RELEASE, BASE, Path(__file__).resolve().parent):
        if path == parent or parent in path.parents or path in parent.parents:
            raise ValueError("output overlaps source, model or script")
    if path.exists():
        raise FileExistsError(f"new output directory required: {path}")
    for item in path.parents:
        if item.is_symlink():
            raise ValueError(f"symlinked output parent: {item}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def source_identity(agent):
    release = read(RELEASE / "RELEASE.json")
    manifest_path = RELEASE / "dataset" / "MANIFEST.json"
    if release["source_dataset_manifest_sha256"] != MANIFEST_SHA or sha(manifest_path) != MANIFEST_SHA:
        raise ValueError("latest dataset identity mismatch")
    manifest = read(manifest_path)
    receipts_path = RELEASE / "validation" / "TOKEN_MASK_ROWS.jsonl"
    if sha(receipts_path) != manifest["token_receipts_sha256"]:
        raise ValueError("token receipts mismatch")
    for name in ("core.py", "loss.py", "budget.py"):
        path = RELEASE / "common" / name
        if sha(path) != release["code_files"]["common/" + name]:
            raise ValueError(f"sealed code drift: {name}")
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "processor_config.json"):
        path = RELEASE / "assets" / "model-processor" / name
        if sha(path) != release["processor_files"][name]:
            raise ValueError(f"sealed processor drift: {name}")
    if sha(BASE / "config.json") != release["model_files"]["config.json"]:
        raise ValueError("base model config drift")
    for name in (f"records/{agent}/train.jsonl", f"records/{agent}/validation.jsonl",
                 f"contract/{agent}_schema.json",
                 "contract/response_system_by_route.json" if agent == "response"
                 else "contract/recommendation_system.txt"):
        if sha(RELEASE / "dataset" / name) != manifest["file_sha256"][name]:
            raise ValueError(f"selected dataset input drift: {name}")
    if manifest["max_length"][agent] != (9216 if agent == "response" else 5120):
        raise ValueError("sequence cap changed")
    return release, manifest, receipts_path


def selected_receipts(agent, receipts_path):
    candidates = []
    with receipts_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["agent"] == agent and row["split"] in ("train", "validation"):
                if row["history_trimmed"] or row["errors"] or not row["builder_equal"] or not row["target_mask_equal"]:
                    raise ValueError(f"invalid token receipt: {row['id']}")
                candidates.append(row)
    candidates.sort(key=lambda row: (-row["full_tokens"], row["split"], row["id"]))
    if len(candidates) < EFFECTIVE_BATCH:
        raise ValueError("fewer than 16 receipt rows")
    selected = candidates[:EFFECTIVE_BATCH]
    if len({(row["split"], row["id"]) for row in selected}) != EFFECTIVE_BATCH:
        raise ValueError("duplicate selected receipt")
    return selected


def load_selected_rows(agent, selected):
    wanted = {(row["split"], row["id"]) for row in selected}
    found = {}
    for split in ("train", "validation"):
        path = RELEASE / "dataset" / "records" / agent / f"{split}.jsonl"
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                key = (split, row["id"])
                if key in wanted:
                    if key in found:
                        raise ValueError(f"duplicate dataset row: {key}")
                    found[key] = row
    if set(found) != wanted:
        raise ValueError("selected source rows missing")
    return found


def prepare(agent, destination):
    release, manifest, receipts_path = source_identity(agent)
    selected = selected_receipts(agent, receipts_path)
    rows = load_selected_rows(agent, selected)
    sys.path.insert(0, str(RELEASE / "common"))
    import core
    from transformers import AutoProcessor

    core.DATASET_ROOT = RELEASE / "dataset"
    core.MAX_LENGTH = manifest["max_length"][agent]
    config = core.STAGE_CONFIGS[agent]
    config["schema_path"] = core.DATASET_ROOT / "contract" / f"{agent}_schema.json"
    if agent == "response":
        config["systems_path"] = core.DATASET_ROOT / "contract" / "response_system_by_route.json"
    else:
        config["system_path"] = core.DATASET_ROOT / "contract" / f"{agent}_system.txt"
    processor = AutoProcessor.from_pretrained(RELEASE / "assets" / "model-processor", local_files_only=True)
    prompts, validator = core.load_contract(agent)
    encoded = []
    for receipt in selected:
        row = rows[(receipt["split"], receipt["id"])]
        item = core.encode_record(agent, row, processor, prompts, validator)
        target_count = sum(label != -100 for label in item["labels"])
        if (len(item["input_ids"]) != receipt["full_tokens"]
                or target_count != receipt["target_tokens"]
                or len(item["input_ids"]) - target_count != receipt["input_tokens"]):
            raise ValueError(f"encoding differs from receipt: {receipt['id']}")
        if len(item["input_ids"]) > core.MAX_LENGTH:
            raise ValueError("encoded row exceeds original cap")
        encoded.append({"id": item["id"], "split": receipt["split"],
                        "input_ids": item["input_ids"], "labels": item["labels"]})
    destination = output_dir(destination)
    payload = {"schema_version": 1, "agent": agent, "source_manifest_sha256": MANIFEST_SHA,
               "release_sha256": sha(RELEASE / "RELEASE.json"), "max_length": core.MAX_LENGTH,
               "processor_tokenizer_sha256": release["processor_files"]["tokenizer.json"],
               "selection": "top 16 full-token train/validation receipts; longest row repeated for primary workload",
               "primary_longest_index": 0, "rows": encoded}
    artifact = destination / "encoded.json"
    write_new(artifact, payload)
    receipt = {"status": "CPU_PREPARED_NO_GPU", "encoded_sha256": sha(artifact),
               "selected": [{"id": row["id"], "split": row["split"],
                             "full_tokens": len(row["input_ids"])} for row in encoded],
               "longest_workload": "encoded row 0 repeated 16 times", "model_loaded": False}
    write_new(destination / "PREPARE.json", receipt)
    return receipt


def acquire_gpu_lock():
    fd = os.open(LOCK, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                          "--format=csv,noheader,nounits"], text=True, timeout=10)
        if output.strip():
            raise RuntimeError("GPU compute process already exists")
        return fd
    except BaseException:
        os.close(fd)
        raise


def validate_payload(path, agent):
    source_identity(agent)
    receipt = read(path.parent / "PREPARE.json")
    if sha(path) != receipt["encoded_sha256"]:
        raise ValueError("encoded artifact changed")
    payload = read(path)
    if (payload["agent"] != agent or payload["source_manifest_sha256"] != MANIFEST_SHA
            or payload["release_sha256"] != sha(RELEASE / "RELEASE.json")
            or payload["max_length"] != (9216 if agent == "response" else 5120)
            or len(payload["rows"]) != EFFECTIVE_BATCH or payload["primary_longest_index"] != 0):
        raise ValueError("encoded artifact contract mismatch")
    if any(len(row["input_ids"]) != len(row["labels"]) or len(row["input_ids"]) > payload["max_length"]
           for row in payload["rows"]):
        raise ValueError("invalid encoded shapes")
    return payload


def quantization(torch):
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_quant_storage=torch.bfloat16)


def model_and_optimizer(engine, agent, torch, *, model_dir=None, release_dir=None,
                        max_seq_length=None):
    if agent not in ("response", "recommendation", "decision"):
        raise ValueError("unknown agent")
    model_dir = Path(model_dir) if model_dir is not None else BASE
    release_dir = Path(release_dir) if release_dir is not None else RELEASE
    sequence_cap = max_seq_length if max_seq_length is not None else (5120 if agent == "recommendation" else 9216)
    if sequence_cap != (5120 if agent == "recommendation" else 9216):
        raise ValueError("agent sequence cap changed")
    from bitsandbytes.optim import PagedAdamW8bit
    sys.path.insert(0, str(release_dir / "common"))
    import core
    qconfig = quantization(torch)
    if engine == "baseline":
        from transformers import AutoModelForMultimodalLM
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        model = AutoModelForMultimodalLM.from_pretrained(
            model_dir, device_map={"": 0}, dtype=torch.bfloat16, attn_implementation="sdpa",
            local_files_only=True, low_cpu_mem_usage=True, quantization_config=qconfig)
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={"use_reentrant": True})
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=core.LORA_R,
            lora_alpha=core.LORA_ALPHA, lora_dropout=core.LORA_DROPOUT,
            target_modules=core.LORA_TARGET_MODULES, bias="none"))
    else:
        from unsloth import FastModel
        model, unused_tokenizer = FastModel.from_pretrained(
            model_name=str(model_dir), max_seq_length=sequence_cap,
            dtype=torch.bfloat16, load_in_4bit=True, quantization_config=qconfig,
            device_map={"": 0}, use_gradient_checkpointing="unsloth",
            random_state=42, local_files_only=True, fast_inference=False)
        del unused_tokenizer  # Never encode with Unsloth's returned tokenizer.
        model.config.use_cache = False
        model = FastModel.get_peft_model(
            model, r=core.LORA_R, lora_alpha=core.LORA_ALPHA, lora_dropout=core.LORA_DROPOUT,
            target_modules=core.LORA_TARGET_MODULES, bias="none",
            finetune_vision_layers=False, finetune_language_layers=True,
            finetune_attention_modules=True, finetune_mlp_modules=True,
            use_gradient_checkpointing="unsloth", random_state=42,
            max_seq_length=sequence_cap)
        if model.max_seq_length != sequence_cap:
            raise RuntimeError("Unsloth sequence cap changed")
    model.train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    count = sum(parameter.numel() for parameter in parameters)
    if count != TRAINABLE_COUNT:
        raise RuntimeError(f"LoRA count drift: {count} != {TRAINABLE_COUNT}")
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if (not names or any("lora_" not in name.lower() or "language_model.layers" not in name.lower()
                         for name in names)):
        raise RuntimeError("trainable weights escaped text-decoder LoRA")
    lora = next(iter(model.peft_config.values()))
    if (lora.r != 16 or lora.lora_alpha != 32 or lora.lora_dropout != 0.05
            or lora.bias != "none" or lora.target_modules != core.LORA_TARGET_MODULES):
        raise RuntimeError("LoRA configuration changed")
    expected_vocab = read(model_dir / "config.json")["text_config"]["vocab_size"]
    if model.get_input_embeddings().weight.shape[0] != expected_vocab:
        raise RuntimeError("base vocabulary changed")
    if not getattr(model, "is_loaded_in_4bit", False):
        raise RuntimeError("model is not NF4 4-bit loaded")
    actual_quant = getattr(model.config, "quantization_config", None)
    if hasattr(actual_quant, "to_dict"):
        actual_quant = actual_quant.to_dict()
    if not isinstance(actual_quant, dict) or not (
        actual_quant.get("load_in_4bit") is True
        and actual_quant.get("bnb_4bit_quant_type") == "nf4"
        and actual_quant.get("bnb_4bit_use_double_quant") is True
        and "bfloat16" in str(actual_quant.get("bnb_4bit_compute_dtype"))
    ):
        raise RuntimeError("actual NF4/double-quant/BF16 configuration changed")
    optimizer = PagedAdamW8bit(parameters, lr=5e-5 if agent == "response" else 1e-4,
                               weight_decay=0.01)
    return model, optimizer, parameters, names


def model_builder_fingerprint(release_dir=None):
    """Stable behavior hash; CLI/reporting edits do not invalidate CE evidence."""
    source_dir = Path(release_dir) if release_dir is not None else RELEASE
    parts = {
        "quantization": hashlib.sha256(inspect.getsource(quantization).encode()).hexdigest(),
        "model_and_optimizer": hashlib.sha256(inspect.getsource(model_and_optimizer).encode()).hexdigest(),
        "sealed_core": sha(source_dir / "common/core.py"),
        "sealed_loss": sha(source_dir / "common/loss.py"),
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def one_update(model, optimizer, parameters, inputs, torch, budget, *, measure):
    from accelerate.utils import convert_to_fp32
    from loss import scored_suffix
    losses = []
    budget.check()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for input_batch in inputs:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(**scored_suffix(input_batch), use_cache=False)
            output = convert_to_fp32(output)
            loss = output.loss / EFFECTIVE_BATCH
        losses.append(loss.detach())
        loss.backward()
        del output, loss
    torch.cuda.synchronize()
    compute_seconds = time.perf_counter() - started
    # Checks deliberately occur outside the measured update interval.
    if not bool(torch.isfinite(torch.stack(losses)).all()):
        raise RuntimeError("non-finite loss")
    mean_loss = float(torch.stack(losses).sum().item())
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if (not gradients or not all(bool(torch.isfinite(grad).all()) for grad in gradients)
            or not any(bool(torch.count_nonzero(grad)) for grad in gradients)):
        raise RuntimeError("non-finite or zero gradient")
    torch.cuda.synchronize()
    update_started = time.perf_counter()
    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    seconds = compute_seconds + time.perf_counter() - update_started
    budget.check(force=True)
    return {"seconds": seconds, "mean_loss": mean_loss} if measure else None


def compare_loss_scalars(native, materialized, fused):
    """Keep acceptance independent of Tensor/logit-return implementation."""
    import math
    values = (float(native), float(materialized), float(fused))
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("non-finite native/fused loss comparison")
    return {
        "status": "PASS" if all(math.isclose(value, values[0], abs_tol=1e-3, rel_tol=1e-3)
                                for value in values[1:]) else "DRIFT",
        "native_selected_ce": values[0], "model_loss": values[1],
        "fused_model_loss": values[2],
        "absolute_delta": abs(values[1] - values[0]),
        "fused_absolute_delta": abs(values[2] - values[0]),
        "atol": 1e-3, "rtol": 1e-3,
    }


def check_native_loss(model, input_batch, torch, *, check_fused=False):
    """Diagnostic only: verify exact selected positions against native CE, outside timing."""
    import torch.nn.functional as functional
    from loss import scored_suffix
    selected = scored_suffix(input_batch)
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(**selected, use_cache=False)
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor):
        return {"status": "UNVERIFIED", "reason": "selected logits unavailable"}
    if not isinstance(getattr(output, "loss", None), torch.Tensor):
        return {"status": "UNVERIFIED", "reason": "model loss unavailable"}
    labels = selected["shift_labels"]
    if logits.ndim != 3 or logits.shape[:2] != labels.shape:
        return {"status": "UNVERIFIED", "reason": "selected-logit shape differs from shift_labels",
                "logits_shape": list(logits.shape), "labels_shape": list(labels.shape)}
    native = functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                      labels.reshape(-1), ignore_index=-100)
    actual = output.loss.float()
    if check_fused:
        # Same weights, exact positions, and eval mode: only switch the CE
        # implementation. Never require materialized logits during all training.
        previous = os.environ.get("UNSLOTH_RETURN_LOGITS")
        try:
            os.environ["UNSLOTH_RETURN_LOGITS"] = "0"
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                fused_output = model(**selected, use_cache=False)
            fused_loss = getattr(fused_output, "loss", None)
            if not isinstance(fused_loss, torch.Tensor):
                return {"status": "UNVERIFIED", "reason": "fused model loss unavailable"}
            result = compare_loss_scalars(native.item(), actual.item(), fused_loss.item())
            result.update(selected_positions=int(labels.numel()),
                          comparison="same_model_eval_materialized_CE_vs_fused_shift_labels",
                          fused_logits_materialized=isinstance(getattr(fused_output, "logits", None), torch.Tensor),
                          training_mode_dropout_equivalence_verified=False)
            return result
        finally:
            if previous is None:
                os.environ.pop("UNSLOTH_RETURN_LOGITS", None)
            else:
                os.environ["UNSLOTH_RETURN_LOGITS"] = previous
    if not bool(torch.isfinite(native) and torch.isfinite(actual)):
        raise RuntimeError("non-finite native loss comparison")
    delta = float((actual - native).abs().item())
    result = {"status": "PASS" if bool(torch.isclose(actual, native, atol=1e-3, rtol=1e-3)) else "DRIFT",
              "model_loss": float(actual.item()), "native_selected_ce": float(native.item()),
              "absolute_delta": delta, "atol": 1e-3, "rtol": 1e-3,
              "selected_positions": int(labels.ne(-100).sum().item())}
    return result


def trace_one(model, optimizer, parameters, input_batch, torch, output):
    from accelerate.utils import convert_to_fp32
    from loss import scored_suffix
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    start_reserved = torch.cuda.memory_reserved()
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, record_shapes=True) as profiler:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = convert_to_fp32(model(**scored_suffix(input_batch), use_cache=False))
            loss = result.loss
        loss.backward()
        torch.cuda.synchronize()
    trace_path = output / "attention_trace.json"
    profiler.export_chrome_trace(str(trace_path))
    operators = [event.key for event in profiler.key_averages()
                 if re.search("attn|attention|flash|sdpa|scaled_dot_product", event.key, re.I)]
    model.zero_grad(set_to_none=True)
    return {"trace": trace_path.name, "trace_sha256": sha(trace_path),
            "attention_operators": sorted(set(operators)), "input_ids_dtype": str(input_batch["input_ids"].dtype),
            "attention_operator_evidence": "IDENTIFIED" if operators else "INCONCLUSIVE",
            "autocast_requested_dtype": "torch.bfloat16", "residual_dtype_observed": False,
            "profiler_extra_peak_reserved_bytes":
            max(0, torch.cuda.max_memory_reserved() - start_reserved),
            "measurement": "one separate forward/backward microbatch; excluded from update timing"}


def run(args):
    if args.mode == "full" and args.timed_updates != 3:
        raise ValueError("full comparison requires 3 timed updates")
    if args.mode == "smoke" and args.timed_updates != 1:
        raise ValueError("smoke comparison requires 1 timed update")
    payload = validate_payload(args.encoded, args.agent)
    destination = output_dir(args.output)
    result = {"status": "FAILED", "engine": args.engine, "agent": args.agent,
              "encoded_sha256": sha(args.encoded), "production_training": False}
    from importlib.metadata import version
    result.update(
        release_sha256=sha(RELEASE / "RELEASE.json"),
        source_manifest_sha256=MANIFEST_SHA,
        model_config_sha256=sha(BASE / "config.json"),
        processor_tokenizer_sha256=sha(RELEASE / "assets/model-processor/tokenizer.json"),
        loader_sha256=sha(Path(__file__)),
        model_builder_sha256=model_builder_fingerprint(),
        packages={name: version(name) for name in
                  ("unsloth", "unsloth-zoo", "torch", "transformers", "peft",
                   "accelerate", "bitsandbytes", "safetensors", "triton")},
    )
    fd = None
    budget = None
    try:
        fd = acquire_gpu_lock()
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        if args.engine == "unsloth":
            actual_return_logits = 1 if args.check_loss else args.return_logits
            os.environ["UNSLOTH_RETURN_LOGITS"] = str(actual_return_logits)
            result["unsloth_return_logits"] = actual_return_logits
            import unsloth  # Apply Unsloth patches before importing transformers in this process.
        import torch
        from transformers import AutoProcessor, DataCollatorForSeq2Seq, set_seed
        sys.path.insert(0, str(RELEASE / "common"))
        from budget import Budget
        set_seed(42)
        budget = Budget(PROCESS_BUDGET, RESERVE)
        processor = AutoProcessor.from_pretrained(RELEASE / "assets" / "model-processor", local_files_only=True)
        collator = DataCollatorForSeq2Seq(tokenizer=processor.tokenizer, padding=True,
                                          label_pad_token_id=-100, pad_to_multiple_of=8,
                                          return_tensors="pt")
        longest = payload["rows"][0]
        one = {"input_ids": longest["input_ids"], "labels": longest["labels"]}
        inputs = [{key: value.to("cuda") for key, value in collator([one]).items()}
                  for _ in range(EFFECTIVE_BATCH)]
        model, optimizer, parameters, names = model_and_optimizer(args.engine, args.agent, torch)
        if args.check_loss:
            result["loss_check"] = check_native_loss(model, inputs[0], torch,
                                                     check_fused=args.check_fused and args.engine == "unsloth")
            if not args.check_fused:
                result["loss_check"].update(comparison="same_model_eval_native1_selected_CE",
                                            fused0_accepted=False,
                                            training_mode_dropout_equivalence_verified=False)
            compiled = Path(os.environ.get("UNSLOTH_COMPILE_LOCATION", ".")) / "unsloth_compiled_module_gemma4_unified.py"
            if compiled.is_file():
                result["compiled_source_sha256"] = sha(compiled)
            if result["loss_check"]["status"] != "PASS":
                raise RuntimeError(f"loss check {result['loss_check']['status']}: "
                                   f"{result['loss_check'].get('reason', 'native CE drift')}")
            budget.close()
            budget.check(force=True)
            result.update(budget.summary())
            result["status"] = "LOSS_CHECK_PASS_ONE_ROW_ONLY"
            result["process_peak_rss_bytes"] = process_peak_rss_bytes()
            budget = None
            write_new(destination / "RESULT.json", result)
            return result
        torch.cuda.reset_peak_memory_stats()
        for _ in range(0 if args.mode == "smoke" else 2):
            one_update(model, optimizer, parameters, inputs, torch, budget, measure=False)
        steps = [one_update(model, optimizer, parameters, inputs, torch, budget, measure=True)
                 for _ in range(args.timed_updates)]
        seconds = [item["seconds"] for item in steps]
        mean_losses = [item["mean_loss"] for item in steps]
        nonpad = sum(int(batch["attention_mask"].sum().item()) for batch in inputs)
        padded = sum(int(batch["input_ids"].numel()) for batch in inputs)
        result.update(status="TIMING_PASS_LOSS_EQUIVALENCE_UNVERIFIED", mode=args.mode,
                      timed_update_seconds=seconds, mean_loss_per_update=mean_losses,
                      samples_per_second=EFFECTIVE_BATCH * len(seconds) / sum(seconds),
                      nonpad_tokens_per_second=nonpad * len(seconds) / sum(seconds),
                      padded_tokens_per_second=padded * len(seconds) / sum(seconds),
                      nonpad_tokens_per_update=nonpad, padded_tokens_per_update=padded,
                      warmup_updates=0 if args.mode == "smoke" else 2,
                      timed_updates=len(steps), microbatch=1, effective_batch=EFFECTIVE_BATCH,
                      trainable_count=sum(parameter.numel() for parameter in parameters),
                      trainable_names_sha256=hashlib.sha256("\n".join(names).encode()).hexdigest(),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(), **budget.summary())
        if args.trace or args.mode == "full":
            result["separate_profiler"] = trace_one(model, optimizer, parameters, inputs[0], torch, destination)
            budget.check(force=True)
        if args.save_snapshot:
            snap = destination / "snapshot"
            snap.mkdir()
            model.save_pretrained(snap / "adapter")
            torch.save(optimizer.state_dict(), snap / "optimizer.pt")
            result["snapshot"] = "disposable only; save/resume compatibility unverified"
            budget.check(force=True)
        budget.close()
        budget.check(force=True)
        result.update(budget.summary())
        result["process_peak_rss_bytes"] = process_peak_rss_bytes()
        budget = None
        write_new(destination / "RESULT.json", result)
        return result
    except BaseException as error:
        result["status"] = "FAILED"
        result["error"] = f"{type(error).__name__}: {error}"
        result["process_peak_rss_bytes"] = process_peak_rss_bytes()
        write_new(destination / "FAILURE.json", result)
        raise
    finally:
        if budget is not None:
            budget.close()
        if fd is not None:
            os.close(fd)


def parse(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--agent", choices=("response", "recommendation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoded", type=Path)
    parser.add_argument("--engine", choices=("baseline", "unsloth"))
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--timed-updates", type=int)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--save-snapshot", action="store_true")
    parser.add_argument("--check-loss", action="store_true",
                        help="separate diagnostic run without timed updates")
    parser.add_argument("--check-fused", action="store_true",
                        help="with --check-loss, compare fused0 as a separate diagnostic")
    parser.add_argument("--return-logits", type=int, choices=(0, 1), default=0,
                        help="Unsloth compiled logits mode; default 0 preserves fused path")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        if args.encoded or args.engine or args.trace or args.save_snapshot or args.check_loss or args.check_fused or args.return_logits != 0:
            parser.error("prepare accepts only agent and output")
    elif args.encoded is None or args.engine is None:
        parser.error("run requires encoded and engine")
    if args.check_loss and (args.trace or args.save_snapshot or args.mode != "smoke" or args.timed_updates):
        parser.error("--check-loss is a separate smoke diagnostic without timing, trace or snapshot")
    if args.check_fused and not args.check_loss:
        parser.error("--check-fused requires --check-loss")
    args.timed_updates = args.timed_updates or (1 if args.mode == "smoke" else 3)
    return args


def main(argv=None):
    args = parse(argv)
    result = prepare(args.agent, args.output) if args.command == "prepare" else run(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
