#!/usr/bin/env python3
"""Fail-closed Unsloth LoRA training from a sealed local/container release.

Only ``preflight`` is CPU-only. ``smoke`` and ``train`` must be launched by the
GPU owner after the disposable benchmark has validated this exact model path.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent
sys.path.insert(0, str(TOOLS / "training_acceleration"))
import benchmark

RELEASE = benchmark.RELEASE
BASE = benchmark.BASE
DATASET = RELEASE / "dataset"
PROCESSOR = RELEASE / "assets/model-processor"
COMMON = HERE.parents[1] / "common"
if not COMMON.is_dir():
    COMMON = RELEASE / "common"
sys.path.insert(0, str(COMMON))
from checkpointing import (SOURCE_SHA, atomic_json, member, plain, publish, read,
                           resume_marker, sha, verify_checkpoint, verify_completion,
                           verify_release)

PROCESS_BUDGET = 32_000_000_000
RESERVE = 512 * 1024**2
EFFECTIVE_BATCH = 16
EXPECTED = {
    "decision": {"train": 4360, "validation": 309, "max_length": 9216,
                 "learning_rate": 1e-4, "target_steps": 1638},
    "response": {"train": 1664, "validation": 152, "max_length": 9216,
                 "learning_rate": 5e-5, "target_steps": 624},
    "recommendation": {"train": 623, "validation": 30, "max_length": 5120,
                       "learning_rate": 1e-4, "target_steps": 234},
}
ISOLATED = Path("/tmp/soomac-training-accel-20261101")
DURABLE_VENV = Path("/home/roma/IRC_SOOMAC3.0/training/learning_first_v43/training_release_20261101_r5/env")
PACKAGES = ("unsloth", "unsloth-zoo", "torch", "transformers", "peft",
            "accelerate", "bitsandbytes", "safetensors", "triton")


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "smoke", "train"), required=True)
    parser.add_argument("--agent", choices=tuple(EXPECTED), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release", type=Path, default=RELEASE / "RELEASE.json")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--model", type=Path, default=BASE)
    parser.add_argument("--processor", type=Path)
    parser.add_argument("--budget-bytes", type=int, default=PROCESS_BUDGET)
    parser.add_argument("--non-allocator-reserve-bytes", type=int)
    parser.add_argument("--execution-environment", choices=("local", "container"), default="local")
    parser.add_argument("--batch", type=int, choices=(1,), required=True)
    parser.add_argument("--resume", choices=("none", "auto"), default="none")
    parser.add_argument("--benchmark-result", type=Path)
    parser.add_argument("--smoke-output", type=Path)
    parser.add_argument("--response-output", type=Path)
    parser.add_argument("--validation-receipt", type=Path)
    parser.add_argument("--loss-evidence", type=Path)
    parser.add_argument("--smoke-max-steps", type=int)
    parser.add_argument("--test-fail-step", type=int, default=0)
    args = parser.parse_args(argv)
    if args.mode == "train" and (args.smoke_max_steps is not None or args.test_fail_step):
        parser.error("full training forbids smoke step/failure overrides")
    if args.mode == "smoke":
        args.smoke_max_steps = args.smoke_max_steps or 10
        if args.smoke_max_steps not in (5, 10):
            parser.error("smoke steps must be 5 or 10")
        if args.test_fail_step not in (0, 5, 10) or args.test_fail_step > args.smoke_max_steps:
            parser.error("smoke failure must follow a saved step 5 or 10")
    if args.mode == "preflight" and (args.smoke_max_steps is not None or args.test_fail_step):
        parser.error("preflight forbids smoke overrides")
    if args.agent != "decision" and args.mode in ("smoke", "train") and args.benchmark_result is None:
        parser.error("GPU modes require a passing, agent-specific benchmark result")
    if args.agent != "decision" and args.mode == "train" and args.smoke_output is None:
        parser.error("full training requires a native save/resume-verified smoke output")
    if args.agent == "decision" and args.mode == "train" and args.validation_receipt is None:
        parser.error("Decision train requires same-image validation receipt")
    if args.agent != "decision" and args.mode == "train" and args.loss_evidence is None:
        parser.error("local full training requires bounded fused-loss evidence")
    if args.mode == "train" and args.agent == "recommendation" and args.response_output is None:
        parser.error("Recommendation full training requires verified Response output")
    if args.response_output is not None and (args.mode != "train" or args.agent != "recommendation"):
        parser.error("Response prerequisite applies only to Recommendation full training")
    if os.environ.get("LAB_TEST_FAIL_AFTER_GLOBAL_STEP") and args.mode != "smoke":
        parser.error("production refuses injected failure environment")
    if args.non_allocator_reserve_bytes is None:
        args.non_allocator_reserve_bytes = (1024**3 if args.execution_environment == "container"
                                            else RESERVE)
    if (args.budget_bytes <= 2 * 1024**3 or not 512 * 1024**2 <= args.non_allocator_reserve_bytes
            < args.budget_bytes):
        parser.error("invalid process budget or non-allocator reserve")
    args.output = args.output.absolute()
    args.release = args.release.absolute()
    args.model = args.model.absolute()
    args.dataset = (args.dataset or args.release.parent / "dataset").absolute()
    args.processor = (args.processor or args.release.parent / "assets/model-processor").absolute()
    if args.benchmark_result is not None:
        args.benchmark_result = args.benchmark_result.absolute()
    if args.smoke_output is not None:
        args.smoke_output = args.smoke_output.absolute()
    if args.response_output is not None:
        args.response_output = args.response_output.absolute()
    if args.validation_receipt is not None:
        args.validation_receipt = args.validation_receipt.absolute()
    if args.loss_evidence is not None:
        args.loss_evidence = args.loss_evidence.absolute()
    if os.environ.get("LAB_RESUME", "0") not in ("0", "1"):
        parser.error("LAB_RESUME must be 0 or 1")
    if os.environ.get("LAB_RESUME") == "1":
        if (args.resume != "auto" or not os.environ.get("RESUME_CHECKPOINT_DIR")
                or not os.environ.get("RESUME_STEP", "").isdigit()):
            parser.error("server resume requires auto marker plus checkpoint path and step")
    return args


def validate_output(path: Path):
    path = path.absolute()
    if ".." in path.parts:
        raise ValueError("output path traversal is forbidden")
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("output path includes symlink")
    path = path.resolve(strict=False)
    for protected in (RELEASE, DATASET, BASE, HERE):
        if path == protected or protected in path.parents or path in protected.parents:
            raise ValueError("output overlaps model, sealed release, or training source")
    return path


def prepare_output(args):
    """Accept only a new local path or an empty container checkpoint mount."""
    path = args.output
    if args.resume == "none":
        if args.execution_environment == "local":
            if path.exists():
                raise FileExistsError("fresh local output must not exist")
            path.mkdir(parents=True, exist_ok=False)
        else:
            plain(path, directory=True)
            for child in path.iterdir():
                if child.name == "lost+found" and child.is_dir() and not child.is_symlink():
                    if any(child.iterdir()):
                        raise FileExistsError("nonempty platform lost+found in fresh mount")
                elif child.name == ".lab-monitor" and child.is_dir() and not child.is_symlink():
                    if any(child.iterdir()):
                        raise FileExistsError("prior progress or resume state in fresh mount")
                else:
                    raise FileExistsError("fresh checkpoint mount contains prior state")
        args.fresh_output_prevalidated = True
    else:
        plain(path, directory=True)


def configure_source(args):
    global RELEASE, BASE, DATASET, PROCESSOR
    RELEASE, BASE, DATASET, PROCESSOR = (args.release.parent, args.model,
                                          args.dataset, args.processor)
    if args.release.name != "RELEASE.json":
        raise ValueError("release must be a sealed RELEASE.json")
    if args.processor != RELEASE / "assets/model-processor":
        raise ValueError("processor must be the sealed release processor")


def configure_environment(args):
    """Set offline, non-runtime caches before Unsloth or Torch imports."""
    cache_root = ISOLATED if args.execution_environment == "local" else Path("/tmp/soomac-unsloth-container")
    cache = {
        "UNSLOTH_COMPILE_LOCATION": cache_root / "compiled",
        "TORCHINDUCTOR_CACHE_DIR": cache_root / "torchinductor",
        "TRITON_CACHE_DIR": cache_root / "triton",
        "HF_HOME": cache_root / "hf",
    }
    for key, value in cache.items():
        value.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(value)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", UNSLOTH_DISABLE_STATISTICS="1",
                      TOKENIZERS_PARALLELISM="false", UNSLOTH_RETURN_LOGITS="1")


def versions():
    return {name: importlib.metadata.version(name) for name in PACKAGES}


def source_preflight(agent):
    release, manifest, release_sha = verify_release(RELEASE / "RELEASE.json", BASE, DATASET)
    spec = EXPECTED[agent]
    if any(manifest["counts"][agent][split] != spec[split] for split in ("train", "validation")):
        raise ValueError("pinned dataset row counts changed")
    if manifest["max_length"][agent] != spec["max_length"]:
        raise ValueError("pinned sequence cap changed")
    if release["source_dataset_manifest_sha256"] != SOURCE_SHA:
        raise ValueError("latest source dataset changed")
    # No target text, weights, or encoded rows leave the local machine.
    return release, manifest, release_sha


def verify_benchmark_result(path, agent, *, budget_bytes=PROCESS_BUDGET, reserve_bytes=RESERVE):
    item = read(path)
    if (item.get("status") != "TIMING_PASS_LOSS_EQUIVALENCE_UNVERIFIED"
            or item.get("engine") != "unsloth" or item.get("agent") != agent
            or item.get("production_training") is not False
            or item.get("release_sha256") != sha(RELEASE / "RELEASE.json")
            or item.get("source_manifest_sha256") != SOURCE_SHA
            or item.get("model_config_sha256") != sha(BASE / "config.json")
            or item.get("processor_tokenizer_sha256") != sha(PROCESSOR / "tokenizer.json")
            or item.get("model_builder_sha256") != benchmark.model_builder_fingerprint(RELEASE)
            or item.get("packages") != versions()
            or item.get("limit_bytes") != budget_bytes
            or item.get("non_allocator_reserve_bytes") != reserve_bytes
            or item.get("unsloth_return_logits") != 1
            or item.get("effective_batch") != EFFECTIVE_BATCH
            or item.get("trainable_count") != benchmark.TRAINABLE_COUNT
            or not math.isfinite(float(item.get("samples_per_second", 0)))
            or float(item.get("samples_per_second", 0)) <= 0):
        raise ValueError("agent-specific Unsloth benchmark gate not satisfied")
    return sha(path)


def verify_loss_evidence(path, args, config):
    """One-row native1 selected-logit CE check, not train-dropout equivalence."""
    item = read(path)
    expected = {"status": "LOSS_CHECK_PASS_ONE_ROW_ONLY", "engine": "unsloth",
                "agent": args.agent, "production_training": False,
                "release_sha256": config["release_sha256"],
                "source_manifest_sha256": config["source_manifest_sha256"],
                "model_config_sha256": config["model_config_sha256"],
                "processor_tokenizer_sha256": config["processor_tokenizer_sha256"],
                "model_builder_sha256": config["model_builder_sha256"],
                "packages": config["packages"]}
    if any(item.get(key) != value for key, value in expected.items()):
        raise ValueError("one-row loss evidence source/model/package identity mismatch")
    loss = item.get("loss_check", {})
    if (item.get("unsloth_return_logits") != 1
            or loss.get("status") != "PASS"
            or loss.get("comparison", "same_model_eval_native1_selected_CE")
            != "same_model_eval_native1_selected_CE"
            or loss.get("training_mode_dropout_equivalence_verified", False) is not False
            or loss.get("fused0_accepted", False) is not False
            or type(loss.get("selected_positions")) is not int
            or loss["selected_positions"] <= 0
            or re.fullmatch(r"[0-9a-f]{64}", str(item.get("compiled_source_sha256"))) is None):
        raise ValueError("one-row native/fused CE proof incomplete")
    for key in ("native_selected_ce", "model_loss", "absolute_delta", "atol", "rtol"):
        if not isinstance(loss.get(key), (int, float)) or not math.isfinite(float(loss[key])):
            raise ValueError(f"nonfinite loss proof value: {key}")
    tolerance = float(loss["atol"]) + float(loss["rtol"]) * abs(float(loss["native_selected_ce"]))
    measured_delta = abs(float(loss["native_selected_ce"]) - float(loss["model_loss"]))
    if (float(loss["atol"]) < 0 or float(loss["rtol"]) < 0
            or float(loss["absolute_delta"]) < 0
            or tolerance <= 0 or measured_delta > min(tolerance, 1e-3)
            or not math.isclose(float(loss["absolute_delta"]), measured_delta,
                                rel_tol=0, abs_tol=1e-7)):
        raise ValueError("native1 selected-logit CE scalar mismatch")
    if args.benchmark_result is not None:
        timing = read(args.benchmark_result)
        if timing.get("encoded_sha256") != item.get("encoded_sha256"):
            raise ValueError("timing/loss evidence used different encoded inputs")
    return sha(path)


def verify_smoke_output(path, agent):
    from checkpointing import verify_completion
    done = verify_completion(path, RELEASE / "RELEASE.json", agent, BASE)
    config = done["config"]
    if config.get("engine") != "unsloth" or config.get("mode") != "smoke":
        raise ValueError("smoke completion belongs to another engine/mode")
    checkpoint = member(path, done["checkpoint"], directory=True)
    evidence = read(checkpoint.parent / "RESTORE_EVIDENCE.json")
    if set(evidence) != {"adapter", "optimizer", "scheduler", "rng"}:
        raise ValueError("smoke lacked native full-state restore evidence")
    if not all(item.get("equal") is True for item in evidence.values()):
        raise ValueError("smoke full-state restore hashes disagree")
    return sha(path / "training_complete.json")


def verify_validation_receipt(path, args, config):
    """Bind Decision train to the tested candidate image's evidence, not throughput."""
    receipt = read(path)
    exact = {
        "schema_version": 1, "status": "PASS", "agent": "decision",
        "release_sha256": config["release_sha256"],
        "source_manifest_sha256": config["source_manifest_sha256"],
        "model_config_sha256": config["model_config_sha256"],
        "processor_tokenizer_sha256": config["processor_tokenizer_sha256"],
        "production_sha256": config["production_sha256"],
        "model_builder_sha256": config["model_builder_sha256"],
        "packages": config["packages"],
        "remote_speed_status": "unmeasured",
        "remote_hardware_fit_status": "unmeasured",
    }
    if any(receipt.get(key) != value for key, value in exact.items()):
        raise ValueError("Decision validation receipt source/code/package identity mismatch")
    digest = receipt.get("tested_candidate_image_digest")
    if not isinstance(digest, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
        raise ValueError("receipt lacks tested candidate image digest")
    if (type(receipt.get("tested_process_budget_bytes")) is not int
            or receipt["tested_process_budget_bytes"] != PROCESS_BUDGET
            or type(receipt.get("tested_process_peak_bytes")) is not int
            or not 0 < receipt["tested_process_peak_bytes"] <= receipt["tested_process_budget_bytes"]
            or type(receipt.get("min_non_allocator_reserve_bytes")) is not int
            or args.non_allocator_reserve_bytes < receipt["min_non_allocator_reserve_bytes"]):
        raise ValueError("remote memory receipt/reserve policy invalid")
    loss = receipt.get("loss_validation", {})
    gradient = receipt.get("gradient_validation", {})
    restore = receipt.get("restore_validation", {})
    if (loss.get("one_eval_row_native1_selected_ce_equal") is not True
            or loss.get("selected_labels_equal") is not True
            or loss.get("fused0_accepted") is not False
            or loss.get("training_mode_dropout_equivalence_verified") is not False
            or re.fullmatch(r"[0-9a-f]{64}", str(loss.get("compiled_source_sha256"))) is None
            or gradient.get("finite_nonzero") is not True
            or restore.get("forced_failure_resume") is not True
            or any(restore.get(key) is not True for key in
                   ("adapter", "optimizer", "scheduler", "rng"))):
        raise ValueError("Decision receipt lacks native loss/gradient/full-state proof")
    files = receipt.get("evidence_files")
    if not isinstance(files, dict) or set(files) != {"loss", "gradient", "restore"}:
        raise ValueError("receipt needs three bounded evidence files")
    for item in files.values():
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ValueError("invalid evidence-file descriptor")
        if sha(member(path.parent, item["path"])) != item["sha256"]:
            raise ValueError("bundled validation evidence changed")
    return sha(path)


def verify_response_completion(path):
    from checkpointing import verify_completion
    done = verify_completion(path, RELEASE / "RELEASE.json", "response", BASE)
    config = done["config"]
    if (config.get("engine") != "unsloth" or config.get("mode") != "train"
            or config.get("epochs") != 6
            or done.get("global_step") != EXPECTED["response"]["target_steps"]
            or done.get("best_adapter_status") != "provisional_eval_loss_semantic_review_pending"):
        raise ValueError("Response is not verified complete under Unsloth")
    checkpoint = member(path, done["checkpoint"], directory=True)
    state = read(checkpoint / "trainer_state.json")
    if abs(float(state.get("epoch", -1)) - 6.0) > 1e-6:
        raise ValueError("Response did not complete all six epochs")
    return sha(path / "training_complete.json")


def configuration(args, release_sha):
    spec = EXPECTED[args.agent]
    smoke = args.mode == "smoke"
    config = {
        "schema_version": 1, "engine": "unsloth", "agent": args.agent,
        "mode": args.mode, "release_sha256": release_sha,
        "source_manifest_sha256": SOURCE_SHA,
        "model_config_sha256": sha(BASE / "config.json"),
        "processor_tokenizer_sha256": sha(PROCESSOR / "tokenizer.json"),
        "model_builder_sha256": benchmark.model_builder_fingerprint(RELEASE),
        "production_sha256": sha(Path(__file__)),
        "packages": versions(), "python_prefix": str(Path(sys.prefix).resolve()),
        "max_length": spec["max_length"],
        "train_rows": spec["train"], "validation_rows": spec["validation"],
        "epochs": 1 if smoke else 6,
        "target_steps": args.smoke_max_steps if smoke else spec["target_steps"],
        "batch": args.batch, "accumulation": 1 if smoke else EFFECTIVE_BATCH // args.batch,
        "effective_batch": args.batch if smoke else EFFECTIVE_BATCH,
        "learning_rate": spec["learning_rate"], "scheduler": "constant" if smoke else "cosine",
        "warmup_ratio": 0 if smoke else 0.03,
        "optimizer": "paged_adamw_8bit", "weight_decay": 0.01,
        "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
        "lora_target_modules": "sealed-core-text-decoder-only",
        "quantization": "nf4-double-quant-bf16",
        "gradient_checkpointing": "unsloth", "history_trimmed": False,
        "seed": 42, "data_seed": 42,
        "requested_budget_bytes": args.budget_bytes,
        "non_allocator_reserve_bytes": args.non_allocator_reserve_bytes,
        "unsloth_return_logits": 1,
        "loss_weighting": "microbatch_mean_divide_by_accumulation",
    }
    if smoke:
        config["smoke_selection"] = smoke_selection(args.agent, read(DATASET / "MANIFEST.json"))
    return config


def resume_source(args, config):
    identity = args.output / "INPUT_IDENTITY.json"
    marker = args.output / ".lab-monitor/resume.json"
    if args.resume == "none":
        if args.output.exists() and not getattr(args, "fresh_output_prevalidated", False):
            raise FileExistsError("fresh output must not exist; never overwrite prior training")
        if identity.exists() or marker.exists():
            raise FileExistsError("fresh output contains prior training identity or resume marker")
        return None, 0
    if not identity.is_file() or not marker.is_file():
        raise ValueError("resume needs existing identity and committed full-state marker")
    if read(identity) != config:
        raise ValueError("resume model/dataset/package/config identity mismatch")
    checkpoint, step = resume_marker(args.output, config)
    if os.environ.get("LAB_RESUME") == "1":
        if (checkpoint != Path(os.environ["RESUME_CHECKPOINT_DIR"]).absolute()
                or step != int(os.environ["RESUME_STEP"])):
            raise ValueError("server resume checkpoint/step differs from committed marker")
    if step >= config["target_steps"]:
        raise ValueError("resume marker already at target; verify completion separately")
    return checkpoint, step


def clone_resume(source, attempt, agent, config):
    """Rebase HF best-checkpoint path without modifying original full state."""
    receipt = verify_checkpoint(source, config)
    dest = attempt / agent
    dest.mkdir(parents=True, exist_ok=False)
    names = {source.name}
    if receipt.get("best_checkpoint_name"):
        names.add(receipt["best_checkpoint_name"])
    for name in sorted(names):
        old = member(source.parent, name, directory=True)
        verify_checkpoint(old, config)
        for child in old.rglob("*"):
            plain(child, directory=child.is_dir())
        new = dest / name
        shutil.copytree(old, new)
        state = read(new / "trainer_state.json")
        if state.get("best_model_checkpoint"):
            state["best_model_checkpoint"] = str(dest / Path(state["best_model_checkpoint"]).name)
        atomic_json(new / "trainer_state.json", state)
        from checkpointing import seal_checkpoint
        seal_checkpoint(new, config)
    return dest / source.name


def acquire_gpu_lock(shared_slot=False):
    """Refuse any other GPU compute process, except in a server-assigned shared slot.

    In a lab shared slot (container mode, server GPU_MODE=shared plus a server byte budget)
    the other tenant on the same physical GPU is expected: the platform reserves the slot and
    Budget caps and observes only our own process. Local and exclusive runs keep the refusal.
    """
    fd = os.open(benchmark.LOCK, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        observed = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                            "--format=csv,noheader,nounits"], text=True, timeout=10)
        if observed.strip():
            if not shared_slot:
                raise RuntimeError("another GPU compute process exists")
            print(json.dumps({"event": "shared_slot_other_gpu_processes",
                              "count": len(observed.split())}), flush=True)
        return fd
    except BaseException:
        os.close(fd)
        raise


def smoke_selection(agent, manifest):
    """Pin longest split-specific sealed receipt IDs without encoding full splits."""
    receipts_path = RELEASE / "validation/TOKEN_MASK_ROWS.jsonl"
    if sha(receipts_path) != manifest["token_receipts_sha256"]:
        raise ValueError("sealed token receipts changed")
    candidates = {"train": [], "validation": []}
    with receipts_path.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            if item["agent"] != agent or item["split"] not in candidates:
                continue
            if (item["history_trimmed"] or item["errors"] or not item["builder_equal"]
                    or not item["target_mask_equal"]):
                raise ValueError("invalid sealed token receipt")
            candidates[item["split"]].append(item)
    selected = {}
    for split, limit in (("train", 16), ("validation", 8)):
        ordered = sorted(candidates[split], key=lambda item: (-item["full_tokens"], item["id"]))
        selected[split] = ordered[:limit]
        if len(selected[split]) != limit or len({item["id"] for item in selected[split]}) != limit:
            raise ValueError("insufficient unique sealed smoke rows")
    identity = {"receipt_sha256": sha(receipts_path),
                "train": [item["id"] for item in selected["train"]],
                "validation": [item["id"] for item in selected["validation"]]}
    identity["selection_sha256"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return identity


def select_smoke_raw(raw, split, smoke_identity):
    by_id = {item["id"]: item for item in raw}
    if len(by_id) != len(raw) or not set(smoke_identity[split]) <= set(by_id):
        raise ValueError("sealed smoke row missing or duplicate")
    return [by_id[row_id] for row_id in smoke_identity[split]]


def enforce_microbatch_mean(trainer, config):
    """Use native per-microbatch CE, then Trainer's accumulation divisor.

    HF Trainer documents this False override for custom loss that does not use
    num_items_in_batch. Our scored-suffix loss is a one-row token mean; passing
    the cross-microbatch token count would change its weighting.
    """
    if (config["batch"] != 1 or config["accumulation"] not in (1, EFFECTIVE_BATCH)
            or trainer.args.per_device_train_batch_size != 1
            or trainer.args.gradient_accumulation_steps != config["accumulation"]
            or trainer.args.n_gpu != 1 or trainer.args.world_size != 1
            or trainer.args.average_tokens_across_devices
            or trainer.compute_loss_func is not None or trainer.label_smoother is not None
            or type(trainer.model_accepts_loss_kwargs) is not bool):
        raise RuntimeError("native one-row microbatch loss prerequisites changed")
    trainer.model_accepts_loss_kwargs = False


def encode_all(agent, manifest, smoke_identity=None):
    """Use only the sealed processor and sealed core encoder; no Unsloth tokenizer."""
    import core
    from transformers import AutoProcessor
    core.DATASET_ROOT = DATASET
    core.MAX_LENGTH = manifest["max_length"][agent]
    cfg = core.STAGE_CONFIGS[agent]
    cfg.update(train_path=DATASET / "records" / agent / "train.jsonl",
               validation_path=DATASET / "records" / agent / "validation.jsonl",
               schema_path=DATASET / "contract" / f"{agent}_schema.json")
    if agent == "response":
        cfg["systems_path"] = DATASET / "contract/response_system_by_route.json"
    else:
        cfg["system_path"] = DATASET / "contract" / f"{agent}_system.txt"
    processor = AutoProcessor.from_pretrained(PROCESSOR, local_files_only=True)
    prompts, validator = core.load_contract(agent)
    rows = {}
    receipts = None
    if smoke_identity is not None:
        actual = smoke_selection(agent, manifest)
        if actual != smoke_identity:
            raise ValueError("smoke row identity changed")
        receipts = {}
        with (RELEASE / "validation/TOKEN_MASK_ROWS.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                item = json.loads(line)
                if (item["agent"] == agent and item["split"] in smoke_identity
                        and item["id"] in smoke_identity[item["split"]]):
                    receipts[(item["split"], item["id"])] = item
    for split in ("train", "validation"):
        raw = core.read_jsonl(cfg[f"{split}_path"])
        if len(raw) != EXPECTED[agent][split]:
            raise ValueError(f"{agent} {split} row count drift")
        if smoke_identity is not None:
            raw = select_smoke_raw(raw, split, smoke_identity)
        rows[split] = [core.encode_record(agent, row, processor, prompts, validator) for row in raw]
        if any(item["length"] > core.MAX_LENGTH for item in rows[split]):
            raise ValueError("encoded row exceeds pinned cap")
        if receipts is not None:
            for item in rows[split]:
                receipt = receipts.get((split, item["id"]))
                scored = sum(label != -100 for label in item["labels"])
                if (receipt is None or item["length"] != receipt["full_tokens"]
                        or scored != receipt["target_tokens"]
                        or item["length"] - scored != receipt["input_tokens"]):
                    raise ValueError("smoke encoding differs from sealed receipt")
        rows[split] = [{"input_ids": item["input_ids"], "labels": item["labels"]}
                       for item in rows[split]]
    return processor, rows


def execute(args, manifest, config, source, start, progress):
    # Unsloth patching must precede every Torch/Transformers/PEFT import.
    import unsloth  # noqa: F401
    import torch
    from torch.utils.data import Dataset
    from transformers import (DataCollatorForSeq2Seq, TrainerCallback,
                              TrainingArguments, set_seed)
    from budget import Budget
    from loss import scored_suffix
    from verification import VerifiedTrainer
    import core

    class EncodedDataset(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            return self.rows[index]

    budget = Budget(args.budget_bytes, args.non_allocator_reserve_bytes)
    budget_closed = False
    attempt = args.output / ("attempt-" + uuid.uuid4().hex)
    stage_dir = attempt / args.agent
    try:
        if source is not None:
            source = clone_resume(source, attempt, args.agent, config)
        else:
            stage_dir.mkdir(parents=True, exist_ok=False)
        processor, rows = encode_all(args.agent, manifest, config.get("smoke_selection"))
        set_seed(42)
        model, optimizer, parameters, names = benchmark.model_and_optimizer(
            "unsloth", args.agent, torch, model_dir=BASE, release_dir=RELEASE,
            max_seq_length=EXPECTED[args.agent]["max_length"])
        if args.mode == "train":
            compiled = (Path(os.environ["UNSLOTH_COMPILE_LOCATION"])
                        / "unsloth_compiled_module_gemma4_unified.py")
            if not compiled.is_file() or sha(compiled) != config["compiled_source_sha256"]:
                raise RuntimeError("generated Unified forward differs from accepted native1 loss proof")
        if len(names) == 0 or len(parameters) == 0:
            raise RuntimeError("missing text-only LoRA parameters")
        collator = DataCollatorForSeq2Seq(tokenizer=processor.tokenizer, padding=True,
                                          label_pad_token_id=-100, pad_to_multiple_of=8,
                                          return_tensors="pt")
        smoke = args.mode == "smoke"
        target = config["target_steps"]
        checks = {"seen": 0, "finite": True, "nonzero": False, "loss": False}
        finite = torch.ones((), dtype=torch.bool, device="cuda")
        nonzero = torch.zeros((), dtype=torch.bool, device="cuda")

        def gradient_check(gradient):
            checks["seen"] += 1
            finite.logical_and_(torch.isfinite(gradient).all())
            nonzero.logical_or_(torch.count_nonzero(gradient) > 0)
            return gradient

        hooks = [parameter.register_hook(gradient_check) for parameter in parameters]

        class Callback(TrainerCallback):
            def __init__(self):
                self.trainer = None
                self.started = None
                self.seconds = 0.0
                self.skipped = 0

            def on_train_begin(self, training_args, state, control, **kwargs):
                progress.write(force=True, phase="training", global_step=int(state.global_step),
                               samples_seen=self.trainer.lab_samples,
                               tokens_seen=self.trainer.lab_tokens)

            def on_step_begin(self, training_args, state, control, **kwargs):
                if training_args.device.type == "cuda":
                    torch.cuda.synchronize(training_args.device)
                self.started = time.perf_counter()

            def on_step_end(self, training_args, state, control, **kwargs):
                if training_args.device.type == "cuda":
                    torch.cuda.synchronize(training_args.device)
                self.seconds += time.perf_counter() - self.started
                self.skipped += int(getattr(self.trainer.accelerator,
                                            "optimizer_step_was_skipped", False))
                budget.check(force=True)
                step = int(state.global_step)
                if step == 5 or step == target:
                    control.should_save = True
                progress.write(phase="training", global_step=step,
                               training_seconds=self.seconds,
                               samples_seen=self.trainer.lab_samples,
                               tokens_seen=self.trainer.lab_tokens,
                               skipped_updates=self.skipped)

            def on_epoch_end(self, training_args, state, control, **kwargs):
                if not smoke:
                    control.should_evaluate = True
                    control.should_save = True

            def on_log(self, training_args, state, control, logs=None, **kwargs):
                if logs and "loss" in logs:
                    value = float(logs["loss"])
                    if not math.isfinite(value):
                        raise RuntimeError("non-finite training loss")
                    checks["loss"] = True
                    progress.write(loss=value)
                if int(state.global_step) >= min(10, target) and hooks:
                    checks["finite"] = bool(finite.item())
                    checks["nonzero"] = bool(nonzero.item())
                    if not checks["loss"] or not checks["seen"] or not checks["finite"] or not checks["nonzero"]:
                        raise RuntimeError(f"LoRA gradient/loss gate failed: {checks}")
                    for hook in hooks:
                        hook.remove()
                    hooks.clear()

        class Training(VerifiedTrainer):
            def __init__(self, *positional, **named):
                self.lab_samples = 0
                self.lab_tokens = 0
                super().__init__(*positional, **named)

            def training_step(self, model, inputs, num_items_in_batch=None):
                if "input_ids" not in inputs:
                    raise ValueError("input_ids required for accounting")
                self.lab_samples += int(inputs["input_ids"].shape[0])
                if "attention_mask" in inputs:
                    self.lab_tokens += int(inputs["attention_mask"].sum().item())
                return super().training_step(model, inputs, num_items_in_batch)

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

            def _save_checkpoint(self, *positional, **named):
                progress.write(force=True, phase="saving")
                result = super()._save_checkpoint(*positional, **named)
                step = int(self.state.global_step)
                checkpoint = stage_dir / f"checkpoint-{step}"
                name = (f"step_{step}_adapter" if smoke else
                        f"step_{step}_epoch_{float(self.state.epoch or 0):.3f}_adapter")
                core.copy_adapter_checkpoint(checkpoint, stage_dir / name, processor)
                publish(args.output, checkpoint, config, step)
                if args.test_fail_step and step >= args.test_fail_step:
                    raise RuntimeError(f"intentional smoke failure after committed step {step}")
                progress.write(force=True, phase="training")
                return result

            def evaluate(self, *positional, **named):
                progress.write(force=True, phase="evaluating")
                try:
                    return super().evaluate(*positional, **named)
                finally:
                    progress.write(force=True, phase="training")

        training_args = TrainingArguments(
            output_dir=str(stage_dir), do_train=True, do_eval=True,
            num_train_epochs=config["epochs"], max_steps=target if smoke else -1,
            per_device_train_batch_size=args.batch, per_device_eval_batch_size=1,
            gradient_accumulation_steps=config["accumulation"],
            learning_rate=config["learning_rate"],
            lr_scheduler_type=config["scheduler"],
            warmup_steps=0 if smoke else math.ceil(target * 0.03),
            weight_decay=0.01, optim="paged_adamw_8bit", max_grad_norm=1.0,
            bf16=True, tf32=True, gradient_checkpointing=False, use_cache=False,
            eval_strategy="steps", save_strategy="steps",
            eval_steps=5 if smoke else 50, save_steps=5 if smoke else 50,
            load_best_model_at_end=True, metric_for_best_model="eval_loss",
            greater_is_better=False, save_total_limit=None,
            logging_strategy="steps", logging_steps=1 if smoke else 5,
            logging_first_step=True, logging_nan_inf_filter=False,
            report_to="none", seed=42, data_seed=42,
            dataloader_num_workers=0, remove_unused_columns=False,
            average_tokens_across_devices=False,
        )
        callback = Callback()
        trainer = Training(model=model, args=training_args,
                           train_dataset=EncodedDataset(rows["train"]),
                           eval_dataset=EncodedDataset(rows["validation"]),
                           data_collator=collator, callbacks=[callback],
                           optimizers=(optimizer, None))
        callback.trainer = trainer
        inferred_loss_kwargs = trainer.model_accepts_loss_kwargs
        enforce_microbatch_mean(trainer, config)
        print(json.dumps({"event": "loss_weighting_configured",
                          "inferred_model_accepts_loss_kwargs": inferred_loss_kwargs,
                          "active_model_accepts_loss_kwargs": trainer.model_accepts_loss_kwargs,
                          "microbatch": args.batch, "accumulation": config["accumulation"]}),
              flush=True)
        progress.write(force=True, phase="training", global_step=start, target_step=target,
                       start_step=start)
        try:
            result = trainer.train(resume_from_checkpoint=str(source) if source else None)
        finally:
            for hook in hooks:
                hook.remove()
        budget.check(force=True)
        if not math.isfinite(float(result.metrics.get("train_loss", float("nan")))):
            raise RuntimeError("training loss missing or non-finite")
        if not checks["loss"] or not checks["finite"] or not checks["nonzero"]:
            raise RuntimeError(f"training safety gate did not pass: {checks}")
        if int(trainer.state.global_step) != target:
            raise RuntimeError("trainer did not complete configured optimizer steps")
        if not smoke and (trainer.state.epoch is None or abs(float(trainer.state.epoch) - 6.0) > 1e-6):
            raise RuntimeError("trainer did not complete all six epochs")
        checkpoint, step = resume_marker(args.output, config)
        if step != target:
            raise RuntimeError("final full-state checkpoint was not committed")
        best_raw = trainer.state.best_model_checkpoint
        if not best_raw:
            raise RuntimeError("eval_loss did not select a best checkpoint")
        best = Path(best_raw)
        best.relative_to(stage_dir)
        verify_checkpoint(best, config)
        if source:
            evidence = read(stage_dir / "RESTORE_EVIDENCE.json")
            if set(evidence) != {"adapter", "optimizer", "scheduler", "rng"}:
                raise RuntimeError("native restore evidence incomplete")
        core.copy_adapter_checkpoint(best, stage_dir / "best_adapter", processor)
        final_name = "smoke_final_adapter" if smoke else "epoch_6_adapter"
        core.copy_adapter_checkpoint(checkpoint, stage_dir / final_name, processor)
        core.write_epoch_metrics(stage_dir, trainer.state.log_history, str(best))
        exports = {}
        for folder in ("best_adapter", final_name):
            for name in ("adapter_config.json", "adapter_model.safetensors"):
                file = member(stage_dir / folder, name)
                exports[file.relative_to(args.output).as_posix()] = sha(file)
        done = {"schema_version": 1, "config": config, "global_step": step,
                "checkpoint": checkpoint.relative_to(args.output).as_posix(),
                "exports": exports, "train_loss": result.metrics["train_loss"],
                "samples_seen": trainer.lab_samples, "tokens_seen": trainer.lab_tokens,
                "training_seconds": callback.seconds, "skipped_updates": callback.skipped,
                "best_adapter_status": "provisional_eval_loss_semantic_review_pending"}
        budget.check(force=True)
        budget.close()
        budget_closed = True
        budget.check(force=True)
        atomic_json(args.output / "GPU_OBSERVATION.json",
                    dict(budget.summary(), violation=budget.violation))
        atomic_json(args.output / "training_complete.json", done)
        verify_completion(args.output, RELEASE / "RELEASE.json", args.agent, BASE)
        progress.write(force=True, phase="completed", global_step=step)
    finally:
        if not budget_closed:
            budget.close()
            atomic_json(args.output / "GPU_OBSERVATION.json",
                        dict(budget.summary(), violation=budget.violation))


def run_validated(args, progress):
    release, manifest, release_sha = source_preflight(args.agent)
    config = configuration(args, release_sha)
    if args.mode == "train" and args.agent == "recommendation":
        config["prerequisite_response_completion_sha256"] = verify_response_completion(args.response_output)
    if args.mode == "preflight":
        result = {"status": "CPU_PREFLIGHT_PASS", "agent": args.agent,
                  "release_sha256": release_sha, "counts": manifest["counts"][args.agent],
                  "max_length": manifest["max_length"][args.agent],
                  "model_loaded": False, "training_started": False}
        progress.write(force=True, phase="completed")
        print(json.dumps(result, sort_keys=True), flush=True)
        return
    receipt_sha = None
    if args.agent == "decision" and args.mode == "train":
        receipt_sha = verify_validation_receipt(args.validation_receipt, args, config)
        config["validation_receipt_sha256"] = receipt_sha
        config["compiled_source_sha256"] = read(args.validation_receipt)["loss_validation"]["compiled_source_sha256"]
    loss_sha = None
    if args.agent != "decision" and args.mode == "train":
        loss_sha = verify_loss_evidence(args.loss_evidence, args, config)
        config["loss_evidence_sha256"] = loss_sha
        config["compiled_source_sha256"] = read(args.loss_evidence)["compiled_source_sha256"]
    source, start = resume_source(args, config)
    benchmark_sha = (verify_benchmark_result(args.benchmark_result, args.agent,
                                             budget_bytes=args.budget_bytes,
                                             reserve_bytes=args.non_allocator_reserve_bytes)
                     if args.agent != "decision" else None)
    if args.mode == "train" and args.agent != "decision":
        smoke_sha = verify_smoke_output(args.smoke_output, args.agent)
    else:
        smoke_sha = None
    gates = {"schema_version": 1, "benchmark_result_sha256": benchmark_sha,
             "smoke_completion_sha256": smoke_sha,
             "loss_evidence_sha256": loss_sha,
             "validation_receipt_sha256": receipt_sha}
    if source is not None and read(args.output / "ENTRY_GATES.json") != gates:
        raise ValueError("resume benchmark/smoke evidence changed")
    if source is None:
        atomic_json(args.output / "INPUT_IDENTITY.json", config)
        atomic_json(args.output / "ENTRY_GATES.json", gates)
        atomic_json(args.output / "SOURCE_PROVENANCE.json", {
            "schema_version": 1, "loader_full_file_sha256": sha(Path(benchmark.__file__)),
            "model_builder_sha256": config["model_builder_sha256"]})
    lock = None
    try:
        shared_slot = (args.execution_environment == "container"
                       and os.environ.get("GPU_MODE") == "shared"
                       and os.environ.get("GPU_MEMORY_BUDGET_BYTES", "").isdigit())
        lock = acquire_gpu_lock(shared_slot=shared_slot)
        execute(args, manifest, config, source, start, progress)
    finally:
        if lock is not None:
            os.close(lock)


def main(argv=None):
    args = parse(argv)
    if args.execution_environment == "container" and (
            HERE != Path("/app/tools/unsloth_training")
            or args.release != Path("/app/RELEASE.json")):
        raise RuntimeError("container mode is reserved for the sealed /app image")
    if (args.execution_environment == "local"
            and Path(sys.prefix).resolve() not in
            ((ISOLATED / "venv").resolve(), DURABLE_VENV.resolve())):
        raise RuntimeError("use only isolated Unsloth venv; original gemma4_env is read-only")
    configure_source(args)
    args.output = validate_output(args.output)
    prepare_output(args)
    os.environ["CHECKPOINT_DIR"] = str(args.output)
    from lab_progress import Progress
    progress = Progress()
    progress.start_heartbeat(10)
    try:
        configure_environment(args)
        run_validated(args, progress)
    except BaseException:
        atomic_json(args.output / "FAILURE.json", {
            "status": "FAILED", "automatic_retry": False,
            "resume_marker_preserved": (args.output / ".lab-monitor/resume.json").is_file()})
        # Spec 2.9 has no failed phase; nonzero exit and FAILURE.json signal failure.
        progress.write(force=True)
        raise
    finally:
        progress.close()


if __name__ == "__main__":
    main()
