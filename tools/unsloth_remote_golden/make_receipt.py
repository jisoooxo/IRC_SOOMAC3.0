#!/usr/bin/env python3
"""Build the Decision GPU validation receipt ONLY from parsed smoke evidence (local, CPU, stdlib).

Inputs are the extracted /checkpoints and results/ (manifest.json, training.log) of the two
remote smoke jobs:
  --job1 / --job1-results  fresh smoke (expected nonzero exit after committed step 5)
  --job2 / --job2-results  "새 작업으로 재개" of job1 (finished step 10 + gpu-validation/)
Every PASS field is recomputed here from the evidence files; nothing is copied as a claim.
Writes <out>/VALIDATION_RECEIPT.json + LOSS/GRADIENT/RESTORE evidence (Dockerfile.final context).
Refuses to write anything if any gate fails, and lists every failed gate.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import sys

PROCESS_BUDGET = 32_000_000_000          # production.PROCESS_BUDGET
RESERVE = 1_073_741_824                  # container non-allocator reserve (run_training.sh ENV)
MIN_DRIVER = (580, 65, 6)
GOLDEN_COMPILED = "a8f33a5684d9b3eb3a6867a843e5b33c7a27900075e0683971d2892a8eb2d987"
# Byte copy of the golden run's generated Unified forward. Unsloth bakes
# compile_threads = min(32, max(4, os.cpu_count())) into it
# (unsloth_zoo/temporary_patches/common.py:170-181,280): an Inductor worker count, not math.
GOLDEN_COMPILED_FILE = Path(__file__).resolve().parent / "lock/golden_unsloth_compiled_module_gemma4_unified.py"
GOLDEN_THREADS = b"'compile_threads': 32"


def golden_compiled_match(remote_sha):
    """Return the compile_threads value N if remote == golden with only that literal set to N."""
    golden = GOLDEN_COMPILED_FILE.read_bytes()
    if hashlib.sha256(golden).hexdigest() != GOLDEN_COMPILED or golden.count(GOLDEN_THREADS) != 1:
        raise ValueError("golden compiled copy changed")
    for threads in range(4, 33):
        variant = golden.replace(GOLDEN_THREADS, f"'compile_threads': {threads}".encode())
        if hashlib.sha256(variant).hexdigest() == remote_sha:
            return threads
    return None
GOLDEN_PACKAGES = {"accelerate": "1.14.0", "bitsandbytes": "0.49.2", "peft": "0.19.1",
                   "safetensors": "0.8.0", "torch": "2.12.1+cu130", "transformers": "5.13.1",
                   "triton": "3.7.1", "unsloth": "2026.10.11", "unsloth-zoo": "2026.10.11"}
EXPECTED = {"release_sha256": "ff3285e46bff4f56c18327adb3b038a1010ab5e7482ca59729ab86c1f3a9459e",
            "source_manifest_sha256": "7850948287237c5ace9d1b0f56931419760fbce3f0afa02cea40681acb6ceffa",
            "production_sha256": "962fc3d54bc278e9b42a22da02cb7085197bcf28981af77358fcdeaf0b32a883",  # shared-slot lock copy
            "model_config_sha256": "478c46e8d2c52d5c2d85bf67e3b3e8c90e7c9d91086cee27e3c267907e936bd9",
            "processor_tokenizer_sha256": "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
            "model_builder_sha256": "6c99024a436e92d887fe705a3c9de5291590f8784639d6c86e83faf0b07b517d"}
IDENTITY_KEYS = tuple(EXPECTED) + ("packages",)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def one(paths, what):
    paths = sorted(paths)
    if len(paths) != 1:
        raise ValueError(f"expected exactly one {what}, found {len(paths)}")
    return paths[0]


def safetensors_tensors(path):
    data = Path(path).read_bytes()
    size = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8:8 + size])
    base = 8 + size
    return {name: data[base + meta["data_offsets"][0]:base + meta["data_offsets"][1]]
            for name, meta in header.items() if name != "__metadata__"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job1", type=Path, required=True)
    parser.add_argument("--job2", type=Path, required=True)
    parser.add_argument("--job1-results", type=Path, required=True)
    parser.add_argument("--job2-results", type=Path, required=True)
    parser.add_argument("--tested-image", required=True, help="smoke image ID sha256:...")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    failed = []

    def gate(name, ok):
        if not ok:
            failed.append(name)
        return ok

    # --- the platform's own records: both jobs ran the tested image; job1 failed on purpose
    man1, man2 = (read(r / "manifest.json") for r in (args.job1_results, args.job2_results))
    log1, log2 = ((r / "training.log").read_text(encoding="utf-8", errors="replace")
                  for r in (args.job1_results, args.job2_results))
    gate("image_job1", man1.get("image_id") == args.tested_image)
    gate("image_job2", man2.get("image_id") == args.tested_image)
    state1, state2 = man1.get("state") or {}, man2.get("state") or {}
    gate("job1_exit_nonzero_not_oom", state1.get("ExitCode") not in (0, None)
         and state1.get("OOMKilled") is False)
    gate("job2_exit_zero_not_oom", state2.get("ExitCode") == 0 and state2.get("OOMKilled") is False)
    gate("job2_is_resume", bool(man2.get("resume")))
    gate("job1_failure_was_intentional",
         "intentional smoke failure after committed step 5" in log1
         and "SMOKE_PHASE1 EXPECTED_FAILURE" in log1)
    gate("driver_logged_pass", "SOOMAC_DRIVER_CHECK status=PASS" in log1
         and "SOOMAC_DRIVER_CHECK status=PASS" in log2)
    gate("job2_validation_logged", "SMOKE_PHASE2 resumed smoke reached step 10" in log2
         and '"event": "SOOMAC_GPU_VALIDATION", "pass": true' in log2)

    # --- job1: forced failure after a committed step-5 full state
    marker1 = read(args.job1 / ".lab-monitor/resume.json")
    gate("job1_failure_recorded", read(args.job1 / "FAILURE.json").get("status") == "FAILED")
    checkpoint5 = args.job1 / marker1["checkpoint"]
    gate("job1_marker_step5", marker1.get("global_step") == 5 and checkpoint5.is_dir())
    # job2 restored from a byte-identical copy of job1's committed checkpoint 5
    files5 = sorted(p.relative_to(args.job1) for p in checkpoint5.rglob("*") if p.is_file())
    gate("job2_copy_of_checkpoint5", bool(files5) and all(
        (args.job2 / rel).is_file() and sha(args.job2 / rel) == sha(args.job1 / rel) for rel in files5))

    # --- job2: resumed from 5, completed 10
    identity = read(args.job2 / "INPUT_IDENTITY.json")
    done = read(args.job2 / "training_complete.json")
    marker2 = read(args.job2 / ".lab-monitor/resume.json")
    gate("identity_expected", all(identity.get(k) == v for k, v in EXPECTED.items()))
    gate("identity_golden_packages", identity.get("packages") == GOLDEN_PACKAGES)
    gate("smoke_config", identity.get("mode") == "smoke" and identity.get("engine") == "unsloth"
         and identity.get("agent") == "decision" and identity.get("target_steps") == 10)
    gate("job2_completed_10", done.get("global_step") == 10 and marker2.get("global_step") == 10
         and math.isfinite(float(done.get("train_loss", float("nan")))))
    stage = Path(one((args.job2).glob(f"{marker2['checkpoint'].split('/')[0]}/decision"), "resumed stage"))
    restore_path = stage / "RESTORE_EVIDENCE.json"
    restore = read(restore_path)
    restore_flags = {k: restore.get(k, {}).get("equal") is True
                     for k in ("adapter", "optimizer", "scheduler", "rng")}
    gate("restore_full_state", set(restore) == set(restore_flags) and all(restore_flags.values()))

    # --- optimizer really updated LoRA: step-5 lora_B nonzero, final differs from step 5
    step5 = safetensors_tensors(one(args.job2.glob("attempt-*/decision/step_5_adapter/adapter_model.safetensors"),
                                    "step_5 adapter"))
    final = safetensors_tensors(stage / "smoke_final_adapter/adapter_model.safetensors")
    lora_b = [n for n in step5 if "lora_B" in n]
    b_nonzero = sum(any(step5[n]) for n in lora_b)
    changed = sum(step5[n] != final.get(n) for n in step5)
    gate("optimizer_update", bool(lora_b) and b_nonzero == len(lora_b) and changed > 0
         and set(step5) == set(final))

    # --- GPU memory: every observed process peak under the production receipt budget
    # job2's copy of the job1 output has the root observation overwritten: read each job's own.
    observations = [read(job / "GPU_OBSERVATION.json") for job in (args.job1, args.job2)]
    validation = args.job2 / "gpu-validation"
    env = read(validation / "ENV_EVIDENCE.json")
    peaks = [o.get("observed_process_peak_bytes") for o in observations] + \
            [env["gpu_budget"].get("observed_process_peak_bytes")]
    gate("gpu_peaks_present", bool(observations) and all(isinstance(p, int) and p > 0 for p in peaks))
    gate("gpu_budget_is_receipt_budget", all(
        o.get("limit_bytes") == PROCESS_BUDGET and o.get("non_allocator_reserve_bytes") == RESERVE
        for o in observations + [env["gpu_budget"]]))
    gate("gpu_no_violation", all(o.get("violation") is None for o in observations)
         and env["gpu_budget"].get("violation") is None)
    peak = max(p for p in peaks if isinstance(p, int)) if peaks else 0
    gate("gpu_peak_within_receipt_budget", 0 < peak <= PROCESS_BUDGET)

    # --- driver / device
    driver = tuple(int(x) for x in str(env.get("driver", "0.0")).split("."))
    gate("driver_cuda13", driver >= MIN_DRIVER and str(env.get("torch_cuda")).startswith("13.0"))

    # --- loss: recompute the native1 equality from the numbers
    loss_doc = read(validation / "LOSS_EVIDENCE.json")
    lc = loss_doc.get("loss_check", {})
    values_ok = all(isinstance(lc.get(k), (int, float)) and math.isfinite(lc[k])
                    for k in ("native_selected_ce", "model_loss", "atol", "rtol"))
    ce_equal = values_ok and abs(lc["model_loss"] - lc["native_selected_ce"]) <= min(
        lc["atol"] + lc["rtol"] * abs(lc["native_selected_ce"]), 1e-3)
    gate("loss_native1_equal", ce_equal and lc.get("status") == "PASS")
    gate("loss_selected_labels", loss_doc.get("selected_labels_equal") is True
         and loss_doc.get("selected_targets", 0) > 0)
    gate("loss_identity", all(loss_doc.get(k) == identity.get(k) for k in IDENTITY_KEYS))
    compiled = loss_doc.get("compiled_source_sha256")
    compile_threads = golden_compiled_match(compiled)
    gate("compiled_equals_golden_except_compile_threads", compile_threads is not None)

    # --- gradient: recompute from the census
    grad = read(validation / "GRADIENT_EVIDENCE.json")
    census = grad.get("census", {})
    with_grad = sum(census.get(k, {}).get("with_grad", 0) for k in ("lora_A", "lora_B"))
    gate("gradient_finite_nonzero", grad.get("all_finite") is True and grad.get("loss_finite") is True
         and with_grad == grad.get("trainable_tensors") and census.get("lora_B", {}).get("nonzero", 0) > 0)
    gate("gradient_identity", all(grad.get(k) == identity.get(k) for k in IDENTITY_KEYS))

    if failed:
        print(json.dumps({"status": "FAIL", "failed_gates": failed}, indent=1))
        return 1

    args.out.mkdir(parents=True, exist_ok=False)
    copies = {"loss": validation / "LOSS_EVIDENCE.json",
              "gradient": validation / "GRADIENT_EVIDENCE.json",
              "restore": restore_path}
    files = {}
    for key, source in copies.items():
        if source.stat().st_size > 65536:
            raise ValueError(f"{key} evidence exceeds 64 KiB")
        shutil.copyfile(source, args.out / source.name)
        files[key] = {"path": source.name, "sha256": sha(args.out / source.name)}
    receipt = {
        "schema_version": 1, "status": "PASS", "agent": "decision",
        **{k: identity[k] for k in IDENTITY_KEYS},
        "remote_speed_status": "unmeasured", "remote_hardware_fit_status": "unmeasured",
        "tested_candidate_image_digest": args.tested_image,
        "tested_process_budget_bytes": PROCESS_BUDGET, "tested_process_peak_bytes": peak,
        "min_non_allocator_reserve_bytes": RESERVE,
        "loss_validation": {"one_eval_row_native1_selected_ce_equal": True,
                            "selected_labels_equal": True, "fused0_accepted": False,
                            "training_mode_dropout_equivalence_verified": False,
                            "compiled_source_sha256": compiled,
                            "compiled_source_equals_golden_except_compile_threads": True,
                            "golden_compiled_source_sha256": GOLDEN_COMPILED,
                            "remote_compile_threads": compile_threads,
                            "golden_numeric_equivalence": "UNKNOWN"},
        "gradient_validation": {"finite_nonzero": True, "optimizer_update_tensors_changed": changed},
        "restore_validation": {"forced_failure_resume": True, **restore_flags},
        "driver": env["driver"], "device_name": env.get("device_name"),
        "evidence_files": files,
    }
    (args.out / "VALIDATION_RECEIPT.json").write_text(json.dumps(receipt, indent=1, sort_keys=True))
    print(json.dumps({"status": "PASS", "receipt": str(args.out / "VALIDATION_RECEIPT.json")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
