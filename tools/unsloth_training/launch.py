#!/usr/bin/env python3
"""Start Recommendation only after verified Response completion and Q gates."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import production


def verified_response(path):
    return production.verify_response_completion(path)


def wait_response_service(unit, output):
    """Observe only the named job. Never stop/restart it or treat failure as done."""
    if not re.fullmatch(r"soomac-unsloth-response-[A-Za-z0-9_-]+\.service", unit):
        raise ValueError("an explicit local Response service is required")
    while True:
        raw = subprocess.check_output(
            ["systemctl", "--user", "show", unit, "-p", "LoadState", "-p", "ActiveState",
             "-p", "Result", "-p", "ExecMainStatus", "-p", "MainPID"], text=True)
        state = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
        if state.get("LoadState") != "loaded":
            raise RuntimeError("Response service not found; no automatic Q start")
        if state.get("ActiveState") in ("active", "activating", "deactivating"):
            print(json.dumps({"event": "waiting_for_response", "unit": unit, **state}), flush=True)
            time.sleep(60)
            continue
        if (state.get("ActiveState") != "inactive" or state.get("Result") != "success"
                or state.get("ExecMainStatus") != "0" or state.get("MainPID") != "0"):
            raise RuntimeError(f"Response did not exit normally: {state}")
        return


def prepare_recommendation(args):
    """Reuse existing checks after verified P completion; no alternate trainer."""
    root = args.recommendation_preflight_root.absolute()
    if args.recommendation_smoke_output.absolute() != root / "smoke":
        raise ValueError("new Q smoke must be inside this preflight root")
    production.validate_output(root)
    root.mkdir(parents=True, exist_ok=False)
    tools = Path(__file__).resolve().parents[1]

    def run_logged(command, filename, expected_failure=False):
        with (root / filename).open("x") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if expected_failure:
            if result.returncode == 0 or "intentional smoke failure after committed step 5" not in (root / filename).read_text():
                raise RuntimeError("Q smoke did not produce the expected saved-step failure")
        elif result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, command)

    run_logged([sys.executable, "-u", str(tools / "training_acceleration/quality_gate.py"),
                "local-objective", "--agent", "recommendation", "--prepared", str(args.recommendation_prepared),
                "--output", str(root / "objective")], "objective.log")
    objective = production.read(root / "objective/LOCAL_OBJECTIVE.json")
    if (objective.get("status") != "LOCAL_REPRESENTATIVE_OBJECTIVE_PASS_NOT_FUNCTIONAL_QUALITY_PASS"
            or objective.get("agent") != "recommendation" or objective.get("max_length") != 5120
            or {r["category"] for r in objective["cases"]} != {"short", "long", "multiturn"}):
        raise ValueError("Q representative objective gate incomplete")
    smoke = [sys.executable, "-u", str(Path(production.__file__).resolve()), "--mode", "smoke",
             "--agent", "recommendation", "--batch", "1", "--output", str(root / "smoke"),
             "--benchmark-result", str(args.recommendation_benchmark_result), "--smoke-max-steps", "10"]
    run_logged(smoke + ["--test-fail-step", "5"], "smoke_save.log", expected_failure=True)
    config = production.read(root / "smoke/INPUT_IDENTITY.json")
    checkpoint, step = production.resume_marker(root / "smoke", config)
    if step != 5:
        raise ValueError("Q expected a verified step5 checkpoint")
    old_sha = production.sha(checkpoint / "adapter_model.safetensors")
    run_logged(smoke + ["--resume", "auto"], "smoke_resume.log")
    production.verify_smoke_output(root / "smoke", "recommendation")
    if production.sha(checkpoint / "adapter_model.safetensors") != old_sha:
        raise ValueError("original Q smoke checkpoint was changed")
    run_logged([sys.executable, "-u", str(tools / "training_acceleration/adapter_reload.py"),
                "--agent", "recommendation", "--smoke", str(root / "smoke"),
                "--prepared", str(args.recommendation_prepared), "--output", str(root / "peft_reload")], "peft_reload.log")
    reload = production.read(root / "peft_reload/PEFT_RELOAD.json")
    if (reload.get("status") != "STANDARD_PEFT_RELOAD_AND_FORWARD_PASS_NOT_FUNCTIONAL_QUALITY_PASS"
            or reload.get("agent") != "recommendation" or reload.get("all_adapter_tensors_exactly_loaded") is not True):
        raise ValueError("Q standard PEFT reload gate incomplete")
    # CPU tensors only: prove actual saved parameter updates, not merely a finite loss.
    from safetensors.torch import load_file
    import torch
    done = production.read(root / "smoke/training_complete.json")
    final = root / "smoke" / done["checkpoint"] / "adapter_model.safetensors"
    before, after = load_file(str(checkpoint / "adapter_model.safetensors")), load_file(str(final))
    if (before.keys() != after.keys() or not any(not torch.equal(before[k], after[k]) for k in before)
            or not any(bool(torch.count_nonzero(v)) for k, v in after.items() if "lora_B" in k)
            or not all(bool(torch.isfinite(v).all()) for v in after.values())):
        raise ValueError("Q saved adapter update missing or nonfinite")
    production.atomic_json(root / "READY.json", {"status": "LOCAL_Q_PREFLIGHT_PASS_NOT_FUNCTIONAL_QUALITY_PASS",
        "objective_sha256": production.sha(root / "objective/LOCAL_OBJECTIVE.json"),
        "peft_reload_sha256": production.sha(root / "peft_reload/PEFT_RELOAD.json"),
        "smoke_receipt_sha256": production.sha(root / "smoke/training_complete.json")})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response-output", type=Path, required=True)
    parser.add_argument("--recommendation-output", type=Path, required=True)
    parser.add_argument("--recommendation-benchmark-result", type=Path, required=True)
    parser.add_argument("--recommendation-smoke-output", type=Path, required=True)
    parser.add_argument("--recommendation-loss-evidence", type=Path, required=True)
    parser.add_argument("--recommendation-batch", type=int, choices=(1,), required=True)
    parser.add_argument("--wait-response-unit", required=True, help="wait read-only for this systemd user job to exit successfully")
    parser.add_argument("--recommendation-preflight-root", type=Path, required=True,
                        help="new directory: run existing Q objective/save/resume/reload checks after P exits")
    parser.add_argument("--recommendation-prepared", type=Path, required=True)
    args = parser.parse_args(argv)
    wait_response_service(args.wait_response_unit, args.response_output.absolute())
    verified_response(args.response_output.absolute())
    production.verify_benchmark_result(args.recommendation_benchmark_result.absolute(), "recommendation")
    prepare_recommendation(args)
    production.verify_smoke_output(args.recommendation_smoke_output.absolute(), "recommendation")
    command = [sys.executable, str(Path(production.__file__).resolve()),
               "--mode", "train", "--agent", "recommendation",
               "--output", str(args.recommendation_output.absolute()),
               "--batch", str(args.recommendation_batch),
               "--benchmark-result", str(args.recommendation_benchmark_result.absolute()),
               "--smoke-output", str(args.recommendation_smoke_output.absolute()),
               "--loss-evidence", str(args.recommendation_loss_evidence.absolute()),
               "--response-output", str(args.response_output.absolute())]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
