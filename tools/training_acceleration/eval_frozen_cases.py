#!/usr/bin/env python3
"""Run frozen D/Q/P quality cases with actual PEFT adapters after GPU training.

Preflight is CPU-only. Inference refuses an occupied GPU, never changes the
runtime adapter paths, and writes a new, immutable result directory. Automated
grades are diagnostics, not independent Korean semantic approval.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/soomac_irc"))

import benchmark
import functional_eval

ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = ("agent_prompts.py", "prompt_examples.py", "prompt_contrast_examples.py",
                "model_runtime.py", "adapter_runtime.py", "prefix_cache.py",
                "decision_model.py", "recommendation_model.py", "response_model.py",
                "response_contract.py", "llm_policy.py", "evaluation_v43.py")


def source_hashes() -> dict[str, str]:
    folder = ROOT / "src/soomac_irc/soomac_irc"
    return {name: benchmark.sha(folder / name) for name in SOURCE_FILES} | {
        "eval_frozen_cases.py": benchmark.sha(Path(__file__)),
        "functional_eval.py": benchmark.sha(Path(functional_eval.__file__)),
    }


def frozen_cases(path: Path) -> list[dict]:
    receipt = benchmark.read(path.parent / "NEW_HOLDOUT_RECEIPT.json")
    if benchmark.sha(path) != receipt["cases_sha256"]:
        raise ValueError("frozen quality cases changed")
    cases = benchmark.read(path)["rows"]
    identifiers = [case["record"]["id"] for case in cases]
    if (len(cases) != 24 or len(set(identifiers)) != 24 or
            {agent: sum(case["agent"] == agent for case in cases)
             for agent in ("decision", "recommendation", "response")} !=
            {"decision": 8, "recommendation": 8, "response": 8}):
        raise ValueError("quality case membership changed")
    return cases


def call_arguments(case: dict) -> tuple[str, tuple]:
    """Rebuild the runtime input and fail if it differs from the frozen case."""
    from soomac_irc.decision_model import build_decision_model_input
    from soomac_irc.response_contract import (
        build_recommendation_model_input, build_response_model_input,
    )

    agent, record = case["agent"], case["record"]
    source = copy.deepcopy(record["input"])
    session = {
        "order": copy.deepcopy(source.get("confirmed_order", source.get("order"))),
        "history": copy.deepcopy(source.get("recent_history") or []),
        "preferences": copy.deepcopy(source["preferences"]),
        "pending": copy.deepcopy(source.get("pending")),
        "action_history": copy.deepcopy(source.get("recent_action_history") or []),
    }
    robot = copy.deepcopy(source["robot_state"])
    if agent == "decision":
        args = (session, source["message"], robot)
        rebuilt = build_decision_model_input(*args)
    elif agent == "recommendation":
        args = (session, {"recommendation": {"action": source["recommendation_action"]}},
                robot, source["current_user_text"])
        rebuilt = build_recommendation_model_input(*args)
    elif agent == "response":
        args = (source["user_text"], session, source["policy"],
                source["applied_this_turn"], source["future_changes"],
                source["recommendation_result"], source["next_prompt"], robot, record["route"])
        rebuilt = build_response_model_input(*args)
    else:
        raise ValueError(f"unknown quality agent: {agent}")
    if rebuilt != source:
        raise ValueError(f"runtime input drift: {record['id']}")
    return agent, args


def predict_one(case: dict, calls: dict) -> dict:
    agent, args = call_arguments(case)
    if agent == "decision":
        calls[agent](*args)
        prediction = copy.deepcopy(calls[agent].last_external_output)
        if not isinstance(prediction, dict):
            raise ValueError(f"Decision returned no sparse JSON: {case['record']['id']}")
    elif agent == "response":
        prediction = {"reply": calls[agent](*args)}
    else:
        prediction = calls[agent](*args)
    return {"id": case["record"]["id"], "prediction": prediction}


def adapter_identity(paths: dict[str, Path]) -> dict:
    result = {}
    for agent, path in paths.items():
        path = path.resolve(strict=True)
        names = ("adapter_config.json", "adapter_model.safetensors")
        if not all((path / name).is_file() for name in names):
            raise ValueError(f"incomplete {agent} PEFT adapter: {path}")
        if (path / "CONVERSION.json").exists():
            raise ValueError(f"{agent}: use original PEFT export, not vLLM conversion")
        result[agent] = {"path": str(path),
                         "file_sha256": {name: benchmark.sha(path / name) for name in names}}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true", help="CPU-only frozen input check")
    parser.add_argument("--out", type=Path, help="new directory; required for inference")
    for agent in ("decision", "recommendation", "response"):
        parser.add_argument(f"--{agent}-adapter", type=Path)
    args = parser.parse_args()
    cases = frozen_cases(args.cases)
    for case in cases:
        call_arguments(case)
    if args.preflight:
        print(json.dumps({"status": "CPU_PREFLIGHT_PASS_NOT_MODEL_QUALITY",
                          "cases": len(cases), "cases_sha256": benchmark.sha(args.cases)}))
        return
    if args.out is None:
        parser.error("--out is required for inference")
    paths = {agent: getattr(args, f"{agent}_adapter")
             for agent in ("decision", "recommendation", "response")}
    if any(path is None for path in paths.values()):
        parser.error("all three original PEFT adapter paths are required")
    identity = adapter_identity(paths)
    resolved_paths = {agent: Path(identity[agent]["path"]) for agent in paths}
    source_identity = source_hashes()
    cases_identity = benchmark.sha(args.cases)
    lock = benchmark.acquire_gpu_lock()
    try:
        output = benchmark.output_dir(args.out)
        benchmark.write_new(output / "IDENTITY.json", {
            "cases_sha256": cases_identity, "adapters": identity,
            "source_sha256": source_identity,
            "process_id": os.getpid(), "training": False, "actual_ros": False})
        from soomac_irc.model_runtime import load_model
        from soomac_irc.decision_model import make_call_decision
        from soomac_irc.recommendation_model import make_call_recommendation
        from soomac_irc.response_model import make_call_response
        model, processor = load_model(*(str(resolved_paths[agent]) for agent in
                                       ("decision", "response", "recommendation")))
        calls = {"decision": make_call_decision(model, processor, adapter_name="decision"),
                 "recommendation": make_call_recommendation(model, processor, adapter_name="recommendation"),
                 "response": make_call_response(model, processor, adapter_name="response")}
        predictions = [predict_one(case, calls) for case in cases]
        if (source_hashes() != source_identity or benchmark.sha(args.cases) != cases_identity
                or adapter_identity(resolved_paths) != identity):
            raise ValueError("evaluation source, cases or adapter changed during inference")
        benchmark.write_new(output / "PREDICTIONS.json", predictions)
        grades = [functional_eval.grade(case, item["prediction"])
                  for case, item in zip(cases, predictions)]
        benchmark.write_new(output / "AUTOMATED_RESULTS.json", {
            "status": "MODEL_PREDICTIONS_COLLECTED_SEMANTIC_REVIEW_REQUIRED",
            "cases_sha256": benchmark.sha(args.cases),
            "predictions_sha256": benchmark.sha(output / "PREDICTIONS.json"),
            "per_agent": {agent: {"count": 8,
                                  "automated_pass": sum(item["agent"] == agent and item["automated_pass"]
                                                        for item in grades)}
                          for agent in calls},
            "observations": grades, "production_quality_pass": False,
            "human_review_required": True})
        print(json.dumps({"output": str(output), "steps": len(predictions),
                          "automated_pass": sum(item["automated_pass"] for item in grades),
                          "semantic_review": "pending"}))
    finally:
        os.close(lock)


if __name__ == "__main__":
    main()
