"""Conditional 6+11-turn evaluation: frozen gold D fields, actual Q/P inference.

Reuses the existing scenario graph and offline physical fixture. No Decision
adapter is loaded, and no robot/remote/production service is changed.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys

import benchmark
from eval_correct_input_epochs import verify_freeze

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/soomac_irc"))
sys.path.insert(0, str(ROOT / "src/soomac_irc/test"))

SOURCE_IDS = {
    "initial": "DA31-CORE-separate-T01-D",
    "cream_select": "DA31-CORE-separate-T02-D",
    "vegetables": "DA31-CORE-separate-T05-D",
    "mushroom_change": "DA31-CORE-separate-T06-D",
    "allergy": "DA31-CORE-separate-T09-D",
    "sausage_select": "DA31-CORE-separate-T10-D",
    "cheese_select": "DA31-CORE-separate-T13-D",
}
COMPOUND_IDS = {
    "cream_execute": "DA31-CORE-compound-T02-D",
    "mushroom_execute": "DA31-CORE-compound-T05-D",
    "allergy_sausage_execute": "DA31-CORE-compound-T07-D",
    "cheese_execute": "DA31-CORE-compound-T09-D",
}
SPLIT_IDS = {
    "cream_execute": "DA31-CORE-separate-T03-D",
    "mushroom_execute": "DA31-CORE-separate-T07-D",
    "sausage_execute": "DA31-CORE-separate-T11-D",
    "cheese_execute": "DA31-CORE-separate-T14-D",
}


def oracle_rows():
    path = benchmark.RELEASE / "dataset/records/decision/train.jsonl"
    manifest = benchmark.read(benchmark.RELEASE / "dataset/MANIFEST.json")
    if benchmark.sha(path) != manifest["file_sha256"]["records/decision/train.jsonl"]:
        raise ValueError("Sealed D gold changed")
    wanted = set(SOURCE_IDS.values()) | set(COMPOUND_IDS.values()) | set(SPLIT_IDS.values())
    with path.open() as stream:
        rows = {row["id"]: row for row in map(json.loads, stream) if row["id"] in wanted}
    if set(rows) != wanted:
        raise ValueError("Missing exact Decision oracle rows")
    return rows


def oracle_factory(compound, rows):
    import manual_scenario_stack_probe as probe
    from soomac_irc.agent_contract import normalize_decision
    steps = probe.scenario(compound)
    state = {"position": 0}

    def factory(model, processor, logger=None, adapter_name=None, **unused):
        def call(session, user_text, robot_state, repair=None):
            if repair is not None or state["position"] >= len(steps):
                raise ValueError("Unexpected repair/additional Decision call in fixed oracle")
            section, tag, expected_text, execute = steps[state["position"]]
            if user_text != expected_text:
                raise ValueError("Scenario utterance drift")
            state["position"] += 1
            source_id = ({**SOURCE_IDS, **(COMPOUND_IDS if compound else SPLIT_IDS)})[tag]
            sparse = copy.deepcopy(rows[source_id]["target"])
            decision = normalize_decision(sparse)
            call.last_external_output = sparse
            call.trace_events.append({"stage": "decision", "gold_injected": True,
                "source_id": source_id, "user_text": user_text,
                "model_input": {"order": copy.deepcopy(session["order"]),
                                "pending": copy.deepcopy(session.get("pending")),
                                "preferences": copy.deepcopy(session["preferences"]),
                                "recent_history": copy.deepcopy(session["history"]),
                                "robot_state": copy.deepcopy(robot_state), "message": user_text},
                "external": copy.deepcopy(sparse), "parsed": copy.deepcopy(decision)})
            return decision
        call.trace_events = []
        call.last_external_output = None
        return call
    return factory


def run(args):
    verify_freeze(args.frozen)
    rows = oracle_rows()
    candidates = benchmark.read(args.frozen / "CANDIDATES.json")["rows"]
    chosen = []
    for agent in ("response", "recommendation"):
        path = getattr(args, agent).resolve()
        match = next(c for c in candidates if c["agent"] == agent and Path(c["path"]).resolve() == path)
        for filename, expected in (("adapter_model.safetensors", match["weights_sha256"]),
                                   ("adapter_config.json", match["config_sha256"])):
            if benchmark.sha(path / filename) != expected:
                raise ValueError("Adapter differs from committed candidate")
        chosen.append(match)
    output = benchmark.output_dir(args.output)
    source_paths = [Path(__file__), ROOT / "src/soomac_irc/test/manual_scenario_stack_probe.py",
                    *(ROOT / "src/soomac_irc/soomac_irc" / n for n in (
                        "llm_langgraph.py", "llm_policy.py", "agent_prompts.py", "response_contract.py",
                        "response_model.py", "recommendation_model.py", "model_runtime.py"))]
    pins = {str(p): benchmark.sha(p) for p in source_paths}
    benchmark.write_new(output / "PLAN.json", {"candidates": chosen, "source_sha256": pins,
        "oracle_source_sha256": benchmark.sha(benchmark.RELEASE / "dataset/records/decision/train.jsonl"),
        "decision_oracle": {key: value["target"] for key, value in rows.items()},
        "decision_model_called": False, "correct_D_assumption": True,
        "matmul_allow_tf32": False,
        "training": False, "actual_ros": False, "physical_simulation_only": True,
        "unseen_generalization": False, "automatic_adoption": False})
    lock = benchmark.acquire_gpu_lock()
    budget = None
    try:
        sys.path.insert(0, str(benchmark.RELEASE / "common"))
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        from budget import Budget
        import torch
        from transformers import AutoProcessor
        import manual_scenario_stack_probe as probe
        import soomac_irc.decision_model as decision_module
        from soomac_irc.model_runtime import load_model
        budget = Budget(benchmark.PROCESS_BUDGET, benchmark.RESERVE)
        torch.backends.cuda.matmul.allow_tf32 = False
        model, unused_processor = load_model(None, str(args.response), str(args.recommendation))
        del unused_processor
        processor = AutoProcessor.from_pretrained(benchmark.RELEASE / "assets/model-processor", local_files_only=True)
        model.requires_grad_(False)
        model.eval()
        original_factory = decision_module.make_call_decision
        results = []
        try:
            with (output / "turns.jsonl").open("x") as stream:
                for compound in (True, False):
                    decision_module.make_call_decision = oracle_factory(compound, rows)
                    results.append(probe.run_episode(compound, model, processor, stream, measure_all=True))
                    budget.check(force=True)
        finally:
            decision_module.make_call_decision = original_factory
        if any(benchmark.sha(Path(p)) != expected for p, expected in pins.items()):
            raise RuntimeError("Scenario code changed during inference")
        benchmark.write_new(output / "RESULT.json", {"episodes": results,
            "state_control_checks_passed": all(r["passed"] for r in results),
            "human_output_review": "PENDING", "decision_model_called": False,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "correct_D_assumption": True, "actual_ros": False, "physical_simulation_only": True,
            "automatic_adoption": False, "budget": budget.summary()})
    finally:
        if budget:
            budget.close()
        os.close(lock)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--recommendation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
