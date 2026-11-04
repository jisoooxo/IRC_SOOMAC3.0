"""Freeze a reviewed evaluation copy; never edits the sealed training release.

The only target corrections are three independently inspected Q normal-default
conflicts. Structural/rule checks are not a blanket semantic certificate.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import benchmark

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/soomac_irc"))
from soomac_irc.response_contract import (build_recommendation_model_input,
                                         build_response_model_input)
from soomac_irc.evaluation_v43 import (grade_recommendation_prediction,
                                      grade_response_prediction)

Q_CORRECTIONS = {"REC1-S0909", "REC1-S0935", "REC1-S0961"}
FROZEN_HOLDOUT_SHA = "855a000c0fbb26546b83c5d2a22ffac3d2592f27161e9ba97fb4c2e125d8aa06"


def digest(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def rebuild(agent, row):
    source = row["input"]
    session = {"order": source.get("confirmed_order", source.get("order")),
               "history": source.get("recent_history") or [],
               "preferences": source["preferences"], "pending": source.get("pending"),
               "action_history": source.get("recent_action_history") or []}
    if agent == "response":
        args = (source["user_text"], session, source["policy"], source["applied_this_turn"],
                source["future_changes"], source["recommendation_result"],
                source["next_prompt"], source["robot_state"], row["route"])
        actual = build_response_model_input(*args)
    elif agent == "recommendation":
        args = (session, {"recommendation": {"action": source["recommendation_action"]}},
                source["robot_state"], source["current_user_text"])
        actual = build_recommendation_model_input(*args)
    else:
        raise ValueError("Only P/Q are permitted")
    if actual != source:
        raise ValueError(f"runtime input drift: {row['id']}")
    return args


def reviewed_row(agent, row):
    result = copy.deepcopy(row)
    reason = "KEEP_CURRENT: input/state/target contract check; no confirmed correction"
    if agent == "recommendation" and row["id"] in Q_CORRECTIONS:
        source = row["input"]
        assert source["pending"] is None and source["order"]["toppings"] == {}
        assert source["preferences"] == [{"value": "꾸덕하게"}]
        assert source["current_user_text"] == "현재 단계에 맞는 재료를 하나 추천해줘"
        assert row["target"]["proposal"]["toppings"] == {"페퍼론치노": "low"}
        # Full histories were inspected: unrelated small talk, no quantity oracle.
        assert not any("페퍼론치노" in message["content"] for message in source["recent_history"])
        result["target"]["proposal"]["toppings"]["페퍼론치노"] = "normal"
        reason = ("CORRECT_EVAL_COPY_ONLY: no item quantity in current request, active preference, "
                  "history, confirmed order or valid pending; sealed Q contract requires normal. "
                  "Taste/cheese dislike does not authorize a new low amount.")
    assert result["input"] == row["input"]
    return result, reason


def candidate_paths():
    run = Path("/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/"
               "runs/r5_unsloth_local_authorized_20261101")
    result = []
    for agent in ("response", "recommendation"):
        marker = benchmark.read(run / agent / ".lab-monitor/resume.json")
        stage = (run / agent / marker["checkpoint"]).parent
        for state_path in stage.glob("checkpoint-*/trainer_state.json"):
            state = benchmark.read(state_path)
            epoch = state.get("epoch")
            if isinstance(epoch, (int, float)) and abs(epoch - round(epoch)) < 1e-8:
                path = state_path.parent
                commit = benchmark.read(path / "CHECKPOINT_COMMIT.json")
                result.append({"agent": agent, "name": f"epoch{int(epoch)}", "epoch": int(epoch),
                               "step": state["global_step"], "path": str(path),
                               "weights_sha256": commit["files"]["adapter_model.safetensors"],
                               "config_sha256": commit["files"]["adapter_config.json"]})
        if agent == "response":
            path = stage / "checkpoint-150"
            commit = benchmark.read(path / "CHECKPOINT_COMMIT.json")
            result.append({"agent": agent, "name": "best150", "epoch": 150 / 104,
                           "step": 150, "path": str(path),
                           "weights_sha256": commit["files"]["adapter_model.safetensors"],
                           "config_sha256": commit["files"]["adapter_config.json"]})
    assert len(result) == 13
    return sorted(result, key=lambda item: (item["agent"], item["step"]))


def prepare(output, holdout):
    if benchmark.sha(holdout) != FROZEN_HOLDOUT_SHA:
        raise ValueError("Recovered holdout differs from the original frozen bytes")
    original, reviewed, audit, sources = [], [], [], {}
    validation = {}
    for agent in ("response", "recommendation"):
        relative = f"records/{agent}/validation.jsonl"
        source = benchmark.RELEASE / "dataset" / relative
        manifest = benchmark.read(benchmark.RELEASE / "dataset/MANIFEST.json")
        if benchmark.sha(source) != manifest["file_sha256"][relative]:
            raise ValueError("Sealed validation changed")
        sources[str(source)] = benchmark.sha(source)
        validation[agent] = [json.loads(line) for line in source.open()]
        for line, row in enumerate(validation[agent], 1):
            rebuild(agent, row)
            fixed, reason = reviewed_row(agent, row)
            grade = (grade_response_prediction(fixed, fixed["target"]["reply"])
                     if agent == "response" else grade_recommendation_prediction(fixed, fixed["target"]))
            if not grade.passed:
                raise ValueError(f"Gold rule checker flagged {row['id']}: {grade.violations}")
            original.append({"agent": agent, "record": row})
            reviewed.append({"agent": agent, "record": fixed})
            audit.append({"agent": agent, "id": row["id"], "source": str(source), "line": line,
                          "source_record_sha256": digest(row), "reviewed_record_sha256": digest(fixed),
                          "decision": "CORRECT" if fixed != row else "KEEP_CURRENT",
                          "reason": reason, "input_rebuild_exact": True, "rule_check_pass": True,
                          "semantic_scope": "rowwise contract review; not a universal language certificate"})
    cases = [case for case in benchmark.read(holdout)["rows"]
             if case["agent"] in ("response", "recommendation")]
    for case in cases:
        case["evaluation_origin"] = "frozen_authored_paraphrase_not_independent_family"
    for agent in ("response", "recommendation"):
        path = benchmark.RELEASE / f"dataset/records/{agent}/train.jsonl"
        sources[str(path)] = benchmark.sha(path)
        for line in path.open():
            row = json.loads(line)
            if row["id"].startswith("DA31-CORE") or (
                    agent == "response" and row["id"] in ("NS01-D01-T01-P", "NS01-D02-T01-P")):
                if agent == "recommendation":
                    row = copy.deepcopy(row)
                    row.setdefault("metadata", {})["required_proposal_fields"] = {"sauce": "크림"}
                cases.append({"agent": agent, "record": row, "evaluation_origin": "train_rehearsal"})
    for case in cases:
        rebuild(case["agent"], case["record"])
    assert {a: sum(c["agent"] == a for c in cases) for a in validation} == {"response": 20, "recommendation": 9}
    output = benchmark.output_dir(output)
    files = {"VALIDATION_ORIGINAL.json": original, "VALIDATION_REVIEWED.json": reviewed,
             "ROW_REVIEW.json": audit, "FUNCTIONAL_CASES.json": cases,
             "CANDIDATES.json": candidate_paths()}
    for name, value in files.items():
        benchmark.write_new(output / name, {"rows": value})
    receipt = {"status": "CPU_REVIEWED_INPUTS_FROZEN_NO_MODEL_RUN", "schema_version": 1,
               "decision_remote_status": "training, per user; not remotely queried",
               "release_sha256": benchmark.sha(benchmark.RELEASE / "RELEASE.json"),
               "source_manifest_sha256": benchmark.sha(benchmark.RELEASE / "dataset/MANIFEST.json"),
               "protected_source_sha256": sources,
               "holdout_sha256": benchmark.sha(holdout),
               "files": {name: benchmark.sha(output / name) for name in files},
               "reviewed_counts": {a: len(rows) for a, rows in validation.items()},
               "corrected_Q_ids": sorted(Q_CORRECTIONS), "P_confirmed_target_corrections": [],
               "limitations": ["Do not infer P loss cause from Q corrections",
                               "P152/Q30 are small, uneven evaluation distributions",
                               "Gold token loss also penalizes valid alternate wording",
                               "No Decision model inference; no ROS or real completion evidence"],
               "source_code_sha256": benchmark.sha(Path(__file__))}
    benchmark.write_new(output / "FREEZE.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.output, args.holdout)
    print(json.dumps({k: result[k] for k in ("status", "reviewed_counts", "corrected_Q_ids")}, ensure_ascii=False))
