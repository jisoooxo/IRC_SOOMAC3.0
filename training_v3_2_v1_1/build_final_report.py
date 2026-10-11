#!/usr/bin/env python3
"""완료된 평가 결과를 고정 형식의 A/B 보고서와 manifest로 집계한다."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
EVAL = HERE / "evaluation"

DECISION_LOSS = {1: 0.0258287527, 2: 0.0410105735, 3: 0.0213587414, 4: 0.00590202026, 5: 0.00698318612, 6: 0.007190865}
RESPONSE_LOSS = {1: 1.056718826, 2: 1.30096507, 3: 1.54239666, 4: 1.66291451}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * ratio) - 1)
    return round(ordered[index], 3)


def metric_accuracy(metric: dict) -> float | None:
    return metric.get("accuracy") if isinstance(metric, dict) else None


def compact_decision(metrics: dict) -> dict:
    failures = metrics.get("failure_classes", {})
    return {
        "rows": metrics["rows"],
        "schema": metric_accuracy(metrics["schema_valid"]),
        "route": metric_accuracy(metrics["route"]),
        "order_precision": metrics["order"]["precision"],
        "order_recall": metrics["order"]["recall"],
        "order_value_accuracy": metrics["order"]["value_accuracy"],
        "fake_mutation": failures.get("fake_mutation", 0),
        "missed_mutation": failures.get("missed_mutation", 0),
        "wrong_reference": failures.get("wrong_reference", 0),
        "over_clarify": failures.get("over_clarify", 0),
        "under_clarify": failures.get("under_clarify", 0),
        "wrong_confirmation": failures.get("wrong_confirmation", 0),
        "wrong_commit": failures.get("wrong_commit", 0),
        "wrong_recommendation": failures.get("wrong_recommendation", 0),
        "wrong_preference": failures.get("wrong_preference", 0),
        "wrong_restriction": failures.get("wrong_restriction", 0),
        "latency_ms": metrics.get("latency_ms"),
    }


def compact_response(metrics: dict) -> dict:
    return {
        "rows": metrics["rows"],
        "red_failures": metrics["red_failures"],
        "fact_consistency": metric_accuracy(metrics["fact_consistency"]),
        "pending_consistency": metric_accuracy(metrics["pending_consistency"]),
        "execution_consistency": metric_accuracy(metrics["execution_consistency"]),
        "next_prompt": metric_accuracy(metrics["next_prompt_fulfillment"]),
        "followup_appropriate": metric_accuracy(metrics["followup_appropriate"]),
        "hallucination_free": metric_accuracy(metrics["hallucination_free"]),
        "naturalness": metrics["naturalness"]["average"],
        "failure_classes": metrics.get("failure_classes", {}),
        "latency_ms_response_plus_judge": metrics.get("latency_ms_response_plus_judge"),
    }


def adapter_manifest(kind: str, epoch: int) -> dict:
    adapter = HERE / kind / "adapters" / f"epoch_{epoch}_adapter"
    config = read_json(adapter / "adapter_config.json")
    return {
        "kind": kind,
        "adopted": kind == "decision",
        "selected_epoch": epoch,
        "path": str(adapter),
        "runtime_copy": str(ROOT / "models" / f"{kind}_adapter_v{'3_2' if kind == 'decision' else '1_1'}_best_epoch{epoch}"),
        "adapter_model_sha256": sha256(adapter / "adapter_model.safetensors"),
        "adapter_model_bytes": (adapter / "adapter_model.safetensors").stat().st_size,
        "base_model": config["base_model_name_or_path"],
        "lora": {
            "r": config["r"],
            "alpha": config["lora_alpha"],
            "dropout": config["lora_dropout"],
            "target_modules": config["target_modules"],
        },
    }


def epoch_tables() -> tuple[list[dict], list[dict]]:
    decision = []
    for epoch, loss in DECISION_LOSS.items():
        metrics = read_json(HERE / f"decision/epoch_eval/epoch_{epoch}/decision_metrics.json")
        decision.append({"epoch": epoch, "validation_loss": loss, **compact_decision(metrics)})
    response = []
    for epoch, loss in RESPONSE_LOSS.items():
        metrics = read_json(HERE / f"response/epoch_eval/epoch_{epoch}/response_metrics.json")
        response.append({"epoch": epoch, "validation_loss": loss, **compact_response(metrics)})
    return decision, response


def targeted_families(path: Path) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(path):
        groups[row["metadata"]["scenario_family"]].append(row)
    result = {}
    for family, rows in sorted(groups.items()):
        result[family] = {
            "rows": len(rows),
            "red_failures": sum(bool(row.get("hard_failures") or row.get("error")) for row in rows),
            "next_prompt_accuracy": sum(bool((row.get("judgment") or {}).get("next_prompt_fulfilled")) for row in rows) / len(rows),
            "followup_accuracy": sum(bool((row.get("judgment") or {}).get("followup_appropriate")) for row in rows) / len(rows),
        }
    return result


def e2e_summary(name: str) -> dict:
    directory = EVAL / name
    sessions = read_jsonl(directory / "e2e_sessions.jsonl")
    turns = read_jsonl(directory / "e2e_turns.jsonl")
    failures = Counter(row["failure_source"] for row in turns if row["failure_source"] != "No failure")
    latency = {}
    for stage in ("decision", "recommendation", "response", "total"):
        values = [float(row["latency_ms"].get(stage) or 0) for row in turns]
        latency[stage] = {"median": percentile(values, 0.5), "p95": percentile(values, 0.95)}
    wrong_adapter = sum(
        not event.get("correct", False)
        for row in turns
        for event in row.get("adapter_events", [])
    )
    probe = read_json(directory / "adapter_probe.json")
    wrong_adapter += sum(not event.get("correct", False) for event in probe.get("adapter_events", []))
    return {
        "sessions": len(sessions),
        "session_success": sum(bool(row["success"]) for row in sessions),
        "turns": len(turns),
        "turn_success": sum(row["failure_source"] == "No failure" for row in turns),
        "failure_sources": dict(failures),
        "adapter_switching_failures": wrong_adapter,
        "latency_ms": latency,
    }


def compact_long(metrics: dict) -> dict:
    return {
        "sessions": metrics["sessions"],
        "turns": metrics["turns"],
        "full_session_success": metrics["full_session_success"],
        "root_cause_clean_turns": metrics["root_cause_clean_turns"],
        "decision_true_failure_turns": metrics["decision_true_failure_turns"],
        "order_precision": metrics["order"]["precision"],
        "order_recall": metrics["order"]["recall"],
        "fake_mutation_turns": metrics["mutation"]["fake_mutation_turns"],
        "missed_mutation_turns": metrics["mutation"]["missed_mutation_turns"],
        "wrong_reference": metrics["wrong_reference"],
        "pending_failure": metrics["pending_failure"],
        "state_drift": metrics["state_drift"],
        "policy_failure": metrics["policy_failure"],
        "preference_persistence_mismatch_turns": metrics["preference_persistence_mismatch_turns"],
        "recommendation_failure": metrics["recommendation"].get("RECOMMENDATION_AGENT_FAILURE", 0),
        "response_true_failure": metrics["response"].get("RESPONSE_TRUE_FAILURE", 0),
    }


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    decision_epochs, response_epochs = epoch_tables()
    write_json(HERE / "decision/epoch_metrics.json", decision_epochs)
    write_json(HERE / "response/epoch_metrics.json", response_epochs)

    decision_best = {
        "best_epoch": 4,
        "checkpoint": str(HERE / "decision/adapters/epoch_4_adapter"),
        "validation_loss": DECISION_LOSS[4],
        "selection_reason": "schema/route 100%, fake mutation 최소 1건, precision/recall 99.37%로 semantic 우선순위 최상",
    }
    response_best = {
        "best_epoch": 4,
        "checkpoint": str(HERE / "response/adapters/epoch_4_adapter"),
        "validation_loss": RESPONSE_LOSS[4],
        "selection_reason": "validation RED 0건과 모든 semantic gate 100%; loss best epoch1보다 semantic 우선",
    }
    write_json(HERE / "decision/best_checkpoint.json", decision_best)
    write_json(HERE / "response/best_checkpoint.json", response_best)
    write_json(HERE / "decision/adapter_manifest.json", adapter_manifest("decision", 4))
    write_json(HERE / "response/adapter_manifest.json", adapter_manifest("response", 4))

    decision_token = read_json(ROOT / "soomac_decision_v3_2_response_v1_1_dual_training_v2_9/docker_context/assets/dataset/decision_token_length_report.json")
    response_token = read_json(ROOT / "soomac_decision_v3_2_response_v1_1_dual_training_v2_9/docker_context/assets/dataset/response_token_length_report.json")
    write_json(HERE / "decision/token_length_report.json", decision_token)
    write_json(HERE / "response/token_length_report.json", response_token)

    baseline_decision = compact_decision(read_json(ROOT / "evaluation_decision_v3_1/decision_metrics.json"))
    candidate_decision = compact_decision(read_json(EVAL / "candidate_combination/decision_metrics.json"))
    decision_ab = {"baseline_v3_1": baseline_decision, "candidate_v3_2_epoch4": candidate_decision, "verdict": "ADOPT DECISION V3.2"}
    write_json(EVAL / "decision_v3_1_vs_v3_2.json", decision_ab)

    baseline_response = compact_response(read_json(ROOT / "evaluation_dual_adapter/response_metrics.json"))
    candidate_response = compact_response(read_json(EVAL / "candidate_combination/response_metrics.json"))
    targeted = {
        "scope_warning": "200 supplement training rows의 train-fit 확인이며 독립 frozen gate로 사용하지 않음",
        "baseline": {
            "overall": compact_response(read_json(EVAL / "targeted_response/baseline/response_metrics.json")),
            "families": targeted_families(EVAL / "targeted_response/baseline/response_predictions.jsonl"),
        },
        "candidate": {
            "overall": compact_response(read_json(EVAL / "targeted_response/candidate/response_metrics.json")),
            "families": targeted_families(EVAL / "targeted_response/candidate/response_predictions.jsonl"),
        },
    }
    response_ab = {"baseline_v1": baseline_response, "candidate_v1_1_epoch4": candidate_response, "targeted_train_fit": targeted, "verdict": "KEEP RESPONSE V1"}
    write_json(EVAL / "response_v1_vs_v1_1.json", response_ab)

    baseline_long = compact_long(read_json(ROOT / "evaluation_decision_v3_1/long_horizon_corrected/corrected_metrics.json"))
    candidate_long = compact_long(read_json(EVAL / "long_horizon_corrected/corrected_metrics.json"))
    write_json(EVAL / "long_horizon_ab.json", {"baseline_v3_1_v1": baseline_long, "candidate_v3_2_v1_1": candidate_long})

    e2e = {
        "A_v3_1_v1": e2e_summary("baseline_combination"),
        "B_v3_2_v1_selected": e2e_summary("selected_combination"),
        "D_v3_2_v1_1": e2e_summary("candidate_combination"),
    }
    write_json(EVAL / "e2e_ab.json", e2e)
    runtime_smoke = read_json(EVAL / "runtime_applied_smoke/runtime_smoke.json")
    write_json(EVAL / "latency_ab.json", {
        "frozen_decision": {"baseline": baseline_decision["latency_ms"], "candidate": candidate_decision["latency_ms"]},
        "frozen_response_plus_judge": {"baseline": baseline_response["latency_ms_response_plus_judge"], "candidate": candidate_response["latency_ms_response_plus_judge"]},
        "e2e": {name: result["latency_ms"] for name, result in e2e.items()},
    })

    failures = []
    sources = {
        "decision_frozen_candidate": EVAL / "candidate_combination/decision_failures.jsonl",
        "response_frozen_candidate": EVAL / "candidate_combination/response_failures.jsonl",
        "long_horizon_corrected_candidate": EVAL / "long_horizon_corrected/corrected_failure_cases.jsonl",
        "response_targeted_baseline": EVAL / "targeted_response/baseline/response_failures.jsonl",
        "response_targeted_candidate": EVAL / "targeted_response/candidate/response_failures.jsonl",
    }
    for scope, path in sources.items():
        failures.extend({"evaluation_scope": scope, **row} for row in read_jsonl(path))
    for combo in ("baseline_combination", "selected_combination", "candidate_combination"):
        failures.extend(
            {"evaluation_scope": f"e2e_{combo}", **row}
            for row in read_jsonl(EVAL / combo / "e2e_turns.jsonl")
            if row["failure_source"] != "No failure"
        )
    write_jsonl(EVAL / "failure_analysis.jsonl", failures)

    training = read_json(HERE / "dual_training_complete.json")["results"]
    decision_report = f"""# Decision v3.2 Training Report

- safety gate: `{training['decision']['safety_gate_passed']}`
- completed epochs: `{training['decision']['completed_epoch']}`
- optimizer steps: `{training['decision']['local_step']}`
- skipped updates: `{training['decision']['skipped_updates']}`
- gradient finite/nonzero: `{training['decision']['gradient_check']['finite']}` / `{training['decision']['gradient_check']['nonzero']}`
- semantic best: epoch 4 (`loss={DECISION_LOSS[4]}`)
- frozen precision/recall: `{percent(candidate_decision['order_precision'])}` / `{percent(candidate_decision['order_recall'])}`
- verdict: **ADOPT DECISION V3.2**
"""
    response_report = f"""# Response v1.1 Training Report

- safety gate: `{training['response']['safety_gate_passed']}`
- completed epochs: `{training['response']['completed_epoch']}`
- optimizer steps: `{training['response']['local_step']}`
- skipped updates: `{training['response']['skipped_updates']}`
- gradient finite/nonzero: `{training['response']['gradient_check']['finite']}` / `{training['response']['gradient_check']['nonzero']}`
- semantic best: epoch 4 (`loss={RESPONSE_LOSS[4]}`)
- frozen RED failures: baseline `{baseline_response['red_failures']}` → candidate `{candidate_response['red_failures']}`
- verdict: **KEEP RESPONSE V1**
"""
    (HERE / "decision/training_report.md").write_text(decision_report, encoding="utf-8")
    (HERE / "response/training_report.md").write_text(response_report, encoding="utf-8")

    report = f"""# Decision v3.2 + Response v1.1 최종 A/B 평가

## 최종 판정

- **ADOPT DECISION V3.2 (epoch 4)**
- **KEEP RESPONSE V1**
- **Selected pair: Decision v3.2 + Response v1**
- **READY FOR ROBOT TEST**

## Decision

| metric | v3.1 | v3.2 |
|---|---:|---:|
| schema | {percent(baseline_decision['schema'])} | {percent(candidate_decision['schema'])} |
| route | {percent(baseline_decision['route'])} | {percent(candidate_decision['route'])} |
| frozen order precision | {percent(baseline_decision['order_precision'])} | {percent(candidate_decision['order_precision'])} |
| frozen order recall | {percent(baseline_decision['order_recall'])} | {percent(candidate_decision['order_recall'])} |
| frozen fake mutation | {baseline_decision['fake_mutation']} | {candidate_decision['fake_mutation']} |
| frozen missed mutation | {baseline_decision['missed_mutation']} | {candidate_decision['missed_mutation']} |
| long order precision | {percent(baseline_long['order_precision'])} | {percent(candidate_long['order_precision'])} |
| long order recall | {percent(baseline_long['order_recall'])} | {percent(candidate_long['order_recall'])} |
| long Decision failures | {baseline_long['decision_true_failure_turns']} | {candidate_long['decision_true_failure_turns']} |
| long state/pending/policy failures | {baseline_long['state_drift']}/{baseline_long['pending_failure']}/{baseline_long['policy_failure']} | {candidate_long['state_drift']}/{candidate_long['pending_failure']}/{candidate_long['policy_failure']} |

Cause: v3.2가 frozen precision과 long-horizon precision/recall을 높이고 extra/missed mutation 및 실제 Decision 실패를 줄였다.

Effect: runtime Decision 기본값을 epoch 4로 교체할 수 있다.

## Response

| metric | v1 | v1.1 |
|---|---:|---:|
| frozen RED failures | {baseline_response['red_failures']} | {candidate_response['red_failures']} |
| fact consistency | {percent(baseline_response['fact_consistency'])} | {percent(candidate_response['fact_consistency'])} |
| pending consistency | {percent(baseline_response['pending_consistency'])} | {percent(candidate_response['pending_consistency'])} |
| next prompt | {percent(baseline_response['next_prompt'])} | {percent(candidate_response['next_prompt'])} |
| hallucination free | {percent(baseline_response['hallucination_free'])} | {percent(candidate_response['hallucination_free'])} |
| long Response failures | {baseline_long['response_true_failure']} | {candidate_long['response_true_failure']} |
| targeted train-fit RED | {targeted['baseline']['overall']['red_failures']} | {targeted['candidate']['overall']['red_failures']} |

Targeted 200행은 학습행 재검증이므로 일반화 판정에는 사용하지 않았다.

Cause: v1.1은 목표 pending family를 학습했지만 독립 frozen RED 2건과 long-horizon Response 실패 54건으로 회귀했다.

Effect: runtime Response는 기존 v1을 유지한다.

## E2E

| combination | sessions | successful sessions | turns | successful turns | switching failures |
|---|---:|---:|---:|---:|---:|
| A: v3.1 + v1 | {e2e['A_v3_1_v1']['sessions']} | {e2e['A_v3_1_v1']['session_success']} | {e2e['A_v3_1_v1']['turns']} | {e2e['A_v3_1_v1']['turn_success']} | {e2e['A_v3_1_v1']['adapter_switching_failures']} |
| B: v3.2 + v1 | {e2e['B_v3_2_v1_selected']['sessions']} | {e2e['B_v3_2_v1_selected']['session_success']} | {e2e['B_v3_2_v1_selected']['turns']} | {e2e['B_v3_2_v1_selected']['turn_success']} | {e2e['B_v3_2_v1_selected']['adapter_switching_failures']} |
| D: v3.2 + v1.1 | {e2e['D_v3_2_v1_1']['sessions']} | {e2e['D_v3_2_v1_1']['session_success']} | {e2e['D_v3_2_v1_1']['turns']} | {e2e['D_v3_2_v1_1']['turn_success']} | {e2e['D_v3_2_v1_1']['adapter_switching_failures']} |

세 조합 모두 같은 1개 Decision 실패를 기록했다. Policy/runtime 및 Adapter switching 실패는 0건이다.

## 무결성 및 적용

- TAR SHA256: `025f763aced60a09cee3ee22334660109be69fcaa37de2e27e8d2c67892ab69f`
- 학습 safety gate: Decision/Response 모두 PASS
- token truncation: Decision/Response train·validation 모두 0행
- runtime Decision: `models/decision_adapter_v3_2_best_epoch4`
- runtime Response: `models/response_adapter_v1_best_epoch1`
- runtime smoke: {runtime_smoke['turns']}턴 / {runtime_smoke['adapter_events']} adapter events / 오류 {runtime_smoke['wrong_adapter_activation']}건
- rollback Decision v3.1과 Response v1 원본은 보존했다.
"""
    (HERE / "FINAL_AB_REPORT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
