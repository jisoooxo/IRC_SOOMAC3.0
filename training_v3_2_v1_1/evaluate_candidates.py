#!/usr/bin/env python3
"""Decision v3.2 / Response v1.1 epoch와 최종 조합을 기존 frozen 평가기로 검증한다."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DUAL_SOURCE = ROOT / "evaluation_dual_adapter"
LONG_SOURCE = ROOT / "evaluation_long_horizon"
CORRECTED_SOURCE = ROOT / "evaluation_long_horizon_corrected"
DATA = ROOT / "data_natural_multiturn_v3_rebuild"
BASELINE_DECISION = ROOT / "models/decision_adapter_v3_1_best_epoch3"
BASELINE_RESPONSE = ROOT / "models/response_adapter_v1_best_epoch1"

sys.path.insert(0, str(DUAL_SOURCE))
sys.path.insert(0, str(ROOT / "src/soomac_irc"))

import evaluate as dual  # noqa: E402
from soomac_irc.agent_contract import new_session_state, new_turn_state  # noqa: E402
from soomac_irc.llm_langgraph import build_graph  # noqa: E402


OUTPUT_NAMES = {
    "DECISION_PREDICTIONS": "decision_predictions.jsonl",
    "DECISION_FAILURES": "decision_failures.jsonl",
    "DECISION_METRICS": "decision_metrics.json",
    "RESPONSE_PREDICTIONS": "response_predictions.jsonl",
    "RESPONSE_REJUDGED": "response_rejudged.jsonl",
    "RESPONSE_FAILURES": "response_failures.jsonl",
    "RESPONSE_METRICS": "response_metrics.json",
    "E2E_SESSIONS": "e2e_sessions.jsonl",
    "E2E_TURNS": "e2e_turns.jsonl",
    "ADAPTER_PROBE": "adapter_probe.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def adapter_info(path: Path) -> dict:
    return {
        "path": str(path),
        "adapter_model_sha256": sha256(path / "adapter_model.safetensors"),
        "adapter_model_bytes": (path / "adapter_model.safetensors").stat().st_size,
        "adapter_config": json.loads((path / "adapter_config.json").read_text(encoding="utf-8")),
    }


def configure(out: Path, decision_path: Path, response_path: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    dual.OUT = out
    dual.DECISION_ADAPTER_PATH = str(decision_path)
    dual.RESPONSE_ADAPTER_PATH = str(response_path)
    for name, filename in OUTPUT_NAMES.items():
        setattr(dual, name, out / filename)
    baseline = {
        "recorded_at": dual.time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "base_model_path": dual.MODEL_PATH,
        "quantization": dual.MODEL_QUANTIZATION,
        "adapters": {
            "decision": adapter_info(decision_path),
            "response": adapter_info(response_path),
        },
        "generation": {
            "decision": {
                "max_new_tokens": dual.DECISION_MAX_TOKENS,
                "input_max_tokens": dual.DECISION_INPUT_MAX_TOKENS,
                "do_sample": False,
                "use_cache": True,
                "xgrammar_any_order": dual.DECISION_KEYS_ANY_ORDER,
            },
            "recommendation": {
                "max_new_tokens": dual.RECOMMENDATION_MAX_TOKENS,
                "do_sample": False,
                "use_cache": True,
                "adapter": "base",
            },
            "response": {
                "max_new_tokens": dual.RESPONSE_MAX_TOKENS,
                "do_sample": False,
                "use_cache": True,
            },
        },
    }
    write_json(out / "baseline.json", baseline)


def validation_rows(kind: str) -> list[dict]:
    metadata = dual.metadata_by_id(kind)
    rows = dual.read_jsonl(DATA / kind / "validation.jsonl")
    for row in rows:
        row["split"] = "validation"
        row["metadata"] = metadata.get(row["id"], {})
    return rows


def targeted_response_rows() -> list[dict]:
    source = ROOT / "data_targeted_upgrade/response_v1_1_candidate"
    provenance = {
        row["id"]: row
        for row in dual.read_jsonl(source / "supplement_provenance.jsonl")
    }
    rows = dual.read_jsonl(source / "supplement_train.jsonl")
    for row in rows:
        info = provenance[row["id"]]
        row["split"] = "targeted_train_fit"
        row["metadata"] = {
            "scenario_family": info["family"],
            "coverage_tags": [info["family"], info["subtype"]],
            "source_type": info["source_type"],
        }
    return rows


def candidate_decision(epoch: int) -> Path:
    return HERE / f"decision/adapters/epoch_{epoch}_adapter"


def candidate_response(epoch: int) -> Path:
    return HERE / f"response/adapters/epoch_{epoch}_adapter"


def run_epoch_decision(epoch: int) -> None:
    out = HERE / f"decision/epoch_eval/epoch_{epoch}"
    configure(out, candidate_decision(epoch), BASELINE_RESPONSE)
    model, processor, tracker, decision, _, _ = dual.load_runtime()
    del model, processor
    dual.run_decision(validation_rows("decision_v3"), decision, tracker)
    print(json.dumps(dual.aggregate_decision(), ensure_ascii=False, indent=2), flush=True)


def run_epoch_response(epoch: int) -> None:
    out = HERE / f"response/epoch_eval/epoch_{epoch}"
    configure(out, BASELINE_DECISION, candidate_response(epoch))
    model, processor, tracker, _, _, response = dual.load_runtime()
    judge = dual.make_judge(model, processor, tracker)
    dual.run_response(validation_rows("response_v1"), response, judge, tracker)
    print(json.dumps(dual.aggregate_response(), ensure_ascii=False, indent=2), flush=True)


def run_targeted_response(variant: str, response_epoch: int) -> None:
    out = HERE / f"evaluation/targeted_response/{variant}"
    response_path = BASELINE_RESPONSE if variant == "baseline" else candidate_response(response_epoch)
    configure(out, BASELINE_DECISION, response_path)
    model, processor, tracker, _, _, response = dual.load_runtime()
    judge = dual.make_judge(model, processor, tracker)
    dual.run_response(targeted_response_rows(), response, judge, tracker)
    print(json.dumps(dual.aggregate_response(), ensure_ascii=False, indent=2), flush=True)


def selected_out() -> Path:
    return HERE / "evaluation/candidate_combination"


def baseline_out() -> Path:
    return HERE / "evaluation/baseline_combination"


def mixed_out() -> Path:
    return HERE / "evaluation/selected_combination"


def configure_selected(decision_epoch: int, response_epoch: int) -> None:
    configure(selected_out(), candidate_decision(decision_epoch), candidate_response(response_epoch))


def configure_baseline() -> None:
    configure(baseline_out(), BASELINE_DECISION, BASELINE_RESPONSE)


def configure_mixed(decision_epoch: int) -> None:
    configure(mixed_out(), candidate_decision(decision_epoch), BASELINE_RESPONSE)


def run_selected_smoke(decision_epoch: int, response_epoch: int) -> None:
    configure_selected(decision_epoch, response_epoch)
    model, processor, tracker, decision, recommendation, response = dual.load_runtime()
    graph = build_graph(decision, recommendation, response)
    session = new_session_state()
    robot = dual.empty_robot("veggie")
    records = []
    for user_text in ("양파 많이 넣어줘", "현재 양파 선택 알려줘", "햄 많이 넣어줘", "아니 실행은 취소해"):
        event_start = len(tracker.events)
        result = graph.invoke(new_turn_state(session, user_text, robot))
        session = copy.deepcopy(result["session"])
        records.append({
            "user": user_text,
            "decision": copy.deepcopy(decision.last_external_output),
            "policy": copy.deepcopy(result.get("policy")),
            "reply": result.get("reply"),
            "pending": copy.deepcopy(session.get("pending")),
            "adapter_events": tracker.since(event_start),
        })
    dual.run_adapter_probe(model, processor, tracker, decision)
    events = [event for row in records for event in row["adapter_events"]]
    probe = json.loads(dual.ADAPTER_PROBE.read_text(encoding="utf-8"))
    events.extend(probe.get("adapter_events", []))
    report = {
        "decision_epoch": decision_epoch,
        "response_epoch": response_epoch,
        "registered_adapters": sorted(getattr(model, "peft_config", {}).keys()),
        "adapter_events": len(events),
        "wrong_adapter_activation": sum(not event.get("correct") for event in events),
        "records": records,
        "probe": probe,
    }
    write_json(selected_out() / "standalone_smoke.json", report)
    if report["wrong_adapter_activation"]:
        raise RuntimeError("candidate adapter switching smoke failed")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def run_runtime_smoke() -> None:
    from soomac_irc import model_runtime

    out = HERE / "evaluation/runtime_applied_smoke"
    configure(out, Path(model_runtime.DECISION_ADAPTER_PATH), Path(model_runtime.RESPONSE_ADAPTER_PATH))
    model, processor, tracker, decision, recommendation, response = dual.load_runtime()
    graph = build_graph(decision, recommendation, response)
    scenarios = (
        ("order_accept", ("양파 많이 넣어줘", "좋아 진행해")),
        ("order_reject", ("버섯 조금 넣어줘", "아니 취소해")),
        ("general", ("양파는 왜 익히면 달아져?",)),
        ("recommendation", ("이 섹션에서 하나 추천해줘", "다른 걸로 추천해줘", "좋아")),
    )
    records = []
    for scenario, turns in scenarios:
        session = new_session_state()
        robot = dual.empty_robot("veggie")
        for user_text in turns:
            event_start = len(tracker.events)
            result = graph.invoke(new_turn_state(session, user_text, robot))
            session = copy.deepcopy(result["session"])
            records.append({
                "scenario": scenario,
                "user": user_text,
                "decision": copy.deepcopy(decision.last_external_output),
                "policy": copy.deepcopy(result.get("policy")),
                "reply": result.get("reply"),
                "pending": copy.deepcopy(session.get("pending")),
                "adapter_events": tracker.since(event_start),
            })
    dual.run_adapter_probe(model, processor, tracker, decision)
    events = [event for row in records for event in row["adapter_events"]]
    probe = json.loads(dual.ADAPTER_PROBE.read_text(encoding="utf-8"))
    events.extend(probe.get("adapter_events", []))
    report = {
        "decision_adapter": model_runtime.DECISION_ADAPTER_PATH,
        "response_adapter": model_runtime.RESPONSE_ADAPTER_PATH,
        "turns": len(records),
        "adapter_events": len(events),
        "wrong_adapter_activation": sum(not event.get("correct") for event in events),
        "stages_seen": sorted({event["stage"] for event in events}),
        "records": records,
        "probe": probe,
    }
    write_json(out / "runtime_smoke.json", report)
    if report["wrong_adapter_activation"]:
        raise RuntimeError("runtime applied adapter switching smoke failed")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def run_selected_frozen(kind: str, decision_epoch: int, response_epoch: int) -> None:
    configure_selected(decision_epoch, response_epoch)
    model, processor, tracker, decision, _, response = dual.load_runtime()
    if kind == "decision":
        dual.run_decision(dual.load_split_rows("decision_v3"), decision, tracker)
        metrics = dual.aggregate_decision()
    else:
        judge = dual.make_judge(model, processor, tracker)
        dual.run_response(dual.load_split_rows("response_v1"), response, judge, tracker)
        metrics = dual.aggregate_response()
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


def run_selected_e2e(decision_epoch: int, response_epoch: int) -> None:
    configure_selected(decision_epoch, response_epoch)
    model, processor, tracker, decision, recommendation, response = dual.load_runtime()
    dual.run_e2e(decision, recommendation, response, tracker)
    dual.run_adapter_probe(model, processor, tracker, decision)


def run_baseline_e2e() -> None:
    configure_baseline()
    model, processor, tracker, decision, recommendation, response = dual.load_runtime()
    dual.run_e2e(decision, recommendation, response, tracker)
    dual.run_adapter_probe(model, processor, tracker, decision)


def run_mixed_e2e(decision_epoch: int) -> None:
    configure_mixed(decision_epoch)
    model, processor, tracker, decision, recommendation, response = dual.load_runtime()
    dual.run_e2e(decision, recommendation, response, tracker)
    dual.run_adapter_probe(model, processor, tracker, decision)


def configure_long() -> object:
    sys.path.insert(0, str(LONG_SOURCE))
    import evaluate_long_horizon as long_eval

    out = HERE / "evaluation/long_horizon"
    long_eval.OUT = out
    long_eval.DUAL_EVAL = selected_out()
    for name, filename in {
        "SESSIONS_PATH": "sessions.jsonl",
        "TURNS_PATH": "turns.jsonl",
        "FAILURES_PATH": "failure_cases.jsonl",
        "OVERMUTATION_PATH": "overmutation_analysis.json",
        "DRIFT_PATH": "state_drift_report.json",
        "METRICS_PATH": "baseline_metrics.json",
        "REPORT_PATH": "LONG_HORIZON_REPORT.md",
    }.items():
        setattr(long_eval, name, out / filename)
    return long_eval


def run_selected_long(decision_epoch: int, response_epoch: int) -> None:
    configure_selected(decision_epoch, response_epoch)
    configure_long().run()


def run_selected_corrected(decision_epoch: int, response_epoch: int) -> None:
    configure_selected(decision_epoch, response_epoch)
    long_eval = configure_long()
    sys.path.insert(0, str(CORRECTED_SOURCE))
    import correct_evaluation as corrected

    corrected.SOURCE = HERE / "evaluation/long_horizon"
    corrected.DUAL = selected_out()
    corrected.OUT = HERE / "evaluation/long_horizon_corrected"
    corrected.long_eval = long_eval
    for name, filename in {
        "CORRECTED_TURNS": "corrected_turns.jsonl",
        "CORRECTED_FAILURES": "corrected_failure_cases.jsonl",
        "CORRECTED_METRICS": "corrected_metrics.json",
        "CORRECTED_OVERMUTATION": "corrected_overmutation_analysis.json",
        "RESPONSE_RECLASSIFICATION": "response_failure_reclassification.json",
        "RECOMMENDATION_ANALYSIS": "recommendation_failure_analysis.json",
        "CONTRACT_CHANGES": "evaluator_contract_changes.md",
        "REPORT": "CORRECTED_LONG_HORIZON_REPORT.md",
    }.items():
        setattr(corrected, name, corrected.OUT / filename)

    original_classifier = corrected.classify_existing_70

    def classify_existing_baseline():
        current_source = corrected.SOURCE
        try:
            corrected.SOURCE = LONG_SOURCE
            return original_classifier()
        finally:
            corrected.SOURCE = current_source

    corrected.classify_existing_70 = classify_existing_baseline
    corrected.main()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=("epoch-decision", "epoch-response", "targeted-response", "smoke", "runtime-smoke", "frozen-decision", "frozen-response", "e2e", "baseline-e2e", "mixed-e2e", "long", "corrected"),
    )
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--decision-epoch", type=int, default=1)
    parser.add_argument("--response-epoch", type=int, default=1)
    parser.add_argument("--variant", choices=("baseline", "candidate"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase == "epoch-decision":
        if args.epoch not in range(1, 7):
            raise ValueError("Decision epoch must be 1..6")
        run_epoch_decision(args.epoch)
    elif args.phase == "epoch-response":
        if args.epoch not in range(1, 5):
            raise ValueError("Response epoch must be 1..4")
        run_epoch_response(args.epoch)
    elif args.phase == "targeted-response":
        if args.variant is None:
            raise ValueError("targeted-response requires --variant")
        run_targeted_response(args.variant, args.response_epoch)
    elif args.phase == "smoke":
        run_selected_smoke(args.decision_epoch, args.response_epoch)
    elif args.phase == "runtime-smoke":
        run_runtime_smoke()
    elif args.phase in {"frozen-decision", "frozen-response"}:
        run_selected_frozen(args.phase.removeprefix("frozen-"), args.decision_epoch, args.response_epoch)
    elif args.phase == "e2e":
        run_selected_e2e(args.decision_epoch, args.response_epoch)
    elif args.phase == "baseline-e2e":
        run_baseline_e2e()
    elif args.phase == "mixed-e2e":
        run_mixed_e2e(args.decision_epoch)
    elif args.phase == "long":
        run_selected_long(args.decision_epoch, args.response_epoch)
    else:
        run_selected_corrected(args.decision_epoch, args.response_epoch)


if __name__ == "__main__":
    main()
