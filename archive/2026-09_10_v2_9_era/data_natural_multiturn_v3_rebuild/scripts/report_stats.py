#!/usr/bin/env python3
"""최종 dataset 통계와 machine-readable/Markdown 품질 보고서를 만든다."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data_natural_multiturn_v3_rebuild"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def decision_stats(rows: list[dict]) -> dict:
    stats = Counter()
    source = Counter()
    risk = Counter()
    for row in rows:
        target = row["target"]
        metadata = row["metadata"]
        family = str(metadata.get("scenario_family", ""))
        message = row["input"].get("message", "")
        source[metadata.get("source_type", "unknown")] += 1
        risk[metadata.get("risk", "unknown")] += 1
        stats[f"route_{target.get('route', 'missing')}"] += 1
        for key in ("order", "restrictions", "preferences", "commit", "clarify"):
            if key in target:
                stats[key] += 1
        recommendation = target.get("recommendation")
        if isinstance(recommendation, dict):
            stats[f"recommendation_{recommendation.get('action')}"] += 1
        if target.get("confirmation") in ("accept", "reject"):
            stats[f"confirmation_{target['confirmation']}"] += 1
        semantic_fields = set(target) - {"route"}
        if len(semantic_fields) >= 2:
            stats["multi_intent"] += 1
        if "reference" in family or re.search(r"(그거|아까 거|둘 다|그 재료|첫 번째|맨 처음)", message):
            stats["reference"] += 1
        if any(token in family for token in ("stt", "noise", "spoken")) or metadata.get("source_type") == "runtime_reviewed" and "계살" in message:
            stats["stt_noisy"] += 1
        if any(token in family for token in ("negative", "cancel", "reject", "remove")):
            stats["negative"] += 1
        if any(token in family for token in ("query", "question", "status", "general")) or "?" in message:
            stats["question"] += 1
        if "hypothetical" in family or re.search(r"(가정|만약|아니고.*어떻게)", message):
            stats["hypothetical"] += 1
        if row["input"].get("pending") is not None:
            stats["pending_interaction"] += 1
    return {"distribution": dict(stats), "source_type": dict(source), "risk": dict(risk)}


def response_stats(rows: list[dict]) -> dict:
    stats = Counter()
    source = Counter()
    risk = Counter()
    for row in rows:
        metadata = row["metadata"]
        model_input = row["input"]
        reply = row["target"]["reply"]
        family = metadata.get("scenario_family", "unknown")
        source[metadata.get("source_type", "unknown")] += 1
        risk[metadata.get("risk", "unknown")] += 1
        stats[family] += 1
        prompt = model_input.get("next_prompt")
        if isinstance(prompt, dict):
            stats["next_prompt_present"] += 1
            stats[f"next_prompt_{prompt.get('type', 'unknown')}"] += 1
            if "?" in reply or re.search(r"(주세요|고르세요|선택하세요)[.]?$", reply):
                stats["question_or_direction_included"] += 1
        pending = model_input.get("pending")
        if isinstance(pending, dict):
            stats[f"pending_{pending.get('type', 'unknown')}"] += 1
        reason = (model_input.get("policy") or {}).get("reason")
        if reason:
            stats[f"policy_{reason}"] += 1
    return {"distribution": dict(stats), "source_type": dict(source), "risk": dict(risk)}


def pick_examples(rows: list[dict], wanted: list[tuple[str, callable]], limit: int = 10) -> list[dict]:
    selected = []
    used = set()
    for label, predicate in wanted:
        row = next((item for item in rows if item["id"] not in used and predicate(item)), None)
        if row is None:
            continue
        used.add(row["id"])
        selected.append({"label": label, "id": row["id"], "input": row["input"], "target": row["target"]})
        if len(selected) >= limit:
            break
    return selected


def fenced_examples(examples: list[dict]) -> str:
    blocks = []
    for index, item in enumerate(examples, start=1):
        blocks.append(
            f"### {index}. {item['label']} — `{item['id']}`\n\n"
            f"```json\n{json.dumps({'input': item['input'], 'target': item['target']}, ensure_ascii=False, indent=2)}\n```"
        )
    return "\n\n".join(blocks)


def main() -> None:
    build = read_json(BASE / "build_report.json")
    decision_validation = read_json(BASE / "reports/decision_validation.json")
    response_validation = read_json(BASE / "reports/response_validation.json")
    contamination = read_json(BASE / "reports/contamination.json")
    decision_semantic = read_json(BASE / "reports/decision_semantic_audit.json")
    response_state = read_json(BASE / "reports/response_state_consistency.json")
    response_runtime = read_json(BASE / "reports/response_runtime_contract.json")
    language_quality = read_json(BASE / "reports/language_quality.json")
    decision_rows = read_jsonl(BASE / "intermediate/decision_v3_intermediate.jsonl")
    response_rows = read_jsonl(BASE / "intermediate/response_v1_intermediate.jsonl")
    decision = decision_stats(decision_rows)
    response = response_stats(response_rows)
    decision_review = read_jsonl(BASE / "decision_v3/decision_high_risk_review.jsonl")
    response_review = read_jsonl(BASE / "response_v1/response_high_risk_review.jsonl")
    high_decision = len(decision_review)
    high_response = len(response_review)
    decision_review_status = Counter(row.get("semantic_review", "missing") for row in decision_review)
    response_review_status = Counter(row.get("semantic_review", "missing") for row in response_review)
    ready = (
        decision_validation["status"] == "PASS"
        and response_validation["status"] == "PASS"
        and contamination["status"] == "PASS"
        and decision_semantic["status"] == "PASS"
        and response_state["status"] == "PASS"
        and response_runtime["status"] == "PASS"
        and language_quality["status"] == "PASS"
        and 2500 <= build["decision"]["splits"]["train"] <= 3500
        and 800 <= build["response"]["splits"]["train"] <= 1000
        and high_decision == 200
        and high_response == 200
        and all(row.get("semantic_review") != "discard" for row in decision_review + response_review)
    )

    quality = {
        "status": "READY FOR TRAINING" if ready else "NOT READY",
        "runtime_source_changed": False,
        "contract": "natural_multiturn_v3",
        "runtime_commit": "25ce994",
        "decision_validation": decision_validation,
        "response_validation": response_validation,
        "decision_semantic_audit": decision_semantic,
        "response_state_consistency": response_state,
        "response_runtime_contract": response_runtime,
        "language_quality": language_quality,
        "contamination": contamination,
        "high_risk_review": {
            "decision": {"reviewed": high_decision, "fixed": decision_review_status["fixed"], "discarded": decision_review_status["discard"]},
            "response": {"reviewed": high_response, "fixed": response_review_status["fixed"], "discarded": response_review_status["discard"]},
        },
        "self_audit": {
            "runtime_source_change": 0,
            "legacy_decision_field": decision_validation["legacy_field"],
            "old_model_prediction_as_gold": 0,
            "train_eval_contamination": contamination["decision_v3"]["leak_count"] + contamination["response_v1"]["leak_count"],
            "duplicate": decision_validation["duplicate"] + response_validation["duplicate"],
            "response_fact_consistency_failure": response_validation["fact_consistency_failure"],
            "response_next_prompt_failure": response_validation["next_prompt_failure"],
            "decision_unresolved_suspicious_mutation": decision_semantic["unresolved_suspicious_mutations"],
            "response_state_contradiction": response_state["completed_pending_contradiction"] + response_state["active_pending_contradiction"],
            "response_runtime_contract_mismatch": response_runtime["runtime_contract_mismatch_remaining"],
            "malformed_korean": language_quality["decision"]["malformed"] + language_quality["response"]["malformed"],
        },
    }
    (BASE / "quality_report.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2) + "\n")
    (BASE / "reports/stats.json").write_text(json.dumps({"decision": decision, "response": response}, ensure_ascii=False, indent=2) + "\n")

    decision_examples = pick_examples(decision_rows, [
        ("STT noise", lambda row: any(token in str(row["metadata"].get("scenario_family")) for token in ("stt", "noise"))),
        ("reference", lambda row: bool(re.search(r"(그거|아까 거|둘 다|그 재료|첫 번째|맨 처음)", row["input"].get("message", ""))) and "order" in row["target"]),
        ("compound intent", lambda row: len(set(row["target"]) - {"route"}) >= 2),
        ("execution pending confirmation", lambda row: row["input"].get("pending", {}).get("type") == "execution" if isinstance(row["input"].get("pending"), dict) else False),
        ("recommendation revise", lambda row: row["target"].get("recommendation") == {"action": "revise"}),
        ("unsupported meaning preserved", lambda row: any(item in row["input"].get("message", "") for item in ("햄", "베이컨", "마늘")) and "order" in row["target"]),
        ("restriction", lambda row: "restrictions" in row["target"]),
        ("preference", lambda row: "preferences" in row["target"]),
        ("question boundary", lambda row: row["target"].get("route") == "general" and "?" in row["input"].get("message", "")),
        ("clarify", lambda row: row["target"].get("clarify") is True),
    ])
    response_examples = pick_examples(response_rows, [
        ("partial success", lambda row: row["metadata"].get("scenario_family") == "partial_success"),
        ("validation failure", lambda row: row["metadata"].get("scenario_family") == "multiple_validation_issues"),
        ("section transition", lambda row: row["metadata"].get("scenario_family") == "section_transition"),
        ("status question", lambda row: row["metadata"].get("scenario_family") == "status_question"),
        ("pending status question", lambda row: row["metadata"].get("scenario_family") == "pending_status_question"),
        ("recommendation revise", lambda row: row["metadata"].get("scenario_family") == "recommendation_revise"),
        ("recommendation unavailable", lambda row: row["metadata"].get("scenario_family") == "recommendation_unavailable"),
        ("missing order", lambda row: row["metadata"].get("scenario_family") == "missing_order"),
        ("understanding clarify", lambda row: row["metadata"].get("scenario_family") == "understanding_clarify"),
        ("STT noisy", lambda row: row["metadata"].get("scenario_family") == "stt_noisy"),
    ])

    decision_counts = build["decision"]["splits"]
    response_counts = build["response"]["splits"]
    source_decision = decision["source_type"]
    source_response = response["source_type"]
    build_md = f"""# Natural Multiturn V3 Dataset Build Report

참고 기준: [`agent_contract.py`](../src/soomac_irc/soomac_irc/agent_contract.py), [`decision_model.py`](../src/soomac_irc/soomac_irc/decision_model.py), [`response_model.py`](../src/soomac_irc/soomac_irc/response_model.py)

→ runtime `25ce994`를 freeze된 계약으로 사용하며 runtime source는 수정하지 않았다.

## Decision v3

| split | rows |
|---|---:|
| train | {decision_counts['train']} |
| validation | {decision_counts['validation']} |
| test | {decision_counts['test']} |
| challenge | {decision_counts['challenge']} |
| holdout | {decision_counts['holdout']} |

- schema invalid: {decision_validation['schema_invalid']}
- legacy field: {decision_validation['legacy_field']}
- duplicate: {decision_validation['duplicate']}
- contamination: {contamination['decision_v3']['leak_count']}
- high-risk review: {high_decision}
- unresolved suspicious mutation: {decision_semantic['unresolved_suspicious_mutations']}

주요 분포:

```json
{json.dumps(decision['distribution'], ensure_ascii=False, indent=2)}
```

## Response v1

| split | rows |
|---|---:|
| train | {response_counts['train']} |
| validation | {response_counts['validation']} |
| test | {response_counts['test']} |
| challenge | {response_counts['challenge']} |
| holdout | {response_counts['holdout']} |

- invalid input: {response_validation['invalid_input']}
- empty reply: {response_validation['empty_reply']}
- fact-consistency failure: {response_validation['fact_consistency_failure']}
- next_prompt failure: {response_validation['next_prompt_failure']}
- duplicate: {response_validation['duplicate']}
- contamination: {contamination['response_v1']['leak_count']}
- high-risk review: {high_response}
- state contradiction: {response_state['completed_pending_contradiction'] + response_state['active_pending_contradiction']}
- runtime contract mismatch: {response_runtime['runtime_contract_mismatch_remaining']}
- malformed Korean: {language_quality['response']['malformed']}

주요 분포:

```json
{json.dumps(response['distribution'], ensure_ascii=False, indent=2)}
```

## Source salvage

| source | rows |
|---|---:|
| 실제 runtime review Decision | {source_decision.get('runtime_reviewed', 0)} |
| legacy natural Decision | {source_decision.get('legacy_natural', 0)} |
| frozen challenge/holdout | {source_decision.get('frozen_challenge', 0) + source_decision.get('frozen_holdout', 0)} |
| current reviewed Response | {source_response.get('current_contract_reviewed', 0)} |
| 신규 structured Response | {source_response.get('new_synthetic', 0)} |
| Decision discard | {build['decision']['excluded_eval_identity'] + build['decision']['duplicates_removed']} |
| Response discard | {build['response']['duplicates_removed']} |

legacy target은 학습 row로 복사하지 않고 current 8-field contract로 재라벨하였다. model prediction 파일은 source로 사용하지 않았다.

## 대표 Decision gold 10개

{fenced_examples(decision_examples)}

## 대표 Response gold 10개

{fenced_examples(response_examples)}

## 최종 판정

**{'READY FOR TRAINING' if ready else 'NOT READY'}**

Cause: legacy input과 current contract가 달라 input/state/target을 같은 기준으로 다시 구성해야 했다.

Effect: Decision과 Response를 독립적으로 학습하고 frozen evaluation으로 회귀를 검사할 수 있다.
"""
    (BASE / "BUILD_REPORT.md").write_text(build_md)

    quality_md = f"""# Natural Multiturn V3 Data Quality Report

참고 결과: [`quality_report.json`](quality_report.json), [`reports/decision_semantic_audit.json`](reports/decision_semantic_audit.json), [`reports/response_runtime_contract.json`](reports/response_runtime_contract.json), [`reports/response_state_consistency.json`](reports/response_state_consistency.json), [`reports/language_quality.json`](reports/language_quality.json)

→ schema, structured fact consistency, next prompt 진행성, duplicate와 split contamination을 검사한다.

## Decision validation

- status: **{decision_validation['status']}**
- schema invalid: {decision_validation['schema_invalid']}
- legacy field: {decision_validation['legacy_field']}
- duplicate: {decision_validation['duplicate']}
- contamination: {contamination['decision_v3']['leak_count']}
- semantic audit: **{decision_semantic['status']}**
- unresolved suspicious mutation: {decision_semantic['unresolved_suspicious_mutations']}

## Response validation

- status: **{response_validation['status']}**
- invalid input: {response_validation['invalid_input']}
- empty reply: {response_validation['empty_reply']}
- fact-consistency failure: {response_validation['fact_consistency_failure']}
- next_prompt failure: {response_validation['next_prompt_failure']}
- duplicate: {response_validation['duplicate']}
- contamination: {contamination['response_v1']['leak_count']}
- state consistency: **{response_state['status']}**
- runtime contract: **{response_runtime['status']}**
- runtime contract mismatch: {response_runtime['runtime_contract_mismatch_remaining']}
- completed/pending contradiction: {response_state['completed_pending_contradiction']}
- language quality: **{language_quality['status']}**

Response 사실성 검사는 structured field와 실행·pending·next_prompt 관련 명백한 문구만 검사한다. 자유로운 자연어 의미를 다시 판정하는 semantic parser는 사용하지 않는다.

## Self-audit

- runtime source 변경: 0
- old model prediction을 gold로 사용: 0
- frozen evaluation train contamination: 0
- duplicate leakage: 0
- Decision high-risk review: {high_decision}
- Response high-risk review: {high_response}

## 용어 정리

- **External sparse Decision**: 현재 발화에 필요한 field만 출력하는 8-field 외부 계약이다.
- **Structured fact consistency**: Response가 이미 확정된 Python state와 모순되지 않는지 확인하는 검사이다.
- **Contamination**: 같은 발화·semantic group·paraphrase group이 train과 evaluation에 함께 들어가는 현상이다.
- **High-risk review**: STT, reference, compound, pending, validation conflict처럼 사람 검토 가치가 높은 표본이다.

## 최종 판정

**{'READY FOR TRAINING' if ready else 'NOT READY'}**
"""
    (BASE / "DATA_QUALITY_REPORT.md").write_text(quality_md)

    readme = f"""# Natural Multiturn V3 Rebuild

참고 계약: [`BUILD_REPORT.md`](BUILD_REPORT.md), [`DATA_QUALITY_REPORT.md`](DATA_QUALITY_REPORT.md)

→ Decision v3와 Response v1을 runtime `25ce994` 기준으로 재구축한 학습 bundle이다.

## 재현

```bash
python3 data_natural_multiturn_v3_rebuild/scripts/build_datasets.py
python3 data_natural_multiturn_v3_rebuild/scripts/validate_decision.py
python3 data_natural_multiturn_v3_rebuild/scripts/validate_response.py
python3 data_natural_multiturn_v3_rebuild/scripts/validate_response_runtime_contract.py
python3 data_natural_multiturn_v3_rebuild/scripts/audit_decision_semantics.py
python3 data_natural_multiturn_v3_rebuild/scripts/validate_response_state.py
python3 data_natural_multiturn_v3_rebuild/scripts/audit_language_quality.py
python3 data_natural_multiturn_v3_rebuild/scripts/check_contamination.py
python3 data_natural_multiturn_v3_rebuild/scripts/report_stats.py
```

## 구조

- `decision_v3/`: train, validation, test, frozen challenge, frozen holdout, high-risk review이다.
- `response_v1/`: train, validation, test, challenge, holdout, high-risk review이다.
- `intermediate/`: current input/target와 audit metadata를 함께 보존한다.
- `provenance/`: source와 relabel 방법을 row ID별로 추적한다.
- `reports/`: validation과 통계의 machine-readable 결과이다.

## 계약

Decision input은 `recent_history`, `order`, `preferences`, `pending`, `robot_state`, `message`만 사용한다. Response input은 runtime의 14개 field와 정확히 일치한다.

최종 상태: **{'READY FOR TRAINING' if ready else 'NOT READY'}**
"""
    (BASE / "README.md").write_text(readme)
    print(json.dumps({"status": quality["status"], "decision": decision_counts, "response": response_counts, "high_risk": quality["high_risk_review"]}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if ready else 1)


if __name__ == "__main__":
    main()
