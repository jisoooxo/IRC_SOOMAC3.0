#!/usr/bin/env python3
"""Decision gold의 과잉 mutation과 high-risk review 완결성을 검사한다."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data_natural_multiturn_v3_rebuild"
SUPPORTED = {"양파", "버섯", "소시지", "게살", "치즈", "페퍼론치노"}
NON_ORDER_TARGETS = {
    "로봇", "갑각류", "유제품", "육류", "파스타", "비건",
    "야채", "채소", "추가 재료", "소세지", "개살", "계살", "패퍼런치노", "페퍼런치노",
}
ORDER_ACTION = re.compile(r"(넣|추가|담|빼|제외|없던\s*걸|취소|바꿔|적용|(?<!설명)(?<!알려)해\s*줘)")
QUESTION = re.compile(r"(뭐|무엇|어떻게|왜|알려|설명|특징|구조|가능해|할 수 있어|\?)")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def unsupported_is_explicit(message: str, item: str) -> bool:
    if item not in message:
        return False
    return bool(ORDER_ACTION.search(message.split(item, 1)[1][:24]))


def main() -> None:
    rows = read_jsonl(BASE / "intermediate/decision_v3_intermediate.jsonl")
    review = read_jsonl(BASE / "decision_v3/decision_high_risk_review.jsonl")
    suspicious = []
    question_order = []
    mixed_rows = []
    unsupported_explicit = 0

    for row in rows:
        message = row["input"].get("message", "")
        target = row["target"]
        if target.get("route") == "mixed":
            mixed_rows.append(row["id"])
        if QUESTION.search(message) and "order" in target:
            question_order.append(row["id"])
        toppings = target.get("order", {}).get("toppings", {})
        for item in toppings:
            if item in SUPPORTED:
                continue
            if item in NON_ORDER_TARGETS or not unsupported_is_explicit(message, item):
                suspicious.append({"id": row["id"], "item": item, "message": message})
            else:
                unsupported_explicit += 1

    review_status = Counter(row.get("semantic_review", "missing") for row in review)
    missing_review_fields = [
        row.get("id") for row in review
        if row.get("semantic_review") not in {"pass", "fixed", "discard"}
        or not isinstance(row.get("review_reason"), str)
        or not isinstance(row.get("error_class"), list)
    ]
    cleanup = Counter(
        error
        for row in rows
        for error in row.get("metadata", {}).get("cleanup_error_class", [])
    )
    status = "PASS" if not suspicious and len(review) == 200 and not missing_review_fields and not review_status["discard"] else "FAIL"
    report = {
        "status": status,
        "rows": len(rows),
        "mixed_route_inspected": len(mixed_rows),
        "question_or_description_with_order_inspected": len(question_order),
        "explicit_unsupported_mutations_preserved": unsupported_explicit,
        "known_fake_mutations_fixed": cleanup["fake_mutation"],
        "fake_mutations_detected": cleanup["fake_mutation"],
        "fake_mutations_discarded": review_status["discard"],
        "known_fake_mutation": len(suspicious),
        "semantic_suspicious": len(suspicious),
        "unresolved_suspicious_mutations": len(suspicious),
        "suspicious_examples": suspicious[:20],
        "high_risk_review": {
            "rows": len(review),
            "status_counts": dict(review_status),
            "missing_review_fields": missing_review_fields,
        },
    }
    path = BASE / "reports/decision_semantic_audit.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
