#!/usr/bin/env python3
"""Response 입력 state의 lifecycle 모순과 review 완결성을 검사한다."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data_natural_multiturn_v3_rebuild"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def item_name(value: object) -> str | None:
    if not isinstance(value, dict):
        return None
    if isinstance(value.get("task"), dict):
        return item_name(value["task"])
    candidate = value.get("item", value.get("class"))
    return candidate if isinstance(candidate, str) else None


def main() -> None:
    rows = read_jsonl(BASE / "intermediate/response_v1_intermediate.jsonl")
    review = read_jsonl(BASE / "response_v1/response_high_risk_review.jsonl")
    completed_pending = []
    active_pending = []
    invalid_robot_state = []
    pending_without_prompt = []
    fact_mismatch = []

    for row in rows:
        model_input = row["input"]
        pending = model_input.get("pending")
        pending_items = set()
        if isinstance(pending, dict):
            pending_items.update(item for item in pending.get("targets", []) if isinstance(item, str))
            if not isinstance(model_input.get("next_prompt"), dict):
                pending_without_prompt.append(row["id"])
        state = model_input.get("robot_state") or {}
        completed_items = {
            item_name(item) for item in state.get("completed_tasks", []) if item_name(item)
        }
        completed_items.update(
            item_name(item) for item in model_input.get("recent_action_history", [])
            if isinstance(item, dict) and item.get("type") == "task_completed" and item_name(item)
        )
        reply = row["target"].get("reply", "")
        claims_completed = bool(re.search(r"(완료됐|완료했|마쳤|작업이 끝났)", reply))
        claims_active = bool(re.search(r"(담는 중|진행하고 있어|담고 있어|실행 중)", reply))
        if claims_completed and not completed_items:
            fact_mismatch.append(row["id"])
        if completed_items & pending_items:
            completed_pending.append(row["id"])
        active = item_name(state.get("active_task"))
        if claims_active and not active:
            fact_mismatch.append(row["id"])
        if active and active in pending_items:
            active_pending.append(row["id"])
        robot_started = state.get("robot_started") is True
        if active and not robot_started:
            invalid_robot_state.append(row["id"])
        if robot_started and not active:
            prompt = model_input.get("next_prompt") or {}
            lifecycle_evidence = bool(
                state.get("task_queue")
                or state.get("completed_tasks")
                or model_input.get("recent_action_history")
                or prompt.get("type") == "section_selection"
            )
            if not lifecycle_evidence:
                invalid_robot_state.append(row["id"])

    review_status = Counter(row.get("semantic_review", "missing") for row in review)
    missing_review_fields = [
        row.get("id") for row in review
        if row.get("semantic_review") not in {"pass", "fixed", "discard"}
        or not isinstance(row.get("review_reason"), str)
        or not isinstance(row.get("error_class"), list)
    ]
    fixed_state_rows = sum(
        "state_contradiction" in row.get("metadata", {}).get("cleanup_error_class", [])
        for row in rows
    )
    failures = completed_pending + active_pending + invalid_robot_state + pending_without_prompt + fact_mismatch + missing_review_fields
    status = "PASS" if not failures and len(review) == 200 and not review_status["discard"] else "FAIL"
    report = {
        "status": status,
        "rows": len(rows),
        "completed_pending_contradiction": len(completed_pending),
        "active_pending_contradiction": len(active_pending),
        "invalid_robot_lifecycle": len(invalid_robot_state),
        "pending_without_next_prompt": len(pending_without_prompt),
        "fact_mismatch": len(fact_mismatch),
        "impossible_state_detected": fixed_state_rows,
        "impossible_state_fixed": fixed_state_rows,
        "impossible_state_discarded": review_status["discard"],
        "known_impossible_state": len(completed_pending) + len(active_pending) + len(invalid_robot_state),
        "examples": {
            "completed_pending": completed_pending[:20],
            "active_pending": active_pending[:20],
            "invalid_robot_lifecycle": invalid_robot_state[:20],
            "pending_without_next_prompt": pending_without_prompt[:20],
            "fact_mismatch": fact_mismatch[:20],
        },
        "high_risk_review": {
            "rows": len(review),
            "status_counts": dict(review_status),
            "missing_review_fields": missing_review_fields,
        },
    }
    path = BASE / "reports/response_state_consistency.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
