#!/usr/bin/env python3
"""Response v1 입력 계약과 명백한 structured fact consistency를 검증한다."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data_natural_multiturn_v3_rebuild"
DATA = BASE / "response_v1"
INTERMEDIATE = BASE / "intermediate/response_v1_intermediate.jsonl"
REPORT = BASE / "reports/response_validation.json"
SPLITS = ("train", "validation", "test", "challenge", "holdout")
INPUT_KEYS = {
    "user_text", "recent_history", "confirmed_order", "preferences", "pending", "policy",
    "applied_this_turn", "future_changes", "recommendation_result", "next_prompt",
    "robot_state", "recent_action_history", "execution_authorized", "starting_now",
}
QUESTION_TYPES = {"execution", "recommendation"}
EXECUTION_CLAIMS = ("시작할게", "시작해요", "시작합니다", "진행할게", "담기 시작")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def asks_question(reply: str) -> bool:
    return "?" in reply or bool(re.search(r"(할까요|일까요|인가요|좋으세요|해주세요)[?.]?$", reply))


def asks_or_directs(reply: str) -> bool:
    return asks_question(reply) or bool(re.search(r"(주세요|고르세요|선택하세요)[.]?$", reply))


def main() -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    metadata = {row["id"]: row.get("metadata", {}) for row in read_jsonl(INTERMEDIATE)}
    failures = []
    counts = {}
    families = Counter()
    next_prompts = Counter()
    seen = {}
    duplicates = []
    for split in SPLITS:
        rows = read_jsonl(DATA / f"{split}.jsonl")
        counts[split] = len(rows)
        for line_number, row in enumerate(rows, start=1):
            row_id = row.get("id", f"{split}:{line_number}")
            model_input = row.get("input")
            target = row.get("target")
            reply = target.get("reply") if isinstance(target, dict) else None
            row_meta = metadata.get(row_id, {})
            family = row_meta.get("scenario_family", "unknown")
            families[family] += 1
            if not isinstance(model_input, dict) or set(model_input) != INPUT_KEYS:
                failures.append({"id": row_id, "split": split, "kind": "invalid_input", "detail": "input_keys"})
                continue
            if not isinstance(target, dict) or set(target) != {"reply"} or not isinstance(reply, str) or not reply.strip():
                failures.append({"id": row_id, "split": split, "kind": "empty_reply", "detail": "target_shape"})
                continue
            prompt = model_input.get("next_prompt")
            prompt_type = prompt.get("type") if isinstance(prompt, dict) else "none"
            next_prompts[prompt_type] += 1
            if prompt_type in QUESTION_TYPES and not asks_question(reply):
                failures.append({"id": row_id, "split": split, "kind": "next_prompt", "detail": prompt_type})
            if prompt_type == "missing_order" and not asks_or_directs(reply):
                failures.append({"id": row_id, "split": split, "kind": "next_prompt", "detail": prompt_type})
            if prompt_type == "section_selection" and not any(word in reply for word in ("넘어", "건너뛰")):
                failures.append({"id": row_id, "split": split, "kind": "next_prompt", "detail": "section_skip_missing"})
            if model_input.get("pending") is not None and prompt is not None:
                pending_type = model_input["pending"].get("type") if isinstance(model_input["pending"], dict) else None
                if pending_type in {"execution", "recommendation"} and not asks_question(reply):
                    failures.append({"id": row_id, "split": split, "kind": "next_prompt", "detail": "pending_question_missing"})
            if not model_input.get("execution_authorized") and any(claim in reply for claim in EXECUTION_CLAIMS):
                failures.append({"id": row_id, "split": split, "kind": "fact_consistency", "detail": "unauthorized_execution_claim"})
            if not model_input.get("starting_now") and any(claim in reply for claim in ("담기 시작", "작업을 진행할게")):
                failures.append({"id": row_id, "split": split, "kind": "fact_consistency", "detail": "empty_starting_now_claim"})
            if "no_forced_question" in row_meta.get("coverage_tags", []) and asks_question(reply):
                failures.append({"id": row_id, "split": split, "kind": "next_prompt", "detail": "forced_question"})
            pending = model_input.get("pending")
            applied = model_input.get("applied_this_turn") or {}
            if isinstance(pending, dict) and pending.get("type") == "recommendation" and not (applied.get("order_changes") or {}):
                if re.search(r"(추가했|반영했|적용했|저장했)", reply):
                    failures.append({"id": row_id, "split": split, "kind": "fact_consistency", "detail": "recommendation_as_confirmed"})
            signature = json.dumps({"route": row.get("route"), "input": model_input, "target": target}, ensure_ascii=False, sort_keys=True)
            if signature in seen:
                duplicates.append({"first": seen[signature], "second": row_id})
            else:
                seen[signature] = row_id
    kinds = Counter(item["kind"] for item in failures)
    report = {
        "status": "PASS" if not failures and not duplicates else "FAIL",
        "counts": counts,
        "invalid_input": kinds["invalid_input"],
        "empty_reply": kinds["empty_reply"],
        "fact_consistency_failure": kinds["fact_consistency"],
        "next_prompt_failure": kinds["next_prompt"],
        "duplicate": len(duplicates),
        "families": dict(families),
        "next_prompt_types": dict(next_prompts),
        "failures": failures[:100],
        "duplicate_examples": duplicates[:20],
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
