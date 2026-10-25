#!/usr/bin/env python3
"""Decision v3 JSONL의 현재 external sparse contract를 검증한다."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data_natural_multiturn_v3_rebuild/decision_v3"
REPORT = ROOT / "data_natural_multiturn_v3_rebuild/reports/decision_validation.json"
SPLITS = ("train", "validation", "test", "challenge", "holdout")
INPUT_KEYS = {"recent_history", "order", "preferences", "pending", "robot_state", "message"}
TARGET_KEYS = {"route", "order", "restrictions", "preferences", "recommendation", "commit", "confirmation", "clarify"}
LEGACY_KEYS = {"mentions", "queries", "order_patch", "restriction_options", "preference_options", "understanding"}
ROUTES = {"task", "general", "mixed"}
AMOUNTS = {"low", "normal", "high", "none"}
REASONS = {"dislike", "cannot_eat", "allergy", "dietary_rule"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_target(target: object) -> list[str]:
    errors = []
    if not isinstance(target, dict):
        return ["target_not_object"]
    unknown = set(target) - TARGET_KEYS
    if unknown:
        errors.append("unknown_target_fields:" + ",".join(sorted(unknown)))
    legacy = set(target) & LEGACY_KEYS
    if legacy:
        errors.append("legacy_fields:" + ",".join(sorted(legacy)))
    if target.get("route") not in ROUTES:
        errors.append("invalid_route")
    order = target.get("order")
    if order is not None:
        if not isinstance(order, dict):
            errors.append("order_not_object")
        else:
            if set(order) - {"sauce", "noodle_type", "noodle_portion", "toppings"}:
                errors.append("unknown_order_field")
            for field in ("sauce", "noodle_type", "noodle_portion"):
                if field in order and not isinstance(order[field], str):
                    errors.append(f"invalid_{field}_type")
            toppings = order.get("toppings")
            if toppings is not None:
                if not isinstance(toppings, dict):
                    errors.append("toppings_not_object")
                else:
                    for item, amount in toppings.items():
                        if not isinstance(item, str) or not item or amount not in AMOUNTS:
                            errors.append("invalid_topping")
    restrictions = target.get("restrictions")
    if restrictions is not None:
        if not isinstance(restrictions, list):
            errors.append("restrictions_not_array")
        else:
            for item in restrictions:
                if not isinstance(item, dict) or set(item) != {"target", "reason", "action"}:
                    errors.append("invalid_restriction_shape")
                    continue
                if not isinstance(item["target"], str) or not item["target"] or item["reason"] not in REASONS or item["action"] not in {"add", "remove"}:
                    errors.append("invalid_restriction_value")
    preferences = target.get("preferences")
    if preferences is not None:
        if not isinstance(preferences, list):
            errors.append("preferences_not_array")
        else:
            for item in preferences:
                if not isinstance(item, dict) or set(item) != {"value", "action"}:
                    errors.append("invalid_preference_shape")
                    continue
                if not isinstance(item["value"], str) or not item["value"] or item["action"] not in {"add", "remove"}:
                    errors.append("invalid_preference_value")
    recommendation = target.get("recommendation")
    if recommendation is not None and (
        not isinstance(recommendation, dict)
        or set(recommendation) != {"action"}
        or recommendation.get("action") not in {"request", "revise"}
    ):
        errors.append("invalid_recommendation")
    if "commit" in target and not isinstance(target["commit"], bool):
        errors.append("invalid_commit")
    if "confirmation" in target and target["confirmation"] not in {"accept", "reject"}:
        errors.append("invalid_confirmation")
    if "clarify" in target and not isinstance(target["clarify"], bool):
        errors.append("invalid_clarify")
    return errors


def main() -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    counts = {}
    target_fields = Counter()
    routes = Counter()
    seen = {}
    duplicate = []
    for split in SPLITS:
        rows = read_jsonl(DATA / f"{split}.jsonl")
        counts[split] = len(rows)
        for line_number, row in enumerate(rows, start=1):
            row_id = row.get("id", f"{split}:{line_number}")
            model_input = row.get("input")
            if not isinstance(model_input, dict) or set(model_input) != INPUT_KEYS:
                failures.append({"id": row_id, "split": split, "error": "invalid_input_keys"})
            elif not isinstance(model_input.get("message"), str) or not model_input["message"].strip():
                failures.append({"id": row_id, "split": split, "error": "empty_message"})
            target_errors = validate_target(row.get("target"))
            for error in target_errors:
                failures.append({"id": row_id, "split": split, "error": error})
            target = row.get("target") if isinstance(row.get("target"), dict) else {}
            routes[target.get("route", "missing")] += 1
            target_fields.update(target.keys())
            signature = json.dumps({"input": model_input, "target": target}, ensure_ascii=False, sort_keys=True)
            if signature in seen:
                duplicate.append({"first": seen[signature], "second": row_id})
            else:
                seen[signature] = row_id
    report = {
        "status": "PASS" if not failures and not duplicate else "FAIL",
        "counts": counts,
        "schema_invalid": len(failures),
        "legacy_field": sum(1 for item in failures if item["error"].startswith("legacy_fields")),
        "duplicate": len(duplicate),
        "routes": dict(routes),
        "target_fields": dict(target_fields),
        "failures": failures[:100],
        "duplicate_examples": duplicate[:20],
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
