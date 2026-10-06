#!/usr/bin/env python3
"""Review generated route-boundary samples, validate them, and merge without overwriting the base."""

from __future__ import annotations

import copy
import difflib
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import jsonschema

from soomac_irc.agent_contract import DECISION_SCHEMA
from soomac_irc.agent_prompts import DECISION_SYSTEM
from legacy_dialogue_focus import (
    BLOCKED_COMMIT_MARKERS,
    EXPLICIT_COMMIT_PHRASES,
    build_reference_context,
)


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
BASE_PATH = BUNDLE / "output" / "final_train.jsonl"

SEMANTIC_FIELDS = {
    "order", "restrictions", "preferences", "recommendation",
    "commit", "confirmation", "queries", "clarify",
}
REFERENCE_WORDS = ("그거", "이거", "방금 그거", "아까 그거", "전에 말한 거", "둘 다", "두 개 다", "첫 번째 거", "두 번째 거")


class DuplicateKeyError(ValueError):
    pass


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(key)
        result[key] = value
    return result


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line, object_pairs_hook=reject_duplicate_keys))
            except Exception as error:
                raise RuntimeError(f"{path}:{line_number}: {error}") from error
    return rows


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_text(value: str) -> str:
    return "".join(value.split())


def normalized_surface(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]", "", value).lower()


def context_identity(row: dict) -> str:
    return canonical({"input": row.get("input"), "target": row.get("target")})


def state_signature(row: dict) -> str:
    model_input = copy.deepcopy(row["input"])
    model_input.pop("message", None)
    return canonical({"state": model_input, "history": row.get("history", []), "target": row["target"]})


def mention_positions(message: str, mentions: list[str]) -> list[int] | None:
    positions = []
    cursor = 0
    for mention in mentions:
        position = message.find(mention, cursor)
        if position < 0:
            return None
        positions.append(position)
        cursor = position + len(mention)
    return positions


def review_row(row: dict, validator: jsonschema.Draft202012Validator) -> list[str]:
    errors = []
    target = row.get("target")
    model_input = row.get("input")
    spec = row.get("_review_spec")

    if not isinstance(target, dict):
        return ["target_not_object"]
    if not isinstance(model_input, dict):
        return ["input_not_object"]
    if not isinstance(spec, dict):
        return ["missing_review_spec"]

    errors.extend(f"schema:{error.message}" for error in validator.iter_errors(target))

    message = model_input.get("message")
    if not isinstance(message, str) or not message.strip():
        errors.append("missing_message")
        message = ""

    if "history" in model_input:
        errors.append("runtime_input_contains_history_copy")

    expected_target = spec.get("expected_target")
    if target != expected_target:
        errors.append("semantic_atom_mismatch")
    if target.get("route") != spec.get("expected_route"):
        errors.append("route_mismatch")

    mentions = target.get("mentions", [])
    expected_mentions = spec.get("expected_mentions", [])
    if mentions != expected_mentions:
        errors.append("mentions_mismatch")
    if len(mentions) != len(set(mentions)):
        errors.append("duplicate_mentions")
    if mention_positions(message, mentions) is None:
        errors.append("mention_not_grounded_or_out_of_order")
    for mention in mentions:
        if mention in REFERENCE_WORDS:
            errors.append(f"reference_word_in_mentions:{mention}")

    for evidence in spec.get("task_evidence", []):
        if evidence not in message:
            errors.append(f"missing_task_evidence:{evidence}")
    for evidence in spec.get("general_evidence", []):
        if evidence not in message:
            errors.append(f"missing_general_evidence:{evidence}")

    semantic_keys = set(target) & SEMANTIC_FIELDS
    route = target.get("route")
    if route == "general" and semantic_keys:
        errors.append(f"general_has_task_semantics:{sorted(semantic_keys)}")
    if route == "mixed":
        if not semantic_keys:
            errors.append("mixed_missing_task_semantic")
        if not spec.get("task_evidence"):
            errors.append("mixed_missing_task_clause")
        if not spec.get("general_evidence"):
            errors.append("mixed_missing_general_clause")

    if spec.get("unsupported_task"):
        if target.get("clarify") is not True:
            errors.append("unsupported_missing_clarify")
        if "order" in target:
            errors.append("unsupported_substituted_to_supported_order")

    if target.get("commit") is True:
        text = compact_text(message)
        if not any(compact_text(phrase) in text for phrase in EXPLICIT_COMMIT_PHRASES):
            errors.append("commit_without_explicit_phrase")
        if any(compact_text(marker) in text for marker in BLOCKED_COMMIT_MARKERS):
            errors.append("commit_with_blocked_marker")

    history = row.get("history", [])
    if len(history) % 2:
        errors.append("history_not_complete_turn_pairs")
    for index, item in enumerate(history):
        expected_role = "user" if index % 2 == 0 else "assistant"
        if item.get("role") != expected_role or not isinstance(item.get("content"), str):
            errors.append("history_role_or_content_invalid")
            break

    reference_context = model_input.get("reference_context", {"status": "none", "targets": []})
    focus = model_input.get("dialogue_focus", {"current": None, "recent": []})
    current_history_turn = len(history) // 2 + 1
    recomputed_reference = build_reference_context(message, focus, current_history_turn)
    if reference_context != recomputed_reference:
        errors.append(
            f"reference_context_mismatch:{canonical(reference_context)}!={canonical(recomputed_reference)}"
        )
    if reference_context.get("status") == "resolved":
        for resolved_target in reference_context.get("targets", []):
            if resolved_target in mentions and resolved_target not in message:
                errors.append(f"resolved_target_copied_to_mentions:{resolved_target}")

    return sorted(set(errors))


def clean_row(row: dict) -> dict:
    cleaned = copy.deepcopy(row)
    cleaned.pop("_review_spec", None)
    cleaned.pop("reviewer_errors", None)
    cleaned.pop("validator_errors", None)
    cleaned["generation_meta"]["reviewed"] = True
    cleaned["generation_meta"]["validated"] = True
    return cleaned


def route_counts(rows: list[dict]) -> dict[str, int]:
    counts = Counter(row["target"]["route"] for row in rows)
    return {route: counts.get(route, 0) for route in ("task", "general", "mixed")}


def validate_hard_eval(
    rows: list[dict],
    validator: jsonschema.Draft202012Validator,
    train_messages: set[str],
) -> tuple[list[dict], dict]:
    passed = []
    failures = []
    for row in rows:
        errors = review_row(row, validator)
        if row["input"]["message"] in train_messages:
            errors.append("hard_eval_exact_train_leakage")
        if errors:
            failed = copy.deepcopy(row)
            failed["reviewer_errors"] = sorted(set(errors))
            failures.append(failed)
        else:
            passed.append(clean_row(row))

    pair_routes = defaultdict(set)
    for row in passed:
        if row.get("eval_category") == "general_task_minimal_pair":
            pair_routes[row["generation_meta"]["pair_id"]].add(row["target"]["route"])
    complete_pairs = sum(routes == {"general", "task"} for routes in pair_routes.values())
    category_counts = Counter(row["eval_category"] for row in passed)
    stats = {
        "raw": len(rows),
        "pass": len(passed),
        "fail": len(failures),
        "category_counts": dict(sorted(category_counts.items())),
        "general_task_minimal_pairs": complete_pairs,
        "exact_train_leakage": 0,
    }
    if failures:
        dump_jsonl(HERE / "hard_eval_fail.jsonl", failures)
    else:
        dump_jsonl(HERE / "hard_eval_fail.jsonl", [])
    return passed, stats


def main() -> None:
    validator = jsonschema.Draft202012Validator(DECISION_SCHEMA)
    base = read_jsonl(BASE_PATH)
    general_raw = read_jsonl(HERE / "generated_general_raw.jsonl")
    mixed_raw = read_jsonl(HERE / "generated_mixed_raw.jsonl")
    hard_raw = read_jsonl(HERE / "hard_eval_raw.jsonl")

    before_counts = route_counts(base)
    if len(base) != 2789 or before_counts != {"task": 2528, "general": 229, "mixed": 32}:
        raise RuntimeError(f"unexpected base dataset: total={len(base)}, routes={before_counts}")

    reviewer_pass = []
    reviewer_reject = []
    for row in general_raw + mixed_raw:
        errors = review_row(row, validator)
        if errors:
            rejected = copy.deepcopy(row)
            rejected["reviewer_errors"] = errors
            reviewer_reject.append(rejected)
        else:
            accepted = copy.deepcopy(row)
            accepted["generation_meta"]["reviewed"] = True
            reviewer_pass.append(accepted)

    dump_jsonl(HERE / "reviewer_pass.jsonl", reviewer_pass)
    dump_jsonl(HERE / "reviewer_reject.jsonl", reviewer_reject)

    existing_identities = {context_identity(row) for row in base}
    accepted_identities = set()
    exact_duplicates = []
    validated_pass = []
    validated_fail = []

    near_buckets: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in base:
        near_buckets[state_signature(row)].append((row.get("id", "base"), row["input"]["message"]))

    near_duplicate_pairs = []
    for row in reviewer_pass:
        errors = []
        errors.extend(f"schema:{error.message}" for error in validator.iter_errors(row["target"]))
        message = row["input"]["message"]
        for mention in row["target"].get("mentions", []):
            if mention not in message:
                errors.append(f"mention_not_grounded:{mention}")

        identity = context_identity(row)
        if identity in existing_identities or identity in accepted_identities:
            duplicate = copy.deepcopy(row)
            duplicate["dedup_reason"] = "exact_full_input_target_duplicate"
            exact_duplicates.append(duplicate)
            continue

        signature = state_signature(row)
        normalized = normalized_surface(message)
        for other_id, other_message in near_buckets[signature]:
            other_normalized = normalized_surface(other_message)
            ratio = difflib.SequenceMatcher(None, normalized, other_normalized).ratio()
            if ratio >= 0.965:
                errors.append(f"near_duplicate:{other_id}:{ratio:.3f}")
                near_duplicate_pairs.append({
                    "id": row["id"],
                    "other_id": other_id,
                    "ratio": round(ratio, 4),
                    "message": message,
                    "other_message": other_message,
                })
                break

        if errors:
            failed = copy.deepcopy(row)
            failed["validator_errors"] = sorted(set(errors))
            validated_fail.append(failed)
            continue

        cleaned = clean_row(row)
        validated_pass.append(cleaned)
        accepted_identities.add(identity)
        near_buckets[signature].append((row["id"], message))

    dump_jsonl(HERE / "validated_pass.jsonl", validated_pass)
    dump_jsonl(HERE / "validated_fail.jsonl", validated_fail)
    dump_jsonl(HERE / "dedup_removed.jsonl", exact_duplicates)
    dump_jsonl(HERE / "rebalance_supplement.jsonl", validated_pass)

    final = base + validated_pass
    if final[: len(base)] != base:
        raise AssertionError("base prefix changed")
    dump_jsonl(HERE / "final_train_rebalanced.jsonl", final)

    final_messages = {row["input"]["message"] for row in final}
    hard_pass, hard_stats = validate_hard_eval(hard_raw, validator, final_messages)
    dump_jsonl(HERE / "hard_eval.jsonl", hard_pass)

    after_counts = route_counts(final)
    total = len(final)
    percentages = {route: round(after_counts[route] * 100 / total, 2) for route in ("task", "general", "mixed")}
    family_counts = Counter(row["scenario_family"] for row in validated_pass)

    contract_payload = canonical({"schema": DECISION_SCHEMA, "system": DECISION_SYSTEM})
    manifest = {
        "contract": {
            "decision_schema_sha256": sha256_text(canonical(DECISION_SCHEMA)),
            "decision_system_sha256": sha256_text(DECISION_SYSTEM),
            "combined_sha256": sha256_text(contract_payload),
            "schema_fields_added": [],
        },
        "before": {"total": len(base), **before_counts, "sha256": sha256_file(BASE_PATH)},
        "generated": {
            "general_raw": len(general_raw),
            "mixed_raw": len(mixed_raw),
            "reviewer_pass": len(reviewer_pass),
            "reviewer_reject": len(reviewer_reject),
            "validator_pass": len(validated_pass),
            "validator_fail": len(validated_fail),
            "dedup_removed": len(exact_duplicates),
            "supplement_final": len(validated_pass),
            "family_counts": dict(sorted(family_counts.items())),
            "near_duplicate_pairs_rejected": len(near_duplicate_pairs),
        },
        "after": {"total": total, **after_counts, "percentage": percentages},
        "preservation": {
            "base_rows_prefix_identical": True,
            "base_rows_preserved": len(base),
            "state_context_in_dedup_identity": True,
            "minimal_pairs_not_deduped_by_surface": True,
            "base_overwritten": False,
        },
        "hard_eval": hard_stats,
        "outputs": {
            "final_train_rebalanced": str(HERE / "final_train_rebalanced.jsonl"),
            "hard_eval": str(HERE / "hard_eval.jsonl"),
        },
    }
    (HERE / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report = f"""# Route rebalance report

## BEFORE

total = {len(base)}
task = {before_counts['task']}
general = {before_counts['general']}
mixed = {before_counts['mixed']}

## GENERATED

general raw = {len(general_raw)}
mixed raw = {len(mixed_raw)}
review pass = {len(reviewer_pass)}
review reject = {len(reviewer_reject)}
validator fail = {len(validated_fail)}
dedup removed = {len(exact_duplicates)}
supplement final = {len(validated_pass)}

## AFTER

total = {total}
task = {after_counts['task']}
general = {after_counts['general']}
mixed = {after_counts['mixed']}

percentage:
task = {percentages['task']:.2f}%
general = {percentages['general']:.2f}%
mixed = {percentages['mixed']:.2f}%

## Validation

- Decision schema violations rejected = {len(validated_fail)}
- exact duplicates removed = {len(exact_duplicates)}
- near duplicates rejected = {len(near_duplicate_pairs)}
- original base rows preserved byte-for-value = {len(base)}
- schema fields added = 0

## Hard eval

- total = {hard_stats['pass']}
- general/task minimal pairs = {hard_stats['general_task_minimal_pairs']}
- mixed boundary = {hard_stats['category_counts'].get('mixed_boundary', 0)}
- reference mixed = {hard_stats['category_counts'].get('reference_mixed', 0)}
- unsupported mixed = {hard_stats['category_counts'].get('unsupported_mixed', 0)}
- query vs general = {hard_stats['category_counts'].get('query_vs_general', 0)}
- commit mixed = {hard_stats['category_counts'].get('commit_mixed', 0)}
- exact train leakage = {hard_stats['exact_train_leakage']}
"""
    (HERE / "report.md").write_text(report, encoding="utf-8")

    print(json.dumps({
        "reviewer_pass": len(reviewer_pass),
        "reviewer_reject": len(reviewer_reject),
        "validator_pass": len(validated_pass),
        "validator_fail": len(validated_fail),
        "dedup_removed": len(exact_duplicates),
        "after": manifest["after"],
        "hard_eval": hard_stats,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
