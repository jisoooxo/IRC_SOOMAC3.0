#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(os.environ.get("DATASET_DIR", "/app/assets/dataset"))
MANIFEST_PATH = ROOT / "manifest.json"

DECISION_INPUT_FIELDS = {
    "recent_history",
    "order",
    "preferences",
    "pending",
    "robot_state",
    "message",
}
RESPONSE_INPUT_FIELDS = {
    "user_text",
    "recent_history",
    "confirmed_order",
    "preferences",
    "pending",
    "policy",
    "applied_this_turn",
    "future_changes",
    "recommendation_result",
    "next_prompt",
    "robot_state",
    "recent_action_history",
    "execution_authorized",
    "starting_now",
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_stage(stage: str, expected_input_fields: set[str], schema_path: Path) -> dict:
    validator = Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8")))
    counts = {}
    seen_ids = set()
    for split in ("train", "validation"):
        rows = read_jsonl(ROOT / stage / "records" / f"{split}.jsonl")
        counts[split] = len(rows)
        for row in rows:
            row_id = row.get("id")
            if not isinstance(row_id, str) or not row_id or row_id in seen_ids:
                raise ValueError(f"{stage}: missing or duplicate id: {row_id}")
            seen_ids.add(row_id)
            model_input = row.get("input")
            if not isinstance(model_input, dict):
                raise ValueError(f"{stage}:{row_id}: input must be an object")
            if set(model_input) != expected_input_fields:
                raise ValueError(
                    f"{stage}:{row_id}: input fields mismatch: "
                    f"expected={sorted(expected_input_fields)} actual={sorted(model_input)}"
                )
            if stage == "response" and row.get("route") not in {
                "task",
                "general",
                "mixed",
            }:
                raise ValueError(f"response:{row_id}: invalid route")
            validator.validate(row.get("target"))
    return counts


def main() -> None:
    required = (
        MANIFEST_PATH,
        ROOT / "decision/records/train.jsonl",
        ROOT / "decision/records/validation.jsonl",
        ROOT / "response/records/train.jsonl",
        ROOT / "response/records/validation.jsonl",
        ROOT / "contract/decision_system.txt",
        ROOT / "contract/decision_schema.json",
        ROOT / "contract/response_schema.json",
        ROOT / "contract/response_system_by_route.json",
    )
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("training_image_spec") != "2.9":
        raise ValueError("manifest training_image_spec must be 2.9")
    if manifest.get("task") != "decision_response_dual_adapter":
        raise ValueError("manifest task mismatch")

    decision_counts = validate_stage(
        "decision",
        DECISION_INPUT_FIELDS,
        ROOT / "contract/decision_schema.json",
    )
    response_counts = validate_stage(
        "response",
        RESPONSE_INPUT_FIELDS,
        ROOT / "contract/response_schema.json",
    )
    expected = manifest["stages"]
    for stage, counts in (("decision", decision_counts), ("response", response_counts)):
        if counts["train"] != expected[stage]["train_rows"]:
            raise ValueError(f"{stage}: train row count mismatch")
        if counts["validation"] != expected[stage]["validation_rows"]:
            raise ValueError(f"{stage}: validation row count mismatch")
    if expected["decision"].get("epochs") != 6:
        raise ValueError("decision epochs must be 6")
    if expected["response"].get("epochs") != 4:
        raise ValueError("response epochs must be 4")

    for relative, expected_hash in manifest.get("sha256", {}).items():
        actual_hash = sha256_file(ROOT / relative)
        if actual_hash != expected_hash:
            raise ValueError(f"dataset hash mismatch: {relative}")

    prompts = json.loads(
        (ROOT / "contract/response_system_by_route.json").read_text(encoding="utf-8")
    )
    if set(prompts) != {"task", "general", "mixed"} or not all(prompts.values()):
        raise ValueError("response route prompts are incomplete")
    print(
        "PASS: dual dataset preflight "
        f"decision={decision_counts} response={response_counts}",
        flush=True,
    )


if __name__ == "__main__":
    main()
