#!/usr/bin/env python3
"""이미지 빌드 중 Response v1.1 내장 자료와 contract를 검증한다."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(os.environ.get("DATASET_DIR", "/app/assets/dataset"))
MANIFEST_PATH = ROOT / "manifest.json"
INPUT_FIELDS = {
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
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    train_path = ROOT / "response/records/train.jsonl"
    validation_path = ROOT / "response/records/validation.jsonl"
    schema_path = ROOT / "contract/response_schema.json"
    systems_path = ROOT / "contract/response_system_by_route.json"
    required = (
        MANIFEST_PATH,
        ROOT / "SHA256SUMS",
        ROOT / "token_length_report.json",
        train_path,
        validation_path,
        schema_path,
        systems_path,
    )
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("training_image_spec") != "2.9":
        raise ValueError("manifest training_image_spec must be 2.9")
    if manifest.get("task") != "response_v1_1_adapter":
        raise ValueError("manifest task mismatch")
    if manifest.get("epochs") != 4 or manifest.get("planned_optimizer_steps") != 276:
        raise ValueError("training schedule mismatch")
    token_audit = json.loads((ROOT / "token_length_report.json").read_text(encoding="utf-8"))
    if token_audit.get("max_length") != 4096 or any(
        split["rows_over_max_after_trim"] != 0
        for split in token_audit.get("splits", {}).values()
    ):
        raise ValueError("Response token length audit failed")

    validator = Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8")))
    counts = {}
    seen_ids = set()
    for split, path in (("train", train_path), ("validation", validation_path)):
        rows = read_jsonl(path)
        counts[split] = len(rows)
        for row in rows:
            row_id = row.get("id")
            if not isinstance(row_id, str) or not row_id or row_id in seen_ids:
                raise ValueError(f"{split}: missing or duplicate id: {row_id}")
            seen_ids.add(row_id)
            model_input = row.get("input")
            if not isinstance(model_input, dict) or set(model_input) != INPUT_FIELDS:
                raise ValueError(f"{split}:{row_id}: Response input contract mismatch")
            if row.get("route") not in {"task", "general", "mixed"}:
                raise ValueError(f"{split}:{row_id}: invalid response route")
            validator.validate(row.get("target"))

    if counts != {"train": 1098, "validation": 101}:
        raise ValueError(f"Response v1.1 row count mismatch: {counts}")
    if counts != {"train": manifest["train_rows"], "validation": manifest["validation_rows"]}:
        raise ValueError(f"dataset row count mismatch: {counts}")
    for relative, expected_hash in manifest.get("sha256", {}).items():
        if sha256_file(ROOT / relative) != expected_hash:
            raise ValueError(f"dataset hash mismatch: {relative}")
    prompts = json.loads(systems_path.read_text(encoding="utf-8"))
    if set(prompts) != {"task", "general", "mixed"} or not all(prompts.values()):
        raise ValueError("response route prompts are incomplete")
    print(f"PASS: Response v1.1 dataset train={counts['train']} validation={counts['validation']}")


if __name__ == "__main__":
    main()
