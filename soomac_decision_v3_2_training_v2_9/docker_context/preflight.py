#!/usr/bin/env python3
"""이미지 빌드 중 Decision v3.2 내장 자료와 contract를 검증한다."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(os.environ.get("DATASET_DIR", "/app/assets/dataset"))
MANIFEST_PATH = ROOT / "manifest.json"
INPUT_FIELDS = {
    "recent_history",
    "order",
    "preferences",
    "pending",
    "robot_state",
    "message",
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
    train_path = ROOT / "decision/records/train.jsonl"
    validation_path = ROOT / "decision/records/validation.jsonl"
    schema_path = ROOT / "contract/decision_schema.json"
    system_path = ROOT / "contract/decision_system.txt"
    required = (
        MANIFEST_PATH,
        ROOT / "SHA256SUMS",
        ROOT / "token_length_report.json",
        train_path,
        validation_path,
        schema_path,
        system_path,
    )
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("training_image_spec") != "2.9":
        raise ValueError("manifest training_image_spec must be 2.9")
    if manifest.get("task") != "decision_v3_2_adapter":
        raise ValueError("manifest task mismatch")
    if manifest.get("epochs") != 6 or manifest.get("planned_optimizer_steps") != 1302:
        raise ValueError("training schedule mismatch")
    token_audit = json.loads((ROOT / "token_length_report.json").read_text(encoding="utf-8"))
    if token_audit.get("max_length") != 4096 or any(
        split["rows_over_max_after_trim"] != 0
        for split in token_audit.get("splits", {}).values()
    ):
        raise ValueError("Decision token length audit failed")

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
            if not isinstance(row.get("input"), dict) or set(row["input"]) != INPUT_FIELDS:
                raise ValueError(f"{split}:{row_id}: Decision input contract mismatch")
            validator.validate(row.get("target"))

    if counts != {"train": manifest["train_rows"], "validation": manifest["validation_rows"]}:
        raise ValueError(f"dataset row count mismatch: {counts}")
    for relative, expected_hash in manifest.get("sha256", {}).items():
        if sha256_file(ROOT / relative) != expected_hash:
            raise ValueError(f"dataset hash mismatch: {relative}")
    if not system_path.read_text(encoding="utf-8").strip():
        raise ValueError("Decision system prompt is empty")
    if counts != {"train": 3465, "validation": 285}:
        raise ValueError(f"Decision v3.2 row count mismatch: {counts}")
    print(f"PASS: Decision v3.2 dataset train={counts['train']} validation={counts['validation']}")


if __name__ == "__main__":
    main()
