#!/usr/bin/env python3
import json
import os
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(os.environ.get("DATASET_DIR", "/app/assets/dataset"))
TRAIN = ROOT / "records/decision_train.jsonl"
VALIDATION = ROOT / "records/decision_validation.jsonl"
SYSTEM = ROOT / "contract/decision_system.txt"
SCHEMA = ROOT / "contract/decision_schema.json"
MANIFEST = ROOT / "manifest.json"


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


def main() -> None:
    for path in (TRAIN, VALIDATION, SYSTEM, SCHEMA, MANIFEST):
        if not path.is_file():
            raise FileNotFoundError(path)

    validator = Draft202012Validator(json.loads(SCHEMA.read_text(encoding="utf-8")))
    seen_ids = set()
    counts = {}
    required_input = {
        "message",
        "order",
        "preferences",
        "recommendation",
        "pending_confirmation",
        "action_history",
        "robot_state",
    }

    for name, path in (("train", TRAIN), ("validation", VALIDATION)):
        rows = read_jsonl(path)
        counts[name] = len(rows)
        for row in rows:
            row_id = row.get("id")
            if not row_id or row_id in seen_ids:
                raise ValueError(f"missing or duplicate id: {row_id}")
            seen_ids.add(row_id)
            model_input = row.get("input")
            if not isinstance(model_input, dict):
                raise ValueError(f"{row_id}: input must be an object")
            missing = sorted(required_input - set(model_input))
            if missing:
                raise ValueError(f"{row_id}: missing input fields: {missing}")
            if "history" in model_input:
                raise ValueError(f"{row_id}: history must stay top-level")
            history = row.get("history", [])
            if not isinstance(history, list):
                raise ValueError(f"{row_id}: history must be a list")
            for index, message in enumerate(history):
                if not isinstance(message, dict):
                    raise ValueError(f"{row_id}: history[{index}] must be an object")
                if message.get("role") not in {"user", "assistant"}:
                    raise ValueError(f"{row_id}: invalid history role at {index}")
                if not isinstance(message.get("content"), str) or not message["content"]:
                    raise ValueError(f"{row_id}: empty history content at {index}")
            validator.validate(row.get("target"))

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = {
        "train": manifest["train_rows"],
        "validation": manifest["validation_rows"],
    }
    if counts != expected:
        raise ValueError(f"unexpected dataset counts: {counts}")
    if manifest.get("epochs") != 6:
        raise ValueError("dataset manifest must declare 6 epochs")
    if not SYSTEM.read_text(encoding="utf-8").strip():
        raise ValueError("decision system prompt is empty")
    print(f"PASS: dataset preflight train={counts['train']} validation={counts['validation']}")


if __name__ == "__main__":
    main()
