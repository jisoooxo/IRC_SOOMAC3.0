#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

from jsonschema import Draft202012Validator

from soomac_irc.agent_contract import DECISION_SCHEMA
from soomac_irc.agent_prompts import DECISION_SYSTEM


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "soomac_decision_migration_merge_bundle"
    / "route_rebalance"
    / "final_train_rebalanced.jsonl"
)
HARD_EVAL = SOURCE.with_name("hard_eval.jsonl")
DATASET = Path(__file__).resolve().parent / "docker_context" / "assets" / "dataset"
SPLIT_SALT = "soomac-decision-v29-epoch6"


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


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_history(row: dict) -> None:
    history = row.get("history", [])
    if not isinstance(history, list):
        raise ValueError(f"{row.get('id')}: history must be a list")
    for index, message in enumerate(history):
        if not isinstance(message, dict):
            raise ValueError(f"{row.get('id')}: history[{index}] must be an object")
        if message.get("role") not in {"user", "assistant"}:
            raise ValueError(f"{row.get('id')}: invalid history role at {index}")
        if not isinstance(message.get("content"), str) or not message["content"]:
            raise ValueError(f"{row.get('id')}: empty history content at {index}")
    if isinstance(row.get("input"), dict) and "history" in row["input"]:
        raise ValueError(f"{row.get('id')}: history must stay top-level")


def split_key(row: dict) -> str:
    return str(row.get("semantic_id") or row.get("paraphrase_group") or row["id"])


def is_validation(row: dict) -> bool:
    digest = hashlib.sha256(f"{SPLIT_SALT}:{split_key(row)}".encode()).hexdigest()
    return int(digest[:8], 16) % 10 == 0


def route_counts(rows: list[dict]) -> dict[str, int]:
    counts = Counter(row["target"]["route"] for row in rows)
    return {route: counts.get(route, 0) for route in ("task", "general", "mixed")}


def main() -> None:
    rows = read_jsonl(SOURCE)
    validator = Draft202012Validator(DECISION_SCHEMA)
    seen_ids = set()
    for row in rows:
        row_id = row.get("id")
        if not row_id or row_id in seen_ids:
            raise ValueError(f"missing or duplicate id: {row_id}")
        seen_ids.add(row_id)
        nested_history = row.get("input", {}).get("history")
        if nested_history is not None:
            if nested_history != row.get("history", []):
                raise ValueError(f"{row_id}: nested history differs from top-level history")
            row["input"].pop("history")
        validate_history(row)
        validator.validate(row["target"])

    train = [row for row in rows if not is_validation(row)]
    validation = [row for row in rows if is_validation(row)]
    train_groups = {split_key(row) for row in train}
    validation_groups = {split_key(row) for row in validation}
    if train_groups & validation_groups:
        raise RuntimeError("semantic group leakage between train and validation")

    records = DATASET / "records"
    contract = DATASET / "contract"
    evaluation = DATASET / "evaluation"
    contract.mkdir(parents=True, exist_ok=True)
    evaluation.mkdir(parents=True, exist_ok=True)
    write_jsonl(records / "decision_train.jsonl", train)
    write_jsonl(records / "decision_validation.jsonl", validation)
    shutil.copyfile(HARD_EVAL, evaluation / "hard_eval.jsonl")
    (contract / "decision_system.txt").write_text(
        DECISION_SYSTEM.strip() + "\n", encoding="utf-8"
    )
    (contract / "decision_schema.json").write_text(
        json.dumps(DECISION_SCHEMA, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "training_image_spec": "2.9",
        "source": str(SOURCE.relative_to(ROOT)),
        "split_method": "semantic_id SHA-256 bucket; bucket 0 is validation",
        "split_salt": SPLIT_SALT,
        "epochs": 6,
        "total_rows": len(rows),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_route_counts": route_counts(train),
        "validation_route_counts": route_counts(validation),
        "train_history_rows": sum(bool(row.get("history")) for row in train),
        "validation_history_rows": sum(bool(row.get("history")) for row in validation),
        "semantic_group_overlap": 0,
        "hard_eval_rows": len(read_jsonl(HARD_EVAL)),
    }
    for relative in (
        "records/decision_train.jsonl",
        "records/decision_validation.jsonl",
        "evaluation/hard_eval.jsonl",
        "contract/decision_system.txt",
        "contract/decision_schema.json",
    ):
        manifest.setdefault("sha256", {})[relative] = sha256_file(DATASET / relative)
    (DATASET / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
