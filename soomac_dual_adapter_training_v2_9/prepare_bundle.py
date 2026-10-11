#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "src" / "soomac_irc"
sys.path.insert(0, str(PACKAGE_ROOT))

from soomac_irc.agent_contract import DECISION_SCHEMA, RESPONSE_SCHEMA
from soomac_irc.agent_prompts import (
    DECISION_SYSTEM,
    GENERAL_RESPONSE_SYSTEM,
    MIXED_RESPONSE_SYSTEM,
    TASK_RESPONSE_SYSTEM,
)


SOURCE_ROOT = ROOT / "data_natural_multiturn_v3_rebuild"
DATASET_ROOT = Path(__file__).resolve().parent / "docker_context" / "assets" / "dataset"

STAGES = {
    "decision": {
        "source": SOURCE_ROOT / "decision_v3",
        "epochs": 6,
        "schema": DECISION_SCHEMA,
    },
    "response": {
        "source": SOURCE_ROOT / "response_v1",
        "epochs": 4,
        "schema": RESPONSE_SCHEMA,
    },
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


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_rows(stage: str, rows: list[dict], schema: dict) -> None:
    validator = Draft202012Validator(schema)
    seen_ids = set()
    for row in rows:
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id or row_id in seen_ids:
            raise ValueError(f"{stage}: missing or duplicate id: {row_id}")
        seen_ids.add(row_id)
        if not isinstance(row.get("input"), dict):
            raise ValueError(f"{stage}:{row_id}: input must be an object")
        validator.validate(row.get("target"))
        if stage == "response" and row.get("route") not in {"task", "general", "mixed"}:
            raise ValueError(f"response:{row_id}: invalid route")


def main() -> None:
    contracts = DATASET_ROOT / "contract"
    contracts.mkdir(parents=True, exist_ok=True)
    (contracts / "decision_system.txt").write_text(
        DECISION_SYSTEM.strip() + "\n", encoding="utf-8"
    )
    write_json(contracts / "decision_schema.json", DECISION_SCHEMA)
    write_json(contracts / "response_schema.json", RESPONSE_SCHEMA)
    write_json(
        contracts / "response_system_by_route.json",
        {
            "task": TASK_RESPONSE_SYSTEM.strip(),
            "general": GENERAL_RESPONSE_SYSTEM.strip(),
            "mixed": MIXED_RESPONSE_SYSTEM.strip(),
        },
    )

    manifest = {
        "training_image_spec": "2.9",
        "task": "decision_response_dual_adapter",
        "stages": {},
        "sha256": {},
    }
    for stage, config in STAGES.items():
        source = config["source"]
        train_rows = read_jsonl(source / "train.jsonl")
        validation_rows = read_jsonl(source / "validation.jsonl")
        validate_rows(stage, train_rows + validation_rows, config["schema"])

        records = DATASET_ROOT / stage / "records"
        records.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / "train.jsonl", records / "train.jsonl")
        shutil.copyfile(source / "validation.jsonl", records / "validation.jsonl")
        manifest["stages"][stage] = {
            "epochs": config["epochs"],
            "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
            "source_train": str((source / "train.jsonl").relative_to(ROOT)),
            "source_validation": str((source / "validation.jsonl").relative_to(ROOT)),
        }

    relative_files = (
        "decision/records/train.jsonl",
        "decision/records/validation.jsonl",
        "response/records/train.jsonl",
        "response/records/validation.jsonl",
        "contract/decision_system.txt",
        "contract/decision_schema.json",
        "contract/response_schema.json",
        "contract/response_system_by_route.json",
    )
    for relative in relative_files:
        manifest["sha256"][relative] = sha256_file(DATASET_ROOT / relative)
    write_json(DATASET_ROOT / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
