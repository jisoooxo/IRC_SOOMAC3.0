#!/usr/bin/env python3
"""Decision v3.2 후보와 현재 runtime contract를 Docker context에 고정한다."""

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

from soomac_irc.agent_contract import DECISION_SCHEMA  # noqa: E402
from soomac_irc.agent_prompts import DECISION_SYSTEM  # noqa: E402


SOURCE = ROOT / "data_targeted_upgrade/decision_v3_2_candidate"
SOURCE_TRAIN = SOURCE / "train.jsonl"
SOURCE_VALIDATION = ROOT / "data_natural_multiturn_v3_rebuild/decision_v3/validation.jsonl"
DATASET = Path(__file__).resolve().parent / "docker_context/assets/dataset"

EXPECTED_TRAIN_SHA256 = "76b2123c0183fcb703163cc5cc0e937d8c97e070c9d0efa3ee417b245e0a5060"
EXPECTED_VALIDATION_SHA256 = "c87ab3908b34c22d8f765a5771187f6557fa98268521ab49841330157bb321df"
EXPECTED_ROWS = {"train": 3465, "validation": 285}
PLANNED_OPTIMIZER_STEPS = 1302
INPUT_FIELDS = {
    "recent_history",
    "order",
    "preferences",
    "pending",
    "robot_state",
    "message",
}
PROVENANCE_FILES = (
    "supplement_train.jsonl",
    "supplement_provenance.jsonl",
    "supplement_design.md",
    "validation_report.json",
    "manifest.json",
    "semantic_review.jsonl",
)


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


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_message(row: dict) -> str:
    return " ".join(row["input"]["message"].lower().split())


def validate_rows(name: str, rows: list[dict]) -> None:
    validator = Draft202012Validator(DECISION_SCHEMA)
    ids = set()
    inputs = set()
    for row in rows:
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id or row_id in ids:
            raise ValueError(f"{name}: missing or duplicate id: {row_id}")
        ids.add(row_id)
        model_input = row.get("input")
        if not isinstance(model_input, dict) or set(model_input) != INPUT_FIELDS:
            raise ValueError(f"{name}:{row_id}: Decision input contract mismatch")
        validator.validate(row.get("target"))
        fingerprint = json.dumps(model_input, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if fingerprint in inputs:
            raise ValueError(f"{name}:{row_id}: duplicate input")
        inputs.add(fingerprint)


def verify_sources() -> None:
    for path in (SOURCE_TRAIN, SOURCE_VALIDATION, *(SOURCE / name for name in PROVENANCE_FILES)):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    actual_train = sha256_file(SOURCE_TRAIN)
    actual_validation = sha256_file(SOURCE_VALIDATION)
    if actual_train != EXPECTED_TRAIN_SHA256:
        raise ValueError(f"train hash mismatch: {actual_train}")
    if actual_validation != EXPECTED_VALIDATION_SHA256:
        raise ValueError(f"validation hash mismatch: {actual_validation}")
    validation_report = json.loads((SOURCE / "validation_report.json").read_text(encoding="utf-8"))
    for key in (
        "schema_invalid",
        "legacy_field",
        "duplicate",
        "frozen_overlap",
        "malformed_korean",
        "semantic_suspicious",
    ):
        if validation_report.get(key) != 0:
            raise ValueError(f"candidate validation failed: {key}={validation_report.get(key)}")


def write_sha256s(paths: list[Path]) -> None:
    lines = [f"{sha256_file(path)}  {path.relative_to(DATASET).as_posix()}" for path in paths]
    (DATASET / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    verify_sources()
    train_rows = read_jsonl(SOURCE_TRAIN)
    validation_rows = read_jsonl(SOURCE_VALIDATION)
    if {"train": len(train_rows), "validation": len(validation_rows)} != EXPECTED_ROWS:
        raise ValueError("dataset row count mismatch")
    validate_rows("train", train_rows)
    validate_rows("validation", validation_rows)

    train_ids = {row["id"] for row in train_rows}
    validation_ids = {row["id"] for row in validation_rows}
    train_messages = {normalized_message(row) for row in train_rows}
    validation_messages = {normalized_message(row) for row in validation_rows}
    if train_ids & validation_ids or train_messages & validation_messages:
        raise ValueError("train/validation contamination detected")

    if DATASET.exists():
        shutil.rmtree(DATASET)
    records = DATASET / "decision/records"
    contracts = DATASET / "contract"
    provenance = DATASET / "provenance"
    records.mkdir(parents=True)
    contracts.mkdir(parents=True)
    provenance.mkdir(parents=True)
    shutil.copyfile(SOURCE_TRAIN, records / "train.jsonl")
    shutil.copyfile(SOURCE_VALIDATION, records / "validation.jsonl")
    for name in PROVENANCE_FILES:
        shutil.copyfile(SOURCE / name, provenance / name)
    (contracts / "decision_system.txt").write_text(DECISION_SYSTEM.strip() + "\n", encoding="utf-8")
    write_json(contracts / "decision_schema.json", DECISION_SCHEMA)

    relative_files = (
        "decision/records/train.jsonl",
        "decision/records/validation.jsonl",
        "contract/decision_system.txt",
        "contract/decision_schema.json",
        *(f"provenance/{name}" for name in PROVENANCE_FILES),
    )
    files = [DATASET / relative for relative in relative_files]
    manifest = {
        "training_image_spec": "2.9",
        "task": "decision_v3_2_adapter",
        "dataset": "decision_v3_1_base_plus_targeted_v3_2_supplement",
        "epochs": 6,
        "learning_rate": 1e-4,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "planned_optimizer_steps": PLANNED_OPTIMIZER_STEPS,
        "gradient_accumulation_steps": 16,
        "source": {
            "train": str(SOURCE_TRAIN.relative_to(ROOT)),
            "validation": str(SOURCE_VALIDATION.relative_to(ROOT)),
        },
        "source_sha256": {
            "train": EXPECTED_TRAIN_SHA256,
            "validation": EXPECTED_VALIDATION_SHA256,
        },
        "contamination": {
            "train_validation_id_overlap": 0,
            "train_validation_message_overlap": 0,
        },
        "sha256": {path.relative_to(DATASET).as_posix(): sha256_file(path) for path in files},
    }
    write_json(DATASET / "manifest.json", manifest)
    write_sha256s(files + [DATASET / "manifest.json"])
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
