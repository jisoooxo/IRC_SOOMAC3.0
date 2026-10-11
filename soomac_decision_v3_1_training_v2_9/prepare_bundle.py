#!/usr/bin/env python3
"""Decision v3.1 후보 데이터와 현재 runtime contract를 Docker context에 고정한다."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "src" / "soomac_irc"
sys.path.insert(0, str(PACKAGE_ROOT))

from soomac_irc.agent_contract import DECISION_SCHEMA  # noqa: E402
from soomac_irc.agent_prompts import DECISION_SYSTEM  # noqa: E402


DOWNLOADS = Path(os.environ.get("V31_SOURCE_DIR", "/home/roma/Downloads"))
SOURCE_TRAIN = DOWNLOADS / "decision_v3_1_train.jsonl"
SOURCE_ZIP = DOWNLOADS / "decision_v3_1_candidate_dataset.zip"
SOURCE_VALIDATION = ROOT / "data_natural_multiturn_v3_rebuild/decision_v3/validation.jsonl"
DATASET = Path(__file__).resolve().parent / "docker_context/assets/dataset"

EXPECTED_TRAIN_SHA256 = "d011cff5dc0460ec55bfd4c7b458fcc88b184064dab69ae0e3e4a963b7f895c4"
EXPECTED_ZIP_SHA256 = "bc31fe6e253a4c6beaac0fbfb3f0c3bc0c9e107622b3373b1754089b44a654b6"
EXPECTED_VALIDATION_SHA256 = "c87ab3908b34c22d8f765a5771187f6557fa98268521ab49841330157bb321df"
INPUT_FIELDS = {
    "recent_history",
    "order",
    "preferences",
    "pending",
    "robot_state",
    "message",
}
PROVENANCE_MEMBERS = {
    "decision_v3_1_candidate_dataset/supplement_train.jsonl": "supplement_train.jsonl",
    "decision_v3_1_candidate_dataset/supplement_provenance.jsonl": "supplement_provenance.jsonl",
    "decision_v3_1_candidate_dataset/supplement_design.md": "supplement_design.md",
    "decision_v3_1_candidate_dataset/validation_report.json": "validation_report.json",
    "decision_v3_1_candidate_dataset/source_manifest.json": "source_manifest.json",
    "decision_v3_1_candidate_dataset/README.md": "README.md",
    "decision_v3_1_candidate_dataset/SHA256SUMS.txt": "SHA256SUMS.txt",
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
    expected = (
        (SOURCE_TRAIN, EXPECTED_TRAIN_SHA256),
        (SOURCE_ZIP, EXPECTED_ZIP_SHA256),
        (SOURCE_VALIDATION, EXPECTED_VALIDATION_SHA256),
    )
    for path, expected_hash in expected:
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"source hash mismatch: {path} expected={expected_hash} actual={actual_hash}")

    with zipfile.ZipFile(SOURCE_ZIP) as archive:
        zipped_train = archive.read("decision_v3_1_candidate_dataset/train.jsonl")
    if hashlib.sha256(zipped_train).hexdigest() != EXPECTED_TRAIN_SHA256:
        raise ValueError("ZIP train.jsonl does not match the standalone Decision v3.1 train file")


def extract_provenance() -> None:
    target = DATASET / "provenance"
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(SOURCE_ZIP) as archive:
        for member, filename in PROVENANCE_MEMBERS.items():
            (target / filename).write_bytes(archive.read(member))


def main() -> None:
    verify_sources()
    train_rows = read_jsonl(SOURCE_TRAIN)
    validation_rows = read_jsonl(SOURCE_VALIDATION)
    validate_rows("train", train_rows)
    validate_rows("validation", validation_rows)

    train_ids = {row["id"] for row in train_rows}
    validation_ids = {row["id"] for row in validation_rows}
    train_messages = {normalized_message(row) for row in train_rows}
    validation_messages = {normalized_message(row) for row in validation_rows}
    if train_ids & validation_ids or train_messages & validation_messages:
        raise ValueError("train/validation contamination detected")

    records = DATASET / "decision/records"
    contracts = DATASET / "contract"
    records.mkdir(parents=True, exist_ok=True)
    contracts.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE_TRAIN, records / "train.jsonl")
    shutil.copyfile(SOURCE_VALIDATION, records / "validation.jsonl")
    (contracts / "decision_system.txt").write_text(
        DECISION_SYSTEM.strip() + "\n", encoding="utf-8"
    )
    write_json(contracts / "decision_schema.json", DECISION_SCHEMA)
    extract_provenance()

    manifest = {
        "training_image_spec": "2.9",
        "task": "decision_v3_1_adapter",
        "dataset": "natural_multiturn_v3_plus_targeted_supplement",
        "epochs": 6,
        "learning_rate": 1e-4,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "planned_optimizer_steps": 1134,
        "gradient_accumulation_steps": 16,
        "source": {
            "train": str(SOURCE_TRAIN),
            "candidate_zip": str(SOURCE_ZIP),
            "validation": str(SOURCE_VALIDATION.relative_to(ROOT)),
        },
        "source_sha256": {
            "train": EXPECTED_TRAIN_SHA256,
            "candidate_zip": EXPECTED_ZIP_SHA256,
            "validation": EXPECTED_VALIDATION_SHA256,
        },
        "contamination": {
            "train_validation_id_overlap": 0,
            "train_validation_message_overlap": 0,
        },
        "sha256": {},
    }
    relative_files = (
        "decision/records/train.jsonl",
        "decision/records/validation.jsonl",
        "contract/decision_system.txt",
        "contract/decision_schema.json",
        *(f"provenance/{filename}" for filename in PROVENANCE_MEMBERS.values()),
    )
    for relative in relative_files:
        manifest["sha256"][relative] = sha256_file(DATASET / relative)
    write_json(DATASET / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
