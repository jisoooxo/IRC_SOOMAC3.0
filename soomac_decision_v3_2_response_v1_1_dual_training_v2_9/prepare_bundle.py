#!/usr/bin/env python3
"""Decision v3.2와 Response v1.1을 단일 듀얼 학습 context에 고정한다."""

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

from soomac_irc.agent_contract import DECISION_SCHEMA, RESPONSE_SCHEMA  # noqa: E402
from soomac_irc.agent_prompts import (  # noqa: E402
    DECISION_SYSTEM,
    GENERAL_RESPONSE_SYSTEM,
    MIXED_RESPONSE_SYSTEM,
    TASK_RESPONSE_SYSTEM,
)


DATASET = Path(__file__).resolve().parent / "docker_context/assets/dataset"
STAGES = {
    "decision": {
        "task": "decision_v3_2_adapter",
        "source": ROOT / "data_targeted_upgrade/decision_v3_2_candidate",
        "train": ROOT / "data_targeted_upgrade/decision_v3_2_candidate/train.jsonl",
        "validation": ROOT / "data_natural_multiturn_v3_rebuild/decision_v3/validation.jsonl",
        "train_sha256": "76b2123c0183fcb703163cc5cc0e937d8c97e070c9d0efa3ee417b245e0a5060",
        "validation_sha256": "c87ab3908b34c22d8f765a5771187f6557fa98268521ab49841330157bb321df",
        "train_rows": 3465,
        "validation_rows": 285,
        "epochs": 6,
        "learning_rate": 1e-4,
        "planned_optimizer_steps": 1302,
        "schema": DECISION_SCHEMA,
    },
    "response": {
        "task": "response_v1_1_adapter",
        "source": ROOT / "data_targeted_upgrade/response_v1_1_candidate",
        "train": ROOT / "data_targeted_upgrade/response_v1_1_candidate/train.jsonl",
        "validation": ROOT / "data_natural_multiturn_v3_rebuild/response_v1/validation.jsonl",
        "train_sha256": "e16b007a0a3c78711951e8b061322fb656f4c09be410f361522aba6d1803d558",
        "validation_sha256": "531fefdc8ec1e1eeda8f85edce34ca9c24832b22a6b7e6dacdb1812e12acac6f",
        "train_rows": 1098,
        "validation_rows": 101,
        "epochs": 4,
        "learning_rate": 5e-5,
        "planned_optimizer_steps": 276,
        "schema": RESPONSE_SCHEMA,
    },
}
PROVENANCE_FILES = (
    "supplement_train.jsonl",
    "supplement_provenance.jsonl",
    "supplement_design.md",
    "validation_report.json",
    "manifest.json",
    "semantic_review.jsonl",
)
INPUT_FIELDS = {
    "decision": {
        "recent_history", "order", "preferences", "pending", "robot_state", "message",
    },
    "response": {
        "user_text", "recent_history", "confirmed_order", "preferences", "pending",
        "policy", "applied_this_turn", "future_changes", "recommendation_result",
        "next_prompt", "robot_state", "recent_action_history", "execution_authorized",
        "starting_now",
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


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_rows(stage: str, rows: list[dict], schema: dict) -> None:
    validator = Draft202012Validator(schema)
    ids = set()
    inputs = set()
    for row in rows:
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id or row_id in ids:
            raise ValueError(f"{stage}: missing or duplicate id: {row_id}")
        ids.add(row_id)
        model_input = row.get("input")
        if not isinstance(model_input, dict) or set(model_input) != INPUT_FIELDS[stage]:
            raise ValueError(f"{stage}:{row_id}: input contract mismatch")
        if stage == "response" and row.get("route") not in {"task", "general", "mixed"}:
            raise ValueError(f"response:{row_id}: invalid route")
        validator.validate(row.get("target"))
        fingerprint_source = (
            model_input
            if stage == "decision"
            else {"route": row["route"], "input": model_input, "target": row["target"]}
        )
        fingerprint = json.dumps(
            fingerprint_source,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if fingerprint in inputs:
            raise ValueError(f"{stage}:{row_id}: duplicate input")
        inputs.add(fingerprint)


def main() -> None:
    if DATASET.exists():
        shutil.rmtree(DATASET)

    contracts = DATASET / "contract"
    contracts.mkdir(parents=True)
    (contracts / "decision_system.txt").write_text(DECISION_SYSTEM.strip() + "\n", encoding="utf-8")
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
        "task": "decision_v3_2_response_v1_1_dual_adapter",
        "stage_order": ["decision", "response"],
        "stages": {},
        "total_optimizer_steps": 1578,
        "sha256": {},
    }
    for stage, config in STAGES.items():
        for source_path, expected_hash in (
            (config["train"], config["train_sha256"]),
            (config["validation"], config["validation_sha256"]),
        ):
            if sha256_file(source_path) != expected_hash:
                raise ValueError(f"{stage}: source hash mismatch: {source_path}")

        train_rows = read_jsonl(config["train"])
        validation_rows = read_jsonl(config["validation"])
        if len(train_rows) != config["train_rows"] or len(validation_rows) != config["validation_rows"]:
            raise ValueError(f"{stage}: row count mismatch")
        validate_rows(stage, train_rows, config["schema"])
        validate_rows(stage, validation_rows, config["schema"])
        if {row["id"] for row in train_rows} & {row["id"] for row in validation_rows}:
            raise ValueError(f"{stage}: train/validation id contamination")

        records = DATASET / stage / "records"
        provenance = DATASET / "provenance" / stage
        records.mkdir(parents=True)
        provenance.mkdir(parents=True)
        shutil.copyfile(config["train"], records / "train.jsonl")
        shutil.copyfile(config["validation"], records / "validation.jsonl")
        for name in PROVENANCE_FILES:
            shutil.copyfile(config["source"] / name, provenance / name)

        manifest["stages"][stage] = {
            "task": config["task"],
            "epochs": config["epochs"],
            "learning_rate": config["learning_rate"],
            "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
            "planned_optimizer_steps": config["planned_optimizer_steps"],
            "source_train": str(config["train"].relative_to(ROOT)),
            "source_validation": str(config["validation"].relative_to(ROOT)),
            "source_train_sha256": config["train_sha256"],
            "source_validation_sha256": config["validation_sha256"],
        }

    # 이미 실제 tokenizer로 검증한 두 보고서를 그대로 포함한다.
    for stage in STAGES:
        bundle = "decision_v3_2" if stage == "decision" else "response_v1_1"
        source = ROOT / f"soomac_{bundle}_training_v2_9/docker_context/assets/dataset/token_length_report.json"
        shutil.copyfile(source, DATASET / f"{stage}_token_length_report.json")

    tracked_files = sorted(path for path in DATASET.rglob("*") if path.is_file())
    manifest["sha256"] = {
        path.relative_to(DATASET).as_posix(): sha256_file(path) for path in tracked_files
    }
    write_json(DATASET / "manifest.json", manifest)
    checksum_files = sorted(path for path in DATASET.rglob("*") if path.is_file())
    (DATASET / "SHA256SUMS").write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.relative_to(DATASET).as_posix()}"
            for path in checksum_files
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
