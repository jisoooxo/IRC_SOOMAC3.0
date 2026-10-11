#!/usr/bin/env python3
"""Token audit를 dataset manifest와 Docker context checksum에 고정한다."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
CONTEXT = HERE / "docker_context"
DATASET = CONTEXT / "assets/dataset"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    report_path = DATASET / "token_length_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("task") != "response_v1_1_adapter":
        raise ValueError("token audit task mismatch")
    for split, metrics in report["splits"].items():
        if metrics["rows_over_max_after_trim"] != 0:
            raise ValueError(f"{split}: token truncation remains")

    manifest_path = DATASET / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["token_length_audit"] = {
        "max_length": report["max_length"],
        "report": "token_length_report.json",
        "sha256": sha256_file(report_path),
        "rows_over_max_after_trim": 0,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    dataset_files = sorted(
        path for path in DATASET.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    (DATASET / "SHA256SUMS").write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.relative_to(DATASET).as_posix()}" for path in dataset_files
        )
        + "\n",
        encoding="utf-8",
    )

    context_files = sorted(
        path
        for path in CONTEXT.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS" and "__pycache__" not in path.parts
    )
    (CONTEXT / "SHA256SUMS").write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.relative_to(CONTEXT).as_posix()}" for path in context_files
        )
        + "\n",
        encoding="utf-8",
    )
    print("PASS: Response v1.1 token audit and SHA256SUMS finalized")


if __name__ == "__main__":
    main()
