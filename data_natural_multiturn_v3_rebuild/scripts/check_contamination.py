#!/usr/bin/env python3
"""split 간 exact/paraphrase/semantic leakage를 provenance 기준으로 검사한다."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data_natural_multiturn_v3_rebuild"
REPORT = BASE / "reports/contamination.json"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalize(value: str) -> str:
    return re.sub(r"[\s\W_]+", "", value.lower())


def audit(name: str) -> dict:
    rows = read_jsonl(BASE / f"intermediate/{name}_intermediate.jsonl")
    by_id = {row["id"]: row for row in rows}
    leaks = []
    for field in ("paraphrase_group", "semantic_id"):
        groups = defaultdict(lambda: defaultdict(list))
        for row in rows:
            value = str(row.get("metadata", {}).get(field, ""))
            if value:
                groups[value][row["split"]].append(row["id"])
        for value, split_map in groups.items():
            if len(split_map) > 1:
                leaks.append({"kind": field, "value": value, "splits": {key: values[:5] for key, values in split_map.items()}})
    messages = defaultdict(lambda: defaultdict(list))
    signatures = defaultdict(lambda: defaultdict(list))
    for row in rows:
        text = row["input"].get("message", row["input"].get("user_text", ""))
        if text:
            messages[normalize(text)][row["split"]].append(row["id"])
        signature = json.dumps({"input": row["input"], "target": row["target"]}, ensure_ascii=False, sort_keys=True)
        signatures[signature][row["split"]].append(row["id"])
    for value, split_map in messages.items():
        if len(split_map) > 1:
            leaks.append({"kind": "exact_message", "value": value, "splits": {key: values[:5] for key, values in split_map.items()}})
    for value, split_map in signatures.items():
        if sum(len(values) for values in split_map.values()) > 1:
            leaks.append({"kind": "exact_row", "value": value[:200], "splits": {key: values[:5] for key, values in split_map.items()}})
    train_ids = {row["id"] for row in rows if row["split"] == "train"}
    eval_ids = {row["id"] for row in rows if row["split"] in {"challenge", "holdout"}}
    return {
        "status": "PASS" if not leaks else "FAIL",
        "rows": len(rows),
        "train": len(train_ids),
        "evaluation": len(eval_ids),
        "leak_count": len(leaks),
        "leaks": leaks[:100],
        "orphan_ids": len(set(by_id) - train_ids - eval_ids - {row["id"] for row in rows if row["split"] in {"validation", "test"}}),
    }


def main() -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    report = {"decision_v3": audit("decision_v3"), "response_v1": audit("response_v1")}
    report["status"] = "PASS" if all(value["status"] == "PASS" for key, value in report.items() if key != "status") else "FAIL"
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
