#!/usr/bin/env python3
"""CPU-only cross-version Decision encoding receipt; prints hashes, never text."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


MANIFEST_SHA = "7850948287237c5ace9d1b0f56931419760fbce3f0afa02cea40681acb6ceffa"


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sequence_sha(values):
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def select_longest_receipt(path):
    selected = None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("agent") == "decision" and row.get("split") == "train":
                if selected is None or (row["full_tokens"], row["id"]) > (selected["full_tokens"], selected["id"]):
                    selected = row
    if selected is None or selected["history_trimmed"] or selected["errors"]:
        raise ValueError("valid longest Decision train receipt missing")
    return selected


def select_record(path, row_id):
    selected = None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("id") == row_id:
                if selected is not None:
                    raise ValueError("duplicate selected Decision ID")
                selected = row
    if selected is None:
        raise ValueError("selected Decision record missing")
    return selected


def encode_receipt(release, processor_dir):
    dataset = release / "dataset"
    if file_sha(dataset / "MANIFEST.json") != MANIFEST_SHA:
        raise ValueError("source manifest changed")
    manifest = json.loads((dataset / "MANIFEST.json").read_text())
    if manifest["max_length"]["decision"] != 9216:
        raise ValueError("Decision cap changed")
    receipt = select_longest_receipt(release / "validation/TOKEN_MASK_ROWS.jsonl")
    row = select_record(dataset / "records/decision/train.jsonl", receipt["id"])
    sys.path.insert(0, str(release / "common"))
    import core
    from jsonschema import Draft202012Validator
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(processor_dir, local_files_only=True)
    core.MAX_LENGTH = 9216
    schema = json.loads((dataset / "contract/decision_schema.json").read_text())
    prompts = {"decision": (dataset / "contract/decision_system.txt").read_text()}
    encoded = core.encode_record("decision", row, processor, prompts, Draft202012Validator(schema))
    if encoded["length"] != receipt["full_tokens"]:
        raise ValueError("processor encoding length differs from sealed receipt")
    input_count = sum(label == -100 for label in encoded["labels"])
    if input_count != receipt["input_tokens"] or encoded["length"] - input_count != receipt["target_tokens"]:
        raise ValueError("sealed target-mask boundary changed")
    return {
        "agent": "decision", "split": "train", "id": receipt["id"],
        "full_tokens": encoded["length"], "input_tokens": input_count,
        "target_tokens": encoded["length"] - input_count,
        "input_ids_sha256": sequence_sha(encoded["input_ids"]),
        "labels_sha256": sequence_sha(encoded["labels"]),
        "source_manifest_sha256": MANIFEST_SHA,
        "processor_tokenizer_sha256": file_sha(processor_dir / "tokenizer.json"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--processor", type=Path, required=True)
    args = parser.parse_args(argv)
    # Prevent an inherited GPU environment from changing the CPU-only intent.
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise ValueError("set CUDA_VISIBLE_DEVICES to empty for CPU-only check")
    print(json.dumps(encode_receipt(args.release, args.processor), sort_keys=True))


if __name__ == "__main__":
    main()
