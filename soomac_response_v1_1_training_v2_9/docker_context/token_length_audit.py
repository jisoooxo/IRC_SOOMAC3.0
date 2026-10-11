#!/usr/bin/env python3
"""실제 Gemma chat template 기준 Response v1.1 token 길이를 계산한다."""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import statistics

from transformers import AutoProcessor


DATASET_ROOT = Path(os.environ.get("DATASET_DIR", "/app/assets/dataset"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/model"))
MAX_LENGTH = 4096
ASSISTANT_END = "<turn|>\n"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def percentile(values: list[int], probability: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * probability) - 1)]


def stats(values: list[int]) -> dict:
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 3),
        "median": statistics.median(values),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def prompt_length(processor, system_prompt: str, model_input: dict) -> int:
    encoded = processor.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(model_input, ensure_ascii=False, separators=(",", ":"))},
        ],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    return int(encoded["input_ids"].shape[1])


def audit_split(processor, systems: dict, path: Path) -> dict:
    input_lengths = []
    output_lengths = []
    raw_totals = []
    fitted_totals = []
    context_trimmed = []
    irreducible = []
    for row in read_jsonl(path):
        output_length = len(
            processor.tokenizer(
                json.dumps(row["target"], ensure_ascii=False, separators=(",", ":")) + ASSISTANT_END,
                add_special_tokens=False,
            )["input_ids"]
        )
        working_input = copy.deepcopy(row["input"])
        system_prompt = systems[row["route"]]
        raw_input_length = prompt_length(processor, system_prompt, working_input)
        fitted_input_length = raw_input_length
        while fitted_input_length + output_length > MAX_LENGTH:
            history = working_input.get("recent_history")
            action_history = working_input.get("recent_action_history")
            if isinstance(history, list) and history:
                del history[:2]
            elif isinstance(action_history, list) and action_history:
                del action_history[0]
            else:
                irreducible.append(row["id"])
                break
            fitted_input_length = prompt_length(processor, system_prompt, working_input)
        if fitted_input_length != raw_input_length:
            context_trimmed.append(row["id"])
        input_lengths.append(raw_input_length)
        output_lengths.append(output_length)
        raw_totals.append(raw_input_length + output_length)
        fitted_totals.append(fitted_input_length + output_length)
    return {
        "input_tokens": stats(input_lengths),
        "output_tokens": stats(output_lengths),
        "total_tokens_raw": stats(raw_totals),
        "total_tokens_after_training_trim": stats(fitted_totals),
        "rows_over_max_before_trim": sum(value > MAX_LENGTH for value in raw_totals),
        "rows_requiring_history_trim": len(context_trimmed),
        "rows_over_max_after_trim": len(irreducible),
        "truncation_rate_after_trim": round(len(irreducible) / len(raw_totals), 8),
        "trimmed_row_ids": context_trimmed,
        "irreducible_row_ids": irreducible,
    }


def main() -> None:
    processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
    systems = json.loads(
        (DATASET_ROOT / "contract/response_system_by_route.json").read_text(encoding="utf-8")
    )
    split_reports = {
        split: audit_split(
            processor,
            systems,
            DATASET_ROOT / f"response/records/{split}.jsonl",
        )
        for split in ("train", "validation")
    }
    report = {
        "task": "response_v1_1_adapter",
        "model": str(MODEL_DIR),
        "max_length": MAX_LENGTH,
        "chat_template": "AutoProcessor.apply_chat_template(enable_thinking=False)",
        "splits": split_reports,
    }
    output = DATASET_ROOT / "token_length_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
