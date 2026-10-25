#!/usr/bin/env python3
"""Build a model-free evidence bundle for Decision Adapter v2.2.5 review."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import shutil
import subprocess


WORKSPACE = Path("/home/roma/IRC_SOOMAC3.0")
BUNDLE_NAME = "decision_adapter_v225_root_cause_review_20260930"
BUNDLE = WORKSPACE / BUNDLE_NAME
ZIP_PATH = WORKSPACE / f"{BUNDLE_NAME}.zip"

IRC = WORKSPACE / "src/soomac_irc/soomac_irc"
DATASET = Path("/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/dataset/soomac_decision_adapter_v2_2_5_semantic_rebalance")
CHECKPOINT = Path("/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/checkpoints/gemma4_decision_nf4_qlora_v2_2_5_semantic_rebalance")
EVALUATION = Path("/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/evaluation")
TRAINER = Path("/home/roma/Downloads/SOOMAC_DECISION_DOCKER_V2_7_REBUILT/docker_context/train_in_container.py")


inventory: list[dict] = []


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_original(source: Path, relative: str, role: str) -> Path:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = BUNDLE / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_hash = sha256(source)
    copied_hash = sha256(destination)
    if source_hash != copied_hash:
        raise RuntimeError(f"copy hash mismatch: {source}")
    inventory.append(
        {
            "bundle_path": relative,
            "original_absolute_path": str(source),
            "role": role,
            "bytes": source.stat().st_size,
            "sha256": source_hash,
        }
    )
    return destination


def write_generated(relative: str, content: str, role: str) -> Path:
    destination = BUNDLE / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    inventory.append(
        {
            "bundle_path": relative,
            "original_absolute_path": "N/A (analysis-generated)",
            "role": role,
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
        }
    )
    return destination


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def semantic_atoms(target: dict) -> set[tuple]:
    atoms: set[tuple] = set()
    order = target.get("order", {})
    for key, value in order.items():
        if key == "toppings":
            atoms.update(("order.topping", k, str(v)) for k, v in value.items())
        else:
            atoms.add((f"order.{key}", "", str(value)))
    for restriction in target.get("restrictions", []):
        atoms.add(("restriction", restriction.get("target"), restriction.get("reason"), restriction.get("action")))
    for preference in target.get("preferences", []):
        atoms.add(("preference", preference.get("value"), preference.get("action")))
    recommendation = target.get("recommendation")
    if recommendation:
        atoms.add(("recommendation.action", recommendation.get("action")))
        if "scope" in recommendation:
            atoms.add(("recommendation.scope", recommendation.get("scope")))
        if "criteria" in recommendation:
            atoms.add(("recommendation.criteria", recommendation.get("criteria")))
    for key in ("commit", "confirmation", "clarify"):
        if key in target:
            atoms.add((key, str(target[key])))
    for query in target.get("queries", []):
        atoms.add(("query", query.get("type"), query.get("target")))
    return atoms


def dataset_summary(path: Path) -> dict:
    rows = read_jsonl(path)
    families = Counter(row.get("scenario_family", "unknown") for row in rows)
    combinations = Counter("+".join(row.get("target", {}).keys()) or "empty" for row in rows)
    atom_counts = Counter(len(semantic_atoms(row.get("target", {}))) for row in rows)
    clarify = sum(row.get("target", {}).get("clarify") is True for row in rows)
    schema_order = ["order", "restrictions", "preferences", "recommendation", "commit", "confirmation", "queries", "clarify"]
    order_index = {key: index for index, key in enumerate(schema_order)}
    key_order_violations = []
    for row in rows:
        keys = list(row.get("target", {}).keys())
        positions = [order_index[key] for key in keys if key in order_index]
        if positions != sorted(positions):
            key_order_violations.append({"id": row.get("id"), "family": row.get("scenario_family"), "keys": keys})

    by_input: defaultdict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = json.dumps(
            {"history": row.get("history", []), "input": row.get("input", {})},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        by_input[key].append(row)
    duplicate_groups = []
    conflict_groups = []
    for group in by_input.values():
        if len(group) < 2:
            continue
        targets = {json.dumps(row.get("target", {}), ensure_ascii=False, sort_keys=True) for row in group}
        record = {"ids": [row.get("id") for row in group], "distinct_targets": len(targets)}
        duplicate_groups.append(record)
        if len(targets) > 1:
            conflict_groups.append(record)

    return {
        "source": str(path),
        "rows": len(rows),
        "family_counts": dict(sorted(families.items())),
        "target_top_level_key_combinations": dict(combinations.most_common()),
        "semantic_atom_count_distribution": {str(k): v for k, v in sorted(atom_counts.items())},
        "single_atom_rows": atom_counts.get(1, 0),
        "two_or_more_atom_rows": sum(v for k, v in atom_counts.items() if k >= 2),
        "clarify_rows": clarify,
        "schema_top_level_key_order_violation_count": len(key_order_violations),
        "schema_top_level_key_order_violations": key_order_violations,
        "exact_full_input_duplicate_group_count": len(duplicate_groups),
        "exact_full_input_conflict_group_count": len(conflict_groups),
        "exact_full_input_duplicate_groups": duplicate_groups,
        "exact_full_input_conflict_groups": conflict_groups,
    }


def prediction_summary(path: Path) -> dict:
    rows = read_jsonl(path)
    families: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        family = row.get("family", row.get("scenario_family", "unknown"))
        families[family][1] += 1
        families[family][0] += int(bool(row.get("ok")))
    correct = sum(bool(row.get("ok")) for row in rows)
    return {
        "source": str(path),
        "total": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else 0.0,
        "family": {
            key: {"correct": value[0], "total": value[1], "accuracy": value[0] / value[1]}
            for key, value in sorted(families.items())
        },
        "failed_ids": [row.get("id") for row in rows if not row.get("ok")],
    }


def main() -> None:
    if BUNDLE.exists() or ZIP_PATH.exists():
        raise FileExistsError(f"refusing to overwrite existing bundle: {BUNDLE} or {ZIP_PATH}")
    BUNDLE.mkdir(parents=True)

    operational_files = [
        "__init__.py",
        "decision_model.py",
        "agent_contract.py",
        "agent_prompts.py",
        "domain.py",
        "llm_policy.py",
        "llm_langgraph.py",
        "llm_langgraph_node.py",
        "model_runtime.py",
    ]
    for name in operational_files:
        copy_original(IRC / name, f"operational_code/{name}", "current IRC_SOOMAC3.0 operational source")

    copy_original(DATASET / "scripts/eval_decision_adapter.py", "evaluation_code/eval_decision_adapter.py", "actual evaluation source")
    copy_original(DATASET / "scripts/eval_semantic_metrics.py", "evaluation_code/eval_semantic_metrics.py", "actual semantic metric source")
    copy_original(TRAINER, "training/train_in_container.py", "actual Docker trainer source")

    training_artifacts = [
        "adapter_config.json",
        "run_manifest.json",
        "training_complete.json",
        "all_results.json",
        "train_results.json",
    ]
    for name in training_artifacts:
        copy_original(CHECKPOINT / name, f"training/results/{name}", "actual v2.2.5 training result/config")
    copy_original(CHECKPOINT / "checkpoint-396/trainer_state.json", "training/results/checkpoint-396_trainer_state.json", "actual Trainer state; records best checkpoint and eval losses")

    contract_files = [
        (DATASET / "contract/decision_system.txt", "dataset_contract/decision_system.txt"),
        (DATASET / "contract/decision_schema.json", "dataset_contract/decision_schema.json"),
        (DATASET / "manifest.json", "dataset_contract/manifest.json"),
        (DATASET / "records/decision_train.jsonl", "dataset_contract/decision_train.jsonl"),
        (DATASET / "records/decision_validation.jsonl", "dataset_contract/decision_validation.jsonl"),
    ]
    for source, relative in contract_files:
        copy_original(source, relative, "actual v2.2.5 dataset/contract used by trainer")

    eval_data = {
        "latest_contract": "latest_contract_eval/latest_contract_eval_sft.jsonl",
        "hard_v221": "hard_eval_v221/hard_eval_v221_sft.jsonl",
        "hard_v222": "hard_eval_v222/hard_eval_v222_sft.jsonl",
        "hard_v223": "hard_eval_v223/hard_eval_v223_sft.jsonl",
        "hard_v224": "hard_eval_v224/hard_eval_v224_sft.jsonl",
        "hard_v225": "hard_eval_v225/hard_eval_v225_sft.jsonl",
        "challenge": "sft/decision_challenge_sft.jsonl",
        "runtime_holdout": "runtime_holdout/real_surface_current_schema_sft.jsonl",
    }
    for suite, relative in eval_data.items():
        copy_original(DATASET / relative, f"evaluation_data/{suite}.jsonl", "actual evaluation input")

    original_results = {
        "latest_contract": ("v225_latest_contract_predictions.jsonl", "v225_latest_contract.log", "v225_latest_contract_semantic.log"),
        "hard_v221": ("v225_hard_v221_predictions.jsonl", "v225_hard_v221.log", "v225_hard_v221_semantic.log"),
        "hard_v222": ("v225_hard_v222_predictions.jsonl", "v225_hard_v222.log", "v225_hard_v222_semantic.log"),
        "hard_v223": ("v225_hard_v223_predictions.jsonl", "v225_hard_v223.log", "v225_hard_v223_semantic.log"),
        "hard_v224": ("v225_hard_v224_predictions.jsonl", "v225_hard_v224.log", "v225_hard_v224_semantic.log"),
        "hard_v225": ("v225_hard_eval_v225_predictions.jsonl", "v225_hard_eval_v225.log", None),
        "challenge": ("v225_challenge_predictions.jsonl", "v225_challenge.log", "v225_challenge_semantic.log"),
        "runtime_holdout": ("v225_runtime_holdout_predictions.jsonl", "v225_runtime_holdout.log", "v225_runtime_holdout_semantic.log"),
    }
    for suite, names in original_results.items():
        prediction_name, log_name, semantic_name = names
        copied_prediction = copy_original(EVALUATION / prediction_name, f"results_any_order_false/{suite}/predictions.jsonl", "actual any_order=False prediction output")
        copy_original(EVALUATION / log_name, f"results_any_order_false/{suite}/generation.log", "actual any_order=False evaluation log")
        if semantic_name:
            copy_original(EVALUATION / semantic_name, f"results_any_order_false/{suite}/semantic_metrics.log", "actual semantic metric log")
        write_generated(
            f"results_any_order_false/{suite}/summary.json",
            json.dumps(prediction_summary(copied_prediction), ensure_ascii=False, indent=2) + "\n",
            "analysis-generated summary of actual any_order=False predictions",
        )

    temporary_results = {
        "latest_contract": "/tmp/v225_anyorder_latest_contract_eval_sft.jsonl",
        "hard_v221": "/tmp/v225_anyorder_hard_eval_v221_sft.jsonl",
        "hard_v222": "/tmp/v225_anyorder_hard_eval_v222_sft.jsonl",
        "hard_v223": "/tmp/v225_anyorder_hard_eval_v223_sft.jsonl",
        "hard_v224": "/tmp/v225_anyorder_hard_eval_v224_sft.jsonl",
        "hard_v225": "/tmp/v225_anyorder_hard225.jsonl",
        "challenge": "/tmp/v225_anyorder_challenge.jsonl",
        "runtime_holdout": "/tmp/v225_anyorder_real_surface_current_schema_sft.jsonl",
    }
    temporary_copies = []
    for suite, source_string in temporary_results.items():
        copied_prediction = copy_original(Path(source_string), f"temporary_any_order_eval/{suite}/predictions.jsonl", "temporary any_order=True prediction output; not operational")
        temporary_copies.append(copied_prediction)
        write_generated(
            f"temporary_any_order_eval/{suite}/summary.json",
            json.dumps(prediction_summary(copied_prediction), ensure_ascii=False, indent=2) + "\n",
            "analysis-generated summary of temporary any_order=True predictions",
        )
    copy_original(Path("/tmp/eval_anyorder_runner.py"), "temporary_any_order_eval/eval_anyorder_runner.py", "temporary in-memory evaluator wrapper; not operational")

    failures = []
    for path in temporary_copies:
        failures.extend(row for row in read_jsonl(path) if not row.get("ok"))
    failures.sort(key=lambda row: str(row.get("id")))
    write_generated(
        "temporary_any_order_eval/failed_9_cases.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in failures),
        "analysis-generated collection of the nine remaining any_order=True failures",
    )

    distribution = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "train": dataset_summary(DATASET / "records/decision_train.jsonl"),
        "validation": dataset_summary(DATASET / "records/decision_validation.jsonl"),
    }
    write_generated(
        "dataset_contract/family_distribution_and_audit_summary.json",
        json.dumps(distribution, ensure_ascii=False, indent=2) + "\n",
        "analysis-generated exhaustive train/validation family and structural audit",
    )

    reproduction = '''#!/usr/bin/env python3
"""Reproduce XGrammar JSON property-order behavior without loading model weights."""
import argparse
import importlib.metadata
import inspect
import json
from pathlib import Path

import xgrammar as xgr
from transformers import AutoProcessor

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", required=True, help="Local Gemma4 base-model directory; weights are not included in this bundle")
parser.add_argument("--schema", default=str(Path(__file__).resolve().parents[1] / "dataset_contract/decision_schema.json"))
args = parser.parse_args()

processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
tokenizer = processor.tokenizer
stop_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<turn|>")]
stop_ids = [value for value in dict.fromkeys(stop_ids) if isinstance(value, int) and value >= 0]
info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=len(tokenizer), stop_token_ids=stop_ids)
compiler = xgr.GrammarCompiler(info)
schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
sample = '{"confirmation":"reject","order":{"toppings":{"소시지":"high"}}}'

print("xgrammar_version=", importlib.metadata.version("xgrammar"))
print("compile_signature=", inspect.signature(xgr.GrammarCompiler.compile_json_schema))
print("sample=", sample)
for any_order in (False, True):
    grammar = compiler.compile_json_schema(schema, any_order=any_order)
    matcher = xgr.GrammarMatcher(grammar)
    accepted = matcher.accept_string(sample)
    print(f"any_order={any_order} accepted={accepted} completed={matcher.is_completed()}")
'''
    write_generated("xgrammar_verification/reproduce_any_order.py", reproduction, "analysis-generated minimal XGrammar reproducer")

    import xgrammar as xgr
    version_text = (
        f"xgrammar_version={importlib.metadata.version('xgrammar')}\n"
        f"compile_json_schema_signature={inspect.signature(xgr.GrammarCompiler.compile_json_schema)}\n"
    )
    write_generated("xgrammar_verification/xgrammar_version_and_signature.txt", version_text, "analysis-generated environment verification")

    callsites = []
    for path in (IRC / "decision_model.py", DATASET / "scripts/eval_decision_adapter.py"):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "compile_json_schema" in line:
                callsites.append(f"{path}:{line_number}:{line.strip()}")
    write_generated("xgrammar_verification/compile_json_schema_callsites.txt", "\n".join(callsites) + "\n", "analysis-generated callsite index")

    readme_head = f'''# Decision Adapter v2.2.5 root-cause review bundle

## Scope and safety

- Operational reference: `/home/roma/IRC_SOOMAC3.0`
- Actual trainer: `{TRAINER}`
- This bundle contains no base-model weights and no LoRA `adapter_model.safetensors`.
- Build/install trees and large training logs are excluded.
- All copied originals were copied byte-for-byte and verified with SHA-256.
- **The actual operational source was not modified.** `git status --short` was empty before packaging.

## any_order=True temporary experiment

The original evaluator contains:

```python
compile_json_schema(schema)
```

XGrammar 0.2.5 defaults this to `any_order=False`. The temporary wrapper in
`temporary_any_order_eval/eval_anyorder_runner.py` loaded the evaluator source in memory and replaced exactly one call with:

```python
compile_json_schema(schema, any_order=True)
```

It did not write the change back to the evaluator or to IRC_SOOMAC3.0. Temporary predictions are isolated under `temporary_any_order_eval/`.

## Result summary

| Scope | any_order=False | temporary any_order=True |
|---|---:|---:|
| Existing 281 | 244/281 | 273/281 |
| New hard v225 96 | 74/96 | 95/96 |
| Challenge 64 | 42/64 | 64/64 |
| Runtime holdout 45 | 41/45 | 44/45 |
| All 377 | 318/377 | 368/377 |

## Remaining nine errors with any_order=True

| ID | Family | Expected | Predicted |
|---|---|---|---|
'''
    failure_lines = []
    for row in failures:
        expected = json.dumps(row.get("expected"), ensure_ascii=False, separators=(",", ":"))
        predicted = json.dumps(row.get("predicted"), ensure_ascii=False, separators=(",", ":"))
        failure_lines.append(f"| `{row.get('id')}` | `{row.get('family')}` | `{expected}` | `{predicted}` |")

    readme_middle = '''

The same records are available as JSONL at `temporary_any_order_eval/failed_9_cases.jsonl`.

## File roles

- `actual ...`: file was used directly by the trainer, evaluator, or current IRC runtime.
- `temporary ...`: output or wrapper created only for the any_order=True investigation.
- `analysis-generated ...`: inventory, summaries, README, or reproducer created for review.
- Every copied file's original absolute path is listed below and in `FILE_ORIGINS.tsv`.

## Complete origin map

| Bundle path | Original absolute path | Role |
|---|---|---|
'''

    origin_lines = []
    for item in sorted(inventory, key=lambda value: value["bundle_path"]):
        origin_lines.append(
            f"| `{item['bundle_path']}` | `{item['original_absolute_path']}` | {item['role']} |"
        )
    readme = readme_head + "\n".join(failure_lines) + readme_middle + "\n".join(origin_lines) + "\n"
    write_generated("README.md", readme, "analysis-generated bundle guide")

    tsv_lines = ["bundle_path\toriginal_absolute_path\trole\tbytes\tsha256"]
    for item in sorted(inventory, key=lambda value: value["bundle_path"]):
        tsv_lines.append(
            "\t".join(str(item[key]) for key in ("bundle_path", "original_absolute_path", "role", "bytes", "sha256"))
        )
    write_generated("FILE_ORIGINS.tsv", "\n".join(tsv_lines) + "\n", "analysis-generated full origin and checksum inventory")

    checksum_lines = []
    for path in sorted(BUNDLE.rglob("*")):
        if path.is_file():
            checksum_lines.append(f"{sha256(path)}  {path.relative_to(BUNDLE)}")
    write_generated("SHA256SUMS.txt", "\n".join(checksum_lines) + "\n", "analysis-generated bundle checksums")

    subprocess.run(["zip", "-q", "-r", str(ZIP_PATH), BUNDLE.name], cwd=WORKSPACE, check=True)
    print(json.dumps({"bundle": str(BUNDLE), "zip": str(ZIP_PATH), "files": sum(1 for p in BUNDLE.rglob('*') if p.is_file()), "zip_bytes": ZIP_PATH.stat().st_size}, ensure_ascii=False))


if __name__ == "__main__":
    main()
