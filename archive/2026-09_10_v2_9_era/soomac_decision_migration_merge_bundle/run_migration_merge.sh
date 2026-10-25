#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-/home/roma/IRC_SOOMAC3.0}"
LEGACY="${2:-$REPO/decision_adapter_v225_root_cause_review_20260930/dataset_contract/decision_train.jsonl}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${3:-$HERE/output}"

python3 "$HERE/migrate_merge_decision_dataset.py" \
  --legacy "$LEGACY" \
  --supplement "$HERE/supplement_685.jsonl" \
  --schema "$HERE/decision_schema_flexible_snapshot.json" \
  --out "$OUT"
