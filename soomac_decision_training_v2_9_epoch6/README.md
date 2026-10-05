# SOOMAC Decision Adapter Training Image — Spec 2.9

- Image default: Gemma 4 12B NF4 QLoRA, 6 epochs.
- Embedded dataset: train 3,049 / validation 290 / hard eval 495.
- SFT messages: current runtime `DECISION_SYSTEM` → top-level `history` messages → current `input` JSON → Decision target.
- Outputs: `best_adapter`, `epoch_6_adapter`, full `checkpoint-*`, metrics, and `.lab-monitor/resume.json` under `CHECKPOINT_DIR`.
- Resume contract: `LAB_RESUME=1`, `RESUME_CHECKPOINT_DIR`, `RESUME_STEP`.

The original `final_train_rebalanced.jsonl` is not modified. Eight records that duplicated the same history in both top-level `history` and `input.history` are normalized only in the embedded training copy; top-level history remains authoritative.

## Build

```bash
cd docker_context
docker build --platform linux/amd64 \
  --build-arg LAB_INPUTS="$(python3 -c 'import json; print(json.dumps(json.load(open("lab-inputs.json")), separators=(",", ":")))')" \
  -t soomac/decision-flexible-v29:epoch6-nf4 .
```

## Output layout

```text
/checkpoints/gemma4_decision_nf4_qlora_flexible_epoch6/
├── best_adapter/
├── epoch_6_adapter/
├── checkpoint-*/
├── run_manifest.json
├── train_results.json
└── training_complete.json
```
