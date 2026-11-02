"""CPU-only import/encode probe (no unsloth import, no CUDA). Dumps loaded distributions."""
import sys, json, os
from pathlib import Path
REL = Path(sys.argv[1]); out = sys.argv[2]
sys.path.insert(0, str(REL / "common"))
import torch, transformers, peft, trl, accelerate, datasets, bitsandbytes, safetensors, jsonschema, numpy  # noqa
from transformers import AutoProcessor, DataCollatorForSeq2Seq, TrainingArguments, Trainer  # noqa
import core, loss, verification, checkpointing, budget, lab_progress  # noqa
assert not torch.cuda.is_initialized()
agent = "decision"
manifest = json.loads((REL / "dataset/MANIFEST.json").read_text())
core.DATASET_ROOT = REL / "dataset"; core.MAX_LENGTH = manifest["max_length"][agent]
cfg = core.STAGE_CONFIGS[agent]
cfg.update(train_path=REL / "dataset/records/decision/train.jsonl",
           validation_path=REL / "dataset/records/decision/validation.jsonl",
           schema_path=REL / "dataset/contract/decision_schema.json",
           system_path=REL / "dataset/contract/decision_system.txt")
processor = AutoProcessor.from_pretrained(REL / "assets/model-processor", local_files_only=True)
prompts, validator = core.load_contract(agent)
raw = core.read_jsonl(cfg["train_path"])
enc = [core.encode_record(agent, r, processor, prompts, validator) for r in raw[:3]]
import importlib.metadata as md
tops = {m.split('.')[0] for m in sys.modules}
pkg = md.packages_distributions()
dists = sorted({d for t in tops for d in pkg.get(t, [])})
json.dump({"train_rows": len(raw), "encoded_lengths": [e["length"] for e in enc],
           "scored": [sum(l != -100 for l in e["labels"]) for e in enc],
           "input_ids_sha": __import__("hashlib").sha256(json.dumps([e["input_ids"] for e in enc]).encode()).hexdigest(),
           "loaded_distributions": dists, "cuda_initialized": torch.cuda.is_initialized()},
          open(out, "w"), indent=1)
print("rows", len(raw), "lengths", [e["length"] for e in enc], "dists", len(dists))
