#!/usr/bin/env python3
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import time

from lab_progress import Progress


APP = Path(os.environ.get("APP_DIR", "/app"))
DATASET_ROOT = Path(os.environ.get("DATASET_DIR", str(APP / "assets" / "dataset")))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/model"))
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", "/checkpoints"))

QUANTIZATION = os.environ.get("TRAIN_QUANTIZATION", "int4").lower()
MODE = os.environ.get("TRAIN_MODE", "smoke").lower()

MAX_LENGTH = 4096
ASSISTANT_END = "<turn|>\n"
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
WARMUP_RATIO = 0.03
DATALOADER_NUM_WORKERS = 4
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
SMOKE_TRAIN_ROWS = 32
SMOKE_VALIDATION_ROWS = 8
SMOKE_MAX_STEPS = 10
SMOKE_SAVE_STEPS = 5
SMOKE_MAX_VRAM_RATIO = 0.95
SAFETY_GATE_STEPS = 10

LORA_TARGET_MODULES = (
    r"model\.language_model\.layers\.\d+\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(gate_proj|up_proj|down_proj))"
)

STAGE_CONFIGS = {
    "decision": {
        "epochs": 6,
        "learning_rate": 1e-4,
        "train_path": DATASET_ROOT / "decision/records/train.jsonl",
        "validation_path": DATASET_ROOT / "decision/records/validation.jsonl",
        "schema_path": DATASET_ROOT / "contract/decision_schema.json",
        "system_path": DATASET_ROOT / "contract/decision_system.txt",
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def package_versions(names: list[str]) -> dict[str, str]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "missing"
    return versions


def planned_steps(stage: str, train_rows: int) -> int:
    if MODE == "smoke":
        return SMOKE_MAX_STEPS
    epochs = int(STAGE_CONFIGS[stage]["epochs"])
    return math.ceil(train_rows / (TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)) * epochs


def adapter_snapshot_name(epoch: float | None, local_step: int) -> str:
    if MODE == "smoke":
        return f"step_{local_step}_adapter"
    epoch_number = int(round(float(epoch or 0)))
    return f"epoch_{epoch_number}_adapter"


def stage_layout() -> tuple[dict[str, int], dict[str, int], int]:
    steps = {
        stage: planned_steps(stage, len(read_jsonl(config["train_path"])))
        for stage, config in STAGE_CONFIGS.items()
    }
    offsets = {"decision": 0}
    return steps, offsets, steps["decision"]


def load_contract(stage: str) -> tuple[dict, object]:
    from jsonschema import Draft202012Validator

    config = STAGE_CONFIGS[stage]
    validator = Draft202012Validator(
        json.loads(config["schema_path"].read_text(encoding="utf-8"))
    )
    if stage != "decision":
        raise ValueError(f"unsupported training stage: {stage}")
    prompts = {"decision": config["system_path"].read_text(encoding="utf-8").strip()}
    return prompts, validator


def build_sft_messages(stage: str, row: dict, prompts: dict) -> list[dict]:
    if stage != "decision":
        raise ValueError(f"unsupported training stage: {stage}")
    system_prompt = prompts["decision"]
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": json.dumps(
                row["input"],
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    ]


def trim_oldest_context(model_input: dict) -> bool:
    history = model_input.get("recent_history")
    if isinstance(history, list) and history:
        del history[:2]
        return True
    return False


def encode_record(stage: str, row: dict, processor, prompts: dict, validator) -> dict:
    validator.validate(row["target"])
    answer_text = json.dumps(
        row["target"],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    answer_ids = processor.tokenizer(
        answer_text + ASSISTANT_END,
        add_special_tokens=False,
    )["input_ids"]
    if not answer_ids:
        raise ValueError(f"{row.get('id')}: assistant target is empty")

    working_row = json.loads(json.dumps(row, ensure_ascii=False))
    while True:
        messages = build_sft_messages(stage, working_row, prompts)
        encoded_prompt = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        prompt_ids = encoded_prompt["input_ids"][0].tolist()
        if len(prompt_ids) + len(answer_ids) <= MAX_LENGTH:
            input_ids = prompt_ids + answer_ids
            return {
                "input_ids": input_ids,
                "labels": [-100] * len(prompt_ids) + answer_ids,
                "length": len(input_ids),
                "id": row.get("id"),
            }
        if not trim_oldest_context(working_row["input"]):
            raise ValueError(
                f"{row.get('id')}: prompt and target exceed MAX_LENGTH={MAX_LENGTH}"
            )


def encode_split(
    stage: str,
    path: Path,
    processor,
    prompts: dict,
    validator,
    limit: int | None = None,
) -> list[dict]:
    encoded = [
        encode_record(stage, row, processor, prompts, validator)
        for row in read_jsonl(path)
    ]
    if limit is not None:
        encoded = sorted(encoded, key=lambda item: item["length"], reverse=True)[:limit]
    return [
        {"input_ids": item["input_ids"], "labels": item["labels"]}
        for item in encoded
    ]


def validate_resume_checkpoint(
    path: Path,
    expected_global_step: int,
    stage: str,
    stage_offset: int,
) -> int:
    if path.is_symlink() or not path.is_dir() or not is_within(path, CHECKPOINT_DIR):
        raise ValueError("RESUME_CHECKPOINT_DIR must be a real directory under CHECKPOINT_DIR")
    relative = path.resolve().relative_to(CHECKPOINT_DIR.resolve())
    if not relative.parts or relative.parts[0] != stage:
        raise ValueError(f"resume checkpoint does not belong to {stage}")
    required = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "trainer_state.json",
    )
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise ValueError(f"resume checkpoint is incomplete: {missing}")
    state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
    local_step = int(state.get("global_step", -1))
    actual_global_step = stage_offset + local_step
    if actual_global_step != expected_global_step:
        raise ValueError(
            "RESUME_STEP mismatch: "
            f"expected={expected_global_step} checkpoint={actual_global_step}"
        )
    return local_step


def publish_resume(checkpoint: Path, global_step: int) -> None:
    if checkpoint.is_symlink() or not checkpoint.is_dir() or not is_within(
        checkpoint, CHECKPOINT_DIR
    ):
        raise ValueError("checkpoint must be a real directory under CHECKPOINT_DIR")
    relative = checkpoint.resolve().relative_to(CHECKPOINT_DIR.resolve()).as_posix()
    write_json(
        CHECKPOINT_DIR / ".lab-monitor" / "resume.json",
        {
            "schema_version": 1,
            "checkpoint": relative,
            "global_step": int(global_step),
        },
    )


def copy_adapter_checkpoint(source: Path, destination: Path, processor) -> None:
    required = ("adapter_config.json", "adapter_model.safetensors")
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise ValueError(f"adapter checkpoint is incomplete: {missing}")
    destination.mkdir(parents=True, exist_ok=True)
    for name in required:
        shutil.copy2(source / name, destination / name)
    processor.save_pretrained(destination)


def checkpoint_stage(path: Path) -> str:
    relative = path.resolve().relative_to(CHECKPOINT_DIR.resolve())
    if not relative.parts or relative.parts[0] not in STAGE_CONFIGS:
        raise ValueError("resume checkpoint must be under decision/")
    return relative.parts[0]


def finalize_completed_checkpoint(
    stage: str,
    checkpoint: Path,
    global_step: int,
    processor,
) -> dict:
    output_dir = CHECKPOINT_DIR / stage
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    best_raw = state.get("best_model_checkpoint")
    best_checkpoint = Path(best_raw) if best_raw else checkpoint
    if not best_checkpoint.is_dir() or not is_within(best_checkpoint, output_dir):
        best_checkpoint = checkpoint
    copy_adapter_checkpoint(best_checkpoint, output_dir / "best_adapter", processor)
    epochs = STAGE_CONFIGS[stage]["epochs"]
    final_name = "smoke_final_adapter" if MODE == "smoke" else f"epoch_{epochs}_adapter"
    copy_adapter_checkpoint(checkpoint, output_dir / final_name, processor)
    result = {
        "stage": stage,
        "global_step": global_step,
        "local_step": int(state["global_step"]),
        "completed_epoch": float(state.get("epoch") or epochs),
        "best_checkpoint": str(best_checkpoint),
        "best_adapter_dir": str(output_dir / "best_adapter"),
        "final_checkpoint": str(checkpoint),
        "final_adapter_dir": str(output_dir / final_name),
        "finalized_from_complete_checkpoint": True,
    }
    write_json(output_dir / "training_complete.json", result)
    publish_resume(checkpoint, global_step)
    return result


def train_stage(
    stage: str,
    *,
    stage_offset: int,
    total_steps: int,
    progress: Progress,
    shared_metrics: dict,
    resume_path: Path | None,
    resume_global_step: int,
    allow_uncommitted_restart: bool = False,
) -> dict:
    import torch
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForMultimodalLM,
        AutoProcessor,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    config = STAGE_CONFIGS[stage]
    smoke = MODE == "smoke"
    epochs = 1 if smoke else int(config["epochs"])
    output_dir = CHECKPOINT_DIR / stage
    completion_path = output_dir / "training_complete.json"

    if completion_path.is_file() and resume_path is not None:
        return json.loads(completion_path.read_text(encoding="utf-8"))
    if completion_path.is_file() and resume_path is None:
        raise RuntimeError(f"Completed output already exists: {output_dir}")
    if resume_path is None and output_dir.exists() and any(output_dir.iterdir()):
        if not allow_uncommitted_restart:
            raise RuntimeError(f"Output directory is not empty: {output_dir}")
        archived = CHECKPOINT_DIR / f"{stage}_uncommitted_{int(time.time())}"
        output_dir.replace(archived)
        print(f"Preserved uncommitted {stage} output at {archived}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress.write(
        force=True,
        phase="loading",
        global_step=stage_offset + (resume_global_step - stage_offset if resume_path else 0),
    )

    prompts, validator = load_contract(stage)
    processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
    train_rows = encode_split(
        stage,
        config["train_path"],
        processor,
        prompts,
        validator,
        SMOKE_TRAIN_ROWS if smoke else None,
    )
    validation_rows = encode_split(
        stage,
        config["validation_path"],
        processor,
        prompts,
        validator,
        SMOKE_VALIDATION_ROWS if smoke else None,
    )
    local_planned_steps = planned_steps(stage, len(read_jsonl(config["train_path"])))

    resume_local_step = 0
    if resume_path is not None:
        resume_local_step = validate_resume_checkpoint(
            resume_path,
            resume_global_step,
            stage,
            stage_offset,
        )
        if resume_local_step == local_planned_steps:
            return finalize_completed_checkpoint(
                stage,
                resume_path,
                resume_global_step,
                processor,
            )

    class EncodedDataset(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            return self.rows[index]

    train_dataset = EncodedDataset(train_rows)
    validation_dataset = EncodedDataset(validation_rows)
    accumulation = 1 if smoke else GRADIENT_ACCUMULATION_STEPS
    warmup_steps = 0 if smoke else max(1, math.ceil(local_planned_steps * WARMUP_RATIO))

    quantization_config = (
        BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        if QUANTIZATION == "int8"
        else BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
    )
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_DIR,
        device_map={"": 0},
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        local_files_only=True,
        quantization_config=quantization_config,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
    )
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES,
            bias="none",
        ),
    )
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable_names:
        raise RuntimeError("No trainable LoRA parameters were created")
    bad_trainable = [
        name
        for name in trainable_names
        if "lora_" in name.lower() and "language_model.layers" not in name.lower()
    ]
    if bad_trainable:
        raise RuntimeError(
            "LoRA was attached outside the text decoder: " + ", ".join(bad_trainable[:10])
        )
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        do_eval=True,
        num_train_epochs=epochs,
        max_steps=SMOKE_MAX_STEPS if smoke else -1,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=accumulation,
        learning_rate=float(config["learning_rate"]),
        lr_scheduler_type="constant" if smoke else "cosine",
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        optim="paged_adamw_8bit",
        max_grad_norm=1.0,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        use_cache=False,
        eval_strategy="steps" if smoke else "epoch",
        save_strategy="steps" if smoke else "epoch",
        eval_steps=SMOKE_SAVE_STEPS if smoke else None,
        save_steps=SMOKE_SAVE_STEPS if smoke else None,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=int(config["epochs"]) + 1,
        logging_strategy="steps",
        logging_steps=1 if smoke else 5,
        logging_first_step=True,
        logging_nan_inf_filter=False,
        report_to="none",
        seed=42,
        data_seed=42,
        dataloader_num_workers=0 if smoke else DATALOADER_NUM_WORKERS,
        remove_unused_columns=False,
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=processor.tokenizer,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    gradient_status = {"seen": 0, "finite": True, "nonzero": False}
    safety_gate = {"loss_seen": False, "passed": False}
    gradient_finite = torch.ones((), dtype=torch.bool, device="cuda")
    gradient_nonzero = torch.zeros((), dtype=torch.bool, device="cuda")

    class ProgressCallback(TrainerCallback):
        def __init__(self):
            self.trainer = None
            self.started = None
            self.seconds = 0.0
            self.skipped = 0

        def sync(self, args):
            if args.device.type == "cuda":
                torch.cuda.synchronize(args.device)

        def on_train_begin(self, args, state, control, **kwargs):
            progress.write(
                force=True,
                phase="training",
                global_step=stage_offset + int(state.global_step),
                target_step=total_steps,
                samples_seen=shared_metrics["samples"],
                tokens_seen=shared_metrics["tokens"],
            )

        def on_step_begin(self, args, state, control, **kwargs):
            self.sync(args)
            self.started = time.perf_counter()

        def on_step_end(self, args, state, control, **kwargs):
            self.sync(args)
            self.seconds += time.perf_counter() - self.started
            skipped = getattr(
                self.trainer.accelerator,
                "optimizer_step_was_skipped",
                False,
            )
            self.skipped += int(skipped)
            progress.write(
                phase="training",
                global_step=stage_offset + int(state.global_step),
                training_seconds=shared_metrics["seconds"] + self.seconds,
                samples_seen=shared_metrics["samples"] + self.trainer.lab_samples,
                tokens_seen=shared_metrics["tokens"] + self.trainer.lab_tokens,
                skipped_updates=shared_metrics["skipped"] + self.skipped,
            )

        def on_log(self, args, state, control, logs=None, **kwargs):
            values = {}
            if logs and "loss" in logs:
                loss = float(logs["loss"])
                if not math.isfinite(loss):
                    raise RuntimeError(
                        f"Safety gate failed: non-finite loss at step {state.global_step}: {loss}"
                    )
                values["loss"] = loss
                safety_gate["loss_seen"] = True
            if logs and "learning_rate" in logs:
                values["learning_rate"] = float(logs["learning_rate"])
            if values:
                progress.write(**values)
            completed_this_run = int(state.global_step) - resume_local_step
            required_gate_steps = min(
                SAFETY_GATE_STEPS,
                local_planned_steps - resume_local_step,
            )
            if completed_this_run >= required_gate_steps and not safety_gate["passed"]:
                gradient_status["finite"] = bool(gradient_finite.item())
                gradient_status["nonzero"] = bool(gradient_nonzero.item())
                if not safety_gate["loss_seen"]:
                    raise RuntimeError("Safety gate failed: no loss was logged")
                if (
                    gradient_status["seen"] == 0
                    or not gradient_status["finite"]
                    or not gradient_status["nonzero"]
                ):
                    raise RuntimeError(f"Safety gate failed: gradients={gradient_status}")
                safety_gate["passed"] = True
                for hook in gradient_hooks:
                    hook.remove()
                gradient_hooks.clear()
                print(
                    f"PASS: {stage} {QUANTIZATION.upper()} 10-step safety gate",
                    flush=True,
                )

        def on_train_end(self, args, state, control, **kwargs):
            progress.write(
                force=True,
                phase="saving",
                global_step=stage_offset + int(state.global_step),
            )

    callback = ProgressCallback()

    class CountingTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            self.lab_samples = 0
            self.lab_tokens = 0
            super().__init__(*args, **kwargs)

        def training_step(self, model, inputs, num_items_in_batch=None):
            if "input_ids" not in inputs:
                raise ValueError("input_ids is required for sample accounting")
            self.lab_samples += int(inputs["input_ids"].shape[0])
            if "attention_mask" in inputs:
                self.lab_tokens += int(inputs["attention_mask"].sum().item())
            return super().training_step(model, inputs, num_items_in_batch)

        def evaluate(self, *args, **kwargs):
            progress.write(force=True, phase="evaluating")
            try:
                return super().evaluate(*args, **kwargs)
            finally:
                progress.write(force=True, phase="training")

        def _save_checkpoint(self, *args, **kwargs):
            progress.write(force=True, phase="saving")
            try:
                result = super()._save_checkpoint(*args, **kwargs)
                local_step = int(self.state.global_step)
                global_step = stage_offset + local_step
                checkpoint = Path(self.args.output_dir) / f"checkpoint-{local_step}"
                adapter_name = adapter_snapshot_name(self.state.epoch, local_step)
                copy_adapter_checkpoint(
                    checkpoint,
                    Path(self.args.output_dir) / adapter_name,
                    processor,
                )
                publish_resume(checkpoint, global_step)
                fail_after = int(os.environ.get("LAB_TEST_FAIL_AFTER_GLOBAL_STEP", "0"))
                if fail_after and global_step >= fail_after:
                    raise RuntimeError(
                        f"Injected resume test failure after global step {global_step}"
                    )
                return result
            finally:
                progress.write(force=True, phase="training")

    trainer = CountingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=processor,
        data_collator=data_collator,
        callbacks=[callback],
    )
    callback.trainer = trainer

    manifest = {
        "training_image_spec": "2.9",
        "task": f"{stage}_adapter",
        "mode": MODE,
        "quantization": QUANTIZATION,
        "model_config_sha256": sha256_file(MODEL_DIR / "config.json"),
        "train_rows": len(train_dataset),
        "validation_rows": len(validation_dataset),
        "planned_optimizer_steps": local_planned_steps,
        "stage_offset": stage_offset,
        "reported_target_step": total_steps,
        "safety_gate_steps": SAFETY_GATE_STEPS,
        "max_length": MAX_LENGTH,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": accumulation,
        "learning_rate": float(config["learning_rate"]),
        "epochs": epochs,
        "resume_enabled": resume_path is not None,
        "resume_local_step": resume_local_step,
        "resume_global_step": resume_global_step if resume_path is not None else 0,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "target_modules": LORA_TARGET_MODULES,
        "packages": package_versions(
            ["torch", "transformers", "peft", "accelerate", "bitsandbytes"]
        ),
    }
    write_json(output_dir / "run_manifest.json", manifest)

    def check_gradient(gradient):
        gradient_status["seen"] += 1
        gradient_finite.logical_and_(torch.isfinite(gradient).all())
        gradient_nonzero.logical_or_(torch.count_nonzero(gradient) > 0)
        return gradient

    gradient_hooks = [
        parameter.register_hook(check_gradient)
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    torch.cuda.reset_peak_memory_stats()
    try:
        train_result = trainer.train(
            resume_from_checkpoint=str(resume_path) if resume_path is not None else None
        )
    finally:
        for hook in gradient_hooks:
            hook.remove()

    peak_allocated_gib = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved_gib = torch.cuda.max_memory_reserved() / 1024**3
    total_vram_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    train_loss = train_result.metrics.get("train_loss")
    if not isinstance(train_loss, (int, float)) or not math.isfinite(train_loss):
        raise RuntimeError(f"Training loss is not finite: {train_loss}")
    if not safety_gate["passed"]:
        raise RuntimeError(
            "Training ended before the 10-step safety gate passed: "
            f"gradients={gradient_status}"
        )
    if smoke and peak_reserved_gib > total_vram_gib * SMOKE_MAX_VRAM_RATIO:
        raise RuntimeError(
            "Smoke VRAM headroom is too small: "
            f"reserved={peak_reserved_gib:.2f}GiB total={total_vram_gib:.2f}GiB"
        )
    if not smoke and (
        trainer.state.epoch is None
        or abs(float(trainer.state.epoch) - float(config["epochs"])) > 1e-6
    ):
        raise RuntimeError(
            f"{stage} training did not reach epoch {config['epochs']}: {trainer.state.epoch}"
        )

    progress.write(force=True, phase="saving")
    best_adapter_dir = output_dir / "best_adapter"
    trainer.save_model(str(best_adapter_dir))
    processor.save_pretrained(best_adapter_dir)
    final_checkpoint = output_dir / f"checkpoint-{int(trainer.state.global_step)}"
    final_name = "smoke_final_adapter" if smoke else f"epoch_{config['epochs']}_adapter"
    final_adapter_dir = output_dir / final_name
    copy_adapter_checkpoint(final_checkpoint, final_adapter_dir, processor)
    final_global_step = stage_offset + int(trainer.state.global_step)
    publish_resume(final_checkpoint, final_global_step)
    trainer.save_metrics("train", train_result.metrics)
    result = {
        "stage": stage,
        "global_step": final_global_step,
        "local_step": int(trainer.state.global_step),
        "train_loss": train_loss,
        "samples_seen": trainer.lab_samples,
        "tokens_seen": trainer.lab_tokens,
        "training_seconds": callback.seconds,
        "skipped_updates": callback.skipped,
        "peak_allocated_gib": round(peak_allocated_gib, 3),
        "peak_reserved_gib": round(peak_reserved_gib, 3),
        "gradient_check": gradient_status,
        "safety_gate_passed": safety_gate["passed"],
        "completed_epoch": float(trainer.state.epoch or 0),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_adapter_dir": str(best_adapter_dir),
        "final_checkpoint": str(final_checkpoint),
        "final_adapter_dir": str(final_adapter_dir),
    }
    write_json(completion_path, result)
    shared_metrics["samples"] += trainer.lab_samples
    shared_metrics["tokens"] += trainer.lab_tokens
    shared_metrics["seconds"] += callback.seconds
    shared_metrics["skipped"] += callback.skipped

    # 종료 전에 Trainer의 순환 참조를 끊어 CUDA 가중치가 실제로 해제되게 한다.
    callback.trainer = None
    del trainer, model, train_dataset, validation_dataset, train_rows, validation_rows
    del gradient_finite, gradient_nonzero, gradient_hooks, check_gradient
    gc.collect()
    torch.cuda.empty_cache()
    print(f"PASS: {stage} training finished: {output_dir}", flush=True)
    return result


def main() -> None:
    if QUANTIZATION not in {"int8", "int4"}:
        raise ValueError("TRAIN_QUANTIZATION must be int8 or int4")
    if MODE not in {"smoke", "full"}:
        raise ValueError("TRAIN_MODE must be smoke or full")
    required = [MODEL_DIR, MODEL_DIR / "config.json", DATASET_ROOT / "manifest.json"]
    for config in STAGE_CONFIGS.values():
        required.extend(
            path
            for key, path in config.items()
            if key.endswith("_path") and isinstance(path, Path)
        )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    import torch
    from transformers import AutoProcessor

    from gpu_budget import configure_torch_memory

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available")
    configure_torch_memory()

    stage_steps, offsets, total_steps = stage_layout()
    resume_enabled = os.environ.get("LAB_RESUME") == "1"
    resume_path = None
    resume_global_step = 0
    if resume_enabled:
        raw_path = os.environ.get("RESUME_CHECKPOINT_DIR", "").strip()
        raw_step = os.environ.get("RESUME_STEP", "").strip()
        if not raw_path or not raw_step:
            raise ValueError(
                "LAB_RESUME=1 requires RESUME_CHECKPOINT_DIR and RESUME_STEP"
            )
        resume_path = Path(raw_path)
        resume_stage = checkpoint_stage(resume_path)
        resume_global_step = int(raw_step)
        validate_resume_checkpoint(
            resume_path,
            resume_global_step,
            resume_stage,
            offsets[resume_stage],
        )

    progress = Progress(
        start_step=resume_global_step,
        target_step=total_steps,
        sample_unit="sequences",
    )
    progress.start_heartbeat(interval=10)
    progress.write(force=True, phase="loading")
    shared_metrics = {"samples": 0, "tokens": 0, "seconds": 0.0, "skipped": 0}

    try:
        result = train_stage(
            "decision",
            stage_offset=offsets["decision"],
            total_steps=total_steps,
            progress=progress,
            shared_metrics=shared_metrics,
            resume_path=resume_path,
            resume_global_step=resume_global_step,
        )
        if not (CHECKPOINT_DIR / "decision/training_complete.json").is_file():
            raise RuntimeError("Decision v3.2 training completion marker is missing")
        write_json(
            CHECKPOINT_DIR / "decision_v3_2_training_complete.json",
            {
                "training_image_spec": "2.9",
                "mode": MODE,
                "quantization": QUANTIZATION,
                "total_optimizer_steps": total_steps,
                "stage_steps": stage_steps,
                "result": result,
            },
        )
        progress.write(
            force=True,
            phase="completed",
            global_step=total_steps,
            samples_seen=shared_metrics["samples"],
            tokens_seen=shared_metrics["tokens"],
            training_seconds=shared_metrics["seconds"],
            skipped_updates=shared_metrics["skipped"],
        )
        print("PASS: Decision v3.2 adapter completed", flush=True)
    finally:
        progress.close()


if __name__ == "__main__":
    main()
