#!/usr/bin/env python3
from __future__ import annotations

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

TRAIN_PATH = DATASET_ROOT / "records/decision_train.jsonl"
VALIDATION_PATH = DATASET_ROOT / "records/decision_validation.jsonl"
SYSTEM_PATH = DATASET_ROOT / "contract/decision_system.txt"
SCHEMA_PATH = DATASET_ROOT / "contract/decision_schema.json"

QUANTIZATION = os.environ.get("TRAIN_QUANTIZATION", "int4").lower()
MODE = os.environ.get("TRAIN_MODE", "smoke").lower()
PROGRESS_START_STEP = int(os.environ.get("LAB_PROGRESS_START_STEP", "0"))
PROGRESS_TARGET_STEP = os.environ.get("LAB_PROGRESS_TARGET_STEP")
PROGRESS_TARGET_STEP = (
    int(PROGRESS_TARGET_STEP) if PROGRESS_TARGET_STEP is not None else None
)
FINAL_STAGE = os.environ.get("LAB_FINAL_STAGE", "1") == "1"
OUTPUT_NAME = os.environ.get("OUTPUT_NAME", "").strip()

MAX_LENGTH = 4096
ASSISTANT_END = "<turn|>\n"
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 1e-4
WARMUP_RATIO = 0.03
NUM_TRAIN_EPOCHS = 6
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
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contract_sha256() -> str:
    payload = {
        "decision_system": SYSTEM_PATH.read_text(encoding="utf-8"),
        "decision_schema": json.loads(SCHEMA_PATH.read_text(encoding="utf-8")),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_history(row: dict) -> list[dict]:
    history = row.get("history", [])
    if not isinstance(history, list):
        raise ValueError(f"{row.get('id')}: history must be a list")
    if isinstance(row.get("input"), dict) and "history" in row["input"]:
        raise ValueError(f"{row.get('id')}: history must stay top-level")
    for index, message in enumerate(history):
        if not isinstance(message, dict):
            raise ValueError(f"{row.get('id')}: history[{index}] must be an object")
        if message.get("role") not in {"user", "assistant"}:
            raise ValueError(f"{row.get('id')}: invalid history role at {index}")
        if not isinstance(message.get("content"), str) or not message["content"]:
            raise ValueError(f"{row.get('id')}: empty history content at {index}")
    return json.loads(json.dumps(history, ensure_ascii=False))


def build_sft_messages(
    row: dict,
    system_prompt: str,
    *,
    history: list[dict] | None = None,
    action_history: list[dict] | None = None,
) -> list[dict]:
    selected_history = validate_history(row) if history is None else history
    model_input = json.loads(json.dumps(row["input"], ensure_ascii=False))
    if action_history is not None:
        model_input["action_history"] = action_history
    return [
        {"role": "system", "content": system_prompt},
        *selected_history,
        {
            "role": "user",
            "content": json.dumps(
                model_input,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    ]


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def validate_resume_checkpoint(path: Path, expected_step: int) -> int:
    if path.is_symlink() or not path.is_dir() or not is_within(path, CHECKPOINT_DIR):
        raise ValueError("RESUME_CHECKPOINT_DIR must be a real directory under CHECKPOINT_DIR")
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
    actual_step = int(state.get("global_step", -1))
    if actual_step != expected_step:
        raise ValueError(
            f"RESUME_STEP mismatch: expected={expected_step} checkpoint={actual_step}"
        )
    return actual_step


def publish_resume(checkpoint: Path, step: int) -> None:
    validate_resume_checkpoint(checkpoint, step)
    write_json(
        CHECKPOINT_DIR / ".lab-monitor" / "resume.json",
        {
            "schema_version": 1,
            "checkpoint_dir": str(checkpoint.resolve()),
            "global_step": step,
            "saved_at": time.time(),
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


def package_versions(names: list[str]) -> dict[str, str]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "missing"
    return versions


def encode_record(row, processor, system_prompt, validator):
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

    history = validate_history(row)
    base_input = json.loads(json.dumps(row["input"], ensure_ascii=False))
    action_history = list(base_input.get("action_history", []))

    while True:
        messages = build_sft_messages(
            {**row, "input": base_input},
            system_prompt,
            history=history,
            action_history=action_history,
        )
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
        if history:
            history = history[2:] if len(history) >= 2 else []
            continue
        if action_history:
            action_history = action_history[1:]
            continue
        raise ValueError(
            f"{row.get('id')}: prompt and target exceed MAX_LENGTH={MAX_LENGTH}"
        )


def encode_split(path, processor, system_prompt, validator, limit=None):
    encoded = [
        encode_record(row, processor, system_prompt, validator)
        for row in read_jsonl(path)
    ]
    if limit is not None:
        encoded = sorted(encoded, key=lambda item: item["length"], reverse=True)[:limit]
    return [
        {"input_ids": item["input_ids"], "labels": item["labels"]}
        for item in encoded
    ]


def main() -> None:
    progress = Progress(
        start_step=PROGRESS_START_STEP,
        target_step=PROGRESS_TARGET_STEP,
        sample_unit="sequences",
    )
    progress.start_heartbeat(interval=10)
    progress.write(force=True, phase="loading")

    try:
        resume_enabled = os.environ.get("LAB_RESUME") == "1"
        resume_path = None
        resume_step = 0
        if resume_enabled:
            raw_path = os.environ.get("RESUME_CHECKPOINT_DIR", "").strip()
            raw_step = os.environ.get("RESUME_STEP", "").strip()
            if not raw_path or not raw_step:
                raise ValueError(
                    "LAB_RESUME=1 requires RESUME_CHECKPOINT_DIR and RESUME_STEP"
                )
            resume_path = Path(raw_path)
            resume_step = validate_resume_checkpoint(resume_path, int(raw_step))
        if QUANTIZATION not in {"int8", "int4"}:
            raise ValueError("TRAIN_QUANTIZATION must be int8 or int4")
        if MODE not in {"smoke", "full"}:
            raise ValueError("TRAIN_MODE must be smoke or full")

        for path in (
            MODEL_DIR,
            MODEL_DIR / "config.json",
            TRAIN_PATH,
            VALIDATION_PATH,
            SYSTEM_PATH,
            SCHEMA_PATH,
        ):
            if not path.exists():
                raise FileNotFoundError(path)

        import torch
        from jsonschema import Draft202012Validator
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

        from gpu_budget import configure_torch_memory

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is not available")
        configure_torch_memory()

        smoke = MODE == "smoke"
        output_name = OUTPUT_NAME or (
            "gemma4_decision_int8_lora_flexible_epoch6"
            if QUANTIZATION == "int8"
            else "gemma4_decision_nf4_qlora_flexible_epoch6"
        )
        if Path(output_name).name != output_name:
            raise ValueError("OUTPUT_NAME must be a single directory name")
        if smoke and not resume_enabled:
            output_name += "_smoke"
        output_dir = resume_path.parent if resume_path is not None else CHECKPOINT_DIR / output_name
        if not resume_enabled and output_dir.exists() and any(output_dir.iterdir()):
            raise RuntimeError(f"Output directory is not empty: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        system_prompt = SYSTEM_PATH.read_text(encoding="utf-8").strip()
        validator = Draft202012Validator(
            json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        )
        processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)

        train_rows = encode_split(
            TRAIN_PATH,
            processor,
            system_prompt,
            validator,
            SMOKE_TRAIN_ROWS if smoke else None,
        )
        validation_rows = encode_split(
            VALIDATION_PATH,
            processor,
            system_prompt,
            validator,
            SMOKE_VALIDATION_ROWS if smoke else None,
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
        planned_steps = (
            SMOKE_MAX_STEPS
            if smoke
            else math.ceil(len(train_dataset) / (TRAIN_BATCH_SIZE * accumulation))
            * NUM_TRAIN_EPOCHS
        )
        warmup_steps = 0 if smoke else max(1, math.ceil(planned_steps * WARMUP_RATIO))

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
                "LoRA was attached outside the text decoder: "
                + ", ".join(bad_trainable[:10])
            )
        model.print_trainable_parameters()

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            do_train=True,
            do_eval=True,
            num_train_epochs=1 if smoke else NUM_TRAIN_EPOCHS,
            max_steps=SMOKE_MAX_STEPS if smoke else -1,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            gradient_accumulation_steps=accumulation,
            learning_rate=LEARNING_RATE,
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
            save_total_limit=7,
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
                target_step = PROGRESS_TARGET_STEP
                if target_step is None:
                    target_step = int(state.max_steps)
                progress.write(
                    force=True,
                    phase="training",
                    start_step=max(PROGRESS_START_STEP, int(state.global_step)),
                    global_step=max(PROGRESS_START_STEP, int(state.global_step)),
                    target_step=target_step,
                    samples_seen=0,
                    tokens_seen=0,
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
                    global_step=int(state.global_step),
                    training_seconds=self.seconds,
                    samples_seen=self.trainer.lab_samples,
                    tokens_seen=self.trainer.lab_tokens,
                    skipped_updates=self.skipped,
                )

            def on_log(self, args, state, control, logs=None, **kwargs):
                values = {}
                if logs and "loss" in logs:
                    loss = float(logs["loss"])
                    if not math.isfinite(loss):
                        raise RuntimeError(
                            f"Safety gate failed: non-finite loss at step "
                            f"{state.global_step}: {loss}"
                        )
                    values["loss"] = loss
                    safety_gate["loss_seen"] = True
                if logs and "learning_rate" in logs:
                    values["learning_rate"] = float(logs["learning_rate"])
                if values:
                    progress.write(**values)
                if (
                    int(state.global_step) >= SAFETY_GATE_STEPS
                    and not safety_gate["passed"]
                ):
                    gradient_status["finite"] = bool(gradient_finite.item())
                    gradient_status["nonzero"] = bool(gradient_nonzero.item())
                    if not safety_gate["loss_seen"]:
                        raise RuntimeError("Safety gate failed: no loss was logged")
                    if (
                        gradient_status["seen"] == 0
                        or not gradient_status["finite"]
                        or not gradient_status["nonzero"]
                    ):
                        raise RuntimeError(
                            f"Safety gate failed: gradients={gradient_status}"
                        )
                    safety_gate["passed"] = True
                    for hook in gradient_hooks:
                        hook.remove()
                    gradient_hooks.clear()
                    print(
                        f"PASS: {QUANTIZATION.upper()} 10-step safety gate; "
                        "continuing the same training run.",
                        flush=True,
                    )

            def on_train_end(self, args, state, control, **kwargs):
                progress.write(
                    force=True,
                    phase="saving",
                    global_step=int(state.global_step),
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
                    step = int(self.state.global_step)
                    checkpoint = Path(self.args.output_dir) / f"checkpoint-{step}"
                    publish_resume(checkpoint, step)
                    fail_after = int(os.environ.get("LAB_TEST_FAIL_AFTER_STEP", "0"))
                    if fail_after and step >= fail_after:
                        raise RuntimeError(
                            f"Injected resume test failure after checkpoint step {step}"
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
            "task": "decision_adapter",
            "mode": MODE,
            "quantization": QUANTIZATION,
            "model_config_sha256": sha256_file(MODEL_DIR / "config.json"),
            "decision_system_sha256": sha256_file(SYSTEM_PATH),
            "decision_schema_sha256": sha256_file(SCHEMA_PATH),
            "contract_sha256": contract_sha256(),
            "sft_message_order": [
                "system:DECISION_SYSTEM",
                "top_level_history",
                "user:input_json",
                "assistant:target_json",
            ],
            "train_rows": len(train_dataset),
            "validation_rows": len(validation_dataset),
            "planned_optimizer_steps": planned_steps,
            "reported_start_step": PROGRESS_START_STEP,
            "reported_target_step": (
                PROGRESS_TARGET_STEP
                if PROGRESS_TARGET_STEP is not None
                else planned_steps
            ),
            "final_stage": FINAL_STAGE,
            "safety_gate_steps": SAFETY_GATE_STEPS,
            "max_length": MAX_LENGTH,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "gradient_accumulation_steps": accumulation,
            "learning_rate": LEARNING_RATE,
            "epochs": 1 if smoke else NUM_TRAIN_EPOCHS,
            "resume_enabled": resume_enabled,
            "resume_step": resume_step,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "target_modules": LORA_TARGET_MODULES,
            "packages": package_versions(
                ["torch", "transformers", "peft", "accelerate", "bitsandbytes"]
            ),
        }
        write_json(output_dir / "run_manifest.json", manifest)

        gradient_status = {"seen": 0, "finite": True, "nonzero": False}
        safety_gate = {"loss_seen": False, "passed": False}
        gradient_finite = torch.ones((), dtype=torch.bool, device="cuda")
        gradient_nonzero = torch.zeros((), dtype=torch.bool, device="cuda")

        def check_gradient(gradient):
            gradient_status["seen"] += 1
            gradient_finite.logical_and_(
                torch.isfinite(gradient).all()
            )
            gradient_nonzero.logical_or_(
                torch.count_nonzero(gradient) > 0
            )
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
        if smoke:
            if peak_reserved_gib > total_vram_gib * SMOKE_MAX_VRAM_RATIO:
                raise RuntimeError(
                    "Smoke VRAM headroom is too small: "
                    f"reserved={peak_reserved_gib:.2f}GiB total={total_vram_gib:.2f}GiB"
                )
        elif trainer.state.epoch is None or abs(float(trainer.state.epoch) - 6.0) > 1e-6:
            raise RuntimeError(f"Full training did not reach epoch 6: {trainer.state.epoch}")

        progress.write(force=True, phase="saving")
        best_adapter_dir = output_dir / "best_adapter"
        trainer.save_model(str(best_adapter_dir))
        processor.save_pretrained(best_adapter_dir)

        final_checkpoint = output_dir / f"checkpoint-{int(trainer.state.global_step)}"
        final_adapter_name = "smoke_final_adapter" if smoke else "epoch_6_adapter"
        final_adapter_dir = output_dir / final_adapter_name
        copy_adapter_checkpoint(final_checkpoint, final_adapter_dir, processor)
        publish_resume(final_checkpoint, int(trainer.state.global_step))
        trainer.save_metrics("train", train_result.metrics)
        write_json(
            output_dir / "training_complete.json",
            {
                "global_step": int(trainer.state.global_step),
                "reported_global_step": (
                    int(trainer.state.global_step)
                ),
                "train_loss": train_loss,
                "samples_seen": trainer.lab_samples,
                "tokens_seen": trainer.lab_tokens,
                "peak_allocated_gib": round(peak_allocated_gib, 3),
                "peak_reserved_gib": round(peak_reserved_gib, 3),
                "gradient_check": gradient_status,
                "safety_gate_passed": safety_gate["passed"],
                "completed_epoch": float(trainer.state.epoch or 0),
                "best_checkpoint": trainer.state.best_model_checkpoint,
                "best_adapter_dir": str(best_adapter_dir),
                "final_checkpoint": str(final_checkpoint),
                "final_adapter_dir": str(final_adapter_dir),
            },
        )
        if FINAL_STAGE:
            progress.write(
                force=True,
                phase="completed",
                global_step=int(trainer.state.global_step),
                samples_seen=trainer.lab_samples,
                tokens_seen=trainer.lab_tokens,
            )
        print(f"PASS: training finished: {output_dir}", flush=True)
    finally:
        progress.close()


if __name__ == "__main__":
    main()
