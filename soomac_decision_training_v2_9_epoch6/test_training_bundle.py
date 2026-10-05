#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
CONTEXT = HERE / "docker_context"
DATASET = CONTEXT / "assets" / "dataset"
sys.path.insert(0, str(CONTEXT))

import train_in_container as trainer_module
from soomac_irc.agent_prompts import DECISION_SYSTEM


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class DummyProcessor:
    def __init__(self):
        self.messages = None
        self.tokenizer = self

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [7, 8]}

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages

        class FakeIds:
            def __getitem__(self, index):
                return self

            def tolist(self):
                return [1, 2, 3]

        return {"input_ids": FakeIds()}

    def save_pretrained(self, path: Path) -> None:
        (path / "processor_saved.txt").write_text("ok\n", encoding="utf-8")


class DummyValidator:
    def validate(self, value) -> None:
        if not isinstance(value, dict):
            raise ValueError("target must be an object")


class TestTrainingBundle(unittest.TestCase):
    def test_current_system_prompt_and_top_level_history_become_chat_messages(self):
        system_prompt = (DATASET / "contract/decision_system.txt").read_text(
            encoding="utf-8"
        ).strip()
        self.assertEqual(system_prompt, DECISION_SYSTEM.strip())
        row = next(
            item
            for item in read_jsonl(DATASET / "records/decision_train.jsonl")
            if item.get("history")
        )
        messages = trainer_module.build_sft_messages(row, system_prompt)
        self.assertEqual(messages[0], {"role": "system", "content": system_prompt})
        self.assertEqual(messages[1:-1], row["history"])
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(json.loads(messages[-1]["content"]), row["input"])
        self.assertNotIn("history", json.loads(messages[-1]["content"]))

        processor = DummyProcessor()
        encoded = trainer_module.encode_record(
            row, processor, system_prompt, DummyValidator()
        )
        self.assertEqual(processor.messages, messages)
        self.assertEqual(encoded["labels"][:3], [-100, -100, -100])

    def test_six_epochs_are_fixed(self):
        self.assertEqual(trainer_module.NUM_TRAIN_EPOCHS, 6)

    def test_resume_manifest_requires_complete_checkpoint(self):
        original = trainer_module.CHECKPOINT_DIR
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "run" / "checkpoint-5"
            checkpoint.mkdir(parents=True)
            for name in (
                "adapter_config.json",
                "adapter_model.safetensors",
                "optimizer.pt",
                "scheduler.pt",
                "rng_state.pth",
            ):
                (checkpoint / name).write_text("ok\n", encoding="utf-8")
            (checkpoint / "trainer_state.json").write_text(
                json.dumps({"global_step": 5}), encoding="utf-8"
            )
            trainer_module.CHECKPOINT_DIR = root
            trainer_module.publish_resume(checkpoint, 5)
            resume = json.loads(
                (root / ".lab-monitor" / "resume.json").read_text(encoding="utf-8")
            )
            self.assertEqual(resume["global_step"], 5)
            self.assertEqual(Path(resume["checkpoint_dir"]), checkpoint)
        trainer_module.CHECKPOINT_DIR = original

    def test_epoch_adapter_is_copied_as_standalone_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "checkpoint-12"
            destination = root / "epoch_6_adapter"
            source.mkdir()
            (source / "adapter_config.json").write_text("{}\n", encoding="utf-8")
            (source / "adapter_model.safetensors").write_bytes(b"weights")
            trainer_module.copy_adapter_checkpoint(
                source, destination, DummyProcessor()
            )
            self.assertTrue((destination / "adapter_config.json").is_file())
            self.assertTrue((destination / "adapter_model.safetensors").is_file())
            self.assertTrue((destination / "processor_saved.txt").is_file())


if __name__ == "__main__":
    unittest.main()
