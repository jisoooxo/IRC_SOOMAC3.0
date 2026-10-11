#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
CONTEXT = HERE / "docker_context"
DATASET = CONTEXT / "assets/dataset"
os.environ["APP_DIR"] = str(CONTEXT)
os.environ["DATASET_DIR"] = str(DATASET)
sys.path.insert(0, str(CONTEXT))
sys.path.insert(0, str(HERE.parent / "src/soomac_irc"))

import train_in_container as trainer  # noqa: E402
from soomac_irc.agent_prompts import (  # noqa: E402
    GENERAL_RESPONSE_SYSTEM,
    MIXED_RESPONSE_SYSTEM,
    TASK_RESPONSE_SYSTEM,
)


def read_first(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8").splitlines()[0])


class TestResponseV11TrainingBundle(unittest.TestCase):
    def test_training_schedule(self):
        original_mode = trainer.MODE
        trainer.MODE = "full"
        try:
            steps, offsets, total = trainer.stage_layout()
        finally:
            trainer.MODE = original_mode
        self.assertEqual(trainer.STAGE_CONFIGS["response"]["epochs"], 4)
        self.assertEqual(trainer.STAGE_CONFIGS["response"]["learning_rate"], 5e-5)
        self.assertEqual(steps, {"response": 276})
        self.assertEqual(offsets, {"response": 0})
        self.assertEqual(total, 276)

    def test_manifest_and_current_prompts(self):
        manifest = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["task"], "response_v1_1_adapter")
        self.assertEqual(manifest["train_rows"], 1098)
        self.assertEqual(manifest["validation_rows"], 101)
        prompts = json.loads(
            (DATASET / "contract/response_system_by_route.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            prompts,
            {
                "task": TASK_RESPONSE_SYSTEM.strip(),
                "general": GENERAL_RESPONSE_SYSTEM.strip(),
                "mixed": MIXED_RESPONSE_SYSTEM.strip(),
            },
        )

    def test_response_messages_match_runtime_contract(self):
        row = read_first(DATASET / "response/records/train.jsonl")
        prompts = json.loads(
            (DATASET / "contract/response_system_by_route.json").read_text(encoding="utf-8")
        )
        messages = trainer.build_sft_messages("response", row, prompts)
        self.assertEqual(messages[0], {"role": "system", "content": prompts[row["route"]]})
        self.assertEqual(json.loads(messages[1]["content"]), row["input"])
        self.assertEqual(len(messages), 2)

    def test_each_full_epoch_has_a_standalone_adapter_name(self):
        original_mode = trainer.MODE
        trainer.MODE = "full"
        try:
            names = [trainer.adapter_snapshot_name(float(epoch), epoch * 69) for epoch in range(1, 5)]
        finally:
            trainer.MODE = original_mode
        self.assertEqual(names, [f"epoch_{epoch}_adapter" for epoch in range(1, 5)])

    def test_resume_checkpoint_is_complete_and_step_matched(self):
        original_root = trainer.CHECKPOINT_DIR
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trainer.CHECKPOINT_DIR = root
            checkpoint = root / "response/checkpoint-69"
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
                json.dumps({"global_step": 69}), encoding="utf-8"
            )
            self.assertEqual(
                trainer.validate_resume_checkpoint(checkpoint, 69, "response", 0),
                69,
            )
            trainer.publish_resume(checkpoint, 69)
            resume = json.loads((root / ".lab-monitor/resume.json").read_text(encoding="utf-8"))
            self.assertEqual(resume["checkpoint"], "response/checkpoint-69")
            self.assertEqual(resume["global_step"], 69)
            with self.assertRaises(ValueError):
                trainer.validate_resume_checkpoint(checkpoint, 70, "response", 0)
        trainer.CHECKPOINT_DIR = original_root


if __name__ == "__main__":
    unittest.main()
