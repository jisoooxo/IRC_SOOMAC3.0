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
from soomac_irc.agent_prompts import DECISION_SYSTEM  # noqa: E402


def read_first(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8").splitlines()[0])


class TestDecisionV31TrainingBundle(unittest.TestCase):
    def test_training_schedule(self):
        original_mode = trainer.MODE
        trainer.MODE = "full"
        try:
            steps, offsets, total = trainer.stage_layout()
        finally:
            trainer.MODE = original_mode
        self.assertEqual(trainer.STAGE_CONFIGS["decision"]["epochs"], 6)
        self.assertEqual(trainer.STAGE_CONFIGS["decision"]["learning_rate"], 1e-4)
        self.assertEqual(steps, {"decision": 1134})
        self.assertEqual(offsets, {"decision": 0})
        self.assertEqual(total, 1134)

    def test_dataset_manifest_and_current_prompt(self):
        manifest = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["task"], "decision_v3_1_adapter")
        self.assertEqual(manifest["train_rows"], 3015)
        self.assertEqual(manifest["validation_rows"], 285)
        prompt = (DATASET / "contract/decision_system.txt").read_text(encoding="utf-8").strip()
        self.assertEqual(prompt, DECISION_SYSTEM.strip())

    def test_decision_messages_match_runtime_contract(self):
        row = read_first(DATASET / "decision/records/train.jsonl")
        prompt = (DATASET / "contract/decision_system.txt").read_text(encoding="utf-8").strip()
        messages = trainer.build_sft_messages("decision", row, {"decision": prompt})
        self.assertEqual(messages[0], {"role": "system", "content": DECISION_SYSTEM.strip()})
        self.assertEqual(json.loads(messages[1]["content"]), row["input"])
        self.assertEqual(len(messages), 2)

    def test_each_full_epoch_has_a_standalone_adapter_name(self):
        original_mode = trainer.MODE
        trainer.MODE = "full"
        try:
            names = [trainer.adapter_snapshot_name(float(epoch), epoch * 189) for epoch in range(1, 7)]
        finally:
            trainer.MODE = original_mode
        self.assertEqual(names, [f"epoch_{epoch}_adapter" for epoch in range(1, 7)])

    def test_resume_checkpoint_is_complete_and_step_matched(self):
        original_root = trainer.CHECKPOINT_DIR
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trainer.CHECKPOINT_DIR = root
            checkpoint = root / "decision/checkpoint-189"
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
                json.dumps({"global_step": 189}), encoding="utf-8"
            )
            self.assertEqual(
                trainer.validate_resume_checkpoint(checkpoint, 189, "decision", 0),
                189,
            )
            trainer.publish_resume(checkpoint, 189)
            resume = json.loads(
                (root / ".lab-monitor/resume.json").read_text(encoding="utf-8")
            )
            self.assertEqual(resume["checkpoint"], "decision/checkpoint-189")
            self.assertEqual(resume["global_step"], 189)
            with self.assertRaises(ValueError):
                trainer.validate_resume_checkpoint(checkpoint, 190, "decision", 0)
        trainer.CHECKPOINT_DIR = original_root


if __name__ == "__main__":
    unittest.main()
