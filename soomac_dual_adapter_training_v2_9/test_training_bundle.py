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
DATASET = CONTEXT / "assets" / "dataset"
os.environ["APP_DIR"] = str(CONTEXT)
os.environ["DATASET_DIR"] = str(DATASET)
sys.path.insert(0, str(CONTEXT))
sys.path.insert(0, str(HERE.parent / "src" / "soomac_irc"))

import train_in_container as trainer
from soomac_irc.agent_prompts import (
    DECISION_SYSTEM,
    GENERAL_RESPONSE_SYSTEM,
    MIXED_RESPONSE_SYSTEM,
    TASK_RESPONSE_SYSTEM,
)


def read_first(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8").splitlines()[0])


class TestDualTrainingBundle(unittest.TestCase):
    def test_stage_parameters_and_optimizer_steps(self):
        original_mode = trainer.MODE
        trainer.MODE = "full"
        try:
            steps, offsets, total = trainer.stage_layout()
        finally:
            trainer.MODE = original_mode
        self.assertEqual(trainer.STAGE_CONFIGS["decision"]["epochs"], 6)
        self.assertEqual(trainer.STAGE_CONFIGS["decision"]["learning_rate"], 1e-4)
        self.assertEqual(trainer.STAGE_CONFIGS["response"]["epochs"], 4)
        self.assertEqual(trainer.STAGE_CONFIGS["response"]["learning_rate"], 5e-5)
        self.assertEqual(steps, {"decision": 978, "response": 228})
        self.assertEqual(offsets, {"decision": 0, "response": 978})
        self.assertEqual(total, 1206)

    def test_decision_messages_match_runtime_contract(self):
        row = read_first(DATASET / "decision/records/train.jsonl")
        prompt = (DATASET / "contract/decision_system.txt").read_text(
            encoding="utf-8"
        ).strip()
        messages = trainer.build_sft_messages(
            "decision",
            row,
            {"decision": prompt},
        )
        self.assertEqual(messages[0], {"role": "system", "content": DECISION_SYSTEM.strip()})
        self.assertEqual(json.loads(messages[1]["content"]), row["input"])
        self.assertEqual(len(messages), 2)

    def test_response_route_prompts_match_runtime_contract(self):
        prompts = json.loads(
            (DATASET / "contract/response_system_by_route.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            prompts,
            {
                "task": TASK_RESPONSE_SYSTEM.strip(),
                "general": GENERAL_RESPONSE_SYSTEM.strip(),
                "mixed": MIXED_RESPONSE_SYSTEM.strip(),
            },
        )
        row = read_first(DATASET / "response/records/train.jsonl")
        messages = trainer.build_sft_messages("response", row, prompts)
        self.assertEqual(messages[0]["content"], prompts[row["route"]])
        self.assertEqual(json.loads(messages[1]["content"]), row["input"])

    def test_full_mode_snapshot_name_is_epoch_based(self):
        original_mode = trainer.MODE
        trainer.MODE = "full"
        try:
            self.assertEqual(trainer.adapter_snapshot_name(3.0, 489), "epoch_3_adapter")
        finally:
            trainer.MODE = original_mode

    def test_response_resume_uses_global_step_with_decision_offset(self):
        original_root = trainer.CHECKPOINT_DIR
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trainer.CHECKPOINT_DIR = root
            checkpoint = root / "response" / "checkpoint-57"
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
                json.dumps({"global_step": 57}), encoding="utf-8"
            )
            local_step = trainer.validate_resume_checkpoint(
                checkpoint,
                expected_global_step=1035,
                stage="response",
                stage_offset=978,
            )
            self.assertEqual(local_step, 57)
            trainer.publish_resume(checkpoint, 1035)
            resume = json.loads(
                (root / ".lab-monitor/resume.json").read_text(encoding="utf-8")
            )
            self.assertEqual(resume["checkpoint"], "response/checkpoint-57")
            self.assertEqual(resume["global_step"], 1035)
        trainer.CHECKPOINT_DIR = original_root


if __name__ == "__main__":
    unittest.main()
