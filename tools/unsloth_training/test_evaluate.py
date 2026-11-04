"""CPU checks binding evaluation to the actual completed Unsloth training."""
import json
from pathlib import Path
import sys
import types
import unittest

import evaluate


class NativeEvaluationControls(unittest.TestCase):
    def test_native_compute_loss_is_exact_production_method(self):
        self.assertTrue(evaluate.exact_loss_method())

    def test_loader_packages_and_original_compiled_source_match_both_training_runs(self):
        identities = evaluate.contract()
        for agent in ("response", "recommendation"):
            self.assertEqual(identities[agent]["identity"]["engine"], "unsloth")
            self.assertEqual(identities[agent]["identity"]["unsloth_return_logits"], 1)
            self.assertEqual(identities[agent]["identity"]["seed"], 42)

    def test_validation_freeze_preserves_serialized_model_input_and_gold_byte_order(self):
        folder = evaluate.ROOT / "training/learning_first_v43/final/goals/GL10/qp_unsloth_validation_20261103_r3/inputs"
        evaluate.verify_frozen(folder)
        frozen = evaluate.b.read(folder / "VALIDATION_ORIGINAL.json")["rows"]
        dump = lambda v: json.dumps(v, ensure_ascii=False, separators=(",", ":"))
        for agent in ("response", "recommendation"):
            path = evaluate.production.DATASET / f"records/{agent}/validation.jsonl"
            with path.open() as stream:
                source = {r["id"]: r for r in map(json.loads, stream)}
            for record in (c["record"] for c in frozen if c["agent"] == agent):
                for field in ("input", "target"):
                    self.assertEqual(dump(record[field]), dump(source[record["id"]][field]))

    def test_cpu_preflight_does_not_import_torch_before_unsloth(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)
        self.assertNotIn("peft", sys.modules)

    def test_generation_prompts_equal_sealed_training_prompts(self):
        from soomac_irc.agent_prompts import (TASK_RESPONSE_SYSTEM, GENERAL_RESPONSE_SYSTEM,
                                             MIXED_RESPONSE_SYSTEM, RECOMMENDATION_SYSTEM)
        root = evaluate.production.DATASET / "contract"
        self.assertEqual(evaluate.b.read(root / "response_system_by_route.json"),
                         {"task": TASK_RESPONSE_SYSTEM, "general": GENERAL_RESPONSE_SYSTEM,
                          "mixed": MIXED_RESPONSE_SYSTEM})
        self.assertEqual((root / "recommendation_system.txt").read_text(), RECOMMENDATION_SYSTEM)

    def test_precision_metadata_only_reads_new_api(self):
        class NewAPIOnly:
            @property
            def fp32_precision(self):
                return "tf32"

            @property
            def allow_tf32(self):
                raise AssertionError("Legacy TF32 query prohibited")

        fake = types.SimpleNamespace(backends=types.SimpleNamespace(
            cuda=types.SimpleNamespace(matmul=NewAPIOnly())))
        self.assertEqual(evaluate.precision_snapshot(fake), {"matmul_fp32_precision": "tf32"})


if __name__ == "__main__":
    unittest.main()
