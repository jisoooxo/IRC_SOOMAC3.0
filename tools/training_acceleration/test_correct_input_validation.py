"""CPU controls for evaluation-copy isolation and causal target selection."""
import contextlib
import copy
import json
from pathlib import Path
import types
import unittest
from unittest.mock import patch

import correct_input_validation as review
import eval_correct_input_epochs as run


class ReviewControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = review.benchmark.RELEASE / "dataset/records/recommendation/validation.jsonl"
        with path.open() as stream:
            cls.rows = {r["id"]: r for r in map(json.loads, stream)}

    def test_only_documented_gold_amount_changes_input_and_source_stay_exact(self):
        original = copy.deepcopy(self.rows["REC1-S0909"])
        snapshot = copy.deepcopy(original)
        fixed, _ = review.reviewed_row("recommendation", original)
        self.assertEqual(original, snapshot)
        self.assertEqual(fixed["input"], snapshot["input"])
        expected = copy.deepcopy(snapshot)
        expected["target"]["proposal"]["toppings"]["페퍼론치노"] = "normal"
        self.assertEqual(fixed, expected)
        unchanged, _ = review.reviewed_row("recommendation", self.rows["REC1-S0910"])
        self.assertEqual(unchanged, self.rows["REC1-S0910"])

    def test_candidate_pending_evidence_prevents_blind_correction(self):
        row = copy.deepcopy(self.rows["REC1-S0909"])
        row["input"]["pending"] = {"type": "recommendation", "candidate": {"toppings": {"페퍼론치노": "low"}}}
        with self.assertRaises(AssertionError):
            review.reviewed_row("recommendation", row)

    def test_invalid_derived_scope_is_rejected_before_inference(self):
        row = copy.deepcopy(self.rows["REC1-S0910"])
        row["input"]["recommendation_scope"]["current_section"] = "wrong"
        with self.assertRaisesRegex(ValueError, "runtime input drift"):
            review.rebuild("recommendation", row)

    def test_ce_scores_prediction_positions_not_label_positions(self):
        import torch
        encoded = {"input_ids": [8, 7, 1, 2], "labels": [-100, -100, 1, 2], "length": 4}
        original = copy.deepcopy(encoded)
        observed = []

        def collator(rows):
            return {"input_ids": torch.tensor([rows[0]["input_ids"]]),
                    "labels": torch.tensor([rows[0]["labels"]]),
                    "attention_mask": torch.ones((1, 4), dtype=torch.long)}

        class Model:
            device = "cpu"

            def __call__(self, **kw):
                observed.extend(kw["logits_to_keep"].tolist())
                self_outer.assertFalse(kw["use_cache"])
                self_outer.assertNotIn("labels", kw)
                return types.SimpleNamespace(logits=torch.tensor([[[0., 9., 0.], [0., 0., 9.]]]))

        self_outer = self
        with patch.object(torch, "autocast", return_value=contextlib.nullcontext()):
            result = run.ce_one(Model(), collator, encoded, torch)
        self.assertEqual(observed, [1, 2])
        self.assertEqual(result["target_tokens"], 2)
        self.assertLess(result["ce"], 0.001)
        self.assertEqual(encoded, original)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
