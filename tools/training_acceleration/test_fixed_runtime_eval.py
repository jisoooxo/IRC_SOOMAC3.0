"""CPU checks for the fresh evaluation's precision and causal target contract."""
import contextlib
import copy
import types
import unittest
from unittest.mock import patch

import eval_fixed_runtime as run


class FixedRuntimeControls(unittest.TestCase):
    def test_ce_disables_autocast_and_scores_causal_prediction_positions(self):
        import torch
        item = {"input_ids": [8, 7, 1, 2], "labels": [-100, -100, 1, 2], "length": 4}
        original = copy.deepcopy(item)
        observed = []

        def collate(rows):
            return {"input_ids": torch.tensor([rows[0]["input_ids"]]),
                    "labels": torch.tensor([rows[0]["labels"]]),
                    "attention_mask": torch.ones((1, 4), dtype=torch.long)}

        class Model:
            device = "cpu"

            def __call__(self, **kwargs):
                observed.extend(kwargs["logits_to_keep"].tolist())
                return types.SimpleNamespace(logits=torch.tensor([[[0., 9., 0.], [0., 0., 9.]]]))

        with patch.object(torch, "autocast", return_value=contextlib.nullcontext()) as autocast:
            result = run.score_ce(Model(), collate, item, torch)
        autocast.assert_called_once_with("cuda", enabled=False)
        self.assertEqual(observed, [1, 2])
        self.assertEqual(result["target_tokens"], 2)
        self.assertLess(result["ce"], .001)
        self.assertEqual(item, original)
        self.assertFalse(torch.cuda.is_initialized())

    def test_repeat_gate_requires_raw_output_and_actual_messages_not_just_parsed_json(self):
        baseline = {"prediction": {"value": 1}, "traces": [{"raw": '{"value":1}', "messages": ["a"]}]}
        changed = copy.deepcopy(baseline)
        self.assertTrue(run.same_prediction(baseline, changed))
        changed["traces"][-1]["messages"] = ["b"]
        self.assertFalse(run.same_prediction(baseline, changed))
        changed = copy.deepcopy(baseline)
        changed["traces"][-1]["raw"] = '{"value": 1}'
        self.assertFalse(run.same_prediction(baseline, changed))

    def test_one_explicit_precision_policy_applies_to_all_candidates(self):
        self.assertFalse(run.POLICY["autocast_enabled"])
        self.assertFalse(run.POLICY["matmul_allow_tf32"])
        self.assertEqual(run.POLICY["bnb_compute_dtype"], "bfloat16")
        self.assertEqual(run.POLICY["adapter_storage_dtype"], "float32")
        self.assertTrue(run.POLICY["new_process_per_checkpoint"])


if __name__ == "__main__":
    unittest.main()
