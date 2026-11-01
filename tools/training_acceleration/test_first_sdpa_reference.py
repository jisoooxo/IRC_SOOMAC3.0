"""CPU-only actual-mask reference tests; run with the isolated torch environment."""
import importlib.util
from pathlib import Path
import unittest

import torch

spec = importlib.util.spec_from_file_location("ablation", Path(__file__).with_name("attention_ablation.py"))
ablation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ablation)


class ActualMaskReference(unittest.TestCase):
    def test_none_without_causal_is_not_triangular(self):
        bias = ablation._sdpa_bias_rows(torch, None, [0, 3], 4, 4, 2, False)
        self.assertTrue(torch.equal(bias, torch.zeros(2, 2, 4)))

    def test_none_with_causal_uses_original_query_indices(self):
        bias = ablation._sdpa_bias_rows(torch, None, [0, 3], 4, 4, 2, True)
        self.assertEqual(torch.isfinite(bias[0]).sum(-1).tolist(), [1, 4])

    def test_boolean_mask_true_means_keep(self):
        mask = torch.eye(4, dtype=torch.bool)[None, None]
        bias = ablation._sdpa_bias_rows(torch, mask, [1, 3], 4, 4, 3, False)
        self.assertEqual(tuple(bias.shape), (3, 2, 4))
        self.assertTrue(torch.equal(torch.isfinite(bias[0]), mask[0, 0, [1, 3]]))

    def test_finite_float_bias_is_not_binarized(self):
        mask = torch.tensor([[0., -0.25, -float("inf"), -3.]])
        bias = ablation._sdpa_bias_rows(torch, mask, [1, 3], 4, 4, 2, False)
        self.assertTrue(torch.equal(bias[0, 0], mask[0]))
        self.assertTrue(torch.equal(bias[1, 1], mask[0]))

    def test_probe_reductions(self):
        x = torch.tensor([1., 2., 3.])
        same = ablation._probe_diff(torch, x, x)
        scaled = ablation._probe_diff(torch, x, 2*x)
        self.assertAlmostEqual(same["cosine"], 1.)
        self.assertEqual(same["relative_l2_error"], 0.)
        self.assertAlmostEqual(scaled["relative_l2_error"], 1.)
        self.assertFalse(torch.cuda.is_initialized())

    def test_bypassed_sdpa_stops_at_layer0_attention(self):
        attention = torch.nn.Identity()
        attention.layer_idx, attention.head_dim = 0, 256
        next_layer = torch.nn.Identity()
        model = torch.nn.Sequential(attention, next_layer)
        reached = []
        next_handle = next_layer.register_forward_pre_hook(lambda *args: reached.append(True))
        guard = ablation._first_sdpa_stop_guard(model)
        try:
            with self.assertRaisesRegex(RuntimeError, "bypassed first SDPA"):
                model(torch.zeros(1))
            self.assertEqual(reached, [])
        finally:
            guard.remove()
            next_handle.remove()


if __name__ == "__main__":
    unittest.main()
