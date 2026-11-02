"""CPU-only isolation tests for final adapter vLLM staging."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from safetensors.torch import load_file, save_file
import torch

import stage_vllm_candidates as subject


class StageVllmCandidatesTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.production = self.root / "production"
        self.production.mkdir()
        (self.production / "sentinel").write_text("preserve", encoding="utf-8")
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps({"text_config": {
            "layer_types": ["sliding_attention", "full_attention"]}}), encoding="utf-8")
        self.sources = {}
        for agent in subject.AGENTS:
            source = self.root / f"source_{agent}"
            source.mkdir()
            (source / "adapter_config.json").write_text('{"r": 2}', encoding="utf-8")
            prefix = "base_model.model.model.language_model.layers.1.self_attn.k_proj"
            save_file({f"{prefix}.lora_A.weight": torch.ones((2, 4)),
                       f"{prefix}.lora_B.weight": torch.ones((4, 2))},
                      str(source / "adapter_model.safetensors"))
            self.sources[agent] = source

    def test_isolated_output_uses_existing_conversion_and_preserves_sources(self):
        before = {agent: subject.sha(path / "adapter_model.safetensors")
                  for agent, path in self.sources.items()}
        output = self.root / "fresh"
        with mock.patch.object(subject.converter, "OUT", self.production):
            receipt = subject.stage(output, self.sources, self.config)
            self.assertEqual(subject.converter.OUT, self.production)
        self.assertEqual((self.production / "sentinel").read_text(), "preserve")
        self.assertEqual(sorted(p.name for p in self.production.iterdir()), ["sentinel"])
        self.assertEqual(receipt["status"], "ISOLATED_VLLM_STAGE_NOT_MODEL_LOAD")
        self.assertEqual(receipt["global_layers"], [1])
        for agent, source in self.sources.items():
            self.assertEqual(subject.sha(source / "adapter_model.safetensors"), before[agent])
            tensors = load_file(str(output / agent / "adapter_model.safetensors"))
            self.assertEqual(len(tensors), 4)
            original_k = next(key for key in tensors if "k_proj" in key and "lora_A" in key)
            self.assertTrue(torch.equal(tensors[original_k],
                                        tensors[original_k.replace("k_proj", "v_proj")]))
            self.assertEqual(json.loads((output / agent / "CONVERSION.json").read_text())
                             ["added_v_proj_tensors"], 2)

    def test_rejects_production_root_and_existing_destination(self):
        with mock.patch.object(subject.converter, "OUT", self.production):
            with self.assertRaises(ValueError):
                subject.stage(self.production, self.sources, self.config)
            existing = self.root / "existing"
            existing.mkdir()
            with self.assertRaises(ValueError):
                subject.stage(existing, self.sources, self.config)
        self.assertEqual((self.production / "sentinel").read_text(), "preserve")

    def test_rejects_converted_adapter_as_source(self):
        (self.sources["response"] / "CONVERSION.json").write_text("{}", encoding="utf-8")
        with mock.patch.object(subject.converter, "OUT", self.production):
            with self.assertRaises(ValueError):
                subject.stage(self.root / "fresh", self.sources, self.config)
        self.assertFalse((self.root / "fresh").exists())


if __name__ == "__main__":
    unittest.main()
