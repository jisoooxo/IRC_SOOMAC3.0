"""CPU-only checks; importing attention_ablation never imports torch or Unsloth."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).with_name("attention_ablation.py")
spec = importlib.util.spec_from_file_location("training_acceleration_attention_ablation", SCRIPT)
ablation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ablation)

CONFIG = Path("/home/roma/Desktop/sLLM/gemma-4-12B-it/config.json")


class AblationCPU(unittest.TestCase):
    def test_import_does_not_pull_cuda(self):
        # The whole point of CPU prep: loading this module must not import torch/unsloth,
        # so it can run while root owns the GPU.
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("unsloth", sys.modules)

    def test_global_layer_indices_match_config(self):
        if not CONFIG.exists():
            self.skipTest("base config not present")
        layer_types = json.loads(CONFIG.read_text())["text_config"]["layer_types"]
        full = tuple(i for i, t in enumerate(layer_types) if t == "full_attention")
        self.assertEqual(full, ablation.GLOBAL_LAYER_INDICES)
        self.assertEqual(len(full), 8)
        self.assertEqual(sum(1 for t in layer_types if t == "sliding_attention"), 40)

    def test_source_manifest_records_hashes_and_lines(self):
        manifest = ablation._source_manifest()
        self.assertIn("unsloth_large_head_attention", manifest)
        entry = manifest["unsloth_large_head_attention"]
        self.assertIn("engaged_calls_counter", entry["lines"])
        self.assertEqual(entry["pull_request"],
                         "https://github.com/unslothai/unsloth/pull/13869")
        # If the reviewed file is present, its hash must be recorded (falsifiable drift check).
        if entry["exists"]:
            self.assertEqual(len(entry["sha256"]), 64)

    def test_cli_prepare_rejects_encoded(self):
        with self.assertRaises(SystemExit):
            ablation.parse(["prepare", "--agent", "response", "--output", "/tmp/x",
                            "--encoded", "/tmp/e.json"])
        args = ablation.parse(["prepare", "--agent", "response", "--output", "/tmp/x"])
        self.assertEqual(args.command, "prepare")

    def test_cli_check_and_compare_require_encoded(self):
        with self.assertRaises(SystemExit):
            ablation.parse(["check", "--agent", "response", "--output", "/tmp/x"])
        args = ablation.parse(["compare", "--agent", "response", "--output", "/tmp/x",
                               "--encoded", "/tmp/e.json"])
        self.assertEqual((args.warmup, args.timed_updates), (3, 3))
        with self.assertRaises(SystemExit):
            ablation.parse(["compare", "--agent", "response", "--output", "/tmp/x",
                            "--encoded", "/tmp/e.json", "--warmup", "0"])

    def test_rss_is_positive(self):
        self.assertGreater(ablation._rss_bytes(), 0)

    def test_bounded_profile_flags(self):
        args = ablation.parse(["check", "--agent", "response", "--output", "/tmp/x",
                               "--encoded", "/tmp/e.json", "--profile-only", "--skip-flex"])
        self.assertTrue(args.profile_only)
        self.assertTrue(args.skip_flex)

    def test_padding_gate_cli(self):
        args = ablation.parse(["padding-cpu", "--agent", "response", "--output", "/tmp/x",
                               "--encoded", "/tmp/e.json"])
        self.assertEqual(args.command, "padding-cpu")
        with self.assertRaises(SystemExit):
            ablation.parse(["padding-gpu", "--agent", "response", "--output", "/tmp/x",
                            "--encoded", "/tmp/e.json"])
        args = ablation.parse(["padding-gpu", "--agent", "response", "--output", "/tmp/x",
                               "--encoded", "/tmp/e.json", "--padding-receipt", "/tmp/p.json"])
        self.assertEqual(args.padding_receipt, Path("/tmp/p.json"))
        args = ablation.parse(["padding-diagnose", "--agent", "response", "--output", "/tmp/x",
                               "--encoded", "/tmp/e.json", "--padding-receipt", "/tmp/p.json"])
        self.assertEqual(args.command, "padding-diagnose")
        args = ablation.parse(["first-sdpa", "--agent", "response", "--output", "/tmp/x",
                               "--encoded", "/tmp/e.json", "--padding-receipt", "/tmp/p.json"])
        self.assertEqual(args.command, "first-sdpa")

    def test_prepare_delegates_and_stamps_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            output.mkdir()
            written = {}

            def fake_write_new(path, value):
                written[path.name] = value

            fake_benchmark = type("B", (), {})()
            fake_benchmark.prepare = lambda agent, dest: {"encoded_sha256": "a" * 64}
            fake_benchmark.write_new = staticmethod(fake_write_new)
            args = ablation.parse(["prepare", "--agent", "response", "--output", str(output)])
            result = ablation.prepare(args, fake_benchmark)
            self.assertEqual(result["status"], "CPU_PREPARED_NO_GPU")
            self.assertIn("ABLATION_PREPARE.json", written)
            self.assertEqual(written["ABLATION_PREPARE.json"]["mode"], "CPU_PREPARED_NO_GPU")
            self.assertIn("reviewed_sources", written["ABLATION_PREPARE.json"])

    def test_cli_compare_run_timing_defaults_off(self):
        # The bounded probe is the default; the hour-long timing must be opt-in.
        args = ablation.parse(["compare", "--agent", "response", "--output", "/tmp/x",
                               "--encoded", "/tmp/e.json"])
        self.assertFalse(args.run_timing)
        args = ablation.parse(["compare", "--agent", "response", "--output", "/tmp/x",
                               "--encoded", "/tmp/e.json", "--run-timing"])
        self.assertTrue(args.run_timing)

    def test_set_hd512_toggles_and_verifies(self):
        # Fake the installed module's [None]-list cache + env read, so this CPU test
        # proves _set_hd512 actually flips _enabled() (not a swallowed no-op) and raises
        # if the flip does not take effect.
        import os as os_module
        import types

        pkg = types.ModuleType("unsloth")
        pkg.__path__ = []
        kernels = types.ModuleType("unsloth.kernels")
        kernels.__path__ = []
        lha = types.ModuleType("unsloth.kernels.large_head_attention")
        lha._ENABLED = [None]

        def fake_enabled():
            if lha._ENABLED[0] is None:
                lha._ENABLED[0] = os_module.environ.get(
                    "UNSLOTH_HD512_ATTENTION", "1") != "0"
            return lha._ENABLED[0]

        lha._enabled = fake_enabled
        saved_env = os_module.environ.get("UNSLOTH_HD512_ATTENTION")
        names = ("unsloth", "unsloth.kernels", "unsloth.kernels.large_head_attention")
        saved_mods = {name: sys.modules.get(name) for name in names}
        sys.modules["unsloth"] = pkg
        sys.modules["unsloth.kernels"] = kernels
        sys.modules["unsloth.kernels.large_head_attention"] = lha
        try:
            lha._ENABLED[0] = True  # a stale verdict that a no-op clear would leave live
            self.assertFalse(ablation._set_hd512(False))
            self.assertEqual(os_module.environ["UNSLOTH_HD512_ATTENTION"], "0")
            self.assertTrue(ablation._set_hd512(True))
            self.assertEqual(os_module.environ["UNSLOTH_HD512_ATTENTION"], "1")
        finally:
            for name, module in saved_mods.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
            if saved_env is None:
                os_module.environ.pop("UNSLOTH_HD512_ATTENTION", None)
            else:
                os_module.environ["UNSLOTH_HD512_ATTENTION"] = saved_env

    def test_benchmark_import_is_cpu_safe(self):
        # The sibling module must import without CUDA so reuse stays CPU-safe.
        with patch.object(ablation, "BENCHMARK_PATH", SCRIPT.with_name("benchmark.py")):
            benchmark = ablation._load_benchmark()
        self.assertTrue(hasattr(benchmark, "model_and_optimizer"))
        self.assertTrue(hasattr(benchmark, "one_update"))
        self.assertEqual(benchmark.EFFECTIVE_BATCH, 16)


if __name__ == "__main__":
    unittest.main()
