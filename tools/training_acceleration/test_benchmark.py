"""CPU-only checks; importing benchmark never imports CUDA or Unsloth."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).with_name("benchmark.py")
spec = importlib.util.spec_from_file_location("training_acceleration_benchmark", SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class BenchmarkCPU(unittest.TestCase):
    def test_constants_and_receipt_selection(self):
        self.assertEqual(benchmark.PROCESS_BUDGET, 32_000_000_000)
        self.assertEqual(benchmark.RESERVE, 512 * 1024**2)
        self.assertEqual(benchmark.EFFECTIVE_BATCH, 16)
        for agent, cap in (("response", 9216), ("recommendation", 5120)):
            _, manifest, receipts = benchmark.source_identity(agent)
            self.assertEqual(manifest["max_length"][agent], cap)
            selected = benchmark.selected_receipts(agent, receipts)
            self.assertEqual(len(selected), 16)
            self.assertEqual(selected[0]["full_tokens"], max(row["full_tokens"] for row in selected))
            self.assertTrue(all(row["full_tokens"] <= cap and not row["history_trimmed"] for row in selected))
            found = benchmark.load_selected_rows(agent, selected)
            self.assertEqual(len(found), 16)

    def test_bad_receipt_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipts.jsonl"
            path.write_text(json.dumps({"agent": "response", "split": "train", "id": "x",
                                        "full_tokens": 2, "history_trimmed": True,
                                        "errors": [], "builder_equal": True,
                                        "target_mask_equal": True}) + "\n")
            with self.assertRaisesRegex(ValueError, "invalid token receipt"):
                benchmark.selected_receipts("response", path)

    def test_output_is_new_and_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fresh"
            self.assertEqual(benchmark.output_dir(path), path)
            with self.assertRaises(FileExistsError):
                benchmark.output_dir(path)
            with self.assertRaisesRegex(ValueError, "overlaps"):
                benchmark.output_dir(benchmark.RELEASE / "accidental")

    def test_cli_requires_separate_engine_and_artifact(self):
        with self.assertRaises(SystemExit):
            benchmark.parse(["run", "--agent", "response", "--output", "/tmp/unused"])
        args = benchmark.parse(["run", "--agent", "response", "--output", "/tmp/unused",
                                "--encoded", "/tmp/encoded.json", "--engine", "baseline"])
        self.assertEqual((args.mode, args.timed_updates), ("smoke", 1))
        args = benchmark.parse(["run", "--agent", "recommendation", "--output", "/tmp/unused",
                                "--encoded", "/tmp/encoded.json", "--engine", "unsloth",
                                "--mode", "full"])
        self.assertEqual(args.timed_updates, 3)
        diagnostic = benchmark.parse(["run", "--agent", "response", "--output", "/tmp/unused",
                                      "--encoded", "/tmp/encoded.json", "--engine", "unsloth",
                                      "--check-loss"])
        self.assertTrue(diagnostic.check_loss)
        self.assertFalse(diagnostic.check_fused)
        self.assertEqual(diagnostic.return_logits, 0)  # run() forces 1 for loss diagnostics.
        fused = benchmark.parse(["run", "--agent", "response", "--output", "/tmp/unused",
                                 "--encoded", "/tmp/encoded.json", "--engine", "unsloth",
                                 "--check-loss", "--check-fused"])
        self.assertTrue(fused.check_fused)
        with self.assertRaises(SystemExit):
            benchmark.parse(["run", "--agent", "response", "--output", "/tmp/unused",
                             "--encoded", "/tmp/encoded.json", "--engine", "unsloth",
                             "--check-fused"])
        with self.assertRaises(SystemExit):
            benchmark.parse(["run", "--agent", "response", "--output", "/tmp/unused",
                             "--encoded", "/tmp/encoded.json", "--engine", "unsloth",
                             "--check-loss", "--mode", "full"])

    def test_gpu_paths_are_not_imported_during_cpu_tests(self):
        with patch.object(benchmark, "acquire_gpu_lock", side_effect=AssertionError("GPU path called")):
            self.assertEqual(benchmark.sha(SCRIPT), benchmark.sha(SCRIPT))

    def test_model_builder_fingerprint_is_stable(self):
        value = benchmark.model_builder_fingerprint()
        self.assertEqual(len(value), 64)
        self.assertEqual(value, benchmark.model_builder_fingerprint(benchmark.RELEASE))

    def test_process_rss_is_process_scoped_linux_high_water(self):
        value = benchmark.process_peak_rss_bytes()
        self.assertIsInstance(value, int)
        self.assertGreater(value, 0)

    def test_native_and_fused_loss_acceptance(self):
        self.assertEqual(benchmark.compare_loss_scalars(0.7945, 0.7945, 0.7946)["status"], "PASS")
        self.assertEqual(benchmark.compare_loss_scalars(0.7945, 0.7945, 0.9)["status"], "DRIFT")
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            benchmark.compare_loss_scalars(0.7945, float("nan"), 0.7945)


if __name__ == "__main__":
    unittest.main()
