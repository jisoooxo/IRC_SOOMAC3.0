"""CPU-only contract tests. Never import Unsloth or initialize CUDA here."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import production
import launch


class ProductionCpuTests(unittest.TestCase):
    def test_full_training_refuses_step_and_failure_overrides(self):
        base = ["--mode", "train", "--agent", "response", "--output", "/tmp/x",
                "--batch", "1", "--benchmark-result", "/tmp/b/RESULT.json",
                "--smoke-output", "/tmp/s", "--loss-evidence", "/tmp/loss.json"]
        with self.assertRaises(SystemExit):
            production.parse(base + ["--smoke-max-steps", "5"])
        with self.assertRaises(SystemExit):
            production.parse(base + ["--test-fail-step", "5"])

    def test_smoke_failure_only_at_committed_step(self):
        base = ["--mode", "smoke", "--agent", "response", "--output", "/tmp/x",
                "--batch", "1", "--benchmark-result", "/tmp/b/RESULT.json"]
        with self.assertRaises(SystemExit):
            production.parse(base + ["--smoke-max-steps", "5", "--test-fail-step", "10"])
        args = production.parse(base + ["--smoke-max-steps", "10", "--test-fail-step", "5"])
        self.assertEqual((args.smoke_max_steps, args.test_fail_step), (10, 5))

    def test_unmeasured_microbatch_rejected(self):
        with self.assertRaises(SystemExit):
            production.parse(["--mode", "smoke", "--agent", "decision",
                              "--output", "/tmp/unused", "--batch", "2"])

    def test_native_microbatch_mean_override_is_explicit_and_bounded(self):
        args = SimpleNamespace(per_device_train_batch_size=1, gradient_accumulation_steps=16,
                               n_gpu=1, world_size=1, average_tokens_across_devices=False)
        trainer = SimpleNamespace(args=args, compute_loss_func=None, label_smoother=None,
                                  model_accepts_loss_kwargs=True)
        config = {"batch": 1, "accumulation": 16}
        production.enforce_microbatch_mean(trainer, config)
        self.assertIs(trainer.model_accepts_loss_kwargs, False)
        trainer.model_accepts_loss_kwargs = True
        args.average_tokens_across_devices = True
        with self.assertRaisesRegex(RuntimeError, "prerequisites changed"):
            production.enforce_microbatch_mean(trainer, config)
        args.average_tokens_across_devices = False
        with self.assertRaisesRegex(RuntimeError, "prerequisites changed"):
            production.enforce_microbatch_mean(trainer, {"batch": 2, "accumulation": 8})
        trainer.model_accepts_loss_kwargs = "unknown"
        with self.assertRaisesRegex(RuntimeError, "prerequisites changed"):
            production.enforce_microbatch_mean(trainer, config)

    def test_actual_hf_trainer_cpu_accumulates_sixteen_row_means(self):
        """Run the HF Trainer CPU path; candidate image runs this same test."""
        import torch
        from transformers import Trainer, TrainingArguments

        class TinyCE(torch.nn.Module):
            accepts_loss_kwargs = True  # Reproduce the candidate's observed flag.

            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.3))
                self.passed_item_counts = []

            def forward(self, input_ids, labels, **kwargs):
                self.passed_item_counts.append(kwargs.get("num_items_in_batch"))
                logits = torch.stack((torch.zeros_like(input_ids, dtype=torch.float32),
                                      self.weight.expand_as(input_ids)), dim=-1)
                return {"loss": torch.nn.functional.cross_entropy(
                    logits.reshape(-1, 2), labels.reshape(-1))}

        with tempfile.TemporaryDirectory() as temp:
            model = TinyCE()
            args = TrainingArguments(output_dir=temp, use_cpu=True, report_to="none",
                                     per_device_train_batch_size=1,
                                     gradient_accumulation_steps=16,
                                     average_tokens_across_devices=False)
            trainer = Trainer(model=model, args=args)
            self.assertIs(trainer.model_accepts_loss_kwargs, True)
            trainer.model_accepts_loss_kwargs = False
            trainer.current_gradient_accumulation_steps = 16
            reference_weight = torch.tensor(0.3, requires_grad=True)
            reference_loss = 0
            for count in range(1, 17):
                labels = torch.tensor([[0] * (count - 1) + [1]])
                inputs = {"input_ids": torch.zeros((1, count), dtype=torch.long),
                          "labels": labels}
                trainer.training_step(model, inputs, num_items_in_batch=torch.tensor(136))
                logits = torch.stack((torch.zeros(count), reference_weight.expand(count)), dim=-1)
                reference_loss = reference_loss + torch.nn.functional.cross_entropy(
                    logits, labels.reshape(-1)) / 16
            reference_loss.backward()
            self.assertEqual(model.passed_item_counts, [None] * 16)
            self.assertAlmostEqual(model.weight.grad.item(), reference_weight.grad.item(), places=6)

    def test_pinned_configuration(self):
        for agent, target, cap, rate in (("decision", 1638, 9216, 1e-4),
                                         ("response", 624, 9216, 5e-5),
                                         ("recommendation", 234, 5120, 1e-4)):
            args = production.parse(["--mode", "train", "--agent", agent,
                                     "--output", "/tmp/unused", "--batch", "1",
                                     ] + (["--validation-receipt", "/tmp/receipt.json"]
                                          if agent == "decision" else
                                          ["--benchmark-result", "/tmp/result.json",
                                           "--smoke-output", "/tmp/smoke",
                                           "--loss-evidence", "/tmp/loss.json"])
                                    + (["--response-output", "/tmp/verified-p"]
                                       if agent == "recommendation" else []))
            with patch.object(production, "sha", return_value="a" * 64), patch.object(
                    production, "versions", return_value={"unsloth": "2026.10.11"}):
                cfg = production.configuration(args, "b" * 64)
            self.assertEqual((cfg["target_steps"], cfg["max_length"], cfg["learning_rate"]),
                             (target, cap, rate))
            self.assertEqual((cfg["batch"], cfg["accumulation"], cfg["effective_batch"]),
                             (1, 16, 16))
            self.assertEqual((cfg["epochs"], cfg["gradient_checkpointing"],
                              cfg["requested_budget_bytes"], cfg["non_allocator_reserve_bytes"]),
                             (6, "unsloth", 32_000_000_000, 512 * 1024**2))
            self.assertEqual(cfg["loss_weighting"], "microbatch_mean_divide_by_accumulation")

    def test_fresh_and_resume_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run"
            args = production.parse(["--mode", "smoke", "--agent", "response",
                                     "--output", str(path), "--batch", "1",
                                     "--benchmark-result", "/tmp/result.json"])
            config = {"engine": "unsloth"}
            self.assertEqual(production.resume_source(args, config), (None, 0))
            path.mkdir()
            with self.assertRaises(FileExistsError):
                production.resume_source(args, config)
            args.resume = "auto"
            with self.assertRaises(ValueError):
                production.resume_source(args, config)
            (path / "INPUT_IDENTITY.json").write_text(json.dumps({"engine": "baseline"}))
            (path / ".lab-monitor").mkdir()
            (path / ".lab-monitor/resume.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                production.resume_source(args, config)

    def test_container_first_run_accepts_only_empty_mount(self):
        with tempfile.TemporaryDirectory() as temp:
            mount = Path(temp) / "checkpoints"
            mount.mkdir()
            args = SimpleNamespace(output=mount, resume="none", execution_environment="container")
            production.prepare_output(args)
            self.assertTrue(args.fresh_output_prevalidated)
            self.assertEqual(production.resume_source(args, {}), (None, 0))
            (mount / ".lab-monitor").mkdir()
            production.prepare_output(args)
            (mount / ".lab-monitor/progress.json").write_text("{}")
            with self.assertRaises(FileExistsError):
                production.prepare_output(args)
            (mount / ".lab-monitor/progress.json").unlink()
            (mount / "INPUT_IDENTITY.json").write_text("{}")
            with self.assertRaises(FileExistsError):
                production.prepare_output(args)
            local = SimpleNamespace(output=mount, resume="none", execution_environment="local")
            with self.assertRaises(FileExistsError):
                production.prepare_output(local)

    def test_heartbeat_precedes_release_verification(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "fresh"
            observed = []

            def fake_preflight(agent):
                observed.append((output / ".lab-monitor/progress.json").is_file())
                return {}, {"counts": {agent: {"train": 1664, "validation": 152}},
                            "max_length": {agent: 9216}}, "a" * 64

            with patch.object(production.sys, "prefix", str(production.DURABLE_VENV)), \
                    patch.object(production, "configure_environment"), \
                    patch.object(production, "source_preflight", side_effect=fake_preflight), \
                    patch.object(production, "configuration", return_value={}):
                production.main(["--mode", "preflight", "--agent", "response",
                                 "--output", str(output), "--batch", "1"])
            self.assertEqual(observed, [True])

    def test_smoke_receipts_select_split_specific_longest_without_history_change(self):
        with tempfile.TemporaryDirectory() as temp:
            release = Path(temp)
            receipt_path = release / "validation/TOKEN_MASK_ROWS.jsonl"
            receipt_path.parent.mkdir()
            receipts = []
            for split, count in (("train", 18), ("validation", 10)):
                for index in range(count):
                    receipts.append({"agent": "response", "split": split,
                                     "id": f"{split}-{index:02d}", "full_tokens": 100 + index,
                                     "input_tokens": 90 + index, "target_tokens": 10,
                                     "history_trimmed": False, "errors": [],
                                     "builder_equal": True, "target_mask_equal": True})
            receipt_path.write_text("".join(json.dumps(item) + "\n" for item in receipts))
            manifest = {"token_receipts_sha256": production.sha(receipt_path)}
            with patch.object(production, "RELEASE", release):
                selected = production.smoke_selection("response", manifest)
            self.assertEqual(len(selected["train"]), 16)
            self.assertEqual(len(selected["validation"]), 8)
            self.assertEqual(selected["train"][0], "train-17")
            self.assertEqual(selected["validation"][0], "validation-09")
            raw = [{"id": item["id"], "history": ["unaltered"], "prompt": "x"}
                   for item in receipts if item["split"] == "train"]
            subset = production.select_smoke_raw(raw, "train", selected)
            self.assertEqual([item["id"] for item in subset], selected["train"])
            self.assertTrue(all(item["history"] == ["unaltered"] for item in subset))
            self.assertEqual(len(raw), 18)
            with patch.object(production, "RELEASE", release):
                self.assertEqual(production.smoke_selection("response", manifest), selected)

    def test_output_path_traversal_and_source_overlap_rejected(self):
        with self.assertRaisesRegex(ValueError, "traversal"):
            production.validate_output(Path("/tmp/../home/roma/IRC_SOOMAC3.0/tools/unsloth_training/output"))
        with self.assertRaisesRegex(ValueError, "overlaps"):
            production.validate_output(production.HERE / "output")

    def test_benchmark_gate_rejects_mismatch(self):
        item = {"status": "TIMING_PASS_LOSS_EQUIVALENCE_UNVERIFIED", "engine": "unsloth",
                "agent": "recommendation", "production_training": False,
                "release_sha256": "a" * 64,
                "source_manifest_sha256": production.SOURCE_SHA,
                "model_config_sha256": "a" * 64,
                "processor_tokenizer_sha256": "a" * 64,
                "model_builder_sha256": "b" * 64,
                "packages": {"unsloth": "2026.10.11"},
                "limit_bytes": 32_000_000_000,
                "non_allocator_reserve_bytes": 512 * 1024**2,
                "unsloth_return_logits": 1,
                "effective_batch": 16, "trainable_count": production.benchmark.TRAINABLE_COUNT,
                "samples_per_second": 0.4}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "RESULT.json"
            path.write_text(json.dumps(item))
            with patch.object(production, "sha", return_value="a" * 64), \
                    patch.object(production, "versions", return_value={"unsloth": "2026.10.11"}), \
                    patch.object(production.benchmark, "model_builder_fingerprint", return_value="b" * 64):
                self.assertEqual(len(production.verify_benchmark_result(path, "recommendation")), 64)
                with self.assertRaises(ValueError):
                    production.verify_benchmark_result(path, "response")
                item["samples_per_second"] = 0
                path.write_text(json.dumps(item))
                with self.assertRaises(ValueError):
                    production.verify_benchmark_result(path, "recommendation")

    def test_decision_train_requires_receipt_but_smoke_does_not(self):
        base = ["--agent", "decision", "--output", "/tmp/d", "--batch", "1",
                "--execution-environment", "container", "--release", "/app/RELEASE.json",
                "--dataset", "/app/assets/dataset", "--model", "/model"]
        with self.assertRaises(SystemExit):
            production.parse(["--mode", "train"] + base)
        smoke = production.parse(["--mode", "smoke"] + base)
        self.assertEqual(smoke.non_allocator_reserve_bytes, 1024**3)
        self.assertEqual(smoke.dataset, Path("/app/assets/dataset"))
        train = production.parse(["--mode", "train"] + base
                                 + ["--validation-receipt", "/app/validation/VALIDATION_RECEIPT.json"])
        self.assertEqual(train.agent, "decision")

    def test_decision_receipt_checks_source_and_bundled_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            files = {}
            for name in ("loss", "gradient", "restore"):
                file = root / f"{name}.json"
                file.write_text(json.dumps({"proof": name}))
                files[name] = {"path": file.name, "sha256": production.sha(file)}
            config = {key: "a" * 64 for key in
                      ("release_sha256", "source_manifest_sha256", "model_config_sha256",
                       "processor_tokenizer_sha256", "production_sha256", "model_builder_sha256")}
            config["packages"] = {"unsloth": "2026.10.11"}
            item = {"schema_version": 1, "status": "PASS", "agent": "decision",
                    **config, "remote_speed_status": "unmeasured",
                    "tested_candidate_image_digest": "sha256:" + "b" * 64,
                    "tested_process_budget_bytes": 32_000_000_000,
                    "tested_process_peak_bytes": 24_000_000_000,
                    "min_non_allocator_reserve_bytes": 1024**3,
                    "loss_validation": {"one_eval_row_native1_selected_ce_equal": True,
                                        "selected_labels_equal": True, "fused0_accepted": False,
                                        "training_mode_dropout_equivalence_verified": False,
                                        "compiled_source_sha256": "c" * 64},
                    "gradient_validation": {"finite_nonzero": True},
                    "restore_validation": {key: True for key in
                                           ("adapter", "optimizer", "scheduler", "rng",
                                            "forced_failure_resume")},
                    "evidence_files": files,
                    "remote_hardware_fit_status": "unmeasured"}
            path = root / "VALIDATION_RECEIPT.json"
            path.write_text(json.dumps(item))
            args = SimpleNamespace(budget_bytes=44_000_000_000,
                                   non_allocator_reserve_bytes=1024**3)
            self.assertEqual(len(production.verify_validation_receipt(path, args, config)), 64)
            item["loss_validation"]["fused0_accepted"] = True
            path.write_text(json.dumps(item))
            with self.assertRaises(ValueError):
                production.verify_validation_receipt(path, args, config)

    def test_launcher_never_runs_q_without_verified_p(self):
        argv = ["--response-output", "/tmp/unfinished-p",
                "--recommendation-output", "/tmp/new-q",
                "--recommendation-benchmark-result", "/tmp/q-result.json",
                "--recommendation-smoke-output", "/tmp/q-smoke",
                "--recommendation-loss-evidence", "/tmp/q-loss.json",
                "--recommendation-batch", "1", "--wait-response-unit", "soomac-unsloth-response-test.service",
                "--recommendation-preflight-root", "/tmp/q-preflight", "--recommendation-prepared", "/tmp/prepared"]
        with patch.object(launch, "wait_response_service"), \
                patch.object(launch, "verified_response", side_effect=ValueError("P not complete")), \
                patch.object(launch.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "P not complete"):
                launch.main(argv)
            run.assert_not_called()

    def test_direct_q_train_requires_response_prerequisite(self):
        base = ["--mode", "train", "--agent", "recommendation", "--output", "/tmp/q",
                "--batch", "1", "--benchmark-result", "/tmp/q-result.json",
                "--smoke-output", "/tmp/q-smoke", "--loss-evidence", "/tmp/q-loss.json"]
        with self.assertRaises(SystemExit):
            production.parse(base)
        args = production.parse(base + ["--response-output", "/tmp/verified-p"])
        self.assertEqual(args.response_output, Path("/tmp/verified-p"))

    def test_queue_waits_for_successful_response_exit(self):
        running = "LoadState=loaded\nActiveState=active\nMainPID=42\nResult=success\nExecMainStatus=0\n"
        done = "LoadState=loaded\nActiveState=inactive\nMainPID=0\nResult=success\nExecMainStatus=0\n"
        with patch.object(launch.subprocess, "check_output", side_effect=[running, done]), \
                patch.object(launch.time, "sleep") as sleep:
            launch.wait_response_service("soomac-unsloth-response-test.service", Path("/tmp/no-receipt"))
            sleep.assert_called_once_with(60)

    def test_queue_rejects_failed_or_missing_response_service(self):
        for state in ["LoadState=loaded\nActiveState=failed\nMainPID=0\nResult=exit-code\nExecMainStatus=1\n",
                      "LoadState=not-found\nActiveState=inactive\n"]:
            with tempfile.TemporaryDirectory() as temp, patch.object(launch.subprocess, "check_output", return_value=state):
                with self.assertRaises(RuntimeError):
                    launch.wait_response_service("soomac-unsloth-response-test.service", Path(temp))

    def test_collected_service_rejected_even_with_completion_file(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / "training_complete.json").write_text("{}")
            argv = ["--response-output", temp, "--recommendation-output", "/tmp/new-q",
                    "--recommendation-benchmark-result", "/tmp/q-result.json",
                    "--recommendation-smoke-output", "/tmp/q-smoke",
                    "--recommendation-loss-evidence", "/tmp/q-loss.json", "--recommendation-batch", "1",
                    "--wait-response-unit", "soomac-unsloth-response-test.service",
                    "--recommendation-preflight-root", "/tmp/q-preflight", "--recommendation-prepared", "/tmp/prepared"]
            with patch.object(launch.subprocess, "check_output", return_value="LoadState=not-found\n"), \
                    patch.object(launch, "verified_response") as verify, \
                    patch.object(launch.subprocess, "run") as run:
                with self.assertRaisesRegex(RuntimeError, "service not found"):
                    launch.main(argv)
                run.assert_not_called()
                verify.assert_not_called()

    def test_queue_does_not_run_q_when_preflight_fails(self):
        argv = ["--response-output", "/tmp/p", "--recommendation-output", "/tmp/new-q",
                "--recommendation-benchmark-result", "/tmp/q-result.json",
                "--recommendation-smoke-output", "/tmp/q-preflight/smoke",
                "--recommendation-loss-evidence", "/tmp/q-loss.json", "--recommendation-batch", "1",
                "--recommendation-preflight-root", "/tmp/q-preflight", "--recommendation-prepared", "/tmp/prepared",
                "--wait-response-unit", "soomac-unsloth-response-test.service"]
        with patch.object(launch, "wait_response_service"), patch.object(launch, "verified_response"), patch.object(production, "verify_benchmark_result"), \
                patch.object(launch, "prepare_recommendation", side_effect=ValueError("Q check failed")), \
                patch.object(launch.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "Q check failed"):
                launch.main(argv)
            run.assert_not_called()

    def test_queue_rejects_unrelated_service(self):
        with patch.object(launch.subprocess, "check_output") as read:
            with self.assertRaises(ValueError):
                launch.wait_response_service("unrelated.service", Path("/tmp/p"))
            read.assert_not_called()

    def test_queue_requires_exit_and_all_preflight_arguments(self):
        base = ["--response-output", "/tmp/p", "--recommendation-output", "/tmp/new-q",
                "--recommendation-benchmark-result", "/tmp/q-result.json",
                "--recommendation-smoke-output", "/tmp/q-preflight/smoke",
                "--recommendation-loss-evidence", "/tmp/q-loss.json", "--recommendation-batch", "1"]
        flags = ["--wait-response-unit", "soomac-unsloth-response-test.service",
                 "--recommendation-preflight-root", "/tmp/q-preflight", "--recommendation-prepared", "/tmp/prepared"]
        for omitted in range(0, len(flags), 2):
            with patch.object(launch.subprocess, "run") as run:
                with self.assertRaises(SystemExit):
                    launch.main(base + flags[:omitted] + flags[omitted+2:])
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
