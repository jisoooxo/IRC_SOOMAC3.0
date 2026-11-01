"""CPU-only static checks for the separate remote Unsloth image package."""
import json
import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest
from .build_final import validate_receipt_files, verify_lineage


HERE = Path(__file__).resolve().parent
RELEASE = Path("/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/releases/conversation_r5_20261101_build05_speed")
MANIFEST_SHA = "7850948287237c5ace9d1b0f56931419760fbce3f0afa02cea40681acb6ceffa"


class RemotePackageCpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dockerfile = (HERE / "Dockerfile").read_text()
        cls.runner = (HERE / "run_training.sh").read_text()

    def test_shell_syntax(self):
        subprocess.run(["sh", "-n", str(HERE / "run_training.sh")], check=True)

    def test_official_immutable_amd64_base(self):
        self.assertRegex(self.dockerfile, r"FROM --platform=linux/amd64 unsloth/unsloth:core-nightly-2026\.10\.28@sha256:[0-9a-f]{64}")
        self.assertNotIn("unsloth/unsloth:latest", self.dockerfile)
        self.assertNotIn("-p 8000", self.dockerfile)

    def test_lab_contract_and_real_embedded_copies(self):
        for text in ('lab.training.spec="2.9"', 'lab.training.progress="1"',
                     'lab.training.resume="output-dir-v1"', 'lab.training.inputs="${LAB_INPUTS}"',
                     'USER 1000:1000', 'ENTRYPOINT ["/app/run_training.sh"]', 'CMD ["train"]',
                     'COPY --chown=1000:1000 bundle/dataset/ /app/assets/dataset/',
                     'COPY --chown=1000:1000 bundle/assets/model-processor/ /app/assets/model-processor/'):
            self.assertIn(text, self.dockerfile)
        self.assertIn('test "$LAB_INPUTS" =', self.dockerfile)

    def test_shared_runner_copy_and_decision_build_gate(self):
        self.assertIn('COPY --chown=1000:1000 source/unsloth_training/production.py', self.dockerfile)
        self.assertIn('COPY --chown=1000:1000 source/training_acceleration/benchmark.py', self.dockerfile)
        self.assertIn('production.py --help > /tmp/soomac-production-help', self.dockerfile)
        self.assertIn('for flag in decision --execution-environment --release --model --dataset --processor --budget-bytes --validation-receipt', self.dockerfile)
        self.assertIn('--agent decision', self.runner)
        self.assertIn('--execution-environment container', self.runner)
        self.assertIn('FROM ${TESTED_CANDIDATE_IMAGE} AS final', self.dockerfile)
        self.assertIn('d.get("tested_candidate_image_digest")==tested', self.dockerfile)
        self.assertIn('COPY --chown=1000:1000 validation/ /app/validation/', self.dockerfile)
        self.assertIn('set(e)=={"loss","gradient","restore"}', self.dockerfile)
        self.assertIn('hexdigest()==x["sha256"]', self.dockerfile)

    def test_resume_and_budget_contract_is_wired_not_claimed_verified(self):
        for text in ('GPU_MEMORY_BUDGET_BYTES', 'GPU_MEMORY_FRACTION', 'LAB_RESUME',
                     'RESUME_CHECKPOINT_DIR', 'RESUME_STEP', '--batch 1',
                     '--non-allocator-reserve-bytes', 'exec /opt/unsloth-venv/bin/python'):
            self.assertIn(text, self.runner)
        self.assertIn('[ "$GPU_MEMORY_BUDGET_BYTES" -lt "$budget" ]', self.runner)
        self.assertIn('resume=auto', self.runner)

    def test_smoke_controls_reach_runner_and_cannot_enter_train(self):
        for text in ('SOOMAC_SMOKE_MAX_STEPS', 'SOOMAC_TEST_FAIL_STEP',
                     '--smoke-max-steps "$smoke_steps"', '--test-fail-step "$fail_step"',
                     'smoke overrides are forbidden outside smoke mode', '"$@"'):
            self.assertIn(text, self.runner)

    def test_final_image_lineage_gate(self):
        candidate = {"Id": "sha256:" + "a" * 64,
                     "RootFS": {"Layers": ["sha256:base", "sha256:candidate"]}}
        final = {"Os": "linux", "Architecture": "amd64",
                 "RootFS": {"Layers": ["sha256:base", "sha256:candidate", "sha256:receipt"]}}
        verify_lineage(candidate, final, candidate["Id"])
        with self.assertRaisesRegex(ValueError, "not a layer extension"):
            verify_lineage(candidate, {**final, "RootFS": {"Layers": ["sha256:base", "sha256:other"]}},
                           candidate["Id"])
        with self.assertRaisesRegex(ValueError, "differs"):
            verify_lineage(candidate, final, "sha256:" + "b" * 64)

    def test_final_build_requires_three_bounded_sha_pinned_evidence_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "validation"
            root.mkdir()
            evidence = {}
            for key in ("loss", "gradient", "restore"):
                path = root / f"{key}.json"
                path.write_text("{}")
                evidence[key] = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            receipt = {"status": "PASS", "agent": "decision",
                       "tested_candidate_image_digest": "sha256:" + "a" * 64,
                       "evidence_files": evidence}
            (root / "VALIDATION_RECEIPT.json").write_text(json.dumps(receipt))
            self.assertEqual(validate_receipt_files(Path(temporary)), receipt["tested_candidate_image_digest"])
            (root / "loss.json").write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                validate_receipt_files(Path(temporary))
            (root / "loss.json").unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                validate_receipt_files(Path(temporary))

    def test_sealed_decision_inputs(self):
        manifest = json.loads((RELEASE / "dataset/MANIFEST.json").read_text())
        declared = json.loads((RELEASE / "remote/lab-inputs.json").read_text())
        self.assertEqual(manifest["max_length"]["decision"], 9216)
        self.assertEqual(manifest["counts"]["decision"]["train"], 4360)
        self.assertEqual(declared["dataset"]["embedded"]["path"], "/app/assets/dataset")
        self.assertIn(MANIFEST_SHA, self.dockerfile)
        self.assertIn('d["file_sha256"].items()', self.dockerfile)
        for name, expected_sha in manifest["file_sha256"].items():
            path = RELEASE / "dataset" / name
            self.assertTrue(path.is_file() and not path.is_symlink(), name)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected_sha, name)


if __name__ == "__main__":
    unittest.main()
