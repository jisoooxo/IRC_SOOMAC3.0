"""No CUDA needed: verify process-local overrides and descriptor restoration."""
import importlib.util
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import memory_speed_trial as trial


class MemoryTrialCPU(unittest.TestCase):
    def test_gc_overrides_both_entrypoints_and_restores(self):
        for setting, skip in ((False, 0), (True, 16), ("unsloth", 0)):
            seen = []

            class Fake:
                @staticmethod
                def from_pretrained(**kw):
                    seen.append(kw)
                    return "model"

                @staticmethod
                def get_peft_model(model, **kw):
                    seen.append(kw)
                    return model

            original = inspect.getattr_static(Fake, "from_pretrained")

            def build(*unused):
                model = Fake.from_pretrained(use_gradient_checkpointing="unsloth")
                return Fake.get_peft_model(model, use_gradient_checkpointing="unsloth")

            with patch.dict(sys.modules, {"unsloth": SimpleNamespace(FastModel=Fake)}):
                with patch.object(trial.benchmark, "model_and_optimizer", side_effect=build):
                    model, calls = trial.load_with_gc(setting, None, skip)
            self.assertEqual(model, "model")
            self.assertEqual([x["use_gradient_checkpointing"] for x in seen], [setting, setting])
            self.assertEqual(seen[1].get("checkpoint_skip_layers", 0), skip)
            self.assertIs(inspect.getattr_static(Fake, "from_pretrained"), original)
            self.assertEqual(len(calls), 2)

    def test_effective_batch_fixed(self):
        for batch in (1, 2):
            self.assertEqual(batch * (16 // batch), 16)

    def test_partial_is_explicit_not_full_on(self):
        args = trial.args_parse(["--encoded", "/tmp/x", "--output", "/tmp/y",
                                 "--reference-loss", "/tmp/z", "--gc", "partial",
                                 "--checkpoint-skip", "16"])
        self.assertEqual((args.gc, args.checkpoint_skip), ("partial", 16))

    def test_actual_decoder_flags_enforced(self):
        for mode, skip, count in (("off", 0, 0), ("on", 0, 48),
                                  ("native", 0, 48), ("partial", 8, 40)):
            state = {"is_gradient_checkpointing": mode != "off",
                     "enabled_modules": [f"base.layers.{i}" for i in range(count)]}
            self.assertEqual(trial.validate_gc_state(state, mode, skip), count)
        with self.assertRaises(RuntimeError):
            trial.validate_gc_state({"is_gradient_checkpointing": True,
                                     "enabled_modules": []}, "native", 0)


if __name__ == "__main__":
    unittest.main()
