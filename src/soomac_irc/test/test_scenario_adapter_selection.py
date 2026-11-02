"""CPU-only CLI controls: no model/server/training is started."""
import json
from pathlib import Path
import tempfile
import unittest

from manual_scenario_stack_probe import evaluation_paths, parse_arguments, response_evidence


class AdapterSelectionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.defaults = {}
        for name in ("decision", "response", "recommendation", "new_response"):
            path = self.root / name
            path.mkdir()
            (path / "adapter_config.json").write_text(json.dumps({"r": 16}))
            (path / "adapter_model.safetensors").write_bytes(b"fixture only")
            if name != "new_response":
                self.defaults[name] = str(path)

    def test_defaults_retained(self):
        args = parse_arguments(["--out", str(self.root / "out")])
        self.assertEqual(evaluation_paths(args, self.defaults), self.defaults)

    def test_one_override_does_not_change_defaults(self):
        original = dict(self.defaults)
        args = parse_arguments(["--out", str(self.root / "out"), "--response-adapter",
                                str(self.root / "new_response"), "--vllm-base-url", "http://127.0.0.1:18001"])
        actual = evaluation_paths(args, self.defaults)
        self.assertEqual(actual["response"], str(self.root / "new_response"))
        self.assertEqual(actual["decision"], original["decision"])
        self.assertEqual(actual["recommendation"], original["recommendation"])
        self.assertEqual(self.defaults, original)

    def test_missing_original_is_rejected(self):
        args = parse_arguments(["--out", str(self.root / "out"), "--response-adapter", str(self.root / "missing")])
        with self.assertRaises(FileNotFoundError):
            evaluation_paths(args, self.defaults)

    def test_converted_override_is_rejected(self):
        path = self.root / "new_response"
        (path / "CONVERSION.json").write_text("{}")
        args = parse_arguments(["--out", str(self.root / "out"), "--response-adapter", str(path)])
        with self.assertRaises(ValueError):
            evaluation_paths(args, self.defaults)

    def test_fixed_and_error_replies_keep_provenance(self):
        fixed = response_evidence([], "담기를 시작합니다.", "compound", "cream_execute")
        self.assertEqual(fixed["source"], "runtime_fixed")
        self.assertIsNone(fixed["automatic_pass"])
        failed = response_evidence([{"stage": "response", "error": {"type": "ValueError"}}],
                                   "주문을 처리하지 못했어요.", "compound", "initial")
        self.assertEqual(failed["source"], "response_error_fallback")
        self.assertFalse(failed["automatic_pass"])

    def test_model_reply_is_automatically_flagged_not_semantically_approved(self):
        from soomac_irc.response_contract import build_response_model_input
        session = {"order": {"sauce": None, "noodle_type": None, "noodle_portion": None,
                             "toppings": {"양파": "low"}, "restrictions": []},
                   "history": [], "preferences": [], "pending": None, "action_history": []}
        robot = {"section": "veggie", "active_task": None, "completed_tasks": [],
                 "robot_started": False, "task_queue": []}
        policy = {"status": "pass", "execute": False, "reason": "state_update_only", "conflicts": []}
        applied = {"order_changes": {"toppings.양파": "low"}, "preference_added": [],
                   "preference_removed": [], "restriction_added": [], "restriction_removed": []}
        model_input = build_response_model_input("양파 조금 골라줘.", session, policy,
                                                 applied, [], None, None, robot, "task")
        evidence = response_evidence([{"stage": "response", "route": "task",
                                       "model_input": model_input}],
                                     "양파를 적게 선택했어요.", "compound", "vegetables")
        self.assertEqual(evidence["source"], "response_model")
        self.assertTrue(evidence["automatic_pass"])
        self.assertEqual(evidence["semantic_review"], "REQUIRES_INDEPENDENT_REVIEW")
        wrong = response_evidence([{"stage": "response", "route": "task",
                                    "model_input": model_input}],
                                  "양파를 보통 양으로 선택했어요. 이미 담았습니다.",
                                  "compound", "vegetables")
        self.assertFalse(wrong["automatic_pass"])
        self.assertIn("selected_amount_contradiction", wrong["text_rule_flags"])
        self.assertIn("unverified_completion_reply", wrong["text_rule_flags"])


if __name__ == "__main__":
    unittest.main()
