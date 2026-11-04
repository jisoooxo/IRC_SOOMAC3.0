"""CPU oracle/harness controls; these are not model quality evidence."""
import contextlib
import copy
import io
import json
import unittest
from unittest.mock import patch

import eval_gold_decision_scenario as run
import manual_scenario_stack_probe as probe
from soomac_irc.agent_contract import empty_order_patch, new_session_state


class GoldDecisionControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = run.oracle_rows()

    def test_oracle_is_exact_gold_without_session_mutation(self):
        session = new_session_state()
        before = copy.deepcopy(session)
        call = run.oracle_factory(True, self.rows)(None, None)
        call(session, probe.scenario(True)[0][2], {"section": "noodle"})
        expected = self.rows[run.SOURCE_IDS["initial"]]["target"]
        self.assertEqual(call.last_external_output, expected)
        self.assertEqual(session, before)
        self.assertTrue(call.trace_events[-1]["gold_injected"])
        self.assertEqual(call.trace_events[-1]["source_id"], run.SOURCE_IDS["initial"])
        with self.assertRaisesRegex(ValueError, "Unexpected repair"):
            call(session, "retry", {}, repair={})

    def test_unexpected_utterance_is_rejected(self):
        call = run.oracle_factory(False, self.rows)(None, None)
        with self.assertRaisesRegex(ValueError, "utterance drift"):
            call(new_session_state(), "different user input", {})

    def test_gold_fields_drive_17_state_transitions_with_stub_outputs(self):
        def recommendation_factory(*args, **kwargs):
            def call(*unused):
                proposal = empty_order_patch()
                proposal["sauce"] = "크림"
                return {"proposal": proposal, "reason_tags": ["느끼한 소스 요청"]}
            call.trace_events = []
            return call

        def response_factory(*args, **kwargs):
            def call(*unused):
                return "CPU 하네스 검사용 응답"
            call.trace_events = []
            return call

        total = 0
        with patch("soomac_irc.recommendation_model.make_call_recommendation", recommendation_factory), \
             patch("soomac_irc.response_model.make_call_response", response_factory), \
             contextlib.redirect_stdout(io.StringIO()):
            for compound in (True, False):
                stream = io.StringIO()
                with patch("soomac_irc.decision_model.make_call_decision",
                           run.oracle_factory(compound, self.rows)):
                    result = probe.run_episode(compound, None, None, stream, measure_all=True)
                turns = [json.loads(line) for line in stream.getvalue().splitlines()]
                self.assertTrue(result["passed"], result["failed_turns"])
                self.assertEqual(len(turns), 6 if compound else 11)
                self.assertTrue(all(t["traces"]["decision"][0]["gold_injected"] for t in turns))
                self.assertEqual(result["reply_sources"]["response_model"], 0)
                total += len(turns)
        self.assertEqual(total, 17)
        import torch
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
