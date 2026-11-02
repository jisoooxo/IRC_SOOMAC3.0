"""CPU-only controls for the frozen-case inference bridge."""
import copy
import unittest

import eval_frozen_cases as runner
from heldout_cases import authored_cases


class FrozenCaseRunnerTests(unittest.TestCase):
    def test_all_cases_rebuild_exact_runtime_input(self):
        cases = authored_cases()
        self.assertEqual(len(cases), 24)
        self.assertEqual([sum(case["agent"] == agent for case in cases)
                          for agent in ("decision", "recommendation", "response")], [8, 8, 8])
        for case in cases:
            agent, args = runner.call_arguments(case)
            self.assertEqual(agent, case["agent"])
            self.assertTrue(args)

    def test_derived_input_drift_is_rejected(self):
        cases = authored_cases()
        recommendation = copy.deepcopy(next(case for case in cases
                                            if case["agent"] == "recommendation"))
        recommendation["record"]["input"]["recommendation_scope"]["current_section"] = "wrong"
        with self.assertRaisesRegex(ValueError, "runtime input drift"):
            runner.call_arguments(recommendation)
        response = copy.deepcopy(next(case for case in cases if case["agent"] == "response"
                                 and case["record"]["id"] == "QG-NEW-P-04"))
        response["record"]["input"]["starting_now"] = []
        with self.assertRaisesRegex(ValueError, "runtime input drift"):
            runner.call_arguments(response)

    def test_prediction_uses_external_sparse_decision_and_reply(self):
        cases = authored_cases()

        def decision(*args):
            decision.last_external_output = {"route": "task", "commit": True}
            return {"route": "task", "commit": True, "other_defaults": "not_exported"}

        def recommendation(*args):
            return {"proposal": {"sauce": "크림", "noodle_type": None,
                                 "noodle_portion": None, "toppings": {}}, "reason_tags": []}

        def response(*args):
            return "치즈는 적게 선택했고 지금 담기 시작합니다."

        calls = {"decision": decision, "recommendation": recommendation, "response": response}
        for agent in calls:
            case = next(case for case in cases if case["agent"] == agent)
            item = runner.predict_one(case, calls)
            self.assertEqual(item["id"], case["record"]["id"])
            if agent == "decision":
                self.assertEqual(item["prediction"], decision.last_external_output)
                self.assertNotIn("other_defaults", item["prediction"])
            elif agent == "response":
                self.assertEqual(item["prediction"], {"reply": response()})
            else:
                self.assertEqual(item["prediction"], recommendation())


if __name__ == "__main__":
    unittest.main()
