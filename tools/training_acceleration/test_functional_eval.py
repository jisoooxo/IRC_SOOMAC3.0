"""Oracle positive/negative controls, not actual adapter evaluation."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent))
import heldout_cases
import functional_eval


class FunctionalControls(unittest.TestCase):
    def test_decision_gold_and_commit_negatives(self):
        cases = [c for c in heldout_cases.authored_cases() if c['agent'] == 'decision']
        for case in cases:
            target = case['record']['target']
            self.assertTrue(functional_eval.grade(case, target)['automated_pass'], case['record']['id'])
            if target.get('commit'):
                bad = {**target, 'commit': False}
                self.assertFalse(functional_eval.grade(case, bad)['automated_pass'])

    def test_recommendation_gold_and_amount_negatives(self):
        for case in heldout_cases.authored_cases():
            if case['agent'] != 'recommendation':
                continue
            target = case['record']['target']
            self.assertTrue(functional_eval.grade(case, target)['automated_pass'], case['record']['id'])
            for key in case['strict_required_proposal_fields']:
                if key.startswith('toppings.'):
                    import copy
                    bad = copy.deepcopy(target)
                    bad['proposal']['toppings'][key.split('.', 1)[1]] = 'high'
                    self.assertFalse(functional_eval.grade(case, bad)['automated_pass'])

    def test_unverified_completion_never_becomes_automatic_pass(self):
        for case in heldout_cases.authored_cases():
            if case['agent'] == 'response':
                self.assertFalse(functional_eval.grade(case, {'reply': '모두 다 담았어요. 작업을 완료했습니다.'})['automated_pass'])

    def test_response_positive_fact_control(self):
        case = next(c for c in heldout_cases.authored_cases() if c['record']['id'] == 'QG-NEW-P-01')
        self.assertTrue(functional_eval.grade(case, {'reply': '양파를 보통 양으로 바꿨어요.'})['automated_pass'])
        self.assertFalse(functional_eval.grade(case, {'reply': '양파를 많이 담았어요.'})['automated_pass'])

    def test_schema_hallucinated_fields_rejected(self):
        case = heldout_cases.authored_cases()[0]
        self.assertFalse(functional_eval.grade(case, {'route': 'task', 'fake_field': True})['automated_pass'])


if __name__ == '__main__':
    unittest.main()
