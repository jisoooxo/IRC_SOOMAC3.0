"""Oracle positive/negative controls, not actual adapter evaluation."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent))
import heldout_cases
import functional_eval


class FunctionalControls(unittest.TestCase):
    def test_decision_gold_state_expectations_replay_actual_graph(self):
        import copy
        from soomac_irc.agent_contract import new_session_state, new_turn_state, normalize_decision
        from soomac_irc.llm_langgraph import build_graph
        from soomac_irc.llm_policy import section_execution_items
        for case in heldout_cases.authored_cases():
            if case['agent'] != 'decision':
                continue
            row = case['record']
            raw, target = row['input'], row['target']
            session = new_session_state()
            session.update(order=copy.deepcopy(raw['order']), pending=copy.deepcopy(raw['pending']),
                           preferences=copy.deepcopy(raw['preferences']), history=copy.deepcopy(raw['recent_history']))
            q_calls = []

            def decision(*_args):
                decision.last_external_output = copy.deepcopy(target)
                return normalize_decision(copy.deepcopy(target))

            def recommendation(*_args):
                q_calls.append(True)
                return {'proposal': {'sauce': '크림', 'noodle_type': None, 'noodle_portion': None, 'toppings': {}},
                        'reason_tags': []}

            result = build_graph(decision, recommendation, lambda *_: '검사용 모의 응답').invoke(
                new_turn_state(session, raw['message'], copy.deepcopy(raw['robot_state'])))
            selected = result['session']['order']
            for key, expected in row['metadata']['required_state_facts'].items():
                if key == 'execute':
                    actual = result['policy']['execute']
                elif key == 'Q_called':
                    actual = bool(q_calls)
                elif key == 'pending':
                    actual = result['session']['pending']
                elif key == 'order_unchanged':
                    actual = selected == raw['order']
                elif key == 'pepper_execution':
                    actual = any(x['item'] == '페퍼론치노' for x in section_execution_items(selected, raw['robot_state']['section']))
                elif key.startswith('restriction.'):
                    actual = next((r['reason'] for r in selected['restrictions'] if r['target'] == key.split('.', 1)[1]), None)
                elif key.startswith('toppings.'):
                    actual = selected['toppings'].get(key.split('.', 1)[1])
                else:
                    actual = selected[key]
                self.assertEqual(actual, expected, (row['id'], key, result['policy']))

    def test_authored_inputs_use_actual_runtime_scope_and_execution_items(self):
        from soomac_irc.llm_policy import build_recommendation_scope, section_execution_items
        for case in heldout_cases.authored_cases():
            if case['agent'] == 'decision':
                continue
            raw = case['record']['input']
            selected = raw.get('confirmed_order', raw.get('order'))
            expected = build_recommendation_scope({'order': selected}, raw['robot_state'])
            self.assertEqual(raw['recommendation_scope'], expected)
            self.assertEqual(expected['current_section'], raw['robot_state']['section'])
            if case['agent'] == 'response':
                items = section_execution_items(selected, raw['robot_state']['section']) if raw['execution_authorized'] else []
                self.assertEqual(raw['starting_now'], items)

    def test_multiturn_amount_and_allergy_examples_have_actual_history(self):
        cases = {c['record']['id']: c['record']['input'] for c in heldout_cases.authored_cases()}
        for case_id in ('QG-NEW-Q-03', 'QG-NEW-P-01', 'QG-NEW-P-02', 'QG-NEW-P-03', 'QG-NEW-P-07', 'QG-NEW-P-08'):
            self.assertEqual([m['role'] for m in cases[case_id]['recent_history']], ['user', 'assistant'])
        scope = cases['QG-NEW-P-02']['recommendation_scope']
        self.assertNotIn('noodle', [s['section'] for s in scope['sections']])
        self.assertIn('sauce', [s['section'] for s in scope['sections']])

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

    def test_frozen_recommendation_oracle_retains_default_amount_checks(self):
        import copy
        cases = {c['record']['id']: {'agent': c['agent'], 'record': c['record']}
                 for c in heldout_cases.authored_cases() if c['agent'] == 'recommendation'}
        for case_id, topping in [('QG-NEW-Q-05', '양파'), ('QG-NEW-Q-06', '버섯')]:
            case = cases[case_id]
            self.assertTrue(functional_eval.grade(case, case['record']['target'])['automated_pass'])
            bad = copy.deepcopy(case['record']['target'])
            bad['proposal']['toppings'][topping] = 'high'
            self.assertIn('strict_authored_proposal_field',
                          {item['code'] for item in functional_eval.grade(case, bad)['failures']})
        case = cases['QG-NEW-Q-03']
        bad = copy.deepcopy(case['record']['target'])
        bad['proposal']['toppings']['치즈'] = 'high'
        self.assertIn('ungrounded_new_amount',
                      {item['code'] for item in functional_eval.grade(case, bad)['failures']})

    def test_unverified_completion_never_becomes_automatic_pass(self):
        for case in heldout_cases.authored_cases():
            if case['agent'] == 'response':
                self.assertFalse(functional_eval.grade(case, {'reply': '모두 다 담았어요. 작업을 완료했습니다.'})['automated_pass'])

    def test_passive_completion_claim_requires_completion_evidence(self):
        case = next(c for c in heldout_cases.authored_cases() if c['record']['id'] == 'QG-NEW-P-05')
        for reply in ('양파는 이미 소량으로 담겨 있어요.', '양파는 이미 소량으로 담겨있습니다.'):
            result = functional_eval.grade(case, {'reply': reply})
            self.assertFalse(result['automated_pass'], reply)
            self.assertTrue(
                {'unverified_completion_reply', 'unverified_item_completion'}
                & {item['code'] for item in result['failures']}, reply,
            )
        for reply in ('양파는 아직 담겨 있지 않아요.', '양파는 안 담겨 있어요.'):
            result = functional_eval.grade(case, {'reply': reply})
            self.assertFalse(
                {'unverified_completion_reply', 'unverified_item_completion'}
                & {item['code'] for item in result['failures']}, reply,
            )
        mixed = functional_eval.grade(case, {'reply': '양파는 안 담겨 있어요. 버섯은 이미 담겨 있어요.'})
        self.assertIn('unverified_completion_reply', {item['code'] for item in mixed['failures']})

    def test_nonspicy_request_allows_safe_full_proposal_but_requires_sauce(self):
        import copy
        case = next(c for c in heldout_cases.authored_cases() if c['record']['id'] == 'QG-NEW-Q-03')
        full = copy.deepcopy(case['record']['target'])
        full['proposal'].update(noodle_type='얇은면', noodle_portion='normal', toppings={'치즈': 'normal'})
        self.assertTrue(functional_eval.grade(case, full)['automated_pass'])
        no_sauce = copy.deepcopy(full)
        no_sauce['proposal']['sauce'] = None
        self.assertFalse(functional_eval.grade(case, no_sauce)['automated_pass'])
        excessive = copy.deepcopy(full)
        excessive['proposal']['toppings']['치즈'] = 'high'
        self.assertFalse(functional_eval.grade(case, excessive)['automated_pass'])
        spicy = copy.deepcopy(full)
        spicy['proposal']['toppings']['페퍼론치노'] = 'normal'
        self.assertFalse(functional_eval.grade(case, spicy)['automated_pass'])

    def test_response_positive_fact_control(self):
        case = next(c for c in heldout_cases.authored_cases() if c['record']['id'] == 'QG-NEW-P-01')
        self.assertTrue(functional_eval.grade(case, {'reply': '양파를 보통 양으로 바꿨어요.'})['automated_pass'])
        self.assertFalse(functional_eval.grade(case, {'reply': '양파를 많이 담았어요.'})['automated_pass'])
        repeated_question = functional_eval.grade(case, {'reply': '양파를 보통 양으로 바꿨어요. 지금 담을까요?'})
        self.assertIn('unwanted_execution_question', {x['code'] for x in repeated_question['failures']})
        self.assertTrue(functional_eval.grade(case,
            {'reply': '양파를 보통 양으로 바꿨어요. 담기 여부는 다시 묻지 않겠습니다.'})['automated_pass'])

    def test_response_does_not_add_unrequested_recommendation(self):
        case = next(c for c in heldout_cases.authored_cases() if c['record']['id'] == 'QG-NEW-P-06')
        self.assertTrue(functional_eval.grade(case,
            {'reply': '버섯을 많이 골라 두셨어요. 추가 추천은 하지 않겠습니다.'})['automated_pass'])
        self.assertTrue(functional_eval.grade(case,
            {'reply': '버섯을 많이 골라 두셨어요. 추가 재료는 추천해 드리지는 않겠습니다.'})['automated_pass'])
        mixed = functional_eval.grade(case, {'reply':
            '버섯을 많이 골라 두셨어요. 추가 재료는 추천해 드리지는 않겠습니다. 치즈는 추천드립니다.'})
        self.assertIn('unwanted_new_recommendation', {x['code'] for x in mixed['failures']})
        for offer in ('치즈도 추천드릴게요.', '치즈도 추천드립니다.', '치즈도 권합니다.'):
            wrong = functional_eval.grade(case, {'reply': f'버섯을 많이 골라 두셨어요. {offer}'})
            self.assertIn('unwanted_new_recommendation', {x['code'] for x in wrong['failures']})

    def test_schema_hallucinated_fields_rejected(self):
        case = heldout_cases.authored_cases()[0]
        self.assertFalse(functional_eval.grade(case, {'route': 'task', 'fake_field': True})['automated_pass'])


if __name__ == '__main__':
    unittest.main()
