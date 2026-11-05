import copy
import io
import json
import unittest
from contextlib import redirect_stdout
import dqp_evaluation as e


class EvaluationTests(unittest.TestCase):
    def calls(self,spec,wrong=False):
        cursor=0
        def decision(session,text,robot,repair=None):
            nonlocal cursor
            if repair is not None:raise AssertionError('unexpected oracle repair')
            case=spec['turns'][cursor];cursor+=1
            self.assertEqual(text,case['text'])
            value=copy.deepcopy(case['decision'])
            if wrong and cursor==1:value={'route':'task','order':{'toppings':{'양파':'low','버섯':'normal'}}}
            decision.last_external_output=value
            decision.trace_events.append(dict(stage='decision',external=value,cpu_fixture_only=True))
            return e.normalize_decision(value)
        def q(*args):
            return dict(proposal=dict(sauce='크림',noodle_type=None,noodle_portion=None,toppings={}),reason_tags=['느끼한 요청'])
        def p(*args):
            p.trace_events.append(dict(stage='response',model_input=e.build_response_model_input(*args),route=args[8]))
            return 'CPU fixture only'
        for call in (decision,q,p):call.trace_events=[]
        return dict(decision=decision,recommendation=q,response=p)

    def test_frozen_suite_has_all_categories_and_disjoint_ids(self):
        rows=e.b.read(e.BASE/'inputs/CASES.json')['rows']
        self.assertEqual(len(rows),len({(r['agent'],r['record']['id']) for r in rows}))
        self.assertEqual({r['record']['target']['route'] for r in rows if r['agent']=='decision'}, {'general','task','mixed'})
        for case in rows:e.args_for(case)

    def test_all_gold_spec_transitions_without_gpu(self):
        for spec in e.extended_specs():
            with self.subTest(episode=spec['name']),redirect_stdout(io.StringIO()):
                output=io.StringIO();result=e.extended_episode(spec,self.calls(spec),output)
                self.assertTrue(result['passed'],result)
                rows=[json.loads(x) for x in output.getvalue().splitlines()]
                for row in rows:
                    if row['policy']['execute']:self.assertEqual(row['traces']['response'],[])
                    self.assertTrue(row['actual_reply_in_history'] if 'actual_reply_in_history' in row else row['checks']['actual_reply_in_history'])
        import torch
        self.assertFalse(torch.cuda.is_initialized())

    def test_wrong_quantity_preserved_and_episode_stops(self):
        spec=next(s for s in e.extended_specs() if s['name']=='cold_quantity_reference')
        stream=io.StringIO()
        with redirect_stdout(io.StringIO()):result=e.extended_episode(spec,self.calls(spec,wrong=True),stream)
        rows=[json.loads(x) for x in stream.getvalue().splitlines()]
        self.assertFalse(result['passed']);self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['after']['order']['toppings']['양파'],'low')
        self.assertEqual(rows[0]['expected_order']['toppings']['양파'],'high')
        self.assertFalse(rows[0]['checks']['order_matches'])


if __name__=='__main__':unittest.main()
