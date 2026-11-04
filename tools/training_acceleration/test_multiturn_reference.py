"""Test authored D assumptions and state/plan fixtures without model inference."""
import contextlib
import io
import json

import pytest

import eval_multiturn_reference as evaluate


@pytest.mark.parametrize('spec', evaluate.scenarios(), ids=lambda x: x['name'])
@pytest.mark.parametrize('mode', ['model', 'fixed'])
def test_fixture_reaches_expected_plans_and_keeps_generated_history(spec, mode):
    counts = {'response': 0, 'recommendation': 0}
    def response(*args):
        counts['response'] += 1
        return f'CPU 응답 {counts["response"]} — 모델 품질 증거 아님'
    def recommendation(*args):
        counts['recommendation'] += 1
        return {'proposal': {'sauce': '크림', 'noodle_type': None, 'noodle_portion': None, 'toppings': {}},
                'reason_tags': ['느끼한 소스 요청']}
    response.trace_events, recommendation.trace_events = [], []
    stream = io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()):
        result = evaluate.run_episode(spec, {'response': response, 'recommendation': recommendation}, stream,
                                      execution_reply_mode=mode)
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert result['passed'], rows[-1]['checks']
    execution_turns = 2 if spec['name'] == 'unresolved_reference_control' else 4
    assert counts['response'] == len(spec['turns']) - (execution_turns if mode == 'fixed' else 0)
    assert counts['recommendation'] == (1 if spec['name'] == 'recommendation_and_short_execution' else 0)
    assert result['execution_turns'] == (2 if spec['name'] == 'unresolved_reference_control' else 4)
    for previous, current in zip(rows, rows[1:]):
        assert previous['after']['history'] == current['before']['history'][:len(previous['after']['history'])]
    if spec['name'] == 'unresolved_reference_control':
        assert rows[3]['policy']['execute'] is rows[4]['policy']['execute'] is False
        assert rows[3]['after']['order'] == rows[4]['after']['order'] == rows[2]['after']['order']
