"""Offline functional grading of frozen new cases; never starts inference/training.

Automated success is not a human semantic approval. Response always needs a
separate grounded-facts review; no exact sentence matching is used.
"""
import argparse
import json
from pathlib import Path
import re
import sys

import benchmark

sys.path.insert(0, str(Path(__file__).parents[2] / 'src/soomac_irc'))
from soomac_irc import evaluation_v43 as evaluator


_AFFIRMATIVE_NEW_RECOMMENDATION = re.compile(
    r'추천(?:해|드려|드릴|드립니다|합니다|드리겠|할게|하겠)|'
    r'권(?:해(?:요|드려|드릴|드립니다)|합니다)'
)
_NEGATED_NEW_RECOMMENDATION = re.compile(
    r'(?:추천|권)(?:해\s*)?(?:드리)?지는?\s*않'
)


def grade(case, prediction):
    from jsonschema import Draft202012Validator
    agent, row = case['agent'], case['record']
    schema = benchmark.read(benchmark.RELEASE / f'dataset/contract/{agent}_schema.json')
    try:
        Draft202012Validator(schema).validate(prediction)
    except Exception as error:
        return {'id': row['id'], 'agent': agent, 'automated_pass': False,
                'failures': [{'code': 'schema', 'detail': str(error)[:500]}], 'semantic_review': 'pending'}
    if agent == 'decision':
        result = evaluator.grade_decision_prediction(row, prediction)
    elif agent == 'recommendation':
        result = evaluator.grade_recommendation_prediction(row, prediction)
    else:
        result = evaluator.grade_response_prediction(row, prediction['reply'])
    failures = [{'code': v.code, 'path': v.path} for v in result.violations]
    if agent == 'response' and (row.get('metadata') or {}).get('no_execution_question'):
        if any(evaluator._is_question(clause) and evaluator._EXECUTION_QUESTION.search(clause)
               for clause in evaluator._clauses(prediction['reply'])):
            failures.append({'code': 'unwanted_execution_question', 'path': 'reply'})
    if agent == 'response' and (row.get('metadata') or {}).get('no_new_recommendation'):
        non_negated_reply = _NEGATED_NEW_RECOMMENDATION.sub('', prediction['reply'])
        if _AFFIRMATIVE_NEW_RECOMMENDATION.search(non_negated_reply):
            failures.append({'code': 'unwanted_new_recommendation', 'path': 'reply'})
    if agent == 'recommendation':
        actual = evaluator._proposal_fields(prediction['proposal'])
        # Existing generic grader deliberately permits flexible implicit amount.
        # These authored cases have an explicit new normal-default policy oracle.
        for field, expected in case.get('strict_required_proposal_fields', {}).items():
            if actual.get(field) != expected:
                failures.append({'code': 'strict_authored_proposal_field', 'path': field,
                                 'expected': expected, 'actual': actual.get(field)})
        default_amount = case.get('strict_new_amount_default')
        if default_amount:
            for field, amount in actual.items():
                if (field == 'noodle_portion' or field.startswith('toppings.')) and amount != default_amount:
                    failures.append({'code': 'ungrounded_new_amount', 'path': field,
                                     'expected': default_amount, 'actual': amount})
    return {'id': row['id'], 'agent': agent, 'automated_pass': not failures,
            'failures': failures, 'semantic_review': 'pending',
            'runtime_transition_executed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    destination = benchmark.output_dir(args.output)
    receipt_path = args.cases.parent / 'NEW_HOLDOUT_RECEIPT.json'
    receipt = benchmark.read(receipt_path)
    if benchmark.sha(args.cases) != receipt['cases_sha256']:
        raise ValueError('functional cases changed after freeze')
    cases = benchmark.read(args.cases)['rows']
    predictions = benchmark.read(args.predictions)
    if not isinstance(predictions, list) or len(predictions) != len(cases):
        raise ValueError('all frozen cases need exactly one prediction')
    by_id = {item['id']: item['prediction'] for item in predictions}
    if len(by_id) != len(cases) or set(by_id) != {c['record']['id'] for c in cases}:
        raise ValueError('missing/duplicate/extra prediction ID')
    observations = [grade(case, by_id[case['record']['id']]) for case in cases]
    result = {'status': 'AUTOMATED_FUNCTIONAL_RESULTS_SEMANTIC_REVIEW_REQUIRED',
        'cases_sha256': receipt['cases_sha256'], 'predictions_sha256': benchmark.sha(args.predictions),
        'per_agent': {agent: {'count': sum(x['agent'] == agent for x in observations),
                            'automated_pass': sum(x['agent'] == agent and x['automated_pass'] for x in observations)}
                      for agent in ('decision', 'recommendation', 'response')},
        'observations': observations, 'production_quality_pass': False,
        'human_review_required': True, 'unassisted_closed_loop_evaluation': 'not_run_here'}
    benchmark.write_new(destination / 'FUNCTIONAL_RESULTS.json', result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
