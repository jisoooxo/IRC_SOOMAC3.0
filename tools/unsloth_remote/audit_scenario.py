#!/usr/bin/env python3
"""Check literal user scenario in sealed build05 without generating/replacing data."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

RELEASE = Path('/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/releases/conversation_r5_20261101_build05_speed')
MANIFEST = '7850948287237c5ace9d1b0f56931419760fbce3f0afa02cea40681acb6ceffa'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    dataset = RELEASE / 'dataset'
    assert sha(dataset / 'MANIFEST.json') == MANIFEST
    manifest = json.loads((dataset / 'MANIFEST.json').read_text())
    selected = {}
    for agent in ('decision', 'recommendation', 'response'):
        relative = f'records/{agent}/train.jsonl'
        source = dataset / relative
        assert sha(source) == manifest['file_sha256'][relative]
        selected[agent] = [r for line in source.open() if (r := json.loads(line))['id'].startswith('DA31-CORE')]
    d = {r['id']: r for r in selected['decision']}
    first = d['DA31-CORE-separate-T01-D']['target']
    assert first['order']['noodle_type'] == '얇은면'
    assert first['order']['noodle_portion'] == 'normal'
    assert first['recommendation']['action'] == 'request'
    for variant in ('compound', 'polite', 'directive'):
        rows = [r for r in selected['decision'] if r['id'].startswith('DA31-CORE-' + variant)]
        commits = [r for r in rows if '바로' in r['input']['message']]
        assert len(rows) == 5 and len(commits) == 4
        assert all(r['target'].get('commit') is True for r in commits)
        assert all(len(r['input']['recent_history']) >= 3 for r in commits)
        # Variation episodes include different event turn indices; compare
        # exact semantic target multiplicity, not unrelated numeric turn IDs.
        signature = lambda row: json.dumps(row['target'], sort_keys=True, ensure_ascii=False)
        canonical = [r for r in selected['decision'] if r['id'].startswith('DA31-CORE-compound-')]
        if variant == 'directive':
            # "그렇게" accepts the actual pending cream candidate instead of
            # spelling cream again. Distinct valid fields, same intended choice.
            accepted = d['DA31-CORE-directive-T02-D']
            assert accepted['target'] == {'route': 'task', 'confirmation': 'accept', 'commit': True}
            assert accepted['input']['pending']['type'] == 'recommendation'
            assert accepted['input']['pending']['candidate'] == {
                'noodle_type': None, 'noodle_portion': None, 'sauce': '크림', 'toppings': {}}
            rows = [r for r in rows if r['id'] != accepted['id']]
            canonical = [r for r in canonical if r['id'] != 'DA31-CORE-compound-T02-D']
        assert Counter(map(signature, rows)) == Counter(map(signature, canonical))
    cream = d['DA31-CORE-compound-T02-D']
    first_text = d['DA31-CORE-separate-T01-D']['input']['message']
    assert first_text in [m['content'] for m in cream['input']['recent_history'] if m['role'] == 'user']
    assert cream['input']['pending']['candidate']['sauce'] == '크림'
    assert cream['target']['order']['sauce'] == '크림' and cream['target']['commit'] is True
    change = d['DA31-CORE-compound-T05-D']
    assert cream['input']['message'] in [m['content'] for m in change['input']['recent_history'] if m['role'] == 'user']
    assert change['input']['order']['toppings'] == {'양파': 'low', '버섯': 'low'}
    assert change['target']['order']['toppings'] == {'버섯': 'normal'}
    assert change['target']['commit'] is True
    meat_row = d['DA31-CORE-compound-T07-D']
    assert meat_row['input']['order']['toppings'] == {'양파': 'low', '버섯': 'normal'}
    assert change['input']['message'] in [m['content'] for m in meat_row['input']['recent_history'] if m['role'] == 'user']
    meat = meat_row['target']
    assert meat['order']['toppings'] == {'게살': 'none', '소시지': 'high'}
    assert meat['restrictions'] == [{'action': 'add', 'reason': 'allergy', 'target': '게살'}]
    assert meat['commit'] is True
    topping_row = d['DA31-CORE-compound-T09-D']
    assert {'reason': 'allergy', 'target': '게살'} in topping_row['input']['order']['restrictions']
    assert topping_row['input']['order']['toppings'] == {'양파': 'low', '버섯': 'normal', '소시지': 'high'}
    assert meat_row['input']['message'] in [m['content'] for m in topping_row['input']['recent_history'] if m['role'] == 'user']
    topping = topping_row['target']
    assert topping['order']['toppings'] == {'치즈': 'low', '페퍼론치노': 'none'}
    assert topping['commit'] is True
    assert len(selected['recommendation']) == 1
    proposal = selected['recommendation'][0]['target']['proposal']
    assert proposal == {'noodle_type': None, 'noodle_portion': None, 'sauce': '크림', 'toppings': {}}
    assert len(selected['decision']) == 26 and len(selected['response']) == 10
    return {'status': 'SEALED_LITERAL_SCENARIO_ASSERTIONS_PASS_NOT_MODEL_FUNCTION_PASS',
            'release': str(RELEASE), 'manifest_sha256': MANIFEST,
            'counts': {a: len(r) for a, r in selected.items()},
            'notes': ['Execution-start P rows intentionally excluded by supplement spec; fixed Python start path.',
                      'Data membership does not prove newly trained adapter behavior. No source or model changed.'],
            'rows': {a: [{'id': r['id'], 'target': r['target'],
                         'history_messages': len(r['input'].get('recent_history', []))} for r in rows]
                     for a, rows in selected.items()}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(result['status'], result['counts'])
