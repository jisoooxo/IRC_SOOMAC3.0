"""New authored functional cases, never imported into the training release.

These are review candidates until oracle audit passes. No fixed Response wording
is used as model-output gold; fields/facts are graded, human semantics stays separate.
"""
import copy


def order(**patch):
    return {'sauce': None, 'noodle_type': None, 'noodle_portion': None,
            'toppings': {}, 'restrictions': [], **patch}


def robot(section):
    return {'section': section, 'active_task': None, 'completed_tasks': [],
            'robot_started': False, 'task_queue': []}


def history(*pairs):
    return [{'role': role, 'content': text} for role, text in pairs]


def scope():
    return {'current_section': 'noodle', 'excluded_past_sections': [], 'sections': [
        {'section': 'noodle', 'fields': ['noodle_type', 'noodle_portion', 'sauce'],
         'options': ['얇은면', '넓은면', '크림', '토마토', '오일'], 'selected': {}},
        {'section': 'veggie', 'fields': ['toppings.양파', 'toppings.버섯'],
         'options': ['양파', '버섯'], 'selected': {}},
        {'section': 'meat', 'fields': ['toppings.소시지', 'toppings.게살'],
         'options': ['소시지', '게살'], 'selected': {}},
        {'section': 'extra', 'fields': ['toppings.치즈', 'toppings.페퍼론치노'],
         'options': ['치즈', '페퍼론치노'], 'selected': {}}]}


def authored_cases():
    cases = []

    def decision(index, text, target, *, section='noodle', selected=None, pending=None, previous=None,
                 required=None, alternatives=None):
        meta = {'dialogue_group': f'QG-NEW-D-{index:02}', 'source_type': 'new_authored_quality_holdout',
                'required_state_facts': required or {}, 'oracle_review': 'pending_independent_review'}
        if alternatives:
            meta['decision_target_alternatives'] = alternatives
        cases.append({'agent': 'decision', 'record': {'id': f'QG-NEW-D-{index:02}',
            'input': {'message': text, 'order': selected or order(), 'pending': pending,
                      'preferences': [], 'recent_history': previous or [], 'robot_state': robot(section)},
            'target': target, 'metadata': meta}})

    decision(1, '납작한 넓은면은 소량으로 골라 놓고, 소스는 진하고 느끼한 쪽 하나 권해 주세요.',
        {'route': 'task', 'order': {'noodle_type': '넓은면', 'noodle_portion': 'low'},
         'recommendation': {'action': 'request'}}, required={'execute': False, 'Q_called': True})
    decision(2, '방금 권한 크림으로 결정했으니 지금 선택까지 마치고 곧바로 담아 주시겠어요?',
        {'route': 'task', 'confirmation': 'accept', 'commit': True},
        selected=order(noodle_type='넓은면', noodle_portion='low'),
        pending={'type': 'recommendation', 'candidate': {'sauce': '크림', 'noodle_type': None,
                 'noodle_portion': None, 'toppings': {}}, 'reason_tags': ['느끼한 취향']},
        previous=history(('user', '오늘은 고소하고 묵직한 소스가 당기네요.'),
                         ('assistant', '크림을 후보로 골라 드렸어요. 선택하시겠어요?')),
        required={'sauce': '크림', 'noodle_portion': 'low', 'execute': True},
        alternatives=[{'route': 'task', 'confirmation': 'accept', 'commit': True},
                      {'route': 'task', 'order': {'sauce': '크림'}, 'confirmation': 'accept', 'commit': True}])
    decision(3, '양파의 소량 선택은 그대로 두고 버섯만 보통 분량으로 고쳐서 바로 투입해 주세요.',
        {'route': 'task', 'order': {'toppings': {'버섯': 'normal'}}, 'commit': True}, section='veggie',
        selected=order(toppings={'양파': 'low', '버섯': 'high'}),
        previous=history(('user', '양파는 소량, 버섯은 넉넉하게 골라 둘게요.'),
                         ('assistant', '두 채소의 선택량을 기록했어요.')),
        required={'toppings.양파': 'low', 'toppings.버섯': 'normal', 'execute': True})
    decision(4, '저는 게살에 알레르기 반응이 있어요. 소시지만 넉넉하게 정해서 지체 없이 담아 주세요.',
        {'route': 'task', 'restrictions': [{'target': '게살', 'reason': 'allergy', 'action': 'add'}],
         'order': {'toppings': {'소시지': 'high'}}, 'commit': True}, section='meat',
        required={'restriction.게살': 'allergy', 'toppings.소시지': 'high', 'execute': True})
    decision(5, '이번 토핑에는 페퍼론치노를 제외하고 치즈 소량만 선택해서 즉시 담도록 해 주세요.',
        {'route': 'task', 'order': {'toppings': {'페퍼론치노': 'none', '치즈': 'low'}}, 'commit': True},
        section='extra', selected=order(toppings={'페퍼론치노': 'normal'}),
        required={'toppings.치즈': 'low', 'pepper_execution': False, 'execute': True})
    decision(6, '조금 전 담자는 실행 확인 질문은 취소해 주세요. 고른 소시지 양은 유지하고 아직 시작하지 마세요.',
        {'route': 'task', 'confirmation': 'reject'}, section='meat',
        selected=order(toppings={'소시지': 'high'}),
        pending={'type': 'execution', 'source': 'preselected', 'section': 'meat',
                 'targets': ['소시지'], 'items': {'소시지': 'high'}},
        required={'toppings.소시지': 'high', 'pending': None, 'execute': False})
    decision(7, '지금 바로 담지는 말아 주세요. 선택 내용은 지금 상태로 두면 됩니다.',
        {'route': 'task'}, selected=order(noodle_type='얇은면', noodle_portion='normal', sauce='토마토'),
        required={'execute': False, 'order_unchanged': True})
    decision(8, '처음에 말씀드린 소량 버섯 주문을 이제 곧바로 진행해 주세요.',
        {'route': 'task', 'commit': True}, section='veggie', selected=order(toppings={'버섯': 'low'}),
        previous=history(('user', '버섯은 조금만 선택해 둘게요.'),
                         ('assistant', '버섯 선택을 기록했어요. 아직 시작하지 않았어요.')),
        required={'toppings.버섯': 'low', 'execute': True})

    q_specs = [
        ('진하고 느끼한 소스를 원해요. 면 얘기 말고 그 취향에 맞는 소스 하나를 골라 주세요.',
         {'sauce': '크림'}, [], []),
        ('느끼함은 피하고 새콤한 소스 하나로 권해 주시겠어요? 면 추천은 필요 없어요.',
         {'sauce': '토마토'}, [], []),
        ('매운 음식을 못 먹는다는 제 말을 기억해서, 소스 후보까지 포함해 순한 쪽으로 알려 주세요.',
         {'sauce': '크림'}, [{'target': '매운맛', 'reason': 'cannot_eat'}], [{'value': '느끼한 소스 선호'}]),
        ('게살 알레르기를 고려해서 고기류 중 안전한 것을 한 가지, 양은 보통으로 추천해 주세요.',
         {'toppings.소시지': 'normal'}, [{'target': '게살', 'reason': 'allergy'}], []),
        ('아직 담지는 말고 채소 중 양파 하나만 새 후보로 권해 주세요. 양은 따로 정하지 않았어요.',
         {'toppings.양파': 'normal'}, [], []),
        ('버섯 하나로 새 채소 추천을 부탁드려요. 추가적인 수량 취향은 없습니다.',
         {'toppings.버섯': 'normal'}, [], []),
        ('페퍼론치노를 못 먹으니 토핑 후보는 치즈로 하나만, 양은 보통으로 제안해 주세요.',
         {'toppings.치즈': 'normal'}, [{'target': '페퍼론치노', 'reason': 'cannot_eat'}], []),
        ('크림과 오일은 먹을 수 없어요. 남은 소스 한 가지로만 새 추천을 부탁드립니다.',
         {'sauce': '토마토'}, [{'target': '크림', 'reason': 'cannot_eat'},
                              {'target': '오일', 'reason': 'cannot_eat'}], [])]
    for index, (text, fields, restrictions, preferences) in enumerate(q_specs, 1):
        proposal = {'sauce': None, 'noodle_type': None, 'noodle_portion': None, 'toppings': {}}
        for key, value in fields.items():
            if key.startswith('toppings.'):
                proposal['toppings'][key.split('.', 1)[1]] = value
            else:
                proposal[key] = value
        cases.append({'agent': 'recommendation', 'strict_required_proposal_fields': fields,
            'record': {'id': f'QG-NEW-Q-{index:02}', 'input': {
                'current_user_text': text, 'event': None, 'input_source': 'user',
                'order': order(restrictions=copy.deepcopy(restrictions)), 'pending': None,
                'preferences': preferences, 'recent_history': [], 'recommendation_action': 'request',
                'recommendation_scope': scope(), 'restrictions': restrictions, 'robot_state': robot('noodle'),
                'supported_domain': {'amounts': ['low', 'normal', 'high'], 'noodle_types': ['얇은면', '넓은면'],
                    'sauces': ['크림', '토마토', '오일'], 'toppings': ['양파', '버섯', '소시지', '게살', '치즈', '페퍼론치노']}},
                'target': {'proposal': proposal, 'reason_tags': []},
                'metadata': {'dialogue_group': f'QG-NEW-Q-{index:02}', 'oracle_review': 'pending_independent_review',
                    'required_proposal_fields': fields, 'proposal_constraints': {'allowed_fields': list(fields),
                                                                              'exact_field_count': len(fields)}}}})

    p_specs = [
        ('소량으로 고른 양파를 보통 양으로 수정만 해 주세요. 지금 담지는 마시고요.',
         'veggie', {'양파': 'normal'}, {'toppings.양파': 'normal'}, False),
        ('버섯 양만 보통으로 고쳤으니 양파를 포함한 선택 채소 전체를 즉시 진행해 주세요. 양파는 앞선 소량 그대로예요.',
         'veggie', {'버섯': 'normal', '양파': 'low'}, {'toppings.버섯': 'normal'}, True),
        ('소시지 넉넉한 선택은 유지하되 잠깐 기다려 주세요. 담기는 아직 시작하지 마세요.',
         'meat', {'소시지': 'high'}, {}, False),
        ('치즈는 소량으로 결정했어요. 페퍼론치노 없이 그 치즈만 곧바로 담아 주세요.',
         'extra', {'치즈': 'low'}, {'toppings.치즈': 'low'}, True),
        ('양파는 소량으로 이미 골랐지만 아직 로봇이 담은 적은 없어요. 선택만 확인해 주세요.',
         'veggie', {'양파': 'low'}, {}, False),
        ('새 추천은 하지 마세요. 버섯을 넉넉히 고른 내용만 알려 주시면 됩니다.',
         'veggie', {'버섯': 'high'}, {}, False),
        ('게살 알레르기를 저장했으니, 고른 소시지 보통 양을 지금 시작해 주세요.',
         'meat', {'소시지': 'normal'}, {}, True),
        ('앞에서 치즈를 넉넉하게 골랐다가 방금 소량으로 바꿨어요. 수정 내용만 확인해 주세요.',
         'extra', {'치즈': 'low'}, {'toppings.치즈': 'low'}, False)]
    for index, (text, section, toppings, changes, execute) in enumerate(p_specs, 1):
        restrictions = [{'target': '게살', 'reason': 'allergy'}] if index == 7 else []
        cases.append({'agent': 'response', 'record': {'id': f'QG-NEW-P-{index:02}', 'route': 'task',
            'input': {'user_text': text, 'confirmed_order': order(toppings=toppings, restrictions=restrictions),
                'applied_this_turn': {'order_changes': changes, 'preference_added': [], 'preference_removed': [],
                                     'restriction_added': [], 'restriction_removed': []},
                'execution_authorized': execute,
                'starting_now': [{'item': item, 'amount': amount} for item, amount in toppings.items()]
                                if execute else [],
                'future_changes': [], 'event': None, 'input_source': 'user',
                'pending': None, 'next_prompt': None, 'preferences': [], 'recent_action_history': [],
                'recent_history': [], 'recommendation_result': None, 'recommendation_scope': scope(),
                'robot_state': robot(section), 'policy': {'status': 'pass', 'execute': execute,
                    'reason': 'commit' if execute else 'state_update_only', 'conflicts': []}},
            'metadata': {'dialogue_group': f'QG-NEW-P-{index:02}', 'oracle_review': 'pending_independent_review',
                         'human_review_required': True, 'no_new_recommendation': True,
                         'no_completion_claim': True, 'no_execution_question': True}}})
    return cases


def freeze(destination):
    """CPU-only membership/schema/leakage freeze, not semantic quality approval."""
    import benchmark
    import quality_gate
    from jsonschema import Draft202012Validator
    corpus = []
    for agent in quality_gate.AGENTS:
        for split in ('train', 'validation'):
            corpus += quality_gate.rows(benchmark.RELEASE / f'dataset/records/{agent}/{split}.jsonl')
    input_hashes = {quality_gate.digest(row['input']) for row in corpus}
    speech = set().union(*(quality_gate.utterances(row) for row in corpus))
    group_ids = set().union(*(quality_gate.groups(row) for row in corpus))
    cases = authored_cases()
    leak_checks = []
    for case in cases:
        agent, row = case['agent'], case['record']
        if agent != 'response':
            schema = benchmark.read(benchmark.RELEASE / f'dataset/contract/{agent}_schema.json')
            Draft202012Validator(schema).validate(row['target'])
            for alternative in row.get('metadata', {}).get('decision_target_alternatives') or []:
                Draft202012Validator(schema).validate(alternative)
        collision = {'input_match': quality_gate.digest(row['input']) in input_hashes,
                     'speech_matches': sorted(quality_gate.utterances(row) & speech),
                     'group_matches': sorted(quality_gate.groups(row) & group_ids)}
        if any(collision.values()):
            raise ValueError(f'new holdout overlaps training/validation: {row["id"]} {collision}')
        leak_checks.append({'id': row['id'], **collision})
    destination = benchmark.output_dir(destination)
    benchmark.write_new(destination / 'NEW_FUNCTIONAL_CASES.json', {'rows': cases})
    receipt = {'status': 'MEMBERSHIP_SCHEMA_EXACT_LEAK_CHECK_PASS_SEMANTIC_REVIEW_PENDING',
        'counts': {agent: sum(c['agent'] == agent for c in cases) for agent in quality_gate.AGENTS},
        'cases_sha256': benchmark.sha(destination / 'NEW_FUNCTIONAL_CASES.json'),
        'source_manifest_sha256': benchmark.MANIFEST_SHA, 'leak_checks': leak_checks,
        'used_for_training': False, 'functional_predictions': 'not_yet_run',
        'semantic_near_duplicate_risk': 'requires_independent_group_and_oracle_review',
        'trained_required_demo': 'must_be_reported_separately_not_blind_holdout'}
    benchmark.write_new(destination / 'NEW_HOLDOUT_RECEIPT.json', receipt)
    return receipt


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(freeze(args.output))
