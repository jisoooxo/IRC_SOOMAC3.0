"""Mutation tests for failures observed in actual Oct-11/GPU logs."""
import pytest

from soomac_irc.agent_contract import normalize_decision
from soomac_irc.evaluation_v43 import (
    grade_decision_prediction, grade_recommendation_prediction, grade_response_prediction,
)


def codes(grade):
    return {item.code for item in grade.violations}


def response_input():
    return {'id': 'response-facts', 'input': {
        'recent_history': [], 'preferences': [], 'pending': None,
        'confirmed_order': {'toppings': {'양파': 'low', '버섯': 'low'}, 'restrictions': []},
        'applied_this_turn': {}, 'execution_authorized': False,
        'robot_state': {'completed_tasks': [{'class': '얇은면', 'repeat_count': 1}]},
    }}


@pytest.mark.parametrize('prediction', [
    {'route': 'task'},
    {'route': 'task', 'order': {'toppings': {'양파': 'low', '버섯': 'low'}}, 'commit': True},
    {'route': 'task', 'order': {'toppings': {'양파': 'low', '버섯': 'low', '게살': 'normal'}}},
])
def test_dataset_target_rejects_omission_extra_commit_and_invented_item(prediction):
    case = {'id': 'revision', 'target': {'route': 'task', 'order': {'toppings': {'양파': 'low', '버섯': 'low'}}}}
    assert 'decision_target_mismatch' in codes(grade_decision_prediction(case, prediction))


def test_normalized_defaults_and_option_order_do_not_cause_false_failure():
    target = {'route': 'task', 'restrictions': [
        {'target': '게살', 'reason': 'allergy', 'action': 'add'},
        {'target': '소시지', 'reason': 'allergy', 'action': 'add'},
    ]}
    prediction = normalize_decision(target)
    prediction['restriction_options'].reverse()
    assert grade_decision_prediction({'target': target}, prediction).passed


def test_normalized_general_defaults_are_not_state_mutations():
    target = {'route': 'general'}
    assert grade_decision_prediction({'target': target}, normalize_decision(target)).passed


def test_reviewed_decision_alternatives_remain_supported():
    case = {'target': {'route': 'task', 'confirmation': 'accept'},
            'input': {'pending': {'type': 'execution'}},
            'metadata': {'decision_target_alternatives': [
                {'route': 'task', 'confirmation': 'accept'}, {'route': 'task', 'commit': True}]}}
    assert grade_decision_prediction(case, {'route': 'task', 'commit': True}).passed


def test_one_requested_two_recommended_fails_but_alternative_item_passes():
    case = {'input': {'order': {'toppings': {}}, 'recommendation_scope': {
        'sections': [{'fields': ['toppings.게살', 'toppings.소시지']}] }},
        'target': {'proposal': {'toppings': {'소시지': 'normal'}}},
        'metadata': {'scenario_family': 'one_of'}}
    both = {'proposal': {'toppings': {'소시지': 'normal', '게살': 'low'}}}
    assert 'proposal_cardinality' in codes(grade_recommendation_prediction(case, both))
    assert grade_recommendation_prediction(case, {'proposal': {'toppings': {'게살': 'high'}}}).passed


@pytest.mark.parametrize('reply,code', [
    ('실행 확인이 남아 있어요. 담아드릴까요?', 'invented_execution_pending'),
    ('게살 알레르기 제한을 새로 저장했어요.', 'false_restriction_update'),
    ('아까 소시지를 좋아한다고 하셨죠.', 'ungrounded_preference_recall'),
    ('크림 소스를 담았어요.', 'unverified_item_completion'),
    ('양파는 많이 선택했어요. 버섯은 조금 선택했어요.', 'selected_amount_contradiction'),
    ('양파는 많이, 버섯은 조금으로 바꿨어요. 지금 담을까요?', 'selected_amount_contradiction'),
])
def test_actual_false_claim_mutations_fail(reply, code):
    assert code in codes(grade_response_prediction(response_input(), reply))


@pytest.mark.parametrize('reply', [
    '얇은면은 담았어요. 크림 소스는 아직 담지 않았어요.',
    '실행 확인은 남아 있지 않아요.',
    '양파는 적은 양으로 선택했어요. 버섯도 조금 골랐어요.',
    '게살 제한은 이미 유지 중이에요. 새로 저장하지 않았어요.',
    '양파는 많이에서 조금으로 바꿨어요. 버섯은 조금으로 유지해요.',
])
def test_positive_and_negative_controls_remain_valid(reply):
    assert grade_response_prediction(response_input(), reply).passed


def test_other_sauce_completion_does_not_ground_cream():
    case = response_input()
    case['input']['robot_state']['completed_tasks'] = [{'class': '토마토', 'repeat_count': 1}]
    assert 'unverified_item_completion' in codes(grade_response_prediction(case, '크림 소스를 담았어요.'))


def test_assistant_speculation_cannot_support_user_preference_recall():
    case = response_input()
    case['input']['recent_history'] = [{'role': 'assistant', 'content': '소시지를 좋아하시는 것 같아요.'}]
    assert 'ungrounded_preference_recall' in codes(grade_response_prediction(case, '소시지를 좋아한다고 하셨죠.'))
    case['input']['recent_history'] = [{'role': 'user', 'content': '소시지를 좋아해.'}]
    assert grade_response_prediction(case, '소시지를 좋아한다고 하셨죠.').passed


def test_real_update_and_real_pending_are_allowed():
    case = response_input()
    case['input']['pending'] = {'type': 'execution'}
    case['input']['applied_this_turn']['restriction_added'] = [{'target': '게살', 'reason': 'allergy'}]
    assert grade_response_prediction(case, '게살 알레르기 제한을 저장했어요. 실행 확인이 남아 있어요.').passed


def test_one_allowed_topping_cannot_be_replaced_by_unrequested_future_sauce():
    case = {'input': {'recommendation_scope': {'sections': [{'fields': ['toppings.치즈','sauce']}]}, 'order': {}},
            'target': {'proposal': {'toppings': {'치즈': 'normal'}}},
            'metadata': {'proposal_constraints': {'exact_field_count': 1, 'allowed_fields': ['toppings.치즈']}}}
    assert 'proposal_outside_requested_fields' in codes(grade_recommendation_prediction(case, {'proposal': {'sauce': '크림'}}))
    assert grade_recommendation_prediction(case, {'proposal': {'toppings': {'치즈': 'low'}}}).passed
