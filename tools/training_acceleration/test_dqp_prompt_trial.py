"""CPU checks for diagnostic continuation, not model accuracy tests."""
import io
import json
import dqp_prompt_trial as trial
from soomac_irc.agent_contract import normalize_decision


def test_wrong_actual_state_is_preserved_until_explicit_correction():
    spec=dict(name='continuation',section='veggie',turns=[
        trial.t('양파 적게 골라줘.',{'order':{'toppings':{'양파':'low'}}},{'toppings':{'양파':'low'}},expected_q=False),
        trial.t('지금 양파 양이 뭐야?',expected_q=False),
        trial.t('양파 적게로 고쳐줘.',{'order':{'toppings':{'양파':'low'}}},expected_q=False)])
    decisions=[{'route':'task','order':{'toppings':{'양파':'high'}}},{'route':'task'},
               {'route':'task','order':{'toppings':{'양파':'low'}}}]
    cursor=[0]
    def d(*args,**kwargs):
        value=decisions[cursor[0]];cursor[0]+=1
        return normalize_decision(value)
    def q(*args):raise AssertionError('Unexpected recommendation')
    def p(*args):return 'CPU fixture only'
    for f in (d,q,p):f.trace_events=[]
    stream=io.StringIO()
    result=trial.episode(spec,dict(decision=d,recommendation=q,response=p),stream,'fixed',cpu=True)
    rows=[json.loads(x) for x in stream.getvalue().splitlines()]
    assert result['observed']==3
    assert rows[0]['after']['order']['toppings']['양파']=='high'
    assert rows[1]['before']['order']['toppings']['양파']=='high'
    assert rows[1]['after']['order']['toppings']['양파']=='high'
    assert rows[2]['after']['order']['toppings']['양파']=='low'
    assert rows[2]['state_plan_passed']
    assert rows[2]['prior_state_failure']==1


def test_user_adjudication_has_separate_explicit_execution_turn():
    spec=next(x for x in trial.specs() if x['name']=='accepted_short_then_explicit_execution')
    assert spec['turns'][1]['text']=='그걸로 담아줘.'
    assert spec['turns'][1]['execute'] is False
    assert spec['turns'][1]['decision']=={'route':'task','confirmation':'accept'}
    assert spec['turns'][2]['execute'] is True
    assert spec['turns'][2]['decision']['commit'] is True
