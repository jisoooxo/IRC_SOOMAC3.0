"""Independent LoRA evaluation. No merging, production swap, ROS or VLM calls."""
from __future__ import annotations
import argparse
import copy
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

import benchmark as b
import eval_vllm_epochs as serving
sys.path.insert(0, str(serving.ROOT / 'src/soomac_irc/test'))
import manual_scenario_stack_probe as probe
import eval_multiturn_reference as ref
import dqp_cold_cases as cold
from soomac_irc.agent_contract import new_session_state, new_turn_state, normalize_decision
from soomac_irc.decision_model import build_decision_model_input
from soomac_irc.llm_langgraph import build_graph
from soomac_irc.llm_policy import section_execution_items, is_future_recommendation_pending
from soomac_irc.response_contract import build_response_model_input, build_recommendation_model_input

BASE = serving.ROOT / 'training/learning_first_v43/final/goals/GL10/dqp_adoption_20261105'
PREVIOUS = BASE.parent / 'qp_vllm_validation_20261103_r4'


def robot_at(section):
    completed = [] if section == 'noodle' else [{'class':'얇은면','repeat_count':2}]
    return dict(section=section,active_task=None,task_queue=[],completed_tasks=completed,
                robot_started=bool(completed),verification_history=[],vlm_enabled=False,vlm_confirmed=False)


def initial_state(section):
    session = new_session_state()
    greeting, prompts = probe.fixture_text()
    if section != 'noodle':
        session['order'].update(sauce='크림',noodle_type='얇은면',noodle_portion='normal')
    session['history'] = [{'role':'assistant','content':greeting if section == 'noodle' else prompts[section]}]
    return session, robot_at(section)


def apply_expected(order, changes):
    for field, value in changes.items():
        if field == 'toppings':
            for item, amount in value.items():
                if amount is None:
                    order[field].pop(item, None)
                else:
                    order[field][item] = amount
        else:
            order[field] = copy.deepcopy(value)


def plans_for(order, section):
    constants = probe.fixture_constants()
    return [{'class':constants['TOPIC_CLASS_NAMES'][item['item']],
             'repeat_count':constants['AMOUNT_TO_COUNT'][item['amount']]}
            for item in section_execution_items(order, section)]


def finish_fixture(session, robot, plans):
    section = robot['section']
    items = section_execution_items(session['order'],section)
    for item, plan in zip(items, plans):
        task = dict(class_=item['item'],repeat_count=plan['repeat_count'])
        task['class'] = task.pop('class_')
        robot['completed_tasks'].append(copy.deepcopy(task))
        session['action_history'].append(dict(type='task_completed',section=section,task=task))
    if not items:
        return
    robot['robot_started'] = True
    if not is_future_recommendation_pending(session.get('pending'), section):
        session['pending'] = None
    following = dict(noodle='veggie',veggie='meat',meat='extra',extra='lid').get(section)
    if following:
        robot['section'] = following
        session['action_history'].append(dict(type='section_transition',section=section,next_section=following,outcome='completed'))
        prompts = probe.fixture_text()[1]
        if following in prompts:
            session['history'].append(dict(role='assistant',content=prompts[following]))


def extended_specs():
    specs = copy.deepcopy(ref.scenarios())
    for spec in specs:
        spec['section'] = 'noodle'
        spec['group'] = 'previously_exposed_reference'
        spec['turns'] = [copy.deepcopy(case) for case in spec['turns']]
        for case in spec['turns']:
            case['changes'] = case.pop('expected_changes')
            case['execute'] = bool(case['expected_plans'])
    for spec in cold.episodes():
        spec['group'] = 'new_authored_selection_suite'
        specs.append(spec)
    return specs


def extended_episode(spec, calls, stream):
    session, robot = initial_state(spec['section'])
    expected = copy.deepcopy(session['order'])
    graph = build_graph(calls['decision'],calls['recommendation'],calls['response'],execution_reply_mode='fixed')
    rows=[]
    for number, case in enumerate(spec['turns'],1):
        before, robot_before = copy.deepcopy(session), copy.deepcopy(robot)
        apply_expected(expected,case['changes'])
        expected_plans = case.get('expected_plans')
        if expected_plans is None:
            expected_plans = plans_for(expected, robot['section']) if case['execute'] else []
        for call in calls.values():
            call.trace_events.clear()
        started=time.monotonic()
        result=graph.invoke(new_turn_state(session,case['text'],robot))
        session=copy.deepcopy(result['session'])
        traces={name:copy.deepcopy(call.trace_events) for name,call in calls.items()}
        plans=plans_for(session['order'],robot['section']) if result['policy']['execute'] else []
        checks=dict(order_matches=session['order']==expected,plans_match=plans==expected_plans,
                    execute_matches=result['policy']['execute'] is case['execute'],
                    history_prefix_preserved=session['history'][:len(before['history'])]==before['history'],
                    actual_reply_in_history=session['history'][-2:]==[dict(role='user',content=case['text']),dict(role='assistant',content=result['reply'])],
                    no_model_errors=not any(t.get('error') for v in traces.values() for t in v))
        row=dict(episode=spec['name'],group=spec['group'],turn=number,user_text=case['text'],before=before,
                 after=copy.deepcopy(session),robot_before=robot_before,decision_final=result['decision'],
                 policy=result['policy'],reply=result['reply'],traces=traces,checks=checks,
                 expected_order=copy.deepcopy(expected),expected_plans=expected_plans,actual_plans=plans,
                 route_expected=case['decision']['route'],route_matches=result['decision']['route']==case['decision']['route'],
                 semantic_criteria=case.get('review'),passed=all(checks.values()),seconds=time.monotonic()-started,
                 physical_simulation_only=True,actual_ros=False,
                 reply_evidence=probe.response_evidence(traces['response'],result['reply'],spec['name'],str(number)))
        stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush();rows.append(row)
        serving.emit('extended_turn',episode=spec['name'],turn=number,passed=row['passed'],reply=row['reply'])
        if not row['passed']:
            break  # No invented successful state after failure.
        if result['policy']['execute']:
            finish_fixture(session,robot,plans)
    return dict(episode=spec['name'],planned=len(spec['turns']),observed=len(rows),
                passed=len(rows)==len(spec['turns']) and all(r['passed'] for r in rows),
                failed_turns=[r['turn'] for r in rows if not r['passed']])


def args_for(case):
    from eval_frozen_cases import call_arguments
    return call_arguments(case)[1]


def capture_gold_cases():
    """CPU-only oracle inputs. Never report these outputs as actual inference."""
    cases=[]
    for spec in cold.episodes():
        session, robot=initial_state(spec['section'])
        for n,turn in enumerate(spec['turns'],1):
            tag=f"COLD-{spec['name']}-{n:02}"
            inp=build_decision_model_input(session,turn['text'],robot)
            cases.append(dict(agent='decision',group='new_authored',record=dict(id=tag+'-D',input=inp,
                target=turn['decision'],metadata=dict(semantic_criteria=turn['review'],authored_before_inference=True))))
            def d(*args,**kwargs):
                if kwargs.get('repair') or len(args)>3 and args[3] is not None:
                    raise RuntimeError('Oracle repair requested')
                d.last_external_output=copy.deepcopy(turn['decision'])
                return normalize_decision(turn['decision'])
            def q(*args):
                return {'proposal':{'sauce':'크림','noodle_type':None,'noodle_portion':None,'toppings':{}},'reason_tags':['느끼한 소스 요청']}
            def p(*args):
                inp=build_response_model_input(*args)
                cases.append(dict(agent='response',group='new_authored',record=dict(id=tag+'-P',input=inp,route=args[8],
                    target={'reply':''},metadata=dict(semantic_criteria=turn['review'],authored_before_inference=True))))
                # Explicit neutral fixture; not generated, not used in actual closed loop.
                return '말씀하신 내용을 확인했어요.'
            result=build_graph(d,q,p,execution_reply_mode='fixed').invoke(new_turn_state(session,turn['text'],robot))
            expected=copy.deepcopy(session['order']);apply_expected(expected,turn['changes'])
            if result['session']['order'] != expected or result['policy']['execute'] is not turn['execute']:
                raise RuntimeError(f'Authored oracle conflicts with runtime: {tag}')
            session=copy.deepcopy(result['session'])
            if result['policy']['execute']:
                finish_fixture(session,robot,plans_for(session['order'],robot['section']))
    return cases


def new_q_cases():
    specs=[('고소하고 느끼한 소스 한 가지만 권해 주세요. 면이나 토핑 추천은 빼고요.','noodle',{'sauce':'크림'},[]),
           ('오늘은 새콤한 소스를 하나 추천해 주세요. 소스만 궁금해요.','noodle',{'sauce':'토마토'},[]),
           ('소스 후보로 오일 하나만 제안해 주세요. 선택은 아직 안 할게요.','noodle',{'sauce':'오일'},[]),
           ('새 채소 후보는 양파만, 양은 보통으로 권해 주세요.','veggie',{'toppings.양파':'normal'},[]),
           ('채소 추천을 버섯만 적은 양으로 받고 싶어요.','veggie',{'toppings.버섯':'low'},[]),
           ('게살 알레르기는 지키고, 소시지 하나만 보통 양으로 권해 주세요.','meat',{'toppings.소시지':'normal'},[{'target':'게살','reason':'allergy'}]),
           ('치즈만 많은 양으로 추천해 주세요. 다른 후보는 필요 없어요.','extra',{'toppings.치즈':'high'},[]),
           ('크림과 오일은 먹을 수 없으니 남은 소스만 하나 권해 주세요.','noodle',{'sauce':'토마토'},[{'target':x,'reason':'cannot_eat'} for x in ('크림','오일')])]
    cases=[]
    for n,(text,section,fields,restrictions) in enumerate(specs,1):
        session,robot=initial_state(section);session['order']['restrictions']=restrictions
        source=build_recommendation_model_input(session,{'recommendation':{'action':'request'}},robot,text)
        proposal=dict(sauce=None,noodle_type=None,noodle_portion=None,toppings={})
        for k,v in fields.items():
            if k.startswith('toppings.'):
                proposal['toppings'][k.split('.')[1]]=v
            else:
                proposal[k]=v
        cases.append(dict(agent='recommendation',group='new_authored',record=dict(id=f'COLD-Q-{n:02}',input=source,
            target=dict(proposal=proposal,reason_tags=[]),metadata=dict(required_proposal_fields=fields,
            proposal_constraints=dict(allowed_fields=list(fields),exact_field_count=len(fields))))))
    return cases


def prepare():
    from audit_dqp_eval_inputs import canonical,norm,utterance,RELEASE
    from jsonschema import Draft202012Validator
    import functional_eval
    output=b.output_dir(BASE/'inputs')
    cases=[]
    old_rows=[json.loads(x) for x in (BASE.parent/'fixed_history_vllm_20261104/TURNS.jsonl').read_text().splitlines()]
    for row in old_rows:
        session=row['before'];robot=row['robot_before'];text=row['user_text']
        target=row['traces']['decision'][0]['external']
        cases.append(dict(agent='decision',group='required_and_reference_gold_state',record=dict(
            id=f"PRIOR-{row['episode']}-{row['turn']}-D",input=build_decision_model_input(session,text,robot),target=target,
            metadata=dict(previous_correct_D_state=True))))
    validation=[json.loads(x) for x in (RELEASE/'records/decision/validation.jsonl').read_text().splitlines()]
    structural=[]
    for row in validation:
        case=dict(agent='decision',record=row)
        try:
            args_for(case)
            grade=functional_eval.grade(case,row['target'])
            structural.append(dict(id=row['id'],builder_exact=True,target_contract_pass=grade['automated_pass'],failures=grade['failures']))
        except Exception as exc:
            structural.append(dict(id=row['id'],builder_exact=False,error=str(exc)))
    eligible={r['id'] for r in structural if r.get('builder_exact') and r.get('target_contract_pass')}
    # Predetermined hash order per route, no candidate scores involved.
    for route in ('general','mixed','task'):
        rows=sorted([r for r in validation if r['target']['route']==route and r['id'] in eligible],key=lambda r:b.sha_text(r['id']) if hasattr(b,'sha_text') else __import__('hashlib').sha256(r['id'].encode()).hexdigest())[:20]
        cases += [dict(agent='decision',group='validation_development_sample',record=r) for r in rows]
    with contextlib.redirect_stdout(io.StringIO()):
        cases+=capture_gold_cases()
    cases+=new_q_cases()
    train={a:[json.loads(x) for x in (RELEASE/'records'/a/'train.jsonl').read_text().splitlines()] for a in ('decision','response','recommendation')}
    audit=[]
    for case in cases:
        args_for(case)
        row=case['record'];schema=b.read(RELEASE/'contract'/f"{case['agent']}_schema.json")
        Draft202012Validator(schema).validate(row['target'])
        if case['group']=='new_authored':
            matches=[r['id'] for r in train[case['agent']] if norm(utterance(r))==norm(utterance(row))]
            exact=[r['id'] for r in train[case['agent']] if canonical(r['input'])==canonical(row['input'])]
            audit.append(dict(agent=case['agent'],id=row['id'],normalized_current_text_train=matches,exact_input_train=exact))
    b.write_new(output/'CASES.json',dict(rows=cases))
    b.write_new(output/'EPISODES.json',dict(required=[probe.scenario(True),probe.scenario(False)],extended=extended_specs()))
    b.write_new(output/'INPUT_AUDIT.json',dict(new_cases=audit,decision_validation_contract=structural,
        limits=['Oracle self-grade is a structural consistency check, not independent semantic validation.',
                'New authored suites share task semantics with training; absence of exact text is not proof of semantic independence.',
                'Authored independent P/D inputs have explicitly synthetic neutral assistant history; actual closed-loop uses real outputs.']))
    pins=serving.source_pins()
    for p in (Path(__file__),Path(cold.__file__),serving.ROOT/'src/soomac_irc/soomac_irc/llm_langgraph_node.py'):
        pins[str(p.resolve())]=b.sha(p)
    old=b.read(PREVIOUS/'inputs/SERVING_FREEZE.json')
    serving.check_pins(old['processor_sha256'])
    b.write_new(output/'FREEZE.json',dict(files={p.name:b.sha(p) for p in output.glob('*.json')},
        source_sha256=pins,processor_sha256=old['processor_sha256'],base_weight_stat=old['base_weight_stat'],
        created_before_model_outputs=True,selection_not_pristine_final_test=True))
    serving.emit('prepared',counts={a:sum(c['agent']==a for c in cases) for a in train},
                 validation_contract_failed=sum(not x.get('target_contract_pass') or not x.get('builder_exact') for x in structural))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare'])
    args=parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    prepare()
