"""Actual P/Q continuation with explicit correct-D assumptions for references.

These are new authored scenario variations, not measurements of Decision accuracy.
No production code/config/adapter, ROS, VLM, training or remote work is changed.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

import benchmark as b
import eval_vllm_epochs as serving
import manual_scenario_stack_probe as probe
from soomac_irc.agent_contract import new_session_state, new_turn_state, normalize_decision
from soomac_irc.llm_langgraph import build_graph
from soomac_irc.llm_policy import is_future_recommendation_pending, section_execution_items

ADOPTION = serving.ROOT / 'training/learning_first_v43/final/goals/GL10/qp_adoption_execution_reply_20261103'


def scenarios():
    def turn(text, decision, changes=None, plans=(), *, require_cream_candidate=False):
        return dict(text=text, decision={'route': 'task', **decision},
                    expected_changes=changes or {}, expected_plans=[
                        {'class': name, 'repeat_count': count} for name, count in plans],
                    require_cream_candidate=require_cream_candidate)
    noodle = {'noodle_type': '얇은면', 'noodle_portion': 'normal'}
    cream = dict(noodle, sauce='크림')
    allergy = [{'target': '게살', 'reason': 'allergy'}]
    add_allergy = [{'target': '게살', 'reason': 'allergy', 'action': 'add'}]
    initial = turn('면은 얇은면으로 보통 양으로 주고, 소스는 느끼한 걸로 추천해줘.',
        {'order': noodle, 'recommendation': {'action': 'request'}}, noodle)
    explicit = turn('면은 얇은면으로 보통 양, 소스는 크림으로 골라줘.', {'order': cream}, cream)
    return [
        {'name': 'recommendation_and_short_execution', 'turns': [
            initial,
            turn('그걸로 담아줘.', {'confirmation': 'accept', 'commit': True},
                 {'sauce': '크림'}, [('noodle_thin', 2)], require_cream_candidate=True),
            turn('양파랑 버섯 둘 다 적게 골라줘.', {'order': {'toppings': {'양파': 'low', '버섯': 'low'}}},
                 {'toppings': {'양파': 'low', '버섯': 'low'}}),
            turn('버섯만 보통 양으로 바꿔줘.', {'order': {'toppings': {'버섯': 'normal'}}},
                 {'toppings': {'버섯': 'normal'}}),
            turn('응, 담아줘.', {'commit': True}, plans=[('onion', 1), ('mushroom', 2)]),
            turn('게살 알레르기가 있어요.', {'restrictions': add_allergy}, {'restrictions': allergy}),
            turn('소시지만 많이 골라줘.', {'order': {'toppings': {'소시지': 'high'}}},
                 {'toppings': {'소시지': 'high'}}),
            turn('그걸로 담아줘.', {'commit': True}, plans=[('sausage', 3)]),
            turn('페퍼론치노는 빼고 치즈만 적게 골라줘.',
                 {'order': {'toppings': {'페퍼론치노': 'none', '치즈': 'low'}}},
                 {'toppings': {'페퍼론치노': None, '치즈': 'low'}}),
            turn('담아줘.', {'commit': True}, plans=[('cheese', 1)]),
        ]},
        {'name': 'correction_then_reference', 'turns': [
            explicit,
            turn('담아줘.', {'commit': True}, plans=[('noodle_thin', 2)]),
            turn('양파는 적게, 버섯은 많이 골라줘.', {'order': {'toppings': {'양파': 'low', '버섯': 'high'}}},
                 {'toppings': {'양파': 'low', '버섯': 'high'}}),
            turn('아니, 버섯도 적게 바꿔줘.', {'order': {'toppings': {'버섯': 'low'}}},
                 {'toppings': {'버섯': 'low'}}),
            turn('그럼 그걸로 담아줘.', {'commit': True}, plans=[('onion', 1), ('mushroom', 1)]),
            turn('게살 알레르기가 있으니까 소시지만 많이 골라줘.',
                 {'restrictions': add_allergy, 'order': {'toppings': {'게살': 'none', '소시지': 'high'}}},
                 {'restrictions': allergy, 'toppings': {'게살': None, '소시지': 'high'}}),
            turn('아니, 소시지는 적게 바꿔줘.', {'order': {'toppings': {'소시지': 'low'}}},
                 {'toppings': {'소시지': 'low'}}),
            turn('응, 그걸로 담아줘.', {'commit': True}, plans=[('sausage', 1)]),
            turn('치즈만 적게 골라줘. 페퍼론치노는 빼줘.',
                 {'order': {'toppings': {'치즈': 'low', '페퍼론치노': 'none'}}},
                 {'toppings': {'치즈': 'low', '페퍼론치노': None}}),
            turn('치즈는 보통으로 바꿔줘.', {'order': {'toppings': {'치즈': 'normal'}}},
                 {'toppings': {'치즈': 'normal'}}),
            turn('그걸로 담아줘.', {'commit': True}, plans=[('cheese', 2)]),
        ]},
        {'name': 'unresolved_reference_control', 'turns': [
            explicit,
            turn('담아줘.', {'commit': True}, plans=[('noodle_thin', 2)]),
            turn('양파는 적게, 버섯은 많이 골라줘.', {'order': {'toppings': {'양파': 'low', '버섯': 'high'}}},
                 {'toppings': {'양파': 'low', '버섯': 'high'}}),
            turn('그거 말고 다른 걸로 바꿔줘.', {'clarify': True}),
            turn('응.', {'clarify': True}),
            turn('버섯만 빼고, 양파는 그대로 담아줘.',
                 {'order': {'toppings': {'버섯': 'none'}}, 'commit': True},
                 {'toppings': {'버섯': None}}, [('onion', 1)]),
        ]},
    ]


def run_episode(spec, calls, stream, *, execution_reply_mode='model'):
    greeting, announcements = probe.fixture_text()
    constants = probe.fixture_constants()
    session, expected = new_session_state(), new_session_state()['order']
    session['history'].append({'role': 'assistant', 'content': greeting})
    robot = dict(section='noodle', active_task=None, task_queue=[], completed_tasks=[],
                 robot_started=False, verification_history=[], vlm_enabled=False, vlm_confirmed=False)
    cursor = 0
    def decision(current, text, current_robot, repair=None):
        nonlocal cursor
        if repair is not None:
            raise RuntimeError('Correct-D oracle unexpectedly requested repair')
        case = spec['turns'][cursor]
        if text != case['text']:
            raise RuntimeError('User text drift')
        if case['require_cream_candidate']:
            pending = current.get('pending') or {}
            candidate = pending.get('candidate') or {}
            if pending.get('type') != 'recommendation' or candidate.get('sauce') != '크림' or candidate.get('toppings'):
                raise RuntimeError('Actual recommendation is not the cream-only referent; stop, do not inject replacement state')
        external = copy.deepcopy(case['decision'])
        decision.last_external_output = external
        value = normalize_decision(external)
        decision.trace_events.append(dict(stage='decision', gold_injected=True,
            oracle_origin='authored_variation_reviewed_against_current_contract',
            user_text=text, model_input=dict(session=copy.deepcopy(current), robot_state=copy.deepcopy(current_robot)),
            external=copy.deepcopy(external), parsed=copy.deepcopy(value)))
        cursor += 1
        return value
    decision.trace_events = []
    graph = build_graph(decision, calls['recommendation'], calls['response'], execution_reply_mode=execution_reply_mode)
    result_rows = []
    for index, case in enumerate(spec['turns'], 1):
        before, robot_before = copy.deepcopy(session), copy.deepcopy(robot)
        for call in (decision, *calls.values()):
            call.trace_events.clear()
        for field, value in case['expected_changes'].items():
            if field == 'toppings':
                for item, amount in value.items():
                    if amount is None:
                        expected['toppings'].pop(item, None)
                    else:
                        expected['toppings'][item] = amount
            else:
                expected[field] = copy.deepcopy(value)
        started = time.perf_counter()
        result = graph.invoke(new_turn_state(session, case['text'], robot))
        seconds = time.perf_counter() - started
        session = copy.deepcopy(result['session'])
        traces = {name: copy.deepcopy(call.trace_events) for name, call in
                  [('decision', decision), *calls.items()]}
        section = robot['section']
        items = section_execution_items(session['order'], section) if result['policy']['execute'] else []
        plans = [{'class': constants['TOPIC_CLASS_NAMES'][item['item']],
                  'repeat_count': constants['AMOUNT_TO_COUNT'][item['amount']]} for item in items]
        checks = dict(order_matches=session['order'] == expected,
            execute_matches=result['policy']['execute'] is bool(case['expected_plans']),
            plans_match=plans == case['expected_plans'],
            history_prefix_preserved=session['history'][:len(before['history'])] == before['history'],
            actual_reply_in_history=session['history'][-2:] == [
                {'role': 'user', 'content': case['text']}, {'role': 'assistant', 'content': result['reply']}],
            no_model_errors=not any(t.get('error') for group in traces.values() for t in group))
        row = dict(episode=spec['name'], turn=index, user_text=case['text'], before=before,
            after=copy.deepcopy(session), robot_before=robot_before, decision=result['decision'],
            policy=result['policy'], reply=result['reply'], traces=traces, actual_plans=plans,
            expected_order=copy.deepcopy(expected), expected_plans=case['expected_plans'], checks=checks,
            passed=all(checks.values()), seconds=seconds, actual_ros=False,
            reply_evidence=probe.response_evidence(traces['response'], result['reply'], spec['name'], str(index)))
        stream.write(json.dumps(row, ensure_ascii=False) + '\n'); stream.flush()
        result_rows.append(row)
        serving.emit('turn', episode=spec['name'], turn=index, reply=result['reply'],
                     plans=plans, passed=row['passed'], seconds=round(seconds, 3))
        if not row['passed']:
            break
        # Same offline physical fixture as the prior evaluation: no ROS or VLM evidence.
        if result['policy']['execute'] and items:
            for item, plan in zip(items, plans):
                task = {'class': item['item'], 'repeat_count': plan['repeat_count']}
                robot['completed_tasks'].append(copy.deepcopy(task))
                session['action_history'].append(dict(type='task_completed', section=section, task=task))
            robot['robot_started'] = True
            if not is_future_recommendation_pending(session.get('pending'), section):
                session['pending'] = None
            next_section = {'noodle': 'veggie', 'veggie': 'meat', 'meat': 'extra', 'extra': 'lid'}[section]
            session['action_history'].append(dict(type='section_transition', section=section,
                next_section=next_section, outcome='completed'))
            robot['section'] = next_section
            if next_section in announcements:
                session['history'].append({'role': 'assistant', 'content': announcements[next_section]})
    return dict(episode=spec['name'], planned=len(spec['turns']), observed=len(result_rows),
                passed=len(result_rows) == len(spec['turns']) and all(x['passed'] for x in result_rows),
                execution_turns=sum(x['policy']['execute'] for x in result_rows))


def run(root):
    serving.check_pins(b.read(ADOPTION / 'FINAL_AUDIT.json')['final_source_sha256'])
    prepared = b.read(ADOPTION / 'PREPARED_ADAPTERS.json')['new']
    for role, row in prepared.items():
        actual = serving.ROOT / 'models/vllm_adapters' / role
        if actual.resolve() != Path(row['path']).resolve() or b.sha(actual / 'adapter_model.safetensors') != row['weights_sha256']:
            raise RuntimeError(f'Active {role} adapter differs from adopted identity')
    old_freeze = b.read(ADOPTION.parent / 'qp_vllm_validation_20261103_r4/inputs/SERVING_FREEZE.json')
    serving.check_pins(old_freeze['processor_sha256'])
    stat = old_freeze['base_weight_stat']; actual = Path(stat['path']).stat()
    if (actual.st_size, actual.st_mtime_ns) != (stat['size'], stat['mtime_ns']):
        raise RuntimeError('Base model stat changed')
    serving.assert_port_free()
    lock = b.acquire_gpu_lock()
    process, output = None, None
    unit = f'soomac-reference-eval-{os.getpid()}'
    try:
        output = b.output_dir(root)
        pins = serving.source_pins()
        pins[str(Path(__file__).resolve())] = b.sha(Path(__file__))
        pins[str(serving.ROOT / 'src/soomac_irc/soomac_irc/llm_langgraph_node.py')] = b.sha(serving.ROOT / 'src/soomac_irc/soomac_irc/llm_langgraph_node.py')
        b.write_new(output / 'PLAN.json', dict(scenarios=scenarios(), selected=prepared, source_sha256=pins,
            decision_model_called=False, actual_ros=False, scope='Authored paraphrases; correct-D assumption; actual P/Q and accumulated history'))
        captured = serving.launcher_contract(serving.ROOT / 'models/vllm_adapters')
        if captured['env'].get('VLLM_BATCH_INVARIANT') != '1':
            raise RuntimeError('Batch invariance must stay enabled')
        b.write_new(output / 'LAUNCH.json', dict(argv=captured['argv'], unit=unit,
            environment={k: captured['env'][k] for k in ['VLLM_BATCH_INVARIANT', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE']},
            intentional_differences=['port18001', 'Decision not loaded']))
        with (output / 'SERVER.log').open('x') as log:
            process = subprocess.Popen(['systemd-run', '--user', '--scope', f'--unit={unit}',
                '-p', 'MemoryMax=40G', '-p', 'MemorySwapMax=0', '--', *captured['argv']],
                cwd=serving.ROOT, env=captured['env'], stdout=log, stderr=subprocess.STDOUT)
            serving.emit('server_starting', unit=unit, pid=process.pid)
            deadline = time.monotonic() + 300
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f'Server exited: {process.returncode}')
                try:
                    with urllib.request.urlopen(serving.ENDPOINT + '/health', timeout=2) as answer:
                        if answer.status == 200:
                            break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError('Startup exceeded 300 seconds')
                time.sleep(2)
            vc, model, processor, calls = serving.client_objects()
            models = vc.served_models(); roots = {m['id']: m.get('root') for m in models}
            if set(roots) != {'gemma4-base', 'response', 'recommendation'}:
                raise RuntimeError('Unexpected served model set')
            vc.check_served_adapters({role: row['source']['path'] for role, row in prepared.items()}, roots)
            b.write_new(output / 'SERVED_MODELS.json', models)
            original = vc.complete_tokens
            counts = {'response': 0, 'recommendation': 0}
            try:
                with (output / 'REQUESTS.jsonl').open('x') as requests, (output / 'TURNS.jsonl').open('x') as turns:
                    def checked(*args, **kwargs):
                        request = serving.request_record(*args, **kwargs)
                        value = original(*args, **kwargs)
                        counts[request['model']] += 1
                        requests.write(json.dumps(dict(request=request, response=value), ensure_ascii=False) + '\n')
                        requests.flush()
                        return value
                    vc.complete_tokens = checked
                    episodes = [run_episode(spec, calls, turns) for spec in scenarios()]
            finally:
                vc.complete_tokens = original
            serving.check_pins(pins)
            b.write_new(output / 'RESULT.json', dict(episodes=episodes, requests=counts,
                actual_ros=False, decision_model_called=False, semantic_review='PENDING'))
            serving.emit('complete', episodes=episodes, requests=counts)
    except BaseException as exc:
        if output is not None:
            b.write_new(output / 'FAILURE.json', dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        try:
            if process is not None:
                stop = subprocess.run(['systemctl', '--user', 'stop', unit + '.scope'], timeout=20,
                    capture_output=True, text=True)
                process.wait(timeout=20)
                b.write_new(output / 'CLEANUP.json', dict(unit=unit, stop_returncode=stop.returncode,
                    process_returncode=process.returncode))
        finally:
            os.close(lock)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    run(args.root.resolve())
