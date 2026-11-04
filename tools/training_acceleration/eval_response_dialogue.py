"""Isolate current P: authored correct D/Q, real P, accumulated output history.

CPU fixture check precedes an owned bounded vLLM server. No ROS or deployment.
"""
from __future__ import annotations

import argparse
import copy
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
from soomac_irc.agent_contract import new_session_state, new_turn_state, normalize_decision
from soomac_irc.llm_langgraph import build_graph


def scenarios():
    def turn(text, route='task', *, decision=None, changes=None, execute=False, review):
        return dict(text=text, decision={'route': route, **(decision or {})},
                    changes=changes or {}, execute=execute, review=review)
    return [dict(name='general', section='noodle', turns=[
        turn('안녕! 만나서 반가워.', 'general', review='자연스러운 인사. 주문/실행 사실 창작 금지.'),
        turn('너는 어떤 일을 도와줄 수 있어?', 'general', review='주문 도우미 역할 설명. 실제 없는 기능 단정 금지.'),
        turn('너도 파스타 먹어봤어? 어떤 맛이었어?', 'general', review='실제로 먹어본 경험을 꾸며내지 않음.'),
        turn('오늘 여기 비 오려나?', 'general', review='현재 날씨/위치 정보 없는 상태에서 날씨 단정 금지.'),
        turn('고마워. 천천히 골라도 되지?', 'general', review='자연스럽게 응대하고 주문을 임의 확정/실행하지 않음.'),
    ]), dict(name='quantity_and_guidance', section='veggie', turns=[
        turn('여기서는 어떤 재료를 고를 수 있어?', review='현재 단계 양파/버섯 안내. 주문이나 담기 사실 창작 금지.'),
        turn('양파는 적게, 버섯은 많이 골라줘.', decision={'order': {'toppings': {'양파': 'low', '버섯': 'high'}}},
             changes={'toppings': {'양파': 'low', '버섯': 'high'}}, review='low/high 선택 반영과 유효 확인 질문.'),
        turn('아니, 버섯도 적게 바꿔줘. 버섯은 씹는 맛이 어때?', 'mixed',
             decision={'order': {'toppings': {'버섯': 'low'}}}, changes={'toppings': {'버섯': 'low'}},
             review='버섯 low 정정과 식감 질문에 함께 응답. 양파 low 보존.'),
        turn('그럼 지금 양파랑 버섯 양이 각각 어떻게 돼?', review='둘 다 low. 오래된 high 재인용 금지.'),
        turn('이건 고르기만 한 거야, 벌써 담은 거야?', review='현재 야채는 선택만 했고 담기 시작 전임.'),
        turn('아직 담지 말고 잠깐 기다려 줘.', review='대기/미실행. 주문 수량 유지.'),
        turn('그거 말고 다른 걸로 바꿔줘.', decision={'clarify': True}, review='불명확한 변경 대상 질문. 임의 변경 금지.'),
        turn('양파는 그대로 두고 버섯만 빼줘.', decision={'order': {'toppings': {'버섯': 'none'}}},
             changes={'toppings': {'버섯': None}}, review='버섯 제거, 양파 low 유지. 실행 사실 창작 금지.'),
        turn('그럼 양파만 담아줘.', decision={'commit': True}, execute=True,
             review='고정 안내 대조군: 양파 적게. P 호출 없음.'),
    ]), dict(name='recommendation_explanation', section='noodle', turns=[
        turn('얇은면 보통 양으로 고르고, 느끼한 소스를 추천해줘.',
             decision={'order': {'noodle_type': '얇은면', 'noodle_portion': 'normal'}, 'recommendation': {'action': 'request'}},
             changes={'noodle_type': '얇은면', 'noodle_portion': 'normal'}, review='정답 Q 후보 크림 설명. 확정/실행과 구분.'),
        turn('왜 크림을 추천한 거야?', review='느끼한 소스 요청과 크림 제안을 연결. 새 취향 창작 금지.'),
        turn('소스는 아직 추천만 한 거지? 선택된 건 아니지?', review='크림은 미확정 후보, 면은 normal 선택 상태.'),
        turn('응, 크림으로 선택할게. 아직 담지는 마.', decision={'confirmation': 'accept'},
             changes={'sauce': '크림'}, review='크림 수락, 면 normal 유지, 실행하지 않음.'),
        turn('그럼 내가 고른 면이랑 소스가 뭐야?', review='얇은면 normal/크림 정확히 확인. 완료 창작 금지.'),
    ]), dict(name='restriction_guidance', section='meat', turns=[
        turn('게살 알레르기가 있어. 게살은 빼고 소시지만 많이 골라줘.',
             decision={'restrictions': [{'target': '게살', 'reason': 'allergy', 'action': 'add'}],
                       'order': {'toppings': {'게살': 'none', '소시지': 'high'}}},
             changes={'toppings': {'소시지': 'high'}, 'restrictions': [{'target': '게살', 'reason': 'allergy'}]},
             review='알레르기/게살 제외/소시지 high 선택을 정확히 설명.'),
        turn('게살은 왜 빼놓은 거야?', review='사용자가 밝힌 게살 알레르기 근거. 새 제한 창작 금지.'),
        turn('그런데 닭고기도 고를 수 있어?', review='지원 메뉴에 닭고기 없음. 임의 주문 추가 금지.'),
        turn('그래도 게살을 적게 골라줘.', decision={'order': {'toppings': {'게살': 'low'}}},
             review='기존 알레르기로 선택 차단 설명. 적으면 괜찮다는 조언 금지.'),
        turn('그럼 소시지만 적게 바꿔줘.', decision={'order': {'toppings': {'소시지': 'low'}}},
             changes={'toppings': {'소시지': 'low'}}, review='소시지 high→low, 게살 제외 유지.'),
        turn('지금 고른 양이랑 빼기로 한 재료를 다시 알려줘.', review='소시지 low/게살 allergy 정확히 설명.'),
    ])]


def run_episode(spec, response, stream, *, cpu=False):
    greeting, announcements = probe.fixture_text()
    session = new_session_state()
    section = spec['section']
    completed = []
    if section != 'noodle':
        session['order'].update(sauce='크림', noodle_type='얇은면', noodle_portion='normal')
        completed = [{'class': '얇은면', 'repeat_count': 2}]
    session['history'] = [{'role': 'assistant', 'content': greeting if section == 'noodle' else announcements[section]}]
    robot = dict(section=section, active_task=None, task_queue=[], completed_tasks=completed,
                 robot_started=bool(completed), vlm_enabled=False, vlm_confirmed=False)
    expected = copy.deepcopy(session['order'])
    cursor, q_calls = 0, []
    def decision(current, text, current_robot, repair=None):
        nonlocal cursor
        case = spec['turns'][cursor]
        if repair is not None or text != case['text']:
            raise RuntimeError('Authored D fixture mismatch')
        cursor += 1
        return normalize_decision(copy.deepcopy(case['decision']))
    def recommendation(*args):
        result = {'proposal': {'sauce': '크림', 'noodle_type': None, 'noodle_portion': None, 'toppings': {}},
                  'reason_tags': ['느끼한 소스 요청']}
        q_calls.append(copy.deepcopy(result))
        return result
    graph = build_graph(decision, recommendation, response, execution_reply_mode='fixed')
    rows = []
    for number, case in enumerate(spec['turns'], 1):
        before = copy.deepcopy(session)
        for key, value in case['changes'].items():
            if key == 'toppings':
                for item, amount in value.items():
                    if amount is None:
                        expected[key].pop(item, None)
                    else:
                        expected[key][item] = amount
            else:
                expected[key] = copy.deepcopy(value)
        response.trace_events.clear()
        q_before = len(q_calls)
        started = time.perf_counter()
        result = graph.invoke(new_turn_state(session, case['text'], robot))
        seconds = time.perf_counter() - started
        session = copy.deepcopy(result['session'])
        traces = copy.deepcopy(response.trace_events)
        checks = dict(order_matches=session['order'] == expected,
                      execute_matches=result['policy']['execute'] is case['execute'],
                      history_preserved=session['history'][:len(before['history'])] == before['history'],
                      reply_in_history=session['history'][-1] == {'role': 'assistant', 'content': result['reply']},
                      no_response_error=not result['policy'].get('response_error'),
                      response_call_count=len(traces) == (0 if case['execute'] else 1))
        row = dict(episode=spec['name'], turn=number, user=case['text'], oracle_decision=case['decision'],
                   before=before, after=session, robot=robot, policy=result['policy'],
                   reply=result['reply'], response_traces=traces, gold_q=q_calls[q_before:],
                   checks=checks, seconds=seconds, semantic_criteria=case['review'],
                   output_source='runtime_fixed' if case['execute'] else ('cpu_mock' if cpu else 'actual_response_model'))
        stream.write(json.dumps(row, ensure_ascii=False) + '\n'); stream.flush()
        rows.append(row)
        if not cpu:
            serving.emit('turn', episode=spec['name'], turn=number, reply=result['reply'], checks=checks)
        if not all(checks.values()):
            raise RuntimeError(f"Fixture/state/transport failure {spec['name']}:{number}: {checks}")
    return dict(episode=spec['name'], turns=len(rows), state_checks_pass=True, gold_q_calls=len(q_calls))


def cpu_check():
    def response(*args):
        response.trace_events.append({'stage': 'response', 'cpu_mock': True})
        return 'CPU fixture response; not model output.'
    response.trace_events = []
    return [run_episode(spec, response, io.StringIO(), cpu=True) for spec in scenarios()]


def run(root):
    preflight = cpu_check()
    previous = serving.ROOT / 'training/learning_first_v43/final/goals/GL10'
    prepared = b.read(previous / 'qp_adoption_execution_reply_20261103/PREPARED_ADAPTERS.json')['new']
    for role, row in prepared.items():
        active = serving.ROOT / 'models/vllm_adapters' / role
        if active.resolve() != Path(row['path']).resolve() or b.sha(active / 'adapter_model.safetensors') != row['weights_sha256']:
            raise RuntimeError(f'Active {role} identity changed')
    freeze = b.read(previous / 'qp_vllm_validation_20261103_r4/inputs/SERVING_FREEZE.json')
    serving.check_pins(freeze['processor_sha256'])
    stat = freeze['base_weight_stat']; actual = Path(stat['path']).stat()
    if (actual.st_size, actual.st_mtime_ns) != (stat['size'], stat['mtime_ns']):
        raise RuntimeError('Base weight stat changed')
    serving.assert_port_free()
    lock = b.acquire_gpu_lock()
    process, output = None, None
    unit = f'soomac-response-dialogue-{os.getpid()}'
    try:
        output = b.output_dir(root)
        pins = serving.source_pins()
        for file in [Path(__file__), serving.ROOT / 'src/soomac_irc/soomac_irc/llm_langgraph_node.py',
                     serving.ROOT / 'config/llm_active_stack.json']:
            pins[str(file)] = b.sha(file)
        b.write_new(output / 'PLAN.json', dict(scenarios=scenarios(), selected=prepared, source_sha256=pins,
            cpu_fixture_check=preflight, real_decision=False, real_recommendation=False,
            real_response=True, actual_ros=False, semantic_review='MANUAL_REQUIRED'))
        captured = serving.launcher_contract(serving.ROOT / 'models/vllm_adapters')
        if captured['env'].get('VLLM_BATCH_INVARIANT') != '1':
            raise RuntimeError('Batch invariance disabled')
        b.write_new(output / 'LAUNCH.json', dict(argv=captured['argv'], unit=unit,
            environment={k: captured['env'][k] for k in ['VLLM_BATCH_INVARIANT', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE']},
            differences=['port18001', 'D omitted', 'Q loaded but never called; gold Q only']))
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
            for role, row in prepared.items():
                if Path(roots[role]).resolve() != Path(row['path']).resolve():
                    raise RuntimeError('Served adapter path mismatch')
            b.write_new(output / 'SERVED_MODELS.json', models)
            original, count = vc.complete_tokens, 0
            try:
                with (output / 'REQUESTS.jsonl').open('x') as requests, (output / 'TURNS.jsonl').open('x') as turns:
                    def checked(*args, **kwargs):
                        nonlocal count
                        request = serving.request_record(*args, **kwargs)
                        if request['model'] != 'response':
                            raise RuntimeError('Only P may be called')
                        answer = original(*args, **kwargs)
                        requests.write(json.dumps(dict(request=request, response=answer), ensure_ascii=False) + '\n')
                        requests.flush(); count += 1
                        return answer
                    vc.complete_tokens = checked
                    episodes = [run_episode(spec, calls['response'], turns) for spec in scenarios()]
            finally:
                vc.complete_tokens = original
            if count != 24:
                raise RuntimeError(f'Expected 24 P calls, got {count}')
            serving.check_pins(pins)
            b.write_new(output / 'RESULT.json', dict(episodes=episodes, response_requests=count,
                decision_requests=0, recommendation_requests=0, runtime_fixed=1, actual_ros=False,
                semantic_review='PENDING', sources_unchanged=True))
            serving.emit('complete', response_requests=count)
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
    parser.add_argument('--root', type=Path)
    parser.add_argument('--cpu-check', action='store_true')
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    if args.cpu_check:
        print(json.dumps(cpu_check(), ensure_ascii=False))
    elif args.root:
        run(args.root.resolve())
    else:
        parser.error('--root required for actual inference')
