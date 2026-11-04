"""Frozen P/Q evaluation through the installed production vLLM/client stack.

Only new evaluation files and an owned, temporary server on 18001 are used.
No Decision inference, HF model loading, training, deployment, or source writes.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

import benchmark as b
import correct_input_validation as reviewed

ROOT = Path(__file__).resolve().parents[2]
CLIENT = Path('/home/roma/miniconda3/envs/soomac_llm/bin/python')
sys.path.insert(0, str(ROOT / 'src/soomac_irc'))
sys.path.insert(0, str(ROOT / 'tools/vllm_server'))
LAUNCHER = ROOT / 'tools/vllm_server/start_vllm_server.sh'
ENDPOINT = 'http://127.0.0.1:18001'


def emit(event, **fields):
    print(json.dumps(dict(event=event, **fields), ensure_ascii=False), flush=True)


def request_record(model, prompt_token_ids, max_tokens, stop_token_ids, grammar=None):
    if model not in ('response', 'recommendation'):
        raise RuntimeError(f'Forbidden adapter request: {model}')
    return dict(model=model, prompt_token_ids=prompt_token_ids, max_tokens=max_tokens,
                stop_token_ids=stop_token_ids, grammar=grammar)


def source_pins():
    paths = [Path(__file__), Path(reviewed.__file__), LAUNCHER,
             ROOT / 'tools/vllm_server/convert_adapters_for_vllm.py',
             ROOT / 'tools/training_acceleration/functional_eval.py',
             ROOT / 'tools/training_acceleration/eval_gold_decision_scenario.py',
             ROOT / 'src/soomac_irc/test/manual_scenario_stack_probe.py']
    paths += list((ROOT / 'src/soomac_irc/soomac_irc').glob('*.py'))
    # VLM is owned by another worker and not used by these text-only calls.
    paths = [p for p in paths if 'vlm' not in p.name and p.name != 'llm_langgraph_node.py']
    return {str(p): b.sha(p) for p in paths}


def check_pins(pins):
    for path, sha in pins.items():
        if b.sha(Path(path)) != sha:
            raise RuntimeError(f'Frozen source changed: {path}')


def client_objects():
    if Path(sys.executable).absolute() != CLIENT:
        raise RuntimeError(f'Use actual client interpreter: {CLIENT}, got {sys.executable}')
    import torch
    torch.set_num_threads(2)
    from transformers import AutoProcessor
    from soomac_irc import vllm_client as vc
    from soomac_irc.response_model import make_call_response
    from soomac_irc.recommendation_model import make_call_recommendation
    vc.VLLM_BASE_URL = ENDPOINT
    processor = AutoProcessor.from_pretrained(b.BASE, local_files_only=True)
    model = vc.VllmModel(max_model_len=10240)
    calls = {'response': make_call_response(model, processor, adapter_name='response'),
             'recommendation': make_call_recommendation(model, processor, adapter_name='recommendation')}
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU client initialized CUDA')
    return vc, model, processor, calls


def cases_from(folder):
    cases = []
    for name, group in [('VALIDATION_REVIEWED.json', 'reviewed_validation'),
                        ('FUNCTIONAL_CASES.json', 'functional')]:
        for case in b.read(folder / name)['rows']:
            case = copy.deepcopy(case)
            case['group'] = group
            case['key'] = f"{case['agent']}:{group}:{case['record']['id']}"
            cases.append(case)
    if len({c['key'] for c in cases}) != 211:
        raise RuntimeError('Expected 211 unique evaluation cases')
    return cases


class CaptureOnly(BaseException):
    pass


def freeze(root, holdout):
    folder = root / 'inputs'
    reviewed.prepare(folder, holdout)
    pins = source_pins()
    for name in ('SERVER_ENVIRONMENT.json', 'CLIENT_ENVIRONMENT.json'):
        info = b.read(root / name)
        for path, sha in info.get('source_sha256', {}).items():
            path = Path(path)
            if not path.is_absolute():
                path = ROOT / path
            if b.sha(path) != sha:
                raise RuntimeError(f'Environment snapshot drift: {path}')
            pins[str(path)] = sha
    release, _, _ = b.source_identity('response')
    b.source_identity('recommendation')
    processor_hashes = {}
    for name, expected in release['processor_files'].items():
        path = b.BASE / name
        if path.is_file():
            processor_hashes[str(path)] = b.sha(path)
            if processor_hashes[str(path)] != expected:
                raise RuntimeError(f'Base processor differs from sealed processor: {name}')
    vc, model, processor, calls = client_objects()
    requests = []
    original = vc.complete_tokens
    try:
        for case in cases_from(folder):
            captured = []
            def capture(*args, **kwargs):
                captured.append(request_record(*args, **kwargs))
                raise CaptureOnly()
            vc.complete_tokens = capture
            call = calls[case['agent']]
            call.trace_events.clear()
            args = reviewed.rebuild(case['agent'], case['record'])
            try:
                call(*args)
            except CaptureOnly:
                pass
            if len(captured) != 1 or len(call.trace_events) != 1:
                raise RuntimeError('Expected one intercepted request')
            trace = call.trace_events[0]
            if trace['model_input'] != case['record']['input'] or trace.get('history_trimmed_messages'):
                raise RuntimeError('Fixed input changed or history was trimmed')
            requests.append(dict(key=case['key'], request=captured[0],
                                 request_sha256=reviewed.digest(captured[0]),
                                 messages=trace['messages']))
    finally:
        vc.complete_tokens = original
    b.write_new(folder / 'REQUESTS.json', {'rows': requests})
    check_pins(pins)
    b.write_new(folder / 'SERVING_FREEZE.json', {
        'status': 'CPU_CAPTURE_NO_MODEL_OUTPUT', 'cases': len(requests),
        'files': {p.name: b.sha(p) for p in folder.glob('*.json')},
        'source_sha256': pins, 'processor_sha256': processor_hashes,
        'base_weight_stat': {'path': str(b.BASE / 'model.safetensors'),
                            'size': (b.BASE / 'model.safetensors').stat().st_size,
                            'mtime_ns': (b.BASE / 'model.safetensors').stat().st_mtime_ns},
        'endpoint': ENDPOINT, 'client_python': sys.executable,
        'no_cuda_initialized': True,
        'scope': 'actual runtime builder, serializer, tokenizer, grammar and output budget; no HTTP sent'})
    emit('tokens_frozen', cases=len(requests), max_prompt_tokens=max(
        len(r['request']['prompt_token_ids']) for r in requests))


def verify(root):
    frozen = b.read(root / 'inputs/SERVING_FREEZE.json')
    for name, sha in frozen['files'].items():
        if b.sha(root / 'inputs' / name) != sha:
            raise RuntimeError(f'Frozen file changed: {name}')
    check_pins(frozen['source_sha256'])
    check_pins(frozen['processor_sha256'])
    check_pins(b.read(root / 'inputs/FREEZE.json')['protected_source_sha256'])
    stat = frozen['base_weight_stat']
    actual = Path(stat['path']).stat()
    if actual.st_size != stat['size'] or actual.st_mtime_ns != stat['mtime_ns']:
        raise RuntimeError('Base weights stat changed')
    return frozen


def stage(root):
    verify(root)
    import torch
    torch.set_num_threads(2)
    from safetensors.torch import load_file
    import convert_adapters_for_vllm as converter
    staging_root = b.output_dir(root / 'staged')
    layers = {i for i, kind in enumerate(b.read(b.BASE / 'config.json')['text_config']['layer_types'])
              if kind == 'full_attention'}
    receipts = []
    for candidate in b.read(root / 'inputs/CANDIDATES.json')['rows']:
        source = Path(candidate['path'])
        for name, key in [('adapter_model.safetensors', 'weights_sha256'),
                          ('adapter_config.json', 'config_sha256')]:
            if b.sha(source / name) != candidate[key]:
                raise RuntimeError(f'Checkpoint commit mismatch: {source / name}')
        destination = staging_root / f"{candidate['agent']}_{candidate['name']}"
        destination.mkdir()
        converter.OUT = destination
        path = converter.build_staging(candidate['agent'], source, layers)
        target = destination / candidate['agent']
        path.rename(target)
        before = load_file(str(source / 'adapter_model.safetensors'))
        after = load_file(str(target / 'adapter_model.safetensors'))
        if len(before) != 656 or len(after) != 672:
            raise RuntimeError('Unexpected converted tensor count')
        for key, value in before.items():
            if value.dtype != after[key].dtype or not torch.equal(value, after[key]):
                raise RuntimeError(f'Conversion changed original tensor: {key}')
        for key in after.keys() - before.keys():
            original = key.replace('.self_attn.v_proj.', '.self_attn.k_proj.')
            match = converter.K_PROJ.search(original)
            if original == key or not match or int(match.group(1)) not in layers:
                raise RuntimeError('Unexpected converted tensor')
            if after[key].dtype != before[original].dtype or not torch.equal(after[key], before[original]):
                raise RuntimeError('K/V LoRA copy differs')
        if b.sha(target / 'adapter_config.json') != candidate['config_sha256']:
            raise RuntimeError('Adapter configuration changed')
        receipts.append(dict(candidate=candidate, path=str(target),
                             weights_sha256=b.sha(target / 'adapter_model.safetensors'),
                             original_tensors_exact=True, copied_v_tensors_exact=16,
                             stored_dtype=sorted({str(v.dtype) for v in after.values()})))
        del before, after
        emit('adapter_staged', agent=candidate['agent'], name=candidate['name'])
    b.write_new(staging_root / 'IDENTITIES.json', {'rows': receipts})


def launcher_contract(adapter_root):
    # Intercept the launcher's final exec; execute no server during this capture.
    env = {k: v for k, v in os.environ.items() if not k.startswith('VLLM_')}
    env.pop('CUDA_VISIBLE_DEVICES', None)
    env.update(ADAPTER_ROOT=str(adapter_root), VLLM_PORT='18001',
               MODEL_PATH=str(b.BASE), PYTHONDONTWRITEBYTECODE='1')
    script = '''exec() { /usr/bin/python3 -c 'import json,os,sys; print(json.dumps({"argv":sys.argv[1:],"env":dict(os.environ)}))' "$@"; }
source "$1"
'''
    captured = json.loads(subprocess.check_output(['bash', '-c', script, 'capture', str(LAUNCHER)],
                                                 env=env, text=True))
    actual = [arg for arg in captured['argv'] if not arg.startswith('decision=')]
    if len(actual) != len(captured['argv']) - 1:
        raise RuntimeError('Expected exactly one omitted Decision module')
    captured['original_argv'] = captured['argv']
    captured['argv'] = actual
    return captured


def assert_port_free():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 18001))


def individual(root, output, candidates, vc, calls):
    import functional_eval
    frozen = {r['key']: r for r in b.read(root / 'inputs/REQUESTS.json')['rows']}
    cases = cases_from(root / 'inputs')
    original = vc.complete_tokens
    summaries = []
    try:
        for candidate in candidates:
            name = f"{candidate['agent']}_{candidate['name']}"
            entries = []
            with (output / f'{name}.jsonl').open('x') as stream:
                selected = [c for c in cases if c['agent'] == candidate['agent']]
                for index, case in enumerate(selected, 1):
                    expected = frozen[case['key']]['request']
                    exchange = []
                    def checked(*args, **kwargs):
                        request = request_record(*args, **kwargs)
                        if request != expected:
                            raise RuntimeError(f"Request token/settings drift: {case['key']}")
                        response = original(*args, **kwargs)
                        exchange.append({'request_sha256': reviewed.digest(request), 'response': response})
                        return response
                    vc.complete_tokens = checked
                    call = calls[case['agent']]
                    call.trace_events.clear()
                    started = time.monotonic()
                    # Request/transport/parse failures stop the run; no retries or fallback scores.
                    value = call(*reviewed.rebuild(case['agent'], case['record']))
                    prediction = {'reply': value} if case['agent'] == 'response' else value
                    traces = copy.deepcopy(call.trace_events)
                    if len(exchange) != 1 or any(t.get('history_trimmed_messages') or
                        t['model_input'] != case['record']['input'] for t in traces):
                        raise RuntimeError('Actual request/input invariant failed')
                    grade = functional_eval.grade(case, prediction)
                    row = dict(key=case['key'], id=case['record']['id'], group=case['group'],
                               origin=case.get('evaluation_origin', 'sealed_validation'),
                               prediction=prediction, grade=grade, traces=traces, exchange=exchange,
                               seconds=time.monotonic() - started)
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                    stream.flush()
                    entries.append(row)
                    if index % 10 == 0 or index == len(selected):
                        emit('generation_progress', candidate=name, completed=index, total=len(selected),
                             automatic_pass=sum(r['grade']['automated_pass'] for r in entries))
            summary = dict(candidate=candidate, rows=len(entries), semantic_review='PENDING',
                           by_group={group: {'rows': sum(r['group'] == group for r in entries),
                                             'automatic_pass': sum(r['group'] == group and
                                                r['grade']['automated_pass'] for r in entries)}
                                     for group in ('reviewed_validation', 'functional')})
            b.write_new(output / f'{name}_SUMMARY.json', summary)
            summaries.append(summary)
    finally:
        vc.complete_tokens = original
    return summaries


def scenario(output, vc, model, processor):
    import eval_gold_decision_scenario as gold
    import manual_scenario_stack_probe as probe
    from soomac_irc import decision_model
    rows = gold.oracle_rows()
    b.write_new(output / 'GOLD_DECISION.json', {'rows': rows,
        'assumption': 'correct Decision fields; no Decision model or actual ROS/VLM'})
    factory, original = decision_model.make_call_decision, vc.complete_tokens
    try:
        with (output / 'SCENARIO_REQUESTS.jsonl').open('x') as requests, (output / 'SCENARIO_TURNS.jsonl').open('x') as turns:
            def checked(*args, **kwargs):
                request = request_record(*args, **kwargs)
                response = original(*args, **kwargs)
                requests.write(json.dumps(dict(request=request, response=response), ensure_ascii=False) + '\n')
                requests.flush()
                return response
            vc.complete_tokens = checked
            episodes = []
            for compound in (True, False):
                decision_model.make_call_decision = gold.oracle_factory(compound, rows)
                episodes.append(probe.run_episode(compound, model, processor, turns, measure_all=True))
        b.write_new(output / 'SCENARIO_RESULT.json', {'episodes': episodes,
            'semantic_review': 'PENDING', 'actual_ros': False, 'decision_model_called': False,
            'physical_simulation_only': True})
    finally:
        decision_model.make_call_decision = factory
        vc.complete_tokens = original


def run_pair(root, label):
    verify(root)
    identities = b.read(root / 'staged/IDENTITIES.json')['rows']
    wanted = {'response': label, 'recommendation': 'epoch6' if label == 'best150' else label}
    selected = [next(r for r in identities if r['candidate']['agent'] == agent and
                     r['candidate']['name'] == name) for agent, name in wanted.items()]
    for row in selected:
        if b.sha(Path(row['path']) / 'adapter_model.safetensors') != row['weights_sha256']:
            raise RuntimeError('Staged weights changed')
        candidate = row['candidate']
        if b.sha(Path(row['path']) / 'adapter_config.json') != candidate['config_sha256']:
            raise RuntimeError('Staged config changed')
        if b.sha(Path(candidate['path']) / 'adapter_model.safetensors') != candidate['weights_sha256']:
            raise RuntimeError('Original checkpoint changed')
    assert_port_free()
    lock = b.acquire_gpu_lock()
    process = None
    unit = f'soomac-qp-vllm-r4-{label}-{os.getpid()}'
    output = None
    try:
        output = b.output_dir(root / 'runs' / label)
        adapters = output / 'adapters'
        adapters.mkdir()
        for row in selected:
            (adapters / row['candidate']['agent']).symlink_to(row['path'], target_is_directory=True)
        captured = launcher_contract(adapters)
        # Store only execution-related environment, never unrelated credentials.
        environment = {k: v for k, v in captured['env'].items() if k.startswith(('VLLM_', 'HF_', 'TRANSFORMERS_', 'CUDA_'))
                       or k in ('DO_NOT_TRACK', 'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE')}
        b.write_new(output / 'LAUNCH.json', dict(argv=captured['argv'], original_argv=captured['original_argv'],
                   environment=environment, owned_unit=unit, candidates=selected,
                   intentional_differences=['port18001', 'isolated adapters', 'Decision module omitted; max_loras unchanged'],
                   source_sha256=source_pins(), client_python=sys.executable))
        with (output / 'SERVER.log').open('x') as log:
            command = ['systemd-run', '--user', '--scope', f'--unit={unit}',
                       '-p', 'MemoryMax=40G', '-p', 'MemorySwapMax=0', '--'] + captured['argv']
            process = subprocess.Popen(command, cwd=ROOT, env=captured['env'], stdout=log, stderr=subprocess.STDOUT)
            emit('server_starting', pair=label, pid=process.pid, unit=unit, log=str(output / 'SERVER.log'))
            deadline = time.monotonic() + 900
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f'Server exited: {process.returncode}; inspect SERVER.log')
                try:
                    with urllib.request.urlopen(ENDPOINT + '/health', timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError('Own evaluation server did not become healthy within 900s')
                time.sleep(2)
            vc, model, processor, calls = client_objects()
            models = vc.served_models()
            roots = {m['id']: m.get('root') for m in models}
            if set(roots) != {'gemma4-base', 'recommendation', 'response'}:
                raise RuntimeError('Unexpected served models')
            if next(m for m in models if m['id'] == 'gemma4-base')['max_model_len'] != 10240:
                raise RuntimeError('Server model length differs')
            for row in selected:
                if Path(roots[row['candidate']['agent']]).resolve() != Path(row['path']).resolve():
                    raise RuntimeError('Server adapter path differs')
            vc.check_served_adapters({r['candidate']['agent']: r['candidate']['path'] for r in selected}, roots)
            b.write_new(output / 'SERVED_MODELS.json', models)
            emit('server_ready', pair=label)
            candidates = [r['candidate'] for r in selected if not (
                label == 'best150' and r['candidate']['agent'] == 'recommendation')]
            summaries = individual(root, output, candidates, vc, calls)
            # Every matched epoch pair receives the same conditional 6+11-turn scenario.
            scenario(output, vc, model, processor)
            verify(root)
            b.write_new(output / 'RESULT.json', {'status': 'GENERATION_COMPLETE_SEMANTIC_REVIEW_PENDING',
                                               'summaries': summaries, 'decision_model_called': False})
            emit('pair_complete', pair=label)
    except BaseException as error:
        if output is not None:
            b.write_new(output / 'FAILURE.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        if process is not None:
            subprocess.run(['systemctl', '--user', 'stop', unit + '.scope'], check=False, timeout=60)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                raise RuntimeError(f'Owned server scope still running: {unit}')
        os.close(lock)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze', 'stage', 'run'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--holdout', type=Path)
    parser.add_argument('--pair', choices=[f'epoch{i}' for i in range(1, 7)] + ['best150'])
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    if args.action == 'freeze':
        freeze(args.root.resolve(), args.holdout)
    elif args.action == 'stage':
        stage(args.root.resolve())
    else:
        if args.pair is None:
            parser.error('--pair is required for run')
        run_pair(args.root.resolve(), args.pair)
