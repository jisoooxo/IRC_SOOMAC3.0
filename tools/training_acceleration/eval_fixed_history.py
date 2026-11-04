"""Current fixed-speech history regression: correct D, real Q/P, 44 turns.

Offline physical fixtures only. Does not assess Decision, ROS, VLM or TTS.
"""
from __future__ import annotations
import argparse
import contextlib
import fcntl
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
from unittest.mock import patch
import benchmark as b
import eval_vllm_epochs as serving
sys.path.insert(0, str(serving.ROOT / 'src/soomac_irc/test'))
import manual_scenario_stack_probe as probe
import eval_gold_decision_scenario as gold
import eval_multiturn_reference as reference


def evaluate(model, processor, calls, stream):
    from soomac_irc import decision_model
    rows = gold.oracle_rows()
    episodes = []
    for compound in (True, False):
        with patch.object(decision_model, 'make_call_decision', gold.oracle_factory(compound, rows)):
            episodes.append(probe.run_episode(compound, model, processor, stream,
                                             measure_all=False, execution_reply_mode='fixed'))
        if not episodes[-1]['passed']:
            raise RuntimeError('Required scenario state failed; preserve observed history')
    for spec in reference.scenarios():
        episodes.append(reference.run_episode(spec, calls, stream, execution_reply_mode='fixed'))
        if not episodes[-1]['passed']:
            raise RuntimeError('Reference scenario state failed; preserve observed history')
    return episodes


def cpu_check():
    def response(*args):
        return 'CPU fixture only'
    def recommendation(*args):
        return {'proposal': {'sauce': '크림', 'noodle_type': None, 'noodle_portion': None, 'toppings': {}},
                'reason_tags': ['느끼한 소스 요청']}
    response.trace_events, recommendation.trace_events = [], []
    with patch('soomac_irc.response_model.make_call_response', lambda *a, **k: response), \
         patch('soomac_irc.recommendation_model.make_call_recommendation', lambda *a, **k: recommendation), \
         contextlib.redirect_stdout(io.StringIO()):
        result = evaluate(None, None, {'response': response, 'recommendation': recommendation}, io.StringIO())
    import torch
    assert not torch.cuda.is_initialized()
    # Mock calls do not emit model traces; their source counters are not evidence.
    for episode in result:
        episode.pop('reply_sources', None)
        episode.pop('response_auto_flagged_turns', None)
        episode['cpu_fixture_only'] = True
    return result


def acquire_lock():
    # Preserve the same GPU lock; only the observed desktop remote-control app is allowed.
    fd = os.open(b.LOCK, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                                       '--format=csv,noheader,nounits'], text=True, timeout=10)
        for line in raw.splitlines():
            pid, name, memory = [part.strip() for part in line.split(',')]
            if name != '/usr/share/rustdesk/rustdesk' or int(memory) > 1024:
                raise RuntimeError('GPU has a non-desktop or unexpectedly large existing process: ' + line)
        used = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                                        '--format=csv,noheader,nounits'], text=True, timeout=10)
        if any(int(x) > 3072 for x in used.splitlines()):
            raise RuntimeError('Unexpected GPU memory baseline')
        return fd, {'compute_rows': raw.splitlines(), 'gpu_used_mib': used.splitlines()}
    except BaseException:
        os.close(fd)
        raise


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
    lock, gpu_baseline = acquire_lock()
    process, output = None, None
    unit = f'soomac-fixed-history-{os.getpid()}'
    try:
        output = b.output_dir(root)
        pins = serving.source_pins()
        for file in [Path(__file__), Path(reference.__file__), serving.ROOT / 'src/soomac_irc/soomac_irc/llm_langgraph_node.py',
                     serving.ROOT / 'config/llm_active_stack.json']:
            pins[str(file)] = b.sha(file)
        b.write_new(output / 'PLAN.json', dict(scenarios={'required': [probe.scenario(True), probe.scenario(False)], 'reference': reference.scenarios()}, selected=prepared, source_sha256=pins, gpu_baseline=gpu_baseline,
            cpu_fixture_check=preflight, real_decision=False, real_recommendation=True,
            real_response=True, actual_ros=False, semantic_review='MANUAL_REQUIRED'))
        captured = serving.launcher_contract(serving.ROOT / 'models/vllm_adapters')
        if captured['env'].get('VLLM_BATCH_INVARIANT') != '1':
            raise RuntimeError('Batch invariance disabled')
        b.write_new(output / 'LAUNCH.json', dict(argv=captured['argv'], unit=unit,
            environment={k: captured['env'][k] for k in ['VLLM_BATCH_INVARIANT', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE']},
            differences=['port18001', 'D omitted', 'fixed speech with current wording; real Q/P']))
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
            original, count = vc.complete_tokens, {'response': 0, 'recommendation': 0}
            try:
                with (output / 'REQUESTS.jsonl').open('x') as requests, (output / 'TURNS.jsonl').open('x') as turns:
                    def checked(*args, **kwargs):
                        request = serving.request_record(*args, **kwargs)
                        answer = original(*args, **kwargs)
                        requests.write(json.dumps(dict(request=request, response=answer), ensure_ascii=False) + '\n')
                        requests.flush(); count[request['model']] += 1
                        return answer
                    vc.complete_tokens = checked
                    episodes = evaluate(model, processor, calls, turns)
            finally:
                vc.complete_tokens = original
            if count != {'response': 26, 'recommendation': 3}:
                raise RuntimeError(f'Expected P26/Q3, got {count}')
            serving.check_pins(pins)
            b.write_new(output / 'RESULT.json', dict(episodes=episodes, requests=count,
                decision_requests=0, runtime_fixed=18, actual_ros=False,
                semantic_review='PENDING', sources_unchanged=True))
            serving.emit('complete', requests=count)
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
