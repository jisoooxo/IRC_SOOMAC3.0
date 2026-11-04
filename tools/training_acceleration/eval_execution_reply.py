"""Evaluate model execution speech on the existing 6+11-turn gold-D scenarios.

Uses the reviewed P best150/Q epoch6 and an owned vLLM scope on port 18001.
No individual-evaluation repeat, real Decision, ROS, training or deployment.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

import benchmark as b
import eval_vllm_epochs as serving


def run(root):
    import eval_gold_decision_scenario as gold
    import manual_scenario_stack_probe as probe
    from soomac_irc import decision_model

    plan = b.read(root / 'PLAN.json')
    previous = root.parent / 'qp_vllm_validation_20261103_r4'
    freeze = b.read(previous / 'inputs/SERVING_FREEZE.json')
    changed = {str(serving.ROOT / p) for p in (
        'src/soomac_irc/soomac_irc/llm_langgraph.py',
        'src/soomac_irc/test/manual_scenario_stack_probe.py')}
    for path, sha in freeze['source_sha256'].items():
        if path not in changed and b.sha(Path(path)) != sha:
            raise RuntimeError(f'Unexpected source drift: {path}')
    serving.check_pins(freeze['processor_sha256'])
    stat = freeze['base_weight_stat']
    actual = Path(stat['path']).stat()
    if (actual.st_size, actual.st_mtime_ns) != (stat['size'], stat['mtime_ns']):
        raise RuntimeError('Base weight stat changed')
    selected = plan['selected']
    for row in selected:
        c = row['candidate']
        for path, expected in (
            (Path(row['path']) / 'adapter_model.safetensors', row['weights_sha256']),
            (Path(c['path']) / 'adapter_model.safetensors', c['weights_sha256']),
            (Path(row['path']) / 'adapter_config.json', c['config_sha256']),
            (Path(c['path']) / 'adapter_config.json', c['config_sha256']),
        ):
            if b.sha(path) != expected:
                raise RuntimeError(f'Adapter identity changed: {path}')
    serving.assert_port_free()
    lock = b.acquire_gpu_lock()
    process = None
    output = b.output_dir(root / 'model_execution_run')
    unit = f'soomac-execution-reply-{os.getpid()}'
    pins = serving.source_pins()
    pins[str(Path(__file__).resolve())] = b.sha(Path(__file__))
    try:
        adapters = output / 'adapters'
        adapters.mkdir()
        for row in selected:
            (adapters / row['candidate']['agent']).symlink_to(row['path'], target_is_directory=True)
        captured = serving.launcher_contract(adapters)
        # Explicit allowlist: the process environment may contain credentials.
        safe_env = {key: captured['env'][key] for key in (
            'VLLM_BATCH_INVARIANT', 'VLLM_PORT', 'VLLM_WORKER_MULTIPROC_METHOD',
            'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'PYTHONDONTWRITEBYTECODE')
            if key in captured['env']}
        b.write_new(output / 'LAUNCH.json', dict(argv=captured['argv'],
            environment=safe_env, owned_unit=unit, source_sha256=pins,
            execution_reply_mode='model', candidates=selected,
            decision_model_called=False, actual_ros=False))
        with (output / 'SERVER.log').open('x') as log:
            process = subprocess.Popen([
                'systemd-run', '--user', '--scope', f'--unit={unit}',
                '-p', 'MemoryMax=40G', '-p', 'MemorySwapMax=0', '--',
                *captured['argv']], cwd=serving.ROOT, env=captured['env'],
                stdout=log, stderr=subprocess.STDOUT)
            serving.emit('server_starting', pid=process.pid, unit=unit)
            deadline = time.monotonic() + 300
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f'Server exited: {process.returncode}')
                try:
                    with urllib.request.urlopen(serving.ENDPOINT + '/health', timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError('Evaluation server startup exceeded 300 seconds')
                time.sleep(2)
            vc, model, processor, _ = serving.client_objects()
            models = vc.served_models()
            roots = {m['id']: m.get('root') for m in models}
            if set(roots) != {'gemma4-base', 'recommendation', 'response'}:
                raise RuntimeError('Unexpected served models')
            vc.check_served_adapters({r['candidate']['agent']: r['candidate']['path'] for r in selected}, roots)
            for row in selected:
                if Path(roots[row['candidate']['agent']]).resolve() != Path(row['path']).resolve():
                    raise RuntimeError('Served adapter path differs')
            b.write_new(output / 'SERVED_MODELS.json', models)
            rows = gold.oracle_rows()
            b.write_new(output / 'GOLD_DECISION.json', {'rows': rows})
            factory, original = decision_model.make_call_decision, vc.complete_tokens
            requests_count = {'response': 0, 'recommendation': 0}
            try:
                with (output / 'SCENARIO_REQUESTS.jsonl').open('x') as requests, \
                     (output / 'SCENARIO_TURNS.jsonl').open('x') as turns:
                    def checked(*args, **kwargs):
                        request = serving.request_record(*args, **kwargs)
                        answer = original(*args, **kwargs)
                        requests_count[request['model']] += 1
                        requests.write(json.dumps({'request': request, 'response': answer}, ensure_ascii=False) + '\n')
                        requests.flush()
                        return answer
                    vc.complete_tokens = checked
                    episodes = []
                    for compound in (True, False):
                        decision_model.make_call_decision = gold.oracle_factory(compound, rows)
                        episodes.append(probe.run_episode(compound, model, processor, turns,
                            measure_all=True, execution_reply_mode='model'))
            finally:
                decision_model.make_call_decision = factory
                vc.complete_tokens = original
            if requests_count != {'response': 17, 'recommendation': 2}:
                raise RuntimeError(f'Expected P17/Q2 requests, got {requests_count}')
            serving.check_pins(pins)
            b.write_new(output / 'RESULT.json', dict(episodes=episodes,
                requests=requests_count, semantic_review='PENDING',
                execution_reply_mode='model', decision_model_called=False, actual_ros=False))
            serving.emit('generation_complete', requests=requests_count)
    except BaseException as error:
        b.write_new(output / 'FAILURE.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        try:
            if process is not None:
                stopped = subprocess.run(['systemctl', '--user', 'stop', unit + '.scope'],
                    check=False, timeout=20, capture_output=True, text=True)
                process.wait(timeout=20)
                b.write_new(output / 'CLEANUP.json', dict(unit=unit,
                    stop_returncode=stopped.returncode, process_returncode=process.returncode))
        finally:
            os.close(lock)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    run(args.root.resolve())
