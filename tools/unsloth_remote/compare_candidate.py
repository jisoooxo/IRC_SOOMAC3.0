#!/usr/bin/env python3
"""FP64 full/per-layer GC comparison against observed A/A repeat noise."""
import argparse
import math
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / 'training_acceleration'))
import benchmark
from quality_gate import _chunked_diff


def require_runtime(report, mode):
    flags = report.get('gradient_checkpointing_flags', {})
    if not flags or not any(flags.values()) or report.get('dropout_restored') is not True:
        raise ValueError(f'GC/dropout runtime evidence missing: {mode}')
    for repeat in report['repeats']:
        if repeat.get('dropout_all_zero') is not True or repeat.get('dropout_modules') != 328:
            raise ValueError(f'diagnostic dropout not zero: {mode}')
        runtime = repeat.get('runtime', {})
        calls = runtime.get('checkpoint_calls', {})
        expected = 'UnslothCheckpointFunction' if mode == 'smart' else 'CheckpointFunction'
        # autograd's worker-thread backward is not observable by sys.setprofile
        # on the main thread. Require real CheckpointFunction.forward calls plus
        # the completed backward's independently stored nonzero gradient; do not
        # invent a missing Python backward callback or relax numeric comparisons.
        if (not any(name == expected + '.forward' and count > 0
                    for name, count in calls.items())
                or not repeat.get('gradient_norm_fp64', 0) > 0):
            raise ValueError(f'actual checkpoint forward/completed backward unproven: {mode}')
        if mode != 'hf':
            drop = runtime.get('fused_dropout', {})
            if (not drop or any(key not in ('0.0', 'forward_p=0.0') for key in drop)
                    or not all(isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in drop.values())):
                raise ValueError(f'fused dropout not proven zero: {mode}: {drop}')


def compare(root):
    import torch
    from safetensors import safe_open
    reports = {name: benchmark.read(root / name / 'CANDIDATE_RESULT.json') for name in ('hf', 'native', 'smart')}
    baseline = reports['hf']
    runtime_issues = {}
    for mode, report in reports.items():
        if report.get('status') != 'GC_PROBE_COMPLETE_NOT_EQUIVALENCE_PASS' or len(report.get('repeats', [])) != 2:
            raise ValueError(f'incomplete probe: {mode}')
        try:
            require_runtime(report, mode)
        except ValueError as error:
            # Preserve numeric failure evidence even if Python profiling did
            # not observe compiled fused dropout. Missing observation NEVER
            # becomes a pass; final selection remains fail-closed below.
            runtime_issues[mode] = str(error)
        for key in ('initial_lora_sha256', 'trainable_names', 'prepared_sha256', 'input_hashes', 'packages',
                    'diagnostic_script_sha256'):
            if report[key] != baseline[key]:
                raise ValueError(f'{mode} differs in controlled identity: {key}')
    spans = {i: [] for i in range(48)}
    offset = 0
    with safe_open(root / 'hf/INITIAL_LORA.safetensors', framework='pt', device='cpu') as initial:
        for name in baseline['trainable_names']:
            length = math.prod(initial.get_slice(name).get_shape())
            match = re.search(r'language_model\.layers\.(\d+)\.', name)
            if not match:
                raise ValueError('unexpected trainable scope')
            spans[int(match.group(1))].append((offset, offset+length))
            offset += length
    if any(not intervals for intervals in spans.values()):
        raise ValueError('decoder layer missing from LoRA comparison')
    # Reduce one layer at a time instead of retaining 6 full FP64 gradient copies.
    gradients = {mode: [torch.load(root / mode / f'gradient{i}.pt', map_location='cpu', weights_only=True)
                         for i in range(2)] for mode in reports}
    if any(g.numel() != offset for pair in gradients.values() for g in pair):
        raise ValueError('gradient shape/parameter order mismatch')
    result = {'status': 'GC_EQUIVALENCE_BLOCKED', 'selected': None,
        'criterion': 'cross-mode loss/logits/gradient within maximum observed same-path two-repeat noise; no raised tolerance',
        'note': 'two repeats bound this observation only, not a statistical population guarantee',
        'same_path': {}, 'candidates': {}, 'production_authorized': False,
        'runtime_control_issues': runtime_issues}
    result['evidence_sha256'] = {str(path.relative_to(root)): benchmark.sha(path)
        for mode in reports for path in [root / mode / 'CANDIDATE_RESULT.json',
            root / mode / 'INITIAL_LORA.safetensors',
            *(root / mode / f'{kind}{i}.pt' for kind in ('gradient', 'logits') for i in range(2))]}
    result['identity'] = {key: baseline[key] for key in
        ('initial_lora_sha256', 'prepared_sha256', 'input_hashes', 'packages', 'diagnostic_script_sha256')}
    for mode in reports:
        result['same_path'][mode] = {
            'gradient': _chunked_diff(torch, *gradients[mode]),
            'loss_delta': abs(reports[mode]['repeats'][0]['loss']-reports[mode]['repeats'][1]['loss']),
            'logits': _chunked_diff(torch, *[torch.load(root / mode / f'logits{i}.pt', map_location='cpu', weights_only=True)
                                            for i in range(2)])}
    for mode in ('native', 'smart'):
        cross = []
        for i in range(2):
            for j in range(2):
                cross.append(_chunked_diff(torch, gradients['hf'][i], gradients[mode][j]))
        noise = max(result['same_path'][n]['gradient']['relative_l2_error'] for n in ('hf', mode))
        observed_cosine = min(result['same_path'][n]['gradient']['cosine'] for n in ('hf', mode))
        loss_noise = max(result['same_path'][n]['loss_delta'] for n in ('hf', mode))
        loss_difference = max(abs(x['loss']-y['loss']) for x in reports['hf']['repeats'] for y in reports[mode]['repeats'])
        logits_a = torch.load(root / 'hf/logits0.pt', map_location='cpu', weights_only=True)
        logits_b = torch.load(root / mode / 'logits0.pt', map_location='cpu', weights_only=True)
        logits_diff = _chunked_diff(torch, logits_a, logits_b)
        del logits_a, logits_b
        logits_noise = max(result['same_path'][n]['logits']['relative_l2_error'] for n in ('hf', mode))
        layers = {}
        for layer, intervals in spans.items():
            chunks = {n: [torch.cat([g[start:end] for start, end in intervals]) for g in gradients[n]]
                      for n in ('hf', mode)}
            aa = {n: _chunked_diff(torch, *chunks[n]) for n in chunks}
            ab = [_chunked_diff(torch, x, y) for x in chunks['hf'] for y in chunks[mode]]
            layer_noise = max(v['relative_l2_error'] for v in aa.values())
            layer_cosine = min(v['cosine'] for v in aa.values())
            layers[str(layer)] = {'same_path': aa, 'cross': ab,
                'within_observed_noise': all(v['finite'] and v['relative_l2_error'] <= layer_noise
                                             and v['cosine'] >= layer_cosine for v in ab)}
        global_pass = all(x['finite'] and x['relative_l2_error'] <= noise
                          and x['cosine'] >= observed_cosine for x in cross)
        numeric_pass = (global_pass and all(v['within_observed_noise'] for v in layers.values())
                  and loss_difference <= loss_noise and logits_diff['relative_l2_error'] <= logits_noise)
        passed = numeric_pass and not any(n in runtime_issues for n in ('hf', mode))
        result['candidates'][mode] = {'passed': passed, 'numeric_pass': numeric_pass, 'gradient_cross': cross,
            'gradient_AA_relativeL2_noise': noise, 'gradient_AA_min_cosine': observed_cosine,
            'loss_cross_max_delta': loss_difference, 'loss_AA_noise': loss_noise,
            'logits_cross': logits_diff, 'logits_AA_relativeL2_noise': logits_noise,
            'layers': layers}
    chosen = next((m for m in ('smart', 'native') if result['candidates'][m]['passed']), None)
    if chosen:
        result.update(status='GC_EQUIVALENCE_PASS', selected=chosen)
    benchmark.write_new(root / 'GC_EQUIVALENCE.json', result)
    summary = {m: {k: v for k, v in item.items() if k not in ('layers', 'gradient_cross')}
               for m, item in result['candidates'].items()}
    print({'status': result['status'], 'selected': chosen, 'candidates': summary})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    compare(parser.parse_args().root)
