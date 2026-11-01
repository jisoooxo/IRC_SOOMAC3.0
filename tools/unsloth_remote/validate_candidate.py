#!/usr/bin/env python3
"""Run inside pinned candidate: one sealed Decision row, isolated GC comparisons.

No production invocation, checkpoint migration or tolerance relaxation. Large tensor
evidence stays in the local output volume, not in review packets or final TAR.
"""
import argparse
import copy
import functools
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import re
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('hf', 'native', 'smart'), required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--initial', type=Path)
    parser.add_argument('--timing', action='store_true')
    parser.add_argument('--equivalence', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, '/app/tools/training_acceleration')
    import benchmark
    benchmark.BASE, benchmark.RELEASE = Path('/model'), Path('/app')
    benchmark.PROCESS_BUDGET, benchmark.RESERVE = 32_000_000_000, 1024**3
    output = benchmark.output_dir(args.output)
    receipt = benchmark.read(args.prepared / 'QUALITY_PREPARE.json')
    if benchmark.sha(args.prepared / 'REPRESENTATIVES.json') != receipt['representatives_sha256']:
        raise ValueError('encoded input changed')
    row = next(r for r in benchmark.read(args.prepared / 'REPRESENTATIVES.json')['rows']
               if r['agent'] == 'decision' and r['category'] == 'long')
    if len(row['input_ids']) != 8608:
        raise ValueError('expected sealed longest Decision input8608')
    if args.timing:
        proof = benchmark.read(args.equivalence) if args.equivalence else {}
        if proof.get('selected') != args.mode or proof.get('status') != 'GC_EQUIVALENCE_PASS':
            raise ValueError('timing requires passing exact selected GC proof')
    fd, budget = None, None
    result = {'status': 'FAILED', 'gc': args.mode, 'production_changed': False,
              'source_manifest_sha256': receipt['source_manifest_sha256'],
              'prepared_sha256': benchmark.sha(args.prepared / 'REPRESENTATIVES.json'),
              'optimizer_updates': 0}
    try:
        fd = benchmark.acquire_gpu_lock()
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', UNSLOTH_RETURN_LOGITS='1')
        if args.mode != 'hf':
            import unsloth  # noqa: F401
        import torch
        from transformers import AutoProcessor, DataCollatorForSeq2Seq, set_seed
        from safetensors.torch import save_file, load_file
        sys.path.insert(0, '/app/common')
        from budget import Budget
        from loss import scored_suffix
        set_seed(42)
        budget = Budget(benchmark.PROCESS_BUDGET, benchmark.RESERVE)
        calls = []
        if args.mode == 'hf':
            model, optimizer, parameters, names = benchmark.model_and_optimizer(
                'baseline', 'decision', torch, model_dir=benchmark.BASE, release_dir=benchmark.RELEASE)
        else:
            from unsloth import FastModel
            setting = True if args.mode == 'native' else 'unsloth'
            descriptors = [inspect.getattr_static(FastModel, n) for n in ('from_pretrained', 'get_peft_model')]
            functions = [FastModel.from_pretrained, FastModel.get_peft_model]
            def wrap(fn, name):
                @functools.wraps(fn)
                def wrapped(*positional, **named):
                    named['use_gradient_checkpointing'] = setting
                    calls.append({'loader': name, 'use_gradient_checkpointing': setting})
                    return fn(*positional, **named)
                return staticmethod(wrapped)
            try:
                FastModel.from_pretrained = wrap(functions[0], 'from_pretrained')
                FastModel.get_peft_model = wrap(functions[1], 'get_peft_model')
                model, optimizer, parameters, names = benchmark.model_and_optimizer(
                    'unsloth', 'decision', torch, model_dir=benchmark.BASE, release_dir=benchmark.RELEASE)
            finally:
                FastModel.from_pretrained, FastModel.get_peft_model = descriptors
        initial = {n: p.detach().cpu().contiguous() for n, p in zip(names, parameters)}
        if args.initial:
            saved = load_file(str(args.initial))
            if set(saved) != set(initial):
                raise ValueError('LoRA parameter names changed')
            with torch.no_grad():
                for name, parameter in zip(names, parameters):
                    parameter.copy_(saved[name].to(parameter.device, dtype=parameter.dtype))
            initial = saved
        elif args.mode != 'hf':
            raise ValueError('Unsloth requires exact HF initial LoRA')
        save_file(initial, str(output / 'INITIAL_LORA.safetensors'))
        result['initial_lora_sha256'] = benchmark.sha(output / 'INITIAL_LORA.safetensors')
        result['trainable_names'] = names
        result['loader_gc_calls'] = calls
        result['packages'] = {k: importlib.metadata.version(k) for k in
                             ('torch', 'transformers', 'peft', 'unsloth', 'unsloth-zoo', 'bitsandbytes')}
        result['checkpoint_functions'] = {n: str(getattr(m, '_gradient_checkpointing_func'))[:250]
            for n, m in model.named_modules() if hasattr(m, '_gradient_checkpointing_func')}
        processor = AutoProcessor.from_pretrained('/app/assets/model-processor', local_files_only=True)
        batch = DataCollatorForSeq2Seq(tokenizer=processor.tokenizer, padding=True,
            label_pad_token_id=-100, pad_to_multiple_of=8, return_tensors='pt')([
                {k: copy.deepcopy(row[k]) for k in ('input_ids', 'labels', 'attention_mask')}])
        batch = {k: v.to('cuda') for k, v in batch.items()}
        selected = scored_suffix(batch)
        def thash(t):
            return hashlib.sha256(t.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
        result['input_hashes'] = {k: thash(v) for k, v in selected.items()}
        dropouts = [(m, m.p) for n, m in model.named_modules()
                    if 'lora_dropout' in n and isinstance(m, torch.nn.Dropout)]
        if not dropouts or any(abs(p-.05) > 1e-9 for m, p in dropouts):
            raise ValueError('original LoRA dropout must be.05')
        buffers = {n: b.detach().cpu().clone() for n, b in model.named_buffers()}
        result['repeats'] = []
        if not args.timing:
            for repeat in range(2):
                with torch.no_grad():
                    for name, parameter in zip(names, parameters):
                        parameter.copy_(initial[name].to(parameter.device, dtype=parameter.dtype))
                    for name, buffer in model.named_buffers():
                        buffer.copy_(buffers[name].to(buffer.device))
                model.train()
                for module, probability in dropouts:
                    module.p = 0.0
                model.zero_grad(set_to_none=True)
                set_seed(42)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    pred = model(**selected, use_cache=False)
                logits = pred.logits.detach().float().cpu()
                loss = float(pred.loss.detach())
                ce = float(torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                    selected['shift_labels'].detach().cpu().reshape(-1)))
                if abs(loss-ce) > .001:
                    raise ValueError('native selected-logit CE drift')
                (pred.loss / 16).backward()
                gradient = torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).detach().float().cpu().flatten()
                                      for p in parameters])
                if not torch.isfinite(gradient).all() or not torch.count_nonzero(gradient):
                    raise ValueError('nonfinite/zero gradient')
                torch.save(logits, output / f'logits{repeat}.pt')
                torch.save(gradient, output / f'gradient{repeat}.pt')
                result['repeats'].append({'loss': loss, 'manual_ce': ce,
                    'gradient_norm_fp64': float(gradient.double().norm()), 'diagnostic_dropout': 0.0})
                print(json.dumps({'phase': 'gc_probe_repeat', 'gc': args.mode, 'repeat': repeat,
                                  **result['repeats'][-1]}), flush=True)
                del gradient, logits, pred
                model.zero_grad(set_to_none=True)
                budget.check(force=True)
            result['status'] = 'GC_PROBE_COMPLETE_NOT_EQUIVALENCE_PASS'
        else:
            for module, probability in dropouts:
                module.p = probability
            model.train()
            batches = [batch]*16
            benchmark.one_update(model, optimizer, parameters, batches, torch, budget, measure=False)
            result['optimizer_updates'] = 1
            timed = []
            for i in range(2):
                item = benchmark.one_update(model, optimizer, parameters, batches, torch, budget, measure=True)
                timed.append(item)
                result['optimizer_updates'] += 1
                print(json.dumps({'phase': 'timed_update', 'gc': args.mode, **item}), flush=True)
            mean = sum(x['seconds'] for x in timed)/len(timed)
            model.save_pretrained(output / 'TIMED_ADAPTER')
            result.update(status='GC_TIMING_AND_ADAPTER_EXPORT_PASS', warmup_updates=1, timed=timed,
                          mean_update_seconds=mean, samples_per_second=16/mean,
                          nonpad_tokens_per_second=len(row['input_ids'])*16/mean, production_dropout=.05)
    finally:
        if budget:
            result.update(budget.summary(), peak_allocated_bytes=budget.torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=budget.torch.cuda.max_memory_reserved())
            budget.close()
        result['cpu_peak_rss_bytes'] = benchmark.process_peak_rss_bytes()
        if Path('/sys/fs/cgroup/memory.peak').exists():
            result['cgroup_memory_peak_bytes'] = int(Path('/sys/fs/cgroup/memory.peak').read_text())
        benchmark.write_new(output / 'CANDIDATE_RESULT.json', result)
        if fd is not None:
            os.close(fd)


if __name__ == '__main__':
    main()
