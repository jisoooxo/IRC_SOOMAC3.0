#!/usr/bin/env python3
"""Bounded training-meaning audit; never starts production or edits sealed inputs.

Reuse benchmark's exact loader and sealed core encoding. Prepare three representative
rows per agent, then run HF/Unsloth in separate processes with identical initial LoRA.
Large local tensor evidence is never intended for external review transmission.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import unicodedata

import benchmark

SOURCE = Path('/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/candidates/compound_semantics_20261101_r2/dataset')
AGENTS = ('decision', 'recommendation', 'response')


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                      separators=(',', ':')).encode()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.open(encoding='utf-8') if line.strip()]


def normalize(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', str(text)).casefold()
                   if c.isalnum())


def utterances(row):
    item = row['input']
    texts = [item[key] for key in ('message', 'current_user_text', 'user_text') if key in item]
    history = item.get('recent_history') or []
    return {normalize(s) for s in [*texts, *(m.get('content', '') for m in history
                    if m.get('role') == 'user')] if normalize(s)}


def group(row):
    return next(iter(sorted(groups(row))), None)


def groups(row):
    meta = row.get('metadata') or {}
    return {(key, str(meta[key])) for key in
            ('dialogue_group', 'conversation_id', 'episode_id', 'scenario_id') if meta.get(key)}


def prepare(destination):
    destination = benchmark.output_dir(destination)
    release = benchmark.read(benchmark.RELEASE / 'RELEASE.json')
    manifest = benchmark.read(benchmark.RELEASE / 'dataset/MANIFEST.json')
    if benchmark.sha(benchmark.RELEASE / 'dataset/MANIFEST.json') != benchmark.MANIFEST_SHA:
        raise ValueError('sealed dataset changed')
    import production
    from transformers import AutoProcessor
    import core
    processor = AutoProcessor.from_pretrained(benchmark.RELEASE / 'assets/model-processor', local_files_only=True)
    all_training = []
    for agent in AGENTS:
        for split in ('train', 'validation'):
            path = benchmark.RELEASE / f'dataset/records/{agent}/{split}.jsonl'
            if benchmark.sha(path) != manifest['file_sha256'][f'records/{agent}/{split}.jsonl']:
                raise ValueError('split changed')
            all_training += rows(path)
    inputs_seen = {digest(row['input']) for row in all_training}
    speech_seen = set().union(*(utterances(row) for row in all_training))
    groups_seen = set().union(*(groups(row) for row in all_training))
    receipts = rows(benchmark.RELEASE / 'validation/TOKEN_MASK_ROWS.jsonl')
    if benchmark.sha(benchmark.RELEASE / 'validation/TOKEN_MASK_ROWS.jsonl') != manifest['token_receipts_sha256']:
        raise ValueError('token receipts changed')
    selected, holdouts, exclusions = [], [], {}
    for agent in AGENTS:
        production.source_preflight(agent)
        core.DATASET_ROOT = benchmark.RELEASE / 'dataset'
        core.MAX_LENGTH = manifest['max_length'][agent]
        cfg = core.STAGE_CONFIGS[agent]
        cfg['schema_path'] = core.DATASET_ROOT / f'contract/{agent}_schema.json'
        if agent == 'response':
            cfg['systems_path'] = core.DATASET_ROOT / 'contract/response_system_by_route.json'
        else:
            cfg['system_path'] = core.DATASET_ROOT / f'contract/{agent}_system.txt'
        prompts, validator = core.load_contract(agent)
        raw = rows(core.DATASET_ROOT / f'records/{agent}/train.jsonl')
        lengths = {r['id']: r for r in receipts if r['agent'] == agent and r['split'] == 'train'}
        chosen = []
        for category, ordering in (
            ('long', sorted(raw, key=lambda r: (-lengths[r['id']]['full_tokens'], r['id']))),
            ('short', sorted(raw, key=lambda r: (lengths[r['id']]['full_tokens'], r['id']))),
            ('multiturn', sorted(raw, key=lambda r: (-len(r['input'].get('recent_history') or []), r['id'])))):
            row = next(r for r in ordering if r['id'] not in chosen)
            chosen.append(row['id'])
            encoded = core.encode_record(agent, copy.deepcopy(row), processor, prompts, validator)
            reference = lengths[row['id']]
            if len(encoded['input_ids']) != reference['full_tokens'] or reference['history_trimmed']:
                raise ValueError('sealed encoding/receipt mismatch')
            selected.append({'agent': agent, 'category': category, 'id': row['id'],
                'input_ids': encoded['input_ids'], 'labels': encoded['labels'],
                'attention_mask': [1]*len(encoded['input_ids']),
                'history_messages': len(row['input'].get('recent_history') or []),
                'source_input_sha256': digest(row['input']),
                'input_ids_sha256': digest(encoded['input_ids']), 'labels_sha256': digest(encoded['labels'])})
        kept, rejected = [], []
        path = SOURCE / f'records/{agent}/holdout.jsonl'
        if benchmark.sha(path) != manifest['file_sha256'][f'records/{agent}/holdout.jsonl']:
            raise ValueError('source holdout differs from pinned manifest')
        for row in rows(path):
            reasons = []
            if digest(row['input']) in inputs_seen:
                reasons.append('same_input')
            if utterances(row) & speech_seen:
                reasons.append('same_normalized_user_utterance_or_history')
            if groups(row) & groups_seen:
                reasons.append('same_dialogue_group')
            if reasons:
                rejected.append({'id': row['id'], 'reasons': reasons})
            else:
                kept.append(row)
        # Freeze membership before model predictions. Old held-out files may have
        # influenced past prompt work; do not pretend these are pristine new gold.
        holdouts += [{'agent': agent, 'record': row} for row in kept]
        exclusions[agent] = {'source_rows': len(kept)+len(rejected), 'kept': len(kept),
                             'excluded': rejected, 'blind_new_authored_set': False}
    benchmark.write_new(destination / 'REPRESENTATIVES.json', {'rows': selected})
    benchmark.write_new(destination / 'FROZEN_EXISTING_HOLDOUT.json', {'rows': holdouts})
    receipt = {'status': 'CPU_SELECTION_FROZEN_NOT_TRAINING_QUALITY_PASS',
        'release_sha256': benchmark.sha(benchmark.RELEASE / 'RELEASE.json'),
        'source_manifest_sha256': benchmark.MANIFEST_SHA,
        'representatives_sha256': benchmark.sha(destination / 'REPRESENTATIVES.json'),
        'holdout_sha256': benchmark.sha(destination / 'FROZEN_EXISTING_HOLDOUT.json'),
        'representatives': [{k: v for k, v in r.items() if k not in ('input_ids', 'labels', 'attention_mask')} |
                            {'length': len(r['input_ids']), 'targets': sum(x != -100 for x in r['labels'])}
                           for r in selected],
        'holdout_leakage_audit': exclusions,
        'new_blind_group_holdout': 'REQUIRED_NOT_YET_AUTHORED_OR_REVIEWED',
        'gates': {'input_label_history_targets_trainable_names': 'exact',
                  'native_ce_absolute_delta_max': 0.001,
                  'gradient_comparison': 'report_full_norm_cosine_relativeL2_no_threshold_relaxation',
                  'functional_safety_and_required_demo': 1.0, 'other_functional_success_rate': .95,
                  'full_state_resume_and_standard_peft_reload': 'required',
                  'production_authorized': False}}
    benchmark.write_new(destination / 'QUALITY_PREPARE.json', receipt)
    return receipt


def tensor_digest(torch, value):
    return hashlib.sha256(value.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def probe(args):
    destination = benchmark.output_dir(args.output)
    prepared = benchmark.read(args.prepared / 'QUALITY_PREPARE.json')
    if benchmark.sha(args.prepared / 'REPRESENTATIVES.json') != prepared['representatives_sha256']:
        raise ValueError('representatives changed')
    selected = benchmark.read(args.prepared / 'REPRESENTATIVES.json')['rows']
    selected += [copy.deepcopy(next(r for r in selected if r['agent'] == agent and r['category'] == 'long'))
                 | {'category': 'long_repeat'} for agent in AGENTS]
    fd, budget = None, None
    result = {'status': 'FAILED', 'engine': args.engine, 'optimizer_updates': 0,
              'production_changed': False, 'representatives_sha256': prepared['representatives_sha256']}
    try:
        fd = benchmark.acquire_gpu_lock()
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', UNSLOTH_RETURN_LOGITS='1')
        if args.engine == 'unsloth':
            import unsloth  # noqa: F401
        import torch
        from transformers import AutoProcessor, DataCollatorForSeq2Seq, set_seed, Trainer, TrainingArguments
        from safetensors.torch import save_file, load_file
        sys.path.insert(0, str(benchmark.RELEASE / 'common'))
        from budget import Budget
        from loss import scored_suffix
        set_seed(42)
        budget = Budget(benchmark.PROCESS_BUDGET, benchmark.RESERVE)
        processor = AutoProcessor.from_pretrained(benchmark.RELEASE / 'assets/model-processor', local_files_only=True)
        model, optimizer, parameters, names = benchmark.model_and_optimizer(args.engine, 'response', torch)
        # Decoder/LoRA graph is shared across D/Q/P. This probe does not claim
        # agent-specific optimizer/scheduler or Q's production5120 loader passed.
        if args.engine == 'baseline':
            save_file({n: p.detach().cpu().contiguous() for n, p in zip(names, parameters)},
                      str(destination / 'INITIAL_LORA.safetensors'))
        else:
            initial = load_file(str(args.initial))
            if set(initial) != set(names):
                raise ValueError('LoRA target names differ')
            with torch.no_grad():
                for n, p in zip(names, parameters):
                    p.copy_(initial[n].to(device=p.device, dtype=p.dtype))
            del initial
        train_args = TrainingArguments(output_dir=str(destination / 'trainer_metadata'), report_to='none',
            per_device_train_batch_size=1, gradient_accumulation_steps=16, average_tokens_across_devices=False)
        trainer = Trainer(model=model, args=train_args, optimizers=(optimizer, None))
        result['inferred_accepts_loss_kwargs'] = trainer.model_accepts_loss_kwargs
        if args.engine == 'unsloth':
            sys.path.insert(0, str(Path(__file__).parents[1] / 'unsloth_training'))
            import production
            production.enforce_microbatch_mean(trainer, {'batch': 1, 'accumulation': 16})
        if trainer.model_accepts_loss_kwargs is not False:
            raise RuntimeError('baseline loss normalization differs; stop before gradient comparison')
        result['active_accepts_loss_kwargs'] = trainer.model_accepts_loss_kwargs
        result['trainable_names_sha256'] = digest(names)
        result['initial_lora_sha256'] = digest([(n, tensor_digest(torch, p)) for n, p in zip(names, parameters)])
        dropouts = [(m, m.p, m.training) for n, m in model.named_modules()
                    if 'lora_dropout' in n and isinstance(m, torch.nn.Dropout)]
        if not dropouts or any(abs(p-.05) > 1e-9 for m, p, training in dropouts):
            raise RuntimeError('LoRA dropout .05 missing')
        collator = DataCollatorForSeq2Seq(tokenizer=processor.tokenizer, padding=True,
            label_pad_token_id=-100, pad_to_multiple_of=8, return_tensors='pt')
        initial_buffers = {n: b.detach().cpu().clone() for n, b in model.named_buffers()}
        cases = []
        for index, row in enumerate(selected):
            with torch.no_grad():
                for name, buffer in model.named_buffers():
                    buffer.copy_(initial_buffers[name].to(buffer.device))
            batch = {k: v.to('cuda') for k, v in collator([
                {k: copy.deepcopy(row[k]) for k in ('input_ids', 'labels', 'attention_mask')}]).items()}
            inputs = scored_suffix(batch)
            model.eval()
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                out = model(**inputs, use_cache=False)
            logits = out.logits.detach().float().cpu()
            ce = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                                   inputs['shift_labels'].detach().cpu().reshape(-1))
            loss = float(out.loss.detach())
            if not math.isfinite(loss) or abs(loss-float(ce)) > .001:
                raise RuntimeError('native CE semantic mismatch')
            torch.save(logits, destination / f'case{index}_logits.pt')
            del out, logits
            model.train()
            for m, p, training in dropouts:
                m.p = 0.0
            model.zero_grad(set_to_none=True)
            set_seed(42)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model(**inputs, use_cache=False)
            train_loss = float(out.loss.detach())
            (out.loss / 16).backward()
            grad = torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).detach().float().cpu().reshape(-1)
                              for p in parameters])
            if not bool(torch.isfinite(grad).all()) or not bool(torch.count_nonzero(grad)):
                raise RuntimeError('nonfinite or zero gradient')
            torch.save(grad, destination / f'case{index}_gradient.pt')
            item = {'agent': row['agent'], 'id': row['id'], 'category': row['category'],
                'padded_length': int(batch['input_ids'].shape[1]),
                'labels_sha256': tensor_digest(torch, inputs['shift_labels']),
                'target_positions_sha256': tensor_digest(torch, inputs['logits_to_keep']),
                'input_ids_sha256': tensor_digest(torch, batch['input_ids']),
                'attention_mask_sha256': tensor_digest(torch, batch['attention_mask']),
                'eval_loss': loss, 'manual_native_ce': float(ce), 'train_dropout0_loss': train_loss,
                'gradient_norm_fp64': float(grad.double().norm()),
                'gradient_finite_nonzero': True, 'diagnostic_dropout': 0.0, 'loss_divisor': 16}
            benchmark.write_new(destination / f'case{index}.json', item)
            cases.append(item)
            print(json.dumps({'phase': 'representative_complete', **item}), flush=True)
            del grad, out, inputs, batch
            model.zero_grad(set_to_none=True)
            for m, p, training in dropouts:
                m.p = p
            budget.check(force=True)
        result.update(status='REPRESENTATIVE_PROBE_COMPLETE_NOT_FINAL_QUALITY_PASS', cases=cases,
            same_shared_decoder_max_length=9216, agent_specific_training_not_exercised=True,
            production_dropout=.05, full_state_resume='not_tested_in_this_probe',
            functional_evaluation='not_run_in_this_probe')
    finally:
        if budget is not None:
            result.update(budget.summary(), peak_allocated_bytes=budget.torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=budget.torch.cuda.max_memory_reserved())
            budget.close()
        result['cpu_peak_rss_bytes'] = benchmark.process_peak_rss_bytes()
        benchmark.write_new(destination / 'MODEL_PROBE.json', result)
        if fd is not None:
            os.close(fd)
    return result


def compare(args):
    import torch
    destination = benchmark.output_dir(args.output)
    a, b = (benchmark.read(p / 'MODEL_PROBE.json') for p in (args.baseline, args.unsloth))
    if (a.get('status') != 'REPRESENTATIVE_PROBE_COMPLETE_NOT_FINAL_QUALITY_PASS'
            or b.get('status') != a['status'] or len(a.get('cases', [])) != 12
            or len(b.get('cases', [])) != 12 or a['engine'] != 'baseline' or b['engine'] != 'unsloth'
            or a['representatives_sha256'] != b['representatives_sha256']):
        raise ValueError('nine representatives plus three A/A repeats required in both engines')
    if a['initial_lora_sha256'] != b['initial_lora_sha256'] or a['trainable_names_sha256'] != b['trainable_names_sha256']:
        raise ValueError('initial LoRA/target names differ')
    comparisons = []
    for i, (x, y) in enumerate(zip(a['cases'], b['cases'])):
        if any(x[k] != y[k] for k in ('agent', 'id', 'category', 'input_ids_sha256', 'attention_mask_sha256',
                                    'labels_sha256', 'target_positions_sha256')):
            raise ValueError('input/label alignment mismatch')
        item = {'agent': x['agent'], 'id': x['id'], 'category': x['category'],
                'eval_loss_delta': abs(x['eval_loss']-y['eval_loss']),
                'train_dropout0_loss_delta': abs(x['train_dropout0_loss']-y['train_dropout0_loss'])}
        for suffix in ('logits', 'gradient'):
            tensors = [torch.load(p / f'case{i}_{suffix}.pt', weights_only=True, map_location='cpu')
                       for p in (args.baseline, args.unsloth)]
            # FP64 bounded reduction for65M gradients: avoid invalid FP32 cosine>1.
            item[suffix] = _chunked_diff(torch, *tensors)
            del tensors
        comparisons.append(item)
    self_repeats = []
    for engine, directory, receipt in (('baseline', args.baseline, a), ('unsloth', args.unsloth, b)):
        for j, agent in enumerate(AGENTS):
            original = next(i for i, case in enumerate(receipt['cases'])
                            if case['agent'] == agent and case['category'] == 'long')
            repeated = 9+j
            if receipt['cases'][original]['id'] != receipt['cases'][repeated]['id']:
                raise ValueError('A/A repeated source differs')
            entry = {'engine': engine, 'agent': agent}
            for suffix in ('logits', 'gradient'):
                pair = [torch.load(directory / f'case{i}_{suffix}.pt', weights_only=True, map_location='cpu')
                        for i in (original, repeated)]
                entry[suffix] = _chunked_diff(torch, *pair)
                del pair
            self_repeats.append(entry)
    result = {'status': 'COMPARISON_RECORDED_NOT_FUNCTIONAL_QUALITY_PASS', 'rows': comparisons,
              'same_engine_repeats': self_repeats,
              'loss_gate_all_pass': all(x['eval_loss_delta'] <= .001 for x in comparisons),
              'diagnostic_gradient_gate_all_pass': all(x['gradient']['allclose_original_tolerance'] for x in comparisons),
              'same_engine_gradient_repeat_gate_all_pass': all(x['gradient']['allclose_original_tolerance'] for x in self_repeats),
              'production_authorized': False, 'full_state_and_functional_gates_still_required': True}
    benchmark.write_new(destination / 'COMPARISON.json', result)
    return result


def _chunked_diff(torch, a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    if a.shape != b.shape:
        raise ValueError('tensor shape differs')
    accum = torch.zeros(4, dtype=torch.float64)
    maximum = 0.0
    allclose = True
    for i in range(0, a.numel(), 1 << 20):
        x, y = a[i:i+(1 << 20)].double(), b[i:i+(1 << 20)].double()
        accum += torch.stack((x.square().sum(), y.square().sum(), (x*y).sum(), (y-x).square().sum()))
        maximum = max(maximum, float((y-x).abs().max()))
        allclose &= bool(torch.allclose(x, y, atol=1e-4, rtol=1e-3))
    an, bn = float(accum[0].sqrt()), float(accum[1].sqrt())
    cosine = (1.0 if an == bn == 0 else 0.0 if an == 0 or bn == 0
              else max(-1.0, min(1.0, float(accum[2])/(an*bn))))
    return {'reference_norm': an, 'candidate_norm': bn,
            'cosine': cosine,
            'relative_l2_error': float(accum[3].sqrt())/max(an, 1e-30),
            'max_absolute': maximum, 'allclose_original_tolerance': allclose,
            'finite': bool(torch.isfinite(accum).all()), 'reduction': 'FP64_chunked'}


def independent_targets(torch, batch, selected):
    """Derive next-token targets from full labels, independently of scored_suffix."""
    labels = batch['labels']
    if labels.shape[0] != 1 or labels[0, 0].item() != -100:
        raise ValueError('one masked-prefix row required')
    target_indexes = (labels[0] != -100).nonzero(as_tuple=False).flatten()
    positions = target_indexes - 1
    targets = labels[:, target_indexes]
    if (not torch.equal(selected['logits_to_keep'], positions)
            or not torch.equal(selected['shift_labels'], targets)):
        raise ValueError('selected causal targets differ from full labels')
    if any(selected[k] is not batch[k] for k in ('input_ids', 'labels', 'attention_mask')):
        raise ValueError('scored_suffix replaced original input/mask/labels')
    return targets


def local_objective(args):
    """Three agent-specific eval CE checks; optimizer/resume tested by production smoke."""
    import production
    production.configure_environment(argparse.Namespace(execution_environment='local'))
    prepared = benchmark.read(args.prepared / 'QUALITY_PREPARE.json')
    if (prepared['release_sha256'] != benchmark.sha(benchmark.RELEASE / 'RELEASE.json')
            or prepared['source_manifest_sha256'] != benchmark.MANIFEST_SHA
            or benchmark.sha(args.prepared / 'REPRESENTATIVES.json') != prepared['representatives_sha256']):
        raise ValueError('prepared representatives identity changed')
    representatives = [r for r in benchmark.read(args.prepared / 'REPRESENTATIVES.json')['rows']
                       if r['agent'] == args.agent]
    if len(representatives) != 3 or {r['category'] for r in representatives} != {'short', 'long', 'multiturn'}:
        raise ValueError('need exactly short/long/multiturn for this agent')
    _, manifest, _ = production.source_preflight(args.agent)
    destination = benchmark.output_dir(args.output)
    result = {'status': 'FAILED', 'agent': args.agent, 'optimizer_updates': 0,
              'production_changed': False, 'functional_quality': 'NOT_TESTED',
              'representatives_sha256': prepared['representatives_sha256'], 'cases': []}
    fd, budget = None, None
    try:
        fd = benchmark.acquire_gpu_lock()
        import unsloth  # noqa: F401
        import torch
        from transformers import AutoProcessor, DataCollatorForSeq2Seq
        from budget import Budget
        from loss import scored_suffix
        import core
        budget = Budget(benchmark.PROCESS_BUDGET, benchmark.RESERVE)
        processor = AutoProcessor.from_pretrained(production.PROCESSOR, local_files_only=True)
        core.DATASET_ROOT = production.DATASET
        core.MAX_LENGTH = manifest['max_length'][args.agent]
        cfg = core.STAGE_CONFIGS[args.agent]
        cfg['schema_path'] = production.DATASET / f'contract/{args.agent}_schema.json'
        if args.agent == 'response':
            cfg['systems_path'] = production.DATASET / 'contract/response_system_by_route.json'
        else:
            cfg['system_path'] = production.DATASET / f'contract/{args.agent}_system.txt'
        prompts, validator = core.load_contract(args.agent)
        source = {r['id']: r for r in rows(production.DATASET / f'records/{args.agent}/train.jsonl')}
        for row in representatives:
            original = source[row['id']]
            encoded = core.encode_record(args.agent, copy.deepcopy(original), processor, prompts, validator)
            if (digest(original['input']) != row['source_input_sha256']
                    or encoded['input_ids'] != row['input_ids'] or encoded['labels'] != row['labels']
                    or row['attention_mask'] != [1] * encoded['length']
                    or len(original['input'].get('recent_history') or []) != row['history_messages']):
                raise ValueError('full representative input/history/labels changed')
        model, optimizer, parameters, names = benchmark.model_and_optimizer(
            'unsloth', args.agent, torch, max_seq_length=core.MAX_LENGTH)
        result['initial_lora_B_all_zero'] = all(not bool(torch.count_nonzero(p))
            for n, p in zip(names, parameters) if 'lora_B' in n)
        model.eval()
        collator = DataCollatorForSeq2Seq(tokenizer=processor.tokenizer, padding=True,
            label_pad_token_id=-100, pad_to_multiple_of=8, return_tensors='pt')
        for row in representatives:
            batch = {k: v.to('cuda') for k, v in collator([
                {k: copy.deepcopy(row[k]) for k in ('input_ids', 'labels', 'attention_mask')}]).items()}
            selected = scored_suffix(batch)
            targets = independent_targets(torch, batch, selected)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                out = model(**selected, use_cache=False)
            ce = torch.nn.functional.cross_entropy(out.logits.float().reshape(-1, out.logits.shape[-1]),
                                                  targets.reshape(-1))
            loss, reference = float(out.loss), float(ce)
            if not math.isfinite(loss) or not math.isfinite(reference) or abs(loss-reference) > .001:
                raise RuntimeError('independent native masked CE mismatch')
            item = {'id': row['id'], 'category': row['category'],
                    'tokens': len(row['input_ids']), 'padded_tokens': int(batch['input_ids'].shape[1]),
                    'history_messages': row['history_messages'], 'targets': targets.numel(),
                    'input_ids_sha256': row['input_ids_sha256'], 'labels_sha256': row['labels_sha256'],
                    'model_loss': loss, 'independent_ce': reference, 'absolute_delta': abs(loss-reference)}
            result['cases'].append(item)
            print(json.dumps({'event': 'local_objective_case', **item}), flush=True)
            del out, ce, batch, selected, targets
            budget.check(force=True)
        result.update(status='LOCAL_REPRESENTATIVE_OBJECTIVE_PASS_NOT_FUNCTIONAL_QUALITY_PASS',
            max_length=core.MAX_LENGTH, production_dropout=.05,
            model_builder_sha256=benchmark.model_builder_fingerprint(), packages=production.versions(),
            release_sha256=prepared['release_sha256'], source_manifest_sha256=benchmark.MANIFEST_SHA)
    finally:
        if budget:
            result.update(budget.summary())
            budget.close()
        result['cpu_peak_rss_bytes'] = benchmark.process_peak_rss_bytes()
        benchmark.write_new(destination / 'LOCAL_OBJECTIVE.json', result)
        if fd is not None:
            os.close(fd)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'probe', 'compare', 'local-objective'))
    parser.add_argument('--agent', choices=('response', 'recommendation'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--engine', choices=('baseline', 'unsloth'))
    parser.add_argument('--initial', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--unsloth', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).parents[1] / 'unsloth_training'))
    if args.command == 'probe' and (not args.prepared or not args.engine or
                                  (args.engine == 'unsloth' and not args.initial)):
        parser.error('probe needs prepared/engine and Unsloth needs initial LoRA')
    if args.command == 'compare' and (not args.baseline or not args.unsloth):
        parser.error('compare needs both probe directories')
    if args.command == 'local-objective' and (not args.prepared or not args.agent):
        parser.error('local-objective needs prepared and agent')
    result = (local_objective(args) if args.command == 'local-objective' else
              prepare(args.output) if args.command == 'prepare' else
              probe(args) if args.command == 'probe' else compare(args))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
