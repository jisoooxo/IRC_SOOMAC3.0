#!/usr/bin/env python3
"""Fresh standard-HF/PEFT load of an Unsloth disposable export, no training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import benchmark


def run(args):
    output = benchmark.output_dir(args.output)
    sys.path.insert(0, str(benchmark.RELEASE / 'common'))
    from checkpointing import verify_completion, member
    complete = verify_completion(args.smoke, benchmark.RELEASE / 'RELEASE.json', 'response', benchmark.BASE)
    if complete['config'].get('engine') != 'unsloth' or complete['config'].get('mode') != 'smoke':
        raise ValueError('only independently generated Unsloth smoke export is allowed')
    checkpoint = member(args.smoke, complete['checkpoint'], directory=True)
    adapter = checkpoint.parent / 'smoke_final_adapter'
    restore = benchmark.read(checkpoint.parent / 'RESTORE_EVIDENCE.json')
    if set(restore) != {'adapter', 'optimizer', 'scheduler', 'rng'} or not all(x['equal'] is True for x in restore.values()):
        raise ValueError('actual native full-state restore proof missing')
    fd, budget = None, None
    result = {'status': 'FAILED', 'optimizer_updates': 0, 'production_changed': False}
    try:
        fd = benchmark.acquire_gpu_lock()
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        if 'unsloth' in sys.modules:
            raise RuntimeError('standard PEFT reload requires fresh process without Unsloth patches')
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor, DataCollatorForSeq2Seq
        from peft import PeftModel, get_peft_model_state_dict
        from safetensors.torch import load_file
        from budget import Budget
        from loss import scored_suffix
        budget = Budget(benchmark.PROCESS_BUDGET, benchmark.RESERVE)
        base = AutoModelForMultimodalLM.from_pretrained(benchmark.BASE, device_map={'': 0},
            dtype=torch.bfloat16, quantization_config=benchmark.quantization(torch),
            attn_implementation='sdpa', local_files_only=True, low_cpu_mem_usage=True)
        model = PeftModel.from_pretrained(base, adapter, is_trainable=False, local_files_only=True)
        model.eval()
        saved = load_file(str(adapter / 'adapter_model.safetensors'))
        loaded = get_peft_model_state_dict(model)
        if set(saved) != set(loaded) or not all(torch.equal(saved[k], loaded[k].detach().cpu()) for k in saved):
            raise RuntimeError('standard PEFT adapter tensor restore mismatch')
        processor = AutoProcessor.from_pretrained(benchmark.RELEASE / 'assets/model-processor', local_files_only=True)
        row = next(r for r in benchmark.read(args.prepared / 'REPRESENTATIVES.json')['rows'] if r['agent'] == 'response' and r['category'] == 'short')
        batch = DataCollatorForSeq2Seq(tokenizer=processor.tokenizer, padding=True,
            label_pad_token_id=-100, pad_to_multiple_of=8, return_tensors='pt')([
                {k: row[k] for k in ('input_ids', 'labels', 'attention_mask')}])
        batch = {k: v.to('cuda') for k, v in batch.items()}
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = model(**scored_suffix(batch), use_cache=False)
        if not torch.isfinite(prediction.loss) or not torch.isfinite(prediction.logits).all():
            raise RuntimeError('reloaded adapter inference is nonfinite')
        result.update(status='STANDARD_PEFT_RELOAD_AND_FORWARD_PASS_NOT_FUNCTIONAL_QUALITY_PASS',
            full_state_native_restore=restore, saved_adapter_tensor_count=len(saved),
            all_adapter_tensors_exactly_loaded=True, representative_id=row['id'],
            forward_loss=float(prediction.loss), checkpoint_step=complete['global_step'],
            complete_receipt_sha256=benchmark.sha(args.smoke / 'training_complete.json'),
            adapter_sha256=benchmark.sha(adapter / 'adapter_model.safetensors'),
            unsloth_imported_in_reload_process=False, functional_validation='not_run_here')
        budget.check(force=True)
    finally:
        if budget:
            result.update(budget.summary(), peak_allocated_bytes=budget.torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=budget.torch.cuda.max_memory_reserved())
            budget.close()
        result['cpu_peak_rss_bytes'] = benchmark.process_peak_rss_bytes()
        benchmark.write_new(output / 'PEFT_RELOAD.json', result)
        if fd is not None:
            os.close(fd)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False))
