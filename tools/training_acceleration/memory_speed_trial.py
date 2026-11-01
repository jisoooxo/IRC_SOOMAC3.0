#!/usr/bin/env python3
"""Disposable GC/batch experiment; original benchmark and production stay unchanged."""
import argparse
import inspect
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

import benchmark


def args_parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoded", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gc", choices=("off", "on", "native", "partial"), required=True,
                        help="on=smart GC/offload; native=full native GC without smart offload")
    parser.add_argument("--checkpoint-skip", type=int, choices=(8, 16, 24), default=16)
    parser.add_argument("--microbatch", choices=(1, 2), type=int, default=1)
    parser.add_argument("--warmup", choices=(0, 1), type=int, default=0)
    parser.add_argument("--updates", choices=(1, 2), type=int, default=1)
    parser.add_argument("--reference-loss", type=Path, required=True)
    parser.add_argument("--profile", action="store_true",
                        help="One extra forward/backward AFTER timing; no extra optimizer update")
    return parser.parse_args(argv)


def gc_state(model):
    flags = []
    functions = {}
    for name, module in model.named_modules():
        if hasattr(module, "gradient_checkpointing"):
            flags.append({"module": name, "enabled": bool(module.gradient_checkpointing)})
        fn = getattr(module, "_gradient_checkpointing_func", None)
        if fn is not None:
            functions[name] = getattr(fn, "__qualname__", str(fn))[:180]
    return {"is_gradient_checkpointing": bool(getattr(model, "is_gradient_checkpointing", False)),
            "unsloth_gradient_checkpointing": getattr(model, "_unsloth_gradient_checkpointing", None),
            "enabled_modules": [item["module"] for item in flags if item["enabled"]],
            "flagged_module_count": len(flags), "checkpoint_functions": functions}


def validate_gc_state(state, mode, skip):
    """Check actual decoder flags, not the loader argument alone."""
    enabled = state["enabled_modules"]
    decoder_count = sum(".layers." in name and name.rsplit(".", 1)[-1].isdigit()
                        for name in enabled)
    expected = 0 if mode == "off" else 48 - (skip if mode == "partial" else 0)
    if decoder_count != expected or bool(state["is_gradient_checkpointing"]) != (mode != "off"):
        raise RuntimeError(f"actual GC decoder flags {decoder_count}, expected {expected}")
    return decoder_count


def profile_backward(model, inputs, parameters, torch):
    """Record runtime functions/kernels without serializing input values or weights."""
    from loss import scored_suffix
    for parameter in parameters:
        parameter.grad = None
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as profiler:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**scored_suffix(inputs), use_cache=False).loss
        loss.backward()
        torch.cuda.synchronize()
    selected = []
    for event in profiler.key_averages():
        if any(word in event.key.lower() for word in
               ("checkpoint", "memcpy", "attention", "cutlass", "_lh_", "_to_copy")):
            selected.append({"name": event.key, "count": event.count})
    for parameter in parameters:
        parameter.grad = None
    return {"scope": "extra single forward/backward after timing; no optimizer update",
            "events": selected}


def offload_state(model):
    import torch.utils.checkpoint as checkpoint
    import unsloth_zoo.gradient_checkpointing as gc
    buffers = getattr(gc, "CPU_BUFFERS", [])
    return {
        "input_embedding_device": str(model.get_input_embeddings().weight.device),
        "block_swap_present": getattr(model, "_unsloth_block_swap", None) is not None,
        "torch_checkpoint_function": checkpoint.CheckpointFunction.__name__,
        "cpu_activation_buffer_count": len(buffers),
        "cpu_activation_buffer_bytes": sum(x.numel() * x.element_size() for x in buffers),
        "offload_generation": getattr(gc, "OFFLOAD_GENERATION", None),
        "nonreentrant_activation_stats": gc.get_activation_offload_stats(),
        "stats_scope": "nonreentrant stats plus smart-GC CPU buffers; embedding offload separate",
    }


def load_with_gc(gc_setting, torch, checkpoint_skip=0):
    """Override the two loader arguments in this process only, not on disk."""
    from unsloth import FastModel
    original_from = inspect.getattr_static(FastModel, "from_pretrained")
    original_peft = inspect.getattr_static(FastModel, "get_peft_model")
    from_fn, peft_fn = FastModel.from_pretrained, FastModel.get_peft_model
    calls = []

    def load(*positional, **named):
        named["use_gradient_checkpointing"] = gc_setting
        calls.append({"function": "from_pretrained", "use_gradient_checkpointing": gc_setting})
        return from_fn(*positional, **named)

    def adapt(*positional, **named):
        named["use_gradient_checkpointing"] = gc_setting
        if checkpoint_skip:
            named["checkpoint_skip_layers"] = checkpoint_skip
        calls.append({"function": "get_peft_model", "use_gradient_checkpointing": gc_setting})
        return peft_fn(*positional, **named)

    try:
        FastModel.from_pretrained = staticmethod(load)
        FastModel.get_peft_model = staticmethod(adapt)
        loaded = benchmark.model_and_optimizer("unsloth", "response", torch)
    finally:
        FastModel.from_pretrained = original_from
        FastModel.get_peft_model = original_peft
    return loaded, calls


def update(model, optimizer, parameters, inputs, torch, budget, phase, progress):
    from accelerate.utils import convert_to_fp32
    from loss import scored_suffix
    accumulation = len(inputs)
    losses = []
    torch.cuda.synchronize()
    started = time.perf_counter()
    for index, batch in enumerate(inputs, 1):
        progress.update(phase=phase + ":forward", microstep=index)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = convert_to_fp32(model(**scored_suffix(batch), use_cache=False))
            loss = output.loss / accumulation
        losses.append(loss.detach())
        progress["phase"] = phase + ":backward"
        loss.backward()
        del output, loss
    torch.cuda.synchronize()
    compute = time.perf_counter() - started
    values = torch.stack(losses)
    gradients = [p.grad for p in parameters if p.grad is not None]
    if (not bool(torch.isfinite(values).all()) or not gradients
            or not all(bool(torch.isfinite(g).all()) for g in gradients)
            or not any(bool(torch.count_nonzero(g)) for g in gradients)):
        raise RuntimeError("nonfinite/zero loss or LoRA gradient")
    torch.cuda.synchronize()
    step_started = time.perf_counter()
    progress["phase"] = phase + ":optimizer"
    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    seconds = compute + time.perf_counter() - step_started
    budget.check(force=True)
    item = {"seconds": seconds, "mean_loss": float(values.sum().item()),
            "finite_nonzero_gradient": True, "optimizer_updates": 1}
    print(json.dumps({"phase": phase, **item}), flush=True)
    return item


def main(argv=None):
    args = args_parse(argv)
    payload = benchmark.validate_payload(args.encoded, "response")
    output = benchmark.output_dir(args.output)
    result = {"status": "FAILED", "production_training": False,
              "gc": args.gc, "microbatch": args.microbatch,
              "checkpoint_skip_layers": args.checkpoint_skip if args.gc == "partial" else 0,
              "accumulation": 16 // args.microbatch, "effective_batch": 16,
              "encoded_sha256": benchmark.sha(args.encoded),
              "reference_loss_sha256": benchmark.sha(args.reference_loss),
              "baseline_builder_sha256": benchmark.model_builder_fingerprint(),
              "trial_script_sha256": benchmark.sha(Path(__file__)),
              "loss_implementation": "native_selected_logits_RETURN_LOGITS1"}
    progress = {"phase": "initialization", "microstep": 0}
    fd = budget = None
    try:
        fd = benchmark.acquire_gpu_lock()
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                          TOKENIZERS_PARALLELISM="false", UNSLOTH_RETURN_LOGITS="1",
                          UNSLOTH_DISABLE_STATISTICS="1")
        import unsloth  # noqa: F401
        import torch
        from transformers import AutoProcessor, DataCollatorForSeq2Seq, TrainingArguments, set_seed
        sys.path.insert(0, str(benchmark.RELEASE / "common"))
        from budget import Budget
        from loss import scored_suffix
        set_seed(42)
        budget = Budget(benchmark.PROCESS_BUDGET, benchmark.RESERVE)
        processor = AutoProcessor.from_pretrained(benchmark.RELEASE / "assets/model-processor",
                                                  local_files_only=True)
        collator = DataCollatorForSeq2Seq(tokenizer=processor.tokenizer, padding=True,
                                         label_pad_token_id=-100, pad_to_multiple_of=8,
                                         return_tensors="pt")
        row = {key: payload["rows"][0][key] for key in ("input_ids", "labels")}
        one = {key: value.to("cuda") for key, value in collator([row]).items()}
        inputs = [{key: value.to("cuda") for key, value in collator([row] * args.microbatch).items()}
                  for _ in range(16 // args.microbatch)]
        gc_setting = {"off": False, "on": "unsloth", "native": True, "partial": True}[args.gc]
        skip = args.checkpoint_skip if args.gc == "partial" else 0
        (model, optimizer, parameters, names), calls = load_with_gc(gc_setting, torch, skip)
        # No Trainer runs in this matched manual-update benchmark. Explicitly
        # validate its requested flag without silently enabling GC in a Trainer.
        training_args = TrainingArguments(output_dir=str(output / "unused-trainer"),
                                          gradient_checkpointing=False, report_to="none")
        result.update(loader_calls=calls, trainer_gradient_checkpointing=training_args.gradient_checkpointing,
                      loop="manual_optimizer_same_as_276s_reference_not_Trainer_training",
                      gc_state_loaded=gc_state(model), offload_state_loaded=offload_state(model))
        if args.gc == "off" and (result["gc_state_loaded"]["is_gradient_checkpointing"]
                                  or result["gc_state_loaded"]["enabled_modules"]):
            raise RuntimeError("Unsloth re-enabled GC despite explicit False")
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            loss = float(model(**scored_suffix(one), use_cache=False).loss.float().item())
        reference = benchmark.read(args.reference_loss)
        if (reference.get("status") != "LOSS_CHECK_PASS_ONE_ROW_ONLY"
                or reference.get("encoded_sha256") != result["encoded_sha256"]):
            raise ValueError("reference loss is not from this exact input")
        wanted = float(reference["loss_check"]["native_selected_ce"])
        result["eval_loss_check"] = {"loss": loss, "reference_loss": wanted,
                                      "absolute_delta": abs(loss - wanted),
                                      "passed": math.isclose(loss, wanted, rel_tol=1e-3, abs_tol=1e-3)}
        if not result["eval_loss_check"]["passed"]:
            raise RuntimeError("same-input native loss differs")
        model.train()
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        result["gc_state_before_update"] = gc_state(model)
        result["offload_state_before_update"] = offload_state(model)
        result["verified_checkpointed_decoder_layers"] = validate_gc_state(
            result["gc_state_before_update"], args.gc, skip)
        if args.gc == "off" and (model.is_gradient_checkpointing
                                  or result["gc_state_before_update"]["enabled_modules"]):
            raise RuntimeError("model.train re-enabled GC")
        for _ in range(args.warmup):
            update(model, optimizer, parameters, inputs, torch, budget, "warmup", progress)
        torch.cuda.reset_peak_memory_stats()
        items = [update(model, optimizer, parameters, inputs, torch, budget, "timed", progress)
                 for _ in range(args.updates)]
        nonpad = sum(int(batch["attention_mask"].sum().item()) for batch in inputs)
        result.update(status="UPDATE_PASS", warmup_updates=args.warmup,
                      timed_updates=items, nonpad_tokens_per_update=nonpad,
                      nonpad_tokens_per_second=nonpad * len(items) / sum(x["seconds"] for x in items),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      gc_state_after_update=gc_state(model),
                      offload_state_after_update=offload_state(model))
        validate_gc_state(result["gc_state_after_update"], args.gc, skip)
        if args.profile:
            result["runtime_profile"] = profile_backward(model, one, parameters, torch)
            result["offload_state_after_profile"] = offload_state(model)
        budget.close()
        budget.check(force=True)
        result.update(budget.summary())
        budget = None
    except BaseException as error:
        result.update(status="OOM" if "out of memory" in str(error).lower() else "FAILED",
                      error=f"{type(error).__name__}: {error}", last_progress=progress)
        if budget is not None:
            result.update(budget.summary())
            try:
                result.update(peak_allocated_bytes=budget.torch.cuda.max_memory_allocated(),
                              peak_reserved_bytes=budget.torch.cuda.max_memory_reserved())
            except Exception:
                pass
        raise
    finally:
        if budget is not None:
            budget.close()
        result["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        benchmark.write_new(output / "RESULT.json", result)
        if fd is not None:
            os.close(fd)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
