#!/usr/bin/env python3
"""Disposable Gemma-4 attention ablation: is the head_dim-512 global kernel actually
selected, and does it change update time / loss / LoRA gradients?

Never trains production. Reuses the sibling benchmark's sealed loader, encoded
artifact, GPU lock, budget and scored_suffix so the model graph, LoRA config and
token masking are identical to the baseline. This file only adds measurement:
per-layer attention-path attribution, a within-process head_dim-512 kernel toggle,
and a flex-attention viability probe. Importing it never imports torch or Unsloth.

Three subcommands, each a new output directory:
  prepare  CPU only; delegates to benchmark.prepare and stamps an environment manifest
           (library versions, source-file sha256 + reviewed line numbers, CPU RSS).
  check    GPU, no timing. First a production-representative, uninstrumented
           model.train() forward+backward profiled for the real attention kernels and
           router counter; then a separate intrusive per-layer recorder captures one
           global/sliding tensor set to probe HD512-vs-SDPA numerics and flex.
  compare  GPU. Bounded by default: a no-step loss / LoRA-gradient equivalence between
           the HD512 OFF and ON paths at identical weights/RNG, which also proves the
           toggle actually changes the kernel. Warm timing (matched warmup + timed
           updates, both variants restored to the same initial state) runs only with
           --run-timing and only if the toggle changed the kernel.
"""

import argparse
import ast
from collections import Counter
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_PATH = SCRIPT_DIR / "benchmark.py"

# Reviewed installed sources (read-only). Hashes are recorded at run time; the line
# numbers are the exact evidence this ablation was designed against, so a drift in
# either file is visible against this record.
REVIEWED_SOURCES = {
    "unsloth_gemma4_flash_sliding": {
        "path": str(SCRIPT_DIR.parents[1] / "training/learning_first_v43/"
                    "training_release_20261101_r5/env/lib/python3.10/site-packages/"
                    "unsloth_zoo/temporary_patches/gemma4_flash_sliding.py"),
        "lines": {"normal_router": 90, "full_band_required": 124},
    },
    "unsloth_gemma4_banded_attention": {
        "path": str(SCRIPT_DIR.parents[1] / "training/learning_first_v43/"
                    "training_release_20261101_r5/env/lib/python3.10/site-packages/"
                    "unsloth_zoo/temporary_patches/gemma4_banded_attention.py"),
        "lines": {"plain_band_mask_check": 75, "compile_rejects_mask_scan": 85,
                  "block_local_sdpa": 119},
    },
    "unsloth_large_head_attention": {
        "path": "/tmp/soomac-training-accel-20261101/venv/lib/python3.10/"
                "site-packages/unsloth/kernels/large_head_attention.py",
        "pull_request": "https://github.com/unslothai/unsloth/pull/13869",
        "lines": {
            "supported_head_dims_512_only": 27,
            "router_head_dim_gt_256_gate": 886,
            "router_rejects_dropout": 887,
            "router_rejects_autocast_dtype_mismatch": 899,
            "router_rejects_math_sdpa_priority": 900,
            "compiled_or_capturing_masked_call_falls_to_sdpa": 912,
            "engaged_calls_counter": 937,
            "fallback_counter": 924,
            "self_check_vs_sdpa_math": 770,
            "install_wraps_ALL_ATTENTION_FUNCTIONS_sdpa": 1022,
            "install_bails_on_forced_float32": 985,
            "enabled_env_UNSLOTH_HD512_ATTENTION": 716,
        },
    },
    "transformers_gemma4_unified": {
        "path": "/home/roma/miniconda3/envs/gemma4_env/lib/python3.10/"
                "site-packages/transformers/models/gemma4_unified/"
                "modeling_gemma4_unified.py",
        "lines": {
            "attention_dispatch_get_interface": 456,
            "global_layer_head_dim_512_k_eq_v": 372,
            "scaling_is_one": 378,
            "attention_dropout_zero_in_train": 466,
            "supports_flex_attn_true": 570,
            "text_forward_builds_full_and_sliding_masks": 664,
            "sliding_mask_is_and_sliding_or_blockwise_multimodal_only": 1239,
        },
    },
    "unsloth_vision_loader": {
        "path": "/tmp/soomac-training-accel-20261101/venv/lib/python3.10/"
                "site-packages/unsloth/models/vision.py",
        "lines": {
            "do_forced_float32_only_if_UNSLOTH_FORCE_FLOAT32": 3669,
            "explicit_flex_from_attn_implementation": 3760,
            "patch_large_head_attention_called_at_load": 4943,
        },
    },
}

# The 8 full_attention layer indices for this exact config (text_config.layer_types).
# head_dim 512, num_global_key_value_heads 1. The other 40 are sliding, head_dim 256.
GLOBAL_LAYER_INDICES = (5, 11, 17, 23, 29, 35, 41, 47)
HD512 = 512
HD256 = 256
HD512_ENV = "UNSLOTH_HD512_ATTENTION"


def _load_benchmark():
    """Import the sibling benchmark module by path. Its top level is stdlib only, so
    this stays CPU-safe; torch/Unsloth are imported lazily inside its GPU helpers."""
    spec = importlib.util.spec_from_file_location("training_acceleration_benchmark",
                                                  BENCHMARK_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rss_bytes():
    """Peak-so-far process RSS in bytes (ru_maxrss is KiB on Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def padding_cpu(args, benchmark):
    """CPU-only gate: remove alignment padding, never real tokens/labels/positions."""
    import torch
    from transformers import AutoProcessor, AutoConfig, DataCollatorForSeq2Seq
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU gate must run before any CUDA initialization")
    destination = benchmark.output_dir(args.output)
    payload = benchmark.validate_payload(args.encoded, args.agent)
    processor = AutoProcessor.from_pretrained(benchmark.RELEASE / "assets/model-processor",
                                              local_files_only=True)
    row = {key: payload["rows"][0][key] for key in ("input_ids", "labels")}
    def collate(multiple):
        return DataCollatorForSeq2Seq(tokenizer=processor.tokenizer, padding=True,
            label_pad_token_id=-100, pad_to_multiple_of=multiple, return_tensors="pt")([copy.deepcopy(row)])
    old, new = collate(8), collate(None)
    valid = old["attention_mask"][0].bool()
    assert int(valid.sum()) == new["input_ids"].shape[1] == 8430
    assert all(torch.equal(old[k][:, valid], new[k]) for k in ("input_ids", "labels"))
    assert bool((old["labels"][:, ~valid] == -100).all())
    old_positions = torch.arange(old["input_ids"].shape[1])[None]
    new_positions = old_positions[:, valid].clone()
    config = AutoConfig.from_pretrained(benchmark.BASE, local_files_only=True).text_config
    config._attn_implementation = "sdpa"
    def masks(batch, positions):
        kwargs = dict(config=config, inputs_embeds=torch.zeros(1, batch["input_ids"].shape[1], 1,
                                                              dtype=torch.bfloat16),
                      attention_mask=batch["attention_mask"], past_key_values=None,
                      position_ids=positions)
        return {"global": create_causal_mask(**kwargs),
                "sliding": create_sliding_window_causal_mask(**kwargs)}
    old_masks, new_masks = masks(old, old_positions), masks(new, new_positions)
    # Execute ONLY these exact installed CPU mask predicates, not the Unsloth
    # package import (which initializes CUDA). No local reimplementation of gates.
    namespace = {"torch": torch}
    sites = Path(REVIEWED_SOURCES["unsloth_gemma4_banded_attention"]["path"]).parents[2]
    functions = [(sites / "unsloth/kernels/large_head_attention.py", {"intervals_from_mask"}),
                 (sites / "unsloth_zoo/temporary_patches/gemma4_banded_attention.py",
                  {"_packed_documents", "_mask_is_plain_band"})]
    source_hashes = {}
    for path, names in functions:
        tree = ast.parse(path.read_text())
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        if {node.name for node in nodes} != names:
            raise RuntimeError("installed mask predicate source changed")
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
        source_hashes[str(path)] = _sha256(path)
    seq = new["input_ids"].shape[1]
    global_ok = (new_masks["global"] is None or
                 namespace["intervals_from_mask"](new_masks["global"], 1, seq) is not None)
    sliding_ok = namespace["_mask_is_plain_band"](new_masks["sliding"], seq, 1024)
    same_semantics = {}
    for name in old_masks:
        old_mask = old_masks[name]
        new_mask = new_masks[name]
        old_valid = old_mask[:, :, valid][:, :, :, valid]
        if new_mask is None:
            new_mask = torch.ones(seq, seq, dtype=torch.bool).tril()[None, None]
        same_semantics[name] = bool(torch.equal(old_valid, new_mask))
    result = {"status": "CPU_PADDING_GATE_PASS" if global_ok and sliding_ok and all(same_semantics.values())
              else "CPU_PADDING_GATE_FAILED_STOP", "production_changed": False,
              "encoded_sha256": benchmark.sha(args.encoded), "padding_side": processor.tokenizer.padding_side,
              "original_length": old["input_ids"].shape[1], "real_tokens": seq,
              "removed_padding_positions": torch.nonzero(~valid).flatten().tolist(),
              "tokens_and_labels_exactly_preserved": True, "pad_labels_all_ignored": True,
              "position_ids_first_last": [int(new_positions[0, 0]), int(new_positions[0, -1])],
              "real_token_mask_semantics_equal": same_semantics,
              "global_mask_none": new_masks["global"] is None, "global_gate": global_ok,
              "sliding_gate": sliding_ok, "predicate_source_hashes": source_hashes,
              "cuda_initialized": torch.cuda.is_initialized()}
    benchmark.write_new(destination / "PADDING_CPU.json", result)
    return result


def padding_gpu(args, benchmark):
    """Last bounded padding-only diagnostic; no optimizer step on failed numerics."""
    destination = benchmark.output_dir(args.output)
    payload = benchmark.validate_payload(args.encoded, args.agent)
    gate = json.loads(args.padding_receipt.read_text())
    if (gate.get("status") != "CPU_PADDING_GATE_PASS"
            or gate.get("encoded_sha256") != benchmark.sha(args.encoded)):
        raise ValueError("matching CPU padding gate required")
    result = {"status": "FAILED", "production_changed": False,
              "encoded_sha256": benchmark.sha(args.encoded), "cpu_gate": gate,
              "tolerances": {"loss_atol_rtol": 1e-3, "tensor_atol": 1e-4, "tensor_rtol": 1e-3},
              "timing": None, "optimizer_updates": 0}
    fd = budget = None
    try:
        fd = benchmark.acquire_gpu_lock()
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                          TOKENIZERS_PARALLELISM="false", UNSLOTH_RETURN_LOGITS="1",
                          UNSLOTH_HD512_ATTENTION="1", UNSLOTH_GEMMA4_FLASH_SLIDING="1")
        if os.environ.get("UNSLOTH_COMPILE_DISABLE") != "1":
            raise ValueError("fresh official compile-OFF process required for controlled router toggles")
        import unsloth  # noqa: F401
        import torch
        from transformers import AutoProcessor, DataCollatorForSeq2Seq, set_seed
        from unsloth_zoo.temporary_patches import gemma4_flash_sliding as sliding
        from unsloth.kernels.large_head_attention import LARGE_HEAD_ENGAGED
        sys.path.insert(0, str(benchmark.RELEASE / "common"))
        from budget import Budget
        from loss import scored_suffix
        set_seed(42)
        budget = Budget(benchmark.PROCESS_BUDGET, benchmark.RESERVE)
        budget.check(force=True)
        p = AutoProcessor.from_pretrained(benchmark.RELEASE / "assets/model-processor", local_files_only=True)
        row = {k: payload["rows"][0][k] for k in ("input_ids", "labels")}
        def collate(multiple):
            batch = DataCollatorForSeq2Seq(tokenizer=p.tokenizer, padding=True,
                label_pad_token_id=-100, pad_to_multiple_of=multiple, return_tensors="pt")([copy.deepcopy(row)])
            return {k: v.to("cuda") for k, v in batch.items()}
        padded, plain = collate(8), collate(None)
        valid = padded["attention_mask"][0].bool()
        padded["position_ids"] = torch.arange(padded["input_ids"].shape[1], device="cuda")[None]
        plain["position_ids"] = padded["position_ids"][:, valid].clone()
        for key in ("input_ids", "labels", "position_ids"):
            if not torch.equal(padded[key][:, valid], plain[key]):
                raise RuntimeError(f"real-token {key} changed")
        model, optimizer, parameters, names = benchmark.model_and_optimizer("unsloth", args.agent, torch)
        budget.check(force=True)
        result.update(library_versions=_library_versions(), resolved_config=_resolved_config(model),
                      trainable_count=sum(p.numel() for p in parameters),
                      position_ids_first_last=[int(plain["position_ids"][0,0]), int(plain["position_ids"][0,-1])])

        def toggle(enabled):
            _set_hd512(enabled)
            os.environ["UNSLOTH_GEMMA4_FLASH_SLIDING"] = "1" if enabled else "0"
            sliding._enabled.cache_clear()
            if sliding._enabled() != enabled:
                raise RuntimeError("sliding router toggle failed")

        def diagnostic(batch, enabled):
            budget.check(force=True)
            toggle(enabled)
            model.train()
            # Correctness-only RNG control: suppress LoRA Dropout execution,
            # NOT its p=.05 setting. Restore training flags before profiling/perf.
            dropouts = [(m, m.training) for m in model.modules() if isinstance(m, torch.nn.Dropout)]
            if not any(abs(m.p - .05) < 1e-9 for m, state in dropouts):
                raise RuntimeError("LoRA dropout .05 missing")
            for module, state in dropouts:
                module.training = False
            captures, handles = {}, []
            for module in model.modules():
                index = getattr(module, "layer_idx", None)
                if hasattr(module, "head_dim") and index in (0, 5):
                    def save_attention(m, unused, output, index=index):
                        tensor = output[0] if isinstance(output, tuple) else output
                        captures.setdefault(str(index), tensor.detach().clone())
                    handles.append(module.register_forward_hook(save_attention))
            model.zero_grad(set_to_none=True)
            set_seed(42)
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(**scored_suffix(batch), use_cache=False)
                loss = float(output.loss.detach().float().item())
                logits = output.logits.detach().clone()
                output.loss.backward()
                grads = _grad_snapshot(torch, parameters)
                finite = bool(torch.isfinite(grads).all()) and bool(torch.count_nonzero(grads))
                if not finite:
                    raise RuntimeError("nonfinite or zero LoRA gradient")
                budget.check(force=True)
                return {"loss": loss, "logits": logits, "grad": grads, "attention": captures}
            finally:
                for handle in handles:
                    handle.remove()
                for module, training in dropouts:
                    module.training = training
                model.zero_grad(set_to_none=True)

        if args.command == "padding-diagnose":
            return _padding_diagnose(torch, model, parameters, names, padded, plain, valid,
                                     scored_suffix, budget, toggle, destination, result, benchmark)
        if args.command == "first-sdpa":
            return _first_sdpa(torch, model, padded, plain, valid, scored_suffix,
                               budget, toggle, destination, result, benchmark)

        def compare_values(reference, candidate, trim_reference):
            diffs = {}
            for label in ("logits", "grad"):
                a, b = reference[label], candidate[label]
                diffs[label] = {**_tensor_diff(b, a),
                    "allclose": bool(torch.allclose(a.float(), b.float(), atol=1e-4, rtol=1e-3))}
            for index in ("0", "5"):
                a, b = reference["attention"][index], candidate["attention"][index]
                if trim_reference:
                    a = a[:, valid]
                diffs["attention_layer" + index] = {**_tensor_diff(b, a),
                    "allclose": bool(torch.allclose(a.float(), b.float(), atol=1e-4, rtol=1e-3))}
            import math
            loss_ok = math.isclose(reference["loss"], candidate["loss"], abs_tol=1e-3, rel_tol=1e-3)
            return {"reference_loss": reference["loss"], "candidate_loss": candidate["loss"],
                    "loss_delta": abs(reference["loss"] - candidate["loss"]),
                    "loss_pass": loss_ok, "tensor_differences": diffs,
                    "passed": loss_ok and all(d["allclose"] for d in diffs.values())}

        padded_ref = diagnostic(padded, False)
        print(json.dumps({"phase": "padded_sdpa_diagnostic_complete", "loss": padded_ref["loss"]}), flush=True)
        plain_ref = diagnostic(plain, False)
        print(json.dumps({"phase": "unpadded_sdpa_diagnostic_complete", "loss": plain_ref["loss"]}), flush=True)
        result["padding_only_equivalence"] = compare_values(padded_ref, plain_ref, True)
        del padded_ref
        result["status"] = ("PADDING_ONLY_PASS_FAST_CHECK_PENDING"
                            if result["padding_only_equivalence"]["passed"]
                            else "PADDING_CORRECTNESS_FAILED_STOP_NO_TIMING")
        benchmark.write_new(destination / "PADDING_EQUIVALENCE.json", result)
        if not result["padding_only_equivalence"]["passed"]:
            result["status"] = "PADDING_CORRECTNESS_FAILED_STOP_NO_TIMING"
            return result
        fast = diagnostic(plain, True)
        print(json.dumps({"phase": "unpadded_fast_diagnostic_complete", "loss": fast["loss"]}), flush=True)
        result["fast_kernel_equivalence"] = compare_values(plain_ref, fast, False)
        del plain_ref, fast
        if not result["fast_kernel_equivalence"]["passed"]:
            result["status"] = "FAST_KERNEL_CORRECTNESS_FAILED_STOP_NO_TIMING"
            return result
        toggle(True)
        result["native_train_profile"] = _native_train_profile(
            torch, model, plain, scored_suffix, LARGE_HEAD_ENGAGED,
            destination / "native_attention_trace.json")
        result["status"] = "PADDING_AND_FAST_CORRECTNESS_PASS_TIMING_PENDING"
        return result
    except BaseException as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if budget is not None:
            result.update(budget.summary(), peak_allocated_bytes=budget.torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=budget.torch.cuda.max_memory_reserved())
            budget.close()
        result["cpu_peak_rss_bytes"] = _rss_bytes()
        filename = {"padding-diagnose": "PADDING_DIAGNOSIS.json",
                    "first-sdpa": "FIRST_SDPA.json"}.get(args.command, "PADDING_GPU.json")
        benchmark.write_new(destination / filename, result)
        if fd is not None:
            os.close(fd)


def _sdpa_bias_rows(torch, mask, q_indices, q_length, k_length, heads, is_causal):
    """CPU reference bias using ACTUAL low-level arguments, never None=>causal.

    Query indices refer to the original full sequence (not the sampled row order).
    Supports broadcast bool keep masks and additive float masks without binarizing
    finite bias values. This diagnostic only handles batch size one.
    """
    bias = torch.zeros((heads, len(q_indices), k_length), dtype=torch.float32)
    if mask is not None:
        expanded = torch.broadcast_to(mask, (1, heads, q_length, k_length))
        selected = expanded[0].index_select(1, torch.tensor(q_indices))
        if selected.dtype == torch.bool:
            bias.masked_fill_(~selected, float("-inf"))
        elif selected.is_floating_point():
            bias += selected.float()
        else:
            raise TypeError("unsupported SDPA mask dtype")
    if is_causal:
        keep = torch.arange(k_length)[None, :] <= torch.tensor(q_indices)[:, None]
        bias.masked_fill_(~keep[None], float("-inf"))
    return bias


def _probe_diff(torch, reference, candidate):
    """Small probe tensors only; FP64 reductions, including norm and L2."""
    x, y = reference.detach().cpu().double(), candidate.detach().cpu().double()
    if x.shape != y.shape:
        raise ValueError("probe shape mismatch")
    an, bn, dn = (float(t.norm()) for t in (x, y, y-x))
    return {"reference_norm": an, "candidate_norm": bn,
            "relative_l2_error": dn / max(an, 1e-12),
            "max_absolute": float((y-x).abs().max()),
            "cosine": float((x*y).sum()) / max(an*bn, 1e-30),
            "finite": bool(torch.isfinite(x).all() and torch.isfinite(y).all()),
            "bitwise_equal": bool(torch.equal(reference, candidate)), "reduction": "FP64"}


def _first_sdpa_stop_guard(model):
    """Fail closed if an alternative attention path bypasses F.sdpa interception."""
    first = [m for m in model.modules()
             if getattr(m, "layer_idx", None) == 0 and hasattr(m, "head_dim")]
    if len(first) != 1:
        raise RuntimeError("cannot locate unique layer0 attention stop boundary")

    def escaped(unused_module, unused_inputs, unused_output):
        raise RuntimeError("layer0 attention bypassed first SDPA interception; stopped")

    return first[0].register_forward_hook(escaped)


def _first_sdpa(torch, model, padded, plain, valid, scored_suffix,
                budget, toggle, destination, result, benchmark):
    """Abort immediately after layer0's first REAL F.sdpa; no later layer/backward.

    Tensor captures stay in process memory. Only scalar/shape/hash metadata and
    CUDA kernel names are serialized. Two prefix-only forwards, zero updates.
    """
    import torch.nn.functional as functional
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    toggle(False)
    model.train()
    modules = [(m, m.p, m.training) for n, m in model.named_modules()
               if "lora_dropout" in n and isinstance(m, torch.nn.Dropout)]
    if not modules or any(abs(p-.05) > 1e-9 for m, p, training in modules):
        raise RuntimeError("expected production LoRA dropout .05")
    original_function = functional.scaled_dot_product_attention
    original_router = ALL_ATTENTION_FUNCTIONS["sdpa"]
    rng_cpu, rng_cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
    buffers = {n: b.detach().cpu().clone() for n, b in model.named_buffers()}
    versions = {n: p._version for n, p in model.named_parameters()}
    current = {}

    class FirstCallComplete(BaseException):
        pass  # Do not let a router's `except Exception` silently retry the model.

    def sha(tensor):
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        return hashlib.sha256(raw.numpy().tobytes()).hexdigest()

    def tensor_meta(tensor):
        return None if tensor is None else {
            "shape": list(tensor.shape), "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype), "device": str(tensor.device),
            "contiguous": tensor.is_contiguous(), "sha256": sha(tensor)}

    def router(module, *args, **kwargs):
        current["layer"] = getattr(module, "layer_idx", None)
        current["attention_type"] = getattr(module, "attention_type", None)
        return original_router(module, *args, **kwargs)

    def intercept(query, key, value, attn_mask=None, dropout_p=0.0,
                  is_causal=False, *, scale=None, enable_gqa=False):
        if current.get("layer") != 0 or "capture" in current:
            raise RuntimeError("first low-level SDPA must be exactly decoder layer0")
        if dropout_p != 0:
            raise RuntimeError("reference requires actual attention dropout zero")
        metadata = {"layer": current["layer"], "attention_type": current["attention_type"],
                    "query": tensor_meta(query), "key": tensor_meta(key),
                    "value": tensor_meta(value), "mask": tensor_meta(attn_mask),
                    "is_causal": bool(is_causal), "scale": scale,
                    "effective_scale": float(scale if scale is not None else query.shape[-1]**-.5),
                    "enable_gqa": bool(enable_gqa), "attention_dropout_p": dropout_p,
                    "autocast_enabled": torch.is_autocast_enabled("cuda"),
                    "autocast_dtype": str(torch.get_autocast_dtype("cuda"))}
        capture = {name: tensor.detach().cpu().clone() for name, tensor in
                   (("q", query), ("k", key), ("v", value))}
        capture["mask"] = None if attn_mask is None else attn_mask.detach().cpu().clone()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profile:
            output = original_function(query, key, value, attn_mask=attn_mask,
                dropout_p=dropout_p, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
            torch.cuda.synchronize()
        metadata["cuda_kernels"] = dict(Counter(e.name for e in profile.events()
            if getattr(e.device_type, "name", "") == "CUDA"))
        metadata["sdpa_operators"] = sorted({e.name for e in profile.events()
                                             if e.name.startswith("aten::") and "attention" in e.name})
        capture.update(out=output.detach().cpu().clone(), metadata=metadata)
        current["capture"] = capture
        raise FirstCallComplete()

    captures = {}
    stop_guard = _first_sdpa_stop_guard(model)
    try:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = router
        functional.scaled_dot_product_attention = intercept
        for m, p, training in modules:
            m.p, m.training = 0.0, True
        for label, batch in (("A_padding2", padded), ("B_padding0", plain)):
            with torch.no_grad():
                for n, b in model.named_buffers():
                    b.copy_(buffers[n].to(b.device))
            torch.set_rng_state(rng_cpu)
            torch.cuda.set_rng_state(rng_cuda)
            current.clear()
            # Keep grad mode/train state as in the previous diagnostics. The sentinel
            # aborts inside layer0; neither loss nor backward is ever reached.
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    model(**scored_suffix(batch), use_cache=False)
            except FirstCallComplete:
                pass
            else:
                raise RuntimeError("model unexpectedly passed first SDPA")
            captures[label] = current.pop("capture")
            captures[label]["metadata"].update(
                position_ids_first_last=[int(batch["position_ids"][0, 0]),
                                         int(batch["position_ids"][0, -1])],
                input_length=int(batch["input_ids"].shape[1]),
                cpu_rng_start_sha=sha(rng_cpu), cuda_rng_start_sha=sha(rng_cuda))
            benchmark.write_new(destination / (label + ".json"), captures[label]["metadata"])
            print(json.dumps({"phase": "first_sdpa_captured", "case": label,
                              "is_causal": captures[label]["metadata"]["is_causal"]}), flush=True)
            budget.check(force=True)
        if any(p._version != versions[n] for n, p in model.named_parameters()):
            raise RuntimeError("parameter mutated before first SDPA")
    finally:
        stop_guard.remove()
        functional.scaled_dot_product_attention = original_function
        ALL_ATTENTION_FUNCTIONS["sdpa"] = original_router
        for m, p, training in modules:
            m.p, m.training = p, training

    a, b = captures["A_padding2"], captures["B_padding0"]
    valid_cpu = valid.cpu()
    n = int(valid_cpu.sum())
    indices = sorted({0, 1, 31, 1023, 1024, n//2, n-1})
    valid_indices = torch.nonzero(valid_cpu).flatten()
    a_indices = valid_indices[indices].tolist()
    heads = b["q"].shape[1]
    def effective_kv(capture, key):
        tensor = capture[key]
        groups = heads // tensor.shape[1]
        if groups != 1:
            if not capture["metadata"]["enable_gqa"] or heads % tensor.shape[1]:
                raise RuntimeError("unexpected runtime KV head mapping")
            tensor = tensor.repeat_interleave(groups, dim=1)
        return tensor

    same_qkv = {"q": bool(torch.equal(a["q"][:, :, valid_cpu], b["q"]))}
    same_qkv.update({key: bool(torch.equal(effective_kv(a, key)[:, :, valid_cpu],
                                           effective_kv(b, key))) for key in ("k", "v")})
    # Check ALL valid query/key mask entries in bounded CPU chunks, not just samples.
    mismatch, max_bias_delta = 0, 0.0
    for start in range(0, n, 256):
        rel = list(range(start, min(start+256, n)))
        bias_a = _sdpa_bias_rows(torch, a["mask"], valid_indices[rel].tolist(),
            a["q"].shape[2], a["k"].shape[2], 1 if a["mask"] is None else a["mask"].shape[-3],
            a["metadata"]["is_causal"])[..., valid_cpu]
        bias_b = _sdpa_bias_rows(torch, b["mask"], rel, n, n,
            1 if b["mask"] is None else b["mask"].shape[-3], b["metadata"]["is_causal"])
        mismatch += int((bias_a != bias_b).sum())
        finite = torch.isfinite(bias_a) & torch.isfinite(bias_b)
        if finite.any():
            max_bias_delta = max(max_bias_delta, float((bias_a-bias_b)[finite].abs().max()))

    def reference(capture, rows, canonical_keys=False):
        # FP32 CPU reference: no TF32/CUDA softmax/fused kernels. Full key range,
        # seven sampled query rows, all16 heads. Not a performance measurement.
        q, k, v = (capture[key][0].float() for key in ("q", "k", "v"))
        groups = heads // k.shape[0]
        if groups != 1:
            if not capture["metadata"]["enable_gqa"] or heads % k.shape[0]:
                raise RuntimeError("unexpected head mapping")
            k, v = k.repeat_interleave(groups, 0), v.repeat_interleave(groups, 0)
        bias = _sdpa_bias_rows(torch, capture["mask"], rows, q.shape[1], k.shape[1],
                               heads, capture["metadata"]["is_causal"])
        if canonical_keys:
            k, v, bias = k[:, valid_cpu], v[:, valid_cpu], bias[..., valid_cpu]
        scores = (q.index_select(1, torch.tensor(rows)) @ k.transpose(-1, -2)) * capture["metadata"]["effective_scale"]
        scores += bias
        probabilities = torch.softmax(scores, dim=-1)
        if not torch.isfinite(probabilities).all():
            raise RuntimeError("reference has empty or nonfinite valid query")
        return probabilities @ v, bias

    ar, abias = reference(a, a_indices)
    br, bbias = reference(b, indices)
    canonical_a, unused = reference(a, a_indices, canonical_keys=True)
    ao = a["out"][0].index_select(1, torch.tensor(a_indices)).float()
    bo = b["out"][0].index_select(1, torch.tensor(indices)).float()
    result.update(status="FIRST_SDPA_CAUSE_PROBE_COMPLETE_NO_TRAINING",
        cases={key: value["metadata"] for key, value in captures.items()},
        diagnostic_lora_dropout=0.0, production_lora_dropout_restored=.05,
        full_model_completed=0, backward_calls=0, first_sdpa_calls=2,
        valid_qkv_bitwise_equal=same_qkv,
        all_valid_bias_mismatch_count=mismatch, all_valid_max_finite_bias_delta=max_bias_delta,
        sampled_relative_query_indices=indices, sampled_A_array_indices=a_indices,
        reference="CPU FP32 matmul/softmax, full keys, sampled queries, all heads",
        fp32_reference_A_vs_B=_probe_diff(torch, ar, br),
        fp32_same_valid_key_shape_A_vs_B=_probe_diff(torch, canonical_a, br),
        fp32_A_full_keys_vs_valid_keys=_probe_diff(torch, ar, canonical_a),
        actual_A_vs_fp32_reference=_probe_diff(torch, ar, ao),
        actual_B_vs_fp32_reference=_probe_diff(torch, br, bo),
        actual_A_vs_B=_probe_diff(torch, ao, bo),
        sampled_nonblocked_counts_A=(abias[0] > -1e20).sum(-1).tolist(),
        sampled_nonblocked_counts_B=(bbias[0] > -1e20).sum(-1).tolist(),
        sampled_nonblocked_padding_key_count_A=int((abias[..., ~valid_cpu] > -1e20).sum()),
        nonblocked_count_note="diagnostic counts use bias>-1e20; FP32 math preserves exact finite biases",
        gpu_kernel_speedup=None, production_transition_authorized=False)
    return result


def _padding_diagnose(torch, model, parameters, names, padded, plain, valid,
                      scored_suffix, budget, toggle, destination, result, benchmark):
    """Five single-row backward diagnostics, no kernel switch or optimizer update."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    import re
    toggle(False)
    model.train()
    dropout_modules = [(n, m, m.p, m.training) for n, m in model.named_modules()
                       if "lora_dropout" in n and isinstance(m, torch.nn.Dropout)]
    if not dropout_modules or any(abs(p - .05) > 1e-9 for n, m, p, state in dropout_modules):
        raise RuntimeError("expected original LoRA dropout .05")
    initial_weights = [p.detach().cpu().clone() for p in parameters]
    initial_buffers = {n: b.detach().cpu().clone() for n, b in model.named_buffers()}
    frozen_versions = {n: p._version for n, p in model.named_parameters() if not p.requires_grad}
    cpu_rng, gpu_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    original_sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]

    def tensor_sha(t):
        return hashlib.sha256(t.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()

    def digest(items):
        h = hashlib.sha256()
        for name, t in items:
            h.update(name.encode())
            h.update(bytes.fromhex(tensor_sha(t)))
        return h.hexdigest()

    def stats(a, b):
        a, b = a.float().reshape(-1), b.float().reshape(-1)
        # FP32 vector_norm/cosine on 65M CPU elements gave cosine >1 even
        # for identical tensors. Accumulate bounded blocks in FP64 instead.
        sums = torch.zeros(5, dtype=torch.float64)
        maximum = reference_maximum = 0.0
        for start in range(0, a.numel(), 1 << 20):
            x, y = a[start:start+(1 << 20)].double(), b[start:start+(1 << 20)].double()
            delta = y-x
            sums += torch.stack((x.square().sum(), y.square().sum(), (x*y).sum(),
                                 delta.square().sum(), delta.abs().sum()))
            maximum = max(maximum, float(delta.abs().max()))
            reference_maximum = max(reference_maximum, float(x.abs().max()))
        an, bn, dn = float(sums[0].sqrt()), float(sums[1].sqrt()), float(sums[3].sqrt())
        cosine = (1.0 if an == bn == 0 else 0.0 if an == 0 or bn == 0
                  else max(-1.0, min(1.0, float(sums[2])/(an*bn))))
        return {"reference_norm": an, "candidate_norm": bn, "cosine": cosine,
                "relative_l2_error": dn/max(an, 1e-12), "max_absolute": maximum,
                "max_relative": maximum/max(reference_maximum, 1e-12),
                "mean_absolute": float(sums[4])/a.numel(), "reduction": "FP64_chunked",
                "allclose": bool(torch.allclose(a, b, atol=1e-4, rtol=1e-3))}

    def run_case(label, batch, probability):
        model.train()
        with torch.no_grad():
            for parameter, saved in zip(parameters, initial_weights):
                parameter.copy_(saved.to(parameter.device))
            for name, buffer in model.named_buffers():
                buffer.copy_(initial_buffers[name].to(buffer.device))
        for name, module, p, state in dropout_modules:
            module.p = probability
            module.training = True
        model.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(gpu_rng)
        before = {"weights_sha": digest(zip(names, parameters)),
                  "buffers_sha": digest(model.named_buffers()),
                  "cpu_rng_sha": tensor_sha(torch.get_rng_state()),
                  "cuda_rng_sha": tensor_sha(torch.cuda.get_rng_state()),
                  "dropout_p_values": sorted({m.p for n, m, p, state in dropout_modules}),
                  "dropout_training_count": sum(m.training for n, m, p, state in dropout_modules)}
        captures = {}

        def recorder(module, query, key, value, attention_mask, *args, **kwargs):
            index = getattr(module, "layer_idx", None)
            if index in (0, 5) and str(index) not in captures:
                captures[str(index)] = {
                    "q": query.detach().cpu(), "k": key.detach().cpu(), "v": value.detach().cpu(),
                    "mask": None if attention_mask is None else attention_mask.detach().cpu(),
                    "is_causal": kwargs.get("is_causal", getattr(module, "is_causal", True))}
            output = original_sdpa(module, query, key, value, attention_mask, *args, **kwargs)
            if index in (0, 5) and "out" not in captures[str(index)]:
                captures[str(index)]["out"] = output[0].detach().cpu()
            return output

        ALL_ATTENTION_FUNCTIONS["sdpa"] = recorder
        selected = scored_suffix(batch)
        metadata = {"target_count": int(selected["logits_to_keep"].numel()),
                    "target_indices": selected["logits_to_keep"].detach().cpu().tolist(),
                    "target_ids_sha": tensor_sha(selected["shift_labels"]),
                    "target_position_ids": batch["position_ids"].index_select(
                        1, selected["logits_to_keep"]).detach().cpu().tolist()}
        budget.check(force=True)
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(**selected, use_cache=False)
            logits, loss = out.logits.detach().float().cpu(), float(out.loss.detach().float())
            out.loss.backward()
            torch.cuda.synchronize()
            gradients = [torch.zeros_like(p, device="cpu", dtype=torch.float32) if p.grad is None
                         else p.grad.detach().float().cpu().clone() for p in parameters]
            if (not all(bool(torch.isfinite(g).all()) for g in gradients)
                    or not any(bool(torch.count_nonzero(g)) for g in gradients)):
                raise RuntimeError("nonfinite or zero gradient")
            after = {"weights_sha": digest(zip(names, parameters)),
                     "dropout_p_values": sorted({m.p for n, m, p, state in dropout_modules}),
                     "frozen_parameter_versions_preserved": frozen_versions == {
                         n: p._version for n, p in model.named_parameters() if not p.requires_grad}}
            if before["weights_sha"] != after["weights_sha"] or not after["frozen_parameter_versions_preserved"]:
                raise RuntimeError("model weights changed in no-optimizer diagnostic")
            budget.check(force=True)
            meta = {"case": label, "loss": loss, "before": before, "after": after,
                    "selected_targets": metadata, "dropout": probability}
            benchmark.write_new(destination / (label + ".json"), meta)
            print(json.dumps(meta), flush=True)
            return {**meta, "logits": logits, "gradients": gradients, "captures": captures}
        finally:
            ALL_ATTENTION_FUNCTIONS["sdpa"] = original_sdpa
            model.zero_grad(set_to_none=True)

    def compare(a, b, padding_change=False):
        grad = stats(torch.cat([g.reshape(-1) for g in a["gradients"]]),
                     torch.cat([g.reshape(-1) for g in b["gradients"]]))
        layer_grad = {}
        for index in range(48):
            indices = [i for i, name in enumerate(names) if f".layers.{index}." in name]
            layer_grad[str(index)] = stats(torch.cat([a["gradients"][i].reshape(-1) for i in indices]),
                                          torch.cat([b["gradients"][i].reshape(-1) for i in indices]))
        differences = {}
        valid_cpu = valid.cpu()
        for index in ("0", "5"):
            ac, bc = a["captures"][index], b["captures"][index]
            d = {}
            for key in ("q", "k", "v"):
                reference = ac[key][:, :, valid_cpu] if padding_change else ac[key]
                d[key] = stats(reference, bc[key])
            reference_out = ac["out"][:, valid_cpu] if padding_change else ac["out"]
            d["sdpa_output"] = stats(reference_out, bc["out"])
            am, bm = ac["mask"], bc["mask"]
            if padding_change:
                am = am[:, :, valid_cpu][:, :, :, valid_cpu]
            if bm is None:
                seq = bc["q"].shape[2]
                bm = torch.ones(seq, seq, dtype=torch.bool).tril()[None, None]
            d["mask_equal"] = bool(torch.equal(am, bm)) if am is not None else ac["mask"] is bc["mask"]
            differences[index] = d
        return {"loss_delta": abs(a["loss"]-b["loss"]), "gradient": grad,
                "logits": stats(a["logits"], b["logits"]), "layer_gradients": layer_grad,
                "attention_inputs": differences,
                "initial_state_and_rng_equal": a["before"] == b["before"],
                "target_ids_equal": a["selected_targets"]["target_ids_sha"] == b["selected_targets"]["target_ids_sha"],
                "scored_position_ids_equal": a["selected_targets"]["target_position_ids"] == b["selected_targets"]["target_position_ids"]}
    try:
        a1 = run_case("A_dropout05_first", padded, .05)
        a2 = run_case("A_dropout05_repeat", padded, .05)
        result["A_repeat_dropout05"] = compare(a1, a2)
        del a1, a2
        a0 = run_case("A_dropout0_first", padded, 0.0)
        a0_repeat = run_case("A_dropout0_repeat", padded, 0.0)
        result["A_repeat_dropout0"] = compare(a0, a0_repeat)
        del a0_repeat
        b0 = run_case("B_dropout0", plain, 0.0)
        result["AB_dropout0"] = compare(a0, b0, padding_change=True)
        result["status"] = "PADDING_FAILURE_DIAGNOSIS_COMPLETE_NO_OPTIMIZER"
        return result
    finally:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = original_sdpa
        for n, m, p, state in dropout_modules:
            m.p, m.training = p, state
        result["restored_dropout_p_values"] = sorted({m.p for n, m, p, state in dropout_modules})


def _sha256(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _source_manifest():
    """Record version/source/hash so the reviewed line numbers are falsifiable."""
    manifest = {}
    for name, entry in REVIEWED_SOURCES.items():
        path = Path(entry["path"])
        record = {"path": str(path), "lines": entry["lines"], "exists": path.exists()}
        if "pull_request" in entry:
            record["pull_request"] = entry["pull_request"]
        record["sha256"] = _sha256(path) if path.exists() else None
        manifest[name] = record
    return manifest


def _library_versions():
    """Imported lazily; importing torch/unsloth pulls CUDA, so only call on GPU paths."""
    versions = {}
    try:
        import torch
        versions["torch"] = torch.__version__
        versions["cuda"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            versions["device"] = torch.cuda.get_device_name(0)
            versions["sm"] = f"{major}.{minor}"
    except Exception as error:  # pragma: no cover - GPU path only
        versions["torch_error"] = f"{type(error).__name__}: {error}"
    for module_name in ("transformers", "unsloth", "unsloth_zoo", "triton", "peft"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception as error:  # pragma: no cover - GPU path only
            versions[module_name] = f"import_error: {type(error).__name__}"
    return versions


# --------------------------------------------------------------------------- prepare


def prepare(args, benchmark):
    """CPU only: reuse the sealed encoder, then stamp an environment manifest beside
    the encoded artifact. No model, no CUDA."""
    before = _rss_bytes()
    receipt = benchmark.prepare(args.agent, args.output)
    manifest = {
        "schema_version": 1,
        "mode": "CPU_PREPARED_NO_GPU",
        "agent": args.agent,
        "encoded_sha256": receipt["encoded_sha256"],
        "reviewed_sources": _source_manifest(),
        "cpu_rss_bytes_before": before,
        "cpu_rss_bytes_after": _rss_bytes(),
        "note": "versions recorded on GPU paths only; importing torch here would load CUDA",
    }
    benchmark.write_new(args.output / "ABLATION_PREPARE.json", manifest)
    return {"status": "CPU_PREPARED_NO_GPU", "output": str(args.output),
            "encoded_sha256": receipt["encoded_sha256"]}


# ------------------------------------------------------------------- GPU shared setup


def _install_attention_recorder(torch, records):
    """Wrap ALL_ATTENTION_FUNCTIONS['sdpa'] (already the HD512 router under Unsloth,
    or the plain sdpa interface under baseline) with a recorder. For every call it
    logs the layer's identity and whether the HD512 Triton op engaged this call,
    measured by the global LARGE_HEAD_ENGAGED counter delta. Returns a restore()."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    try:
        from unsloth.kernels.large_head_attention import LARGE_HEAD_ENGAGED
    except Exception:
        LARGE_HEAD_ENGAGED = None

    def recorder(module, query, key, value, attention_mask, *rest, **kwargs):
        before = dict(LARGE_HEAD_ENGAGED) if LARGE_HEAD_ENGAGED is not None else None
        label = (f"attn.L{getattr(module, 'layer_idx', -1)}."
                 f"{getattr(module, 'layer_type', '?')}.hd{query.shape[-1]}")
        torch.cuda.nvtx.range_push(label)
        try:
            output = original(module, query, key, value, attention_mask, *rest, **kwargs)
        finally:
            torch.cuda.nvtx.range_pop()
        engaged = fallback = None
        if before is not None:
            engaged = LARGE_HEAD_ENGAGED["calls"] - before["calls"]
            fallback = LARGE_HEAD_ENGAGED["fallback"] - before["fallback"]
        records.append({
            "layer_idx": getattr(module, "layer_idx", None),
            "layer_type": getattr(module, "layer_type", None),
            "head_dim": int(query.shape[-1]),
            "is_sliding": bool(getattr(module, "is_sliding", None)),
            "query_dtype": str(query.dtype),
            "num_query_heads": int(query.shape[1]),
            "num_kv_heads": int(key.shape[1]),
            "mask_is_none": attention_mask is None,
            "mask_shape": None if attention_mask is None else list(attention_mask.shape),
            "mask_dtype": None if attention_mask is None else str(attention_mask.dtype),
            "hd512_kernel_engaged": engaged,
            "hd512_kernel_fallback": fallback,
        })
        return output

    ALL_ATTENTION_FUNCTIONS["sdpa"] = recorder

    def restore():
        ALL_ATTENTION_FUNCTIONS["sdpa"] = original

    return restore


def _router_installed(torch):
    """Registered != selected: report whether the HD512 router wraps sdpa at all."""
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        current = ALL_ATTENTION_FUNCTIONS["sdpa"]
        return bool(getattr(current, "_unsloth_large_head", False)
                    or getattr(getattr(current, "_unsloth_original", None),
                               "_unsloth_large_head", False))
    except Exception:
        return False


def _tensor_diff(a, b):
    """absolute / relative / cosine / non-finite between two tensors, in fp32."""
    import torch
    a32, b32 = a.detach().float().reshape(-1), b.detach().float().reshape(-1)
    nonfinite = int((~torch.isfinite(a32)).sum() + (~torch.isfinite(b32)).sum())
    diff = (a32 - b32).abs()
    denom = b32.abs().max().clamp_min(1e-12)
    cos = torch.nn.functional.cosine_similarity(a32, b32, dim=0, eps=1e-12)
    return {
        "max_absolute": float(diff.max()),
        "mean_absolute": float(diff.mean()),
        "max_relative": float((diff.max() / denom)),
        "cosine": float(cos),
        "nonfinite_count": nonfinite,
        "shape": list(a.shape),
    }


# --------------------------------------------------------------------------- check


def _resolved_config(model):
    """Snapshot the config fields the router and mask builder actually branch on, so a
    drift in the loaded model (vs the plan's config.json facts) is visible in the JSON."""
    config = getattr(model, "config", None)
    text = getattr(config, "text_config", config)
    fields = ("head_dim", "global_head_dim", "num_attention_heads",
              "num_key_value_heads", "num_global_key_value_heads",
              "attention_dropout", "attention_k_eq_v", "sliding_window")
    resolved = {name: getattr(text, name, None) for name in fields}
    resolved["attn_implementation"] = getattr(config, "_attn_implementation", None)
    layer_types = getattr(text, "layer_types", None)
    if layer_types is not None:
        resolved["global_layer_indices"] = [
            i for i, t in enumerate(layer_types) if t == "full_attention"]
        resolved["sliding_layer_count"] = sum(
            1 for t in layer_types if t == "sliding_attention")
    return resolved


def _native_train_profile(torch, model, batch, scored_suffix, engaged_counter, trace_path=None):
    """Requirement 2: the production-representative measurement. No wrapper on
    ALL_ATTENTION_FUNCTIONS, no NVTX, no eval/no_grad -- model.train() and one real
    microbatch forward+backward under torch.profiler, so the router sees the exact
    mask / compile / dropout conditions a real update does. record_shapes=True tags
    every attention CUDA kernel with its input shapes, so head_dim 512 (global) and
    head_dim 256 (sliding) are told apart by the real 8k-sequence shapes themselves --
    no synthetic short-sequence probe, and no claim that kernel choice is independent
    of sequence length."""
    import torch.profiler as torch_profiler
    was_training = model.training
    before = dict(engaged_counter) if engaged_counter is not None else None
    model.train()
    model.zero_grad(set_to_none=True)
    activities = [torch_profiler.ProfilerActivity.CPU, torch_profiler.ProfilerActivity.CUDA]
    try:
        with torch_profiler.profile(activities=activities, record_shapes=True) as profiler:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(**scored_suffix(batch), use_cache=False)
                loss = output.loss
            loss.backward()
            torch.cuda.synchronize()
        loss_value = float(loss.detach().item())
    finally:
        model.zero_grad(set_to_none=True)
        if not was_training:
            model.eval()
    if trace_path is not None:
        profiler.export_chrome_trace(str(trace_path))
    engaged = fallback = None
    if before is not None:
        engaged = engaged_counter["calls"] - before["calls"]
        fallback = engaged_counter["fallback"] - before["fallback"]
    attention_kernels = []
    for event in profiler.key_averages(group_by_input_shape=True):
        name = event.key or ""
        if event.self_device_time_total <= 0:
            continue
        if not any(tag in name.lower() for tag in
                   ("fmha", "attention", "flash", "cutlass", "sdp", "efficient", "_lh_")):
            continue
        shapes = [list(s) for s in (event.input_shapes or []) if s]
        head_dims = sorted({s[-1] for s in shapes
                            if isinstance(s, list) and s and isinstance(s[-1], int)})
        attention_kernels.append({
            "kernel": name, "count": int(event.count),
            "input_shapes": shapes, "head_dims_seen": head_dims,
            "device_time_us": float(event.self_device_time_total)})
    # CUDA kernel names alone do not carry the query shapes. Preserve the real
    # CPU attention operators' shapes and their directly correlated CUDA kernels
    # as a separate table; never infer hd512 use from a Python counter alone.
    operators = {}
    cuda_counts = Counter()
    for event in profiler.events():
        name = event.name
        if (getattr(event.device_type, "name", "") == "CUDA"
                and any(word in name.lower() for word in
                        ("fmha", "_lh_fwd", "_lh_bwd", "flash_fwd", "flash_bwd"))):
            cuda_counts[name] += 1
        shapes = event.input_shapes or []
        if (name.startswith("aten::") and "attention" in name
                and shapes and isinstance(shapes[0], (list, tuple))
                and shapes[0] and shapes[0][-1] in (256, 512)):
            query_shape = tuple(shapes[0])
            key = (name, query_shape)
            item = operators.setdefault(key, {"operator": name, "query_shape": list(query_shape),
                                              "head_dim": query_shape[-1], "count": 0,
                                              "direct_cuda_kernels": Counter()})
            item["count"] += 1
            for kernel in event.kernels:
                item["direct_cuda_kernels"][kernel.name] += 1
    for item in operators.values():
        item["direct_cuda_kernels"] = dict(item["direct_cuda_kernels"])
    return {
        "mode": "NATIVE_TRAIN_FORWARD_BACKWARD_UNINSTRUMENTED",
        "production_representative": True,
        "note": "model.train()+backward, no ALL_ATTENTION_FUNCTIONS wrapper; the "
                "intrusive per-layer recorder below is a separate, non-production probe",
        "loss": loss_value,
        "hd512_kernel_engaged": engaged,
        "hd512_kernel_fallback": fallback,
        "attention_cuda_kernels": attention_kernels,
        "attention_operators_by_real_query_shape": list(operators.values()),
        "attention_kernel_event_counts": dict(cuda_counts)}


def _mask_contract_summary(torch, captured, window=None):
    """Scalar-only evidence for router rejection; never alter or export the mask."""
    mask = captured["mask"]
    if mask is None:
        return {"mask_is_none": True}
    seq = int(captured["query"].shape[2])
    indices = torch.arange(seq, device=mask.device)
    empty, holes, band_mismatch = [], [], []
    lows, highs = [], []
    for start in range(0, seq, 1024):
        chunk = mask[:, 0, start:start + 1024]
        keep = chunk if mask.dtype == torch.bool else chunk == 0
        lo = torch.where(keep, indices, seq).amin(-1)
        hi = torch.where(keep, indices, -1).amax(-1)
        count = keep.sum(-1)
        empty.extend((torch.nonzero(count[0] == 0).flatten() + start).tolist())
        holes.extend((torch.nonzero((count[0] > 0) & (count[0] != hi[0] - lo[0] + 1)).flatten()
                      + start).tolist())
        lows.append(lo)
        highs.append(hi)
        if window:
            rows = indices[start:start + 1024]
            band = ((indices[None, :] <= rows[:, None])
                    & (indices[None, :] > rows[:, None] - window))
            mismatch = (keep != band).any(-1)
            band_mismatch.extend((torch.nonzero(mismatch[0]).flatten() + start).tolist())
    lo, hi = torch.cat(lows, dim=-1), torch.cat(highs, dim=-1)
    return {"mask_is_none": False, "mask_shape": list(mask.shape),
            "empty_row_count": len(empty), "first_empty_rows": empty[:16],
            "noncontiguous_nonempty_row_count": len(holes),
            "lo_decreases": int((lo[:, 1:] < lo[:, :-1]).sum().item()),
            "hi_decreases": int((hi[:, 1:] < hi[:, :-1]).sum().item()),
            "window": window, "plain_band_mismatch_rows": len(band_mismatch),
            "first_band_mismatch_rows": band_mismatch[:16]}


def _probe_hd512_vs_sdpa_numeric(torch, captured):
    """Requirement 3 (layer probe): on one real global layer's captured q/k/v/mask, run
    the HD512 Triton kernel and plain SDPA on the SAME inputs, same scale=1.0, real GQA,
    no dropout, and diff the output. One layer only, so no full-model math reference.

    The captured tensors are detached clones, so the memory they hold is separate from
    the production forward peak and must not be read as production VRAM.

    Mask handling matches the kernel's own convention (intervals_from_mask: bool
    True=keep, additive float 0=keep), so the comparison stays apples-to-apples:
      * a bool mask is passed to SDPA as-is (True=keep) -- an earlier version replaced
        it with zeros_like, which silently removed ALL blocking and made any match
        meaningless;
      * an additive float mask is passed as the 0/-inf bias SDPA expects;
      * a None mask is the pure-causal case, which the kernel supports via dense causal
        intervals, so it is compared too (it used to be reported SKIPPED)."""
    import torch.nn.functional as functional
    from unsloth.kernels.large_head_attention import (
        large_head_attention, intervals_from_mask, _dense_intervals)

    query, key, value, mask = (captured["query"], captured["key"],
                               captured["value"], captured["mask"])
    batch, q_heads, seq, head_dim = query.shape
    groups = q_heads // key.shape[1]
    result = {"mask_is_none": mask is None, "head_dim": int(head_dim),
              "num_query_heads": int(q_heads), "num_kv_heads": int(key.shape[1]),
              "captured_tensors_detached_clone_not_production_peak": True}

    if mask is None:
        intervals = _dense_intervals(batch, seq, True, query.device)
        result["mask_mode"] = "NONE_PURE_CAUSAL_DENSE_INTERVALS"
    else:
        intervals = intervals_from_mask(mask, batch, seq)
        result["mask_mode"] = "ADDITIVE_FLOAT_BIAS" if mask.is_floating_point() else "BOOL_KEEP"
        result["intervals_qualified"] = intervals is not None
        if intervals is None:
            result["status"] = "SDPA_ONLY_MASK_NOT_INTERVAL"
            return result

    k_full = key.repeat_interleave(groups, 1)
    v_full = value.repeat_interleave(groups, 1)
    kernel_out = large_head_attention(query, key, value, *intervals, 1.0)
    if mask is None:
        sdpa_out = functional.scaled_dot_product_attention(
            query, k_full, v_full, is_causal=True, scale=1.0).transpose(1, 2)
    else:
        # float -> additive 0/-inf bias; bool -> True=keep. Both are native SDPA inputs
        # and both match intervals_from_mask's keep convention.
        sdpa_out = functional.scaled_dot_product_attention(
            query, k_full, v_full, attn_mask=mask, scale=1.0).transpose(1, 2)
    result["status"] = "COMPARED"
    result["hd512_vs_sdpa"] = _tensor_diff(kernel_out, sdpa_out)
    return result


def _probe_flex(torch, captured_global, captured_sliding):
    """Requirement 4 (LAYER-WISE probe only): does flex_attention run on the real
    captured global (head_dim 512) and sliding (head_dim 256) tensors AND reproduce
    SDPA's output+gradient using the REAL captured allow-relations? The captured dense
    mask is turned into a BlockMask so padding and the exact window are preserved -- an
    earlier version invented a fresh causal/window mask, which is not equivalent to what
    the model actually masked.

    This is a single-layer viability check. It does NOT prove a full-model flex load:
    that needs a separate fresh process that loads native1 with
    attn_implementation='flex_attention' and runs the real Gemma-4 mask builder over all
    40 sliding + 8 global layers forward+backward. This script does not do that; the plan
    records it as the required follow-up.

    Both the eager entry and the torch.compile'd entry are tried, because eager flex
    emits a slow/partial path and the compiled entry is the officially recommended one;
    reporting only eager would misstate viability. If both raise, the path ends here."""
    try:
        from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    except Exception as error:
        return {"status": "FLEX_UNAVAILABLE", "scope": "LAYER_WISE_PROBE_ONLY",
                "reason": f"{type(error).__name__}: {error}"}
    import torch.nn.functional as functional

    compiled_flex = torch.compile(flex_attention, dynamic=False)

    def run_one(captured, is_sliding):
        query, key, value, mask = (captured["query"], captured["key"],
                                   captured["value"], captured["mask"])
        batch, q_heads, seq, head_dim = query.shape
        groups = q_heads // key.shape[1]
        k_full = key.repeat_interleave(groups, 1)
        v_full = value.repeat_interleave(groups, 1)

        # Build mask_mod from the REAL captured mask, not a reinvented window, so padding
        # and the exact per-row key range match what the model masked.
        if mask is None:
            def mask_mod(b, h, q_idx, kv_idx):
                return q_idx >= kv_idx
            mask_source = "NONE_PURE_CAUSAL"
        elif mask.is_floating_point():
            dense = mask
            def mask_mod(b, h, q_idx, kv_idx):
                return dense[b, 0, q_idx, kv_idx] >= 0  # additive bias: 0 = keep
            mask_source = "CAPTURED_ADDITIVE_FLOAT_BIAS"
        else:
            dense = mask
            def mask_mod(b, h, q_idx, kv_idx):
                return dense[b, 0, q_idx, kv_idx]  # bool: True = keep
            mask_source = "CAPTURED_BOOL_KEEP"

        out = {"head_dim": int(head_dim), "is_sliding": bool(is_sliding),
               "mask_source": mask_source}
        mask_batch = 1 if (mask is None or mask.shape[0] == 1) else batch
        try:
            block_mask = create_block_mask(mask_mod, B=mask_batch, H=None,
                                           Q_LEN=seq, KV_LEN=seq, device=query.device)
        except Exception as error:
            out["status"] = "UNSUPPORTED_BLOCKMASK"
            out["reason"] = f"{type(error).__name__}: {error}"
            return out

        # Reference: SDPA on the same inputs/mask, forward + grad (q only, enough to show
        # the backward path agrees).
        q_ref = query.detach().clone().requires_grad_()
        ref = functional.scaled_dot_product_attention(
            q_ref, k_full, v_full,
            attn_mask=(None if mask is None else mask),
            is_causal=(mask is None), scale=1.0)
        ref_grad = torch.autograd.grad(ref.sum(), q_ref)[0]

        for entry_name, fn in (("eager", flex_attention), ("compiled", compiled_flex)):
            try:
                q_flex = query.detach().clone().requires_grad_()
                flex_out = fn(q_flex, k_full, v_full, block_mask=block_mask, scale=1.0)
                flex_grad = torch.autograd.grad(flex_out.sum(), q_flex)[0]
                out[entry_name] = {
                    "status": "RAN",
                    "output_vs_sdpa": _tensor_diff(flex_out, ref),
                    "qgrad_vs_sdpa": _tensor_diff(flex_grad, ref_grad)}
            except Exception as error:
                out[entry_name] = {"status": "UNSUPPORTED",
                                   "reason": f"{type(error).__name__}: {error}"}
        out["status"] = "PROBED"
        return out

    return {"status": "PROBED", "scope": "LAYER_WISE_PROBE_ONLY",
            "full_model_flex_requires": "separate fresh-process native1 load with "
            "attn_implementation=flex_attention over all 40 sliding + 8 global layers "
            "forward+backward; not done by this script",
            "global_head_dim_512": run_one(captured_global, is_sliding=False),
            "sliding_head_dim_256": run_one(captured_sliding, is_sliding=True)}


def check(args, benchmark):
    destination = benchmark.output_dir(args.output)
    payload = benchmark.validate_payload(args.encoded, args.agent)
    result = {"status": "FAILED", "command": "check", "engine": args.engine,
              "agent": args.agent, "production_training": False,
              "reviewed_sources": _source_manifest(),
              "encoded_sha256": benchmark.sha(args.encoded),
              "compile_disable_env": os.environ.get("UNSLOTH_COMPILE_DISABLE", "0")}
    fd = restore = budget = None
    try:
        fd = benchmark.acquire_gpu_lock()
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                          TOKENIZERS_PARALLELISM="false")
        if args.engine == "unsloth":
            os.environ[HD512_ENV] = "1"
            # RETURN_LOGITS=1 keeps the native loss path, so an attention change is not
            # confused with the fused-CE delta root measured (fused .79728025 vs native
            # .79451740). This ablation isolates attention, not the loss kernel.
            os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
            import unsloth  # noqa: F401  apply patches before transformers import
        import torch
        from transformers import AutoProcessor, DataCollatorForSeq2Seq, set_seed
        sys.path.insert(0, str(benchmark.RELEASE / "common"))
        from budget import Budget
        from loss import scored_suffix
        set_seed(42)
        result["library_versions"] = _library_versions()
        budget = Budget(benchmark.PROCESS_BUDGET, benchmark.RESERVE)
        processor = AutoProcessor.from_pretrained(
            benchmark.RELEASE / "assets" / "model-processor", local_files_only=True)
        collator = DataCollatorForSeq2Seq(tokenizer=processor.tokenizer, padding=True,
                                          label_pad_token_id=-100, pad_to_multiple_of=8,
                                          return_tensors="pt")
        one = {"input_ids": payload["rows"][0]["input_ids"],
               "labels": payload["rows"][0]["labels"]}
        batch = {key: value.to("cuda") for key, value in collator([one]).items()}
        model, optimizer, parameters, names = benchmark.model_and_optimizer(
            args.engine, args.agent, torch)
        result["hd512_router_installed"] = _router_installed(torch)
        result["resolved_config"] = _resolved_config(model)

        # Requirement 2 FIRST: the production-representative path. Uninstrumented
        # model.train() forward+backward profiled before any wrapper is installed, so the
        # recorded kernels / shapes / router counter reflect what a real update does.
        engaged_counter = None
        if args.engine == "unsloth":
            from unsloth.kernels.large_head_attention import LARGE_HEAD_ENGAGED
            engaged_counter = LARGE_HEAD_ENGAGED
        result["native_train_profile"] = _native_train_profile(
            torch, model, batch, scored_suffix, engaged_counter,
            destination / "native_attention_trace.json")
        result["native_trace_sha256"] = _sha256(destination / "native_attention_trace.json")
        benchmark.write_new(destination / "NATIVE_PROFILE.json", result)
        budget.check(force=True)
        if args.profile_only:
            result.update(status="NATIVE_PROFILE_COMPLETE", cpu_rss_bytes=_rss_bytes(),
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
            budget.close()
            budget.check(force=True)
            result.update(budget.summary())
            budget = None
            benchmark.write_new(destination / "CHECK.json", result)
            return result

        # Intrusive attribution probe (NOT the production path): wrap
        # ALL_ATTENTION_FUNCTIONS to label each layer and capture one global/sliding
        # tensor set for the numeric/flex probes. A kernel engaging HERE does not prove
        # production training uses it -- the wrapper itself can change router conditions.
        records = []
        restore = _install_attention_recorder(torch, records)
        _stash = {}
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        recorder = ALL_ATTENTION_FUNCTIONS["sdpa"]

        def capturing_recorder(module, query, key, value, attention_mask, *rest, **kwargs):
            head_dim = int(query.shape[-1])
            bucket = "global" if head_dim == HD512 else "sliding"
            if bucket not in _stash:
                _stash[bucket] = {"query": query.detach().clone(),
                                  "key": key.detach().clone(),
                                  "value": value.detach().clone(),
                                  "mask": None if attention_mask is None
                                  else attention_mask.detach().clone()}
            return recorder(module, query, key, value, attention_mask, *rest, **kwargs)

        ALL_ATTENTION_FUNCTIONS["sdpa"] = capturing_recorder
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(**scored_suffix(batch), use_cache=False)
        ALL_ATTENTION_FUNCTIONS["sdpa"] = recorder  # back to plain recorder
        budget.check(force=True)
        result["router_mask_contracts"] = {
            name: _mask_contract_summary(torch, tensors,
                                        window=1024 if name == "sliding" else None)
            for name, tensors in _stash.items()}

        # Per-layer attribution (INTRUSIVE PROBE, not the production path): one record
        # per layer_idx. The per-layer hd512 engaged counts come from the wrapped path,
        # so read the native_train_profile above for the production verdict.
        by_layer = {}
        for row in records:
            by_layer.setdefault(row["layer_idx"], row)
        result["per_layer_attention"] = [by_layer[k] for k in sorted(by_layer)]
        result["per_layer_attention_is_intrusive_probe"] = True
        result["global_layers_hd512_engaged"] = sum(
            1 for r in result["per_layer_attention"]
            if r["head_dim"] == HD512 and (r["hd512_kernel_engaged"] or 0) > 0)
        result["global_layers_total"] = sum(
            1 for r in result["per_layer_attention"] if r["head_dim"] == HD512)
        result["sliding_layers_total"] = sum(
            1 for r in result["per_layer_attention"] if r["head_dim"] == HD256)

        # Requirement 3 (layer probe): HD512 kernel vs SDPA on a real global layer.
        if args.engine == "unsloth" and "global" in _stash:
            result["hd512_vs_sdpa_layer_probe"] = _probe_hd512_vs_sdpa_numeric(
                torch, _stash["global"])
        else:
            result["hd512_vs_sdpa_layer_probe"] = {
                "status": "SKIPPED", "reason": "needs unsloth engine and a captured global layer"}
        budget.check(force=True)

        # Requirement 4: flex viability probe (layer-wise).
        if args.skip_flex:
            result["flex_probe"] = {"status": "SKIPPED_USER_BOUNDED_DEDICATED_KERNEL_SCOPE"}
        elif "global" in _stash and "sliding" in _stash:
            result["flex_probe"] = _probe_flex(torch, _stash["global"], _stash["sliding"])
        else:
            result["flex_probe"] = {"status": "SKIPPED",
                                    "reason": "missing captured global/sliding layer"}
        budget.check(force=True)

        result["cpu_rss_bytes"] = _rss_bytes()
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        budget.close()
        budget.check(force=True)
        result.update(budget.summary())
        budget = None
        result["status"] = "CHECK_COMPLETE"
        benchmark.write_new(destination / "CHECK.json", result)
        return result
    except BaseException as error:
        result["status"] = "FAILED"
        result["error"] = f"{type(error).__name__}: {error}"
        benchmark.write_new(destination / "CHECK_FAILURE.json", result)
        raise
    finally:
        if restore is not None:
            restore()
        if budget is not None:
            budget.close()
        if fd is not None:
            os.close(fd)


# --------------------------------------------------------------------------- compare


def _set_hd512(enabled):
    """Flip the HD512 router on/off and verify the flip actually took effect.

    The installed source does NOT use lru_cache: it caches the env read in a module
    list ``_ENABLED = [None]`` and exposes ``_enabled.cache_clear`` as a lambda that
    just resets that list (large_head_attention.py:711-720). Swallowing a failed
    cache_clear would leave the previous verdict live, so 'off' and 'on' would run the
    SAME kernel and any timing gap would be noise. Reset the list directly, read
    ``_enabled()`` back, and raise if it disagrees with the requested state."""
    from unsloth.kernels.large_head_attention import _enabled, _ENABLED
    os.environ[HD512_ENV] = "1" if enabled else "0"
    _ENABLED[0] = None  # force the next _enabled() to re-read the env
    observed = _enabled()
    if observed != enabled:
        raise RuntimeError(
            f"HD512 toggle did not take effect: requested {enabled}, _enabled()={observed}")
    return observed


def _grad_snapshot(torch, parameters):
    """Flat fp32 copy of current LoRA gradients for equivalence diffing."""
    grads = [p.grad.detach().float().reshape(-1) for p in parameters if p.grad is not None]
    return torch.cat(grads) if grads else torch.zeros(1, device="cuda")


def _snapshot_training_state(torch, optimizer, parameters):
    """Clone just the trainable (LoRA) params + optimizer state + RNG. The 4-bit base is
    frozen, so this is small; it lets the OFF and ON timing runs start from identical
    weights/optimizer/RNG instead of ON inheriting OFF's optimizer.step mutations."""
    return {
        "params": [p.detach().clone() for p in parameters],
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }


def _restore_training_state(torch, optimizer, parameters, snapshot):
    """Reset weights/optimizer/RNG to a snapshot so each timing variant is comparable."""
    with torch.no_grad():
        for parameter, saved in zip(parameters, snapshot["params"]):
            parameter.copy_(saved)
    optimizer.load_state_dict(snapshot["optimizer"])
    torch.set_rng_state(snapshot["cpu_rng"])
    torch.cuda.set_rng_state_all(snapshot["cuda_rng"])


def _forward_backward_no_step(model, parameters, inputs, torch, scored_suffix):
    """One accumulation pass (no optimizer.step): loss sum and a LoRA-grad snapshot.
    Weights are never mutated, so two configs compare at identical weights."""
    from accelerate.utils import convert_to_fp32
    from transformers import set_seed
    model.zero_grad(set_to_none=True)
    set_seed(42)  # same RNG so LoRA dropout masks match across configs; dropout stays 0.05
    losses = []
    for input_batch in inputs:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = convert_to_fp32(model(**scored_suffix(input_batch), use_cache=False))
            loss = output.loss / benchmark_EFFECTIVE_BATCH
        losses.append(loss.detach())
        loss.backward()
        del output, loss
    torch.cuda.synchronize()
    snapshot = _grad_snapshot(torch, parameters)
    total = float(torch.stack(losses).sum().item())
    model.zero_grad(set_to_none=True)
    return total, snapshot


benchmark_EFFECTIVE_BATCH = 16  # mirrors benchmark.EFFECTIVE_BATCH; asserted at run time


def compare(args, benchmark):
    global benchmark_EFFECTIVE_BATCH
    benchmark_EFFECTIVE_BATCH = benchmark.EFFECTIVE_BATCH
    destination = benchmark.output_dir(args.output)
    payload = benchmark.validate_payload(args.encoded, args.agent)
    result = {"status": "FAILED", "command": "compare", "engine": "unsloth",
              "agent": args.agent, "production_training": False,
              "reviewed_sources": _source_manifest(),
              "note": "within-process head_dim-512 toggle; bounded equivalence by default, "
                      "timing only with --run-timing and only if the toggle changed the kernel"}
    fd = budget = None
    original_hd512_env = os.environ.get(HD512_ENV)
    try:
        fd = benchmark.acquire_gpu_lock()
        # RETURN_LOGITS=1: native loss path, so attention and the fused-CE delta are not
        # confused (fused .79728025 vs native .79451740, root's measurement).
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                          TOKENIZERS_PARALLELISM="false", UNSLOTH_RETURN_LOGITS="1")
        os.environ[HD512_ENV] = "1"
        import unsloth  # noqa: F401
        import torch
        from transformers import AutoProcessor, DataCollatorForSeq2Seq, set_seed
        sys.path.insert(0, str(benchmark.RELEASE / "common"))
        from budget import Budget
        from loss import scored_suffix
        from unsloth.kernels.large_head_attention import LARGE_HEAD_ENGAGED
        set_seed(42)
        result["library_versions"] = _library_versions()
        budget = Budget(benchmark.PROCESS_BUDGET, benchmark.RESERVE)
        processor = AutoProcessor.from_pretrained(
            benchmark.RELEASE / "assets" / "model-processor", local_files_only=True)
        collator = DataCollatorForSeq2Seq(tokenizer=processor.tokenizer, padding=True,
                                          label_pad_token_id=-100, pad_to_multiple_of=8,
                                          return_tensors="pt")
        one = {"input_ids": payload["rows"][0]["input_ids"],
               "labels": payload["rows"][0]["labels"]}
        inputs = [{key: value.to("cuda") for key, value in collator([one]).items()}
                  for _ in range(benchmark.EFFECTIVE_BATCH)]
        model, optimizer, parameters, names = benchmark.model_and_optimizer(
            "unsloth", args.agent, torch)
        result["hd512_router_installed"] = _router_installed(torch)
        nonpad = sum(int(b["attention_mask"].sum().item()) for b in inputs)

        # --- Bounded probe FIRST: no optimizer.step, identical weights/RNG/dropout. This
        # is the ~minutes check; it also decides whether a longer timing run is even
        # meaningful (does the toggle actually change which kernel runs?).
        equivalence = {}
        for label, enabled in (("hd512_off", False), ("hd512_on", True)):
            observed = _set_hd512(enabled)
            before = dict(LARGE_HEAD_ENGAGED)
            loss, snapshot = _forward_backward_no_step(
                model, parameters, inputs, torch, scored_suffix)
            equivalence[label] = {
                "loss_sum": loss,
                "hd512_enabled_observed": observed,
                "hd512_calls": LARGE_HEAD_ENGAGED["calls"] - before["calls"],
                "hd512_fallback": LARGE_HEAD_ENGAGED["fallback"] - before["fallback"],
                "_snapshot": snapshot}
            budget.check(force=True)
        off, on = equivalence["hd512_off"], equivalence["hd512_on"]
        result["numeric_equivalence"] = {
            "hd512_off_loss_sum": off["loss_sum"],
            "hd512_on_loss_sum": on["loss_sum"],
            "loss_absolute_delta": abs(on["loss_sum"] - off["loss_sum"]),
            "hd512_off_calls": off["hd512_calls"],
            "hd512_on_calls": on["hd512_calls"],
            "hd512_off_enabled_observed": off["hd512_enabled_observed"],
            "hd512_on_enabled_observed": on["hd512_enabled_observed"],
            "lora_grad_diff": _tensor_diff(on["_snapshot"], off["_snapshot"]),
        }
        del off["_snapshot"], on["_snapshot"]

        # Gate: a timing comparison is only meaningful if OFF and ON actually ran
        # different kernels. If they did not, measuring time would measure noise.
        result["counter_suggests_route_change"] = (
            on["hd512_calls"] != off["hd512_calls"] and on["hd512_calls"] > 0)
        # A compiled kernel may never touch this Python counter. This legacy
        # within-process toggle cannot provide the required fresh-process CUDA
        # proof, so it must not authorize a long timing run from counters alone.
        toggle_changed_kernel = None
        result["toggle_changed_kernel"] = toggle_changed_kernel
        result["kernel_proof_status"] = "UNVERIFIED_COUNTERS_ARE_NOT_CUDA_PROOF"
        result["speedup_off_over_on"] = None

        # --- Timing: OPT-IN (--run-timing) and only if the toggle changed the kernel.
        # Each variant is restored to the same initial weights/optimizer/RNG, so ON does
        # not inherit OFF's optimizer.step mutations.
        if not args.run_timing:
            timing = None
            result["timing_skipped_reason"] = (
                "bounded probe default; rerun with --run-timing for warmup+timed updates")
        elif toggle_changed_kernel is not True:
            timing = None
            result["timing_skipped_reason"] = (
                "legacy within-process counter toggle is not fresh-process CUDA proof; "
                "timing is blocked rather than reporting a false speedup")
        else:
            timing = {}
            snapshot_state = _snapshot_training_state(torch, optimizer, parameters)
            for label, enabled in (("hd512_off", False), ("hd512_on", True)):
                _restore_training_state(torch, optimizer, parameters, snapshot_state)
                _set_hd512(enabled)
                before = dict(LARGE_HEAD_ENGAGED)
                for _ in range(args.warmup):
                    benchmark.one_update(model, optimizer, parameters, inputs, torch,
                                         budget, measure=False)
                torch.cuda.reset_peak_memory_stats()
                steps = [benchmark.one_update(model, optimizer, parameters, inputs, torch,
                                              budget, measure=True)
                         for _ in range(args.timed_updates)]
                seconds = [s["seconds"] for s in steps]
                timing[label] = {
                    "timed_update_seconds": seconds,
                    "mean_update_seconds": sum(seconds) / len(seconds),
                    "mean_loss_per_update": [s["mean_loss"] for s in steps],
                    "nonpad_tokens_per_second":
                        nonpad * len(seconds) / sum(seconds),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "hd512_calls": LARGE_HEAD_ENGAGED["calls"] - before["calls"],
                    "hd512_fallback": LARGE_HEAD_ENGAGED["fallback"] - before["fallback"],
                    "warmup_updates": args.warmup, "timed_updates": len(steps)}
                budget.check(force=True)
            off_t, on_t = timing["hd512_off"], timing["hd512_on"]
            result["speedup_off_over_on"] = (
                off_t["mean_update_seconds"] / on_t["mean_update_seconds"]
                if on_t["mean_update_seconds"] else None)
        result["timing"] = timing
        result["microbatch"] = 1
        result["effective_batch"] = benchmark.EFFECTIVE_BATCH
        result["nonpad_tokens_per_update"] = nonpad
        result["cpu_rss_bytes"] = _rss_bytes()
        budget.close()
        budget.check(force=True)
        result.update(budget.summary())
        budget = None
        result["status"] = ("COMPARE_COMPLETE_TIMING_AND_EQUIVALENCE" if timing
                            else "COMPARE_EQUIVALENCE_ONLY")
        benchmark.write_new(destination / "COMPARE.json", result)
        return result
    except BaseException as error:
        result["status"] = "FAILED"
        result["error"] = f"{type(error).__name__}: {error}"
        benchmark.write_new(destination / "COMPARE_FAILURE.json", result)
        raise
    finally:
        # Restore the env and the cached verdict so a later run in this interpreter is not
        # silently pinned to this run's last toggle.
        if original_hd512_env is None:
            os.environ.pop(HD512_ENV, None)
        else:
            os.environ[HD512_ENV] = original_hd512_env
        module = sys.modules.get("unsloth.kernels.large_head_attention")
        if module is not None and hasattr(module, "_ENABLED"):
            module._ENABLED[0] = None  # only if already imported; never load CUDA here
        if budget is not None:
            budget.close()
        if fd is not None:
            os.close(fd)


# --------------------------------------------------------------------------- CLI


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check", "compare", "padding-cpu", "padding-gpu", "padding-diagnose", "first-sdpa"))
    parser.add_argument("--agent", choices=("response", "recommendation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoded", type=Path)
    parser.add_argument("--engine", choices=("baseline", "unsloth"), default="unsloth")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timed-updates", type=int, default=3)
    parser.add_argument("--run-timing", action="store_true",
                        help="compare: run the warmup+timed updates (minutes to ~1h). "
                             "Default is the bounded equivalence probe only.")
    parser.add_argument("--profile-only", action="store_true",
                        help="check: native training profiler only, no intrusive/layer/Flex probes")
    parser.add_argument("--skip-flex", action="store_true",
                        help="check: keep dedicated-kernel investigation bounded; skip Flex probes")
    parser.add_argument("--padding-receipt", type=Path)
    args = parser.parse_args(argv)
    if args.command in ("padding-gpu", "padding-diagnose", "first-sdpa") and args.padding_receipt is None:
        parser.error("padding-gpu requires --padding-receipt")
    if args.command == "prepare":
        if args.encoded is not None:
            parser.error("prepare builds the encoded artifact; do not pass --encoded")
    else:
        if args.encoded is None:
            parser.error(f"{args.command} requires --encoded")
        if args.warmup < 1 or args.timed_updates < 1:
            parser.error("warmup and timed-updates must be >= 1")
    return args


def main(argv=None):
    args = parse(argv)
    benchmark = _load_benchmark()
    if args.command == "prepare":
        result = prepare(args, benchmark)
    elif args.command == "padding-cpu":
        result = padding_cpu(args, benchmark)
    elif args.command in ("padding-gpu", "padding-diagnose", "first-sdpa"):
        result = padding_gpu(args, benchmark)
    elif args.command == "check":
        result = check(args, benchmark)
    else:
        result = compare(args, benchmark)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
