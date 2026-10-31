# Local evidence (measured 2026-11-01 on this machine)

GPU: NVIDIA RTX PRO 4500 Blackwell, 32623 MiB, cc 12.0, driver 580.178.04, idle VRAM used 869 MiB.
Env gemma4_env: torch 2.12.1+cu130, transformers 5.13.1, peft 0.19.1, bitsandbytes 0.49.2, triton 3.7.1, xgrammar 0.2.5, torchao 0.17.0. No vllm/sglang/flash-attn/flashinfer installed.

Model: /home/roma/Desktop/sLLM/gemma-4-12B-it, Gemma4UnifiedForConditionalGeneration, 48 layers (40 sliding w=1024 + 8 full),
head_dim 256 (sliding, 8 KV heads), global_head_dim 512 (1 KV head, attention_k_eq_v), 16 q heads, hidden 3840, inter 15360, vocab 262144.
Loader: src/soomac_irc/soomac_irc/model_runtime.py:253-269 (bnb NF4, double quant, bf16 compute, quant_storage bf16, attn sdpa).
3 LoRA (r16, alpha32, q/k/v/o/gate/up/down of language_model only, peft_version 0.21.0 in adapter_config) loaded by name, switched per call (model_runtime.py:287-336).
Generate: decision_model.py:155-162, response_model.py:68-85 — greedy, use_cache, XGrammar HF LogitsProcessor (decision uses any_order=True). VLM call_vlm disables adapter (model_runtime.py:198-202).
No cross-call KV reuse anywhere.

## Logged latency (logs/llm/2026-10-3*/agent_turns.jsonl, llm_trace latency_ms wrapper incl. tokenize+generate+decode)
decision n=78: in 6212-6826 tok; out=10 -> 3.65-4.14 s; out=47 -> 6.44 s  => slope ~75 ms/output token, intercept ~2.9 s
response n=50: in ~6000 out 23-38 -> 4.3-5.8 s ; in 2567 out 25 -> 2.3 s
recommendation n=7: in 3832 out 56-58 -> 4.7-5.1 s

## Static prefix (tokenized with the real tokenizer on logged messages; prefix = tokens identical to system-only render)
decision: total 6209-6481, static prefix 6040 (93-97%)
recommendation: 3832 total, 3338 static (87%)
response/task: 5962-6611 total, 5280 static (80-89%)

## SDPA microbench, torch 2.12.1 on SM120, L=6300, bf16, 16 heads
With enable_gqa=True: masked hd256 -> only MATH works (80.9 ms); hd512 -> only MATH (130 ms).
With KV repeated to 16 heads (what HF does when mask!=None or head_dim>256; transformers/integrations/sdpa_attention.py:28-38):
 hd256 causal no mask: FLASH 2.6 ms, EFFICIENT 6.8 ms, MATH 86 ms, CUDNN unavailable
 hd256 sliding bool mask: FLASH unavailable, EFFICIENT 16.2 ms, MATH 80.8
 hd512 causal: FLASH unavailable ("requires head dim <= 256"), EFFICIENT 14.0 ms, MATH 133
 hd512 + mask: EFFICIENT 31.1 ms
 CUDNN attention not available for any of these on SM120.
Gemma4 HF builds a sliding_window mask for sliding layers (modeling_gemma4_unified.py:666) -> sliding layers use EFFICIENT ~16 ms.
=> attention per 6.3k prefill ≈ 40*16 + 8*(14..31) ≈ 0.75-0.9 s.

## Linear microbench (q 3840->4096, o 4096->3840, gate/up 3840->15360, down 15360->3840; k/v excluded), x48 layers
prefill M=6300: bf16 864 ms, bnb nf4 921 ms (bf16 ~146 TFLOPS achieved)
decode M=1: bf16 26 ms/token, nf4 14 ms/token   (logged decode ≈ 75 ms/token -> ~60 ms/token is non-GEMM overhead: python/launch/PEFT LoRA/XGrammar/attention/repeat_kv)

## Derived estimate of a Decision call (in 6.3k, out ~20): prefill ~2.5-2.9 s (linear 0.9 + attn 0.8 + rest ~0.8-1.2 unattributed), decode ~1.5 s.
