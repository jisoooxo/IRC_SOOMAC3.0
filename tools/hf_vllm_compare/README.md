# HF / vLLM current-stack comparison

User-authorized GPU evaluation only. Production nodes, prompts, adapters, environments and launcher are read-only.

Plan: `docs/2026-11-05_hf_vllm_compare_plan.md`.
Evidence: `training/learning_first_v43/final/goals/GL10/hf_vllm_compare_20261105/`.

1. `prepare.py`: verify source/converted LoRA tensor equality including tied-K/V copies; pin original assets and freeze12 current-prompt requests plus36 scenario turns.
2. `run.py`: sequential fresh processes, original STT/TTS resident, owned resource-limited scopes, port18001, NVML monitoring and cleanup. Output directories must be new; existing evidence is never silently overwritten.
3. `worker.py`: three frozen-input sweeps and real36-turn graph, no physical actions. `worker_lora_bf16.py` is an extra precision control changing only process-local HF LoRA tensors, not original checkpoints.
4. `summarize.py`: recompute statistics into SUMMARY.json; repeat JSON agreement is not a semantic correctness oracle.

Initial6-condition controller was already running when the BF16 control was added; run the seventh condition separately after its cleanup. This is an experiment harness, not an operational launch script. `hf_nocache` disables prefix reuse only; normal within-generation KV cache remains enabled. No history shortening, adapter merge, new training, model download or installation.

Frozen-input timings compare the same request IDs/grammar/limits. Generated histories can diverge in closed-loop scenarios, so their wall times compare complete deployment behavior, not identical compute. HF and vLLM library/native-kernel versions differ. Original HF PEFT promotes LoRA toFP32; explicit BF16 control is recorded separately. The invariant-off condition still has no physical side effects. STT/TTS residency is not live speech pipeline latency.

Recovery: first vLLM no-cache attempt ended after6 calls (causeUNKNOWN); retry is `_r2`. HF BF16 preparation triggered wetext ModelScope auxiliary downloads/HTTP403; owned scopes were stopped and evidence preserved. `hold_tts_offline.py` pins only wetext snapshot lookup to existing localfiles, then executes the unchanged original helper with text_frontend=False. Final HF BF16 result is `_r2`. Some auxiliary cachefiles were updated by the originallibrary; no claim of whole-filesystem immutability. Production source/package/adapter preservation has separate AFTER.json evidence.

`report.py` regenerates `docs/2026-11-05_hf_vllm_compare_generated.md` and OUTPUTS.md. The curated final review is `docs/2026-11-05_hf_vllm_compare_review.md`; regeneration does not overwrite its adjudication or audit limitations.
