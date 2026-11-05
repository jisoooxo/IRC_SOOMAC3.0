# Isolated CosyVoice vLLM trial (2026-11-05)

Experimental node only. Production TTS/STT/LLM files and original environments are not edited. Do not adopt by changing the production interpreter to this venv: its HF path produced degenerate 0.04s audio.

- `audio_node.py`: production callback subclass with process-local factory patch, isolated ROS domain217, artifact-only WAVs, captured publications, actual speaker writes. STT uses stored audio, no microphone.
- `measure.py`: one fresh stack, port18001, owned systemd scopes, flock, sampled resource guard and cleanup. Original Gemma16K/.58 and D6/Q6/Pbest150; independent TTS vLLM0.17 with explicit512MiB KV, max_len4096/max_seqs1. `--post-only` fills missing baseline stress/ASR diagnostics and is not a repeated full scenario.
- `llm_worker.py`: real graph and current shared-prefix startup warmup; frozen36-turn text scenarios, robot-completion fixtures, no physical robot actions.
- `summarize.py` and `verify_preservation.py`: statistics/audio rejection and original source/package/adapter identity, GPU/port/scope cleanup checks.
- `.venv/` and `model/`: ignored local dependencies and symlink/export overlay. Original CosyVoice/TTS files remain intact. Inspect `INSTALL_ACTUAL.json` and `PIP_CHECK.log` before any reuse.

Plan/result: `docs/2026-11-05_tts_vllm_trial_plan.md`, `docs/2026-11-05_tts_vllm_trial_review.md`.
Artifacts: `training/learning_first_v43/final/goals/GL10/tts_vllm_trial_20261105/`.

A fresh, uniquely named condition can be launched with the original cosyvoice3 Python (controller needs pynvml); rerunning an existing label intentionally fails to preserve results. `run_matrix.py --resume` is a record of the recovered experiment, not an idempotent startup command.

Statistics refresh (reads measurements; rewrites only the summary artifact):

```bash
cd /home/roma/IRC_SOOMAC3.0
python3 tools/tts_vllm_trial/summarize.py
```

The command writes only SUMMARY.json in the trial artifacts. First audio means first software speaker.write, not acoustic onset. RTF is generation duration/audio duration, with playback on a separate consumer. vLLM export BF16, dependency versions and official top-k sampling differ from original HF FP32/RAS; per-command seed resets do not reset the independent vLLM engine RNG. This is a candidate-package comparison, not a pure engine ablation. ASR checks do not establish voice quality. DONE means diagnostics completed, not all contracts passed.
