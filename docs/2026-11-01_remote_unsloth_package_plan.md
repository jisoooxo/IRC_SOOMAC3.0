# Decision Unsloth remote package plan (2026-11-01)

This is a **two-stage package awaiting validation**, not a trained model or submit-ready TAR. The
existing Decision job remains untouched. See the [lab spec v2.9](</home/roma/Downloads/TRAINING_IMAGE_SPEC.ko_v2_9 (2).md>)
→ image labels, embedded files, heartbeat, resource clamp, and full-state
save/fail/resume are mandatory. See the prior
[feasibility note](2026-11-01_remote_unsloth_feasibility.md)
→ model-family feasibility is distinct from project-runner correctness.

## Current feasibility

The official [Unsloth Docker image](https://hub.docker.com/r/unsloth/unsloth)
documents `core` as a script-oriented image with PyTorch 2.11/CUDA 12.8,
linux/amd64 sm_120 kernels and RTX PRO 6000 support; it requires driver
>=570.26. [NVIDIA CUDA 12.8 notes](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/)
also list sm_120 and the 570.26 Linux driver floor. The Dockerfile pins the
official amd64 `core-nightly-2026.10.28` image by full digest. Actual lab
driver, package versions *inside that digest*, xformers/Unsloth operations,
Decision training, speed, and memory fit remain unmeasured. The base image's
Studio/port/tunnel command is replaced by a foreground batch `train` entrypoint.

The shared `tools/unsloth_training/production.py` now has the Decision and
container CPU path. The pinned base image was pulled, and candidate v2
(`sha256:67a6f1d8eeda17e87408d1fd6a2f7052412256982376f18c053bb7fd6a1dea41`)
built and passed UID1000 CPU preflight. The first candidate GPU smoke loaded
the official stack but stopped before step 1 at the Trainer loss-kwargs guard;
its owner patched the loss-kwargs scaling guard. The exact 16-row HF Trainer
CPU gradient test passed in the official image (transformers 5.17.0), and
candidate v3 (`sha256:bf465ad8543b8a280f8322980a879fb46cba09d2356ee8d49581c082a95c2d3a`)
contains production SHA `86638bad3614a633b6fa023694f5c7b37ca840e9e37177be0eeff9d6a27e3208`.
Candidate v3 has not been GPU-tested, so no GPU training validation is claimed. The
final stage has not been built because the required GPU evidence is absent.
No remote upload/POST or existing D modification was done in this package task.

CPU-only UID1000 check of the pinned base image found torch 2.11.0+cu128,
transformers 5.17.0, PEFT 0.21.2, Unsloth/unsloth-zoo 2026.10.10 and
bitsandbytes 0.50.2; the lab HF preparation imports also succeeded with
network disabled. These versions differ from the local isolated Python, so a
local training receipt cannot substitute for candidate-image GPU validation.
For one sealed longest Decision train row (`SCV2-D-0042`, 8608 tokens), CPU
`core.encode_record` with the sealed processor produced identical local
transformers 5.13.1 and container 5.17.0 hashes:
`input_ids_sha256=4317447fdcc1539ef7df03e588d18152af5dad5da98d925abb6550ceddd460c2`,
`labels_sha256=442e9b3b85bbf20cdc184f423fb0f9541899ab2e2e0b1333cf378e1f9c27396f`;
mask boundary 8593/15 also matched the sealed receipt. This is one-row
encoding evidence, not full-dataset or GPU-loss equivalence.

## Packaging and acceptance gates

The candidate context must contain exact copies of sealed build05 release,
Decision dataset and processor under `/app/assets`, the new shell entrypoint,
and the D-capable shared `production.py` plus loader helper. The final context
adds `VALIDATION_RECEIPT.json` and three bounded SHA-pinned evidence JSON files
as a validation-only layer. Build that stage from the exact GPU-tested candidate
candidate tag, checking its immutable image ID against the receipt before and
after the final build. The `build_final.py` helper also verifies that final
RootFS layers extend the tested candidate's layers. Docker BuildKit did not
resolve the local candidate by digest-only `FROM`, so a tag plus these ID and
layer-prefix gates is required. The release's compact `lab-inputs.json`
must exactly match the label. Use linux/amd64, UID:GID 1000:1000, `/model` and
`/data` read-only, `/checkpoints` writeable, `HOME=/tmp`, `CMD train` and no
runtime install. The shell clamps requested bytes to the server byte limit;
the future runner must also clamp to server fraction and NVML total before
Torch allocation. A static Dockerfile is insufficient to prove this.

Before a TAR can be called ready: finish the D-capable path-parameterized
runner; verify its sealed processor/label-mask and Unsloth loss/gradients
against baseline; check the actual remote driver; build/inspect the pinned
candidate image; run CPU import/progress checks as UID1000; run short GPU
smoke under the provided server budget; prove complete save → forced failure
→ resume, including optimizer/scheduler/RNG/step and bad-checkpoint rejection;
then build and **retest the final receipt-bearing image** and inspect the
saved TAR. Upload/start only after the root operator verifies
slot/job noncollision. Existing D is never an automatic replace target.

The bundled receipt may report the local candidate smoke budget of
32,000,000,000 bytes and observed peak, but must mark remote hardware fit and
remote throughput **unmeasured**. The remote request ceiling is 44,000,000,000
bytes with fixed microbatch 1/accumulation 16 and 1 GiB reserve; it is still
clamped by the server byte limit, server fraction and NVML total. A larger
ceiling is not a claim of larger actual use or fit.

## Decision speed experiment

Use the sealed `TOKEN_MASK_ROWS.jsonl` receipts to select the **longest
Decision train row** without re-encoding every row; the observed largest
`full_tokens` is 8608, within the 9216 cap. Select that row 16 times and
encode only those 16 through sealed `core.encode_record` and processor. Keep
the exact input IDs/labels artifact and SHA-256 local, then run baseline and
Unsloth in independent processes with identical seed 42, NF4/BF16,
double-quant, LoRA rank16/alpha32/dropout.05, microbatch1/effective16,
optimizer, LR, scored-suffix loss and no history trim. First run one update
each for smoke; only then warmup 2 plus timed 3 optimizer updates. Record
per-update sync time, mean/median, actual process/NVML peak, padded and
nonpad tokens/s, samples/s, loss/gradient finite checks separately from
timing, versions and image digest. Attention operator/dtype trace is diagnostic,
outside timing.

The same-model diagnostic found fused `UNSLOTH_RETURN_LOGITS=0` loss drift
(`0.00276285`) beyond the native CE tolerance. It is not approved for
production; Decision candidate must use `UNSLOTH_RETURN_LOGITS=1`, prove
selected-logit native CE agreement, and time **that** mode. A separate
`--check-fused` diagnostic may record the failed experiment but is not an
acceptance gate. The evidence is pinned by `model_builder_sha256`, covering
the loader/quantization functions and sealed core/loss sources without making
CLI/reporting-only changes invalidate the CE proof.

The existing 186.37 seconds/update figure is one cold baseline update, so it
is **not** a speedup denominator for the warmed three-update statistic. A
like-for-like repeated baseline on the same pre-encoded rows is required.
No faster/fit claim follows from local P/Q or an unrun image. After gates,
choose remote D only if it preserves correctness and meaningfully reduces
same-workload per-update time without violating the live job's resource slot.
