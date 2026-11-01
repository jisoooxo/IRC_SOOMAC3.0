# Decision Unsloth remote image package — validation pending

This directory prepares a separate batch-entrypoint image for lab spec 2.9.
The shared `tools/unsloth_training/production.py` has a Decision/container
CPU path and its candidate image built. The first candidate GPU smoke loaded
the official stack but stopped before step 1 at the Trainer loss-kwargs guard.
The patched production source passed the same 16-row HF Trainer CPU test in
the official image and was packaged as candidate v3, not yet GPU-tested.
GPU correctness and resume gates remain open. This is **not** a submit-ready
TAR. Do not upload/start/replace the running
Decision job from this package.

## Pinned platform and inputs

- Official `unsloth/unsloth:core-nightly-2026.10.28`, linux/amd64 image digest
  `sha256:852e00064749edbaecb57f8388a6776915bfe744cb5238d34b809d152623d87f`.
  The digest pins the Python/CUDA/Unsloth stack; no runtime package installation.
- Official image documentation specifies PyTorch 2.11/CUDA 12.8, sm_120 SASS,
  RTX PRO 6000 support and NVIDIA driver >=570.26. The actual lab driver and
  container kernel operation have not been checked.
- Sealed build05 release: `/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/releases/conversation_r5_20261101_build05_speed`.
  Source dataset manifest SHA-256 is
  `7850948287237c5ace9d1b0f56931419760fbce3f0afa02cea40681acb6ceffa`.
  Use the release's exact `remote/lab-inputs.json` as the Docker label. The
  external model cache must provide `config.json` and `model.safetensors` under
  `MODEL_DIR`; this image does not contain model weights.

## Staging and two image stages

Create a **new temporary build context** (not the sealed release), with these
real files, not symlinks. `bundle/` is a byte-for-byte copy of the sealed
release; `package/run_training.sh` is this directory's script;
`source/` contains the shared D-capable production loader and benchmark helper.
The Dockerfile's `candidate` stage needs no validation receipt; the `final`
stage requires the candidate's GPU validation receipt and its three bounded
evidence JSON files. The shared Decision runner is now staged in the candidate.

```text
context/
  bundle/RELEASE.json
  bundle/common/...
  bundle/dataset/...
  bundle/assets/model-processor/...
  bundle/remote/lab-inputs.json
  package/run_training.sh
  package/test_processor_encoding.py
  source/unsloth_training/production.py
  source/training_acceleration/benchmark.py
  validation/VALIDATION_RECEIPT.json  # final stage only
  validation/<loss,gradient,restore evidence>.json  # final stage only
```

After the D runner's CPU gate passes, stage a context in `/tmp`, preserving
the sealed bundle. The `validation/` directory may remain empty for the
candidate stage. These commands are **not** authorization to launch GPU work;
the root operator owns Docker/GPU scheduling.

```bash
repo=/home/roma/IRC_SOOMAC3.0
release=/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/releases/conversation_r5_20261101_build05_speed
context_dir="$(mktemp -d /tmp/soomac-unsloth-d.XXXXXX)"
mkdir -p "$context_dir/bundle" "$context_dir/package" "$context_dir/source/unsloth_training" "$context_dir/source/training_acceleration" "$context_dir/validation"
cp -a "$release/." "$context_dir/bundle/"
cp "$repo/tools/unsloth_remote/run_training.sh" "$context_dir/package/run_training.sh"
cp "$repo/tools/unsloth_remote/test_processor_encoding.py" "$context_dir/package/test_processor_encoding.py"
cp "$repo/tools/unsloth_training/production.py" "$context_dir/source/unsloth_training/production.py"
cp "$repo/tools/training_acceleration/benchmark.py" "$context_dir/source/training_acceleration/benchmark.py"
test -z "$(find "$context_dir/bundle" -type l -print -quit)"
lab_inputs="$(python3 -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1])),separators=(",",":")))' "$context_dir/bundle/remote/lab-inputs.json")"
docker build --platform linux/amd64 --target candidate --build-arg "LAB_INPUTS=$lab_inputs" -f "$repo/tools/unsloth_remote/Dockerfile" -t soomac/decision-unsloth:candidate-20261101 "$context_dir"
```

Variables alive at each command: `repo`, `release`, `context_dir`, then
`lab_inputs` are defined above before use. No source or existing TAR is
deleted/overwritten. The Dockerfile verifies compact label equality, every
manifest-listed dataset file, and every sealed processor file. Record candidate image ID, actual package versions,
driver, uid, entrypoint/CMD and embedded paths. After root's candidate smoke,
save/fail/resume and loss/gradient gates, write the receipt and its three
SHA-pinned, at-most-64-KiB evidence JSON files into `$context_dir/validation/`
and run the bounded helper below. Docker BuildKit cannot resolve a local
digest-only `FROM`, so the helper checks the candidate tag's image ID against
the receipt before and after building, then verifies the final RootFS layers
extend that exact candidate. It refuses an existing final tag.

```bash
python3 "$repo/tools/unsloth_remote/build_final.py" \
  --context "$context_dir" \
  --candidate-tag soomac/decision-unsloth:candidate-20261101-v3 \
  --final-tag soomac/decision-unsloth:final-20261101-v1
```

이 자리에서 살아있는 변수: `repo`(위 staging 첫 줄), `context_dir`(위 `mktemp` 줄).

Test that
**final image** again on the GPU, then save it with
`docker image save` (not `docker export`) to a new path and record TAR SHA-256.
Candidate digest can be in its receipt; a final image digest cannot be stored
inside that same image without self-reference. Keep final image/TAR digest in
an external package record. Uploading reserves a shared slot; check job/slot
noncollision immediately before any upload.

## Required Decision runner interface

`run_training.sh` passes `--mode train|smoke|preflight --agent decision
--execution-environment container`, the
exact release, `DATASET_DIR`, `MODEL_DIR`, sealed processor path,
`CHECKPOINT_DIR`, clamped byte budget, reserve, batch 1/effective 16, and
`--resume none|auto`. Smoke-only `SOOMAC_SMOKE_MAX_STEPS=5|10` and
`SOOMAC_TEST_FAIL_STEP=0|5|10` pass through to the shared runner; train and
preflight reject these overrides. It rejects absent server budget/fraction in GPU modes
and invalid resume parameters. Decision `train` must verify the bundled
`/app/validation/VALIDATION_RECEIPT.json`; `smoke`/`preflight` may run without
it. The runner must additionally:

1. Start schema-1 progress and 10-second heartbeat before heavy import/load;
   count actual optimizer updates/samples, not microbatches.
2. Clamp process use to server `GPU_MEMORY_BUDGET_BYTES`,
   `GPU_MEMORY_FRACTION` and NVML total, with 1 GiB non-allocator reserve;
   fail if any cap cannot be established.
3. Keep sealed Decision data, label mask, full history, processor, NF4/BF16
   double-quant, rank 16/alpha 32/dropout .05, effective batch 16 and 9216 cap.
4. Save complete model/optimizer/scheduler/step/RNG state, atomically publish
   `.lab-monitor/resume.json` only after a full save, and fail closed on damaged
   or mismatched `LAB_RESUME=1` restore.
5. Run first a short smoke with save → forced failure → resume, then separate
   loss/gradient/parameter and runtime equivalence checks before any full run.

CPU-only package check:

```bash
python3 -m unittest tools.unsloth_remote.test_remote_package -v
```

This test does not establish GPU compatibility, training correctness, image
buildability, or server acceptance. Check remote driver/slot/job state before
any upload; do not stop or change the existing Decision job.
