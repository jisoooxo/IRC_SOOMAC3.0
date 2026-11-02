#!/bin/sh
# Remote Decision entrypoint for the golden-environment candidate.
# Same contract as tools/unsloth_remote/run_training.sh (candidate v3), plus:
#  - interpreter is the reproduced golden conda Python at /opt/soomac/py
#  - GPU modes refuse a host driver older than CUDA 13.0's minimum (580.65.06),
#    because torch 2.12.1+cu130 cannot run there. Unknown driver is reported, not passed.
#  - the GPU validation receipt lives in /app/gpu-validation, because the sealed
#    release already owns /app/validation (TOKEN_MASK_ROWS.jsonl etc.).
set -eu

PY=/opt/soomac/py/bin/python
MIN_DRIVER=580.65.06

mode="${SOOMAC_MODE:-${1:-train}}"
case "$mode" in train|smoke|preflight) ;; *) echo "invalid mode: $mode" >&2; exit 64 ;; esac
if [ "$#" -gt 1 ]; then echo "unexpected command arguments" >&2; exit 64; fi
if [ "$mode" = smoke ]; then
    smoke_steps="${SOOMAC_SMOKE_MAX_STEPS:-10}"
    fail_step="${SOOMAC_TEST_FAIL_STEP:-0}"
    case "$smoke_steps" in 5|10) ;; *) echo "smoke steps must be 5 or 10" >&2; exit 64 ;; esac
    case "$fail_step" in 0|5|10) ;; *) echo "invalid smoke failure step" >&2; exit 64 ;; esac
    if [ "$fail_step" -gt "$smoke_steps" ]; then echo "failure step exceeds smoke steps" >&2; exit 64; fi
    set -- --smoke-max-steps "$smoke_steps" --test-fail-step "$fail_step"
else
    if [ -n "${SOOMAC_SMOKE_MAX_STEPS:-}" ] || [ -n "${SOOMAC_TEST_FAIL_STEP:-}" ]; then
        echo "smoke overrides are forbidden outside smoke mode" >&2; exit 64
    fi
    set --
fi
dataset="${DATASET_DIR:-/app/assets/dataset}"
model="${MODEL_DIR:-/model}"
output="${CHECKPOINT_DIR:-/checkpoints}"
for path in "$dataset" "$model" "$output"; do
    if [ ! -d "$path" ]; then echo "required directory missing: $path" >&2; exit 66; fi
done
if [ ! -w "$output" ]; then echo "checkpoint directory is not writable: $output" >&2; exit 73; fi
if [ ! -f /app/tools/unsloth_training/production.py ]; then
    echo "shared Decision Unsloth runner is absent" >&2; exit 78
fi

if [ "$mode" != preflight ]; then
    case "${GPU_MEMORY_BUDGET_BYTES:-}" in
        ''|*[!0-9]*) echo "GPU_MEMORY_BUDGET_BYTES must be supplied by server" >&2; exit 64 ;;
    esac
    if [ -z "${GPU_MEMORY_FRACTION:-}" ]; then
        echo "GPU_MEMORY_FRACTION must be supplied by server" >&2; exit 64
    fi
    driver=""
    if [ -r /proc/driver/nvidia/version ]; then
        # "...Kernel Module for x86_64  580.178.04  Release Build..." : skip the x86_64 digits.
        driver="$(head -n 1 /proc/driver/nvidia/version | grep -oE '[0-9]{3}\.[0-9]+(\.[0-9]+)?' | head -n 1)"
    fi
    if [ -z "$driver" ] && command -v nvidia-smi >/dev/null 2>&1; then
        driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 | tr -d ' ' || true)"
    fi
    # Fail closed: an unreadable or malformed driver version is not a pass.
    if ! printf '%s' "$driver" | grep -Eqx '[0-9]{3}\.[0-9]+(\.[0-9]+)?'; then
        echo "SOOMAC_DRIVER_CHECK status=UNKNOWN driver='$driver' required>=$MIN_DRIVER" >&2
        exit 69
    fi
    lowest="$(printf '%s\n%s\n' "$MIN_DRIVER" "$driver" | sort -V | head -n 1)"
    if [ "$lowest" != "$MIN_DRIVER" ]; then
        echo "SOOMAC_DRIVER_CHECK status=FAIL driver=$driver required>=$MIN_DRIVER (CUDA 13.0 runtime)" >&2
        exit 69
    fi
    echo "SOOMAC_DRIVER_CHECK status=PASS driver=$driver required>=$MIN_DRIVER" >&2
fi
case "${SOOMAC_REMOTE_REQUESTED_BUDGET_BYTES:-}" in
    ''|*[!0-9]*) echo "invalid requested GPU budget" >&2; exit 64 ;;
esac
budget="$SOOMAC_REMOTE_REQUESTED_BUDGET_BYTES"
if [ -n "${GPU_MEMORY_BUDGET_BYTES:-}" ] && [ "$GPU_MEMORY_BUDGET_BYTES" -lt "$budget" ]; then
    budget="$GPU_MEMORY_BUDGET_BYTES"
fi
case "${SOOMAC_NON_ALLOCATOR_RESERVE_BYTES:-}" in
    ''|*[!0-9]*) echo "invalid non-allocator reserve" >&2; exit 64 ;;
esac

if [ "${LAB_RESUME:-0}" = 1 ]; then
    case "${RESUME_STEP:-}" in ''|*[!0-9]*) echo "invalid RESUME_STEP" >&2; exit 64 ;; esac
    checkpoint="${RESUME_CHECKPOINT_DIR:-}"
    case "$checkpoint" in "$output"/*) ;; *) echo "resume path outside CHECKPOINT_DIR" >&2; exit 64 ;; esac
    case "$checkpoint" in *'/../'*|*'/./'*|*'/..'|*'/.'|*'//'* ) echo "unsafe resume path" >&2; exit 64 ;; esac
    if [ ! -e "$checkpoint" ] || [ -L "$checkpoint" ]; then echo "resume checkpoint missing or linked" >&2; exit 66; fi
    resume=auto
elif [ "${LAB_RESUME:-0}" = 0 ]; then
    if [ -n "${RESUME_STEP:-}" ] || [ -n "${RESUME_CHECKPOINT_DIR:-}" ]; then
        echo "resume variables require LAB_RESUME=1" >&2; exit 64
    fi
    resume=none
else
    echo "invalid LAB_RESUME" >&2; exit 64
fi

# The shared runner additionally clamps to NVML total and server fraction.
exec "$PY" /app/tools/unsloth_training/production.py \
    --mode "$mode" --agent decision --execution-environment container \
    --release /app/RELEASE.json \
    --dataset "$dataset" --model "$model" --processor /app/assets/model-processor \
    --output "$output" --budget-bytes "$budget" \
    --non-allocator-reserve-bytes "$SOOMAC_NON_ALLOCATOR_RESERVE_BYTES" \
    --batch 1 --resume "$resume" \
    --validation-receipt /app/gpu-validation/VALIDATION_RECEIPT.json \
    "$@"
