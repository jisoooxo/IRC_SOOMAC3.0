#!/bin/sh
set -eu

export CHECKPOINT_DIR="${CHECKPOINT_DIR:-/checkpoints}"
export HOME="${LAB_HOME:-/tmp/lab-home}"
export HF_HOME="${HF_HOME:-/tmp/huggingface}"
export PYTHONUNBUFFERED=1

mkdir -p "$HOME" "$HF_HOME"

if [ ! -d "$CHECKPOINT_DIR" ] || [ ! -w "$CHECKPOINT_DIR" ]; then
    echo "Checkpoint directory is missing or not writable: $CHECKPOINT_DIR" >&2
    exit 1
fi

if [ -n "${GPU_MEMORY_FRACTION:-}" ]; then
    export XLA_PYTHON_CLIENT_MEM_FRACTION="$GPU_MEMORY_FRACTION"
fi

if [ "$#" -eq 0 ]; then
    echo "Training command is missing." >&2
    exit 64
fi

exec "$@"
