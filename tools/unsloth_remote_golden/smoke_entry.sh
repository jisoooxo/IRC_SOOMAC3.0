#!/bin/sh
# Smoke-validation entrypoint of the derived image (packages/training code/model policy = c3).
# It only drives the SAME /app/run_training.sh that the final train image uses.
#   job 1 (fresh mount)        : smoke 10 steps; production fails on purpose after committed step 5.
#                                Exit is nonzero so the lab can "새 작업으로 재개" this job.
#   job 2 (LAB_RESUME=1, step 5): finish steps 6..10 from checkpoint 5 (native full-state restore),
#                                then one-row loss/gradient/env check into /checkpoints/gpu-validation.
set -eu
PY=/opt/soomac/py/bin/python
output="${CHECKPOINT_DIR:-/checkpoints}"
if [ "$#" -ne 1 ] || [ "$1" != smoke-validation ]; then
    echo "this image only runs: smoke-validation" >&2; exit 64
fi
export SOOMAC_SMOKE_MAX_STEPS=10

if [ "${LAB_RESUME:-0}" = 1 ]; then
    if [ "${RESUME_STEP:-}" != 5 ]; then
        echo "SMOKE_PHASE2 expects resume from committed step 5, got '${RESUME_STEP:-}'" >&2; exit 64
    fi
    export SOOMAC_TEST_FAIL_STEP=0
    /app/run_training.sh smoke
    echo "SMOKE_PHASE2 resumed smoke reached step 10; starting one-row GPU validation" >&2
    exec "$PY" /app/gpu_validate.py --output "$output/gpu-validation"
fi

export SOOMAC_TEST_FAIL_STEP=5
set +e
/app/run_training.sh smoke
status=$?
set -e
if [ "$status" -eq 0 ]; then
    echo "SMOKE_PHASE1 FAIL: forced failure after step 5 did not fire" >&2; exit 70
fi
if "$PY" -I -c 'import json,sys; from pathlib import Path; root=Path(sys.argv[1]); d=json.loads((root/".lab-monitor/resume.json").read_text()); c=root/d["checkpoint"]; assert d["global_step"]==5 and c.is_dir() and not c.is_symlink(), d; print(json.dumps(d))' "$output"; then
    echo "SMOKE_PHASE1 EXPECTED_FAILURE exit=$status after committed step 5: resume this job (새 작업으로 재개)" >&2
else
    echo "SMOKE_PHASE1 FAIL exit=$status and no committed step-5 marker: do not resume" >&2
fi
exit "$status"
