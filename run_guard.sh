#!/usr/bin/env bash
set -euo pipefail

PY=python
SCRIPT1=train_guard.py
SCRIPT2=eval_guard.py

MODEL_SIZE=${1:-"1.7B"}        # arg1
TASK=${2:-"RETURN"}            # arg2
START_STAGE=${3:-1}            # arg3
END_STAGE=${4:-10}             # arg4
BATCH_SIZE=${BATCH_SIZE:-4}
OUTDIR=${OUTDIR:-"eval_results"}

for stage in $(seq "$START_STAGE" "$END_STAGE"); do
  echo "=== Running: model_size=$MODEL_SIZE task=$TASK stage=$stage ==="
  $PY $SCRIPT1 \
    --model_size "$MODEL_SIZE" \
    --task "$TASK" \
    --stage "$stage"
  $PY $SCRIPT2 \
    --model_size "$MODEL_SIZE" \
    --task "$TASK" \
    --stage "$stage" \
    --batch_size "$BATCH_SIZE" \
    --out_dir "$OUTDIR"
done