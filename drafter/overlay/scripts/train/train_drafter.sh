#!/usr/bin/env bash
# Train one of the paper's drafter configurations on online target features.
#
#   CUDA_VISIBLE_DEVICES=0,1,2,3 DSPARK_TRAIN_DATA=<online_train.jsonl> \
#     bash scripts/train/train_drafter.sh config/dspark/attnconv_qwen3_4b_b7_10ep.py [--opts k=v ...]
#
# train.py starts one worker per visible GPU.  Every paper run used global batch 512
# with local batch 1; the gradient accumulation is 512 / (#GPUs x 1), so any GPU
# count dividing 512 sees the same 512 samples per optimizer step.  The paper's runs
# used 4 H100 80GB GPUs (accumulation 128).
#
# Resuming is automatic: the trainer resumes from <checkpoint_dir>/step_latest when
# it exists (see resume_drafter.sh for the checks worth running first).
set -euo pipefail
CFG=${1:?usage: train_drafter.sh <config.py> [--opts key=value ...]}
shift
test -f "$CFG" || { echo "no such config: $CFG" >&2; exit 1; }
GPUS=${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES}
N=$(awk -F, '{print NF}' <<< "$GPUS")
GB=$(grep -oE "global_batch_size=[0-9]+" "$CFG" | head -1 | cut -d= -f2)
LB=$(grep -oE "local_batch_size=[0-9]+" "$CFG" | head -1 | cut -d= -f2)
test $(( GB % (N * LB) )) -eq 0 || {
  echo "global_batch_size=$GB is not divisible by $N GPUs x local_batch_size=$LB" >&2; exit 1; }
echo "gpus=$N global_batch=$GB local_batch=$LB grad_accum=$(( GB / (N * LB) ))"

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
exec python train.py --config "$CFG" "$@"
