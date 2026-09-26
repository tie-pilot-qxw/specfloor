set -uo pipefail
export CUDA_VISIBLE_DEVICES=4
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/DeepSpec
/workspace/sglang017-env/bin/python -m measurement.probe_rpre --corpus C0 \
  --corpus-file /workspace/measurement_runs/calib_20260818/C0/arena8k.jsonl \
  --cheap /workspace/measurement_runs/calib_20260818/C0/arena8k.ladder.M512.jsonl \
  --drafter /workspace/dflash_sgl --out /workspace/measurement_runs/rpre/arena8k.rpre.jsonl \
  --anchors 96 --paths 256 --split --kv-budget-gib 3 2>&1 | grep -avE "Loading weights" | tail -5
echo "-- arena8k rows: $(wc -l < /workspace/measurement_runs/rpre/arena8k.rpre.jsonl)"
echo "===== ARENA RERUN DONE ====="
