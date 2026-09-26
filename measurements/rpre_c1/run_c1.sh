set -uo pipefail
export CUDA_VISIBLE_DEVICES=4
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/workspace/sglang017-env/bin/python
cd /workspace/DeepSpec
# C1 gsm8k: the serving law. mu and p_verify are BOTH the warped distribution.
echo "############ gsm8k C1 (T=0.7 top-p=0.8 top-k=20) ############"
$PY -m measurement.probe_rpre --corpus C1 \
  --corpus-file /workspace/measurement_runs/gate_c1/gsm8k.C1.jsonl \
  --cheap /workspace/measurement_runs/rpre_c1/gsm8k.C1.anchors.jsonl \
  --drafter /workspace/dflash_sgl --out /workspace/measurement_runs/rpre_c1/gsm8k.rpre.jsonl \
  --anchors 96 --paths 256 --split --kv-budget-gib 6 2>&1 | grep -avE "Loading weights" | tail -6
echo "-- gsm8k C1 rows: $(wc -l < /workspace/measurement_runs/rpre_c1/gsm8k.rpre.jsonl)"
# C0 arena8k, retried with the memory fixes + per-anchor OOM guard
echo ""; echo "############ arena8k C0 (retry) ############"
$PY -m measurement.probe_rpre --corpus C0 \
  --corpus-file /workspace/measurement_runs/calib_20260818/C0/arena8k.jsonl \
  --cheap /workspace/measurement_runs/calib_20260818/C0/arena8k.ladder.M512.jsonl \
  --drafter /workspace/dflash_sgl --out /workspace/measurement_runs/rpre/arena8k.rpre.jsonl \
  --anchors 96 --paths 256 --split --kv-budget-gib 4 2>&1 | grep -avE "Loading weights" | tail -6
echo "-- arena8k C0 rows: $(wc -l < /workspace/measurement_runs/rpre/arena8k.rpre.jsonl)"
echo ""; echo "===== C1 + ARENA DONE ====="
