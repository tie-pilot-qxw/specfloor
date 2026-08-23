set -uo pipefail
# arena8k only, at a KV budget that does not truncate it. The 3 GiB pass in
# run.sh dropped exactly the 15 longest contexts, which on the long-context
# domain is a length-dependent selection rather than a random loss, so that
# domain is redrawn here and the 3 GiB arena8k output is discarded.
#
# The budget needed differs by order because the two probes hold different
# things resident: the order-1 head runs against a cached prefix, the order-0
# backbone materialises full-vocabulary logits for the whole chunk. 8 GiB is
# enough for order 1 (96/96, chunk 112); order 0 needs 12 (96/96, chunk 168)
# where 8 still skipped 6 anchors on OOM.
export CUDA_VISIBLE_DEVICES=5
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/workspace/sglang017-env/bin/python
D=/workspace/measurement_runs/calib_20260818/C0
OUT=/workspace/measurement_runs/srv
cd /workspace/DeepSpec
run () {   # order drafter kv_gib
  echo ""; echo "############ arena8k (order $1, $3 GiB) ############"
  $PY -m measurement.probe_rpre --corpus C0 --corpus-file $D/arena8k.jsonl \
      --cheap $D/arena8k.ladder.M512.jsonl --drafter "$2" --order "$1" --cond both \
      --out $OUT/arena8k.srv$1.jsonl --anchors 96 --paths 256 --split \
      --kv-budget-gib "$3" 2>&1 | grep -avE "Loading weights" | tail -6
  echo "-- arena8k order $1 rows: $(wc -l < $OUT/arena8k.srv$1.jsonl)"
}
run 1 deepseek-ai/dspark_qwen3_4b_block7 8
run 0 /workspace/dflash_sgl 12
echo ""; echo "===== ARENA RERUN DONE ====="
