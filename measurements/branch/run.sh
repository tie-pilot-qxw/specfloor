set -uo pipefail
# 5.6: the K-median of the realisation family in total variation. This
# DECOMPOSES the order-0 floor into a commitment cost and a residual; it is
# NOT a tree drafter's acceptance ceiling. The min over the K centres sits
# inside the expectation, so it prices an oracle router that knows which path
# it is on. The width-K acceptance ceiling is a different, union-of-events
# object and is measured separately in the coverage table of the same section.
#
# K=1 must be in --widths: it recovers T^(0) exactly on these same rollouts,
# which is what makes every "removed" figure a within-anchor difference rather
# than a comparison of two runs.
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/workspace/sglang017-env/bin/python
D=/workspace/measurement_runs/calib_20260818/C0
OUT=/workspace/measurement_runs/branch
mkdir -p $OUT; cd /workspace/DeepSpec

run () {
  echo ""; echo "######## $1"
  $PY -m measurement.probe_kmedian --corpus C0 --corpus-file $D/$2 --cheap $D/$3 \
      --out $OUT/$1.tb.jsonl --anchors 32 --paths 256 --top-k 256 \
      --widths 1,2,4 --restarts 3 --iters 12 2>&1 \
    | grep -avE "Loading weights|Capturing|it/s\]$" | tail -6
  echo "-- $1 rows: $(wc -l < $OUT/$1.tb.jsonl)"
}
run gsm8k   gsm8k.jsonl   ladder.M512.jsonl
run mbpp    mbpp.jsonl    mbpp.ladder.M512.jsonl
run alpaca  alpaca.jsonl  alpaca.ladder.M512.jsonl
run arena8k arena8k.jsonl arena8k.ladder.M512.jsonl
echo ""; echo "===== K-MEDIAN DONE ====="
