set -uo pipefail
# 5.11: the single-slot best response. Reopen slot k, hold the shipped proposal
# at the other six, and water-fill the reopened slot per anchor.
#
# probe_br records ONLY (realised token, target p, drafter q) per path per slot
# -- no [M, V] rows -- which is what makes M=1024 affordable here when the
# floor probes run at 256. Everything that costs thought rather than GPU time
# (the water filling, the two-fold cross-fit, the epsilon smoothing) lives in
# br_report, so re-splitting or re-smoothing never triggers another GPU pass.
#
# Both orders share the anchor files and --paths with the rest of the paper, so
# the paths are the same population; they are NOT bit-identical to the M=256
# probes, which draw 256.
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/workspace/sglang017-env/bin/python
D=/workspace/measurement_runs/calib_20260818/C0
OUT=/workspace/measurement_runs/br
DRAFT0=deepseek-ai/dflash_qwen3_4b_block7
DRAFT1=deepseek-ai/dspark_qwen3_4b_block7
mkdir -p $OUT; cd /workspace/DeepSpec

run () {  # $1 domain  $2 corpus  $3 anchors  $4 order  $5 drafter
  echo ""; echo "######## $1 (order $4)"
  $PY -m measurement.probe_br --corpus C0 --corpus-file $D/$2 --cheap $D/$3 \
      --drafter $5 --order $4 --out $OUT/$1.br$4.jsonl \
      --anchors 96 --paths 1024 --kv-budget-gib 4 2>&1 \
    | grep -avE "Loading weights|Capturing|it/s\]$" | tail -6
  echo "-- $1 rows: $(wc -l < $OUT/$1.br$4.jsonl)"
}

for ORD in 0 1; do
  DR=$DRAFT0; [ "$ORD" = "1" ] && DR=$DRAFT1
  run gsm8k   gsm8k.jsonl   ladder.M512.jsonl        $ORD $DR
  run mbpp    mbpp.jsonl    mbpp.ladder.M512.jsonl   $ORD $DR
  run alpaca  alpaca.jsonl  alpaca.ladder.M512.jsonl $ORD $DR
  run arena8k arena8k.jsonl arena8k.ladder.M512.jsonl $ORD $DR
done
echo ""; echo "===== BR 4B DONE ====="
