#!/bin/bash
# t1/ rerun with the slot-alignment fix in probe_tk. See PREDICTION.md.
# Everything except the fix is identical to t1/run_t1.sh: same corpora, same
# ladder, same anchors, same seeds, same top-k, both path counts.
set -uo pipefail
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export HF_HUB_OFFLINE=1
export HF_HOME=/workspace/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/workspace/specfloor
export SPECFLOOR_DEEPSPEC=/workspace/DeepSpec
export SPECFLOOR_EVAL_ROOT=/workspace/DeepSpec/eval_datasets
PY=/workspace/sglang017-env/bin/python
D=/workspace/measurement_runs/calib_20260818/C0
OUT=/workspace/measurement_runs/t1_fix
mkdir -p $OUT
QUIET="Capturing|Multi-thread|it/s\]|Loading|Fetching|UserWarning|_warn_once"

run () {   # $1 domain  $2 corpus  $3 ladder  $4 paths  $5 suffix
  echo ""; echo "############ $1  M=$4 ############"
  [ -s $OUT/$1$5.jsonl ] || $PY -m specfloor.probe_tk --corpus C0 \
      --corpus-file $D/$2 --cheap $D/$3 --out $OUT/$1$5.jsonl \
      --anchors 96 --paths $4 --top-k 256 --rungs 0,1 --split 2>&1 \
    | grep -avE "$QUIET" | tail -8
  echo "-- $1$5 rows: $(wc -l < $OUT/$1$5.jsonl)"
}
for M in 1024 256; do
  [ $M = 1024 ] && S=.t01 || S=.t01.m256
  run gsm8k   gsm8k.jsonl   ladder.M512.jsonl        $M $S
  run mbpp    mbpp.jsonl    mbpp.ladder.M512.jsonl   $M $S
  run alpaca  alpaca.jsonl  alpaca.ladder.M512.jsonl $M $S
  run arena8k arena8k.jsonl arena8k.ladder.M512.jsonl $M $S
done
echo ""; echo "===== T1_FIX ALL DONE ====="
