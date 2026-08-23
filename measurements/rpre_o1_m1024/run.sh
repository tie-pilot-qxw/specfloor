#!/bin/bash
# The order-1 run of rpre_o1/ repeated at four times the paths, to settle
# whether the 2.2x disagreement between the two T^(1) estimators at slot 6 is
# resolution or something else. PREDICTION.md states what each outcome means
# and was written before this ran.
#
# Everything except --paths is identical to rpre_o1/run.sh: same corpus files,
# same anchor source, same drafter, same seeds. The per-anchor random stream is
# seeded from (prompt_id, t, SEED), so the first 256 draws are NOT the same as
# the M=256 run -- chunking differs, and the docstring in probe_rpre.rollout
# says why that changes how the stream is consumed. The comparison is therefore
# paired per (anchor, slot) but not per path.
#
# --kv-budget-gib is raised from 4 to 12 because the chunk count, not the
# budget, is what the extra paths cost; at 4 GiB this would prefill three times
# as often for no reason. It does not enter any estimate.
set -uo pipefail
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export HF_HUB_OFFLINE=1
export HF_HOME=/workspace/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/workspace/specfloor
export SPECFLOOR_DEEPSPEC=/workspace/DeepSpec
PY=/workspace/sglang017-env/bin/python
D=/workspace/measurement_runs/calib_20260818/C0
OUT=/workspace/measurement_runs/rpre_o1_m1024
DRAFT=deepseek-ai/dspark_qwen3_4b_block7
mkdir -p $OUT

run () {
  echo ""; echo "############ $1 (order 1, M=1024) ############"
  $PY -m specfloor.probe_rpre --corpus C0 --corpus-file $D/$2 --cheap $D/$3 \
      --drafter $DRAFT --order 1 --cond both --out $OUT/$1.rpre1.jsonl \
      --anchors 96 --paths 1024 --split --kv-budget-gib 12 2>&1 \
    | grep -avE "Loading|Fetching|it/s\]|UserWarning|SOLUTION|flex_attention|_warn_once|debug your score_mod|This will allow" | tail -6
  echo "-- $1 rows: $(wc -l < $OUT/$1.rpre1.jsonl)"
}
run gsm8k   gsm8k.jsonl   ladder.M512.jsonl
run mbpp    mbpp.jsonl    mbpp.ladder.M512.jsonl
run alpaca  alpaca.jsonl  alpaca.ladder.M512.jsonl
run arena8k arena8k.jsonl arena8k.ladder.M512.jsonl
echo ""; echo "===== ORDER-1 M=1024 DONE ====="
