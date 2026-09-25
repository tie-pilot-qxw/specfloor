#!/usr/bin/env bash
# T^(0) and T^(1) at block length 16 (slots 0-15), for the longer-block check.
#
# This replaces a first gamma=16 run that was made with an older copy of
# probe_tk predating b6e6597, the forced-suffix slot off-by-one: its T^(1) was
# the too-small pre-fix estimator. Its T^(0) did not depend on the bug, and this
# run reproduces it, because anchors, seeds and paths are unchanged.
#
# Anchors come from the gamma=7 M=512 ladder: shuffled with the frozen seed,
# first 96 per domain, and skipped when fewer than 16 continuation tokens
# remain, which leaves 372 of the 384. Their inclusion probabilities pi are
# those of the gamma=7 eligible population.
#
#   CALIB=<decompressed calib_20260818/C0> OUT=<dir> GPUS="0 1 2 3" bash run.sh
#
# runs one domain per listed GPU, in parallel, and waits for all of them.
set -uo pipefail
PY=${PY:-python}
CALIB=${CALIB:?set CALIB to the directory holding the C0 corpora and M=512 ladders}
OUT=${OUT:?set OUT to the output directory}
read -r -a GPU_LIST <<< "${GPUS:-0}"
DOMAINS=(gsm8k mbpp alpaca arena8k)
export SPECFLOOR_GAMMA=16
mkdir -p "$OUT"

ladder () { [ "$1" = gsm8k ] && echo ladder.M512.jsonl || echo "$1.ladder.M512.jsonl"; }

run () {   # $1 domain  $2 gpu
  local out="$OUT/$1.g16.jsonl"
  if [ -s "$out" ]; then echo "$1: $out exists, skipped"; return; fi
  echo "$1: gpu $2 start $(date -u +%FT%TZ)"
  CUDA_VISIBLE_DEVICES=$2 "$PY" -m specfloor.probe_tk --corpus C0 \
      --corpus-file "$CALIB/$1.jsonl" --cheap "$CALIB/$(ladder "$1")" \
      --out "$out" --anchors 96 --paths 1024 --top-k 256 --rungs 0,1 --split \
      > "$OUT/$1.log" 2>&1
  echo "$1: exit $? rows $(wc -l < "$out" 2>/dev/null || echo 0) $(date -u +%FT%TZ)"
}

i=0
for d in "${DOMAINS[@]}"; do
  run "$d" "${GPU_LIST[$(( i % ${#GPU_LIST[@]} ))]}" &
  i=$(( i + 1 ))
  # With fewer GPUs than domains, wait for a round before starting the next.
  [ $(( i % ${#GPU_LIST[@]} )) -eq 0 ] && wait
done
wait
