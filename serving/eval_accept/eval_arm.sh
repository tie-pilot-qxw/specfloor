#!/usr/bin/env bash
# Serve one draft checkpoint, record per-request accepted length, stop the server.
#
#   PY=<python with patched SGLang> EVAL_DATASETS=<DeepSpec>/eval_datasets \
#     eval_arm.sh <checkpoint> <out.json> <gpu> [port]
#
# Settings (environment):   TEMP    CAP   OFFICIAL_SUBSET  SEED    CONC
#   one-epoch components:   0.0     50    (unset)          (unset) 32
#   training trajectory:    1.0     500   1                980406  32
# ALGO=DFLASH serves a DFlash/DFlash2 checkpoint (the DFlash2 reproduction row);
# the default DSPARK serves DSpark-format checkpoints, including ours.
#
# Accepted length is invariant to batching, so requests run 32 at a time; speed
# is not, and must never be read off this harness (see ../sweep for timing).
set -euo pipefail
CKPT=$1; OUT=$2; GPU=$3; PORT=${4:-30090}
PY=${PY:-python}
: "${EVAL_DATASETS:?set EVAL_DATASETS to the DeepSpec eval_datasets directory}"
HERE=$(cd "$(dirname "$0")" && pwd)
# The block size comes from the checkpoint, never from a default: serving an
# 8-block checkpoint at 7 does not fail, it regroups across the wrong boundary.
BS=${BS:-$(sed -n 's/.*"block_size"[ \t]*:[ \t]*\([0-9][0-9]*\).*/\1/p' "$CKPT/config.json" | head -1)}
: "${BS:=7}"
# The two algorithms read --speculative-num-draft-tokens differently:
#   DSPARK  gamma = block size, so num_draft_tokens = BS + 1 counts the bonus token;
#   DFLASH  num_draft_tokens is the block size, whose first position is the anchor.
if [ "${ALGO:-DSPARK}" = "DFLASH" ]; then NUM_DRAFT=$BS; else NUM_DRAFT=$((BS + 1)); fi
LOG=${OUT%.json}.server.log

"$PY" -m sglang.launch_server \
  --model-path Qwen/Qwen3-4B \
  --speculative-algorithm "${ALGO:-DSPARK}" \
  --speculative-draft-model-path "$CKPT" \
  --speculative-dspark-block-size "$BS" \
  --speculative-num-draft-tokens "$NUM_DRAFT" \
  --mem-fraction-static "${MEM_FRAC:-0.55}" --dtype bfloat16 --trust-remote-code \
  --host 127.0.0.1 --port "$PORT" --base-gpu-id "$GPU" \
  > "$LOG" 2>&1 &
SERVER=$!
trap 'kill -TERM "$SERVER" 2>/dev/null || true' EXIT
for _ in $(seq 1 180); do
  curl -s -o /dev/null --max-time 3 "http://127.0.0.1:$PORT/health" && break
  kill -0 "$SERVER" 2>/dev/null || { tail -40 "$LOG" >&2; exit 1; }
  sleep 5
done
curl -s -o /dev/null --max-time 3 "http://127.0.0.1:$PORT/health" || { echo "server did not come up" >&2; exit 1; }
# What the server actually built, next to the number it produces.
grep -oE "markov_head=[A-Za-z]*|DFlash2?DraftModel" "$LOG" | sort -u | head -2 || true

LABEL=${LABEL:-$(basename "$OUT" .json)} EVAL_DATASETS="$EVAL_DATASETS" \
  "$PY" "$HERE/sglang_paired_accept.py" "http://127.0.0.1:$PORT" "$OUT" \
  "${TEMP:-0.0}" "${CONC:-32}" "${CAP:-50}"
