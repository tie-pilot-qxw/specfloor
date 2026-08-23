set -uo pipefail
# gsm8k lost its slot to a race: pick_gpu saw the card free, and a neighbour's
# 56 GiB job landed before sglang finished starting. Rerun pinned to a card
# that is empty rather than merely under threshold.
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${1:-6}
PY=/workspace/sglang017-env/bin/python
D=/workspace/measurement_runs/calib_20260818/C0
OUT=/workspace/measurement_runs/tk20
cd /workspace/DeepSpec
echo "== gsm8k top-20 rerun on GPU $CUDA_VISIBLE_DEVICES"
TK_PROFILE=1 $PY -m measurement.probe_tk --corpus C0 --corpus-file $D/gsm8k.jsonl \
    --cheap $D/ladder.M512.jsonl --out $OUT/gsm8k.t01.tk20.jsonl \
    --anchors 96 --paths 256 --top-k 20 --rungs 0,1 --split 2>&1 \
  | grep -avE "Capturing|Multi-thread|it/s\]$" | tail -12
echo "-- gsm8k rows: $(wc -l < $OUT/gsm8k.t01.tk20.jsonl)"
