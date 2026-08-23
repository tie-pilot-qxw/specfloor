set -uo pipefail
export CUDA_VISIBLE_DEVICES=4
PY=/workspace/sglang017-env/bin/python
D=/workspace/measurement_runs/calib_20260818/C0
cd /workspace/DeepSpec
# Pilot: 16 anchors of gsm8k at M=1024, rungs 0 AND 1 so T^(0)-T^(1) is a paired
# difference on identical paths. Purpose is to time it before committing four
# domains -- rung 1 does 2 scoring passes over M sequences for each of 6 slots,
# i.e. ~12k sequence-scorings per anchor against rung 0's one pass.
date +"start %H:%M:%S"
$PY -m measurement.probe_tk --corpus C0 --corpus-file $D/gsm8k.jsonl \
    --cheap $D/ladder.M512.jsonl --out /workspace/measurement_runs/t1/PILOT.jsonl \
    --anchors 16 --paths 1024 --top-k 256 --rungs 0,1 --split 2>&1 \
    | grep -avE "Capturing|Multi-thread|it/s\]$" | tail -8
date +"end   %H:%M:%S"
echo "-- rows: $(wc -l < /workspace/measurement_runs/t1/PILOT.jsonl)"
echo "===== PILOT DONE ====="
