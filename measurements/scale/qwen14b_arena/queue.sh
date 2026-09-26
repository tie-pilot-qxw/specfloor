#!/bin/bash
# Wait for a card with room for 14B in bf16 (28 GiB of weights plus a KV cache
# large enough that sglang's own minimum-viable check passes at ~0.53 of an
# 80 GiB card), then run. The cards here are shared, so a fixed --gpu would
# either sit idle or OOM depending on a neighbour.
NEED=46000        # MiB free
GEMMA=/workspace/measurement_runs/scale/gemma12b_fix/run.log

# Do not contend with our own other job. The first attempt picked a card during
# the gap between gemma's ladder and its probe_tk, and the two then shared it
# until the ladder here died of OOM. Wait it out; the corpus, which is the long
# part, is already on disk and the guards below skip it.
while ! grep -q "RE-ANCHORED ALL DONE" $GEMMA 2>/dev/null; do sleep 120; done
echo "$(date +%H:%M) gemma finished; looking for a card"

while true; do
  pick=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
         | awk -v n=$NEED -F', ' '{f=$3-$2; if (f>n) {print $1, f}}' | sort -k2 -rn | head -1 | cut -d' ' -f1)
  if [ -n "$pick" ]; then
    echo "$(date +%H:%M) starting on GPU $pick"
    GPU=$pick bash /workspace/measurement_runs/scale/qwen14b_arena/run.sh
    rc=$?
    [ $rc -eq 0 ] && exit 0
    echo "$(date +%H:%M) attempt on GPU $pick exited $rc; retrying"
  fi
  sleep 300
done
