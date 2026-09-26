set -uo pipefail
export CUDA_VISIBLE_DEVICES=4
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "measurement.probe_rpre" > /dev/null; do sleep 20; done
cd /workspace/DeepSpec
# Same C1 prefixes, same anchors, C0 LAW. Differencing this against the C1 run
# isolates the sampling law; differencing the C0-corpus run against it would
# also fold in a different anchor composition.
echo "############ gsm8k: C1 corpus / C1 anchors, C0 law ############"
/workspace/sglang017-env/bin/python -m measurement.probe_rpre --corpus C0 \
  --corpus-file /workspace/measurement_runs/gate_c1/gsm8k.C1.jsonl \
  --cheap /workspace/measurement_runs/rpre_c1/gsm8k.C1.anchors.jsonl \
  --drafter /workspace/dflash_sgl \
  --out /workspace/measurement_runs/rpre_c1/gsm8k.c1prefix_c0law.jsonl \
  --anchors 96 --paths 256 --split --kv-budget-gib 6 2>&1 | grep -avE "Loading weights" | tail -4
echo "-- rows: $(wc -l < /workspace/measurement_runs/rpre_c1/gsm8k.c1prefix_c0law.jsonl)"
echo "===== LAW ABLATION DONE ====="
