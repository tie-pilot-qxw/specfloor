set -uo pipefail
# Serving reweighting. NOTE the KV budget here is 3 GiB against 4 in
# arena8k's output from THIS script is discarded -- at 3 GiB that domain lost
# exactly its 15 longest contexts, and rerun_arena.sh redraws it at a budget
# that keeps all 96.
# measurement_runs/rpre/run_rpre.sh, and chunk size is a deterministic function
# of that budget, so the sampler consumes its stream differently and these are a
# DIFFERENT draw from the same law on the same anchors -- chunk 26 against 34 on
# gsm8k. Pooled T and R still agree with the earlier run to <= 0.0044, which is
# the cross-implementation scale, but per-cell agreement is Monte Carlo, not
# exact. Everything this run is FOR is within-run: R_free and R_serve are the
# same TV on the same paths under two weightings, so their difference is paired
# regardless.
#
# What is new: the probe now stores, per path per slot, the accept factor
# a_k = min(1, q_k(Z_k)/p_k(Z_k)). Its running product along a path is that
# path's probability of REACHING the slot, so it reweights the free-rollout
# population into the serving one and gives R_serve alongside R_free. The same
# product gives S_j = E[prod a_i] directly, which is the quantity that prod(1-R_i)
# only approximates.
export CUDA_VISIBLE_DEVICES=5
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/workspace/sglang017-env/bin/python
D=/workspace/measurement_runs/calib_20260818/C0
OUT=/workspace/measurement_runs/srv
mkdir -p $OUT; cd /workspace/DeepSpec
run () {   # order out_suffix drafter domain corpus cheap
  echo ""; echo "############ $4 (order $1) ############"
  $PY -m measurement.probe_rpre --corpus C0 --corpus-file $D/$5 --cheap $D/$6 \
      --drafter "$3" --order "$1" --cond both --out $OUT/$4.$2.jsonl \
      --anchors 96 --paths 256 --split --kv-budget-gib 3 2>&1 \
    | grep -avE "Loading weights" | tail -4
  echo "-- $4 order $1 rows: $(wc -l < $OUT/$4.$2.jsonl)"
}
for d in gsm8k mbpp alpaca arena8k; do
  case $d in gsm8k) c=ladder.M512.jsonl;; *) c=$d.ladder.M512.jsonl;; esac
  run 0 srv0 /workspace/dflash_sgl $d $d.jsonl $c
done
for d in gsm8k mbpp alpaca arena8k; do
  case $d in gsm8k) c=ladder.M512.jsonl;; *) c=$d.ladder.M512.jsonl;; esac
  run 1 srv1 deepseek-ai/dspark_qwen3_4b_block7 $d $d.jsonl $c
done
echo ""; echo "===== SERVING REWEIGHTING ALL DONE ====="
