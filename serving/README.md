# Serving: the prefix-attention drafter in SGLang

This directory reproduces the paper's serving results (Sec. 6 and App.
*Prefix-attention drafter: implementation and evaluation*): accepted length and
speedup of our Qwen3-4B drafter against the released DSpark checkpoint and AR
decoding, the concurrency sweep, the training trajectory and the one-epoch
component comparison.

| Directory | Contents |
|---|---|
| [`sglang/`](sglang/) | The SGLang patch (against `sgl-project/sglang@702de2631`) that serves our drafter, with its tests, install notes and switches |
| [`sweep/`](sweep/) | The serving-sweep collector: request preparation, one-arm runner, recovery, comparison |
| [`eval_accept/`](eval_accept/) | The accepted-length harness behind the training trajectory and the one-epoch components |
| [`draft_forward_probe/`](draft_forward_probe/) | CUDA-event timing of the draft forward alone (not in the paper) |
| [`results/`](results/) | Every serving record behind the paper, and GPU-free scripts that regenerate the tables from them |

To check the paper's numbers, no GPU or model is needed:

```bash
cd results && python summarize_serving_sweep.py && python summarize_accept_evals.py
```

The rest of this page is about producing new measurements.

## Models

| Role | Hugging Face repository | Revision |
|---|---|---|
| Target | `Qwen/Qwen3-4B` | `1cfa9a7208912126459214e8b04321603b3df60c` |
| DSpark baseline | `deepseek-ai/dspark_qwen3_4b_block7` | `3457dff1417cb84927f6098a5fcb7cee85c934b7` |
| Ours (10 epochs, step 26160) | `TIE-Pilot/dspark-attnconv-block7-qwen3-4b` (repository root; `step_*/` hold the epoch checkpoints) | `6e97f0159b527322a95737417c0bf6cb96d39e74` |
| One-epoch components | `TIE-Pilot/deepspec-drafter-ablations`, `<name>/step_2616` | `43549d81df38db655f4cae70a7f345bcc1cda486` |

The served files are byte-identical to these: ours has `model.safetensors`
SHA-256 `4effe8f1…` and `config.json` `751e42e3…`, as recorded in every sweep
manifest; each one-epoch checkpoint's weights and config match the served copy.
Their `config.json` already carries the `dflash_config` block SGLang reads
for the short convolution and slot embeddings. A newly trained checkpoint
needs that block added (the drafter code's `bridge_ckpt_config.py` does it).

| Component row | `<name>` | Served as |
|---|---|---|
| Vanilla DSpark | `dspark_b7_qwen3_4b_1ep` | `DSPARK`, block 7 |
| DFlash2 reproduction | `official_dflash2_b8_qwen3_4b_1ep` | `ALGO=DFLASH`, block 8 (anchor + 7) |
| Vanilla + our short convolution | `dspark_b7_qwen3_4b_1ep_shortconv` | `DSPARK`, block 7 |
| Vanilla + slot embeddings | `slotembed_b7_qwen3_4b` | `DSPARK`, block 7 |
| Prefix-attention head | `attnhead_b7_qwen3_4b` | `DSPARK`, block 7 |
| Head + our convolution + slot embeddings | `attnconv_b7_qwen3_4b` | `DSPARK`, block 7 |

## The serving sweep

One H100 80GB per arm, bf16, seven speculative positions, chain verification
without confidence-based early stopping, 3,030 rows over nine tasks
(GSM8K 500, MATH-500 500, AIME25 30, HumanEval 164, MBPP 256, LiveCodeBench
500, MT-Bench 80, Alpaca 500, Arena-Hard-v2 500), maximum 2,048 new tokens.

```bash
cd sweep
python prepare_requests.py --eval-datasets <DeepSpec>/eval_datasets --out prepared_requests.jsonl
for T in 0 1; do
  python run_arm.py --prepared prepared_requests.jsonl --out runs/t$T-ar       --arm baseline --temperature $T --gpu 0 --port 30487
  python run_arm.py --prepared prepared_requests.jsonl --out runs/t$T-official --arm official --temperature $T --gpu 1 --port 30488 --checkpoint <dspark_qwen3_4b_block7>
  python run_arm.py --prepared prepared_requests.jsonl --out runs/t$T-ours     --arm ours     --temperature $T --gpu 2 --port 30489 --checkpoint <ours>
  python compare_runs.py --baseline runs/t$T-ar --official runs/t$T-official --ours runs/t$T-ours
done
```

`prepare_requests.py` reads DeepSpec's `eval_datasets/` (unchanged since
upstream DeepSpec `afdfa7c`), selects rows with DeepSpec's seeded shuffle
(seed 980406), applies the Qwen3-4B chat template with thinking disabled, and
refuses to write anything but the archived set (SHA-256 `b5efd513…`).

`run_arm.py` launches the server with the archived flags and environment
(`sweep/common.py`), waits until the GPU is empty, requires the server to own
at least 74 GB of it, samples the GPU's process set every two seconds, and
aborts on any change. Each point runs two 64-token warmup waves, then every
task in order under one timer per task, highest concurrency first. Output files
follow the archived layout, so `results/summarize_serving_sweep.py` checks
apply to them. An interrupted point resumes at task boundaries with
`resume_point.py`. `--shared-gpu`, `--tasks` and `--limit` run a functional
smoke test on a busy GPU; such output is marked in its manifest and is not a
timing run.

The archived collector is `results/serving_sweep_20260922/sources/`; the
scripts here are it with paths turned into arguments and the two temperature
variants merged. For the same row they send the byte-identical request body,
build the identical launch command for all nine archived runs, and produce
identical per-task summaries from the archived rows.

## Trajectory and component evaluations

```bash
cd eval_accept
export PY=<python with patched SGLang> EVAL_DATASETS=<DeepSpec>/eval_datasets
# one-epoch components (temperature 0, 430 rows)
./eval_arm.sh <ablations>/attnconv_b7_qwen3_4b/step_2616 out/lat_attnconv.json 0
ALGO=DFLASH ./eval_arm.sh <ablations>/official_dflash2_b8_qwen3_4b_1ep/step_2616 out/lattice_dflash2_repro.json 0
# trajectory (temperature 1, 3,030 rows): each epoch checkpoint, and the DSpark baseline
TEMP=1.0 CAP=500 OFFICIAL_SUBSET=1 SEED=980406 ./eval_arm.sh <ours>/step_2616 out/paper_10ep_ep1.json 0
```

Accepted length here is batch-invariant in expectation, so requests run 32 at a
time; timing must come from the sweep. These evaluations ran on an earlier
state of the runtime branch without the short-conv fusion (see
[`sglang/README.md`](sglang/README.md#revisions)). Re-running the
head + convolution + slot-embedding component on the current patch gave 4.789
against the archived 4.766, on identical prompts. At temperature 0 and
concurrency 32 greedy outputs are not batch-invariant, so 146 of 430 texts
matched and accepted length moves by a few hundredths between runs; the
slot-embedding row (−0.01) is within that range.

## Reproducibility notes

- **Temperature 0** is reproducible request by request at C=1. A smoke run of
  this patch (`run_arm.py --shared-gpu`, 12 archived rows from GSM8K,
  HumanEval and MT-Bench) matched the archived output text, token count and
  verification steps on 12/12 rows for ours and 12/12 for DSpark.
- **Temperature 1** is reproducible in distribution only. The target samples
  with a per-request seed, but the drafter's lattice walk (and upstream DSpark's
  sampler) draw uniforms from the process-wide CUDA generator, so the accepted
  path depends on everything the server ran before. The same smoke run at
  temperature 1 matched 0/12 texts, with pooled τ 4.31 against 4.45 over those
  12 rows.
- **Timing** requires an exclusive H100 80GB and includes prefill and HTTP
  streaming; arms of the archived sweep ran in parallel on different cards of
  one host. One pass per point does not establish significance for
  percent-level differences.
- `SGLANG_DFLASH_FUSE_CONV=1` is required to reproduce the sweep and is set
  by `sweep/common.py`; the runtime default is 0.

## Draft-forward probe

```bash
cd draft_forward_probe
python run_draft_forward_probe.py --gpu 0 --out probe_out --python <python with patched SGLang> \
    --official-checkpoint <dspark_qwen3_4b_block7> --ours-checkpoint <ours>
```

It times the actual SGLang draft forward (and the full proposal) with CUDA
events, 64 graph replays after 12 warmups per shape, at batch sizes 1–32 and
fixed 512/1024-token prefixes, excluding target verification, KV commit,
prefill and HTTP. The archived run is `results/draft_forward_sweep/`; it is a
diagnostic and is not reported in the paper.
