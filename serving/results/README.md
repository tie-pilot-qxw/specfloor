# Serving records and the scripts that turn them into the paper's tables

Everything here runs without a GPU. Each script validates the raw records
before aggregating them and writes to `generated/`, which is committed so the
outputs can be read without running anything.

```bash
python summarize_serving_sweep.py         # sweep, main and per-task serving tables
python summarize_accept_evals.py          # training trajectory and one-epoch components
python summarize_draft_forward_sweep.py   # draft-forward latency (not in the paper)
python summarize_historical_20260915.py   # superseded single-request runs (not in the paper)
```

| Paper item | Generated file | Records |
|---|---|---|
| Table *serving sweep* (`tab:serving-sweep`), and the 3.35–5.10% / 3.08–5.95% gain ranges | `serving_sweep.tex` | `serving_sweep_20260922/` |
| Main serving table (`tab:solution-serving`) | `solution_serving.tex` | `serving_sweep_20260922/` |
| Per-task serving table (`tab:solution-serving-details`) | `solution_serving_details.tex` | `serving_sweep_20260922/` |
| Abstract: +3.93% mean accepted length | `acceptance_summary.json` (temperature 1, `ours / official − 1`) | `serving_sweep_20260922/` |
| Training trajectory (`tab:solution-trajectory`) | `solution_trajectory.tex`, `accept_evals.json` | `accept_evals/paper_10ep_*` |
| One-epoch components (`tab:solution-components`) | `solution_components.tex`, `accept_evals.json` | `accept_evals/lat_*`, `lattice_dflash2_repro` |

Every row of these generated tables appears verbatim in the paper source.
`serving_sweep.tex` equals the paper's table except for its generator comment.

## `serving_sweep_20260922/`: the serving sweep

AR, the released DSpark checkpoint and ours at temperatures 0 and 1 and
concurrency C ∈ {1, 4, 8, 16, 32} on the same 3,030 prompts: 30 points and
90,900 request records. Raw JSONL is gzip-compressed losslessly; `archive.json`
maps every point to its rows, summary, completion marker and source run, and
maps the original source hashes to the snapshots in `sources/`, which are the
exact collector and runtime files that ran. `protocol/` holds the run notes and
the recovery scripts used at the time; `../sweep/` is the parametrised form of
the same collector. Five of the snapshots are SGLang files (Apache-2.0):
`scheduler.py`, `req_time_stats.py` and `batch_result_processor.py` are
unmodified upstream `702de2631`, and `dspark.py` and `dspark_draft.py` are the
modified versions that `../sglang/sglang-dspark.patch` produces (before the
removals listed there). They are kept byte for byte because their hashes are
the provenance record.

`summarize_serving_sweep.py` checks, for every row: prompt identity against
`prepared_requests.jsonl.gz`, prompt token count, sampling settings, finish
reason, no retraction or cache reuse, and that speculation was active. It then
recomputes each per-task summary from the rows, checks the recorded
comparisons and source hashes, and regenerates the tables.

**Units.** Within task *d* at concurrency *C*, accepted length is total output
tokens over total verification steps (the bonus token included), and
throughput is total output tokens over the task's end-to-end elapsed time
(including tokenization, prefill and HTTP streaming). Speedup divides by AR's
throughput for the same task and C. Macro values average the nine tasks
equally. The main table's τ averages each task over the five concurrencies,
then the nine tasks; its S is the C=1 point. The five concurrencies are
different conditions on the same prompts, not independent repetitions, and each
point is one pass, which does not establish significance for percent-level
differences.

**Recoveries.** The temperature-0 AR and ours C=1 points stopped after GSM8K
on a 15 s `nvidia-smi` telemetry timeout, and the temperature-1 official C=1
point stopped after GSM8K in an earlier attempt. All three were resumed at
whole-task boundaries, so no task timer spans a server restart; each
`*.recovery.json` and `recovery_*/` directory records what was kept and
re-measured, and `diagnostic_excerpts.log.txt` records the excluded failed
attempts. Arms ran in parallel on different H100 cards of one host;
`*.gpu_ready.json` and `*.gpu_samples.jsonl.gz` record exclusivity for every
point.

## `accept_evals/`: trajectory and component evaluations

Per-request accepted length from `../eval_accept/sglang_paired_accept.py`,
recorded September 1–9, 2026, before the serving optimizations.

- `paper_official` and `paper_10ep_ep{1..9}` / `paper_10ep_final`: temperature 1,
  seed 980406, the 3,030 rows (DeepSpec's seeded subset, `OFFICIAL_SUBSET=1`),
  at the ten epoch checkpoints of the final run.
- `lat_*` and `lattice_dflash2_repro`: temperature 0, the first 50 rows of each
  task (all 30 of AIME25), 430 rows, one-epoch checkpoints.

`manifest.json` records each file's raw SHA-256, the checkpoint served, the
speculative settings from the server log, and the head the server built.

## `draft_forward_sweep/`: draft-forward latency (not in the paper)

Median of 64 CUDA-event samples of the actual draft forward (folded proposal
head included) at batch sizes 1–32 and fixed 512/1024-token prefixes, one
launch per arm on one H100. Target verification, KV commit, prefill and HTTP
are excluded. This measures draft cost, not end-to-end speedup.
`../draft_forward_probe/` produced it.

## `historical_20260915/`: superseded single-request runs (not in the paper)

The sequential temperature-0/1 timings and the three-launch temperature-1
acceptance that the paper reported before the sweep replaced them, on runtime
`8f51a8a`. `manifest.json` maps each file to its original path and raw SHA-256.
The summarizer reproduces the earlier `solution_extensions.json` values to
within floating-point summation order (≤ 1e-15).
