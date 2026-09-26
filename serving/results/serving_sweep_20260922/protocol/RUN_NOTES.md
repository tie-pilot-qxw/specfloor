# Run notes: serving sweep, September 22, 2026

These notes record how the archived sweep was run and which attempts were
excluded. The recovery scripts in this directory are the ones that ran, kept
byte for byte; their absolute paths are those of the original machine.

## Fixed settings

- One H100 80GB per arm, with exactly one compute process on the card during
  measurement. Arms ran in parallel on different cards of one host; each run
  directory name records its card and attempt number.
- One pass per arm. Concurrency points run in the order 32, 16, 8, 4, 1.
- All 3,030 prepared requests over the nine tasks, with the original seed,
  selection, chat template (thinking disabled) and a 2,048-token output cap.
  Prompt identities and input token counts were checked against earlier AR
  records.
- `--mem-fraction-static 0.95` with no manual KV-token cap, so each arm gets
  the largest pool its footprint allows; actual memory reports are retained.
- BF16, TP 1, the default FA3 backend, 32 maximum running requests and decode
  graph batch 32, prefill CUDA graphs disabled, chunked prefill 8,192, maximum
  prefill 16,384 tokens.
- Prefix caching disabled. The cache is flushed before warmup and again before
  measurement at each point. Warmup is two waves of 32 requests drawn across
  all nine tasks, capped at 64 generated tokens; warmup rows are not measured.
- Temperature 1 differs from temperature 0 only in `temperature=1.0` and
  `sampling_seed=980406` (warmup included), with no added top-k or top-p.
  Sampling seeds do not make AR and speculative outputs identical.

## Timing and aggregation

Throughput is a task's output tokens divided by the `perf_counter` wall time
from dispatch of its first request to completion of its last response. It
includes tokenization, prefill, decode, scheduling and HTTP streaming; chat
formatting happens before timing. Exact seconds are stored. Macro speedup
averages the nine task-level speedups over AR at the same concurrency.
Accepted length is a task's output tokens over its verification rounds,
macro-averaged over tasks.

A process-local import hook (`sources/*sitecustomize.py`) exports the
scheduler's existing timestamps for prefill completion and request completion,
without modifying installed SGLang files. Their difference is scheduler
post-prefill elapsed time, including scheduling delays and finalization; it is
not GPU kernel time. The per-request decode rate derived from it uses N-1
output tokens, excluding the token produced by prefill.

Raw rows also keep first-chunk timing, output hashes, queue timing, cached and
retracted token counts, and verification counts. Server snapshots keep the
resolved configuration, memory capacity and decode-batch counters.

## Validity checks during the run

GPU process sets were sampled every two seconds; a change invalidated the task
in progress. Errors, retractions and unexpected cache reuse stopped the arm.
Runtime and collector sources were hashed and checked between points, and a
point counts as complete only with its completion marker; partial JSONL is
never a result. Sampling cannot rule out interference between samples.

## Excluded attempts

Startup-only failures were retried, each attempt kept in its own directory;
no retry was made once formal measurement had begun.

- Temperature-0 AR: an earlier startup on another card was killed before
  measurement, and further attempts met concurrent allocations from other
  processes and were stopped by the exclusivity check. The archived AR run is
  the first attempt on a card confirmed empty.
- Temperature-1 official DSpark: attempts 001 and 002 failed the startup
  exclusivity checks and measured nothing.

## Recoveries

Each recovery keeps the tasks already committed, then re-measures only whole
remaining tasks with the identical server command, prepared requests, warmup
and aggregation, so no task timer spans a server restart. A recovered point is
a resumed pass, not an extra replicate. Each run directory's `*.recovery.json`
and `recovery_*/` record what was kept and what was re-measured.

- Temperature-0 AR, C=1: the telemetry thread's `nvidia-smi` query exceeded
  its 15 s limit and the collector rejected the task in progress. C=4 to C=32
  were complete and C=1 had committed GSM8K. The other eight tasks were
  resumed with `resume_ar_serving_sweep.py`. Only the telemetry timeout
  changed, to 60 s; query times are recorded and any process-set change or
  query failure still invalidates a task.
- Temperature-0 ours, C=1: the same timeout, at the same stage; resumed with
  `resume_serving_sweep.py`.
- Temperature-1 official DSpark, C=1 (attempt 003): after GSM8K completed, an
  additional compute process appeared during MATH-500, which was rejected
  before its rows were committed. GSM8K's last request had finished about 141
  s before the other process was first sampled. The remaining eight tasks were
  resumed on a free card with `resume_serving_temp1.py`. Its first recovery
  server collided with this sweep's own ours-arm startup on the same card and
  both ran out of memory while allocating KV pools, before any measurement;
  the retained GSM8K rows were checked unchanged, and the replacement recovery
  waited until the ours arm had finished.

The temperature-1 sweep first ran C=1 for all three arms; C=32 to C=4 were
added in a second queue that started only after every C=1 arm had its
completion marker. No C=1 request was repeated.
