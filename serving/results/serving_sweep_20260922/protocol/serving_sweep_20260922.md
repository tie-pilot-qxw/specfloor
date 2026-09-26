# Serving sweep: AR, official DSpark, and ours, one pass

Active supervisor: `refine_exp/eval_out/serving_sweep_20260922_gpu3_clean`.
Its `status.json` links the current `attempt_NNN` run directory.
The preceding `serving_sweep_20260922_gpu6` directory contains a startup/smoke
attempt whose scheduler received SIGKILL before formal measurement; it is excluded.
The GPU 6 retry stayed in the wait phase and ended on an `nvidia-smi` timeout.
The user then reassigned the run to the empty GPU 3; the GPU 6 waiter is stopped.
The first GPU 3 startup overlapped with the user's other job and was stopped by
the exclusivity check before measurement. The user then confirmed GPU 3 for this
run. Startup-only failures can be retried, preserving each attempt separately;
automatic retry stops once any formal measurement has begun.
The earlier `gpu3_auto` attempts also encountered concurrent allocations and were
excluded. After the user cleared the other job, `gpu3_clean` was started. The
near-full allocation check uses the sole compute process's memory, not aggregate
GPU memory, which can lag briefly when another process exits.

The user initially requested AR and official DSpark first, with near-full GPU
memory use, then authorized parallel execution on different empty H100 cards.
The added arms are official DSpark and ours; the existing AR pass continues.
This does not replace the paper's existing single-request table automatically.

Parallel queue and eventual combined comparison:
`refine_exp/eval_out/serving_sweep_20260922_parallel`. Its `status.json` records
GPU assignments and per-arm run directories. The parallel runners import the same
collector and verify its hashes against AR's manifest. The active AR source files
were not modified to implement parallel execution.

If official DSpark starts on another card, the queue stops the original serial
coordinator only after all five AR completion markers exist; `parallel_handoff.json`
documents that intentional stop. The old coordinator may label this SIGTERM as a
failure, but its completed AR points remain valid. If no card frees in time, the
original serial official run remains the fallback. Card assignment is recorded;
cross-card differences are not estimated by this single pass.

## Completion notifications

`refine_exp/monitor_serving_sweep.py` watches all three arms (including the queued
`ours`, which is the attn-head checkpoint with `markov_head_type=attn`) and the
parallel controller. It reports completed arms, failed/exited processes, ten-minute
server-log stalls, and the final comparison through this installation's
`codex queue --thread ... --message ...`. The queue command accepted a labelled
self-test message for the current thread. Monitoring logs and acknowledgements live
under `serving_sweep_20260922_parallel/monitor/`.

This monitor only reads benchmark state and sends messages to the requesting Codex
session; it does not run models, occupy GPUs, or modify benchmark code. It validates
five 3,030-row completion markers before declaring an arm complete, and recognizes
the intentional AR handoff. Its 24-hour limit generates a notification to renew it.
Delivery requires the local Codex session/app server to remain available; accepted
queue submission alone is not proof that a stopped session has resumed.

The self-test was subsequently received by the current Codex conversation after
the sending turn ended; completion and failure events have also resumed it.

## AR C=1 recovery

At 07:41 UTC the original AR coordinator failed because its telemetry thread's
`nvidia-smi` query exceeded 15 seconds. The collector rejected the current task
at its post-measurement guard and the coordinator shut down its own server.
C=4/8/16/32 remain complete. At C=1, only the 500 GSM8K rows and their full-task
elapsed timer were committed; subsequent task data were not saved or accepted.

`refine_exp/resume_ar_serving_sweep.py` retains those GSM8K rows and resumes the
other eight tasks on GPU 3 with the identical server command, prepared requests,
source hashes, two 32-request warmup waves, and per-task wall-time calculation.
Each recovered task is timed wholly within the restarted server; no task timer
spans the restart. This is a resumed point, not an uninterrupted C=1 run or an
additional replicate. Its single-pass precision limits still apply.

Original partial files and failed status are copied under the run's timestamped
`recovery_*` directory. Recovered task records, server snapshots, GPU samples,
runner hash, and process identity are recorded there; `baseline.c1.recovery.json`
links the provenance. The canonical C=1 completion marker is written only after
all nine tasks match the original request identities and reproduce their summaries.
The original server log is retained and appended with a recovery boundary.

Only the recovery telemetry query timeout changes, from 15 to 60 seconds. Query
start/end times are recorded; process-set changes or further query failures still
invalidate the affected task. This does not change the inference or timing code.
The running attn-head collector and its hashed dependencies remain unchanged.

The attn-head coordinator subsequently reported the same 15-second telemetry
timeout and stopped during C=1 as well. Its C=4/8/16/32 and C=1 GSM8K were intact.
`resume_serving_sweep.py` applies the same task-level recovery to that arm on GPU
6. Neither arm accepts the unsaved task that overlapped its telemetry failure.
The previous paragraph's unchanged-source statement refers to the original
collector and runtime; recovery uses a separately recorded coordinator.

The original parallel controller exited as incomplete after the attn-head failure.
`watch_recovered_serving_sweep.py` now observes the existing workers and generates
the same comparison schema once every completion marker is present. It launches
no inference jobs. The original control status and process record are preserved
as `pre_recovery.status.json` and `original.controller.process.json`; the existing
Codex notifier watches the replacement controller through its updated PID record.

## Fixed settings

- GPU 3, H100 80GB. Run only with one compute process sampled on the GPU.
- One independent pass per arm, temperature 0. Official/ours may run concurrently
  on other exclusive H100 cards under the user's updated instruction.
- Concurrency points 1, 4, 8, 16, 32, executed in order 32, 16, 8, 4, 1 for both arms.
- All 3,030 requests across the existing nine tasks. Original seed, selection,
  chat template (thinking disabled), and maximum output length of 2,048 tokens.
  Prepared prompt identities and input token counts were checked against the old AR records.
- `--mem-fraction-static 0.95`, no manually imposed KV-token cap. Each arm gets
  the largest automatically allocated pool under that fraction. Pool capacities
  may differ because model footprints differ; retain actual memory reports.
- BF16, TP1, FA3 default backend, maximum running requests and decode graph batch 32;
  prefill CUDA graphs disabled, chunked prefill 8,192, max prefill tokens 16,384.
- Prefix caching disabled. Flush before warmup and again before measurement at each
  point. Warm up with two waves of 32 requests drawn across all nine tasks, capped
  at 64 generated tokens. Warmup requests do not replace any measured rows.

## Timing and aggregation

The primary throughput is task output tokens divided by exact `perf_counter`
wall time from request dispatch through all completed responses. This includes
server tokenization, prefill, decode, scheduling, and HTTP streaming. Chat formatting
is prepared before timing. Exact seconds are stored in JSON; printed rounded minutes
are never numerical inputs. The macro speedup averages the nine task-level speedups
over AR at the same concurrency. Acceptance is task total output tokens divided by
total verify rounds, then macro-averaged over tasks.

A process-local import hook exports the existing scheduler timestamps for prefill
completion and request completion. Their difference is **scheduler post-prefill
elapsed time**, including scheduling delays and request finalization. It is not
GPU kernel time. The auxiliary per-request decode rate uses N-1 output tokens,
excluding the one prefill output token. At concurrency greater than one, dividing
by the sum of request decode times is a per-request rate, not system throughput.
The hook does not modify installed SGLang files or its forward implementation.

Raw rows also retain first-stream-chunk token counts/times, output hashes, queue
timing, cached token counts, retractions, token counts, and verification counts.
Server snapshots retain the reported decode-batch-size/graph counters, memory
capacity, and full resolved configuration. A reported batch-size histogram is
not the same as CUDA graph padding sizes.

## Validity and operation

Four regression tests check exact elapsed-time aggregation, the prefill-token
subtraction, and rejection of cache reuse/retractions/missing timing or speculation.
The AR smoke request verified live export of scheduler decode time.

During measurement, GPU process snapshots are recorded every two seconds. A changed
process set invalidates the run; errors, retractions, and unexpected cache reuse stop
the coordinator. Sampling cannot rule out interference between samples. Runtime and
collector sources are hashed and checked between points. Completed points have a
separate completion marker; partial JSONL files are not complete results.

The coordinator owns only the servers it starts (or the explicitly recorded initial
AR process). Cleanup uses those process groups, never a port-based process search or
another user's GPU process. When switching arms, it waits for the GPU to be empty
before immediately starting the next server. `status.json` and `runner.log` expose
progress; `comparison.json` is emitted only after both arms finish all points.

One pass is a diagnostic replication. It does not establish the significance of
percent-level differences. Retain existing paper results until the completed new
pass has been reviewed.
