# Temperature-1, concurrency-1 supplemental pass

The user subsequently extended this to a full C=1/4/8/16/32 sweep. The existing
C=1 queue and recovery keep running. The added queue is
`refine_exp/eval_out/serving_temp1_sweep_20260922`, managed by
`queue_serving_temp1_sweep.py` with independent `watch_temp1_sweep_queue.py`.
It waits until the C=1 queue has all three valid completion markers and its
comparison, then allocates empty GPUs to run each arm at C=32,16,8,4. This avoids
the two generations of queues competing for the same newly empty card. No C=1
request is repeated. The final full-sweep comparison links the original C=1
directories alongside the new four-point directories; it does not relabel or
copy a partial C=1 result as complete.

The added worker `run_serving_temp1_batch_arm.py` imports the same frozen temp=1
collector. Existing active manifests and source hashes remain unchanged. The
dependency gate was checked to prevent GPU queries/launches while C=1 is pending,
and completion requires all four 3,030-row markers per new arm. Startup failures
can retry; failures after formal measurement trigger notification for review.

User requested this after completion of the temperature-0 concurrency sweep.
Run directory: `refine_exp/eval_out/serving_temp1_c1_20260922`.

AR, official DSpark, and the attn-conv step-26160 head each run one pass at C=1.
All three use the frozen T=0 prepared requests (3,030 rows over nine tasks), same
server command except GPU/port/speculation arm, 95% static memory allocation,
and the same warmup, exact task timers, row validation, and aggregation.

The separate `serving_temp1_client.py` differs from the frozen T=0 client only in
temperature, sampling seed, row labels, and its module description. Formal and
warmup requests send temperature=1.0 and sampling_seed=980406, matching the old
`speedup_fulleval.py` temperature-1 request parameters. No additional top-k/top-p
override is introduced; server information snapshots retain generation defaults.
Maximum generation remains 2,048 tokens (64 for warmup). Sampling seeds do not
imply identical sampled text across AR and speculative decoding implementations.

Primary metric remains end-to-end output tokens per exact task elapsed second,
including prefill/HTTP. Task speedup ratios are averaged equally across nine tasks.
Scheduler post-prefill N-1 token rates remain secondary diagnostics, not GPU-only
timing. This is one supplemental pass, not a repeated-trial confidence estimate.

`queue_serving_temp1.py` starts AR first on a free GPU, then official and attn head
as GPUs become free. It never evicts another process. Each server validates one
compute process and at least 74,000 MiB allocated to that process before timing.
GPU monitoring uses the recovered T=0 60-second telemetry timeout and retains
sample timestamps. Failure after any formal output stops that arm for review;
startup failures can retry up to three attempts, preserving attempt directories.

The queue reports starts, completion, failures, stalls, and final comparison via
`codex queue` to the existing user thread. `watch_temp1_queue.py` independently
reports controller exit or a ten-minute stale heartbeat. Both end after all work
is complete. Process identities, logs, manifests/source snapshots, and raw results
are kept under the run directory. Existing temperature-0 and paper results remain
unchanged.

## Official DSpark recovery after GPU interference

The official arm's first two GPU-6 startup attempts failed the near-full/exclusive
checks and produced no formal measurements. Attempt 003 passed startup, completed
GSM8K, then observed an additional compute PID during math500. The guard rejected
math500 before committing its rows. The retained GSM8K's last request finished at
Unix time 1790100969.682; the foreign process was first sampled at 1790101110.643,
about 141 seconds later. Sampling does not rule out interference between samples.

`resume_serving_temp1.py` preserves the 500 GSM8K rows and its exact task timer,
waits for any exclusive free GPU, and resumes only the remaining eight whole tasks
using the same seeded temperature-1 requests and warmup. A new GPU assignment and
the original failed files are recorded in the timestamped recovery directory.
No other user's process is stopped. A further recovery failure triggers review.

The original queue continues running AR and attn head. While it does so, its
official-arm state remains the original failure; `recovery.status.json` is the
combined current view. `watch_temp1_recovery.py` monitors the recovered worker and
sends completion/failure/stall events to the same Codex thread. After all original
unaffected workers have completed and been reaped, it generates the final comparison,
stops only the now-idle original controller by its verified PID, and publishes the
combined complete status. Original startup/measurement failures remain archived.

The first recovery server then collided with this experiment's own attn-head
startup on newly free GPU 3. Both independently selected that GPU and ran out of
memory while allocating KV pools at 19:11 UTC; the resulting SIGKILL/exit -9 was
before formal recovery measurement. This was an internal scheduling race, not
evidence of another user's process killing the server. The retained GSM8K bytes
were checked unchanged. Attn head's subsequent attempt on GPU 3 started measuring.

The replacement official recovery now uses `--wait-for-arm ours --control ...`.
It does not query/select GPUs until the original queue has marked attn head complete
and its valid 3,030-row completion marker exists. The queue sets complete only after
the worker exits, so no attn startup retry can compete with this recovery. Tests
confirmed that a marker alone does not release the dependency. The later full-sweep
queue already waits for all three C=1 arms, preventing a second allocation race.
