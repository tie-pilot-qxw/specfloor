# Completed serving sweep

All 15 concurrency points passed raw-row, input-identity, exact task-summary and comparison checks (45,450 measured rows).

Primary metric: task output tokens / exact request wall time, including prefill and HTTP. Each speedup is calculated within each of nine tasks, then averaged equally. C is request concurrency, not a fixed physical decode batch.

| Concurrency | Official / AR | Attn head / AR | Attn head gain over official |
|---:|---:|---:|---:|
| 1 | 3.616x | 3.757x | +3.99% |
| 4 | 3.415x | 3.536x | +3.65% |
| 8 | 3.141x | 3.301x | +5.10% |
| 16 | 2.896x | 3.032x | +4.77% |
| 32 | 3.112x | 3.209x | +3.35% |

The direct attn/official column averages task ratios; it is not the quotient of the two macro AR speedups.

Acceptance stays around 5.16–5.18 for official DSpark and 5.31–5.33 for attn head. The mean advantage over official persists across the sweep; it does not imply every task is faster (LiveCodeBench at C=32: attn/official 0.96184).

One pass per arm. AR uses GPU 3; official and attn head use GPU 6. AR and attn C=1 resumed at whole-task boundaries after telemetry query timeouts; retained GSM8K rows and timers are byte-for-byte/numerically unchanged. Each other task has a fresh complete task timer after restart, with original warmup and inference settings. No completed point was rerun. This is not an uninterrupted C=1 run. Percent-level statistical significance is not established.

The prior paper single-request table is unchanged. Full per-task values: comparison.json. Validation/provenance: final_validation.json and each arm’s C=1 recovery records.
