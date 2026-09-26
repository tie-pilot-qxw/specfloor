# Temperature-1 full serving sweep: complete

All 15 points passed raw-row, prompt-identity, sampling-parameter, exact task-summary and comparison validation: 45,450 measured rows. C=1 is reused from the original queue, without rerunning completed tasks.

Primary metric is end-to-end task output throughput, including prefill, scheduling and HTTP. Ratios are calculated per task, then averaged equally across nine tasks. C is request concurrency, not fixed physical decode batch.

| C | Official / AR | Attn head / AR | Attn gain over official | Official acceptance | Attn acceptance |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.339x | 3.518x | +5.40% | 4.748 | 4.941 |
| 4 | 3.133x | 3.222x | +3.08% | 4.761 | 4.937 |
| 8 | 2.977x | 3.098x | +4.38% | 4.756 | 4.932 |
| 16 | 2.689x | 2.838x | +5.95% | 4.743 | 4.933 |
| 32 | 2.695x | 2.831x | +5.27% | 4.758 | 4.956 |

C=1 scheduler decode-only diagnostic: official / AR 3.463x; attn / AR 3.661x; mean task attn/official improvement 5.74%. This excludes prefill and its first output token, but includes decode scheduling and finalization; it is not CUDA-event GPU kernel timing. At C>1 the sum of per-request decode durations must not be called system elapsed time.

Acceptance is task total output tokens divided by verify rounds, then averaged across tasks; it includes the extra token advanced by a verify round. It stays approximately 4.74–4.76 for official and 4.93–4.96 for attn across concurrency.

Relative to AR, speedup decreases up to C=16, then is nearly flat at C=32. Attn remains faster than official on the task-average measure across all five points; individual task ratios are in comparison.json. A single pass cannot establish statistical significance of percent-level differences.

Protocol: temperature=1, sampling_seed=980406, 3,030 requests per point, max_new_tokens=2048, 95% static memory configuration, two 32-request warmup waves excluded from measurement. Exact seconds, not rounded printed minutes, determine rates.

AR C=1 ran on GPU 2; attn C=1 on GPU 3. Official C=1 preserved GSM8K from GPU 6 and resumed the other eight whole tasks on GPU 0. GPU interference and an internal startup allocation race were logged; failed/unsaved tasks and startup attempts are excluded. Each retained task has a complete timer within one server session. Additional C=32/16/8/4 arms ran on GPUs 0 (AR), 2 (official), and 3 (attn). GPU sampling cannot exclude interference between samples.

The prior paper single-request results remain unchanged. Source paths, raw results and recovery provenance are retained; final_validation.json records the hashes and checks.
