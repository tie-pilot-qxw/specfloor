# MC convergence calibration — 2026-08-18

The data behind `MAIN_M = 512` in `DeepSpec/measurement/config.py` and
`paper/PROTOCOL.md` §4. Rescued out of the container's `/tmp`, which does not
survive a restart; this represents several GPU-hours and is not cheap to redo.

Target Qwen3-4B, corpus C0 (T=1, top-p=1, top-k off, thinking off), γ=7,
slot 6, 96 prompts per domain, 256-anchor stratified sample.

## What is here

`C0/<domain>.ladder.M<M>.jsonl` — one **fixed-M** arm, escalation disabled
(`--m-base M --m-max M`). Arms are nested by construction: `sample_paths` seeds
request *i* with `seed_base + i`, so M=32's paths are the first 32 of M=64's
(verified 8/8 on this build; depends on `enable_deterministic_inference`).

| domain | corpus file | arms | anchors |
|---|---|---|---|
| gsm8k | `gsm8k.jsonl` | 32 / 64 / 128 / 256 / 512 | 252 (70 prompts) |
| mbpp | `mbpp.jsonl` | 32 / 64 / 128 / 256 / 512 / **1024** | 252 (69 prompts) |
| alpaca | `alpaca.jsonl` | 32 / 64 / 128 / 256 / 512 | 252 (69 prompts) |
| arena-hard-v2 | `arena-hard-v2.jsonl` | 32 / 64 / 128 / 256 | 252 (74 prompts) |
| arena-hard-v2 **8k** | `arena8k.jsonl` | 32 / 64 / 128 / 256 / 512 | 247 (74 prompts) |

**Use `arena8k`, not `arena-hard-v2`.** The latter was generated with
`--max-new-tokens 2048` and came back **14.58% right-censored**, far over the
0.5% acceptance criterion — long answers are cut mid-stream, so its anchor
population is biased toward the early part of long responses. The M comparison
on it is still *internally* valid (all arms share one anchor set) but it does not
represent the domain. `arena8k` regenerated at 8192 and censored **0.00%**.
gsm8k, mbpp and alpaca were all 0.00% at 2048.

`verify.*.json`, `v2.*.json` — the transformers-vs-sglang cross-checks behind
PROTOCOL §7's floor on δ (40 anchors × 7 slots × 16 shared paths; worst
cross-backend |ΔΔCE| = 0.308 nat, 0/280 classification flips).

## Reproducing the verdict

```bash
cd /workspace/DeepSpec
python -m measurement.mc_ladder --arms '<dir>/C0/mbpp.ladder.M*.jsonl' --ref 512
```

Converged M per domain, each verified against the **next rung up** — the top rung
of a ladder can never certify itself:

| domain | converged M | verified vs | worst statistic |
|---|---|---|---|
| gsm8k | 128 | 512 | 0.14 |
| alpaca | 256 | 512 | 0.17 |
| arena8k | 256 | 512 | 0.11 |
| **mbpp** | **512** | 1024 | 0.09 |

Global `MAIN_M = 512`. The requirement is **statistic-dependent**: on mbpp against
M=1024, incidence/p50/p90 are converged at M=128 (0.12 / 0.07 / 0.17) while
`E[dCE]` and `E[dCE|inf]` are not (0.67 / 0.65 at 128, 0.76 / 0.92 at 256).

Do not read a single M vs 2M step in isolation — mbpp's `E[dCE]` runs
1.116 → 0.924 → 0.785 → **0.798** → 0.681 → 0.678, and the off-trend M=256 arm
makes the 256→512 test fail while the sequence is converging.
