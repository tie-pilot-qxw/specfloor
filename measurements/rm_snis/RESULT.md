# R_m: conditional vs interventional — full run, 2026-08-19

Four domains, C0, γ=7, slot 6 selection at ε_R=0.05, 64 mixed paths, ESS gate ≥32/64,
ratio of means, 4 000-sample bootstrap over **prompts**.

**698 anchors / 221 prompts.** Every cell has ~200 clusters, comfortably above the
protocol's ≥30. These numbers are quotable; the earlier 21-anchor pilot was not.

## Pooled

| m | R_m conditional (SNIS) | R_m interventional (all earlier runs) | shift (pp) | ESS med | cells dropped |
|---|---|---|---|---|---|
| 1 | **89.5%** [84.5, 93.9] | 76.2% [72.6, 79.9] | **−13.3** [−16.6, −9.9] | 53 | 43% |
| 2 | **98.8%** [98.2, 99.3] | 94.3% [93.0, 95.4] | **−4.5** [−5.6, −3.6] | 57 | 37% |
| 4 | **99.9%** [99.8, 100.0] | 99.2% [98.8, 99.4] | **−0.7** [−1.1, −0.5] | 63 | 22% |

## By domain (conditional)

| domain | anchors / prompts | R₁ | R₂ | R₄ |
|---|---|---|---|---|
| gsm8k | 160 / 50 | 94.0% [90.4, 97.6] | 99.6% [98.5, 100.2] | 99.9% |
| mbpp | 166 / 55 | 95.1% [93.7, 97.8] | 98.7% [98.2, 99.7] | 99.9% |
| alpaca | 224 / 56 | 83.5% [76.2, 92.8] | 98.2% [96.9, 99.3] | 99.8% |
| arena-hard-v2 | 148 / 60 | 90.7% [81.8, 98.3] | 99.2% [98.2, 99.8] | 100.0% |

Chat is the hard case at order 1 (alpaca 83.5%) and closes by order 2, which is the
same domain ordering the old estimator showed — the correction moves the level, not
the ranking.

## Three internal consistency checks it passes

1. **The shift is negative everywhere and its CI excludes zero** at m=1 and m=2 in all
   four domains. Only gsm8k m=4 includes zero ([−2.2, +0.1]), which is where the
   correction should vanish.
2. **|shift| decreases monotonically in m** (13.3 → 4.5 → 0.7 pooled). It must: the
   correction reweights the segment *before* the reveal, and that segment shrinks as m
   grows. At m=k it is empty and the estimators coincide identically.
3. **ESS *increases* with m** (53 → 57 → 63) and the drop rate falls (43% → 37% → 22%),
   for the same reason: a shorter early segment means less spread in the weights. This
   is the opposite of grouping, which starves as m grows.

## What it changes

- R₁ was previously reported as erratic and sometimes negative and blamed on a small
  denominator. **It was mostly the estimator.** Order-1 alone recovers ~90%.
- The old "order-2 recovers 80–100%" becomes **order-2 recovers 98.8% [98.2, 99.3]**.
- The shipped DSpark head is `vanilla` = order 1. So the multilag / windowmix arms of
  the capacity study were competing for the **10.5%** that order-1 leaves — of a
  quantity whose median is 0. That is the most likely explanation of their ±0.00.

## Caveat that remains

43% of eligible cells are dropped by the ESS gate at m=1. The estimator is unbiased on
what survives, but survival is not random — it favours anchors whose revealed token is
probable under most sampled prefixes. Re-run at 128 mixed paths to check the gate is
not selecting a biased subpopulation before this goes in a paper.

## Reproduce

```bash
python -m measurement.probe_rm --corpus C0 --corpus-file <C0>/<dom>.jsonl \
    --cheap <C0>/<dom>.ladder.M512.jsonl --out <dom>.rm.jsonl \
    --budget 256 --mixed-paths 64
python -m measurement.rm_compare --rm '<dir>/*.rm.jsonl' --ess 32 --by-domain
```
