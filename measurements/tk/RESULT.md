# $T^{(0)}$ — the realisation-blind acceptance floor, measured

Run `bm21pvj2z` + `rerun_gsm8k`, 2026-08-19. Qwen3-4B target, C0 corpus
(T=1, untruncated), 96 anchors per domain, M=256 free rollouts per anchor,
top-k=256 per position, half-sample split, HT weights $1/\pi$, prompt-level
cluster bootstrap B=10000.

Reproduce:

```
bash measurement_runs/tk/run_t0.sh                       # four domains, rung 0
python -m measurement.tk_report --tk 'measurement_runs/tk/*.t0.jsonl'
```

## What the number is

$$T_k^{(m)} \;=\; \mathbb{E}_{W_m}\Big[\min_q \mathbb{E}_{Z\mid W_m} \operatorname{TV}(p_Z, q)\Big],
\qquad W_m = (X, Z_{k-m:k-1})$$

A drafter on rung $m$ sees only $W_m$; it must commit to a single $q$ before the
target's realisations $Z$ are known. $\operatorname{TV}$ is exactly the rejection
probability of the standard acceptance rule ($\alpha = 1 - \operatorname{TV}$),
so $T_k^{(m)}$ is the per-slot rejection loss that **no** rung-$m$ drafter can
beat, at any capacity, against the free-rollout target. Rung 0 is DFlash
(prompt only); rung 1 is DSpark (`markov_rank > 0`).

Two things this is **not**. It is not an acceptance rate under the serving law —
the serving law conditions on the accepted prefix, and a $\gamma=2$
counterexample with free-rollout $T_2 = 0.5$ but served $\alpha_2 = 1.0$ is in
`calib_20260818/derivations/`. And $T$ is a *floor*, so it is only half of the
observed loss; the drafter-side gap $G_{\text{pre}} = R_{\text{pre}} - T^{(0)}$
needs $R_{\text{pre}} = \mathbb{E}_Z[\operatorname{TV}(p_Z, q_{\text{DFlash}})]$,
which is not measured yet.

## Result, rung 0

Pooled over the four domains (384 anchors, 170 prompts):

| slot $k$ | $T_k^{(0)}$ | 95% CI | curse | resid |
|---|---|---|---|---|
| 0 | 0.0000 | [0.0000, 0.0000] | +0.0000 | 8.4e-06 |
| 1 | 0.0773 | [0.0552, 0.1022] | +0.0008 | 1.7e-05 |
| 2 | 0.1167 | [0.0898, 0.1462] | +0.0028 | 1.2e-05 |
| 3 | 0.1707 | [0.1363, 0.2099] | −0.0000 | 2.4e-05 |
| 4 | 0.2020 | [0.1631, 0.2447] | −0.0008 | 1.9e-05 |
| 5 | 0.2436 | [0.2003, 0.2896] | +0.0032 | 1.9e-05 |

By domain, at the two ends of the block:

| domain | $T_1^{(0)}$ | $T_5^{(0)}$ | 95% CI at $k=5$ |
|---|---|---|---|
| alpaca | 0.0810 | 0.3141 | [0.2443, 0.3855] |
| arena8k | 0.0926 | 0.2571 | [0.1858, 0.3382] |
| mbpp | 0.0391 | 0.2073 | [0.1563, 0.2592] |
| gsm8k | 0.0536 | 0.1560 | [0.1014, 0.2146] |

## Readings

**Slot 0 is exactly zero, in every domain.** Nothing precedes slot 0, so a
realisation-blind proposal forfeits nothing there and the estimator must return
0. It does, to $10^{-16}$. This is the pipeline's arithmetic check, not a
finding.

**The floor grows monotonically with depth, in every domain.** Pooled, it
roughly triples from slot 1 to slot 5. The ceiling on a block drafter degrades
along the block for *informational* reasons — the drafter is blind to more and
more of what the target actually sampled — independently of how much capacity it
has. This is the mechanism the block-length curve has been attributed to
capacity.

**The open/closed split separates the domains at both ends of the block.** The
two open-ended domains sit above the two constrained ones at slot 1 (alpaca
0.081, arena8k 0.093 vs gsm8k 0.054, mbpp 0.039) and again at slot 5 (0.314,
0.257 vs 0.207, 0.156). Where many continuations are valid, which one the target
sampled carries a lot of information, and a blind drafter forfeits it.

The *within-group* ordering is not stable and should not be read as a finding:
mbpp is the lowest domain at slot 1 but gsm8k is the lowest at slot 5, and the
per-domain CIs overlap heavily (at slot 5, gsm8k [0.101, 0.215] against mbpp
[0.156, 0.259]). What survives is the two-group split, not a ranking of four.
Cheap-to-draft and easy-to-solve are different axes, and it is the first one $T$
measures.

**A drafter that is blind at slot 5 cannot exceed 76% acceptance there** (pooled,
$1 - 0.244$), against 92% at slot 1. Any measured rung-0 acceptance below those
lines is capacity; the distance to them is not.

## Gates

* **Winner's curse.** $q^*$ is fitted on the sampled paths and scored on the
  same paths, which biases $T$ down. `T_split` refits on half and scores on the
  other half: the correction is $\le 0.006$ everywhere and $\le 0.003$ in the
  pooled rows, i.e. second-order against $T \approx 0.24$, and it changes sign
  across cells. Reported, not applied.
* **Truncation.** TV is a full-vocabulary distance; mass outside the per-path
  top-256 can only be bounded. Measured residual mass is $1$–$3 \times 10^{-5}$,
  two orders below the $10^{-3}$ gate. 4–8 of 384 anchors per slot exceed the
  gate and are dropped and counted, never averaged in.
* **ESS.** 256.0 = M exactly, as it must be: rung 0 has no importance
  weighting. The column is a no-op here and only becomes informative at rung
  $\ge 1$, where the SNIS weights $p(z^* \mid X, s)$ concentrate.

## Provenance note

`gsm8k.t0.jsonl.CORRUPT-two-writers` is the first pass and is **not** used. A
still-live process from the previous (dense-barycentre) run shared its output
file with the new run: two writers, independent file offsets, records spliced at
overlapping byte ranges. Two lines failed to parse outright and an unknown
number of the survivors are silent splices of two anchors, so the domain was
re-run single-writer rather than filtered. `tk_report` now counts malformed
lines rather than skipping them silently.

## Next

1. $R_{\text{pre}} = \mathbb{E}_Z[\operatorname{TV}(p_Z, q_{\text{DFlash}})]$ —
   load `dflash_sgl` (`markov_rank: 0`) and score the same anchors, to split the
   observed loss into the floor $T^{(0)}$ and the capacity gap $G_{\text{pre}}$.
2. Rung 1 ($T^{(1)}$, SNIS): the DSpark rung. ~12× the cost of rung 0, and the
   first place the ESS gate does any work.
