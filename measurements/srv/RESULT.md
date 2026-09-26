# Serving reweighting — what the free-rollout law costs, and whether τ factorises

Qwen3-4B, C0, γ=7, four domains × 96 anchors = 384 anchors over 170 prompts,
M=256 paths, full vocabulary, exact TV. Two drafters:
`deepseek-ai/dflash_qwen3_4b_block7` (order 0) and
`deepseek-ai/dspark_qwen3_4b_block7` (order 1, oracle-conditioned).

```
bash measurement_runs/srv/run.sh          # four domains, 3 GiB budget
bash measurement_runs/srv/rerun_arena.sh  # arena8k redrawn: order 1 at 8 GiB, order 0 at 12
python -m measurement.srv_report --rpre 'measurement_runs/srv/*.srv0.jsonl' --by-domain
python -m measurement.srv_report --rpre 'measurement_runs/srv/*.srv1.jsonl' --by-domain
```

## The one new recorded quantity

A path that realised `Z` is accepted at slot `i` with probability
`a_i = min(1, q_i(Z_i)/p_i(Z_i))`, and conditional on the path the accept coins
are independent across slots. So `W_{k-1} = prod_{i<k} a_i` is that path's
probability of REACHING slot k. Everything else follows from recording it:

* `R_serve = E[W TV] / E[W]` — the same TV on the same paths, reweighted into
  the population a serving stack actually visits.
* `S_j = E[W_j]` — `P(J > j)` directly, which `prod_i (1 - R_i)` only approximates.

Both weightings run on the same paths, so the difference is paired within anchor
and is bootstrapped as one quantity (prompt-level cluster bootstrap, B=10000).

**Validation.** `E_mu[a_k] = 1 - R_k` identically, and the two sides come from
disjoint code: a min-ratio at the drawn token against a total variation over the
full simplex. Pooled they agree to at most 0.0015 in magnitude at every slot on
both drafters; per-anchor discrepancies sit on the M^(-1/2) ≈ 0.06 scale.

## Result 1: the free-rollout risk is HIGH, by one to six points

Pooled, four domains:

| k | DFlash R_free | R_serve | diff | 95% CI | DSpark R_free | R_serve | diff | 95% CI |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.2384 | 0.2175 | −0.0209 | [−.032, −.010] | 0.1367 | 0.1212 | −0.0155 | [−.034, −.004] |
| 2 | 0.3457 | 0.3114 | −0.0343 | [−.046, −.024] | 0.2063 | 0.1812 | −0.0251 | [−.045, −.011] |
| 3 | 0.4266 | 0.3719 | −0.0547 | [−.071, −.040] | 0.2673 | 0.2450 | −0.0223 | [−.050, +.007] |
| 4 | 0.4973 | 0.4567 | −0.0406 | [−.059, −.024] | 0.2854 | 0.2714 | −0.0140 | [−.038, +.013] |
| 5 | 0.5689 | 0.5261 | −0.0428 | [−.068, −.017] | 0.3453 | 0.3008 | −0.0446 | [−.065, −.026] |
| 6 | 0.6353 | 0.5835 | −0.0518 | [−.076, −.032] | 0.3658 | 0.3508 | −0.0150 | [−.039, +.009] |

The sign is what survival predicts — the trajectories a drafter finds hard are
preferentially the ones that already ended the block. Intervals exclude zero at
all six slots for DFlash and at three of six for DSpark. The size is the useful
part: the free-rollout number is a **conservative proxy**, not a different
quantity, so the floors and gaps measured under μ understate the drafter's
serving position rather than mis-specifying it.

The correction is larger for the weaker drafter, in every domain separately,
which is the mechanism working: DFlash rejects more, so survival selects harder,
so the surviving population sits further from μ. DSpark accepts more and its
surviving population is closer to μ. Mean |diff| over slots 1–6, DFlash against
DSpark: alpaca 0.042/0.017, arena8k 0.033/0.024, gsm8k 0.065/0.013,
mbpp 0.045/0.038, pooled 0.041/0.023.

## Result 2: τ does not factorise, and the direction is the safe one

`E[prod a_i] / prod E[a_i]`, pooled:

| slot | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| DFlash | 1.000 | 1.093 | 1.337 | 1.945 | 3.133 | 5.692 | **12.33** |
| DSpark | 1.000 | 1.039 | 1.110 | 1.310 | 1.580 | 1.986 | **2.63** |

It exceeds 1 at every slot and grows with depth, so multiplying per-slot means
badly understates survival. This is exactly the non-negative dependence condition
an accepted length has to be inverted through (paper §5.9), now measured rather
than assumed. In accepted length, pooled:

| | DFlash | DSpark |
|---|---|---|
| τ from E[prod a_i] | **4.574** | **5.232** |
| τ from prod E[a_i]  | 3.397 | 4.386 |

The domain spread is wide and ordered the way the risk is: DFlash τ runs 3.15
(alpaca) to 6.21 (gsm8k), and its last-slot dependence ratio runs 3.1 (gsm8k) to
50.8 (alpaca) — where per-slot risk is high, survival concentrates on a few easy
trajectories and the factorised estimate is worst.

These are C0 (T=1) numbers on this paper's four-domain mix and are not comparable
to a temperature-0 benchmark figure.

## What is NOT here

No serving-conditioned floor. The population reaching slot k is ν_k(q), a
function of the proposal being optimised, so `min_q E_{ν_k(q)}[TV]` is solved by
rejecting hard trajectories early. `R_serve` is well defined only because q is
fixed — it reweights a given drafter's risk, it does not optimise anything. The
serving-native object is `max_q E_q[τ]`, a sequential decision problem whose
local objective carries a continuation value, and it is out of scope.

## Provenance

Chunk size is a deterministic function of the KV budget and the sampler consumes
its stream per chunk, so these are a **different draw from the same law** than
`measurement_runs/rpre/` on the same anchors — chunk 26 against 34 on gsm8k.
Pooled T and R agree with that run to ≤ 0.0044, and the order-1 decomposition
reproduces it quantity by quantity to ≤ 0.004 with the three-term identity
closing exactly; that is the replication reported in the paper's §5.5. Nothing
here depends on it: R_free and R_serve are the same TV on the same paths under
two weightings, so their difference is paired regardless of which draw it is.

arena8k was redrawn because at 3 GiB it lost exactly its 15 longest contexts —
a length-dependent cut on the long-context domain, not a random loss. Its 3 GiB
output is discarded and is not used anywhere above.
