# specfloor

Information floors and model gaps for block speculative decoding.

A block drafter proposes γ tokens from a single forward pass, so every slot must
commit to a distribution before the target's realisations at earlier slots exist.
Such drafters lose a large share of proposed tokens to rejection, and two very
different explanations fit the same acceptance rate: the drafter cannot see the
earlier realisations, or it models badly what it can already see. They prescribe
opposite work, and the accepted length everyone reports cannot separate them.

This package separates them by measuring the information half directly.

## The one identity everything rests on

Under the speculative accept rule the probability that a drafted token survives
verification against the target is exactly

```
alpha(p, q) = sum_v min(p(v), q(v)) = 1 - TV(p, q)
```

Rejection loss is therefore a *distance*, and one can ask how small it could
possibly be for a proposal allowed to see only the last `m` realised tokens:

```
T_k^(m) := E_{W_m} [ min_q  E[ TV(p_Z, q) | W_m ] ],     W_m = (X, Z_{k-m}, ..., Z_{k-1})
```

`T^(m)` is the **information floor**. It contains no drafter — it is a property
of the target and the factorisation — and every drafter of that shape must pay
it. What a real drafter loses *above* its own floor is the **model gap**
`G = R - T^(m)`.

The minimisation sits **inside** the outer expectation, so it can be solved
separately at each prefix from ordinary sampled continuations. That is what makes
the floor measurable rather than merely definable.

## Two halves, two dependency sets

|  | needs | modules |
|---|---|---|
| **Floor** `T^(m)` | the target alone | `probe_tk`, `probe_cheap`, `probe_kmedian`, `probe_api_floor`, `probe_rm`, `mc_ladder` |
| **Gap** `G = R − T` | a real drafter too | `probe_rpre`, `probe_br`, `eval_nll` |
| **All post-processing** | nothing but Python | every `*_report`, `stats`, `test_estimators` |

The split is not a packaging accident — it is the same split the measurement
makes. Running a DFlash/DSpark block forward pass is what needs
[DeepSpec](https://github.com/deepseek-ai/DeepSpec); computing a floor does not.

So **23 of the 26 modules import with no DeepSpec at all**, including every
report. The three that need it resolve their symbols lazily through
`specfloor._deepspec`, and a missing install fails with an instruction rather
than a traceback:

```
$ python -m specfloor.probe_rpre ...
This probe measures a DRAFTER's risk R, which requires running a DFlash/DSpark
block forward pass, and that lives in DeepSpec:
    git clone https://github.com/deepseek-ai/DeepSpec
    export SPECFLOOR_DEEPSPEC=/path/to/DeepSpec
Measuring the information floor T^(m) needs none of this ...
```

## Install

```bash
pip install -e .                     # floors, gaps from recorded runs, all reports
export SPECFLOOR_DEEPSPEC=/path/to/DeepSpec    # additionally: measure a drafter's R
export SPECFLOOR_EVAL_ROOT=/path/to/eval_datasets
```

`sglang` is needed only by the probes that sample rollouts locally; the API probe
and every report run without it.

## Start here

Nothing below touches a GPU, and it is the fastest way to see whether the
package is working:

```bash
python -m specfloor.test_estimators
```

It checks the identities the measurements depend on against closed forms and
brute force — that the barycentre solver attains the common-level quantile, that
the K-median hits the exact `(M−K)/M` optimum on point masses, that the water
fill beats a dense grid search, and that `Δτ` equals its decomposition
`E[W·F·Δa]` to `1e-9` at every slot. If a change breaks an estimator, this fails
before any GPU time is spent.

## What the probes measure

| module | quantity | needs a drafter |
|---|---|---|
| `probe_tk` | `T^(0)`, `T^(1)` by importance sampling on a top-K read | no |
| `probe_cheap` | free rollouts, anchors, the CE companion | no |
| `probe_kmedian` | the K-median of the realisation family in TV | no |
| `probe_api_floor` | `T^(m)` on a target behind an HTTP endpoint | no |
| `probe_rm` | the log-loss companion `R_m` | no |
| `probe_rpre` | `R` and `G`, exact TV over the full vocabulary | **yes** |
| `probe_br` | per-path accept factors, for the best-response analysis | **yes** |

Reports are pure post-processing over the recorded `jsonl` and never touch a GPU:
`rpre_report`, `tk_report`, `kmedian_report`, `api_floor_report`, `srv_report`,
`rm_compare`, `br_report`, `br_iter`.

## Two invariants the code will not break

**It does not silently degrade.** A probe that needs a drafter and cannot find
one raises. A cell whose importance weights are too degenerate is dropped *and
counted*, never reported. A truncated read that cannot resolve a cell does not
guess it. Every gate reports the population it removed, because a gate is a
selection rule and an ungated report of a gated population is a different
estimand.

**It does not mix the two laws.** `R` and `T^(m)` average over the target's own
rollouts; accepted length weights slot `k` by the probability of surviving to it,
and the surviving paths are exactly the ones the drafter found easy. Those are
different populations. `srv_report` measures how far apart they are; nothing
converts between them behind your back.

## Reading the numbers

`T^(m)` is an oracle quantity, deliberately. The minimisation is inside the outer
expectation, so it grants a separately chosen optimum at every prefix — the
admissible object is any measurable map from the permitted information to the
simplex. It constrains only *what may be looked at*, never *how well it is used*.

Consequently `G` is a **ceiling on closable headroom, not a forecast**. Writing
`T^arch` for the same minimisation restricted to what a given architecture can
represent,

```
T^(m)  <=  T^arch  <=  R
```

and this package measures the outer two. The slack is not closable by anything in
the class, at any width or any amount of data. The floor direction is untouched
by this: `1 - T^(m)` upper-bounds acceptance for *any* proposal, computable or
not.

## Run order

See [`RUNBOOK.md`](RUNBOOK.md) for the full sequence and the flags each step
needs.
