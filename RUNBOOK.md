# Runbook

The run order, the flags each step needs, and the reasons behind the choices
that are not obvious. For what the quantities *are*, see `README.md`.

Two steps gate everything else and neither needs a GPU: `test_estimators`
checks the estimator identities against closed forms, and `mc_ladder` is what
justifies the path count. Run both before spending GPU time.

## Conventions, applied everywhere

- **Rollout policy = the corpus's own policy.** C0 rolls out at T=1 untruncated,
  so paths, labels and the probabilities being read all come from `p` and CE is a
  genuine conditional-entropy estimate. C1 rolls out under the deployment recipe,
  which is what makes `CE_B` comparable to the drafter's NLL on the same anchors.
- **Probabilities are always read from the raw head** — no temperature, no
  truncation. We measure `p`, not the sampling process.
- `CE_B` is **log of the mean**; `CE_commit` is **mean of the log**. The Jensen
  gap between them is across-path heterogeneity of the predictive, *not* a chain
  drafter's regret.
- `H*_D ≥ CE_B`, because the drafter sees the prefix only through five projected
  target layers. So the decomposition reports **bounds**: `G_info ≥ ΔCE`,
  `G_model ≤ L_D − CE_B`, with `G_info + G_model = L_D − CE_A` exact.

## Files

Grouped by what they need, because that is the split that matters: measuring a
floor needs the target alone, measuring a gap needs a drafter as well.

**Needs nothing but Python** — pure post-processing over recorded `jsonl`.

| | |
|---|---|
| `config.py` | every frozen constant; no probe takes its own default |
| `stats.py` | hierarchical bootstrap (outer unit = prompt), weighted aggregation, decomposition bounds |
| `anchors.py` | eligible population → strata → weighted sample with `pi` |
| `test_estimators.py` | estimator identities against closed forms; gates the run, no GPU |
| `mc_ladder.py` | the M-convergence evidence for the path count |
| `rpre_report.py` | `T`, `R`, `G` per slot, paired bootstrap |
| `tk_report.py` | `T^(0)` and `T^(1)` from the importance-sampling arm |
| `kmedian_report.py` | the K-median decomposition of the floor |
| `api_floor_report.py` | the floor measured through an endpoint |
| `srv_report.py` | free-rollout vs survival-weighted risk, and `E[prod a]` |
| `rm_compare.py` | the log-loss companion, and the gate-relaxation check |
| `br_report.py` | single-slot best response, cross-fitted |
| `br_iter.py` | the same swept to a fixed point; both sweep orders |

**Needs the target** — samples rollouts or calls an endpoint.

| | backend |
|---|---|
| `backend.py` | the only place the sglang logprob API is interpreted | sglang |
| `backend_api.py` | remote endpoint, same interface; probes what it can actually do | HTTP |
| `corpus.py` | C0/C1/C2 generation to natural EOS, censoring flagged | sglang |
| `verify_backend.py` | cross-checks sglang against transformers; gates the run | both |
| `probe_cheap.py` | free rollouts, `CE_A`, `CE_B`, `ΔCE`; frozen M ladder | sglang |
| `probe_tk.py` | `T^(0)` and `T^(1)` by importance sampling on a top-K read | sglang |
| `probe_kmedian.py` | the K-median of the realisation family in TV | sglang |
| `probe_rm.py` | the log-loss companion `R_m` on the informative subset | sglang |
| `probe_api_floor.py` | `T^(m)` on a target reachable only over HTTP | HTTP |

**Needs a drafter too** — the only three that reach for DeepSpec, and they do it
lazily through `_deepspec.py`, so the rest of the package imports without it.

| | backend |
|---|---|
| `probe_rpre.py` | `R` and `G`, exact TV over the full vocabulary | transformers + DeepSpec |
| `probe_br.py` | per-path accept factors for the best-response analysis | transformers + DeepSpec |
| `eval_nll.py` | `L_D = −log q_D(y_realised)`; no decay, no aux, no mixture | transformers + DeepSpec |

## Two gates before any main run

```bash
python -m specfloor.test_estimators                       # no GPU, seconds
python -m specfloor.verify_backend --phase hf  --corpus-file <C1 corpus> --out /tmp/v
python -m specfloor.verify_backend --phase sgl --corpus-file <C1 corpus> --out /tmp/v
python -m specfloor.verify_backend --phase cmp --out /tmp/v
```

Run the backend check on **C1**, not only C0. C1 is temp-0.7/top-p-0.8/top-k-20, which is where a
temperature-contaminated read would show up; a C0-only check passes while the real run is wrong.

`test_estimators.py` exists because the jackknife bias correction once returned a *negative*
cross-entropy — impossible by definition — and nothing downstream noticed until a population mean
came out absurd three stages later.

## Why `MAIN_M = 128`, and why the ladder is not optional

`MAIN_M` was first guessed at 64. The ladder rejected it: on C0/gsm8k (252 anchors / 70 prompts,
slot 6) the shift to `M=256`, measured in each statistic's own sampling-CI half-widths, was

| | from M=64 | from M=128 |
|---|---|---|
| `P(dCE>eps)` | 0.27 | 0.13 |
| `E[dCE]` | 0.16 | 0.01 |
| `E[dCE \| inf]` | 0.34 | 0.08 |
| `p90 \| inf` | **0.75** | **0.00** |

M=64 is systematically low on precisely the conditional and tail statistics the heavy-tail story
rests on. This is a *bias*, not noise: `-log(mean p)` carries a finite-M bias of ≈ CV²/2M, and the
anchors that carry the story are the high-CV ones. **Boundary escalation cannot fix it** — those
anchors are not near the threshold, they are far above it with a badly estimated value. That is why
the ladder is the primary evidence and not a formality.

Two operational rules fall out:

- **`--max-running 64` at M ≥ 256.** An unbounded fan-out killed the engine (SIGQUIT, child died)
  and left a **0-byte output file** behind. `probe_cheap` now hard-fails when the record count does
  not match, and `mc_ladder` refuses to run with an empty arm rather than silently comparing the
  survivors — a crashed run that still produces output is more dangerous than one that produces none.
- **M=32 must never be reported**, not even as exploratory: `E[dCE]` came out 0.947 against a
  converged 0.695, a 36% error.

## Running against a remote endpoint

For targets too large to host locally. Probe first — never infer capability from the URL, since a
host that looks like sglang may be a proxy:

```bash
python -m specfloor.backend_api --api-base <url> --api-model <name>
```

| tier | endpoint exposes | what works |
|---|---|---|
| **A** | native `/generate` + `token_ids_logprob` | everything, code unchanged |
| **B** | `/v1/completions` with `echo` + `logprobs` over token ids | `CE_A` and corpus only |
| **C** | chat, `top_logprobs ≤ 20`, no echo | corpus generation only |

Tier C is **refused by default**, and the reason is the measurement rather than the code.
`p(y_k)` is observable only when `y_k` is in the top 20 — but the anchors that carry the whole
result are exactly the ones where the realized token is improbable under a wrong path, i.e. *not*
in the top 20. The censoring is correlated with the estimand. It can be made rigorous as interval
bounds (`p ∈ [0, p_20]`), but that turns every downstream quantity into an interval and would mean
rewriting the bootstrap, the incidence definition and the aggregation around interval arithmetic.
Not attempted. Independently, without `echo` nothing can be scored in place, so `CE_B` costs one
request per (path, slot) — ~900 per anchor at M=128, ~2.2M per domain.

**The supported route for a large target is to rent a GPU host and run `sglang.launch_server` on
it** (Tier A). The probe distinguishes an *unreachable* endpoint from an *incapable* one and
refuses to report a tier for the former — otherwise a typo reads as "your API cannot do this".

## Why sglang for the target, transformers for the drafter

Every probe needs `M` sampled continuations of the **same** prefix. Under
transformers that is `M` independent KV caches — on Qwen3-4B, 144 KiB per token
per path, so `M=128` over an 8k prefix is **141 GiB**. The old code worked around
this by chunking the path batch and re-prefilling the prefix per chunk. sglang's
RadixAttention stores the shared prefix once, so the same job is
`8000 + 128·7 ≈ 8.9k` token-slots ≈ **1.25 GiB** — it fits in the scraps of a
shared GPU, and the second (scoring) pass is nearly free because the prefix is
already resident.

`eval_nll.py` stays on transformers and cannot move: it calls
`draft._forward_backbone`, `compute_logits` and `markov_head.apply_block_logits`
directly and builds `create_dspark_attention_mask` itself — all below the level
sglang exposes.

### Three sglang behaviours the probes depend on

These are not defaults; getting any of them wrong is silent, not loud.

1. **Raw probabilities.** sglang's decode path does
   `logits.div_(temperatures)` **in place** before computing logprobs
   (`srt/layers/sampler.py:189`), so *output* logprobs are temperature-scaled.
   Under C1 (T=0.7) every probability would have come back rescaled. `backend.py`
   sets `SGLANG_RETURN_ORIGINAL_LOGPROB=1` *and* reads everything from the
   **input** side, which is a plain `log_softmax` of the unmodified logits and so
   is raw by construction rather than by flag.
2. **`logprob_start_len` is off by one on purpose.** Entry 0 of
   `input_token_logprobs` is always `None` — the token at position `s` never gets
   a logprob. To score from position `p` you must send `s = p−1` and read from
   index 1. Sending `s = p` silently returns `None` for the first slot.
3. **Radix sharing is not automatic within a batch.** Prefix matching happens at
   admission against the tree *as it is at that instant*, and the default
   schedule policy is `fcfs`. Firing `M` requests at a cold tree makes each one
   prefill the whole prefix. Every probe calls `warm_prefix()` first.

`backend.py` additionally **refuses to start** if sglang has disabled the radix
cache or if deterministic inference did not take — the first would cost an
`M`-fold slowdown while still producing correct numbers, and the second would
make `sampling_seed` silently ignored and the run irreproducible.

## Run order

Matches `PROTOCOL.md` §10. Steps 1–2 need no GPU.

```bash
# 1. corpus  (C0 = intrinsic, C1 = deployment-matched; same prompt IDs)
python -m specfloor.corpus --corpus C0 --domain gsm8k --out runs/C0/gsm8k.jsonl
python -m specfloor.corpus --corpus C1 --domain gsm8k --out runs/C1/gsm8k.jsonl

# 2. anchor population + stratified sample (prints cell occupancy and thin cells)
python -m specfloor.anchors --corpus-file runs/C0/gsm8k.jsonl \
    --out runs/C0/gsm8k.anchors.jsonl --budget 2500

# 3. MC convergence ladder BEFORE the main run -- this is what justifies MAIN_M.
#    --m-base X --m-max X pins each arm at a FIXED M (escalation cannot fire).
#    The arms are nested by construction: sample_paths seeds request i with
#    seed_base+i, so M=32's paths are the first 32 of M=64's. mc_ladder checks
#    this and bootstraps the M-to-M difference as a PAIRED quantity.
for M in 32 64 128 256; do
  python -m specfloor.probe_cheap --corpus C0 --corpus-file runs/C0/gsm8k.jsonl \
      --anchors runs/C0/gsm8k.anchors.jsonl --out runs/pilot/ladder.M$M.jsonl \
      --m-base $M --m-max $M --max-running 64      # <-- see note below
done
python -m specfloor.mc_ladder --arms 'runs/pilot/ladder.M*.jsonl' --ref 128

# 4. cheap pass, full budget
python -m specfloor.probe_cheap --corpus C0 --corpus-file runs/C0/gsm8k.jsonl \
    --anchors runs/C0/gsm8k.anchors.jsonl --out runs/C0/gsm8k.cheap.jsonl

# 5. expensive R_m, on the informative subset only
python -m specfloor.probe_rm --corpus C0 --corpus-file runs/C0/gsm8k.jsonl \
    --cheap runs/C0/gsm8k.cheap.jsonl --out runs/C0/gsm8k.rm.jsonl

# 6. drafter NLL on the SAME anchors  (C1 is the one the claim rests on)
python -m specfloor.eval_nll --corpus-file runs/C1/gsm8k.jsonl \
    --anchors runs/C1/gsm8k.anchors.jsonl \
    --draft deepseek-ai/dspark_qwen3_4b_block7 --out runs/C1/gsm8k.nll.jsonl

# 7. report
python -m specfloor.stats --cheap 'runs/C1/*.cheap.jsonl' \
    --nll 'runs/C1/*.nll.jsonl' --by-stratum
```

Steps 1–7 measure the cheap surrogate and the drafter's loss. The floor itself,
and the gap a real drafter leaves against it, come from four further probes that
share the anchors and the ladder file but not the estimator.

```bash
# 8. the floors. T^(0) is the unconditional barycentre radius, T^(1) the one a
#    head that has seen Z_{k-1} is allowed to reach. --split holds out half the
#    paths to price the plug-in bias of the conditional rung, which is the only
#    rung where it is not negligible.
python -m specfloor.probe_tk --corpus C0 --corpus-file runs/C0/gsm8k.jsonl \
    --cheap runs/C0/gsm8k.ladder.M512.jsonl --out runs/tk/gsm8k.t01.jsonl \
    --anchors 96 --paths 1024 --top-k 256 --rungs 0,1 --split
python -m specfloor.tk_report --tk 'runs/tk/*.t01.jsonl' --by-domain

# 9. the gap. R for the REAL drafter on the same paths, full vocabulary, so
#    G = R - T is a difference of two numbers from one rollout rather than two
#    runs. --order 0 scores a product-measure backbone against T^(0); --order 1
#    scores a markov head against T^(1), and additionally splits R^self into
#    T^(1) + G_post + exposure.
python -m specfloor.probe_rpre --corpus C0 --corpus-file runs/C0/gsm8k.jsonl \
    --cheap runs/C0/gsm8k.ladder.M512.jsonl --drafter <dflash> --order 0 \
    --cond both --out runs/rpre/gsm8k.r0.jsonl --anchors 96 --paths 256 --split
python -m specfloor.probe_rpre --corpus C0 --corpus-file runs/C0/gsm8k.jsonl \
    --cheap runs/C0/gsm8k.ladder.M512.jsonl --drafter <dspark> --order 1 \
    --cond both --out runs/rpre/gsm8k.r1.jsonl --anchors 96 --paths 256 --split
python -m specfloor.rpre_report --rpre 'runs/rpre/*.r0.jsonl' --order 0
python -m specfloor.rpre_report --rpre 'runs/rpre/*.r1.jsonl' --order 1

# 10. free-rollout law against the serving one. No new forward passes: probe_rpre
#     already records a_k = min(1, q_k(Z_k)/p_k(Z_k)) per path per slot, whose
#     running product is that path's probability of REACHING the slot. Reweighting
#     by it gives R_serve beside R_free on identical paths, and E[prod a] gives
#     P(J > j) directly instead of through prod(1 - R_i).
python -m specfloor.srv_report --rpre 'runs/rpre/*.r0.jsonl' --by-domain

# 11. floors on a target that only exists behind an API. One chat completion with
#     logprobs and top_logprobs=20 returns a free rollout AND p(.|X, Z_<k) at
#     every slot, which is all a floor needs. Gaps are NOT measurable this way --
#     no endpoint exposes the backbone tap a drafter reads.
python -m specfloor.probe_api_floor --model deepseek-chat --domain gsm8k \
    --out runs/api/gsm8k.api.jsonl --anchors 64 --paths 256 --gamma 7
python -m specfloor.api_floor_report --api 'runs/api/*.api.jsonl'

# 12. how much of the floor is a COMMITMENT cost. T^(0) is the best single blind
#     proposal; the K-median is the best K of them with an oracle picking per
#     path. K=1 is the same estimator with one centre, so it reproduces T^(0) on
#     the same paths and the difference is within-anchor. NOT a tree drafter's
#     ceiling -- acceptance over a candidate set is a union event, this puts the
#     min inside the expectation. Lloyd is a LOCAL optimum, so the removed
#     fraction is a LOWER bound.
python -m specfloor.probe_kmedian --corpus C0 --corpus-file runs/C0/gsm8k.jsonl \
    --cheap runs/C0/gsm8k.ladder.M512.jsonl --out runs/branch/gsm8k.tb.jsonl \
    --anchors 32 --paths 256 --widths 1,2,4
python -m specfloor.kmedian_report --branch 'runs/branch/*.tb.jsonl'

# 11. single-slot best response. probe_br records ONLY (realised token, target
#     p, drafter q) per path per slot -- no [M, V] rows -- which is what makes
#     M=1024 affordable here when the floor probes run at 256. The water
#     filling and the cross-fit live in br_report, so re-splitting or
#     re-smoothing never costs another GPU pass.
python -m specfloor.probe_br --corpus C0 --corpus-file runs/C0/gsm8k.jsonl \
    --cheap runs/C0/gsm8k.ladder.M512.jsonl --drafter $DRAFT --order 0 \
    --out runs/br/gsm8k.br0.jsonl --anchors 96 --paths 1024
python -m specfloor.br_report --br 'runs/br/*.br0.jsonl' --by-domain

# 12. the same swept to a fixed point. No GPU at all: the water fill returns a
#     distribution on exactly the realised tokens, so every round's accept
#     factors are readable from the files step 11 already wrote. Both sweep
#     orders are run -- compare them only once BOTH have converged, since
#     before that the gap is a rate difference and says nothing.
python -m specfloor.br_iter --br 'runs/br/*.br0.jsonl' --rounds 8 --by-domain

# 13. what a top-20 read costs, if you need to defend a truncated endpoint.
#     Re-read the SAME anchors with --top-k 20 and nothing else changed, so
#     the comparison is paired per (anchor, slot). It must be paired: a top-20
#     read fails the residual gate far more often, so the two columns as
#     printed describe different sub-populations.
python -m specfloor.probe_tk --corpus C0 --corpus-file runs/C0/gsm8k.jsonl \
    --cheap runs/C0/gsm8k.ladder.M512.jsonl --out runs/tk20/gsm8k.t01.tk20.jsonl \
    --anchors 96 --paths 256 --top-k 20 --rungs 0,1 --split
```

`--kv-budget-gib` sets the chunk size, and the chunk is where the sampler's
stream is consumed, so two runs at different budgets are different draws from the
same law rather than the same paths. Within a run everything is paired, which is
what the difference quantities (`G = R - T`, `R_serve - R_free`) rely on; across
runs, expect Monte Carlo agreement, not equality. Raise the budget if a probe
reports skipped anchors — those skips are OOM on the longest contexts, which is
a length-dependent loss rather than a random one.

## Guardrails built into the code

- `corpus.py` refuses to pass silently if the censor rate exceeds 0.5%.
- `anchors.py` prints every cell below the headline minimum so thin cells are
  visible before, not after, the expensive pass.
- `stats.py` suppresses conditional quantiles when `n_informative < 128` and
  labels the cell exploratory instead of printing a number nobody should trust.
- `probe_cheap.py` records the realised `M` per anchor, so escalation is auditable
  and cannot be mistaken for a post-hoc decision.

## Not covered here

Part III/IV serving quantities (`q₂`, `conf`, `α`, rescue, `j*`, hit@1, speedup)
run against a live server, not this offline path. They belong on **C1 only** —
C2 is thinking-on and the drafter was trained thinking-off, so any drafter
quantity measured there tests transfer, not context depth.
