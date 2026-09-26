# $R_{\text{pre}}$ and $G_{\text{pre}}$ — how far the real DFlash is from its own floor

Qwen3-4B target, official `deepseek-ai/dflash_qwen3_4b_block7` drafter
(`markov_rank: 0`, block 7, no head weights at all), 96 anchors per domain,
$M=256$ free rollouts, **full vocabulary, exact TV — no top-$K$ truncation**,
HT weights $1/\pi$, prompt-level cluster bootstrap $B=10^4$.

```
bash measurement_runs/rpre/run_rpre.sh                       # C0, four domains
bash measurement_runs/rpre_c1/run_c1.sh                      # C1 gsm8k + law ablation
python -m measurement.rpre_report --rpre 'measurement_runs/rpre/*.rpre.jsonl' \
       --tk 'measurement_runs/tk/*.t0.jsonl' --recall
```

## The metric this paper is about

Under lossless speculative sampling the drafter proposes from a full
distribution $q$ and the verifier accepts with probability
$\min(1, p(y)/q(y))$, so $\alpha=\sum_v\min(p,q)=1-\mathrm{TV}(p,q)$. The
order-0 ceiling is therefore

$$\alpha_k\ \le\ 1-T_k^{(0)},\qquad T_k^{(0)}=\min_q\mathbb{E}_Z[\mathrm{TV}(p_Z,q)]$$

and $G_k=R_k-T_k^{(0)}\ge0$ is what better order-0 modelling could still remove.
**This is the object. Everything in the Recall section below is a diagnostic and
is not an acceptance rate.**

## Result: C0, four domains, 384 anchors / 170 prompts

| slot | $T^{(0)}$ | $R_{\text{pre}}$ | $G_{\text{pre}}$ | 95% CI | $G/R$ | vs `probe_tk` |
|---|---|---|---|---|---|---|
| 0 | 0.0000 | 0.1359 | 0.1359 | [0.094, 0.183] | 100% | +0.0000 |
| 1 | 0.0776 | 0.2375 | 0.1598 | [0.122, 0.203] | 67.3% | +0.0004 |
| 2 | 0.1211 | 0.3462 | 0.2250 | [0.177, 0.278] | 65.0% | +0.0044 |
| 3 | 0.1724 | 0.4258 | 0.2534 | [0.204, 0.307] | 59.5% | +0.0017 |
| 4 | 0.2060 | 0.4978 | 0.2919 | [0.242, 0.345] | 58.6% | +0.0040 |
| 5 | 0.2458 | 0.5685 | 0.3227 | [0.268, 0.378] | 56.8% | +0.0023 |
| 6 | 0.2861 | 0.6359 | 0.3497 | [0.300, 0.401] | 55.0% | — |

**The headline is negative for the paper's original thesis.** $G_{\text{pre}}$ is
large: 55–67% of DFlash's per-slot rejection loss across the block is *not*
information-theoretic. The draft's anticipated value was 0.024; the measured
value at slot 5 is 0.323, thirteen times larger. The capacity route is not
provably closed.

$G$ is bootstrapped as a **paired** difference — same anchor, same paths, same
$p_Z$ — never as a difference of two separately bootstrapped means.

$T$ here and $T$ from `probe_tk` are independent: different engine (local HF vs
sglang), different truncation (full vocabulary vs top-256), different barycentre
implementation (order-statistic walk vs sparse bisection). They agree to
$+0.0004$…$+0.0044$ across the pooled slots, which is the strongest evidence
either number is right.

Whether the serving stack warps $q$ by the full law or by temperature only makes
no difference at C0 and $\le0.008$ at C1, so that implementation ambiguity is
not load-bearing.

## Robustness: a truncated law makes it worse, not better

**C0 is the deployment protocol, not an idealisation.** `eval.py` defaults to
`--temperature 1.0`, and neither it nor anything under `deepspec/eval/` mentions
`top_p` or `top_k`; `deepspec/utils/sampling.py` samples a temperature-scaled
softmax with no truncation. So the numbers above are already measured under the
law the drafter is evaluated at. C1 ($T=0.7$, top-$p$ 0.8, top-$k$ 20) is the
law the drafter's TRAINING DATA was generated at
(`scripts/data/generate_train_data.py`), and is carried here only to check that
the decomposition is not an artefact of an untruncated law. Under C1, $\mu$ and
$p_{\text{verify}}$ are both the warped distribution, and both are warped in the
code that produced these numbers. gsm8k, **same anchors and same prefixes, only
the law changed**:

| slot | $T$ C0 | $T$ C1 | $R$ C0 | $R$ C1 | $G$ C0 | $G$ C1 | $G/R$ |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0.0204 | 0.0196 | 0.0204 | 0.0196 | 100%→100% |
| 3 | 0.0917 | 0.0571 | 0.1904 | 0.1812 | 0.0987 | 0.1241 | 52%→69% |
| 5 | 0.1466 | 0.0905 | 0.2568 | 0.2172 | 0.1102 | 0.1267 | 43%→58% |
| 6 | 0.1640 | 0.0947 | 0.3223 | 0.2873 | 0.1583 | 0.1926 | 49%→67% |

Truncation removes **42%** of the information floor at slot 6 but only **11%** of
DFlash's actual loss, so $G/R$ *rises* to 67% — the gap is larger under the
truncated law, not smaller, so the finding is not an artefact of $T{=}1$. If DFlash's loss were tail
miscalibration, cutting the tail to 20 tokens would remove it. It does not:
the loss is in the head of the distribution.

**A correction that this ablation forced.** Comparing the C0-corpus run against
the C1-corpus run suggested slot-0 $R$ fell 4× (0.0794 → 0.0196) and we read that
as tail mismatch. Controlling the anchors kills it: 0.0204 vs 0.0196. The 4× was
**corpus composition**, not law — $T{=}1$-generated text is itself harder for a
drafter trained near $T{=}0.7$. Corpus and law are separate effects and only the
first is real here.

## Diagnostic only: Recall@K and its coverage ceiling

**These are not acceptance rates.** A guess set chosen from $X$ alone hits with
the mixture's top-$K$ mass, which is what $\alpha$ becomes only under the
artificial restriction that the proposal be a point mass. Reported because
candidate-set arguments in the literature are stated in this metric.

C0, four domains, greedy-target convention (the correct token is the target's
own $\arg\max$ on each sampled path):

| slot | DFlash @1 | ceiling @1 | DFlash @16 | ceiling @16 |
|---|---|---|---|---|
| 0 | 0.859 | 1.000 | 1.000 | 1.000 |
| 3 | 0.614 | 0.830 | 0.906 | 0.996 |
| 6 | 0.422 | 0.711 | 0.799 | 0.983 |

The ceiling is a **counting** object: bucket the 256 paths by the target's greedy
next token, and take the top-$K$ masses of that histogram. No TV anywhere.

The 0.983 at slot 6 says a commit-blind top-16 lattice contains the right token
98.3% of the time. It does **not** say any drafter accepts at 98.3%: the
acceptance value of proposing those 16 candidates is a width-16 branch ceiling
$1-T_k^{\text{branch},16}$, which is a $K$-median TV problem and is not measured
here.

## Cross-estimator gate

$\max_a\bar p_k(a)\le 1-T_k^{(0)}$ must hold at every anchor, because $\delta_a$
is an admissible $q$ in the minimisation defining $T^{(0)}$. The two sides come
from disjoint code. **2660 (anchor, slot) cells, zero violations**, maximum
deviation $8.7\times10^{-8}$ on cells where both sides exceed 0.9999. Slack
$(1-T^{(0)})-\max_a\bar p_k$: 0.112 at slot 0 falling to 0.041 at slot 6 — how
much the point-mass restriction gives away.

## Scope

Everything here is **DFlash 1**, the only system whose block proposal is a true
product measure and therefore the only one that genuinely sits at $T^{(0)}$. A
sampled markov chain (DSpark) and a causal path selector are both order-1
conditional proposals; $T^{(0)}$ bounds them only loosely and the tight,
serving-relevant floor is $T^{(1)}$, which is **not measured**. Nothing in this
file is a measurement of DSpark or of any selector.
