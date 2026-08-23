r"""Free-rollout risk against serving risk, and $\mathbb E[\prod]$ against $\prod\mathbb E$.

  python -m specfloor.srv_report --rpre 'measurement_runs/srv/*.srv0.jsonl'

Two questions, one recorder.

**How much does the outer path law matter?** Everything else in this paper is
averaged over the target's own block law $\mu$, because that is what a floor is
defined against. Serving does not sample slot $k$ from $\mu$: it reaches slot $k$
only on blocks that survived, and survival is not independent of the trajectory.
The conversion is exact and needs nothing new from the model. A path that
realised $Z$ is accepted at slot $i$ with probability
$a_i = \min(1, q_i(Z_i)/p_i(Z_i))$, so

    W_{k-1} = prod_{i<k} a_i     is that path's probability of reaching slot k,
    R_k^serve = E[W_{k-1} TV] / E[W_{k-1}],
    S_k       = E[W_k]           is P(J > k).

$R^{\text{free}}$ and $R^{\text{serve}}$ differ only in the weight, on the same
paths, so their difference is a within-anchor quantity and is bootstrapped as one.

**Does the accepted length factorise?** $S_j$ is an expectation of a product;
$\prod_{i\le j}(1-R_i)$ is a product of expectations. They agree only if the
per-slot accept factors are uncorrelated along a path, which is exactly the
condition a published accepted length has to be inverted through. Both are
printed. Note the asymmetry in what is exact: $1-R_i$ is computed from the full
simplex and carries no sampling error in the accept factor, while $S_j$ is a
Monte Carlo average of a product, so a small positive gap is the interesting
direction and a small negative one is noise.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random


def load(pattern):
    by = {}
    for f in sorted(glob.glob(pattern)):
        dom = os.path.basename(f).split(".")[0]
        for line in open(f):
            line = line.strip()
            if line:
                by.setdefault(dom, []).append(json.loads(line))
    return by


def wmean(pairs):
    sw = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / sw if sw > 0 else float("nan")


def serve_risk(cs):
    r"""Serving risk as a RATIO OF SUMS, not a mean of per-anchor ratios.

        R^serve_k = E[W_k * TV] / E[W_k],   W_k = prod_{i<k} a_i

    Each record already carries its own anchor's ratio in `R_serve`, and its
    arrival mass E[W_k] is `S[k-1]` (1 at slot 0, since nothing precedes it).
    Averaging the ratios would weight a rarely-reached anchor the same as one
    the serving path visits constantly, which is not the serving population.
    The two differ by a lot here: at slot 6 the mean of ratios reads 0.58 and
    the ratio of sums reads 0.21, because arrival mass and risk are strongly
    anti-correlated -- the anchors that survive to slot 6 are the ones the
    drafter was already agreeing with.
    """
    num = sum(c["Rs"] * c["w"] * c["D"] for c in cs
              if c.get("Rs") is not None and c.get("D") is not None)
    den = sum(c["w"] * c["D"] for c in cs
              if c.get("Rs") is not None and c.get("D") is not None)
    return num / den if den > 0 else float("nan")


def boot(cells, key, B, seed):
    """Prompt-level cluster bootstrap of a Hájek-weighted column.

    `key` may also be one of the two derived quantities, which are ratios of
    sums and cannot be written as a weighted mean of a column.
    """
    by = {}
    for c in cells:
        by.setdefault(c["pid"], []).append(c)
    ks = list(by)
    rng = random.Random(seed)
    draws = []
    for _ in range(B):
        s = []
        for _ in range(len(ks)):
            s += by[ks[rng.randrange(len(ks))]]
        if key == "Rs_pop":
            v = serve_risk(s)
        elif key == "d_pop":
            v = serve_risk(s) - wmean([(c["Rf"], c["w"]) for c in s
                                       if c.get("Rf") is not None])
        else:
            v = wmean([(c[key], c["w"]) for c in s if c.get(key) is not None])
        if not math.isnan(v):
            draws.append(v)
    if not draws:
        return float("nan"), float("nan")
    draws.sort()
    q = lambda p: draws[max(0, min(len(draws) - 1, int(p * len(draws))))]
    return q(0.025), q(0.975)


def cells(recs, k):
    out = []
    for r in recs:
        sk = str(k)
        rf = (r.get("R") or {}).get(sk)
        rs = (r.get("R_serve") or {}).get(sk)
        if rf is None or rs is None:
            continue
        # arrival mass at slot k: E[prod_{i<k} a_i], which is S[k-1]. Slot 0 is
        # reached by every path, so its arrival mass is 1 by definition.
        D = 1.0 if k == 0 else (r.get("S") or {}).get(str(k - 1))
        out.append(dict(pid=r["prompt_id"], w=1.0 / max(r.get("pi", 1.0), 1e-9),
                        Rf=rf, Rs=rs, D=D, d=rs - rf,
                        T=(r.get("T") or {}).get(sk),
                        S=(r.get("S") or {}).get(sk),
                        ab=(r.get("abar") or {}).get(sk)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rpre", required=True)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--by-domain", action="store_true")
    args = ap.parse_args()

    by = load(args.rpre)
    if not by:
        raise SystemExit(f"no records matched {args.rpre!r}")
    allr = [r for rs in by.values() for r in rs]
    slots = sorted({int(k) for r in allr for k in (r.get("R") or {})})
    print(f"M={allr[0].get('M')}  order={allr[0].get('order')}  "
          f"bootstrap B={args.boot} over PROMPTS  Hájek weights 1/pi")

    groups = ([(d, by[d]) for d in sorted(by)] if args.by_domain else []) + \
             [("POOLED", allr)]
    for name, recs in groups:
        print(f"\n===== {name}  ({len(recs)} anchors, "
              f"{len({r['prompt_id'] for r in recs})} prompts) =====")
        print(f"{'k':>2} {'n':>4} | {'R free':>8} {'R serve':>8} {'serve-free':>11} "
              f"{'95% CI on diff':>17} | {'E[prod a]':>10} {'prod E[a]':>10} {'ratio':>7}")
        prodE = 1.0
        for k in slots:
            cs = cells(recs, k)
            if len(cs) < 5:
                continue
            g = lambda key: wmean([(c[key], c["w"]) for c in cs
                                   if c.get(key) is not None])
            Rf, Rs = g("Rf"), serve_risk(cs)
            lo, hi = boot(cs, "d_pop", args.boot, args.seed)
            # prod E[a_i] uses the EXACT marginal 1 - R_i, not the sampled abar:
            # the accept factor's mean is 1 - TV identically, and the TV side is
            # computed over the whole simplex rather than from the drawn tokens.
            prodE *= (1.0 - Rf)
            S = g("S")
            print(f"{k:>2} {len(cs):>4} | {Rf:>8.4f} {Rs:>8.4f} {Rs - Rf:>11.4f} "
                  f"[{lo:>6.4f},{hi:>6.4f}] | {S:>10.4f} {prodE:>10.4f} "
                  f"{S / prodE if prodE > 0 else float('nan'):>7.3f}")
        # accepted length from the two routes
        EJ_s = sum(wmean([(c["S"], c["w"]) for c in cells(recs, k)
                          if c.get("S") is not None]) for k in slots)
        pe, EJ_p = 1.0, 0.0
        for k in slots:
            cs = cells(recs, k)
            pe *= (1.0 - wmean([(c["Rf"], c["w"]) for c in cs]))
            EJ_p += pe
        print(f"   E[J] from E[prod a] = {EJ_s:.3f}   (tau = {1 + EJ_s:.3f})")
        print(f"   E[J] from prod E[a] = {EJ_p:.3f}   (tau = {1 + EJ_p:.3f})"
              f"   -- understates by {EJ_s - EJ_p:+.3f}")

    print("\nR serve is the same TV on the same paths, reweighted by each path's")
    print("probability of reaching the slot. A large negative serve-free difference")
    print("at deep slots means the hard trajectories had already been rejected, so")
    print("the free-rollout number is not what a serving stack experiences there.")
    print("ratio > 1 means the accept factors are positively dependent along a path,")
    print("which is the condition an accepted length has to be inverted through.")


if __name__ == "__main__":
    main()
