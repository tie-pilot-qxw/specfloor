r"""Single-slot best-response headroom: water filling on one half, $\tau$ on the other.

  python -m specfloor.br_report --br 'measurement_runs/br/*.br0.jsonl' --by-domain

**The estimand is a population quantity, not a property of these paths.** What we want is

    q_k*  =  argmax_{q_k}  E_{Z ~ mu}[ S_k(Z) F_k(Z) a_k(Z; q_k) ],

with the expectation against the target's own continuation law. The rollouts are Monte Carlo draws
from that law, so the water filling of probes/br can only solve the empirical version, and reporting
its value on the paths it was fitted to would measure sampling noise. The objective is *more* prone
to this than the floor is: the weight $c_r = S_{r,k}F_{r,k}$ is not 1 but a product of accept
factors, so a handful of long-surviving paths carry most of the leverage, and one rare token on a
high-$c$ path looks like a bargain that a fresh draw will not reproduce.

So every number here is **cross-fitted**. Each anchor's paths are split in half by index; the
proposal is water-filled on one half and $\tau$ is evaluated on the other; the halves are swapped and
the two folds averaged. `dtau_train` is printed beside `dtau_held` and their difference is a result
in its own right -- it prices how hard this policy is to estimate, exactly as `curse` does for the
floor in §5.3.

**Where unallocated mass goes.** Water filling stops when every path in the cell is saturated, which
can happen with mass to spare: $\lambda = 1 - \sum_v q^\star(v) > 0$. That leftover is worth nothing
in-sample and everything out-of-sample, because a held-out path may realise a token the fit half
never produced. We give it to the drafter's own proposal,

    q_fit(v)  =  waterfill(v)  +  lambda * q_base(v),

which is a proper distribution, needs $q_{\mathrm{base}}$ only at the tokens actually realised (all
this probe records), and leaves every token with at least $\lambda q_{\mathrm{base}}(v)$ so that
support truncation cannot by itself drive the held-out number negative. $\lambda$ is reported.

**Two biases, opposite in sign.** The fitted mass is optimistic (it chases sampling noise) and the
fitted support is pessimistic (it is confined to what the fit half realised, up to the $\lambda$
term). Neither is removed by cross-fitting; cross-fitting only stops the first from being counted as
signal. Because $q_{\mathrm{base}}$ is itself feasible, the population optimum is at least
$\tau_{\mathrm{base}}$, so **a negative held-out $\Delta\tau$ means the estimate failed, not that the
oracle is worse than the drafter** -- and at deep slots, where survival concentrates, that is the
expected outcome rather than a surprise. The effective sample size of $c$ is printed for that reason.

**What the cross-fit does NOT certify, and this is the important one.** Splitting rollouts guards
against overfitting the *realisations*. It does not guard against the prefix, because both halves
share one. The water filling is solved separately at every anchor, so what it delivers is a map

    X  |-->  q*_{k,X}

built by seeing that prefix's own continuation law many times. Held-out rollouts ask "does this
prefix-specific proposal survive fresh draws from the same prefix"; they never ask "could anything
compute q*_{k,X'} for a prefix X' it has not seen". That second question is the whole modelling
problem and it is assumed away here.

This is not a defect peculiar to this measurement -- it is exactly the convention of $T^{(m)}$, whose
minimisation also sits inside the outer expectation and so also grants a separate optimum per prefix
(paper §3.1, §3.5). Both quantities live in the same oracle class: **any measurable map from the
permitted information to the simplex**. What is forbidden is only the realisation that has not
happened yet.

**So name it accordingly.** This is a *single-slot serving oracle*, not trainable headroom:

    tau_base  <=  tau(best DFlash-shaped slot k)  <=  tau(q_{-k}, q*_{k,X})    <-- measured here

Calling it "what DFlash would gain from better modelling of slot k" is too strong by the whole
distance between the middle and right terms. What it does answer cleanly, because both sides oracle
the prefix map away identically, is: **under the same information set, how far apart are the
TV-optimal proposal and the serving-optimal one?** That difference is attributable to the objective
and to nothing else.

**And it reopens ONE slot.** It is a best response to the drafter's other six, not a piece of a
jointly optimal proposal: $S_k$ depends on $q_{<k}$ and $F_k$ on $q_{>k}$, so changing a neighbour
changes this slot's answer. Iterating to a fixed point would give a coordinate-wise optimum, still
not the global $\max_q \mathbb{E}_q[\tau]$.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random


# ------------------------------------------------------------------ paths ----
def per_path(row):
    """Reassemble a row's slot-wise records into one entry per path.

    Returns {gid: {"a": [...], "tok": [...], "p": [...], "q": [...],
                   "cond": [...], "alive": [...]}} with a = 0 at slots the path
    did not reach -- the target's own rollout ended there, so the block ends
    there and every later factor is zero too.
    """
    K = row["K"]
    out = {}
    for k, sl in enumerate(row["slots"]):
        for j, g in enumerate(sl["gid"]):
            e = out.setdefault(g, dict(a=[0.0] * K, tok=[None] * K, p=[0.0] * K,
                                       q=[0.0] * K, cond=[None] * K,
                                       alive=[False] * K, qtv=[None] * K,
                                       qtvx=[None] * K))
            p, q = sl["p"][j], sl["q"][j]
            e["tok"][k] = sl["tok"][j]
            e["p"][k] = p
            e["q"][k] = q
            e["cond"][k] = sl["cond"][j]
            e["alive"][k] = True
            e["a"][k] = min(1.0, q / p) if p > 0 else 0.0
            if "qtv" in sl:
                e["qtv"][k] = sl["qtv"][j]
                e["qtvx"][k] = sl["qtv_xf"][j]
    return out


def tau_of(paths, K, override=None):
    """1 + mean over paths of sum_j prod_{i<=j} a_i, optionally replacing slot k.

    override = (k, fn) where fn(entry) -> a_k for that path.
    """
    if not paths:
        return float("nan")
    tot = 0.0
    for e in paths:
        run, acc = 1.0, 0.0
        for i in range(K):
            ai = e["a"][i]
            if override is not None and i == override[0]:
                ai = override[1](e) if e["alive"][i] else 0.0
            run *= ai
            if run <= 0.0:
                break
            acc += run
        tot += acc
    return 1.0 + tot / len(paths)


def sk_fk(e, k, K):
    """(S_k, F_k) for one path under the BASE proposal: reach, and what follows."""
    S = 1.0
    for i in range(k):
        S *= e["a"][i]
    F, run = 1.0, 1.0
    for j in range(k + 1, K):
        run *= e["a"][j]
        if run <= 0.0:
            break
        F += run
    return S, F


# --------------------------------------------------------- water filling ----
def waterfill(items):
    """Exact argmax of sum_r c_r min(1, q(y_r)/p_r) over the simplex.

    items: list of (token, p, c). For token v the objective is piecewise linear
    in x = q(v) with slope sum_{r: y_r = v, p_r > x} c_r / p_r, which decreases
    as paths saturate at x = p_r. Each token therefore contributes segments of
    decreasing slope, and the optimum pours each next unit of mass into the
    highest remaining slope until one unit is spent.

    Returns (dict v -> mass, leftover) where leftover is 1 - sum of masses, the
    mass no segment had any use for.
    """
    by = {}
    for v, p, c in items:
        if p > 0 and c > 0:
            by.setdefault(v, []).append((p, c))
    segs = []
    for v, lst in by.items():
        lst.sort()                                   # ascending p
        # slope on (p_{j-1}, p_j] is the total c/p of paths with p_r >= p_j,
        # accumulated from the top down.
        suffix = 0.0
        rev = []
        for p, c in reversed(lst):
            suffix += c / p
            rev.append((p, suffix))
        rev.reverse()                                # ascending p again
        prev = 0.0
        for p, slope in rev:
            if p > prev:
                segs.append((slope, p - prev, v))
                prev = p
    segs.sort(key=lambda s: -s[0])
    q, budget = {}, 1.0
    for slope, width, v in segs:
        if budget <= 0:
            break
        take = min(width, budget)
        q[v] = q.get(v, 0.0) + take
        budget -= take
    return q, max(0.0, budget)


def fit_slot(fit_paths, k, K, order):
    """Water-fill slot k per information cell. Returns {cell: (q_dict, leftover)}."""
    cells = {}
    for e in fit_paths:
        if not e["alive"][k]:
            continue
        S, F = sk_fk(e, k, K)
        c = S * F
        if c <= 0:
            continue
        w = 0 if order == 0 else e["cond"][k]
        cells.setdefault(w, []).append((e["tok"][k], e["p"][k], c))
    return {w: waterfill(items) for w, items in cells.items()}


def ess_of(fit_paths, k, K):
    cs = []
    for e in fit_paths:
        if not e["alive"][k]:
            continue
        S, F = sk_fk(e, k, K)
        cs.append(S * F)
    s = sum(cs)
    return (s * s) / sum(c * c for c in cs) if s > 0 and any(cs) else 0.0


def tv_scorer(k, field):
    """a_k under the TV barycentre -- the proposal T^(0) selects at this slot."""
    def fn(e):
        qv = e[field][k]
        p = e["p"][k]
        if qv is None or p <= 0:
            return 0.0
        return min(1.0, qv / p)
    return fn


def scorer(fitted, k, order, eps):
    """a_k under the fitted proposal, mixed with the base at weight eps."""
    def fn(e):
        w = 0 if order == 0 else e["cond"][k]
        got = fitted.get(w)
        qb = e["q"][k]
        if got is None:                      # cell unseen in the fit half
            qv = qb
        else:
            qd, leftover = got
            qv = qd.get(e["tok"][k], 0.0) + leftover * qb
        qv = (1.0 - eps) * qv + eps * qb
        p = e["p"][k]
        return min(1.0, qv / p) if p > 0 else 0.0
    return fn


# ---------------------------------------------------------------- report ----
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


def anchor_cells(row, eps_list):
    """Cross-fitted dtau per slot for one anchor, plus diagnostics."""
    K = row["K"]
    order = row.get("order", 0)
    paths = list(per_path(row).values())
    if len(paths) < 8:
        return None
    half = len(paths) // 2
    folds = [(paths[:half], paths[half:]), (paths[half:], paths[:half])]

    res = {}
    for k in range(K):
        acc = {e: [] for e in eps_list}
        acc_tr, acc_tv = [], []
        lam, essv = [], []
        for fit, test in folds:
            fitted = fit_slot(fit, k, K, order)
            if not fitted:
                continue
            essv.append(ess_of(fit, k, K))
            lam.append(wmean([(lv, 1.0) for _, lv in fitted.values()]))
            base_test = tau_of(test, K)
            base_fit = tau_of(fit, K)
            if test and test[0]["qtv"][k] is not None:
                acc_tv.append(tau_of(test, K, override=(k, tv_scorer(k, "qtvx")))
                              - base_test)
            for e in eps_list:
                fn = scorer(fitted, k, order, e)
                acc[e].append(tau_of(test, K, override=(k, fn)) - base_test)
            fn0 = scorer(fitted, k, order, eps_list[0])
            acc_tr.append(tau_of(fit, K, override=(k, fn0)) - base_fit)
        if not acc_tr:
            continue
        res[k] = dict(
            held={e: sum(v) / len(v) for e, v in acc.items() if v},
            tv=(sum(acc_tv) / len(acc_tv)) if acc_tv else None,
            train=sum(acc_tr) / len(acc_tr),
            ess=sum(essv) / len(essv) if essv else 0.0,
            lam=sum(lam) / len(lam) if lam else 0.0,
            n=len(paths),
        )
    return res


def boot(cells, key, B, seed):
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
        v = wmean([(c[key], c["w"]) for c in s if c.get(key) is not None])
        if not math.isnan(v):
            draws.append(v)
    if not draws:
        return float("nan"), float("nan")
    draws.sort()
    q = lambda p: draws[max(0, min(len(draws) - 1, int(p * len(draws))))]
    return q(0.025), q(0.975)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--br", required=True)
    ap.add_argument("--eps", default="0,0.1,0.25",
                    help="mix weights toward the base proposal; the first is the "
                         "one train-vs-held-out is reported at")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--by-domain", action="store_true")
    args = ap.parse_args()

    eps_list = [float(x) for x in args.eps.split(",") if x.strip() != ""]
    by = load(args.br)
    if not by:
        raise SystemExit(f"no records matched {args.br!r}")

    prepared = {}
    for dom, rows in by.items():
        out = []
        for r in rows:
            cells = anchor_cells(r, eps_list)
            if cells:
                out.append((r, cells))
        prepared[dom] = out
    allr = [x for v in prepared.values() for x in v]
    K = allr[0][0]["K"]
    order = allr[0][0].get("order", 0)
    print(f"M={allr[0][0].get('M')}  order={order}  2-fold cross-fit  "
          f"bootstrap B={args.boot} over PROMPTS  Hajek weights 1/pi")

    groups = ([(d, prepared[d]) for d in sorted(prepared)] if args.by_domain else []) + \
             [("POOLED", allr)]
    for name, rows in groups:
        print(f"\n===== {name}  ({len(rows)} anchors, "
              f"{len({r['prompt_id'] for r, _ in rows})} prompts) =====")
        head = "  ".join(f"{'dtau@eps=' + str(e):>13}" for e in eps_list)
        print(f"{'k':>2} {'n':>4} | {head} | {'dtau TV':>9} | {'dtau train':>10} "
              f"{'fit-score':>9} | {'ESS(c)':>7} {'lambda':>7}")
        for k in range(K):
            cs = []
            for r, cells in rows:
                if k not in cells:
                    continue
                c = cells[k]
                d = dict(pid=r["prompt_id"], w=1.0 / max(r.get("pi", 1.0), 1e-9),
                         train=c["train"], ess=c["ess"], lam=c["lam"], tv=c["tv"])
                for e in eps_list:
                    d[f"h{e}"] = c["held"].get(e)
                cs.append(d)
            if len(cs) < 5:
                continue
            g = lambda key: wmean([(c[key], c["w"]) for c in cs
                                   if c.get(key) is not None])
            vals = "  ".join(f"{g('h' + str(e)):>13.4f}" for e in eps_list)
            tr = g("train")
            h0 = g("h" + str(eps_list[0]))
            tv = g("tv")
            tvs = f"{tv:>9.4f}" if not math.isnan(tv) else f"{'-':>9}"
            print(f"{k:>2} {len(cs):>4} | {vals} | {tvs} | {tr:>10.4f} "
                  f"{tr - h0:>9.4f} | {g('ess'):>7.1f} {g('lam'):>7.4f}")
        # the interval on the headline column only
        e0 = eps_list[0]
        print(f"   95% CI on dtau@eps={e0}, per slot:")
        for k in range(K):
            cs = [dict(pid=r["prompt_id"], w=1.0 / max(r.get("pi", 1.0), 1e-9),
                       v=cells[k]["held"].get(e0))
                  for r, cells in rows if k in cells]
            cs = [c for c in cs if c["v"] is not None]
            if len(cs) < 5:
                continue
            lo, hi = boot(cs, "v", args.boot, args.seed)
            print(f"     k={k}  [{lo:+.4f}, {hi:+.4f}]")

    print("\ndtau is CROSS-FITTED: the proposal is water-filled on half the paths")
    print("and tau is scored on the other half, both folds, averaged. 'fit-score'")
    print("is how much the in-sample number overstates it -- large means this")
    print("policy is hard to estimate, which ESS(c) predicts. lambda is the mass")
    print("water filling left unspent, handed to the drafter's own proposal.")
    print("dtau TV is the same slot handed the TV BARYCENTRE instead -- the")
    print("proposal T^(0) selects -- cross-fitted on the same split, so the two")
    print("columns differ only by the objective and not by how they were fitted.")
    print("A NEGATIVE held-out dtau means the estimate failed, not that the oracle")
    print("is worse: the base proposal is feasible, so the population optimum is")
    print("at least tau_base. One slot is reopened; this is a best response to the")
    print("other six, not part of a jointly optimal proposal.")


if __name__ == "__main__":
    main()
