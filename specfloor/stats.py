"""Inference: hierarchical bootstrap, weighted aggregation, tail reporting,
and the information/model gap bounds.

Two things this file exists to prevent:

1. Bootstrapping anchors as if they were i.i.d. They are not -- anchors cluster
   within sequence and sequences within prompt. The outer resampling unit is the
   PROMPT; anchors ride along with it.

2. Quoting an unweighted pooled number. Stratified sampling deliberately
   over-represents thin cells, so aggregation uses inverse-probability weights
   (workload-native) or equal domain weights (macro) -- and always says which.

Usage:
  python -m specfloor.stats --cheap runs/C0/*.cheap.jsonl
  python -m specfloor.stats --cheap runs/C1/gsm8k.cheap.jsonl \
      --nll runs/C1/gsm8k.nll.jsonl        # adds the decomposition bounds
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import random
import statistics as st

from specfloor import config as C


# ------------------------------------------------------------- bootstrap ----
def hierarchical_bootstrap(records, statistic, B=C.BOOTSTRAP_B, seed=C.SEED):
    """records: [{prompt_id, ...}]; statistic: list[record] -> float."""
    by_prompt = collections.defaultdict(list)
    for r in records:
        by_prompt[r["prompt_id"]].append(r)
    prompts = list(by_prompt)
    n = len(prompts)
    rng = random.Random(seed)

    point = statistic(records)
    draws = []
    for _ in range(B):
        rep = []
        for _ in range(n):
            rep.extend(by_prompt[prompts[rng.randrange(n)]])
        try:
            v = statistic(rep)
        except (ZeroDivisionError, ValueError, st.StatisticsError):
            continue
        if v == v and abs(v) != float("inf"):     # NaN/inf would corrupt sort()
            draws.append(v)
    n_bad = B - len(draws)
    if n_bad > 0.01 * B:
        print(f"   !! {n_bad}/{B} bootstrap replicates were NaN/inf and were "
              f"dropped -- the CI below is NOT trustworthy")
    if len(draws) < 100 or point != point:
        return point, float("nan"), float("nan")
    draws.sort()
    lo = draws[int((1 - C.CI) / 2 * len(draws))]
    hi = draws[int((1 + C.CI) / 2 * len(draws)) - 1]
    return point, lo, hi


def fmt(point, lo, hi, pct=False, nd=3):
    s = 100 if pct else 1
    u = "%" if pct else ""
    return f"{point*s:.{nd}f}{u} [{lo*s:.{nd}f}, {hi*s:.{nd}f}]"


# ---------------------------------------------------------- aggregation -----
def wmean(vals, weights):
    tot = sum(weights)
    return sum(v * w for v, w in zip(vals, weights)) / tot if tot else float("nan")


def native_weights(records):
    """Inverse inclusion probability -- undoes the stratification distortion."""
    return [1.0 / max(r.get("pi", 1.0), 1e-9) for r in records]


def n_clusters(records) -> int:
    """Distinct PROMPTS, which is the resampling unit and therefore the only
    honest sample size.

    Anchors are not independent: one long sequence can contribute dozens of them,
    so a raw record count sails past MIN_INFORMATIVE_PER_CELL while the effective
    sample size is a handful of prompts. Every suppression gate below counts
    clusters, not rows.
    """
    return len({r["prompt_id"] for r in records})


# -------------------------------------------------------------- tails -------
def tail_report(records, slot, thr=C.INFORMATIVE_THRESHOLD, label=""):
    """Frequency x severity, rather than one quantile carrying the story."""
    def dce(r):
        return r["dCE"][slot]

    w = native_weights(records)

    def incidence(rs):
        ww = native_weights(rs)
        return wmean([1.0 if dce(r) > thr else 0.0 for r in rs], ww)

    def mean_all(rs):
        return wmean([dce(r) for r in rs], native_weights(rs))

    def mean_inf(rs):
        sub = [r for r in rs if dce(r) > thr]
        if not sub:
            raise ValueError
        return wmean([dce(r) for r in sub], native_weights(sub))

    inf_recs = [r for r in records if dce(r) > thr]
    n_inf = len(inf_recs)
    n_inf_cl = n_clusters(inf_recs)
    # How many anchors could have their classification flipped by MC noise. A
    # thresholded statistic is only safe when this is small: E[1(dCE_hat > eps)]
    # equals P(dCE > eps) only if the noise is small WHERE THE THRESHOLD IS.
    # Individually noisy anchors far from eps are harmless; these are not.
    n_amb = sum(1 for r in records if slot in (r.get("ambiguous_slots") or []))
    print(f"\n-- {label} slot {slot}   n={len(records)} anchors / "
          f"{n_clusters(records)} prompts   informative={n_inf} / {n_inf_cl} prompts")
    print(f"   P(dCE>{thr})            {fmt(*hierarchical_bootstrap(records, incidence), pct=True, nd=1)}")
    if n_amb:
        print(f"      !! {n_amb} ({n_amb/len(records):.1%}) of these anchors sit "
              f"within {C.AMBIGUITY_K}*SE of the threshold -- quote the "
              f"incidence WITH this number")
    else:
        print(f"      0 anchors within {C.AMBIGUITY_K}*SE of the threshold: "
              f"MC noise and eps are disjoint here, so the incidence is "
              f"noise-safe")
    print(f"   E[dCE]                  {fmt(*hierarchical_bootstrap(records, mean_all))}")
    if n_inf_cl >= C.MIN_INFORMATIVE_PER_CELL:
        print(f"   E[dCE | informative]    {fmt(*hierarchical_bootstrap(records, mean_inf))}")
        sub = sorted(((dce(r), 1.0 / max(r.get("pi", 1.0), 1e-9))
                      for r in records if dce(r) > thr), key=lambda x: x[0])
        tot = sum(w for _, w in sub)

        def wq(f):                       # weighted quantile, same weights as
            acc = 0.0                    # every other line in this block
            for v, w in sub:
                acc += w
                if acc >= f * tot:
                    return v
            return sub[-1][0]

        print(f"   p50 / p90 | informative {wq(.50):.3f} / {wq(.90):.3f}"
              f"   (workload-native weights)")
    else:
        print(f"   conditional quantiles SUPPRESSED (informative anchors span "
              f"{n_inf_cl} prompts < {C.MIN_INFORMATIVE_PER_CELL}; {n_inf} raw "
              f"anchors is NOT the sample size) -- exploratory")


# ------------------------------------------------------ decomposition -------
def decomposition(cheap, nll, slot):
    """Bounds, not point estimates.  H*_D >= CE_B because the drafter sees the
    prefix only through five projected target layers, so:

        G_info  >= dCE                    (a LOWER bound on the information gap)
        G_model <= L_D - CE_B             (an UPPER bound on avoidable error)
        G_info + G_model = L_D - CE_A     (measured exactly)
    """
    key = lambda r: (r["corpus"], r["prompt_id"], r["t"])
    cheap_c = {r.get("corpus") for r in cheap}
    nll_c = {r.get("corpus") for r in nll}
    if None in cheap_c | nll_c:
        raise SystemExit("records without a `corpus` field -- regenerate; "
                         "prompt_id is corpus-independent by design, so an "
                         "unkeyed join silently pairs different tokens")
    if cheap_c != nll_c:
        raise SystemExit(f"cross-corpus join refused: cheap={sorted(cheap_c)} "
                         f"nll={sorted(nll_c)}")
    nmap = {key(r): r for r in nll}
    joined = [(c, nmap[key(c)]) for c in cheap if key(c) in nmap]
    if not joined:
        print("\n!! no overlapping anchors between cheap probe and eval NLL")
        return

    recs = [{"prompt_id": c["prompt_id"], "pi": c.get("pi", 1.0),
             "total": n["L_D_base"][slot] - c["ce_A"][slot],
             "info_lb": c["dCE"][slot],
             "model_ub": n["L_D_base"][slot] - c["ce_B"][slot]} for c, n in joined]

    def share(rs):
        w = native_weights(rs)
        tot = wmean([r["total"] for r in rs], w)
        return wmean([r["info_lb"] for r in rs], w) / tot if tot > 0 else float("nan")

    print(f"\n== information / model decomposition, slot {slot}   n={len(recs)}")
    print(f"   estimand: L_D_base (path-marginal, matches CE_B's information set)")
    for name, f in (("L_D - CE_A  (total, exact)", lambda rs: wmean([r["total"] for r in rs], native_weights(rs))),
                    ("G_info  >= dCE", lambda rs: wmean([r["info_lb"] for r in rs], native_weights(rs))),
                    ("G_model <= L_D - CE_B", lambda rs: wmean([r["model_ub"] for r in rs], native_weights(rs)))):
        print(f"   {name:<28} {fmt(*hierarchical_bootstrap(recs, f))}")
    p, lo, hi = hierarchical_bootstrap(recs, share)
    print(f"   {'information share >=':<28} {fmt(p, lo, hi, pct=True, nd=1)}")
    print(f"   -> claim holds a fortiori if the lower CI bound exceeds 50%")


def _ratio_of_means(rs, m, k):
    """R_m as a RATIO OF MEANS, not a mean of per-anchor ratios.

    Per-anchor R_m = num/den divides by a quantity that is small for some
    anchors, so its mean is dominated by whichever anchor happened to draw the
    smallest denominator -- the estimator has no finite variance to speak of.
    Aggregating the numerator and the denominator separately and dividing once
    is a well-defined population quantity and is stable by construction:

        R_m = E[CE_B - CE_M_m] / E[CE_B - CE_A]

    Both expectations use the same inverse-probability weights, so the
    stratification is undone exactly as everywhere else.
    """
    num, den, w = [], [], []
    for r in rs:
        n = (r.get("num", {}).get(str(m), {}) or {}).get(str(k))
        d = (r.get("den", {}) or {}).get(str(k))
        if n is None or d is None or n != n or d != d:
            continue
        num.append(n); den.append(d)
        w.append(1.0 / max(r.get("pi", 1.0), 1e-9))
    if not num:
        return float("nan")
    dd = wmean(den, w)
    return wmean(num, w) / dd if dd > 0 else float("nan")


def rm_report(rm, slot_range=None):
    """R_m with CIs and the H2 wording gate. Without this the R_m pass produced
    a file nobody read, so H2 could not be adjudicated at all."""
    if not rm:
        return
    K = C.GAMMA
    slots = slot_range or range(1, K)
    legacy = not any("num" in r for r in rm)
    print(f"\n== R_m  (n={len(rm)} eligible anchors over {n_clusters(rm)} "
          f"prompts, dCE > {C.RM_ELIGIBILITY_THRESHOLD})")
    print(f"   estimand: R_m = E[dCE - dCE_m | dCE > {C.RM_ELIGIBILITY_THRESHOLD}]"
          f" / E[dCE | dCE > {C.RM_ELIGIBILITY_THRESHOLD}]")
    print( "   NOTE the conditioning: this is recovery over the mass carried by "
           "ELIGIBLE anchors,\n   not over all informative anchors. Quote it that way.")
    if legacy:
        print("   !! this file predates sample splitting: its CE_B is the same "
              "estimate that\n      selected the anchor, so R_m is biased "
              "TOWARD 1. Rerun probe_rm.")
    else:
        print("   headline = ratio of means; CE_B re-estimated on draws "
              "independent of the selection")
    print(f"   {'slot':>5}" + "".join(f"{'R_'+str(m):>26}" for m in C.RM_ORDERS))
    for k in slots:
        row = f"   {k:>5}"
        for m in C.RM_ORDERS:
            key = "R" if legacy else "num"
            sub = [r for r in rm
                   if (r["R"][str(m)].get(str(k)) is not None if legacy
                       else (r.get("num", {}).get(str(m), {}) or {}).get(str(k))
                       is not None)]
            if n_clusters(sub) < C.MIN_INFORMATIVE_PER_CELL:
                row += f"{'n<min':>26}"
                continue
            if legacy:
                f = lambda rs: wmean([rs_["R"][str(m)][str(k)] for rs_ in rs],
                                     native_weights(rs))
            else:
                f = lambda rs, _m=m, _k=k: _ratio_of_means(rs, _m, _k)
            row += f"{fmt(*hierarchical_bootstrap(sub, f), pct=True, nd=1):>26}"
        print(row)

    # H2 wording gate -- decided by the CI, never written in advance
    if legacy:
        sub = [r for r in rm if r["R"]["2"].get(str(K - 1)) is not None]
        f = lambda rs: wmean([r["R"]["2"][str(K - 1)] for r in rs],
                             native_weights(rs))
    else:
        sub = [r for r in rm
               if (r.get("num", {}).get("2", {}) or {}).get(str(K - 1)) is not None]
        f = lambda rs: _ratio_of_means(rs, 2, K - 1)
    if n_clusters(sub) >= C.MIN_INFORMATIVE_PER_CELL:
        _, lo, hi = hierarchical_bootstrap(sub, f)
        if lo > 0.8:
            v = "near-complete recovery"
        elif lo > 0.5:
            v = "a substantial fraction"
        elif hi < 0.5:
            v = "FALSIFIED: the second-order-local explanation does not hold here"
        else:
            v = "indeterminate at this n"
        print(f"\n   H2 gate on R_2 at the deepest slot: LCB={lo:.3f} UCB={hi:.3f}"
              f"\n   -> permitted wording: {v}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cheap", nargs="+", required=True)
    ap.add_argument("--nll", nargs="*", default=[])
    ap.add_argument("--rm", nargs="*", default=[])
    ap.add_argument("--slot", type=int, default=C.GAMMA - 1)
    ap.add_argument("--by-stratum", action="store_true")
    args = ap.parse_args()

    cheap = []
    for pat in args.cheap:
        for path in sorted(glob.glob(pat)):
            cheap += [json.loads(l) for l in open(path) if l.strip()]
    print(f"loaded {len(cheap)} cheap-probe anchors "
          f"from {len(args.cheap)} pattern(s)")
    if any("converged" in r for r in cheap):
        print("  !! this file predates the fixed-M protocol (it carries a "
              "`converged` flag from\n     the retired per-anchor SE rule). "
              "Its numbers are still usable, but the\n     threshold-ambiguity "
              "diagnostic below is unavailable. Rerun probe_cheap.")
    ms = sorted({r.get("M") for r in cheap if r.get("M")})
    if len(ms) > 1:
        print(f"  note: mixed M across anchors {ms} -- boundary escalation fired; "
              f"this is expected")

    tail_report(cheap, args.slot, label="ALL (workload-native weights)")

    if args.by_stratum:
        by = collections.defaultdict(list)
        for r in cheap:
            by[r["stratum"]].append(r)
        for s in sorted(by):
            tail_report(by[s], args.slot, label=f"stratum {s}")

    if args.rm:
        rm = []
        for pat in args.rm:
            for path in sorted(glob.glob(pat)):
                rm += [json.loads(l) for l in open(path) if l.strip()]
        rm_report(rm)

    if args.nll:
        nll = []
        for pat in args.nll:
            for path in sorted(glob.glob(pat)):
                nll += [json.loads(l) for l in open(path) if l.strip()]
        decomposition(cheap, nll, args.slot)


if __name__ == "__main__":
    main()
