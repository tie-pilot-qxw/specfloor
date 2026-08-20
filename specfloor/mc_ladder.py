"""MC convergence ladder: the evidence that MAIN_M is large enough.

The M-convergence evidence. With the per-anchor SE target withdrawn, this is the
ONLY formal justification for MAIN_M -- there is no per-anchor accuracy
guarantee standing behind it. The claim it must support is narrow and specific:

    the AGGREGATE statistics we publish do not move materially between
    M = MAIN_M and the largest M we can afford

"materially" is defined against the sampling CI, not against zero. A shift of
0.002 nats matters if the CI half-width is 0.001 and is irrelevant if it is 0.4.

What makes this comparison sharp is that all arms read ONE anchor file, so
anchor-to-anchor variation -- which dwarfs MC variation -- cancels in the
difference. The M-to-M shift is therefore bootstrapped as a PAIRED quantity,
with the anchor's own prompt as the resampling unit exactly as in stats.py.

The arms are also nested in their PATHS, which tightens the comparison further.
sample_paths seeds request i with seed_base + i, so M=32's requests carry the
same seeds as the first 32 of M=64's; under deterministic inference those are the
same paths. VERIFIED on this build: sampling 8 and then 16 paths from one prefix
with the same seed base gave 8/8 identical leading paths. (This depends on
`enable_deterministic_inference` -- without it sampling_seed is silently ignored
and the arms would be independent, not nested. backend.py refuses to start in
that case.)

The pairing does not RELY on path nesting: it is by anchor, and valid either way.
`check_nesting` verifies the anchor sets; the path-level property is a bonus that
makes the M-to-M differences smaller than the arm-to-arm CIs would suggest.

Usage:
  python -m specfloor.mc_ladder --arms 'runs/pilot/ladder.M*.jsonl' --slot 6
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import re

from specfloor import config as C
from specfloor.stats import (fmt, hierarchical_bootstrap, n_clusters,
                               native_weights, wmean)

# Two-tier standard, on the POINT estimate of the shift.
#   RATIO_OK   preferred: residual M-bias is negligible against sampling noise.
#   RATIO_MAX  hard ceiling: above this the M-bias is a material fraction of the
#              interval and the statistic cannot be quoted at this M at all.
# Between the two, the number is usable but the residual must be reported with
# it. Keeping the tiers distinct matters because domains differ: a cell can sit
# in the middle band for lack of power rather than for real M-sensitivity, and
# collapsing that into a single pass/fail hides which one it is.
RATIO_OK = 0.25
RATIO_MAX = 0.50


def load_arms(patterns):
    arms = {}
    for pat in patterns:
        matched = sorted(glob.glob(pat))
        if not matched:
            raise SystemExit(f"pattern matched no files: {pat}")
        for path in matched:
            recs = [json.loads(l) for l in open(path) if l.strip()]
            if not recs:
                # An empty arm means a crashed run, not an absent one. Skipping
                # it silently would drop an arm from the ladder and still print
                # a confident verdict over the survivors.
                raise SystemExit(
                    f"{path} is EMPTY -- that arm crashed. Delete it and rerun; "
                    f"do not compare the remaining arms as if it were absent.")
            ms = {r["M"] for r in recs}
            if len(ms) != 1:
                raise SystemExit(
                    f"{path} mixes M={sorted(ms)} -- an arm must be a FIXED M. "
                    f"Rerun with --m-base X --m-max X so escalation cannot fire.")
            arms[ms.pop()] = recs
    if len(arms) < 2:
        raise SystemExit(f"need at least two arms, found {sorted(arms)}")
    return dict(sorted(arms.items()))


def check_nesting(arms):
    """The arms must cover the same anchors; report whether they do."""
    keys = {m: {(r["prompt_id"], r["t"]) for r in recs} for m, recs in arms.items()}
    common = set.intersection(*keys.values())
    sizes = {m: len(k) for m, k in keys.items()}
    print(f"anchors per arm: {sizes}")
    print(f"common to all arms: {len(common)}  "
          f"({n_clusters([{'prompt_id': p} for p, _ in common])} prompts)")
    if len(common) < min(sizes.values()):
        print(f"   !! arms are not on identical anchor sets; the comparison uses "
              f"the {len(common)} in common")
    return common


def restrict(recs, common):
    return [r for r in recs if (r["prompt_id"], r["t"]) in common]


# ------------------------------------------------------------ statistics ----
def make_stats(slot, thr):
    def dce(r):
        return r["dCE"][slot]

    def incidence(rs):
        return wmean([1.0 if dce(r) > thr else 0.0 for r in rs], native_weights(rs))

    def mean_all(rs):
        return wmean([dce(r) for r in rs], native_weights(rs))

    def mean_inf(rs):
        sub = [r for r in rs if dce(r) > thr]
        if not sub:
            raise ValueError
        return wmean([dce(r) for r in sub], native_weights(sub))

    def wq(rs, f):
        sub = sorted(((dce(r), 1.0 / max(r.get("pi", 1.0), 1e-9))
                      for r in rs if dce(r) > thr), key=lambda x: x[0])
        if not sub:
            raise ValueError
        tot = sum(w for _, w in sub)
        acc = 0.0
        for v, w in sub:
            acc += w
            if acc >= f * tot:
                return v
        return sub[-1][0]

    def ambiguity(rs):
        """Fraction of anchors whose informative/not classification is still
        noise-limited. This is the statistic that most directly answers "is M
        large enough for the THRESHOLD to mean anything" -- SE falls as
        1/sqrt(M), so this must fall with M or the incidence headline is not
        supportable at any M we can afford."""
        return sum(1.0 for r in rs
                   if slot in (r.get("ambiguous_slots") or [])) / max(1, len(rs))

    return [
        ("P(dCE>eps)", incidence, True),
        ("ambiguous", ambiguity, True),
        ("E[dCE]", mean_all, False),
        ("E[dCE | inf]", mean_inf, False),
        ("p50 | inf", lambda rs: wq(rs, .50), False),
        ("p90 | inf", lambda rs: wq(rs, .90), False),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--slot", type=int, default=C.GAMMA - 1)
    ap.add_argument("--eps", type=float, default=C.INFORMATIVE_THRESHOLD)
    ap.add_argument("--ref", type=int, default=C.MAIN_M,
                    help="the M being justified")
    args = ap.parse_args()

    arms = load_arms(args.arms)
    common = check_nesting(arms)
    arms = {m: restrict(r, common) for m, r in arms.items()}
    Ms = sorted(arms)
    top = Ms[-1]
    print(f"\nslot {args.slot}, eps={args.eps}, "
          f"reference M={args.ref}, richest M={top}\n")

    stats = make_stats(args.slot, args.eps)
    table = {}
    for name, f, pct in stats:
        row = {}
        for m in Ms:
            try:
                row[m] = hierarchical_bootstrap(arms[m], f)
            except (ValueError, ZeroDivisionError):
                row[m] = (float("nan"),) * 3
        table[name] = row

    print("  " + " " * 14 + "".join(f"{'M=' + str(m):>28}" for m in Ms))
    for name, _f, pct in stats:
        cells = "".join(f"{fmt(*table[name][m], pct=pct, nd=1 if pct else 4):>28}"
                        for m in Ms)
        print(f"  {name:<14}{cells}")

    # ---- the actual criterion -------------------------------------------
    print(f"\n== convergence criterion: does the statistic move between "
          f"M={args.ref} and M={top},")
    print(f"   compared with the sampling CI at M={args.ref}?\n")
    verdict_ok = True
    any_over_max = False
    for name, f, pct in stats:
        if name == "ambiguous":
            # NOT a stability statistic. SE falls as 1/sqrt(M), so this SHOULD
            # shrink with M; holding still would be the bad outcome. It is a
            # LEVEL to report at the reference M, not a shift to bound.
            lo_m, ref_v = Ms[0], table[name][args.ref][0]
            print(f"  [note ] {'ambiguous':<14} "
                  f"{table[name][lo_m][0]:.1%} at M={lo_m} -> {ref_v:.1%} at "
                  f"M={args.ref} -> {table[name][top][0]:.1%} at M={top}")
            print(f"         {'':<14} falls with M as it must; quote the "
                  f"M={args.ref} value beside every incidence figure")
            continue
        ref, hi = table[name].get(args.ref), table[name].get(top)
        if ref is None or hi is None or ref[0] != ref[0] or hi[0] != hi[0]:
            print(f"  {name:<14} unavailable")
            continue
        shift = abs(hi[0] - ref[0])
        # Denominator is the CI half-width at the RICHER arm: the error budget
        # is "MC error as a fraction of sampling uncertainty", and the best
        # available estimate of that uncertainty is the one from the most
        # precise arm. Numerically almost identical to using the reference arm
        # (the sampling CI is driven by prompt resampling, not by M), but it is
        # the right reference by definition.
        half = (hi[2] - hi[1]) / 2
        # paired: the arms are nested, so the difference is far better
        # determined than either endpoint. Bootstrap it directly.
        pair = {(r["prompt_id"], r["t"]): r for r in arms[top]}
        merged = [{**r, "_hi": pair[(r["prompt_id"], r["t"])]}
                  for r in arms[args.ref]]

        def diff(rs, _f=f):
            return _f([x["_hi"] for x in rs]) - _f(rs)

        try:
            d, dlo, dhi = hierarchical_bootstrap(merged, diff)
        except (ValueError, ZeroDivisionError):
            d = dlo = dhi = float("nan")
        s = 100 if pct else 1
        u = "%" if pct else ""
        # The criterion is MAGNITUDE relative to the sampling CI, NOT statistical
        # significance of the difference. The arms are nested, so the paired CI
        # is very tight and almost any real shift excludes zero -- using that as
        # the test would reject every M we could afford. The paired CI is still
        # printed, because it distinguishes "small and real" from "small and
        # noise", which changes whether the residual is worth reporting.
        ratio = shift / half if half == half and half > 0 else float("inf")
        ok = ratio < RATIO_OK
        over_max = ratio >= RATIO_MAX
        verdict_ok &= ok
        any_over_max |= over_max
        real = not (dlo <= 0 <= dhi) if dlo == dlo else False
        tag = "ok   " if ok else ("OVER " if over_max else "band ")
        print(f"  [{tag}] {name:<14} "
              f"shift {shift*s:+.4f}{u} = {ratio:.2f} x CI half-width"
              f"   paired {d*s:+.4f} [{dlo*s:+.4f}, {dhi*s:+.4f}]{u}"
              f"{'  (real)' if real else '  (within noise)'}")

    # A convergence test whose denominator is huge passes for the wrong reason:
    # the criterion is shift / CI half-width, so a thin sample with wide CIs
    # certifies any M. Report the power explicitly rather than let a vacuous
    # pass read like evidence.
    n_cl = n_clusters(arms[args.ref])
    thin = n_cl < 30
    print()
    if thin:
        print(f"  !! POWER WARNING: only {n_cl} prompts in this cell. The "
              f"criterion divides by the\n     sampling CI, so a small sample "
              f"makes every M look converged. Treat a pass here\n     as "
              f"'not contradicted' rather than 'validated'.")
    if verdict_ok:
        print(f"VERDICT: M={args.ref} PASSES the preferred bar"
              f"{' (but see the power warning)' if thin else ''} -- every "
              f"statistic's shift to\n  M={top} is under {RATIO_OK} x its own "
              f"sampling CI half-width.")
    elif any_over_max:
        print(f"VERDICT: M={args.ref} FAILS -- at least one statistic exceeds "
              f"the {RATIO_MAX} hard ceiling.\n  The residual M-bias is a "
              f"material fraction of the interval; this M cannot be used.")
    else:
        print(f"VERDICT: M={args.ref} is in the MIDDLE BAND "
              f"[{RATIO_OK}, {RATIO_MAX}) -- usable, but the residual\n  "
              f"M-sensitivity must be reported alongside the statistics marked "
              f"`band` above.\n  Check whether that is real M-dependence or "
              f"just low power: if the paired CI\n  covers zero, it is the "
              f"latter and more ANCHORS (not more paths) is the fix.")
    print("\nNOTE: this justifies M for the CHEAP pass only. R_m rides on "
          "RM_MIXED_PATHS and\n  has its own path count; the jackknife guard "
          "matters there, not here.")


if __name__ == "__main__":
    main()
