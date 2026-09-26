r"""What committing to one proposal costs: $T^{(0)} - T^{(0),K}$, pooled.

  python -m specfloor.kmedian_report --branch 'measurement_runs/branch/*.tb.jsonl'

$K = 1$ is computed by the same estimator on the same paths, so it reproduces
$T^{(0)}$ and the difference is a within-anchor quantity, bootstrapped as one and
clustered on prompts.

**This is not a tree drafter's ceiling.** The $K$-median puts $\min_j$ inside the
expectation and so prices an oracle that knows which of its $K$ proposals suits
the realised path. A verification tree accepts on a UNION over its candidate set
and gets no such oracle. See probe_kmedian's docstring.

**Read the sign, not the size, when the removed fraction is small.** Lloyd returns
a local optimum, so every $T^{(0),K}$ here is an upper bound on the true
$K$-median and every removed fraction is a lower bound. A large one is a finding;
a small one is not evidence that width is worth little. The restart spread beside
each cell is the only direct handle on how far the solver might have stopped
short -- zero spread across independent seeds means every restart agreed, which is
weak evidence of a global optimum and is reported as such, not as a certificate.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random

from specfloor.records import open_text


def load(pattern):
    by = {}
    for f in sorted(glob.glob(pattern)):
        dom = os.path.basename(f).split(".")[0]
        for line in open_text(f):
            line = line.strip()
            if line:
                by.setdefault(dom, []).append(json.loads(line))
    return by


def wmean(pairs):
    sw = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / sw if sw > 0 else float("nan")


def cells(recs, W, k):
    out = []
    for r in recs:
        one = (r.get("Tb") or {}).get("1", {}).get(str(k))
        wid = (r.get("Tb") or {}).get(str(W), {}).get(str(k))
        if one is None or wid is None:
            continue
        out.append(dict(pid=r["prompt_id"], w=1.0 / max(r.get("pi", 1.0), 1e-9),
                        T1=one, TW=wid, gain=one - wid,
                        sp=(r.get("spread") or {}).get(str(W), {}).get(str(k))))
    return out


def boot(cs, key, B, seed):
    by = {}
    for c in cs:
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


def spread_summary(recs, widths):
    """Restart spread over every (anchor, K > 1, slot) cell, weighted by 1/pi.

    The per-cell means in the table hide the shape: most cells agree across
    restarts exactly and a thin tail does not. Returned: the weighted share of
    cells with zero spread, the weighted 90th percentile, the largest single
    spread, and the largest per-(K, slot) weighted mean.
    """
    cells = []
    for r in recs:
        w = 1.0 / max(r.get("pi", 1.0), 1e-9)
        for W in widths:
            for k, s in ((r.get("spread") or {}).get(str(W)) or {}).items():
                if s is not None:
                    cells.append((s, w, W, k))
    if not cells:
        return None
    tot = sum(c[1] for c in cells)
    acc, p90 = 0.0, None
    for s, w, _, _ in sorted(cells, key=lambda c: c[0]):
        acc += w
        if p90 is None and acc >= 0.9 * tot:
            p90 = s
    means = {}
    for s, w, W, k in cells:
        means.setdefault((W, k), []).append((s, w))
    return dict(zero=sum(c[1] for c in cells if c[0] == 0) / tot, p90=p90,
                max=max(c[0] for c in cells),
                max_cell_mean=max(wmean(v) for v in means.values()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", required=True)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--by-domain", action="store_true")
    args = ap.parse_args()

    by = load(args.branch)
    if not by:
        raise SystemExit(f"no records matched {args.branch!r}")
    allr = [r for rs in by.values() for r in rs]
    widths = [W for W in (allr[0].get("widths") or []) if W != 1]
    slots = sorted({int(k) for r in allr for k in (r.get("Tb") or {}).get("1", {})})
    print(f"M={allr[0].get('M')}  widths={allr[0].get('widths')}  "
          f"bootstrap B={args.boot} over PROMPTS  Hajek weights 1/pi")

    groups = ([(d, by[d]) for d in sorted(by)] if args.by_domain else []) + \
             [("POOLED", allr)]
    for name, recs in groups:
        print(f"\n===== {name}  ({len(recs)} anchors, "
              f"{len({r['prompt_id'] for r in recs})} prompts) =====")
        for W in widths:
            print(f"\n  width K={W}")
            print(f"  {'k':>2} {'n':>4} | {'T^(0)':>8} {'T^(0),K':>9} "
                  f"{'gain':>8} {'95% CI on gain':>17} {'ceiling':>8} "
                  f"{'restart spread':>15}")
            for k in slots:
                cs = cells(recs, W, k)
                if len(cs) < 5:
                    continue
                g = lambda key: wmean([(c[key], c["w"]) for c in cs
                                       if c.get(key) is not None])
                T1, TW, gn = g("T1"), g("TW"), g("gain")
                lo, hi = boot(cs, "gain", args.boot, args.seed)
                sp = g("sp")
                print(f"  {k:>2} {len(cs):>4} | {T1:>8.4f} {TW:>9.4f} {gn:>8.4f} "
                      f"[{lo:>6.4f},{hi:>6.4f}] {1 - TW:>8.4f} {sp:>15.2e}")

    sp = spread_summary(allr, widths)
    if sp:
        print(f"\nrestart spread over all K>1 cells: {sp['zero']:.0%} of the weight "
              f"has zero spread, p90 {sp['p90']:.3f}, max {sp['max']:.2f}; the "
              f"largest per-(K, slot) mean is {sp['max_cell_mean']:.3f}")

    print("\nT^(0),K is an UPPER bound on the true K-median, so the gain is a LOWER")
    print("bound on what K proposals remove. A large gain is a finding; a small one")
    print("is consistent with the solver having stopped short and establishes nothing.")
    print("This prices an ORACLE ROUTER, not a verification tree's acceptance.")


if __name__ == "__main__":
    main()
