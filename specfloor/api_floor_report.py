"""Pool probe_api_floor cells into the same table shape the local runs report.

  python -m specfloor.api_floor_report --in 'measurement_runs/api_v4/*.jsonl'

Two differences from the local reports, both forced by what an endpoint can do
and both visible in the output rather than buried.

RESIDUAL, NOT ESS, IS THE GATE HERE. The order-0 estimator carries no importance
weights, and the order-1 estimator is grouping rather than SNIS, so effective
sample size is not the binding quantity. What binds is the top-20 read: mass the
endpoint does not report is mass the TV cannot charge for, and it is charged
against T in the direction that flatters the target. Cells above --resid-gate
are dropped and counted.

ANCHORS ARE NOT PAIRED ACROSS TARGETS. The floor is a property of the target's
own conditional structure at positions the target itself would occupy, so each
target is anchored inside its OWN generations. Prompts are shared, responses are
not, and the comparison is therefore between domains rather than between matched
cells. A paired comparison is not available at any sample size, because the two
targets do not write the same text.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random


def load(pattern, gate):
    rows, dropped, kept = [], 0, 0
    for f in sorted(glob.glob(pattern)):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            for k, c in (r.get("slots") or {}).items():
                if c.get("resid", 0.0) > gate:
                    dropped += 1
                    continue
                kept += 1
                rows.append((r.get("domain", "?"), int(k), r["prompt_id"], c))
    return rows, dropped, kept


def boot(vals, keys, B, seed):
    """Prompt-level cluster bootstrap; unweighted, since anchors here are drawn
    uniformly within a response rather than by a stratified scheme with known pi."""
    by = {}
    for v, k in zip(vals, keys):
        by.setdefault(k, []).append(v)
    ks = list(by)
    rng = random.Random(seed)
    draws = []
    for _ in range(B):
        s = []
        for _ in range(len(ks)):
            s += by[ks[rng.randrange(len(ks))]]
        if s:
            draws.append(sum(s) / len(s))
    if not draws:
        return float("nan"), float("nan")
    draws.sort()
    q = lambda p: draws[max(0, min(len(draws) - 1, int(p * len(draws))))]
    return q(0.025), q(0.975)


def table(name, rows, args):
    sel = [r for r in rows if name in ("POOLED", r[0])]
    if not sel:
        return
    npr = len({r[2] for r in sel})
    print(f"\n===== {name}  ({len({(r[2], id(r[3])) for r in sel})} cells, {npr} prompts) =====")
    print(f"{'k':>2} {'n':>4} | {'T0':>7} {'95% CI':>17} {'T0split':>8} | {'T1plug':>7} "
          f"{'T1split':>8} {'cov':>5} {'grp':>5} | {'dT1/T0':>7} {'resid':>9} {'supp':>6}")
    for k in sorted({r[1] for r in sel}):
        cs = [r[3] for r in sel if r[1] == k]
        ks = [r[2] for r in sel if r[1] == k]
        if len(cs) < 5:
            continue
        m = lambda f: (lambda v: sum(v) / len(v) if v else float("nan"))(
            [x for x in (f(c) for c in cs) if x is not None and not math.isnan(x)])
        T0 = m(lambda c: c["T0"])
        T1s = m(lambda c: (c.get("T1") or {}).get("split"))
        lo, hi = boot([c["T0"] for c in cs], ks, args.boot, args.seed)
        d = (T0 - T1s) / T0 if T0 > 0 else float("nan")
        print(f"{k:>2} {len(cs):>4} | {T0:>7.4f} [{lo:>6.4f},{hi:>6.4f}] "
              f"{m(lambda c: c.get('T0_split')):>8.4f} | "
              f"{m(lambda c: (c.get('T1') or {}).get('plug')):>7.4f} {T1s:>8.4f} "
              f"{m(lambda c: (c.get('T1') or {}).get('cov')):>5.2f} "
              f"{m(lambda c: (c.get('T1') or {}).get('groups')):>5.1f} | "
              f"{d:>6.1%} {m(lambda c: c['resid']):>9.1e} {m(lambda c: c['support']):>6.0f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--resid-gate", type=float, default=1e-3)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--by-domain", action="store_true")
    args = ap.parse_args()

    rows, dropped, kept = load(args.inp, args.resid_gate)
    if not rows:
        raise SystemExit(f"no cells matched {args.inp!r}")
    print(f"top-20 read; residual gate <= {args.resid_gate:g}; {kept} cells kept, "
          f"{dropped} dropped ({100*dropped/max(kept+dropped,1):.1f}%)")
    print(f"bootstrap B={args.boot} over PROMPTS; unweighted means")
    if args.by_domain:
        for d in sorted({r[0] for r in rows}):
            table(d, rows, args)
    table("POOLED", rows, args)
    print("\nT0 is the product-measure floor and T1 the order-1 floor, both for the")
    print("TARGET alone. dT1/T0 is what one realised token removes. No drafter is")
    print("involved and no gap is estimated: R needs the target's hidden-state taps.")


if __name__ == "__main__":
    main()
