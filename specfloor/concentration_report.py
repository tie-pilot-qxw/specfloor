r"""How unevenly the floor is spread over contexts, and how concentrated the target's paths are.

  python -m specfloor.concentration_report

Two descriptive readings of the same anchors, both Hajek-weighted by 1/pi, so a
share of anchors below means a share of the WEIGHTED population of blocks, not
of the rows that happen to be in the file.

**Where the floor sits** (Sec. 3.1, from the full-vocabulary M=256 run). At a
slot, the share of anchors whose floor is below INFORMATIVE_THRESHOLD, the share
of the total floor they carry, and the share carried by the top decile --
anchors at or above the weighted 90th percentile. A mean of 0.286 can be a
mixture of many near-zero blocks and a few heavily branching ones, and these
three numbers say which.

**How many paths carry the mass** (App. "Continuation concentration", from the
M=1024 top-256 run, which records it). probe_tk stores the collision-equivalent
effective support N_eff = 1 / sum_z p(z)^2 of each j-token prefix, estimated
without the plug-in bias. Its weighted quantiles, and the weighted correlation
of log N_eff with the slot-k floor, overall and within each domain, show that
the floor tracks branching rather than context length or domain alone.

Weighted quantiles are the smallest value whose cumulative weight share reaches
q, the convention stats.tail_report uses.
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from specfloor import config as C
from specfloor.records import archived, load_by_domain, weight


def wquantile(values, weights, q):
    o = np.argsort(values, kind="stable")
    v, w = np.asarray(values, float)[o], np.asarray(weights, float)[o]
    return float(v[np.searchsorted(np.cumsum(w) / w.sum(), q)])


def wcorr(x, y, w):
    x, y, w = (np.asarray(a, float) for a in (x, y, w))
    mx, my = (w * x).sum() / w.sum(), (w * y).sum() / w.sum()
    return float((w * (x - mx) * (y - my)).sum()
                 / math.sqrt((w * (x - mx) ** 2).sum() * (w * (y - my) ** 2).sum()))


def floor_at(r, slot):
    """The order-0 floor at a slot, from either record shape: probe_rpre keeps
    it flat, probe_tk nests it under its conditioning order."""
    T = r.get("T") or {}
    if isinstance(T.get("0"), dict):
        return T["0"].get(str(slot))
    return T.get(str(slot))


def floor_concentration(recs, slot, eps=C.INFORMATIVE_THRESHOLD):
    rs = [r for r in recs if floor_at(r, slot) is not None]
    v = np.array([floor_at(r, slot) for r in rs])
    w = np.array([weight(r) for r in rs])
    mass = v * w
    top = v >= wquantile(v, w, 0.9)
    return dict(n=len(rs), mean=float(mass.sum() / w.sum()),
                median=wquantile(v, w, 0.5),
                below=float(w[v < eps].sum() / w.sum()),
                below_mass=float(mass[v < eps].sum() / mass.sum()),
                top_decile_mass=float(mass[top].sum() / mass.sum()))


def effective_support(by_domain, slot):
    """Weighted N_eff quantiles for the slot-long prefix, and corr with T_slot."""
    def rows(recs):
        return [r for r in recs if (r.get("n_eff") or {}).get(str(slot)) is not None
                and floor_at(r, slot) is not None]

    def summary(rs):
        ne = [r["n_eff"][str(slot)] for r in rs]
        T = [floor_at(r, slot) for r in rs]
        w = [weight(r) for r in rs]
        return dict(n=len(rs), median=wquantile(ne, w, 0.5), p90=wquantile(ne, w, 0.9),
                    min=min(ne), max=max(ne),
                    corr_log=wcorr(np.log(ne), T, w), corr_raw=wcorr(ne, T, w))

    allr = rows([r for recs in by_domain.values() for r in recs])
    return summary(allr), {d: summary(rows(recs)) for d, recs in sorted(by_domain.items())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--floors", default=archived("rpre/*.rpre.jsonl.gz"),
                    help="records carrying the per-anchor order-0 floor")
    ap.add_argument("--support", default=archived("t0_s6/*.t0.jsonl.gz"),
                    help="probe_tk records carrying n_eff")
    ap.add_argument("--slots", default="1,6")
    args = ap.parse_args()

    by, _ = load_by_domain(args.floors)
    recs = [r for rs in by.values() for r in rs]
    print(f"floor concentration: {len(recs)} anchors, Hajek weights 1/pi, "
          f"threshold {C.INFORMATIVE_THRESHOLD}")
    print(f"{'slot':>4} {'mean':>7} {'median':>7} | {'share < thr':>11} "
          f"{'their floor':>11} | {'top decile':>10}")
    for k in (int(s) for s in args.slots.split(",")):
        c = floor_concentration(recs, k)
        print(f"{k:>4} {c['mean']:>7.4f} {c['median']:>7.4f} | {c['below']:>11.1%} "
              f"{c['below_mass']:>11.2%} | {c['top_decile_mass']:>10.1%}")

    sby, _ = load_by_domain(args.support)
    slot = C.GAMMA - 1
    pooled, per = effective_support(sby, slot)
    print(f"\neffective support of the {slot}-token prefix (M="
          f"{next(iter(sby.values()))[0].get('M')}), against T_{slot}^(0) on the same paths")
    print(f"{'':>10} {'n':>4} {'median':>7} {'p90':>7} {'range':>16} | "
          f"{'corr(log N,T)':>13} {'corr(N,T)':>9}")
    for name, s in [("POOLED", pooled)] + list(per.items()):
        print(f"{name:>10} {s['n']:>4} {s['median']:>7.2f} {s['p90']:>7.1f} "
              f"{s['min']:>7.1f}-{s['max']:<8.1f} | {s['corr_log']:>+13.3f} "
              f"{s['corr_raw']:>+9.3f}")
    print("\nN_eff is recorded as M when no two rollouts share the prefix; it is a")
    print("descriptive concentration measure, not an input to any floor.")


if __name__ == "__main__":
    main()
