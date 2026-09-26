r"""The headline shares, with intervals: G/R, what one token removes, and the floors behind them.

  python -m specfloor.ratio_report

rpre_report bootstraps G, a paired difference. The paper also quotes SHARES --
the part of a drafter's rejection that is model gap, G/R, and the part of the
order-0 floor one realised token removes, (T0 - T1)/T0. A share is a ratio of
two Hajek means over the same anchors, so its interval comes from resampling
prompts once and dividing inside every replicate (stats.ratio_interval), never
from the two means' separate intervals, which would ignore that they move
together.

What is printed, all at the final slot unless a slot is named:

* the floor intervals: T0 from the M=1024 top-256 run, T1 from the
  partitioning route on the M=256 full-vocabulary run, and the share one token
  removes, whose T0 and T1 are read off the SAME paths;
* the slot-1 order-1 residual, zero by identity, which is the partitioning
  estimator's numerical resolution;
* the paired difference between the two T1 routes on shared anchors;
* for each target, the DFlash decomposition T0 / R / G/R and the DSpark one
  T1 / R_oracle / G1/R_oracle, which are the bars of the cross-target figure.

Qwen3-4B reads its decompositions from rpre/ and rpre_o1/; the three larger
targets from scale/, where Qwen3-14B's arena-hard lives in its regenerated run.
"""

from __future__ import annotations

import argparse

from specfloor import config as C
from specfloor.records import archived, load_by_domain, weight
from specfloor.stats import ratio_interval

# target -> (order-0 DFlash globs, order-1 DSpark globs), inside measurements/
TARGETS = {
    "Qwen3-4B": (["rpre/*.rpre.jsonl.gz"], ["rpre_o1/*.rpre1.jsonl.gz"]),
    "Qwen3-8B": (["scale/qwen8b/*.srv0.jsonl.gz"], ["scale/qwen8b/*.srv1.jsonl.gz"]),
    "Qwen3-14B": (["scale/qwen14b/*.srv0.jsonl.gz", "scale/qwen14b_arena/*.srv0.jsonl.gz"],
                  ["scale/qwen14b/*.srv1.jsonl.gz", "scale/qwen14b_arena/*.srv1.jsonl.gz"]),
    "Gemma-4-12B": (["scale/gemma12b_fix/*.srv0.jsonl.gz"],
                    ["scale/gemma12b_fix/*.srv1.jsonl.gz"]),
}


def records(patterns):
    """Every record under globs inside measurements/, in sorted file order."""
    out = []
    for p in patterns:
        by, _ = load_by_domain(archived(p))
        out += [r for recs in by.values() for r in recs]
    return out


def hajek(recs, y):
    pairs = [(y(r), weight(r)) for r in recs if y(r) is not None]
    return sum(v * w for v, w in pairs) / sum(w for _, w in pairs)


def interval(recs, y, x=None, B=C.BOOTSTRAP_B, seed=C.SEED):
    """Hajek mean of y -- or, with x, the ratio of the Hajek means of y and x --
    over the records where both are defined; (point, lo, hi, n)."""
    rows = [(r["prompt_id"], weight(r), y(r), 1.0 if x is None else x(r)) for r in recs]
    rows = [t for t in rows if t[2] is not None and t[3] is not None]
    p, lo, hi = ratio_interval([t[0] for t in rows], [t[1] * t[2] for t in rows],
                               [t[1] * t[3] for t in rows], B, seed)
    return float(p), float(lo), float(hi), len(rows)


def flat(field, k):
    return lambda r: (r.get(field) or {}).get(str(k))


def t1_split(k):
    return lambda r: ((r.get("T1") or {}).get(str(k)) or {}).get("split")


def nested(field, m, k):
    return lambda r: ((r.get(field) or {}).get(str(m)) or {}).get(str(k))


def minus(a, b):
    return lambda r: None if a(r) is None or b(r) is None else a(r) - b(r)


def floor_intervals(t0, o1, k, B=C.BOOTSTRAP_B, seed=C.SEED):
    """T0 (top-256, M=1024), T1 (partitioning, split-half), share removed."""
    return dict(
        T0=interval(t0, nested("T", 0, k), B=B, seed=seed),
        T1=interval(o1, t1_split(k), B=B, seed=seed),
        share=interval(o1, minus(flat("T", k), t1_split(k)), flat("T", k), B, seed))


def route_gap(snis, k):
    """Per-anchor partitioning minus reweighting T1, both held-out estimates."""
    other = {(r["prompt_id"], r["t"]): r for r in snis}

    def d(r):
        s = other.get((r["prompt_id"], r["t"]))
        b = None if s is None else nested("T_split", 1, k)(s)
        a = t1_split(k)(r)
        return None if a is None or b is None else a - b
    return d


def route_difference(o1, snis, k, B=C.BOOTSTRAP_B, seed=C.SEED):
    """The two T1 routes paired on shared anchors, weighted as rpre_o1."""
    return interval(o1, route_gap(snis, k), B=B, seed=seed)


def dflash_share(recs, k, B=C.BOOTSTRAP_B, seed=C.SEED):
    """T0, R and G/R = (R - T0)/R at slot k."""
    return dict(T=interval(recs, flat("T", k), B=B, seed=seed)[0],
                R=interval(recs, flat("R", k), B=B, seed=seed)[0],
                share=interval(recs, minus(flat("R", k), flat("T", k)), flat("R", k), B, seed))


def dspark_share(recs, k, B=C.BOOTSTRAP_B, seed=C.SEED):
    """T1, R_oracle, T0 and G1/R_oracle = (R_oracle - T1)/R_oracle at slot k."""
    rs = [r for r in recs if t1_split(k)(r) is not None]
    return dict(T1=interval(rs, t1_split(k), B=B, seed=seed)[0],
                R=interval(rs, flat("R", k), B=B, seed=seed)[0],
                T0=interval(rs, flat("T", k), B=B, seed=seed)[0],
                share=interval(rs, minus(flat("R", k), t1_split(k)), flat("R", k), B, seed))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=C.BOOTSTRAP_B)
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()
    B, seed, K = args.boot, args.seed, C.GAMMA

    t0 = records(["t0_s6/*.t0.jsonl.gz"])
    o1 = records(["rpre_o1/*.rpre1.jsonl.gz"])
    snis = records(["t1_fix/*.t01.jsonl.gz"])
    print(f"Qwen3-4B floors, {len(o1)} anchors; prompt bootstrap B={B}, "
          f"{C.CI:.0%} intervals, Hajek weights 1/pi")
    print(f"{'k':>2} | {'T0 top-256, M=1024':>26} | {'T1 partitioning':>26} | "
          f"{'(T0 - T1)/T0, same paths':>26}")
    for k in range(1, K):
        f = floor_intervals(t0, o1, k, B, seed)
        print(f"{k:>2} | {f['T0'][0]:.4f} [{f['T0'][1]:.4f},{f['T0'][2]:.4f}]"
              f"{'':>5} | {f['T1'][0]:.4f} [{f['T1'][1]:.4f},{f['T1'][2]:.4f}]{'':>5} |"
              f" {f['share'][0]:6.1%} [{f['share'][1]:6.1%},{f['share'][2]:6.1%}]")
    print("   slot 1: T1 is zero by identity; what the estimator returns there is "
          "its numerical resolution.")
    d = route_difference(o1, snis, K - 1, B, seed)
    print(f"\nT1 at slot {K - 1}, partitioning minus reweighting route, paired on "
          f"{d[3]} anchors: {d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}]")
    worst = max(abs(hajek(o1, route_gap(snis, k))) for k in range(1, K))
    print(f"largest |paired difference| over slots 1-{K - 1}: {worst:.4f}")

    print(f"\nfinal slot k={K - 1}, per target")
    print(f"{'':>12} | {'DFlash T0':>9} {'R':>7} {'G/R':>24} | "
          f"{'DSpark T1':>9} {'R_orac':>7} {'G1/R_orac':>24} {'R_orac > T0':>11}")
    for name, (g0, g1) in TARGETS.items():
        a = dflash_share(records(g0), K - 1, B, seed)
        b = dspark_share(records(g1), K - 1, B, seed)
        print(f"{name:>12} | {a['T']:>9.4f} {a['R']:>7.4f} "
              f"{a['share'][0]:6.1%} [{a['share'][1]:6.1%},{a['share'][2]:6.1%}] | "
              f"{b['T1']:>9.4f} {b['R']:>7.4f} "
              f"{b['share'][0]:6.1%} [{b['share'][1]:6.1%},{b['share'][2]:6.1%}] "
              f"{'yes' if b['R'] > b['T0'] else 'NO':>11}")
    print("\nG/R is the share of the drafter's rejection its own information does not")
    print("force; 1 - G/R is the share no proposal of that conditioning order removes.")


if __name__ == "__main__":
    main()
