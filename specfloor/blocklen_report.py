r"""The floor at a longer block: T^(0) and T^(1) at every slot of a gamma=16 run.

  python -m specfloor.blocklen_report                 # measurements/g16/
  python -m specfloor.blocklen_report --tk 'runs/g16/*.g16.jsonl'

probe_tk run with SPECFLOOR_GAMMA=16 records the same fields as the gamma=7
replication arm, over sixteen slots. This prints what the longer-block check
reads off it: the pooled floors at every slot, the share of the order-0 floor
one realised token removes, the final-slot floor per domain, and the slot-6
order-1 interval beside the gamma=7 run's at the same slot.

T^(1) here is the reweighting route's estimate on all paths, the column the
gamma=7 replication run (t1_fix/) reports; T_split is printed as the fit check.

Anchors are the gamma=7 ones with at least 16 tokens of continuation, weighted
by their gamma=7 inclusion probabilities, so pooled numbers describe a slightly
smaller population than the gamma=7 run's. The per-anchor seeds are the same,
so the first seven tokens of every path coincide with t1_fix/'s and slots 0-6
of the two runs are the same measurement: the paired comparison printed last
should agree to engine precision, and a gap there means the two runs did not
compute the same estimator. That is how the first gamma=16 run, made with a
probe_tk that predated the forced-suffix slot fix, gave itself away.
"""

from __future__ import annotations

import argparse
import statistics

from specfloor import config as C
from specfloor.records import archived, load_by_domain, weight
from specfloor.ratio_report import interval, nested, records


def floor_mean(recs, m, k, field="T"):
    pairs = [(v, weight(r)) for r in recs
             if (v := nested(field, m, k)(r)) is not None]
    return sum(v * w for v, w in pairs) / sum(w for _, w in pairs) if pairs else float("nan")


def table(by):
    """Per slot: T0, T1, T1_split and the share of T0 that T1 removes."""
    recs = [r for rs in by.values() for r in rs]
    K = 1 + max(int(k) for r in recs for k in (r.get("T") or {}).get("0", {}))
    rows = []
    for k in range(K):
        T0 = floor_mean(recs, 0, k)
        T1 = floor_mean(recs, 1, k) if k >= 1 else None
        rows.append(dict(k=k, T0=T0, T1=T1,
                         T1s=floor_mean(recs, 1, k, "T_split") if k >= 1 else None,
                         removed=None if T1 is None or T0 <= 0 else 1 - T1 / T0))
    return recs, K, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tk", default=archived("g16/*.g16.jsonl.gz"))
    ap.add_argument("--short", default="t1_fix/*.t01.jsonl.gz",
                    help="the gamma=7 run to set slot 6 beside (inside measurements/)")
    ap.add_argument("--boot", type=int, default=C.BOOTSTRAP_B)
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()

    by, bad = load_by_domain(args.tk)
    if not by:
        raise SystemExit(f"no probe_tk records matched {args.tk!r}")
    for dom, n in sorted(bad.items()):
        print(f"!! {dom}: {n} malformed line(s)")
    recs, K, rows = table(by)
    ctx = statistics.median(r["context"] for r in recs)
    print(f"gamma={K}: {len(recs)} anchors over {len({r['prompt_id'] for r in recs})} "
          f"prompts, median context {ctx:g}; M={recs[0].get('M')} "
          f"top-{recs[0].get('top_k')}; Hajek weights 1/pi")
    print(f"{'k':>3} | {'T^(0)':>7} {'T^(1)':>8} {'T^(1) split':>11} | {'removed':>8}")
    for r in rows:
        t1 = "--" if r["T1"] is None else f"{r['T1']:.4f}"
        ts = "--" if r["T1s"] is None else f"{r['T1s']:.4f}"
        rm = "--" if r["removed"] is None else f"{r['removed']:.1%}"
        print(f"{r['k']:>3} | {r['T0']:>7.4f} {t1:>8} {ts:>11} | {rm:>8}")
    deep = [r for r in rows if r["k"] >= 2]
    print(f"\nslots 2-{K - 1}: one realised token removes "
          f"{min(r['removed'] for r in deep):.1%}-{max(r['removed'] for r in deep):.1%}"
          f" of T^(0); T^(1) never exceeds {max(r['T1'] for r in rows[1:]):.4f}")
    last = rows[-1]["T0"]
    print(f"slot {K - 1}: T^(0) = {last:.4f}, so an all-parallel drafter's per-slot "
          f"acceptance there is at most {1 - last:.1%}")
    print("slot %d T^(0) by domain: " % (K - 1) + "  ".join(
        f"{d} {floor_mean(rs, 0, K - 1):.3f}" for d, rs in sorted(by.items())))

    short = {(r["prompt_id"], r["t"]): r for r in records([args.short])}
    mine = [r for r in recs if (r["prompt_id"], r["t"]) in short]
    theirs = [short[(r["prompt_id"], r["t"])] for r in mine]
    K7 = 1 + max(int(s) for r in theirs for s in (r.get("T") or {}).get("0", {}))
    print(f"\nagainst {args.short} on the {len(mine)} shared anchors, slots 1-{K7 - 1}"
          f" (same seeds, same paths):")
    for m in (0, 1):
        d = [floor_mean(mine, m, k) - floor_mean(theirs, m, k) for k in range(1, K7)]
        print(f"  T^({m}) gamma={K} minus gamma={K7}: "
              + " ".join(f"{x:+.4f}" for x in d))
    k = min(6, K - 1)
    a = interval(recs, nested("T", 1, k), B=args.boot, seed=args.seed)
    b = interval(list(short.values()), nested("T", 1, k), B=args.boot, seed=args.seed)
    print(f"  T^(1) at slot {k}: gamma={K} {a[0]:.4f} [{a[1]:.3f},{a[2]:.3f}]   "
          f"gamma={K7} {b[0]:.4f} [{b[1]:.3f},{b[2]:.3f}]   (prompt bootstrap)")


if __name__ == "__main__":
    main()
