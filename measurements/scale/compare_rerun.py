"""Check a re-run against the run it replaces before anything is overwritten.

The OOM fix changes no arithmetic -- it removes a retained autograd graph from
a forward pass -- so a re-run should differ from its predecessor only by the
anchors that used to be skipped. What it cannot promise is bit-exactness: the
probe SAMPLES, and the target's kernels are not bitwise reproducible across
processes, so a logit difference of 1e-7 occasionally flips a drawn token and
that path's whole trajectory diverges. With M paths per anchor a handful of
flips move a per-anchor mean by O(1/M), which is 4e-3 at M = 256 and shows up
as a few large-looking per-cell deltas.

Three things are therefore checked, in increasing order of what they can tell
you:

  chunk      must match EXACTLY. It is a deterministic function of (context,
             K, budget) and it controls how the sampler's random stream is
             consumed, so a mismatch means the two runs were not given the
             same budget and nothing below is comparable.
  aggregate  the HT-weighted mean over SHARED anchors, per slot, which is what
             the paper reports. This is the number that must not move.
  per-cell   reported for orientation only, with the count above tolerance.

  python compare_rerun.py <old.jsonl> <new.jsonl> [--tol 5e-3]
"""
from __future__ import annotations

import argparse
import json
import math


def load(path):
    rows = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                rows[(r["prompt_id"], r["t"])] = r
    return rows


def wmean(pairs):
    W = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / W if W else float("nan")


def leaves(a, b, path=""):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            out += leaves(a.get(k), b.get(k), f"{path}.{k}")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not (isinstance(a, float) and math.isnan(a)):
            out.append((path, a, b))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--tol", type=float, default=5e-3,
                    help="per-cell delta above which a cell is counted as moved")
    ap.add_argument("--agg-tol", type=float, default=2e-3,
                    help="aggregate shift above which the re-run is refused")
    args = ap.parse_args()

    old, new = load(args.old), load(args.new)
    shared = sorted(set(old) & set(new))
    gained = sorted(set(new) - set(old))
    lost = sorted(set(old) - set(new))
    print(f"old {len(old):3d} rows   new {len(new):3d} rows   shared {len(shared)}   "
          f"gained {len(gained)}   lost {len(lost)}")

    bad_chunk = [k for k in shared if old[k].get("chunk") != new[k].get("chunk")]
    if bad_chunk:
        print(f"  !! chunk differs on {len(bad_chunk)}/{len(shared)} anchors, e.g. "
              f"{old[bad_chunk[0]]['chunk']} -> {new[bad_chunk[0]]['chunk']} at "
              f"ctx {old[bad_chunk[0]]['context']}. The two runs had different "
              f"kv budgets; they are not comparable. DO NOT INSTALL")
        return 1
    print(f"  chunk identical on all {len(shared)} shared anchors")

    # Only the risk/floor fields are in [0, 1] and comparable; `alive`, `rec`
    # and the group counts are integers whose deltas mean something else.
    VAL = (".T", ".R", ".G", ".R_self", ".R_temp", ".T_split", ".T1", ".S",
           ".abar", ".R_serve")
    moved = worst = 0
    for k in shared:
        for p, x, y in leaves(old[k], new[k]):
            if not p.startswith(VAL) or p.endswith(("groups", "groups2")):
                continue   # group counts are integers; their deltas mean something else
            d = abs(x - y)
            worst = max(worst, d)
            moved += d > args.tol

    fail = False
    for fld in ("T", "R", "R_self", "G"):
        rows = [k for k in shared if isinstance(old[k].get(fld), dict) and old[k][fld]]
        if not rows:
            continue
        line, hi = [], 0.0
        for s in range(7):
            a = [(old[k][fld][str(s)], 1 / old[k]["pi"]) for k in rows
                 if old[k][fld].get(str(s)) is not None]
            b = [(new[k][fld][str(s)], 1 / new[k]["pi"]) for k in rows
                 if (new[k].get(fld) or {}).get(str(s)) is not None]
            if not a or not b:
                continue
            d = wmean(b) - wmean(a)
            hi = max(hi, abs(d))
            line.append(f"{d:+.5f}")
        print(f"  {fld:7s} aggregate shift by slot: " + " ".join(line) +
              f"   max |shift| {hi:.5f}")
        # With a handful of shared anchors the aggregate has no power: a single
        # divergent path moves it by O(1/(n*M)). Report it, do not gate on it.
        if len(rows) >= 30:
            fail |= hi > args.agg_tol

    print(f"  per-cell: {moved} value leaves moved by more than {args.tol:g}, "
          f"largest {worst:.3e} (path divergence, not arithmetic)")
    if len(shared) < 30:
        print(f"  note: only {len(shared)} shared anchors, so the aggregate is "
              f"not gated on -- it has no power at this n")
    if lost:
        print(f"  !! {len(lost)} anchors present before and missing now")
    if gained:
        ctx = sorted(new[k]["context"] for k in gained)
        print(f"  recovered {len(gained)} anchors, context {ctx[0]}-{ctx[-1]} "
              f"(median {ctx[len(ctx) // 2]})")
    if fail or lost:
        print("  DO NOT INSTALL")
        return 1
    print("  OK to install")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
