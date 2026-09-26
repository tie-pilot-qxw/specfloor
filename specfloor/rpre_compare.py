"""Two order-1 drafters on the SAME anchors: paired per-slot rejection-risk differences.

`rpre_report` reads one arm. Comparing two by eye across two of its tables throws away
the pairing that makes the comparison sharp: both runs roll out the target from the same
corpus, the same anchor file and the same per-anchor seed, so T^(0) and T^(1) are the
same numbers in both and every difference is a within-anchor difference of the drafter
alone. Bootstrapping the two means separately would give intervals several times wider
and would answer a question nobody asked.

So this pairs on (domain, prompt_id, t), REFUSES to run if the two anchor sets are not
identical -- an unpaired comparison that merely looks paired is the failure mode worth
spending an assert on -- and cluster-bootstraps the DIFFERENCE over prompts, which is the
same outer unit rpre_report uses.

    python -m specfloor.rpre_compare --a 'runs/rpre_o1/*.rpre1.jsonl' \
                                     --b 'runs/rpre_o1_ours/*.rpre1.jsonl' \
                                     --label-a dspark --label-b ours

Columns are the decomposition of eq. (3) in the paper: R^self = T^(1) + G_post + E^exp.
A NEGATIVE diff on Gpost is an improvement (less risk the floor does not excuse); a
positive diff on exp means the head became MORE sensitive to a wrong predecessor. Both
are reported because a head can buy the first with the second.
"""

from __future__ import annotations

import argparse
import random

from specfloor import config as C
from specfloor.rpre_report import cells1, load, wmean

COLS = [("T1s", "T1"), ("Rorc", "R orac"), ("Gpost", "Gpost"),
        ("exp", "expo"), ("Rself", "R self")]


def keyed(by_dom, k):
    """(domain, prompt_id, t) -> cell, for one slot."""
    out = {}
    for dom, recs in by_dom.items():
        for r, c in zip(recs, cells1(recs, k)):
            out[(dom, r["prompt_id"], r["t"])] = c
    return out


def paired(a, b, k):
    ka, kb = keyed(a, k), keyed(b, k)
    common = sorted(set(ka) & set(kb))
    only_a, only_b = len(set(ka) - set(kb)), len(set(kb) - set(ka))
    assert not (only_a or only_b), (
        f"slot {k}: anchor sets differ ({only_a} only in A, {only_b} only in B). "
        f"These runs are not paired -- compare them with rpre_report, or re-run B on "
        f"A's corpus and anchor files.")
    return [(key, ka[key], kb[key]) for key in common]


def boot_diff(rows, col, B, seed):
    """Prompt-level cluster bootstrap of the weighted mean of (B - A)."""
    by = {}
    for (dom, pid, _t), ca, cb in rows:
        if ca.get(col) is None or cb.get(col) is None:
            continue
        by.setdefault((dom, pid), []).append((cb[col] - ca[col], ca["w"]))
    keys = list(by)
    if not keys:
        return None, None, None
    point = wmean([p for kk in keys for p in by[kk]])
    rng = random.Random(seed)
    draws = []
    for _ in range(B):
        samp = []
        for _ in range(len(keys)):
            samp += by[keys[rng.randrange(len(keys))]]
        draws.append(wmean(samp))
    draws.sort()
    return point, draws[int(0.025 * B)], draws[int(0.975 * B)]


def decomposition(a, b, K=None):
    """Per slot, both arms: the paired Hajek means of eq. (3)'s terms.

    {slot: {metric: (a, b)}} for floor T^(1), oracle risk, model gap G^(1),
    exposure and self-conditioned risk. Slot 0 has no predecessor to reveal, so
    its population floor is exactly zero and the gap there is the whole oracle
    risk; the estimator returns ~1e-8 at that slot, and it is zeroed rather than
    left to leak into the gap. These are the values behind the solution figure
    and its table.
    """
    if K is None:
        K = 1 + max(int(s) for r in next(iter(a.values())) for s in (r.get("R") or {}))
    out = {}
    for k in range(K):
        rows = paired(a, b, k)
        res = {}
        for c in (0, 1):
            w = [row[1 + c]["w"] for row in rows]
            get = lambda key: wmean([(row[1 + c][key], wt) for row, wt in zip(rows, w)])
            floor = 0.0 if k == 0 else get("T1s")
            orac, self_ = get("Rorc"), get("Rself")
            res[c] = dict(floor=floor, oracle_risk=orac, model_gap=orac - floor,
                          exposure=self_ - orac, self_risk=self_)
        out[k] = {m: (res[0][m], res[1][m]) for m in res[0]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="baseline glob")
    ap.add_argument("--b", required=True, help="arm glob")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--boot", type=int, default=C.BOOTSTRAP_B)
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()

    a, bad_a = load(args.a)
    b, bad_b = load(args.b)
    assert a and b, f"nothing loaded: A={sorted(a)} B={sorted(b)}"
    assert sorted(a) == sorted(b), (
        f"different domains: A={sorted(a)} vs B={sorted(b)}")
    if bad_a or bad_b:
        print(f"unparseable lines: A={bad_a} B={bad_b}")
    K = 1 + max(int(s) for r in next(iter(a.values())) for s in (r.get("R") or {}))
    print(f"{args.label_b} - {args.label_a}   domains {sorted(a)}   "
          f"cluster bootstrap over prompts, B={args.boot}")
    for col, name in COLS:
        print(f"\n--- {name} ---")
        print(f"{'k':>2} {'n':>5} {args.label_a:>9} {args.label_b:>9} "
              f"{'diff':>9}   95% CI on the paired diff")
        for k in range(K):
            rows = paired(a, b, k)
            va = wmean([(ca[col], ca["w"]) for _, ca, _ in rows if ca.get(col) is not None])
            vb = wmean([(cb[col], cb["w"]) for _, _, cb in rows if cb.get(col) is not None])
            d, lo, hi = boot_diff(rows, col, args.boot, args.seed + k)
            if d is None:
                print(f"{k:>2} {len(rows):>5} {'--':>9} {'--':>9}")
                continue
            tag = "DOWN" if hi < 0 else ("UP" if lo > 0 else "")
            print(f"{k:>2} {len(rows):>5} {va:>9.4f} {vb:>9.4f} "
                  f"{d:>+9.4f}   [{lo:+.4f},{hi:+.4f}] {tag}")


if __name__ == "__main__":
    main()
