r"""Iterated best response: sweep the slots, refitting each against the others.

  python -m specfloor.br_iter --br 'measurement_runs/br/*.br0.jsonl' --rounds 4

**What this adds to br_report.** `br_report` reopens ONE slot and holds the other
six at the shipped proposal, so its seven columns are seven separate deviations
from the same base point and cannot be added. This module runs the obvious next
thing: sweep slot by slot, and when slot k is refitted let it see the actions
already installed at the other slots. Repeat until it stops moving.

**This needs no GPU.** `probe_br` recorded, per path per slot, the realised token
and the target and drafter masses at it. The water filling at slot k produces a
distribution supported on exactly the tokens some path realised there, so its
mass at any recorded path's token is readable offline. Every accept factor of
every round is therefore computable from the files already on disk -- which is
the whole reason that probe stores scalars instead of [M, V] rows.

**What converges, and to what.** Each coordinate update is the exact maximiser of
tau in that coordinate (br_report's water filling, appendix A.7), so tau on the
FIT half is non-decreasing along the sweep and bounded above by gamma + 1. It
converges. What it converges to is a coordinate-wise fixed point -- no single
slot can improve alone -- which is a Nash point of the deviation game, NOT the
global optimum: tau is not jointly concave in (q_0, ..., q_{K-1}), and sweeping
in a different order can land somewhere else. Both sweep orders are reported for
that reason, and the gap between them is a lower bound on how non-concave the
joint problem is here.

    tau_base  <=  max_k tau(single deviation at k)     <-- br_report, 5.11
              <=  tau(coordinate-wise fixed point)     <-- HERE
              <=  tau*  (jointly optimal prefix-specific proposal)

**The oracle got bigger, and this is the main caveat.** br_report prices a
prefix-specific oracle at ONE slot. After a full sweep every slot is oracled, so
this is a prefix-specific oracle over the whole block. It is an upper bound on
what a deployable block policy of this factorisation could reach, and a wide one.
It does not become trainable headroom by being iterated.

**Cross-fitting discipline.** The split is by path index, identical to
`br_report`'s, and is held FIXED across all rounds: the actions of every round
are fitted on one half and tau is always scored on the other. Re-splitting per
round would let each round launder the previous round's overfitting through a
fresh test half, and the compounding would be invisible. The fit-half trajectory
is printed beside the held-out one precisely so the divergence between them is
legible -- with seven coupled oracles fitted on ~512 paths, the two are expected
to separate more than they do for a single slot, and by how much is the result.
"""

from __future__ import annotations

import argparse
import math

from .br_report import (boot, fit_slot, load, per_path, scorer, tau_of, waterfill,
                        wmean)


# ------------------------------------------------- accept factors under acts --
def a_vec(e, K, acts):
    """This path's accept factors with the installed actions applied.

    A slot the target's own rollout never reached contributes 0 and truncates
    the block, exactly as in the base measurement -- no fitted action can
    resurrect a path the target ended.
    """
    out = []
    for i in range(K):
        if not e["alive"][i]:
            out.append(0.0)
            continue
        fn = acts.get(i)
        out.append(fn(e) if fn is not None else e["a"][i])
    return out


def tau_from_a(paths, K, acts):
    if not paths:
        return float("nan")
    tot = 0.0
    for e in paths:
        av = a_vec(e, K, acts)
        run, acc = 1.0, 0.0
        for i in range(K):
            run *= av[i]
            if run <= 0.0:
                break
            acc += run
        tot += acc
    return 1.0 + tot / len(paths)


def sk_fk_a(av, k, K):
    """(reach, continuation) at slot k from an accept-factor vector.

    Deliberately never reads av[k]: the weight a slot's own refit is scored
    against must not contain that slot's current action, or the update would
    chase its own tail.
    """
    S = 1.0
    for i in range(k):
        S *= av[i]
    F, run = 1.0, 1.0
    for j in range(k + 1, K):
        run *= av[j]
        if run <= 0.0:
            break
        F += run
    return S, F


def fit_slot_iter(fit_paths, k, K, order, acts):
    """Water-fill slot k with the reach and continuation the OTHER slots now give."""
    cells = {}
    for e in fit_paths:
        if not e["alive"][k]:
            continue
        S, F = sk_fk_a(a_vec(e, K, acts), k, K)
        c = S * F
        if c <= 0:
            continue
        w = 0 if order == 0 else e["cond"][k]
        cells.setdefault(w, []).append((e["tok"][k], e["p"][k], c))
    return {w: waterfill(items) for w, items in cells.items()}


# ----------------------------------------------------------------- sweeping --
def sweep(fit, test, K, order, rounds, eps, forward):
    """Coordinate ascent over the slots. Returns per-round (fit, held) dtau."""
    order_k = list(range(K)) if forward else list(range(K - 1, -1, -1))
    acts = {}
    base_fit = tau_from_a(fit, K, {})
    base_test = tau_from_a(test, K, {})
    traj = []
    for _ in range(rounds):
        for k in order_k:
            fitted = fit_slot_iter(fit, k, K, order, acts)
            if not fitted:
                continue
            acts[k] = scorer(fitted, k, order, eps)
        traj.append((tau_from_a(fit, K, acts) - base_fit,
                     tau_from_a(test, K, acts) - base_test))
    return traj


def singles(fit, test, K, order, eps):
    """br_report's single-slot held-out dtau at EVERY slot, on this same fold.

    Returned per slot, never reduced per anchor. The reduction has to happen
    after aggregation: max_k E[dtau_k] is the largest entry of the 5.11 table,
    while E[max_k dtau_k] silently adds a second oracle that picks the slot per
    prefix, and on this data the two differ by a factor of three. Uses
    br_report's own fit_slot/tau_of so that at eps=0 these ARE the 5.11 numbers.
    """
    base_test = tau_of(test, K)
    out = []
    for k in range(K):
        fitted = fit_slot(fit, k, K, order)
        if not fitted:
            out.append(None)
            continue
        out.append(tau_of(test, K, override=(k, scorer(fitted, k, order, eps)))
                   - base_test)
    return out


def anchor_traj(row, rounds, eps, forward):
    K = row["K"]
    order = row.get("order", 0)
    paths = list(per_path(row).values())
    if len(paths) < 8:
        return None
    half = len(paths) // 2
    folds = [(paths[:half], paths[half:]), (paths[half:], paths[:half])]

    fit_acc = [[] for _ in range(rounds)]
    held_acc = [[] for _ in range(rounds)]
    sing_acc = [[] for _ in range(K)]
    for fit, test in folds:
        for r, (f, h) in enumerate(sweep(fit, test, K, order, rounds, eps, forward)):
            fit_acc[r].append(f)
            held_acc[r].append(h)
        for k, d in enumerate(singles(fit, test, K, order, eps)):
            if d is not None:
                sing_acc[k].append(d)
    return dict(
        fit=[sum(v) / len(v) for v in fit_acc],
        held=[sum(v) / len(v) for v in held_acc],
        single=[(sum(v) / len(v)) if v else None for v in sing_acc],
        n=len(paths),
    )


# ------------------------------------------------------------------ report --
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--br", required=True)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--eps", type=float, default=0.0)
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--by-domain", action="store_true")
    args = ap.parse_args()

    by = load(args.br)
    if not by:
        raise SystemExit(f"no records matched {args.br!r}")

    prepared = {}
    for dom, rows in by.items():
        out = []
        for r in rows:
            for fwd in (True, False):
                t = anchor_traj(r, args.rounds, args.eps, fwd)
                if t:
                    out.append((r, fwd, t))
        prepared[dom] = out
    allr = [x for v in prepared.values() for x in v]
    if not allr:
        raise SystemExit("no anchor produced a trajectory")
    K = allr[0][0]["K"]
    order = allr[0][0].get("order", 0)
    print(f"M={allr[0][0].get('M')}  order={order}  K={K}  rounds={args.rounds}  "
          f"eps={args.eps}\n2-fold cross-fit, split held FIXED across rounds  "
          f"bootstrap B={args.boot} over PROMPTS  Hajek weights 1/pi")

    groups = ([(d, prepared[d]) for d in sorted(prepared)] if args.by_domain else []) + \
             [("POOLED", allr)]
    for name, rows in groups:
        n_anchor = len({(r["prompt_id"], r.get("t")) for r, _, _ in rows})
        print(f"\n===== {name}  ({n_anchor} anchors, "
              f"{len({r['prompt_id'] for r, _, _ in rows})} prompts) =====")
        # the two reference points, both aggregated BEFORE any max or sum
        sel_f = [(r, t) for r, f, t in rows if f is True]
        per_slot = []
        for k in range(K):
            cs = [dict(w=1.0 / max(r.get("pi", 1.0), 1e-9), v=t["single"][k])
                  for r, t in sel_f if t["single"][k] is not None]
            per_slot.append(wmean([(c["v"], c["w"]) for c in cs]) if cs else float("nan"))
        best1 = max(v for v in per_slot if not math.isnan(v))
        sum1 = sum(v for v in per_slot if not math.isnan(v))
        print("  single-slot reference (5.11, eps=0): " +
              " ".join(f"{v:.4f}" for v in per_slot))
        print(f"  best single slot max_k E[dtau_k] = {best1:.4f}   "
              f"naive sum sum_k E[dtau_k] = {sum1:.4f}")

        print(f"{'sweep':>8} {'round':>6} | {'dtau held':>10} {'95% CI':>20} "
              f"| {'dtau fit':>9} {'fit-score':>10} | {'vs best1':>9} {'vs sum1':>8}")
        for fwd in (True, False):
            sel = [(r, t) for r, f, t in rows if f is fwd]
            if not sel:
                continue
            lab = "0..K-1" if fwd else "K-1..0"
            for rd in range(args.rounds):
                cs = [dict(pid=r["prompt_id"],
                           w=1.0 / max(r.get("pi", 1.0), 1e-9),
                           h=t["held"][rd], f=t["fit"][rd])
                      for r, t in sel]
                g = lambda key: wmean([(c[key], c["w"]) for c in cs
                                       if c.get(key) is not None])
                lo, hi = boot(cs, "h", args.boot, args.seed)
                h, f = g("h"), g("f")
                print(f"{lab:>8} {rd + 1:>6} | {h:>10.4f} [{lo:+.4f},{hi:+.4f}] "
                      f"| {f:>9.4f} {f - h:>10.4f} | {h / best1:>8.2f}x "
                      f"{h / sum1:>7.2f}x")

        # monotonicity of the fit half is a property of exact coordinate ascent
        bad = 0
        for _, _, t in rows:
            for rd in range(1, args.rounds):
                if t["fit"][rd] < t["fit"][rd - 1] - 1e-9:
                    bad += 1
        print(f"   fit-half tau non-decreasing across rounds: "
              f"{'YES' if bad == 0 else f'NO ({bad} violations)'}")

    print("\nEach coordinate update is the exact maximiser of tau in that slot,")
    print("so the FIT half is monotone by construction and the printed check is")
    print("an arithmetic assertion, not a finding. The held-out column is the")
    print("measurement. 'vs best1' is the ratio to the largest entry of the 5.11")
    print("table, 'vs sum1' to the sum of all seven -- the sum 5.11 says may not")
    print("be taken, printed to show by how much it is wrong and which way.")
    print("Above 1.00x the deviations are COMPLEMENTARY: repairing one slot")
    print("raises what repairing another is worth, because reach is shared.")
    print("The limit is a COORDINATE-WISE fixed point, not the joint optimum.")
    print("Compare the two sweep orders only once BOTH have converged: before")
    print("that the gap is a rate difference and says nothing about the problem.")
    print("A gap that SURVIVES convergence is order dependence, i.e. a witness")
    print("that tau is not jointly concave in the block proposal.")
    print("After a full sweep every slot is oracled per prefix, so this is an")
    print("upper bound on a whole-block prefix-specific oracle -- a wider one")
    print("than 5.11's, and still not trainable headroom.")


if __name__ == "__main__":
    main()
