"""G_pre = R_pre - T^(0): how much of DFlash's rejection loss capacity could remove.

  python -m specfloor.rpre_report --rpre 'measurement_runs/rpre/*.rpre.jsonl' \
                                    [--tk 'measurement_runs/tk/*.t0.jsonl']

R is the drafter's actual expected TV rejection loss against the free-rollout
target; T is the minimum over ALL rung-0 proposals on the same paths. G = R - T
is therefore a CEILING on what any parallel-capacity increase could buy, not an
estimate of what it would buy.

G is a PAIRED difference -- same anchor, same sampled paths, same p_Z -- so it is
bootstrapped as a difference, never as the difference of two separately
bootstrapped means. The two CIs would be much wider and would not answer the
question, because R and T move together across anchors.

Passing --tk adds the cross-check that matters most for trusting either number:
probe_tk computed T on a different engine from a top-256 read, this probe
computes it locally over the full vocabulary. They estimate the same quantity
and are matched here per (domain, slot).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random

from specfloor import config as C


def load(pattern):
    by, bad = {}, {}
    for f in sorted(glob.glob(pattern)):
        dom = os.path.basename(f).split(".")[0]
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                by.setdefault(dom, []).append(json.loads(line))
            except ValueError:
                bad[dom] = bad.get(dom, 0) + 1
    return by, bad


def wmean(pairs):
    sw = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / sw if sw > 0 else float("nan")


def cells(recs, k):
    """(prompt_id, T, R, G, T_split, ht_weight, R_temp, G_temp) at slot k.

    R uses q warped by the SAME law as the target (temperature, top-p, top-k),
    which is what a serving stack that post-processes both sides does. R_temp
    warps q by temperature only. Which one is right is an implementation
    question about the serving stack, not about the drafter, and the two differ
    enough on C1 to matter -- so both are carried rather than one being chosen
    silently here. Note the floor T is unaffected either way: it minimises over
    ALL distributions, warped or not, so it stays a valid lower bound.
    """
    out = []
    for r in recs:
        sk = str(k)
        T, R = (r.get("T") or {}).get(sk), (r.get("R") or {}).get(sk)
        if T is None or R is None:
            continue
        rt = (r.get("R_temp") or {}).get(sk)
        out.append((r["prompt_id"], T, R, R - T,
                    (r.get("T_split") or {}).get(sk),
                    1.0 / max(r.get("pi", 1.0), 1e-9),
                    rt, None if rt is None else rt - T))
    return out


def boot(cs, idx, B, seed):
    by = {}
    for c in cs:
        by.setdefault(c[0], []).append(c)
    keys = list(by)
    rng = random.Random(seed)
    draws = []
    for _ in range(B):
        samp = []
        for _ in range(len(keys)):
            samp += by[keys[rng.randrange(len(keys))]]
        v = wmean([(c[idx], c[5]) for c in samp if c[idx] is not None])
        if not math.isnan(v):
            draws.append(v)
    if not draws:
        return float("nan"), float("nan")
    q = lambda p: sorted(draws)[max(0, min(len(draws) - 1, int(p * len(draws))))]
    return q((1 - C.CI) / 2), q(1 - (1 - C.CI) / 2)


def table(name, recs, args, slots, tk=None):
    npr = len({r["prompt_id"] for r in recs})
    print(f"\n===== {name}  ({len(recs)} anchors, {npr} prompts) =====")
    hdr = (f"{'k':>2} {'n':>4} | {'T (floor)':>9} {'R (q warped)':>12} "
           f"{'R (q temp)':>10} | {'G = R-T':>9} {'95% CI on G':>17} {'G/R':>6}")
    if tk is not None:
        hdr += f" | {'T probe_tk':>10} {'diff':>7}"
    print(hdr)
    for k in slots:
        cs = cells(recs, k)
        if len(cs) < 5:
            continue
        T = wmean([(c[1], c[5]) for c in cs])
        R = wmean([(c[2], c[5]) for c in cs])
        G = wmean([(c[3], c[5]) for c in cs])
        rt = [c for c in cs if c[6] is not None]
        Rt = wmean([(c[6], c[5]) for c in rt]) if rt else float("nan")
        lo, hi = boot(cs, 3, args.boot, args.seed)
        line = (f"{k:>2} {len(cs):>4} | {T:>9.4f} {R:>12.4f} {Rt:>10.4f} | "
                f"{G:>9.4f} [{lo:>6.4f},{hi:>6.4f}] {G / R if R else float('nan'):>6.1%}")
        if tk is not None:
            t2 = tk.get((name, k))
            line += (f" | {t2:>10.4f} {T - t2:>+7.4f}" if t2 is not None
                     else f" | {'--':>10} {'--':>7}")
        print(line)


# ------------------------------------------------------------- order 1 ---
def cells1(recs, k):
    r"""One dict per anchor at slot $k$ for the order-1 decomposition.

    Four quantities, all read off the SAME 256 paths so every difference below
    is paired within an anchor:

      T0    the order-0 floor -- what the block is worth with the chain off;
      T1    the order-1 floor, conditional on the realised $Z_{k-1}$;
      Rorc  DSpark's loss when the chain is fed that realisation, which is what
            a slot reached under left-to-right verification was actually fed;
      Rself DSpark's loss on its own free-rollout token.

    dT1 = T0 - T1 is the information the conditioning token carries. Gpost =
    Rorc - T1 is what the head still owes its own floor. exp = Rself - Rorc is
    exposure, and it is kept out of Gpost rather than folded into it.
    """
    out = []
    for r in recs:
        sk = str(k)
        T0 = (r.get("T") or {}).get(sk)
        t1 = (r.get("T1") or {}).get(sk)
        Ro = (r.get("R") or {}).get(sk)
        if T0 is None or not t1 or Ro is None:
            continue
        T1s, T1p = t1.get("split"), t1.get("plug")
        Rs = (r.get("R_self") or {}).get(sk)
        out.append(dict(
            pid=r["prompt_id"], w=1.0 / max(r.get("pi", 1.0), 1e-9),
            T0=T0, T1p=T1p, T1s=T1s, cov=t1.get("cov"), grp=t1.get("groups"),
            Rorc=Ro, Rself=Rs,
            dT1=None if T1s is None else T0 - T1s,
            Gpost=None if T1s is None else Ro - T1s,
            exp=None if Rs is None else Rs - Ro))
    return out


def bootg(cs, key, B, seed):
    """Prompt-level cluster bootstrap of one HT-weighted column of cells1."""
    by = {}
    for c in cs:
        by.setdefault(c["pid"], []).append(c)
    keys = list(by)
    rng = random.Random(seed)
    draws = []
    for _ in range(B):
        samp = []
        for _ in range(len(keys)):
            samp += by[keys[rng.randrange(len(keys))]]
        v = wmean([(c[key], c["w"]) for c in samp if c.get(key) is not None])
        if not math.isnan(v):
            draws.append(v)
    if not draws:
        return float("nan"), float("nan")
    draws.sort()
    q = lambda p: draws[max(0, min(len(draws) - 1, int(p * len(draws))))]
    return q((1 - C.CI) / 2), q(1 - (1 - C.CI) / 2)


def table1(name, recs, args, slots):
    npr = len({r["prompt_id"] for r in recs})
    print(f"\n===== {name}  ({len(recs)} anchors, {npr} prompts) =====")
    print(f"{'k':>2} {'n':>4} | {'T0':>7} {'T1plug':>7} {'T1split':>7} {'cov':>5} "
          f"{'grp':>4} | {'R orac':>7} {'R self':>7} | {'dT1':>7} {'Gpost':>7} "
          f"{'95% CI on Gpost':>17} {'expo':>7}")
    for k in slots:
        cs = cells1(recs, k)
        if len(cs) < 5:
            continue
        g = lambda key: wmean([(c[key], c["w"]) for c in cs if c.get(key) is not None])
        lo, hi = bootg(cs, "Gpost", args.boot, args.seed)
        print(f"{k:>2} {len(cs):>4} | {g('T0'):>7.4f} {g('T1p'):>7.4f} "
              f"{g('T1s'):>7.4f} {g('cov'):>5.2f} {g('grp'):>4.1f} | "
              f"{g('Rorc'):>7.4f} {g('Rself'):>7.4f} | {g('dT1'):>7.4f} "
              f"{g('Gpost'):>7.4f} [{lo:>6.4f},{hi:>6.4f}] {g('exp'):>7.4f}")


def recall_table(name, recs, slots, conv="greedy", Ks=(1, 8, 16)):
    """Recall@K against the commit-blind ceiling -- a DIFFERENT floor from T.

    The ceiling is the mixture's top-K mass (sampled target) or the top-K mass of
    the target's own argmax distribution (greedy target). Its minimiser is a mean,
    not the quantile that minimises TV, so this table and the T/G table are in
    different metrics and their rows must never be differenced against each other.
    """
    rows = [r for r in recs if r.get("rec")]
    if not rows:
        return
    print(f"\n----- {name}: Recall@K vs the commit-blind ceiling "
          f"({conv} target) -----")
    print(f"{'k':>2} " + " ".join(f"{'@%d DFlash' % K:>11} {'ceil':>7} {'gap':>7}"
                                  for K in Ks))
    for k in slots:
        cells, w = {}, []
        for r in rows:
            d = (r.get("rec") or {}).get(str(k))
            if not d:
                continue
            wt = 1.0 / max(r.get("pi", 1.0), 1e-9)
            w.append(wt)
            for K in Ks:
                cells.setdefault(K, [[], []])
                cells[K][0].append((d.get(f"dflash_{conv}@{K}"), wt))
                cells[K][1].append((d.get(f"ceil_{conv}@{K}"), wt))
        if not w:
            continue
        line = f"{k:>2} "
        for K in Ks:
            a = wmean([(v, x) for v, x in cells[K][0] if v is not None])
            c = wmean([(v, x) for v, x in cells[K][1] if v is not None])
            line += f" {a:>11.3f} {c:>7.3f} {c - a:>7.3f}"
        print(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rpre", required=True)
    ap.add_argument("--recall", action="store_true",
                    help="also print the exact-match Recall@K table and its ceiling")
    ap.add_argument("--recall-target", default="greedy", choices=("greedy", "sampled"))
    ap.add_argument("--tk", default=None, help="probe_tk output, for the T cross-check")
    ap.add_argument("--boot", type=int, default=C.BOOTSTRAP_B)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--resid-gate", type=float, default=float("inf"),
                    help="drop cells whose top-k truncation residual exceeds this. OFF by default: a gate is a selection rule and the cells it removes are the heavy-tailed ones, which are the high-floor ones, so it biases the mean down by orders of magnitude more than the residual it insures against. The residual is reported instead, and bounds |T - T~| two-sidedly.")
    args = ap.parse_args()

    by, bad = load(args.rpre)
    if not by:
        raise SystemExit(f"no records matched {args.rpre!r}")
    for dom, n in sorted(bad.items()):
        print(f"!! {dom}: {n} malformed line(s)")

    tk = None
    if args.tk:
        tby, _ = load(args.tk)
        tk = {}
        allt = []
        for dom, recs in tby.items():
            for k in range(16):
                cs = []
                for r in recs:
                    v = (r.get("T") or {}).get("0", {}).get(str(k))
                    res = (r.get("resid") or {}).get("0", {}).get(str(k))
                    if v is None or (res is not None and res > args.resid_gate):
                        continue
                    cs.append((v, 1.0 / max(r.get("pi", 1.0), 1e-9)))
                if len(cs) >= 5:
                    tk[(dom, k)] = wmean(cs)
                    allt += cs
                    tk.setdefault(("POOLED", k), None)
        # pooled needs one pass over every domain's cells at that slot
        for k in range(16):
            cs = []
            for dom, recs in tby.items():
                for r in recs:
                    v = (r.get("T") or {}).get("0", {}).get(str(k))
                    res = (r.get("resid") or {}).get("0", {}).get(str(k))
                    if v is None or (res is not None and res > args.resid_gate):
                        continue
                    cs.append((v, 1.0 / max(r.get("pi", 1.0), 1e-9)))
            if len(cs) >= 5:
                tk[("POOLED", k)] = wmean(cs)

    allr = [r for rs in by.values() for r in rs]
    slots = sorted({int(k) for r in allr for k in (r.get("T") or {})})
    print(f"M={allr[0].get('M')}  full-vocabulary exact TV  bootstrap B={args.boot} "
          f"over PROMPTS  CI={C.CI}  HT weights 1/pi")

    if int(allr[0].get("order", 0)) >= 1:
        print("order 1: R is scored against the CONDITIONAL floor T1, and T0 is")
        print("carried on the same paths so dT1 = T0 - T1 (the value of the")
        print("revealed token) and Gpost = R - T1 (what the head still owes) are")
        print("both within-anchor differences.")
        for dom in sorted(by):
            table1(dom, by[dom], args, slots)
        table1("POOLED", allr, args, slots)
        print("\ncov is the share of paths sitting in a conditioning group of size >= 2,")
        print("i.e. the sub-population T1split describes; T1plug uses every path but")
        print("reads 0 on singleton groups, so the two bracket the conditional floor.")
        print("expo = R_self - R_oracle is exposure and is NOT part of Gpost: under")
        print("left-to-right verification a slot that is reached was fed the")
        print("realisation, so R_oracle is the serving quantity.")
        return

    print("G is bootstrapped as a PAIRED difference (same anchor, same paths).")
    for dom in sorted(by):
        table(dom, by[dom], args, slots, tk)
    table("POOLED", allr, args, slots, tk)
    if args.recall:
        for dom in sorted(by):
            recall_table(dom, by[dom], slots, args.recall_target)
        recall_table("POOLED", allr, slots, args.recall_target)
        print("\nThe Recall ceiling is the mixture's top-K mass -- a MEAN. The TV floor")
        print("T is a quantile. Same family, different functionals: do not difference")
        print("a row of this table against a row of the one above. The @1-to-@K span")
        print("is the provable ceiling on what any re-ranking selector over a")
        print("commit-blind top-K lattice can recover, with no oracle in it.")

    print("\nG = R - T >= 0 by construction: T minimises over all q and q_DFlash is one.")
    print("G is the CEILING on what parallel capacity could buy, not the expected gain.")
    print("G/R is the share of DFlash's own rejection loss that is not information-")
    print("theoretic -- 1 - G/R is the share no rung-0 model of any size can remove.")


if __name__ == "__main__":
    main()
