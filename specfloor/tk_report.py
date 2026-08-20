"""Report the realisation-blind acceptance floor T_k^(m) from probe_tk output.

  python -m specfloor.tk_report --tk 'measurement_runs/tk/*.t0.jsonl'

T_k^(m) = E_{W_m}[ min_q E_{Z|W_m} TV(p_Z, q) ] is the TV distance no drafter
that sees only W_m = (X, Z_{k-m:k-1}) can beat, so 1 - T_k^(m) is an upper
bound on that rung's per-slot acceptance rate against the free-rollout target.
Slot 0 has no preceding realisation to be blind to, so T_0 = 0 identically --
it is printed as an arithmetic check on the pipeline, not as a result.

Three things are read alongside every T:

* T_split -- q* fitted on half the sampled paths and scored on the other half.
  Fitting and scoring on the SAME draws biases T down (the barycentre is
  chosen to be close to exactly these paths), so T_split >= T in expectation
  and the gap is the winner's curse. The correction is reported, never
  silently applied; if it is comparable to T itself the slot is not reportable.

* resid -- the probability mass the top-k truncation missed, averaged over
  paths. TV is a total-variation distance over the FULL vocabulary, and mass
  outside the top-k can only be bounded, not resolved. Slots above
  --resid-gate are dropped and counted.

* ess -- effective sample size of the path weights. At rung 0 there is no
  importance weighting so ess == M by construction; it becomes informative at
  rung >= 1, where the SNIS weights p(z* | X, s) concentrate.

Weighting follows stats.py: anchors carry Horvitz-Thompson weights 1/pi from
the stratified sampler, and the bootstrap resamples PROMPTS, not anchors,
because anchors from one prompt share a prefix.
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
    """Records grouped by domain. Malformed lines are counted, never skipped
    silently -- two probe processes sharing an output file interleave writes at
    overlapping offsets and produce spliced records, some of which still parse."""
    by, bad = {}, {}
    for f in sorted(glob.glob(pattern)):
        dom = os.path.basename(f).split(".")[0]
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                bad[dom] = bad.get(dom, 0) + 1
                continue
            by.setdefault(dom, []).append(r)
    return by, bad


def wmean(pairs):
    """pairs of (value, weight) -> weighted mean, nan if no weight."""
    sw = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / sw if sw > 0 else float("nan")


def cells(recs, m, k, resid_gate, ess_gate):
    """(prompt_id, T, T_split, resid, ess, ht_weight) for eligible anchors."""
    out, drop_resid, drop_ess = [], 0, 0
    for r in recs:
        T = (r.get("T") or {}).get(m, {}).get(k)
        if T is None:
            continue
        res = (r.get("resid") or {}).get(m, {}).get(k)
        if res is not None and res > resid_gate:
            drop_resid += 1
            continue
        e = (r.get("ess") or {}).get(m, {}).get(k)
        if e is not None and e < ess_gate:
            drop_ess += 1
            continue
        ts = (r.get("T_split") or {}).get(m, {}).get(k)
        w = 1.0 / max(r.get("pi", 1.0), 1e-9)
        out.append((r["prompt_id"], T, ts, res, e, w))
    return out, drop_resid, drop_ess


def boot(cs, B, seed, idx=1):
    """Cluster bootstrap over prompts; returns a CI on the HT-weighted mean."""
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
        draws.append(wmean([(c[idx], c[5]) for c in samp if c[idx] is not None]))
    draws = [d for d in draws if not math.isnan(d)]
    if not draws:
        return float("nan"), float("nan")
    q = lambda p: sorted(draws)[max(0, min(len(draws) - 1, int(p * len(draws))))]
    return q((1 - C.CI) / 2), q(1 - (1 - C.CI) / 2)


def table(name, recs, args, rungs, slots):
    npr = len({r["prompt_id"] for r in recs})
    print(f"\n===== {name}  ({len(recs)} anchors, {npr} prompts) =====")
    print(f"{'m':>2} {'k':>2} {'n':>4} {'prompt':>6} | {'T^(m)':>7} {'CI':>17} |"
          f" {'T_split':>8} {'curse':>7} | {'resid':>8} {'ESS':>7}")
    for m in rungs:
        for k in slots:
            cs, dr, de = cells(recs, m, k, args.resid_gate, args.ess_gate)
            if len(cs) < 5:
                continue
            T = wmean([(c[1], c[5]) for c in cs])
            lo, hi = boot(cs, args.boot, args.seed, 1)
            sp = [c for c in cs if c[2] is not None]
            Ts = wmean([(c[2], c[5]) for c in sp]) if sp else float("nan")
            res = wmean([(c[3], c[5]) for c in cs if c[3] is not None])
            ess = wmean([(c[4], c[5]) for c in cs if c[4] is not None])
            flag = ""
            if not math.isnan(Ts) and T > 0 and abs(Ts - T) > args.curse_frac * T:
                flag = "  <-- curse > %d%% of T" % (100 * args.curse_frac)
            print(f"{m:>2} {k:>2} {len(cs):>4} {len({c[0] for c in cs}):>6} |"
                  f" {T:>7.4f} [{lo:>6.4f},{hi:>6.4f}] |"
                  f" {Ts:>8.4f} {Ts - T:>+7.4f} | {res:>8.1e} {ess:>7.1f}{flag}")
            if dr or de:
                print(f"        dropped: {dr} on resid>{args.resid_gate:g}, "
                      f"{de} on ESS<{args.ess_gate:g}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tk", required=True, help="glob of probe_tk output files")
    ap.add_argument("--resid-gate", type=float, default=1e-3,
                    help="max top-k truncation mass tolerated per cell")
    ap.add_argument("--ess-gate", type=float, default=0.0,
                    help="min effective sample size (informative at rung >= 1)")
    ap.add_argument("--curse-frac", type=float, default=0.10,
                    help="flag a slot when |T_split - T| exceeds this fraction of T")
    ap.add_argument("--boot", type=int, default=C.BOOTSTRAP_B)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--pooled-only", action="store_true")
    args = ap.parse_args()

    by, bad = load(args.tk)
    if not by:
        raise SystemExit(f"no records matched {args.tk!r}")
    for dom, n in sorted(bad.items()):
        print(f"!! {dom}: {n} malformed line(s) skipped -- concurrent writers?")

    allr = [r for rs in by.values() for r in rs]
    rungs = sorted({m for r in allr for m in (r.get("T") or {})}, key=int)
    slots = sorted({k for r in allr for d in (r.get("T") or {}).values() for k in d},
                   key=int)

    print(f"M={allr[0].get('M')} top_k={allr[0].get('top_k')}  "
          f"resid gate<={args.resid_gate:g}  bootstrap B={args.boot} over PROMPTS  "
          f"CI={C.CI}  HT weights 1/pi")
    if not args.pooled_only:
        for dom in sorted(by):
            table(dom, by[dom], args, rungs, slots)
    table("POOLED", allr, args, rungs, slots)

    print("\nT_0 = 0 exactly is a pipeline check: slot 0 has no preceding")
    print("realisation, so a realisation-blind proposal loses nothing there.")
    print("curse = T_split - T; it is second-order when |curse| << T. 1 - T^(m)")
    print("upper-bounds rung m's per-slot acceptance against the free-rollout")
    print("target -- it is NOT an acceptance rate under the serving law.")


if __name__ == "__main__":
    main()
