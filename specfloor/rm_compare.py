"""R_m, conditional (SNIS) vs interventional, with prompt-level CIs.

Separate from stats.py on purpose. stats.py reports the protocol's headline R_m
and knows nothing about the estimator correction; this answers one question --
how much did the intervention-vs-conditioning bug move the number, and does the
shift survive a cluster bootstrap -- and is meant to be read once and then
folded into stats.py if the answer is "a lot".

  python -m specfloor.rm_compare --rm 'runs/*.rm.jsonl' [--ess 32]

Estimator notes that matter for reading the output:

* RATIO OF MEANS, not mean of ratios. R_m is a normalised ratio whose
  denominator (CE_B - CE_A) can be near zero; averaging per-anchor ratios lets
  one small denominator dominate. Numerator and denominator are accumulated
  separately across anchors and divided once, which is also what stats.py does.

* The bootstrap resamples PROMPTS, not anchors. Anchors from one prompt share a
  prefix and are not independent; in the pilot 21 anchors were 10 prompts, so
  treating anchors as the unit would have inflated the apparent n by 2x.

* ESS gates the SNIS column only. Self-normalised importance sampling degrades
  when the revealed suffix is improbable under most sampled prefixes -- exactly
  the interesting regime -- so cells below the gate are dropped and COUNTED,
  never silently averaged. The interventional column is recomputed on the same
  surviving cells so the two are always compared on identical data.
"""

from __future__ import annotations

import argparse
import glob
import json
import random

from specfloor import config as C
from specfloor.records import open_text


def load(pattern):
    recs = []
    for f in sorted(glob.glob(pattern)):
        with open_text(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    r["_src"] = f
                    recs.append(r)
    return recs


def cells(recs, m, ess_gate):
    """(prompt_id, num_cond, num_interv, den) per eligible (anchor, slot)."""
    out, dropped_ess, dropped_den = [], 0, 0
    for r in recs:
        K = len(r.get("ce_A") or [])
        for k in range(1, K):
            sk = str(k)
            den = (r.get("den") or {}).get(sk)
            if den is None or den <= C.RM_ELIGIBILITY_THRESHOLD:
                dropped_den += 1
                continue
            cM = (r.get("ce_M") or {}).get(m, {}).get(sk)
            iM = (r.get("ce_M_interv") or {}).get(m, {}).get(sk)
            if cM is None or iM is None:
                continue
            e = (r.get("ess") or {}).get(m, {}).get(sk)
            if e is None or e < ess_gate:
                dropped_ess += 1
                continue
            b = r["ce_B"][k]
            out.append((r["prompt_id"], b - cM, b - iM, den, e))
    return out, dropped_ess, dropped_den


def ratio(cs, idx):
    num = sum(c[idx] for c in cs)
    den = sum(c[3] for c in cs)
    return num / den if den else float("nan")


def boot(cs, B, seed):
    """Hierarchical bootstrap: resample PROMPTS with replacement."""
    by = {}
    for c in cs:
        by.setdefault(c[0], []).append(c)
    keys = list(by)
    rng = random.Random(seed)
    dc, di, ds = [], [], []
    for _ in range(B):
        samp = []
        for _ in range(len(keys)):
            samp += by[keys[rng.randrange(len(keys))]]
        rc, ri = ratio(samp, 1), ratio(samp, 2)
        dc.append(rc); di.append(ri); ds.append(ri - rc)
    q = lambda v, p: sorted(v)[max(0, min(len(v) - 1, int(p * len(v))))]
    lo, hi = (1 - C.CI) / 2, 1 - (1 - C.CI) / 2
    return ((q(dc, lo), q(dc, hi)), (q(di, lo), q(di, hi)), (q(ds, lo), q(ds, hi)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rm", required=True, help="glob of probe_rm output files")
    ap.add_argument("--ess", type=float, default=32.0,
                    help="minimum effective sample size per cell for the SNIS column")
    ap.add_argument("--boot", type=int, default=C.BOOTSTRAP_B)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--by-domain", action="store_true")
    args = ap.parse_args()

    recs = load(args.rm)
    if not recs:
        raise SystemExit(f"no records matched {args.rm!r}")

    groups = {"ALL": recs}
    if args.by_domain:
        for r in recs:
            groups.setdefault(r["prompt_id"].split(":")[0], []).append(r)

    print(f"eps_R={C.RM_ELIGIBILITY_THRESHOLD}  ESS gate>={args.ess:g}  "
          f"bootstrap B={args.boot} over PROMPTS  CI={C.CI}")
    for gname, grecs in groups.items():
        print(f"\n===== {gname}  ({len(grecs)} anchors, "
              f"{len({r['prompt_id'] for r in grecs})} prompts) =====")
        print(f"{'m':>2} {'cells':>6} {'prompts':>8} | {'R cond (SNIS)':>22} "
              f"{'R interv (old)':>22} | {'shift (pp)':>20} {'ESS med':>8}")
        for m in (str(x) for x in C.RM_ORDERS):
            cs, dropped, _ = cells(grecs, m, args.ess)
            if len(cs) < 5:
                print(f"{m:>2} {len(cs):>6}   -- too few cells to report --")
                continue
            npr = len({c[0] for c in cs})
            rc, ri = ratio(cs, 1), ratio(cs, 2)
            (cl, ch), (il, ih), (sl, sh) = boot(cs, args.boot, args.seed)
            ess = sorted(c[4] for c in cs)[len(cs) // 2]
            warn = "  <-- n_clusters < 30" if npr < 30 else ""
            print(f"{m:>2} {len(cs):>6} {npr:>8} | "
                  f"{100*rc:>7.1f}% [{100*cl:>5.1f},{100*ch:>5.1f}] "
                  f"{100*ri:>7.1f}% [{100*il:>5.1f},{100*ih:>5.1f}] | "
                  f"{100*(ri-rc):>+7.1f} [{100*sl:>+5.1f},{100*sh:>+5.1f}] {ess:>8.1f}"
                  f"{warn}")
            if dropped:
                print(f"   {dropped} cells dropped by the ESS gate "
                      f"({100*dropped/(dropped+len(cs)):.0f}% of eligible)")

    print("\nShift is (interventional - conditional). It is NEGATIVE when the old")
    print("estimator UNDERSTATED locality. A CI on the shift excluding 0 means the")
    print("correction is larger than prompt-level sampling noise.")


if __name__ == "__main__":
    main()
