r"""rho_{k,m}: the share of the missing path information the last m tokens recover.

  python -m specfloor.mi_report                        # the archived M=512 run
  python -m specfloor.mi_report --rm 'measurements/rm_snis/*.rm.jsonl.gz'

probe_rm records, per eligible anchor and slot k, the missing information
d_k = C_{k,0} - C_{k,k} (`den`) and the part the last m realised tokens recover,
n_{k,m} = C_{k,0} - C_{k,m} (`num`), both in nats of log loss. The estimand is a
RATIO OF POPULATION SUMS,

    rho_{k,m} = sum_i n_{i,k,m} / pi_i  /  sum_i d_{i,k} / pi_i ,

so an anchor with little missing information contributes little, instead of
contributing one noisy per-anchor ratio at full weight. Pooling adds the same
numerators and denominators over slots m+1..GAMMA-1; at k = m the m tokens are
the whole realised prefix and the ratio is 1 by construction, so it is omitted.

**Why 1/pi is the whole weight.** probe_rm measures a second-stage subsample of
the anchors whose screening ladder shows dCE_6 > RM_ELIGIBILITY_THRESHOLD, and
multiplies that stage's inclusion probability into the record's pi. `audit`
recomputes the eligible set from the archived ladders and requires that every
eligible anchor was measured, with the ladder's pi, so the second stage had
probability one and nothing else needs reweighting. An earlier aggregation
applied one pooled 256-anchor budget across all four domains to all 698
anchors, which down-weighted the thin strata by as much as 0.175; it is the
reason this check exists rather than an assumption.

**The ESS gate is a sensitivity arm, off by default.** CE_M is an importance
estimate, and cells whose weights concentrate can be dropped with --ess; the
dropped cells are counted. The paper's table uses every eligible cell.

Intervals resample prompts with all their anchors (stats.ratio_interval), in
first-encounter order of the sorted domain files, and are clipped to [0, 100]%.
"""

from __future__ import annotations

import argparse
import collections
import glob
import os

import numpy as np

from specfloor import config as C
from specfloor.records import archived, domain_of, read_jsonl, weight
from specfloor.stats import ratio_interval

ORDERS = C.RM_ORDERS


def load(pattern):
    """Records over the sorted domain files. Only `<domain>.rm.jsonl*` files
    count: rm_snis/ also holds a gsm8k.rm_snis side file from the estimator
    comparison, which is not part of the run."""
    rows = []
    for f in sorted(glob.glob(pattern)):
        if not os.path.basename(f).split(".", 1)[1].startswith("rm.jsonl"):
            continue
        for r in read_jsonl(f):
            r["_domain"] = domain_of(f)
            rows.append(r)
    return rows


def screening(pattern):
    """{domain: screening records} -- the files probe_rm drew its anchors from.
    The archive's gsm8k ladder was written before files carried a domain prefix
    and is named plain `ladder.M512.jsonl`."""
    out = {}
    for f in sorted(glob.glob(pattern)):
        dom = domain_of(f)
        out["gsm8k" if dom == "ladder" else dom] = read_jsonl(f)
    return out


def audit(rows, screen_pattern, slot=C.GAMMA - 1, eps=C.RM_ELIGIBILITY_THRESHOLD):
    """Per domain: screened, eligible, measured, and whether 1/pi is exact."""
    screens = screening(screen_pattern)
    out = {}
    for dom in sorted({r["_domain"] for r in rows}):
        if dom not in screens:
            raise SystemExit(f"no screening file for {dom} in {screen_pattern!r}")
        screen = screens[dom]
        elig = {(r["prompt_id"], r["t"]): r for r in screen if r["dCE"][slot] > eps}
        got = {(r["prompt_id"], r["t"]): r for r in rows if r["_domain"] == dom}
        strata = collections.Counter(r["stratum"] for r in elig.values())
        exact = set(got) == set(elig) and all(
            got[k]["pi"] == elig[k]["pi"] for k in got)
        out[dom] = dict(screened=len(screen), eligible=len(elig), measured=len(got),
                        strata=len(strata), largest_stratum=max(strata.values()),
                        exact=exact)
    return out


def cell(r, m, k, ess_gate):
    n = ((r.get("num") or {}).get(str(m)) or {}).get(str(k))
    d = (r.get("den") or {}).get(str(k))
    if n is None or d is None:
        return None
    e = ((r.get("ess") or {}).get(str(m)) or {}).get(str(k))
    if ess_gate and (e is None or e < ess_gate):
        return None
    return n, d


def columns(K=C.GAMMA):
    """(m, k) for every reported cell, then (m, 'pooled') per order."""
    cols = [(m, k) for m in ORDERS for k in range(m + 1, K)]
    return cols + [(m, "pooled") for m in ORDERS]


def table(rows, ess_gate=0.0, B=C.BOOTSTRAP_B, seed=C.SEED, K=C.GAMMA):
    """{(m, k): dict(rho, lo, hi, cells)} in percent; k='pooled' pools slots."""
    cols = columns(K)
    num = np.zeros((len(rows), len(cols)))
    den = np.zeros_like(num)
    cells = np.zeros(len(cols), dtype=int)
    for i, r in enumerate(rows):
        w = weight(r)
        for j, (m, k) in enumerate(cols):
            slots = range(m + 1, K) if k == "pooled" else (k,)
            for kk in slots:
                c = cell(r, m, kk, ess_gate)
                if c is None:
                    continue
                num[i, j] += w * c[0]
                den[i, j] += w * c[1]
                cells[j] += 1
    point, lo, hi = ratio_interval([r["prompt_id"] for r in rows], num, den, B, seed)
    return {col: dict(rho=100 * point[j], lo=max(0.0, 100 * lo[j]),
                      hi=min(100.0, 100 * hi[j]), cells=int(cells[j]))
            for j, col in enumerate(cols)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rm", default=archived("rm_m512/*.rm.jsonl.gz"))
    ap.add_argument("--screen", default=archived("calib_20260818/C0/*ladder.M512.jsonl.gz"),
                    help="glob of the per-domain files probe_rm screened (its --cheap)")
    ap.add_argument("--ess", type=float, default=0.0,
                    help="drop cells whose importance ESS is below this")
    ap.add_argument("--boot", type=int, default=C.BOOTSTRAP_B)
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()

    rows = load(args.rm)
    if not rows:
        raise SystemExit(f"no probe_rm records matched {args.rm!r}")
    paths = sorted({r.get("mixed_paths") for r in rows})
    print(f"{len(rows)} eligible anchors over {len({r['prompt_id'] for r in rows})} "
          f"prompts; M={paths}; ESS gate {args.ess:g}; bootstrap B={args.boot} "
          f"over prompts")
    bad = False
    for dom, a in audit(rows, args.screen).items():
        print(f"  {dom:<8} screened {a['screened']:>4}  eligible {a['eligible']:>4}  "
              f"measured {a['measured']:>4}  over {a['strata']} strata"
              f"  {'1/pi exact' if a['exact'] else '!! measured set is not the eligible set'}")
        bad |= not a["exact"]
    if bad:
        raise SystemExit("second-stage selection is not the identity; 1/pi would "
                         "misweight this sample -- reweight before reporting")

    t = table(rows, args.ess, args.boot, args.seed)
    print(f"\n{'slot':>6}" + "".join(f"{'rho_' + str(m):>24}" for m in ORDERS))
    for k in list(range(2, C.GAMMA)) + ["pooled"]:
        line = f"{k:>6}"
        for m in ORDERS:
            c = t.get((m, k))
            line += (f"{c['rho']:>8.1f}% [{c['lo']:5.1f},{c['hi']:5.1f}]"
                     if c else f"{'--':>24}")
        print(line)
    print("\ncells  " + "  ".join(f"m={m}: {t[(m, 'pooled')]['cells']}" for m in ORDERS))
    print("A dash marks k <= m, where the m tokens are the whole realised prefix.")


if __name__ == "__main__":
    main()
