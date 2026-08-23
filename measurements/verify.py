"""Recompute the paper's headline numbers straight from this archive.

The point is not to reproduce a run -- that needs GPUs -- but to close the last
gap between what was measured and what was written. Every number below is read
off the gzipped records in this directory with the estimator the paper
describes: Hajek weights 1/pi from the stratified sampler, no residual gate,
and the printed value beside the one in the paper.

    python -m measurements.verify           # from the repository root

A row that disagrees is a bug in the paper, not in the run.
"""
from __future__ import annotations

import glob
import gzip
import json
import os
import pathlib

HERE = pathlib.Path(__file__).resolve().parent


def load(pattern, skip=("SMOKE", "PILOT", "CORRUPT", "pre-oomfix")):
    """Every record under a glob, gzipped or not."""
    out = []
    for path in sorted(glob.glob(str(HERE / pattern))):
        if any(s in os.path.basename(path) for s in skip):
            continue
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    return out


def wmean(pairs):
    W = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / W if W else float("nan")


def w(r):
    return 1.0 / max(r.get("pi", 1.0), 1e-9)


def cell(recs, field, slot, order=None):
    vals = []
    for r in recs:
        d = r.get(field) or {}
        if order is not None:
            d = d.get(order) or {}
        v = d.get(str(slot))
        if v is not None:
            vals.append((v, w(r)))
    return wmean(vals) if vals else float("nan")


CHECKS = []


def check(name, paper, got, tol=5e-4):
    ok = abs(paper - got) <= tol
    CHECKS.append(ok)
    flag = "ok " if ok else "!! "
    print(f"  {flag}{name:<52s} paper {paper:>8.4f}   archive {got:>8.4f}")


def main():
    print("Sec 3  the order-0 floor, Qwen3-4B, top-256, M=1024")
    t0 = load("t0_s6/*.t0.jsonl.gz")
    print(f"       {len(t0)} anchors")
    for slot, paper in ((1, 0.0761), (3, 0.1715), (6, 0.2854)):
        check(f"T^(0) at slot {slot}", paper, cell(t0, "T", slot, "0"))

    print("Sec 3  full vocabulary, M=256, exact TV (the answer key)")
    rp = load("rpre/*.rpre.jsonl.gz")
    check("T^(0) at slot 6", 0.2861, cell(rp, "T", 6))

    print("Sec 4  one realised token")
    t1 = load("t1/*.t01.jsonl.gz")
    for slot, paper in ((2, 0.0040), (5, 0.0217)):
        check(f"T^(1) at slot {slot}", paper, cell(t1, "T", slot, "1"))

    print("App    the two T^(1) estimators, same 384 anchors (app:t1gap)")
    o1 = load("rpre_o1/*.rpre1.jsonl.gz")
    grp = {(r["prompt_id"], r["t"]): r for r in o1}
    #   slot 1 is T^(1) = 0 by identity. SNIS returns it exactly -- its free
    #   segment is empty there, so all M sequences coincide and the family it
    #   minimises over is a point mass. Grouping does not: it compares paths
    #   that shared a batched bf16 decode, which is the 1e-3 arithmetic floor.
    def cond(recs, key, slot):
        vals = [((r.get("T1") or {}).get(str(slot), {}).get(key), w(r))
                for r in recs]
        return wmean([(v, x) for v, x in vals if v is not None])
    for slot, a_fit, b_fit in ((1, 0.0000, 0.0009), (6, 0.0185, 0.0327)):
        check(f"slot {slot}  SNIS fit-and-score", a_fit, cell(t1, "T", slot, "1"))
        check(f"slot {slot}  grouping fit-and-score", b_fit, cond(o1, "plug", slot))
    check("slot 6  grouping held-out", 0.0413, cond(o1, "split", 6))
    paired = [(g - s, w(grp[k])) for k, r in ((k, r) for k, r in
              ((( r["prompt_id"], r["t"]), r) for r in t1))
              if k in grp
              for s in [(r.get("T_split") or {}).get("1", {}).get("6")]
              for g in [(grp[k].get("T1") or {}).get("6", {}).get("split")]
              if s is not None and g is not None]
    check("slot 6  paired, held-out vs held-out", 0.0203, wmean(paired))
    #   the pre-registered resolution test: four times the paths, same anchors,
    #   corpora and seeds. rpre_o1_m1024/PREDICTION.md was written before it ran.
    o1k = load("rpre_o1_m1024/*.rpre1.jsonl.gz")
    check("slot 6  grouping M=1024 fit-and-score", 0.0352, cond(o1k, "plug", 6))
    check("slot 6  grouping M=1024 held-out", 0.0393, cond(o1k, "split", 6))
    #   the same test on the other estimator. Neither column is unbiased -- one
    #   is the winner's curse, the other a half-sample fit -- so an estimator
    #   carrying only those two has a STABLE MIDPOINT as M grows, and its
    #   fit-and-score must rise. Grouping does both; SNIS does neither.
    a256 = load("t1/*.t01.m256.jsonl.gz")
    check("slot 6  SNIS M=256 fit-and-score", 0.0192, cell(a256, "T", 6, "1"))
    check("slot 6  SNIS M=256 held-out", 0.0258, cell(a256, "T_split", 6, "1"))
    for name, lo, hi in (("SNIS", (0.0192 + 0.0258) / 2, (0.0185 + 0.0210) / 2),
                         ("grouping", (0.0327 + 0.0413) / 2, (0.0352 + 0.0393) / 2)):
        print(f"  -- {name:8s} midpoint  M=256 {lo:.4f} -> M=1024 {hi:.4f}"
              f"   shift {hi - lo:+.4f}")
    print(f"  -- grouping larger on {sum(1 for v, _ in paired if v > 0)}"
          f"/{len(paired)} anchors")

    print("Sec 5  DFlash against its own floor, four domains")
    check("R at slot 6", 0.6359, cell(rp, "R", 6))
    check("G/R at slot 6", 0.550,
          (cell(rp, "R", 6) - cell(rp, "T", 6)) / cell(rp, "R", 6), tol=2e-3)

    print("Sec 6  three larger targets, slot 6")
    for tag, name, paper_t, paper_r in (
            ("qwen8b", "Qwen3-8B", 0.3831, 0.6734),
            ("qwen14b", "Qwen3-14B", 0.3411, 0.6341),
            ("gemma12b", "Gemma-4-12B", 0.3322, 0.7880)):
        rr = load(f"scale/{tag}/*.srv0.jsonl.gz")
        check(f"{name} T^(0)", paper_t, cell(rr, "T", 6))
        check(f"{name} R", paper_r, cell(rr, "R", 6))

    print("Sec 6  DeepSeek-V4-Pro through its API, top-20, unweighted")
    api = load("api_v4/v4pro*.jsonl.gz")
    print(f"       {len(api)} anchors over "
          f"{len({r['prompt_id'] for r in api})} prompts")
    for slot, paper in ((1, 0.0588), (6, 0.2452)):
        cs = [r["slots"][str(slot)]["T0"] for r in api
              if (r.get("slots") or {}).get(str(slot))]
        check(f"T^(0) at slot {slot}", paper, sum(cs) / len(cs))
    res = [r["slots"]["6"]["resid"] for r in api
           if (r.get("slots") or {}).get("6")]
    print(f"  -- truncation band at slot 6: {sum(res)/len(res):.1e} "
          f"(the reason no cell is gated)")

    print("Sec 7  the serving reweighting, DFlash")
    srv = load("srv/*.srv0.jsonl.gz")
    check("R free at slot 6", 0.6353, cell(srv, "R", 6))
    check("R serve at slot 6", 0.5835, cell(srv, "R_serve", 6))
    tau = 1 + sum(cell(srv, "S", k) for k in range(7))
    check("tau from the recorded joint", 4.574, tau, tol=5e-3)

    bad = CHECKS.count(False)
    print(f"\n{len(CHECKS) - bad}/{len(CHECKS)} agree" +
          ("" if not bad else f"   -- {bad} DISAGREE"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
