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

    print("Sec 4  one realised token, partitioning route, paired on one rollout set")
    o1 = load("rpre_o1/*.rpre1.jsonl.gz")

    def cond(recs, key, slot):
        vals = [((r.get("T1") or {}).get(str(slot), {}).get(key), w(r))
                for r in recs]
        return wmean([(v, x) for v, x in vals if v is not None])

    for slot, t0, t1 in ((1, 0.0776, 0.0010), (3, 0.1727, 0.0210),
                         (6, 0.2864, 0.0413)):
        check(f"T^(0) at slot {slot}", t0, cell(o1, "T", slot))
        check(f"T^(1) at slot {slot} (held-out)", t1, cond(o1, "split", slot))

    print("App    T^(0) alignment: two implementations, no shared estimator code")
    #   probe_tk reaches slot k by teacher-forced rescoring and has to pick the
    #   right row; probe_rpre reads p from the forward that samples the token
    #   and picks none. T^(0) steps 0.034-0.078 between adjacent slots, so a
    #   one-slot misalignment would show up at that size. It does not.
    tk = load("t0_s6/*.t0.jsonl.gz")
    rp = load("rpre/*.rpre.jsonl.gz")
    worst, prev = 0.0, None
    for slot in range(7):
        a_, b_ = cell(tk, "T", slot, "0"), cell(rp, "T", slot)
        if b_ != b_:
            continue
        if prev is not None and abs(b_ - prev) > 0:
            worst = max(worst, abs(a_ - b_) / abs(b_ - prev))
        prev = b_
    ok = worst < 0.25
    CHECKS.append(ok)
    print(f"  {'ok ' if ok else '!! '}{'worst gap as a share of the slot step':<52s}"
          f" limit    <25%   archive {worst:>7.1%}")

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
