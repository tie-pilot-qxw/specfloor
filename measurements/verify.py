"""Recompute every number the paper reads off this archive, with the package's own reports.

The point is not to reproduce a run -- that needs GPUs -- but to close the last
gap between what was measured and what was written. Each number below is
recomputed from the gzipped records in this directory by the specfloor module
that reports it, with the estimator the paper describes: Hajek weights 1/pi
from the stratified sampler, no residual gate, prompt-cluster bootstraps.

    python -m measurements.verify            # from the repository root
    python -m measurements.verify --fast     # skip the best-response blocks (~4 min)

A value is compared at the precision the paper prints it: "0.078" must be
within 0.0005, "67.3" (a percentage) within 0.05 points. Three outcomes:

  ok    the archive gives the printed value;
  fix   the paper prints a stale value, recorded beside the one the archive
        gives, which the paper should print instead -- counted separately so
        the list of corrections is explicit rather than tolerated;
  !!    the archive agrees with neither: a bug in the paper or in a report.

Bootstrap intervals reuse each report's own resampling code, so an interval
printed by a report and the one checked here are the same draws.
"""
from __future__ import annotations

import argparse
import math
import re
import statistics
import sys

import numpy as np

from specfloor import config as C
from specfloor import concentration_report as CR
from specfloor import kmedian_report as KM
from specfloor import mi_report as MI
from specfloor import ratio_report as RR
from specfloor import rpre_compare as RC
from specfloor import rpre_report as RP
from specfloor import srv_report as SR
from specfloor.records import archived, load_by_domain, weight

OK, FIX, BAD = [], [], []
_CACHE = {}


def load(pattern):
    """Every record under a glob inside measurements/, sorted file order."""
    if pattern not in _CACHE:
        by, bad = load_by_domain(archived(pattern))
        assert not bad, f"{pattern}: malformed lines {bad}"
        _CACHE[pattern] = [r for recs in by.values() for r in recs]
    return _CACHE[pattern]


def by_domain(pattern):
    by, _ = load_by_domain(archived(pattern))
    return by


def hajek(recs, y):
    pairs = [(v, weight(r)) for r in recs if (v := y(r)) is not None]
    return sum(v * w for v, w in pairs) / sum(w for _, w in pairs)


# ------------------------------------------------------------- comparing ---
def tolerance(printed):
    """Half a unit in the last printed digit: "0.078" -> 5e-4, "4.8e-3" -> 5e-5."""
    m = re.fullmatch(r"([+-]?)(\d*)\.?(\d*)(?:e([+-]?\d+))?", printed.strip())
    assert m, printed
    return 0.5 * 10.0 ** (-len(m.group(3)) + int(m.group(4) or 0)) * 1.0001


def check(name, paper, got, pct=False, erratum=None):
    """Compare at the paper's printed precision; `erratum` is what it should print."""
    v = 100 * got if pct else got
    unit = "%" if pct else ""
    target = erratum if erratum is not None else paper
    good = abs(float(target) - v) <= tolerance(target)
    shown = f"{v:.4g}" if abs(v) < 1e-3 and v != 0 else f"{v:.4f}".rstrip("0").rstrip(".")
    if good and erratum is None:
        OK.append(name)
        print(f"  ok   {name:<60s} {paper + unit:>10}  archive {shown}{unit}")
    elif good:
        FIX.append((name, paper, erratum))
        print(f"  fix  {name:<60s} {paper + unit:>10}  archive {shown}{unit}"
              f"  -> print {erratum}{unit}")
    else:
        BAD.append(name)
        print(f"  !!   {name:<60s} {paper + unit:>10}  archive {shown}{unit}"
              + ("" if erratum is None else f"  (erratum {erratum})"))


def check_ci(name, paper, lo, hi, pct=False, erratum=None):
    """paper as "[.094,.183]"; each endpoint at its own printed precision."""
    ps = paper.strip("[]").split(",")
    es = erratum.strip("[]").split(",") if erratum else [None, None]
    for tag, p, v, e in (("lo", ps[0], lo, es[0]), ("hi", ps[1], hi, es[1])):
        check(f"{name} {tag}", p, v, pct, None if e == p else e)


def check_true(name, claim, ok):
    (OK if ok else BAD).append(name)
    print(f"  {'ok ' if ok else '!! '}  {name:<60s} {claim}")


def section(title):
    print(f"\n{title}")


# ------------------------------------------------------------------ Sec 2 ---
def setup():
    section("App. Experimental setup: the anchor sample")
    rp = load("rpre/*.rpre.jsonl.gz")
    doms = by_domain("rpre/*.rpre.jsonl.gz")
    check("anchors", "384", len(rp))
    check("prompts", "170", len({r["prompt_id"] for r in rp}))
    check_true("96 anchors per domain", "96 x 4",
               all(len(v) == 96 for v in doms.values()) and len(doms) == 4)
    arena = sorted(r["context"] for r in doms["arena8k"])
    check("arena-hard median context", "2250", statistics.median(arena))
    check("arena-hard p90 context (nearest rank)", "8226",
          float(np.percentile(arena, 90, method="nearest")))
    check("arena-hard max context", "8638", max(arena))
    meds = {d: statistics.median(r["context"] for r in doms[d])
            for d in ("gsm8k", "mbpp", "alpaca")}
    check_true("other three medians \"near 440\"",
               " ".join(f"{d} {v:g}" for d, v in meds.items()),
               all(abs(v - 440) < 60 for v in meds.values()))


# ---------------------------------------------------------------- Sec 3.1 ---
def parallel_blindness():
    section("Sec. 3.1 / Fig. t1: the order-0 floor, full vocabulary, M=256")
    o1 = load("rpre_o1/*.rpre1.jsonl.gz")
    for k, v in zip(range(1, 7), ("0.0776", "0.1212", "0.1727", "0.2059", "0.2459", "0.2864")):
        check(f"T^(0) slot {k} (Fig. t1)", v, hajek(o1, RR.flat("T", k)))
    check("T^(0) slot 1 (text)", "0.078", hajek(o1, RR.flat("T", 1)))
    check("T^(0) slot 6 (text)", "0.286", hajek(o1, RR.flat("T", 6)))
    check("acceptance cap at slot 1", "92", 1 - hajek(o1, RR.flat("T", 1)), pct=True)
    check("acceptance cap at slot 6", "71", 1 - hajek(o1, RR.flat("T", 6)), pct=True)
    check("T^(0) slot 0 (identity)", "0", abs(hajek(o1, RR.flat("T", 0))))

    section("Sec. 3.1 / Fig. domain floors: T^(0) per domain, rpre/")
    doms = by_domain("rpre/*.rpre.jsonl.gz")
    fig = {"alpaca": ("0.0811", "0.1469", "0.2206", "0.2849", "0.3225", "0.3685"),
           "arena8k": ("0.0924", "0.1322", "0.1813", "0.2117", "0.2561", "0.2993"),
           "mbpp": ("0.0396", "0.0991", "0.1373", "0.1620", "0.2066", "0.2343"),
           "gsm8k": ("0.0563", "0.0739", "0.1229", "0.1451", "0.1652", "0.2009")}
    per = {d: [hajek(doms[d], RR.flat("T", k)) for k in range(1, 7)] for d in fig}
    for d, vals in fig.items():
        for k, v in zip(range(1, 7), vals):
            check(f"{d} T^(0) slot {k}", v, per[d][k - 1])
    check_true("open-ended above constrained at slots 1-6",
               "min(alpaca, arena) > max(gsm8k, mbpp)",
               all(min(per["alpaca"][i], per["arena8k"][i]) >
                   max(per["gsm8k"][i], per["mbpp"][i]) for i in range(6)))

    section("Sec. 3.1: how the floor is spread over anchors (concentration_report)")
    rp = load("rpre/*.rpre.jsonl.gz")
    c1, c6 = CR.floor_concentration(rp, 1), CR.floor_concentration(rp, 6)
    check_true("slot 1 share of anchors below 0.01 (\"two thirds\")",
               f"{c1['below']:.1%}", 0.6 < c1["below"] < 0.7)
    check("slot 1 their share of the floor", "0.4", c1["below_mass"], pct=True)
    check("slot 1 top-decile share of the floor", "60", c1["top_decile_mass"], pct=True)
    check("slot 6 mean", "0.286", c6["mean"])
    check("slot 6 weighted median", "0.199", c6["median"])
    check_true("slot 6 share below 0.01 (\"one quarter\")", f"{c6['below']:.1%}",
               0.2 < c6["below"] < 0.3)
    check("slot 6 top-decile share", "27", c6["top_decile_mass"], pct=True)


# ---------------------------------------------------------------- Sec 3.2 ---
def conditioning_order():
    section("Sec. 3.2 / Fig. t1: one realised token, partitioning route")
    o1 = load("rpre_o1/*.rpre1.jsonl.gz")
    for k, v in zip(range(2, 7), ("0.0048", "0.0210", "0.0256", "0.0287", "0.0413")):
        check(f"T^(1) slot {k} (Fig. t1)", v, hajek(o1, RR.t1_split(k)))
    shares = [RR.interval(o1, RR.minus(RR.flat("T", k), RR.t1_split(k)),
                          RR.flat("T", k), B=2)[0] for k in range(2, 7)]
    check("smallest share removed, slots 2-6 (\"86-100%\")", "86", min(shares), pct=True)
    check_true("share removed at slot 1 is 100% by identity", "T^(1)_1 = 0 exactly", True)
    check("largest T^(1) over slots 1-6 (\"0.041 or below\")", "0.041",
          max(hajek(o1, RR.t1_split(k)) for k in range(1, 7)))

    section("Sec. 3.2 / tab:mi-locality: rho_{k,m}, M=512, no ESS gate (mi_report)")
    rows = MI.load(archived("rm_m512/*.rm.jsonl.gz"))
    check("eligible anchors", "698", len(rows))
    check("their prompts", "221", len({r["prompt_id"] for r in rows}))
    check_true("every eligible anchor measured, pi as screened",
               "second stage is the identity",
               all(a["exact"] for a in MI.audit(rows, archived("calib_20260818/C0/*ladder.M512.jsonl.gz")).values()))
    t = MI.table(rows)
    paper = {
        (1, 2): ("94.2", "[86.6,98.5]"), (1, 3): ("95.8", "[92.4,98.2]"),
        (1, 4): ("94.9", "[92.6,96.8]"), (1, 5): ("94.6", "[89.7,97.6]"),
        (1, 6): ("92.9", "[86.3,97.3]"), (2, 3): ("98.8", "[97.2,99.8]"),
        (2, 4): ("99.4", "[98.8,99.9]"), (2, 5): ("99.0", "[98.1,99.9]"),
        (2, 6): ("98.7", "[97.4,99.9]"), (4, 5): ("99.9", "[99.8,100.0]"),
        (4, 6): ("99.7", "[99.4,99.9]"),
        (1, "pooled"): ("94.3", "[91.8,96.2]"), (2, "pooled"): ("99.0", "[98.3,99.5]"),
        (4, "pooled"): ("99.8", "[99.7,99.9]")}
    for (m, k), (v, ci) in paper.items():
        c = t[(m, k)]
        check(f"rho_{m} slot {k}", v, c["rho"] / 100, pct=True)
        check_ci(f"rho_{m} slot {k} CI", ci, c["lo"] / 100, c["hi"] / 100, pct=True)
    check("order-1 cells, no gate", "3490", t[(1, "pooled")]["cells"])
    g32, g256 = MI.table(rows, 32, B=2), MI.table(rows, 256, B=2)
    check("order-1 cells at ESS >= 32", "3234", g32[(1, "pooled")]["cells"])
    check("rho_1 pooled at ESS >= 32", "95.2", g32[(1, "pooled")]["rho"] / 100, pct=True)
    check("order-1 cells at ESS >= M/2", "2234", g256[(1, "pooled")]["cells"])
    check("rho_1 pooled at ESS >= M/2", "93.9", g256[(1, "pooled")]["rho"] / 100, pct=True)
    for m in (2, 4):
        vals = [x[(m, "pooled")]["rho"] for x in (t, g32, g256)]
        check_true(f"rho_{m} pooled moves < 0.4 points across gates",
                   f"range {max(vals) - min(vals):.2f}", max(vals) - min(vals) < 0.4)
    snis = MI.table(MI.load(archived("rm_snis/*.rm.jsonl.gz")), 32, B=2)
    check("rho_1 pooled, M=64 at ESS >= M/2", "92.7", snis[(1, "pooled")]["rho"] / 100,
          pct=True)
    check("one token recovers 92.9-95.8% at every slot: min", "92.9",
          min(t[(1, k)]["rho"] for k in range(2, 7)) / 100, pct=True)
    check("two tokens recover at least", "98.7",
          min(t[(2, k)]["rho"] for k in range(3, 7)) / 100, pct=True)

    section("Sec. 3.2 / App. Continuation concentration (concentration_report)")
    pooled, per = CR.effective_support(by_domain("t0_s6/*.t0.jsonl.gz"), 6)
    check("effective support, weighted median", "1.8", pooled["median"])
    check("effective support, weighted p90", "18.9", pooled["p90"])
    check_true("spans more than two orders of magnitude",
               f"{pooled['min']:.1f}-{pooled['max']:.1f}", pooled["max"] / pooled["min"] > 100)
    check("corr(log N_eff, T_6), pooled", "+0.90", pooled["corr_log"])
    for d, v in (("alpaca", "+0.845"), ("arena8k", "+0.915"), ("gsm8k", "+0.920"),
                 ("mbpp", "+0.904")):
        check(f"corr(log N_eff, T_6), {d}", v, per[d]["corr_log"])
    check("corr(N_eff, T_6), pooled", "+0.53", pooled["corr_raw"])
    raw = [s["corr_raw"] for s in per.values()]
    check("corr(N_eff, T_6), smallest within a domain", "+0.53", min(raw))
    check("corr(N_eff, T_6), largest within a domain", "+0.65", max(raw))


# ---------------------------------------------------------------- Sec 3.3 ---
def drafter_gaps():
    section("Sec. 3.3 / tab:dflash-details: DFlash against T^(0) (rpre_report)")
    rp = load("rpre/*.rpre.jsonl.gz")
    T = ("0.00", "0.08", "0.12", "0.17", "0.21", "0.25", "0.29")
    R = ("0.14", "0.24", "0.35", "0.43", "0.50", "0.57", "0.64")
    G = ("0.14", "0.16", "0.23", "0.25", "0.29", "0.32", "0.35")
    CI = ("[.094,.183]", "[.122,.203]", "[.177,.278]", "[.204,.307]", "[.242,.345]",
          "[.268,.378]", "[.300,.401]")
    GR = ("100", "67.3", "65.0", "59.5", "58.6", "56.8", "55.0")
    fig_T = ("0.0000", "0.0776", "0.1211", "0.1724", "0.2060", "0.2458", "0.2861")
    fig_R = ("0.1359", "0.2375", "0.3462", "0.4258", "0.4978", "0.5685", "0.6359")
    for k in range(7):
        cs = RP.cells(rp, k)
        t = RP.wmean([(c[1], c[5]) for c in cs])
        r = RP.wmean([(c[2], c[5]) for c in cs])
        g = RP.wmean([(c[3], c[5]) for c in cs])
        lo, hi = RP.boot(cs, 3, C.BOOTSTRAP_B, C.SEED)
        check(f"T^(0) slot {k}", T[k], t)
        check(f"R slot {k}", R[k], r)
        check(f"G^(0) slot {k}", G[k], g)
        # the paper rounded 0.17648 via 0.1765 to .177
        check_ci(f"G^(0) slot {k} CI", CI[k], lo, hi,
                 erratum="[.176,.278]" if k == 2 else None)
        check(f"G^(0)/R slot {k}", GR[k], g / r, pct=True)
        check(f"Fig. stack DFlash T^(0) slot {k}", fig_T[k], t)
        check(f"Fig. stack DFlash R slot {k}", fig_R[k], r)
    shares = [RP.wmean([(c[3], c[5]) for c in RP.cells(rp, k)]) /
              RP.wmean([(c[2], c[5]) for c in RP.cells(rp, k)]) for k in range(1, 7)]
    check("gap share of rejection, slots 1-6: low (\"55-67%\")", "55", min(shares), pct=True)
    check("gap share of rejection, slots 1-6: high", "67", max(shares), pct=True)

    section("Sec. 3.3 / tab:o1: DSpark against T^(1) (rpre_report, order 1)")
    o1 = load("rpre_o1/*.rpre1.jsonl.gz")
    tab = {2: ("4.8e-3", "0.21", "0.20", "[.155,.252]", "0.12", "0.32"),
           4: ("0.03", "0.28", "0.26", "[.203,.315]", "0.18", "0.46"),
           6: ("0.04", "0.37", "0.33", "[.273,.381]", "0.21", "0.58")}
    for k, (t1, ro, g1, ci, ex, rs) in tab.items():
        cs = RP.cells1(o1, k)
        g = lambda key: RP.wmean([(c[key], c["w"]) for c in cs if c.get(key) is not None])
        lo, hi = RP.bootg(cs, "Gpost", C.BOOTSTRAP_B, C.SEED)
        check(f"T^(1) split slot {k}", t1, g("T1s"))
        check(f"R_oracle slot {k}", ro, g("Rorc"))
        check(f"G^(1) slot {k}", g1, g("Gpost"))
        check_ci(f"G^(1) slot {k} CI", ci, lo, hi)
        check(f"E_expo slot {k}", ex, g("exp"))
        check(f"R_self slot {k}", rs, g("Rself"))
    cov = [RP.wmean([(c["cov"], c["w"]) for c in RP.cells1(o1, k)]) for k in (2, 4, 6)]
    check("split-half coverage, low (\"0.98-0.99\")", "0.98", min(cov))
    check("split-half coverage, high", "0.99", max(cov))
    share = [RP.wmean([(c["Gpost"], c["w"]) for c in RP.cells1(o1, k)]) /
             RP.wmean([(c["Rorc"], c["w"]) for c in RP.cells1(o1, k)]) for k in range(1, 7)]
    check("G^(1) share of R_oracle, slots 1-6: low (\"89-100%\")", "89", min(share), pct=True)
    check_true("G^(1) share of R_oracle, slots 1-6: at most 100%", f"{max(share):.1%}",
               max(share) <= 1.0)
    ex6 = RP.wmean([(c["exp"], c["w"]) for c in RP.cells1(o1, 6)])
    check("exposure difference at slot 6", "0.214", ex6)
    fig = {"floor": ("0.0000", "0.0000", "0.0048", "0.0210", "0.0256", "0.0287", "0.0413"),
           "Rorc": ("0.1073", "0.1357", "0.2055", "0.2679", "0.2824", "0.3452", "0.3667"),
           "Rself": ("0.1073", "0.2199", "0.3224", "0.4015", "0.4591", "0.5396", "0.5810")}
    for k in range(7):
        cs = RP.cells1(o1, k)
        g = lambda key: RP.wmean([(c[key], c["w"]) for c in cs if c.get(key) is not None])
        # slots 0 and 1 are zero by identity; the figure draws the identity
        check(f"Fig. stack DSpark floor slot {k}", fig["floor"][k],
              0.0 if k < 2 else g("T1s"))
        check(f"Fig. stack DSpark R_oracle slot {k}", fig["Rorc"][k], g("Rorc"))
        check(f"Fig. stack DSpark R_self slot {k}", fig["Rself"][k], g("Rself"))


# ------------------------------------------------------------------ Sec 4 ---
def scale():
    section("Sec. 4 / Fig. cross-target: final-slot decompositions (ratio_report)")
    fig = {"Qwen3-4B": ("0.286", "0.636", "0.041", "0.367"),
           "Qwen3-8B": ("0.383", "0.673", "0.060", "0.399"),
           "Qwen3-14B": ("0.353", "0.656", "0.056", "0.412"),
           "Gemma-4-12B": ("0.242", "0.676", "0.031", "0.397")}
    ci = {"Qwen3-4B": ("[49.5,60.7]", "[49.4,60.5]"),
          "Qwen3-8B": ("[36.2,50.4]", "[36.2,50.5]"),
          # the paper's interval is from the superseded arena-hard run
          "Qwen3-14B": ("[39.2,53.4]", "[38.6,53.8]"),
          "Gemma-4-12B": ("[58.9,69.4]", "[59.2,69.5]")}
    gr, dr = [], []
    for name, (g0, g1) in RR.TARGETS.items():
        a = RR.dflash_share(RR.records(g0), 6)
        b = RR.dspark_share(RR.records(g1), 6)
        t0, r0, t1, r1 = fig[name]
        check(f"{name} DFlash T^(0)", t0, a["T"])
        # the figure rounded Qwen3-14B's 0.65549 via 0.6555 to 0.656
        check(f"{name} DFlash R", r0, a["R"], erratum="0.655" if name == "Qwen3-14B" else None)
        check(f"{name} DSpark T^(1)", t1, b["T1"])
        check(f"{name} DSpark R_oracle", r1, b["R"])
        paper, now = ci[name]
        check_ci(f"{name} G^(0)/R CI", paper, a["share"][1], a["share"][2], pct=True,
                 erratum=None if paper == now else now)
        check_true(f"{name} R_oracle exceeds T^(0)", "DSpark above the order-0 floor",
                   b["R"] > b["T0"])
        gr.append(a["share"][0])
        dr.append(b["share"][0])
    check("DFlash G/R at slot 6, lowest target (\"43-64%\")", "43", min(gr), pct=True)
    check("DFlash G/R at slot 6, highest target", "64", max(gr), pct=True)
    check("DSpark G/R_oracle at slot 6, lowest (\"85-92%\")", "85", min(dr), pct=True)
    check("DSpark G/R_oracle at slot 6, highest", "92", max(dr), pct=True)
    # carried from the original checks: the larger targets at four places
    for tag, name, pt, pr in (("qwen8b", "Qwen3-8B", "0.3831", "0.6734"),
                              ("qwen14b", "Qwen3-14B", "0.3531", "0.6555"),
                              ("gemma12b_fix", "Gemma-4-12B", "0.2423", "0.6764")):
        rr = RR.records([g for g in RR.TARGETS[name][0]])
        check(f"{name} T^(0) slot 6 (4 d.p.)", pt, hajek(rr, RR.flat("T", 6)))
        check(f"{name} R slot 6 (4 d.p.)", pr, hajek(rr, RR.flat("R", 6)))

    section("Sec. 4 / App. Vocabulary truncation: DeepSeek-V4-Pro, top-20, unweighted")
    api = [r for p in ("api_v4/v4pro.jsonl.gz", "api_v4/v4pro2.jsonl.gz") for r in load(p)]
    check("anchors", "120", len(api))
    at = lambda k: [r["slots"][str(k)] for r in api if (r.get("slots") or {}).get(str(k))]
    check("anchors with a complete slot-6 record", "118", len(at(6)))
    mean = lambda xs: sum(xs) / len(xs)
    for k, v in ((1, "0.0588"), (6, "0.2452")):
        check(f"T^(0) slot {k}", v, mean([c["T0"] for c in at(k)]))
    check("T^(0) slot 6 (text)", "0.25", mean([c["T0"] for c in at(6)]))
    check("held-out T^(1) slot 6 (\"about 0.03\")", "0.03",
          mean([c["T1"]["split"] for c in at(6)]))
    removed = [1 - mean([c["T1"]["split"] for c in at(k)]) / mean([c["T0"] for c in at(k)])
               for k in range(1, 7)]
    check("share one token removes, slots 1-6: low (\"86-90%\")", "86", min(removed), pct=True)
    check("share one token removes, slots 1-6: high", "90", max(removed), pct=True)
    check("mean omitted mass at slot 6", "2.2e-4", mean([c["resid"] for c in at(6)]))
    dom = {}
    for r in api:
        c = (r.get("slots") or {}).get("6")
        if c:
            dom.setdefault(r["domain"], []).append(c["T0"])
    dm = {d: mean(v) for d, v in dom.items()}
    open_, closed = [v for d, v in dm.items() if d in ("alpaca", "arena8k", "arena-hard-v2")], \
                    [v for d, v in dm.items() if d in ("gsm8k", "mbpp")]
    check_true("open-ended domains have the higher slot-6 floors",
               " ".join(f"{d} {v:.3f}" for d, v in sorted(dm.items())),
               bool(open_) and bool(closed) and min(open_) > max(closed))


# ------------------------------------------------------------------ Sec 5 ---
def serving():
    section("Sec. 5 / tab:serving-risk-details (srv_report)")
    tab = {
        "srv/*.srv0.jsonl.gz": ("DFlash",
            ("0.24", "0.35", "0.43", "0.50", "0.57", "0.64"),
            ("0.17", "0.20", "0.17", "0.19", "0.22", "0.21"),
            ("-0.07", "-0.15", "-0.26", "-0.31", "-0.35", "-0.42"),
            # the paper's DFlash intervals sit ~0.001 from srv_report's draws
            (("[-.098,-.046]", "[-.098,-.047]"), ("[-.185,-.111]", "[-.185,-.112]"),
             ("[-.311,-.211]", "[-.312,-.211]"), ("[-.365,-.253]", "[-.366,-.254]"),
             ("[-.421,-.286]", "[-.421,-.287]"), ("[-.493,-.356]", "[-.494,-.357]")),
            ("1.000", "1.093", "1.337", "1.945", "3.133", "5.692", "12.33"),
            "4.574", "3.397"),
        "srv/*.srv1.jsonl.gz": ("DSpark",
            ("0.14", "0.21", "0.27", "0.29", "0.35", "0.37"),
            ("0.10", "0.15", "0.13", "0.14", "0.18", "0.16"),
            ("-0.03", "-0.05", "-0.13", "-0.15", "-0.17", "-0.21"),
            (("[-.056,-.017]", None), ("[-.081,-.033]", None),
             ("[-.172,-.095]", "[-.173,-.097]"), ("[-.194,-.101]", None),
             ("[-.224,-.120]", None), ("[-.262,-.159]", None)),
            ("1.000", "1.039", "1.110", "1.310", "1.580", "1.986", "2.63"),
            "5.232", "4.386"),
    }
    for pattern, (name, Rf, Rs, D, CIs, dep, tau_joint, tau_ind) in tab.items():
        recs = load(pattern)
        prodE, EJ_s, EJ_p = 1.0, 0.0, 0.0
        for k in range(7):
            cs = SR.cells(recs, k)
            rf = SR.wmean([(c["Rf"], c["w"]) for c in cs])
            S = SR.wmean([(c["S"], c["w"]) for c in cs])
            prodE *= 1 - rf
            EJ_s += S
            EJ_p += prodE
            check(f"{name} E[prod a]/prod E[a] slot {k}", dep[k], S / prodE)
            if k == 0:
                continue
            rs = SR.serve_risk(cs)
            lo, hi = SR.boot(cs, "d_pop", C.BOOTSTRAP_B, C.SEED)
            check(f"{name} R free slot {k}", Rf[k - 1], rf)
            check(f"{name} R serve slot {k}", Rs[k - 1], rs)
            check(f"{name} serve - free slot {k}", D[k - 1], rs - rf)
            paper, now = CIs[k - 1]
            check_ci(f"{name} serve - free slot {k} CI", paper, lo, hi, erratum=now)
        check(f"{name} tau from the joint survival", tau_joint, 1 + EJ_s)
        check(f"{name} tau from independent slots", tau_ind, 1 + EJ_p)
        if name == "DFlash":
            check("DFlash R free at slot 6 (text)", "0.635", rf)
            check("DFlash R serve at slot 6 (text)", "0.211", rs)
        else:
            check("DSpark R_oracle free at slot 6 (text)", "0.366", rf)
            check("DSpark R serve at slot 6 (text)", "0.158", rs)


def best_response(sweeps=True):
    from specfloor import br_iter as BI
    from specfloor import br_report as BR

    section("Sec. 5 / App. Full single-slot profiles (br_report, 2-fold cross-fit)")
    eps = [0.0, 0.1, 0.25]
    paper = {
        "br0": ("DFlash", ("0.221", "0.183", "0.166", "0.124", "0.135", "0.093", "0.056"),
                ("[.148,.300]", "[.143,.229]", "[.129,.209]", "[.092,.161]",
                 "[.087,.208]", "[.059,.136]", "[.038,.077]"), "2.3e-3"),
        "br1": ("DSpark", ("0.266", "0.213", "0.232", "0.185", "0.146", "0.137", "0.062"),
                None, "5.2e-3")}
    gain = {}
    for tag, (name, vals, cis, fit) in paper.items():
        by = BR.load(archived(f"br/*.{tag}.jsonl.gz"))
        rows = {d: [(r, c) for r in rs if (c := BR.anchor_cells(r, eps))]
                for d, rs in by.items()}
        allr = [x for v in rows.values() for x in v]

        def slot_means(rs, e):
            return [RP.wmean([(c[k]["held"][e], weight(r)) for r, c in rs if k in c])
                    for k in range(7)]
        held = slot_means(allr, 0.0)
        gain[name] = held
        worst_fit = 0.0
        for k in range(7):
            check(f"{name} dtau^BR slot {k}", vals[k], held[k])
            tr = RP.wmean([(c[k]["train"], weight(r)) for r, c in allr if k in c])
            worst_fit = max(worst_fit, tr - held[k])
            if cis:
                cs = [dict(pid=r["prompt_id"], w=weight(r), v=c[k]["held"][0.0])
                      for r, c in allr if k in c]
                lo, hi = BR.boot(cs, "v", 4000, C.SEED)
                check_ci(f"{name} dtau^BR slot {k} CI", cis[k], lo, hi)
        check(f"{name} largest in-sample minus held-out", fit, worst_fit)
        for e in eps[1:]:
            m = slot_means(allr, e)
            check_true(f"{name} endpoint ordering kept at eps={e}",
                       "slot 0 above slot 6", m[0] > m[6])
        if name == "DFlash":
            per = {d: slot_means(v, 0.0) for d, v in rows.items()}
            check_true("slot 6 is the minimum in all four domains", "",
                       all(int(np.argmin(v)) == 6 for v in per.values()))
            peaks = {d: int(np.argmax(v)) for d, v in per.items()}
            check_true("slot 0 is the maximum in three; mbpp peaks at slot 1",
                       str(peaks), peaks == {"alpaca": 0, "arena8k": 0, "gsm8k": 0, "mbpp": 1})

    rp = load("rpre/*.rpre.jsonl.gz")
    G = [RP.wmean([(c[3], c[5]) for c in RP.cells(rp, k)]) for k in range(7)]
    check("model gap grows slot 0 -> 6 (x)", "2.6", G[6] / G[0])
    check("single-slot gain falls slot 0 -> 6 (x)", "3.9", gain["DFlash"][0] / gain["DFlash"][6])
    for k, v in enumerate(("0.221", "0.183", "0.166", "0.124", "0.135", "0.093", "0.056")):
        check(f"Fig. slot priority serving gain slot {k}", v, gain["DFlash"][k])
    # the figure annotates three decimals
    for k, v in enumerate(("0.136", "0.160", "0.225", "0.253", "0.292", "0.323", "0.350")):
        check(f"Fig. slot priority model gap slot {k}", v, G[k])
    if not sweeps:
        return

    section("Sec. 5 / App. Coordinate sweeps (br_iter)")
    paper = {"br0": ("DFlash", "0.98", ("2.15", "[1.883,2.430]"), ("2.07", "[1.794,2.357]"),
                     "2.20"),
             "br1": ("DSpark", "1.24", ("2.45", "[2.127,2.794]"), ("2.46", "[2.128,2.796]"),
                     "1.98")}
    ends = {}
    rounds = 8          # both directions of both drafters have stopped moving by 7
    for tag, (name, sum1, fwd_p, bwd_p, ratio) in paper.items():
        by = BI.load(archived(f"br/*.{tag}.jsonl.gz"))
        allr = [r for rs in by.values() for r in rs]
        out = {}
        for fwd in (True, False):
            trajs = [(r, t) for r in allr if (t := BI.anchor_traj(r, rounds, 0.0, fwd))]
            cs = [dict(pid=r["prompt_id"], w=weight(r), h=t["held"][-1]) for r, t in trajs]
            out[fwd] = (RP.wmean([(c["h"], c["w"]) for c in cs]), *BR.boot(cs, "h", 4000, C.SEED))
            last = RP.wmean([(t["held"][-2], weight(r)) for r, t in trajs])
            check_true(f"{name} sweep {'0->6' if fwd else '6->0'} converged",
                       f"last round moved {abs(out[fwd][0] - last):.1e}",
                       abs(out[fwd][0] - last) < 1e-4)
            if fwd:
                singles = [RP.wmean([(t["single"][k], weight(r)) for r, t in trajs
                                     if t["single"][k] is not None]) for k in range(7)]
        check(f"{name} sum of isolated gains", sum1, sum(singles))
        for fwd, (v, ci) in ((True, fwd_p), (False, bwd_p)):
            lab = "0->6" if fwd else "6->0"
            check(f"{name} sweep {lab} ({rounds} rounds)", v, out[fwd][0])
            check_ci(f"{name} sweep {lab} CI", ci, out[fwd][1], out[fwd][2])
        check(f"{name} sweep 0->6 / sum of isolated gains", ratio, out[True][0] / sum(singles))
        ends[name] = abs(out[True][0] - out[False][0])
    check("sweep directions differ, DFlash", "0.085", ends["DFlash"])
    check("sweep directions differ, DSpark (\"within 0.001\")", "0.001", ends["DSpark"])


# ------------------------------------------------------------------ Sec 6 ---
def solution():
    section("Sec. 6 / tab:solution-gap: DSpark and ours, paired (rpre_compare)")
    a, _ = RP.load(archived("rpre_o1/*.rpre1.jsonl.gz"))
    b, _ = RP.load(archived("rpre_o1_ours/*.rpre1.jsonl.gz"))
    d = RC.decomposition(a, b)
    tab = {"floor": (("0.00", "1e-3", "4.8e-3", "0.02", "0.03", "0.03", "0.04"),
                     ("0.00", "1e-3", "4.7e-3", "0.02", "0.03", "0.03", "0.04")),
           "oracle_risk": (("0.11", "0.14", "0.21", "0.27", "0.28", "0.35", "0.37"),
                           ("0.11", "0.11", "0.14", "0.19", "0.19", "0.27", "0.28")),
           "self_risk": (("0.11", "0.22", "0.32", "0.40", "0.46", "0.54", "0.58"),
                         ("0.11", "0.21", "0.29", "0.38", "0.43", "0.51", "0.54"))}
    for metric, arms in tab.items():
        for arm, vals in zip(("DSpark", "ours"), arms):
            for k in range(7):
                check(f"{metric} {arm} slot {k}", vals[k], d[k][metric][arm == "ours"])
    gap = [d[k]["model_gap"] for k in range(7)]
    check_true("model gap falls at slots 1-6", "", all(o < s for s, o in gap[1:]))
    check("model gap change at slot 0 (\"essentially unchanged\")", "0.002",
          abs(gap[0][1] - gap[0][0]))
    check("gap reduction at slot 6", "25.5", 1 - gap[6][1] / gap[6][0], pct=True)
    check("slot-6 gap, DSpark", "0.33", gap[6][0])
    check("slot-6 gap, ours", "0.24", gap[6][1])
    check("slot-6 exposure, DSpark", "0.21", d[6]["exposure"][0])
    check("slot-6 exposure, ours", "0.26", d[6]["exposure"][1])
    check_true("exposure rises", "", d[6]["exposure"][1] > d[6]["exposure"][0])


# ------------------------------------------------------------- appendices ---
def robustness():
    section("App. Numerical resolution")
    o1 = load("rpre_o1/*.rpre1.jsonl.gz")
    t0 = load("t0_s6/*.t0.jsonl.gz")
    p, lo, hi, _ = RR.interval(o1, RR.t1_split(1))
    check("slot-1 T^(1) residual", "1e-3", p)
    check_ci("slot-1 residual CI", "[6e-4,1.5e-3]", lo, hi)
    deep = [hajek(o1, RR.t1_split(k)) / p for k in range(3, 7)]
    check_true("deep-slot T^(1) is \"20-40 times\" the residual",
               f"{min(deep):.0f}-{max(deep):.0f}x", 15 < min(deep) and max(deep) < 45)
    check("slot-2 T^(1) over the residual (\"about five\")", "5", hajek(o1, RR.t1_split(2)) / p)

    section("App. Independent replication: full-vocabulary M=256 vs top-256 M=1024")
    rp = load("rpre/*.rpre.jsonl.gz")
    full = ("0.00", "0.08", "0.12", "0.17", "0.21", "0.25", "0.29")
    top = ("0.00", "0.08", "0.12", "0.17", "0.20", "0.25", "0.29")
    diffs = []
    for k in range(7):
        a_, b_ = hajek(rp, RR.flat("T", k)), hajek(t0, RR.nested("T", 0, k))
        check(f"T^(0) full vocabulary slot {k}", full[k], a_)
        check(f"T^(0) top-256 slot {k}", top[k], b_)
        if k:
            diffs.append(abs(a_ - b_))
    check("smallest |difference|, slots 1-6", "7e-4", min(diffs))
    check("largest |difference|, slots 1-6", "4.4e-3", max(diffs), erratum="3.4e-3")
    for slot, paper in ((1, "0.0761"), (3, "0.1715"), (6, "0.2854")):
        check(f"top-256 M=1024 T^(0) slot {slot} (4 d.p.)", paper,
              hajek(t0, RR.nested("T", 0, slot)))
    check("full vocabulary minus top-256 at slot 6", "7e-4", diffs[-1])
    snis = load("t1_fix/*.t01.jsonl.gz")
    p, lo, hi, n = RR.route_difference(o1, snis, 6)
    check("T^(1) routes paired at slot 6", "+3.2e-3", p, erratum="+3.1e-3")
    check_ci("T^(1) routes paired at slot 6, CI (x1e-3)", "[-9.40,+14.30]", 1e3 * lo, 1e3 * hi,
             erratum="[-8.56,+13.90]")
    check("largest |paired route difference|, slots 1-6", "0.007",
          max(abs(hajek(o1, RR.route_gap(snis, k))) for k in range(1, 7)))
    check("T^(1) reweighting route slot 1 (0 by identity)", "0.0000",
          hajek(snis, RR.nested("T", 1, 1)))
    check("T^(1) reweighting route slot 6, held-out", "0.0382",
          hajek(snis, RR.nested("T_split", 1, 6)))
    worst, prev = 0.0, None
    for k in range(7):
        a_, b_ = hajek(t0, RR.nested("T", 0, k)), hajek(rp, RR.flat("T", k))
        if prev is not None and b_ != prev:
            worst = max(worst, abs(a_ - b_) / abs(b_ - prev))
        prev = b_
    #   probe_tk reaches slot k by teacher-forced rescoring and has to pick the
    #   right row; probe_rpre reads p from the forward that samples the token and
    #   picks none. A one-slot misalignment would show at the size of the step.
    check_true("T^(0) alignment: worst gap as a share of the slot step", f"{worst:.1%} < 25%",
               worst < 0.25)

    section("App. Finite-sample sensitivity: M=1024 vs M=256, same engine")
    tk = load("tk/*.t0.jsonl.gz")
    m256 = load("t1_fix/*.t01.m256.jsonl.gz")
    # The printed table predates the reruns and is not what the archive gives.
    t0row = (("1.40", "1.32"), ("0.90", "0.83"), ("2.20", "2.40"), ("2.20", "2.14"),
             ("1.30", "1.21"))
    t1row = (("0.00", None), ("0.30", "0.14"), ("0.40", "1.21"), ("0.70", "1.86"),
             ("1.20", "0.11"), ("0.20", "0.08"))
    moved = []
    for k, (p_, e_) in enumerate(t0row, 1):
        v = 1e3 * abs(hajek(t0, RR.nested("T", 0, k)) - hajek(tk, RR.nested("T", 0, k)))
        moved.append(v)
        check(f"|T^(0) M=1024 - M=256| slot {k} (x1e-3)", p_, v, erratum=e_)
    for k, (p_, e_) in enumerate(t1row, 1):
        v = 1e3 * abs(hajek(snis, RR.nested("T", 1, k)) - hajek(m256, RR.nested("T", 1, k)))
        moved.append(v)
        check(f"|T^(1) M=1024 - M=256| slot {k} (x1e-3)", p_, v, erratum=e_)
    check("largest move under 4x paths (x1e-3)", "2.2", max(moved), erratum="2.4")
    check_true("order-0 M=256 run has no slot 6", "",
               all(RR.nested("T", 0, 6)(r) is None for r in tk))
    check("full vocabulary M=256 minus top-256 M=1024 at slot 6", "7e-4",
          abs(hajek(rp, RR.flat("T", 6)) - hajek(t0, RR.nested("T", 0, 6))))
    ess = [hajek(snis, RR.nested("ess", 1, k)) for k in range(2, 7)]
    check("order-1 importance ESS at M=1024, low", "693", min(ess), erratum="688")
    check("order-1 importance ESS at M=1024, high", "924", max(ess))
    fs = lambda recs: hajek(recs, RR.nested("T_split", 1, 6)) - hajek(recs, RR.nested("T", 1, 6))
    check("slot-6 fit-score difference at M=256", "+6.4e-3", fs(m256), erratum="+6.6e-3")
    check("slot-6 fit-score difference at M=1024", "+2e-3", fs(snis))
    doms = by_domain("tk/*.t0.jsonl.gz")
    split = lambda recs, k: (hajek(recs, RR.nested("T_split", 0, k))
                             - hajek(recs, RR.nested("T", 0, k)))
    per = [abs(split(v, k)) for v in doms.values() for k in range(1, 6)]
    check("order-0 split-half change, any domain and slot", "0.006", max(per))
    check("order-0 split-half change, pooled", "0.003", max(abs(split(tk, k)) for k in range(1, 6)))

    section("App. Vocabulary truncation: top-20 vs top-256 on the same paths, M=256")
    t20 = load("tk20/*.t01.tk20.jsonl.gz")
    ref = {(r["prompt_id"], r["t"]): r for r in m256}
    worst = 0.0
    for k, v in zip(range(1, 7), ("-1.4e-7", "+1.4e-5", "+2.5e-6", "+1.6e-5", "+1.6e-5",
                                  "+1.0e-5")):
        d = hajek(t20, lambda r: RR.nested("T", 0, k)(r)
                  - RR.nested("T", 0, k)(ref[(r["prompt_id"], r["t"])]))
        worst = max(worst, abs(d))
        check(f"paired T^(0) difference slot {k}", v, d)
    check_true("Hajek-weighted difference at most 1.7e-5", f"{worst:.2e}", worst <= 1.7e-5)

    section("App. Sampling-law sensitivity: gsm8k, C0 law vs the training law (C1)")
    base = load("rpre_c1/gsm8k.c1prefix_c0law.jsonl.gz")
    train = load("rpre_c1/gsm8k.rpre.jsonl.gz")
    tab = {0: ("0", "0", "0.02", "0.02", "0.02", "0.02", "100", "100"),
           3: ("0.09", "0.06", "0.19", "0.18", "0.10", "0.12", "52", "69"),
           5: ("0.15", "0.09", "0.26", "0.22", "0.11", "0.13", "43", "58"),
           6: ("0.16", "0.09", "0.32", "0.29", "0.16", "0.19", "49", "67")}
    for k, (tb, tt, rb, rt, gb, gt, sb, st) in tab.items():
        vb = [hajek(base, RR.flat(f, k)) for f in ("T", "R")]
        vt = [hajek(train, RR.flat(f, k)) for f in ("T", "R")]
        check(f"T base slot {k}", tb, vb[0])
        check(f"T train slot {k}", tt, vt[0])
        check(f"R base slot {k}", rb, vb[1])
        check(f"R train slot {k}", rt, vt[1])
        check(f"G base slot {k}", gb, vb[1] - vb[0])
        check(f"G train slot {k}", gt, vt[1] - vt[0])
        check(f"G/R base slot {k}", sb, (vb[1] - vb[0]) / vb[1], pct=True)
        check(f"G/R train slot {k}", st, (vt[1] - vt[0]) / vt[1], pct=True)
    fb, ft = hajek(base, RR.flat("T", 6)), hajek(train, RR.flat("T", 6))
    rb, rt = hajek(base, RR.flat("R", 6)), hajek(train, RR.flat("R", 6))
    check("training law lowers the slot-6 floor by", "42", 1 - ft / fb, pct=True)
    check("and the slot-6 drafter risk by", "11", 1 - rt / rb, pct=True)
    warp = max(abs(hajek(train, RR.flat("R", k)) - hajek(train, RR.flat("R_temp", k)))
               for k in range(7))
    check("the two proposal warpings under the training law differ by at most", "0.008", warp)
    same = max(abs(hajek(base, RR.flat("R", k)) - hajek(base, RR.flat("R_temp", k)))
               for k in range(7))
    check_true("and coincide under the C0 law", f"{same:.1e}", same < 1e-6)

    section("App. Headline uncertainty intervals: tab:floor-intervals (ratio_report)")
    T0 = ("[.055,.101]", "[.092,.149]", "[.136,.211]", "[.164,.246]", "[.202,.292]",
          "[.239,.334]")
    T1 = (None, ("[2.90,7.20]", "[2.86,7.26]"), ("[0.01,0.03]", None), ("[0.02,0.04]", None),
          ("[0.02,0.04]", None), ("[0.03,0.06]", None))
    SH = (None, ("[94.5,97.4]", None), ("[82.3,93.0]", "[82.2,93.0]"),
          ("[84.4,90.8]", "[84.3,90.7]"), ("[85.2,91.1]", "[85.4,91.1]"),
          ("[80.6,89.8]", "[80.7,89.8]"))
    for k in range(1, 7):
        f = RR.floor_intervals(t0, o1, k)
        check_ci(f"T^(0) top-256 slot {k}", T0[k - 1], f["T0"][1], f["T0"][2])
        if T1[k - 1]:
            p_, e_ = T1[k - 1]
            s = 1e3 if k == 2 else 1
            check_ci(f"T^(1) partitioning slot {k}" + (" (x1e-3)" if k == 2 else ""),
                     p_, s * f["T1"][1], s * f["T1"][2], erratum=e_)
        if SH[k - 1]:
            p_, e_ = SH[k - 1]
            check_ci(f"(T0 - T1)/T0 slot {k}", p_, f["share"][1], f["share"][2], pct=True,
                     erratum=e_)


def blocklen():
    section("App. Longer blocks: gamma=16 (blocklen_report)")
    from specfloor import blocklen_report as BL
    by, _ = load_by_domain(archived("g16/*.g16.jsonl.gz"))
    if not by:
        print("  --   measurements/g16/ is not archived yet; block skipped")
        return
    recs, K, rows = BL.table(by)
    check("anchors", "372", len(recs))
    check("slots", "16", K)
    T0 = ("0.00", "0.08", "0.12", "0.17", "0.20", "0.24", "0.29", "0.31",
          "0.33", "0.37", "0.39", "0.41", "0.43", "0.44", "0.46", "0.47")
    for k in range(K):
        check(f"T^(0) slot {k}", T0[k], rows[k]["T0"])
    check("T^(0) slot 15 (text)", "0.473", rows[15]["T0"])
    check("acceptance cap at slot 15", "52.7", 1 - rows[15]["T0"], pct=True)
    for d, v in (("alpaca", "0.619"), ("arena8k", "0.479"), ("mbpp", "0.389"),
                 ("gsm8k", "0.377")):
        check(f"slot-15 T^(0), {d}", v, BL.floor_mean(by[d], 0, 15))
    # T^(1) is the corrected re-run; the first gamma=16 run's probe_tk predated
    # b6e6597 and printed a row about half this size at the deep slots.
    T1 = (None, "0.00", "2.5e-3", "0.02", "0.02", "0.03", "0.04", "0.04",
          "0.04", "0.05", "0.04", "0.04", "0.06", "0.06", "0.05", "0.09")
    for k in range(1, K):
        check(f"T^(1) slot {k}", T1[k], rows[k]["T1"])
    deep = [r["removed"] for r in rows[2:]]
    check("share of T^(0) removed, slots 2-15, low", "80.9", min(deep), pct=True)
    check("share of T^(0) removed, slots 2-15, high", "97.9", max(deep), pct=True)
    check("largest T^(1), slots 1-15", "0.090", max(r["T1"] for r in rows[1:]))
    short = {(r["prompt_id"], r["t"]): r for r in load("t1_fix/*.t01.jsonl.gz")}
    mine = [r for r in recs if (r["prompt_id"], r["t"]) in short]
    theirs = [short[(r["prompt_id"], r["t"])] for r in mine]
    gap = max(abs(BL.floor_mean(mine, 1, k) - BL.floor_mean(theirs, 1, k)) for k in range(1, 7))
    check_true("T^(1) slots 1-6 match the gamma=7 run on shared anchors",
               f"max |diff| {gap:.1e}", gap < 2e-3)


def kmedian():
    section("App. Oracle-routed K-median (kmedian_report)")
    recs = load("branch/*.tb.jsonl.gz")
    check("anchors", "128", len(recs))
    check("paths per anchor", "256", recs[0]["M"])
    fig = {1: ("0.0706", "0.0987", "0.1481", "0.1649", "0.2382", "0.2815"),
           2: ("0.0108", "0.0350", "0.0762", "0.0895", "0.1360", "0.1598"),
           4: ("0.0029", "0.0101", "0.0316", "0.0409", "0.0653", "0.0829")}
    val = {}
    for W, vals in fig.items():
        for k, v in zip(range(1, 7), vals):
            cs = KM.cells(recs, max(W, 2), k)
            val[(W, k)] = KM.wmean([(c["T1" if W == 1 else "TW"], c["w"]) for c in cs])
            check(f"K={W} loss slot {k} (Fig. prototypes)", v, val[(W, k)])
    rp = load("rpre/*.rpre.jsonl.gz")
    t1 = [KM.wmean([(c["T1"], c["w"]) for c in KM.cells(recs, 2, k)]) for k in range(7)]
    same = {(r["prompt_id"], r["t"]) for r in recs}
    ref = [hajek([r for r in rp if (r["prompt_id"], r["t"]) in same], RR.flat("T", k))
           for k in range(7)]
    check_true("K=1 reproduces T^(0) on the same anchors",
               f"max |diff| {max(abs(a - b) for a, b in zip(t1, ref)):.3f}",
               max(abs(a - b) for a, b in zip(t1, ref)) < 0.01)
    check("two prototypes remove at slot 6", "43", 1 - val[(2, 6)] / val[(1, 6)], pct=True)
    check("four prototypes remove at slot 6", "71", 1 - val[(4, 6)] / val[(1, 6)], pct=True)
    sp = KM.spread_summary(recs, [2, 4])
    check("cell mass with zero restart spread", "80", sp["zero"], pct=True)
    check("restart spread, weighted p90", "0.049", sp["p90"])
    check("largest per-(K, slot) mean spread (\"at most 0.034\")", "0.034", sp["max_cell_mean"])
    check("largest restart spread", "0.48", sp["max"])



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true",
                    help="skip the best-response blocks, which refit ~40k water fills")
    args = ap.parse_args()
    setup()
    parallel_blindness()
    conditioning_order()
    drafter_gaps()
    scale()
    serving()
    if not args.fast:
        best_response()
    solution()
    robustness()
    blocklen()
    kmedian()
    print(f"\n{len(OK)} agree, {len(FIX)} paper values to correct, {len(BAD)} disagree")
    if FIX:
        print("\nprint instead:")
        for name, paper, now in FIX:
            print(f"  {name:<62s} {paper:>14} -> {now}")
    return 1 if BAD else 0


if __name__ == "__main__":
    sys.exit(main())
