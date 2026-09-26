"""Every value the paper's figures plot, computed from measurements/ by the package's reports.

Nothing here is typed in. Each function reads the archived records through the
specfloor module that reports the same quantity in text form, so a figure and
the table beside it cannot drift apart. measurements/verify.py checks these
values against the paper.
"""
from __future__ import annotations

import numpy as np

from specfloor import kmedian_report as KM
from specfloor import ratio_report as RR
from specfloor import rpre_compare as RC
from specfloor import rpre_report as RP
from specfloor.records import archived, load_by_domain, weight

SLOTS = np.arange(7)
DOMAINS = (("alpaca", "alpaca"), ("arena8k", "arena-hard"), ("mbpp", "mbpp"),
           ("gsm8k", "gsm8k"))


def _pool(pattern):
    by, _ = load_by_domain(archived(pattern))
    return by, [r for recs in by.values() for r in recs]


def _hajek(recs, y):
    pairs = [(v, weight(r)) for r in recs if (v := y(r)) is not None]
    return sum(v * w for v, w in pairs) / sum(w for _, w in pairs)


def conditioning_order():
    """Fig. t1: T^(0) and T^(1) on the same M=256 full-vocabulary paths.

    The slot-1 order-1 floor is zero by identity -- the predecessor fixes the
    whole realised prefix -- and is drawn as such; what the estimator returns
    there (1e-3) is its numerical resolution, reported in the appendix.
    """
    _, o1 = _pool("rpre_o1/*.rpre1.jsonl.gz")
    slots = np.arange(1, 7)
    t0 = np.array([_hajek(o1, RR.flat("T", k)) for k in slots])
    t1 = np.array([0.0] + [_hajek(o1, RR.t1_split(k)) for k in slots[1:]])
    return slots, t0, t1


def domain_floors():
    """Fig. domain floors: T^(0) per domain, slots 1-6, full vocabulary."""
    by, _ = _pool("rpre/*.rpre.jsonl.gz")
    slots = np.arange(1, 7)
    return slots, [(label, np.array([_hajek(by[d], RR.flat("T", k)) for k in slots]))
                   for d, label in DOMAINS]


def prototypes():
    """Fig. prototypes: oracle-routed K-median loss for K = 1, 2, 4."""
    _, recs = _pool("branch/*.tb.jsonl.gz")
    slots = np.arange(1, 7)
    out = []
    for W in (1, 2, 4):
        col = "T1" if W == 1 else "TW"
        out.append((W, np.array([KM.wmean([(c[col], c["w"])
                                           for c in KM.cells(recs, max(W, 2), k)])
                                 for k in slots])))
    return slots, out


def cross_target():
    """Fig. cross-target: final-slot floor and risk, DFlash and DSpark."""
    names, dfl, dfr, dsf, dsr = [], [], [], [], []
    for name, (g0, g1) in RR.TARGETS.items():
        a = RR.dflash_share(RR.records(g0), 6, B=2)
        b = RR.dspark_share(RR.records(g1), 6, B=2)
        names.append(name)
        dfl.append(a["T"]), dfr.append(a["R"]), dsf.append(b["T1"]), dsr.append(b["R"])
    return names, *(np.array(v) for v in (dfl, dfr, dsf, dsr))


def dflash_decomposition():
    """T^(0) and R per slot for DFlash, from rpre_report's cells."""
    _, rp = _pool("rpre/*.rpre.jsonl.gz")
    floor = np.array([RP.wmean([(c[1], c[5]) for c in RP.cells(rp, k)]) for k in SLOTS])
    risk = np.array([RP.wmean([(c[2], c[5]) for c in RP.cells(rp, k)]) for k in SLOTS])
    return floor, risk


def dspark_decomposition():
    """T^(1), R_oracle, R_self per slot for DSpark; slots 0-1 floors are zero
    by identity and drawn so."""
    _, o1 = _pool("rpre_o1/*.rpre1.jsonl.gz")
    cols = {}
    for key in ("T1s", "Rorc", "Rself"):
        cols[key] = np.array([RP.wmean([(c[key], c["w"]) for c in RP.cells1(o1, k)
                                        if c.get(key) is not None]) for k in SLOTS])
    cols["T1s"][:2] = 0.0
    return cols["T1s"], cols["Rorc"], cols["Rself"]


def single_slot_gains(tag="br0"):
    """Held-out dtau^BR per slot (br_report, 2-fold cross-fit, eps = 0)."""
    from specfloor import br_report as BR
    by = BR.load(archived(f"br/*.{tag}.jsonl.gz"))
    rows = [(r, c) for rs in by.values() for r in rs if (c := BR.anchor_cells(r, [0.0]))]
    return np.array([RP.wmean([(c[k]["held"][0.0], weight(r)) for r, c in rows if k in c])
                     for k in SLOTS])


def solution():
    """Per-slot decomposition of DSpark and ours on the same anchors."""
    a, _ = RP.load(archived("rpre_o1/*.rpre1.jsonl.gz"))
    b, _ = RP.load(archived("rpre_o1_ours/*.rpre1.jsonl.gz"))
    return RC.decomposition(a, b)
