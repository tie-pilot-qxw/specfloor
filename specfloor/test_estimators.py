"""Estimator sanity tests. Run before any main pass; cheap, no GPU.

    python -m specfloor.test_estimators

These exist because the jackknife bias correction silently produced a NEGATIVE
cross-entropy on a heavy-tailed sample -- a value that is impossible by
definition, not merely inaccurate. Nothing downstream would have caught it: the
number flowed into CE_B, into dCE, into the ratio R_m, and only showed up as an
absurd population mean three stages later.

Every fixture below is a property that must hold for ANY input, so a future
change to the estimator either keeps them or is wrong.
"""

from __future__ import annotations

import math
import sys

from specfloor.probe_cheap import ce_b_jackknife, ambiguous_slots
from specfloor import config as C

FAIL = []


def check(cond, name, detail=""):
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


# ---------------------------------------------------------------------------
# ce_mixed_all: does the SNIS column actually recover the CONDITIONAL quantity?
#
# R_m names p(y_k | X, Z_{k-m:k-1}), which integrates the earlier segment over
# its POSTERIOR given the reveal. Forcing the reveal onto prior-sampled prefixes
# computes an INTERVENTIONAL quantity instead. Both are now returned; this
# checks each converges to the right thing against brute-force enumeration.
# ---------------------------------------------------------------------------
def _chain_logp(seq, P):
    """Toy autoregressive chain on {0,1}: p(1|...) depends on the last 2 tokens."""
    import math as _m
    tab = {(0, 0): 0.15, (0, 1): 0.75, (1, 0): 0.35, (1, 1): 0.60}
    ctx = tuple(seq[-2:]) if len(seq) >= 2 else (0, 0)
    p1 = tab[ctx]
    return _m.log(p1), _m.log(1.0 - p1)


class _FakeEngine:
    """Exact scorer for the toy chain, matching teacher_forced_nll's contract."""

    def teacher_forced_nll(self, seqs, start_lens):
        out = []
        for s, st in zip(seqs, start_lens):
            row = []
            for pos in range(st, len(s)):
                l1, l0 = _chain_logp(s[:pos], None)
                row.append(-(l1 if s[pos] == 1 else l0))
            out.append(row)
        return out


def test_ce_mixed_all_snis():
    import itertools, math as _m, random as _random
    from specfloor.probe_rm import ce_mixed_all

    PREFIX, K, m, k = [0, 1], 3, 1, 2      # reveal 1 token, predict slot k=2
    gt = [1, 1, 1]

    # brute force over the early segment s = path[:k-m] (one token here)
    def seqp(seq):
        lp = 0.0
        for pos in range(len(PREFIX), len(seq)):
            l1, l0 = _chain_logp(seq[:pos], None)
            lp += l1 if seq[pos] == 1 else l0
        return _m.exp(lp)

    early = list(itertools.product([0, 1], repeat=k - m))
    pri = {s: seqp(list(PREFIX) + list(s)) for s in early}
    Z = sum(pri.values())
    pri = {s: v / Z for s, v in pri.items()}
    w, f = {}, {}
    for s in early:
        base = list(PREFIX) + list(s)
        l1, l0 = _chain_logp(base, None)
        w[s] = _m.exp(l1 if gt[k - m] == 1 else l0)          # p(reveal | s)
        l1b, l0b = _chain_logp(base + [gt[k - m]], None)
        f[s] = _m.exp(l1b if gt[k] == 1 else l0b)            # p(y_k | s, reveal)

    true_cond = (sum(pri[s] * w[s] * f[s] for s in early)
                 / sum(pri[s] * w[s] for s in early))
    true_intv = sum(pri[s] * f[s] for s in early)

    # draw many paths from the prior so the estimators can converge
    rng = _random.Random(0)
    drawn = []
    for _ in range(20000):
        seq = list(PREFIX)
        for _ in range(K):
            l1, _l0 = _chain_logp(seq, None)
            seq.append(1 if rng.random() < _m.exp(l1) else 0)
        drawn.append(seq[len(PREFIX):])

    snis, interv, ess = ce_mixed_all(_FakeEngine(), PREFIX, gt, drawn, K,
                                     (m,), range(1, K))
    got_c, got_i = _m.exp(-snis[m][k]), _m.exp(-interv[m][k])
    check(abs(got_c - true_cond) < 5e-3,
       "SNIS column recovers the CONDITIONAL p(y_k | X, reveal)",
       f"got={got_c:.5f} true={true_cond:.5f}")
    check(abs(got_i - true_intv) < 5e-3,
       "interventional column recovers the FORCED-reveal quantity",
       f"got={got_i:.5f} true={true_intv:.5f}")
    check(abs(true_cond - true_intv) > 1e-3,
       "the two estimands genuinely differ on this fixture (else the test is vacuous)",
       f"cond={true_cond:.5f} interv={true_intv:.5f} gap={true_intv-true_cond:+.5f}")
    check(0 < ess[m][k] <= len(drawn), "ESS is reported and in range",
       f"ess={ess[m][k]:.1f}/{len(drawn)}")


# ---------------------------------------------------------------------------
# tv_barycentre: does the common-level quantile actually attain the minimum,
# and does it return a PROBABILITY VECTOR?
#
# The per-coordinate quantile is a STEP function of beta, so sum(q(beta)) jumps
# and generically never equals 1 exactly. Bisecting alone therefore returns a
# sub- or super-probability vector, and scoring TV against it silently yields
# ~0.5 everywhere -- which is what the first live run of probe_tk produced.
# Interpolating across the jump lands on the member of the argmin SET that has
# the right mass. The degenerate fixture below is the one that exposed it.
# ---------------------------------------------------------------------------
def test_tv_barycentre():
    import itertools
    from specfloor.probe_tk import tv_barycentre, wloss

    def grid_min(ps, ws, V, G=60):
        best = float("inf")
        for c in itertools.combinations(range(1, G + V), V - 1):
            prev, q = 0, []
            for x in c:
                q.append((x - prev - 1) / G); prev = x
            q.append(1.0 - sum(q))
            if q[-1] < -1e-12:
                continue
            best = min(best, wloss(ps, ws, {i: q[i] for i in range(V)}))
        return best

    V = 4
    fixtures = {
        "distinct point masses (the degenerate case)":
            [{0: 1.0}, {1: 1.0}, {2: 1.0}, {3: 1.0}],
        "two modes":
            [{0: 0.9, 1: 0.1}] * 3 + [{2: 0.9, 3: 0.1}] * 2,
        "one mode plus an outlier":
            [{0: 0.97, 1: 0.03}] * 5 + [{3: 1.0}],
    }
    for name, ps in fixtures.items():
        ps = [{k: v / sum(p.values()) for k, v in p.items()} for p in ps]
        ws = [1.0] * len(ps)
        q = tv_barycentre(ps, ws)
        check(abs(sum(q.values()) - 1.0) < 1e-9,
              f"{name}: barycentre is a probability vector",
              f"sum(q)={sum(q.values()):.9f}")
        got, ref = wloss(ps, ws, q), grid_min(ps, ws, V)
        check(got <= ref + 2.0 / 60,
              f"{name}: attains the grid minimum",
              f"T={got:.5f} grid={ref:.5f}")
    # M distinct point masses: any single q is right for one path and wrong for
    # the rest, so the floor is exactly (M-1)/M. A closed form to anchor on.
    ps = [{i: 1.0} for i in range(5)]
    q = tv_barycentre(ps, [1.0] * 5)
    check(abs(wloss(ps, [1.0] * 5, q) - 4 / 5) < 1e-6,
          "M distinct point masses: T = (M-1)/M exactly",
          f"got {wloss(ps, [1.0]*5, q):.6f} want 0.800000")


def test_kmedian():
    """The K-median must reduce to T^(0) at K=1, be monotone, and use weights.

    The closed form to anchor on: M distinct point masses. One centre is right
    for one path and wrong for the other M-1, so T^(0) = (M-1)/M; K centres
    cover K of them, so T^(branch,K) = (M-K)/M exactly. Lloyd has to find that,
    and it is the only fixture here whose optimum is known rather than bounded.
    """
    import random
    from specfloor.probe_tk import floor_from
    from specfloor.probe_kmedian import kmedian

    rng = random.Random(7)

    def mk(support, conc):
        w = [rng.gammavariate(conc, 1.0) for _ in support]
        s = sum(w)
        return {v: x / s for v, x in zip(support, w)}

    ps = [mk(range(20), 0.4) for _ in range(200)]
    ws = [1.0] * len(ps)
    T, _ = floor_from(ps, ws)
    b1, sp1 = kmedian(ps, ws, 1)
    check(b1 == T, "K=1 IS the single barycentre, bit for bit",
          f"branch={b1!r} floor_from={T!r}")
    check(sp1 == 0.0, "K=1 has no restart spread", f"spread={sp1!r}")

    vals = [kmedian(ps, ws, K, restarts=3, iters=12)[0] for K in (1, 2, 4, 8)]
    check(all(a >= b - 1e-9 for a, b in zip(vals, vals[1:])),
          "objective is non-increasing in K",
          " ".join(f"{v:.4f}" for v in vals))

    # separated clusters: one centre cannot cover both, two can.
    ps2 = [mk(range(0, 10), 6.0) for _ in range(100)] + \
          [mk(range(50, 60), 6.0) for _ in range(100)]
    ws2 = [1.0] * len(ps2)
    v1 = kmedian(ps2, ws2, 1)[0]
    v2, sp2 = kmedian(ps2, ws2, 2, restarts=3, iters=12)
    check(v2 < 0.5 * v1, "two separated clusters collapse at K=2",
          f"K=1 {v1:.4f} -> K=2 {v2:.4f}")
    check(sp2 < 1e-6, "and every restart finds the same split",
          f"spread={sp2:.2e}")

    # weights are not decoration: tilting the mass must move the K=1 centre.
    wtilt = [9.0] * 100 + [1.0] * 100
    check(kmedian(ps2, wtilt, 1)[0] < v1 - 0.05,
          "weights move the single centre toward the heavy cluster",
          f"tilted {kmedian(ps2, wtilt, 1)[0]:.4f} vs flat {v1:.4f}")

    # exact optimum, distinct point masses: T^(branch,K) = (M - K)/M.
    M = 6
    pm = [{i: 1.0} for i in range(M)]
    wm = [1.0] * M
    for Kb in (1, 2, 3):
        got = kmedian(pm, wm, Kb, restarts=4, iters=12)[0]
        check(abs(got - (M - Kb) / M) < 1e-9,
              f"point masses, K={Kb}: attains the exact (M-K)/M optimum",
              f"got {got:.6f} want {(M - Kb) / M:.6f}")

def test_best_response():
    """Water filling is the exact optimum, and dtau really is E[S_k F_k da_k].

    The second is the load-bearing one: the whole single-slot construction rests
    on tau - 1 splitting into a part with no a_k in it plus E[S_k a_k F_k], so it
    is checked against tau computed directly rather than assumed.
    """
    import itertools
    import random
    from specfloor.br_report import waterfill, tau_of, sk_fk

    rng = random.Random(11)

    def obj(items, q):
        return sum(c * min(1.0, q.get(v, 0.0) / p) for v, p, c in items)

    def grid_max(items, V, G=40):
        best = -1.0
        for cut in itertools.combinations(range(1, G + V), V - 1):
            prev, qs = 0, []
            for x in cut:
                qs.append((x - prev - 1) / G)
                prev = x
            qs.append(1.0 - sum(qs))
            if qs[-1] < -1e-12:
                continue
            best = max(best, obj(items, {i: qs[i] for i in range(V)}))
        return best

    for trial in range(4):
        V = 3
        items = [(rng.randrange(V), round(rng.uniform(0.05, 0.6), 3),
                  round(rng.uniform(0.1, 3.0), 3)) for _ in range(7)]
        q, left = waterfill(items)
        check(obj(items, q) >= grid_max(items, V) - 2.0 / 40,
              f"water fill attains the grid optimum (trial {trial})",
              f"wf={obj(items, q):.4f} grid={grid_max(items, V):.4f}")
        check(abs(sum(q.values()) + left - 1.0) < 1e-9,
              f"allocated + leftover = 1 (trial {trial})")

    K = 5

    def mk():
        return dict(a=[round(rng.uniform(0.2, 0.95), 4) for _ in range(K)],
                    alive=[True] * K, tok=[0] * K, p=[0.5] * K,
                    q=[0.5] * K, cond=[0] * K)

    for k in range(K):
        paths = [mk() for _ in range(40)]
        d = 0.31
        got = (tau_of(paths, K, override=(k, lambda e: min(1.0, e["a"][k] + d)))
               - tau_of(paths, K))
        want = sum(sk_fk(e, k, K)[0] * sk_fk(e, k, K)[1]
                   * (min(1.0, e["a"][k] + d) - e["a"][k]) for e in paths) / len(paths)
        check(abs(got - want) < 1e-9,
              f"slot {k}: dtau equals mean[S_k F_k da_k]",
              f"tau route {got:+.6f} against decomposition {want:+.6f}")

    dead = mk()
    dead["a"][2] = 0.0
    dead["alive"][2] = False
    check(abs(tau_of([dead], K) - (1 + dead["a"][0] + dead["a"][0] * dead["a"][1]))
          < 1e-12, "a slot the target did not reach truncates the block")

def main() -> None:
    print("ce_b_jackknife -- domain properties")

    # The regression fixture. plug-in 2.18 nats, raw jackknife -13.22.
    dom = [0.9] + [1e-9] * 7
    plug, corr, se, ok = ce_b_jackknife(dom)
    check(corr >= 0.0, "one dominant path: CE >= 0",
          f"plug={plug:.4f} corrected={corr:.4f} jk_ok={ok}")
    check(not ok, "one dominant path: flagged as unusable")
    check(abs(corr - plug) < 1e-12, "one dominant path: falls back to plug-in")

    # CE = -log(pbar) with pbar in (0,1], so CE >= 0 for every input, and the
    # correction removes an UPWARD bias so it can never exceed the plug-in.
    cases = {
        "uniform": [0.3] * 8,
        "mild tail": [0.5, 0.2, 0.05] + [1e-4] * 5,
        "all tiny": [1e-9] * 8,
        "all near 1": [0.999] * 16,
        "two-scale": [0.4, 0.4] + [1e-12] * 30,
        "single huge + noise": [0.99] + [1e-7] * 63,
        "n=3 minimum": [0.5, 0.2, 0.1],
        "descending decade": [10.0 ** -i for i in range(1, 9)],
    }
    bad_neg = bad_dir = 0
    for name, pg in cases.items():
        p, c, s, o = ce_b_jackknife(pg)
        if c < 0:
            bad_neg += 1
        if c > p + 1e-9:
            bad_dir += 1
    check(bad_neg == 0, "CE >= 0 on every fixture", f"{len(cases)} fixtures")
    check(bad_dir == 0, "corrected <= plug-in on every fixture "
                        "(bias is upward, so a correction must reduce)")

    # plug-in is exactly -log(mean p); this pins the definition, not the fix.
    pg = [0.2, 0.4, 0.6]
    p, _, _, _ = ce_b_jackknife(pg)
    check(abs(p - (-math.log(sum(pg) / len(pg)))) < 1e-12,
          "plug-in equals -log(mean p)")

    # A perfectly homogeneous sample has zero MC bias and zero SE, so the
    # correction must be an exact no-op -- the sharpest test of the formula.
    p, c, s, o = ce_b_jackknife([0.25] * 32)
    check(abs(c - p) < 1e-9 and s < 1e-9,
          "homogeneous sample: correction and SE both vanish",
          f"plug={p:.6f} corrected={c:.6f} se={s:.2e}")

    # No data must yield NO ESTIMATE (NaN + jk_ok False), not a fabricated
    # number. 0.0 here would silently read as "CE is zero", i.e. p = 1.
    try:
        p, c, s, o = ce_b_jackknife([])
        check(c != c and not o, "empty sample: NaN and flagged, not fabricated",
              f"corrected={c} jk_ok={o}")
    except Exception as e:          # noqa: BLE001
        check(False, f"empty sample: raised {type(e).__name__}")

    # Non-empty degenerate inputs must return a finite, non-negative CE.
    for name, pg in (("n=1", [0.5]), ("n=2", [0.5, 0.1]),
                     ("all zero", [0.0] * 8), ("one zero", [0.0, 0.4, 0.4])):
        try:
            p, c, s, o = ce_b_jackknife(pg)
            check(c == c and c >= 0.0 and abs(c) != float("inf"),
                  f"degenerate {name}: finite, non-negative",
                  f"corrected={c:.4f}")
        except Exception as e:      # noqa: BLE001 - the point is that it cannot
            check(False, f"degenerate {name}: raised {type(e).__name__}")

    # Monotone in the mean: scaling every probability down raises CE by exactly
    # -log(scale), so the estimator cannot be inventing curvature.
    a = [0.4, 0.2, 0.1, 0.05] * 4
    b = [x / 10 for x in a]
    pa, _, sa, _ = ce_b_jackknife(a)
    pb, _, sb, _ = ce_b_jackknife(b)
    check(abs((pb - pa) - math.log(10)) < 1e-9,
          "scaling p by 1/10 shifts CE by log 10")
    check(abs(sa - sb) < 1e-9, "SE is scale-free (it is a CV)")

    print("\nambiguous_slots -- threshold gate")
    eps = C.INFORMATIVE_THRESHOLD
    # far from the threshold with a huge SE: NOT ambiguous, because no plausible
    # amount of noise moves a dCE of 40 nats across a 0.01 boundary
    check(ambiguous_slots([40.0], [5.0]) == [],
          "large dCE + large SE is not ambiguous")
    # right at the threshold with a small SE: IS ambiguous
    check(ambiguous_slots([eps + 0.001], [0.002]) == [0],
          "dCE at the threshold with small SE is ambiguous")
    check(ambiguous_slots([0.5], [0.001]) == [],
          "dCE clear of the threshold with small SE is not ambiguous")
    check(ambiguous_slots([0.02], [float("inf")]) == [],
          "unusable SE is skipped rather than counted")
    check(ambiguous_slots([float("nan")], [0.001]) == [],
          "NaN dCE does not crash the gate")

    print()
    print("ce_mixed_all -- interventional vs conditional")
    test_ce_mixed_all_snis()

    print()
    print("tv_barycentre -- attains the min AND returns a distribution")
    test_tv_barycentre()

    print()
    print("kmedian -- reduces to T^(0) at K=1 and is monotone in K")
    test_kmedian()

    print()
    print("best response -- water filling is exact and dtau decomposes")
    test_best_response()

    print()
    if FAIL:
        print(f"{len(FAIL)} FAILED: {FAIL}")
        sys.exit(1)
    print("all estimator properties hold.")


if __name__ == "__main__":
    main()
