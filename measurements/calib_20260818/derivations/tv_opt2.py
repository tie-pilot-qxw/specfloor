"""T_k = min_q E_Z[TV(p_Z,q)] over the simplex: is the argmin the mixture?

Claim to test: q*(v) = Quantile_beta({p_Z(v)}_Z) at ONE common level beta fixed
by sum_v q*(v) = 1.  Checked against exhaustive grid search over the simplex
(no solver, so no convergence to doubt).  Also checks the sandwich
  (1/2) E_{Z,Z'}[TV(p_Z,p_Z')]  <=  T_k  <=  E_Z[TV(p_Z, pbar)].
"""
import itertools, random, statistics

random.seed(7)
V, STEP = 4, 0.01                       # grid resolution 1/100 per coordinate
G = round(1 / STEP)


def tv(a, b):
    return 0.5 * sum(abs(x - y) for x, y in zip(a, b))


def obj(q, P):
    return sum(tv(p, q) for p in P) / len(P)


def grid_min(P):
    best, argbest = float("inf"), None
    for c in itertools.combinations(range(1, G + V), V - 1):   # compositions of G
        prev, q = 0, []
        for x in c:
            q.append((x - prev - 1) * STEP)
            prev = x
        q.append((G + V - 1 - prev - 1 + 1) * STEP - STEP)
        q[-1] = 1.0 - sum(q[:-1])
        if q[-1] < -1e-12:
            continue
        f = obj(q, P)
        if f < best:
            best, argbest = f, list(q)
    return best, argbest


def quantile(xs, beta):
    """Linear-interpolated empirical quantile (numpy 'linear' convention)."""
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    h = (len(s) - 1) * beta
    lo = int(h)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (h - lo) * (s[hi] - s[lo])


def quantile_solution(P):
    cols = list(zip(*P))
    f = lambda b: sum(quantile(c, b) for c in cols) - 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):                                       # bisection
        mid = (lo + hi) / 2
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    beta = (lo + hi) / 2
    return [quantile(c, beta) for c in cols], beta


def norm(v):
    s = sum(v)
    return [x / s for x in v]


CASES = {
    "flat random":      [norm([random.random() for _ in range(V)]) for _ in range(9)],
    "peaked random":    [norm([random.random() ** 6 for _ in range(V)]) for _ in range(9)],
    "two hard modes":   [[.90, .06, .03, .01]] * 5 + [[.06, .90, .03, .01]] * 4,
    "mode + spread":    [[.85, .08, .04, .03]] * 6 + [norm([random.random() for _ in range(V)]) for _ in range(3)],
    "near-degenerate":  [norm([.97 + random.uniform(-.01, .01), .01, .01, .01]) for _ in range(9)],
}

print(f"{'case':18s} {'T(grid)':>9s} {'T(quant)':>9s} {'mixture':>9s} {'median':>9s} "
      f"{'beta':>6s} {'LB':>8s}   verdict")
bad = 0
for name, P in CASES.items():
    Tg, qg = grid_min(P)
    qq, beta = quantile_solution(P)
    Tq = obj(qq, P)
    mix = [sum(p[v] for p in P) / len(P) for v in range(V)]
    med = norm([statistics.median([p[v] for p in P]) for v in range(V)])
    lb = 0.5 * statistics.mean(tv(P[i], P[j]) for i in range(len(P))
                               for j in range(len(P)) if i != j)
    ok = (Tq <= Tg + 2 * STEP) and (lb <= Tg + 1e-9) and (Tg <= obj(mix, P) + 1e-9)
    bad += not ok
    print(f"{name:18s} {Tg:9.5f} {Tq:9.5f} {obj(mix,P):9.5f} {obj(med,P):9.5f} "
          f"{beta:6.3f} {lb:8.5f}   {'ok' if ok else 'FAIL'}"
          f"   mixture excess {100*(obj(mix,P)-Tq)/max(Tq,1e-12):+.1f}%")

print(f"\n{'FAILURES: %d' % bad if bad else 'all cases: quantile attains the grid optimum; LB <= T <= mixture'}")
print(f"(grid step {STEP} on the {V}-simplex => grid optimum is only accurate to ~{STEP})")
