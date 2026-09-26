"""How loose is the sandwich, and is the quantile solution robust at scale?

Cross-checks the common-level-quantile characterisation against projected
subgradient descent (works for any V, unlike grid search), then measures how
wide [LB, mixture-UB] is relative to the true T_k -- the number that decides
whether PROTOCOL should quote a sandwich or compute T_k directly.
"""
import random, statistics

random.seed(11)


def tv(a, b):
    return 0.5 * sum(abs(x - y) for x, y in zip(a, b))


def obj(q, P):
    return sum(tv(p, q) for p in P) / len(P)


def project(v):
    """Euclidean projection onto the probability simplex (Duchi et al.)."""
    u = sorted(v, reverse=True)
    css, rho, theta = 0.0, 0, 0.0
    for i, x in enumerate(u):
        css += x
        if x - (css - 1) / (i + 1) > 0:
            rho, theta = i + 1, (css - 1) / (i + 1)
    return [max(x - theta, 0.0) for x in v]


def subgrad(P, iters=60000):
    V = len(P[0])
    q = [1.0 / V] * V
    best, bq = obj(q, P), q[:]
    for t in range(1, iters + 1):
        g = [0.0] * V
        for p in P:
            for v in range(V):
                if q[v] > p[v]:
                    g[v] += 0.5
                elif q[v] < p[v]:
                    g[v] -= 0.5
        g = [x / len(P) for x in g]
        step = 0.5 / (t ** 0.5)
        q = project([q[v] - step * g[v] for v in range(V)])
        f = obj(q, P)
        if f < best:
            best, bq = f, q[:]
    return best, bq


def quantile(xs, beta):
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    h = (len(s) - 1) * beta
    lo = int(h); hi = min(lo + 1, len(s) - 1)
    return s[lo] + (h - lo) * (s[hi] - s[lo])


def quantile_solution(P):
    cols = list(zip(*P))
    lo, hi = 0.0, 1.0
    for _ in range(120):
        mid = (lo + hi) / 2
        if sum(quantile(c, mid) for c in cols) - 1.0 < 0:
            lo = mid
        else:
            hi = mid
    b = (lo + hi) / 2
    return [quantile(c, b) for c in cols], b


def norm(v):
    s = sum(v)
    return [x / s for x in v]


def mk(V, M, kind):
    if kind == "unimodal":
        base = norm([0.9] + [random.random() * 0.02 for _ in range(V - 1)])
        return [norm([max(x + random.gauss(0, 0.01), 1e-9) for x in base]) for _ in range(M)]
    if kind == "bimodal":            # the branch case: paths split between 2 futures
        a = norm([0.88, 0.05] + [random.random() * 0.01 for _ in range(V - 2)])
        b = norm([0.05, 0.88] + [random.random() * 0.01 for _ in range(V - 2)])
        return [norm([max(x + random.gauss(0, 0.01), 1e-9) for x in (a if random.random() < .55 else b)])
                for _ in range(M)]
    if kind == "mode+spread":        # most paths agree, a minority scatters
        base = norm([0.85, 0.08] + [random.random() * 0.02 for _ in range(V - 2)])
        return [norm([random.random() ** 2 for _ in range(V)]) if random.random() < .3
                else norm([max(x + random.gauss(0, 0.01), 1e-9) for x in base]) for _ in range(M)]
    return [norm([random.random() ** 4 for _ in range(V)]) for _ in range(M)]


print(f"{'family':13s} {'V':>3s} {'M':>3s} {'T*(subgrad)':>11s} {'T(quantile)':>11s} {'beta':>6s} "
      f"{'LB':>8s} {'UB=mix':>8s} {'UB excess':>9s} {'sandwich width / T':>18s}")
rows = []
for kind in ("unimodal", "bimodal", "mode+spread", "heavy-tail"):
    for V, M in ((8, 24), (32, 64)):
        P = mk(V, M, kind)
        Ts, _ = subgrad(P, iters=4000 if V > 8 else 8000)
        qq, beta = quantile_solution(P)
        Tq = obj(qq, P)
        mix = [sum(p[v] for p in P) / M for v in range(V)]
        UB = obj(mix, P)
        LB = 0.5 * statistics.mean(tv(P[i], P[j]) for i in range(M) for j in range(M) if i != j)
        rows.append((kind, Tq, LB, UB))
        print(f"{kind:13s} {V:3d} {M:3d} {Ts:11.5f} {Tq:11.5f} {beta:6.3f} "
              f"{LB:8.5f} {UB:8.5f} {100*(UB-Tq)/Tq:8.1f}% {100*(UB-LB)/Tq:17.0f}%")
        assert Tq <= Ts + 1e-6, f"quantile worse than subgradient: {Tq} vs {Ts}"
        assert LB <= Tq + 1e-9 <= UB + 1e-9, "sandwich violated"

print("\nquantile solution >= subgradient in every case (it is the optimum).")
w = [100 * (u - l) / t for _, t, l, u in rows]
print(f"sandwich width as % of T_k: min {min(w):.0f}%  median {statistics.median(w):.0f}%  max {max(w):.0f}%")
