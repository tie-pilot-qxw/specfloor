"""Can T^(m) be estimated without exact-suffix occupancy?

Three estimators of  T = E_W[ min_q E_{Z|W} TV(p_Z, q) ]  on a synthetic system
where the truth is computable by enumeration:

  (A) GROUPING   free-sample M paths, bucket by exact suffix, solve per bucket.
                 Posterior-correct, but a singleton bucket gives TV=0 by
                 construction, so T is pushed DOWN as m grows.
  (B) SNIS       force the target suffix w onto every sampled prefix and weight
                 by u_i = p(w | X, s_i). Every path contributes to the cell.
  (C) SNIS+SPLIT fit q* on half the (weighted) sample, evaluate on the other
                 half. Guards the separate downward bias from optimising q and
                 scoring it on the SAME draws.
"""
import itertools, math, random

random.seed(0)
V, S, W = 6, 5, 4          # vocab, #early states, #suffix values


def mk_system(rng):
    pri = [rng.random() for _ in range(S)]
    pri = [x / sum(pri) for x in pri]
    pw = []                                  # p(w | s)
    for _ in range(S):
        r = [rng.random() ** 2 for _ in range(W)]
        pw.append([x / sum(r) for x in r])
    py = {}                                  # p(Y | s, w)
    for s in range(S):
        for w in range(W):
            r = [rng.random() ** 3 for _ in range(V)]
            py[(s, w)] = [x / sum(r) for x in r]
    return pri, pw, py


def tv(a, b):
    return 0.5 * sum(abs(x - y) for x, y in zip(a, b))


def wquantile(xs, ws, beta):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    tot = sum(ws)
    c = 0.0
    for i in order:
        c += ws[i]
        if c >= beta * tot:
            return xs[i]
    return xs[order[-1]]


def barycentre(ps, ws):
    """Weighted TV barycentre: common-level weighted quantile, level by norm."""
    cols = list(zip(*ps))
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if sum(wquantile(list(c), ws, mid) for c in cols) - 1.0 < 0:
            lo = mid
        else:
            hi = mid
    b = (lo + hi) / 2
    return [wquantile(list(c), ws, b) for c in cols]


def wloss(ps, ws, q):
    tot = sum(ws)
    return sum(w * tv(p, q) for p, w in zip(ps, ws)) / tot if tot else float("nan")


def truth(pri, pw, py):
    """Exact T by enumeration: outer over w, inner over s | w."""
    T = 0.0
    for w in range(W):
        pwm = sum(pri[s] * pw[s][w] for s in range(S))
        post = [pri[s] * pw[s][w] / pwm for s in range(S)]
        ps = [py[(s, w)] for s in range(S)]
        q = barycentre(ps, post)
        T += pwm * wloss(ps, post, q)
    return T


def sample_paths(pri, pw, M, rng):
    out = []
    for _ in range(M):
        r, c = rng.random(), 0.0
        for s in range(S):
            c += pri[s]
            if r <= c:
                break
        r, c = rng.random(), 0.0
        for w in range(W):
            c += pw[s][w]
            if r <= c:
                break
        out.append((s, w))
    return out


def est_group(paths, py):
    buckets = {}
    for s, w in paths:
        buckets.setdefault(w, []).append(py[(s, w)])
    M = len(paths)
    T, singles = 0.0, 0
    for w, ps in buckets.items():
        ws = [1.0] * len(ps)
        if len(ps) == 1:
            singles += 1
        q = barycentre(ps, ws)
        T += (len(ps) / M) * wloss(ps, ws, q)
    return T, singles, len(buckets)


def est_snis(paths, pri, pw, py, split=False, rng=None):
    """Outer average over the REALISED w of each path (correct marginal);
    inner conditional by weighting every sampled prefix by p(w | s)."""
    prefixes = [s for s, _ in paths]
    T, ess_all = 0.0, []
    for _, w in paths:
        u = [pw[s][w] for s in prefixes]
        ps = [py[(s, w)] for s in prefixes]
        su = sum(u)
        if su <= 0:
            continue
        ess_all.append(su * su / sum(x * x for x in u))
        if not split:
            q = barycentre(ps, u)
            T += wloss(ps, u, q)
        else:
            idx = list(range(len(ps)))
            rng.shuffle(idx)
            a, b = idx[::2], idx[1::2]
            q = barycentre([ps[i] for i in a], [u[i] for i in a])
            T += wloss([ps[i] for i in b], [u[i] for i in b], q)
    n = len(paths)
    return T / n, (sorted(ess_all)[len(ess_all) // 2] if ess_all else 0.0)


rng = random.Random(7)
print(f"{'M':>5} | {'truth':>8} | {'(A) group':>10} {'singletons':>10} {'buckets':>8} |"
      f" {'(B) SNIS':>9} {'ESS med':>8} | {'(C) split':>9}")
for trial in range(3):
    pri, pw, py = mk_system(rng)
    T = truth(pri, pw, py)
    print(f"--- system {trial+1}: true T = {T:.5f} ---")
    for M in (8, 32, 128, 512):
        g, sing, nb = est_group(sample_paths(pri, pw, M, rng), py)
        paths = sample_paths(pri, pw, M, rng)
        b, ess = est_snis(paths, pri, pw, py)
        c, _ = est_snis(paths, pri, pw, py, split=True, rng=random.Random(1))
        print(f"{M:>5} | {T:>8.5f} | {g:>10.5f} {sing:>10} {nb:>8} |"
              f" {b:>9.5f} {ess:>8.1f} | {c:>9.5f}")
