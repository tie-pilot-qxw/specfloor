"""Gate the full-vocab torch barycentre against (a) the verified sparse one,
(b) grid search, (c) the closed form T=(M-1)/M on point masses."""
import sys, math, random, torch
sys.path.insert(0, "/workspace/DeepSpec")
from measurement.probe_rpre import barycentre, mean_tv
from measurement.probe_tk import tv_barycentre, wloss

torch.manual_seed(0); random.seed(0)
ok = True

# (c) closed form: M point masses on distinct coordinates -> T = (M-1)/M
for M in (2, 5, 17, 64):
    V = M + 3
    P = torch.zeros(M, V, dtype=torch.float64)
    for i in range(M):
        P[i, i] = 1.0
    T = mean_tv(P, barycentre(P))
    exp = (M - 1) / M
    good = abs(T - exp) < 1e-9
    ok &= good
    print(f"point-mass M={M:3d}: T={T:.12f} expect={exp:.12f} {'ok' if good else 'FAIL'}")

# (a)+(b) random families: torch-full vs sparse-dict vs grid search over the simplex
for trial in range(4):
    M, V = random.choice([(8, 12), (32, 40), (64, 20), (7, 200)])
    P = torch.rand(M, V, dtype=torch.float64) ** 3
    P = P / P.sum(1, keepdim=True)
    T_new = mean_tv(P, barycentre(P))
    ps = [{v: float(P[i, v]) for v in range(V) if P[i, v] > 0} for i in range(M)]
    ws = [1.0] * M
    T_old = wloss(ps, ws, tv_barycentre(ps, ws))
    # grid: no q in a random search should beat the claimed minimum
    best = min(mean_tv(P, (lambda z: z / z.sum())(torch.rand(V, dtype=torch.float64) ** 3))
               for _ in range(4000))
    best = min(best, min(mean_tv(P, P[i]) for i in range(M)))     # each p_i is a candidate too
    good = abs(T_new - T_old) < 1e-6 and T_new <= best + 1e-9
    ok &= good
    print(f"random M={M:3d} V={V:3d}: full={T_new:.9f} sparse={T_old:.9f} "
          f"best_random={best:.9f} {'ok' if good else 'FAIL'}")

# mass check: the returned q must be a probability vector
for trial in range(3):
    P = torch.rand(23, 50, dtype=torch.float64); P = P / P.sum(1, keepdim=True)
    m = float(barycentre(P).sum())
    good = abs(m - 1.0) < 1e-9 and float(barycentre(P).min()) >= -1e-12
    ok &= good
    print(f"mass check: sum(q)={m:.12f} {'ok' if good else 'FAIL'}")

print("\nALL PASS" if ok else "\nFAILURES ABOVE")
sys.exit(0 if ok else 1)
