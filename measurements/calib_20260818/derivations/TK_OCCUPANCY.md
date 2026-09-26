# T^(m): exact-suffix grouping vs SNIS — 2026-08-19

The question: `E_{Z|W_m}` needs posterior samples. Exact-suffix grouping gets them
free (bucket free rollouts by realised suffix) but starves as m grows — a singleton
bucket returns TV = 0 by construction, so T is pushed **down**. Is that fatal?

**No.** Grouping is a *hard* assignment; SNIS is a *soft* one. Force the target
suffix onto every sampled prefix and weight by `u_i = p(w | X, s_i)`; then every
path contributes to the cell and occupancy stops binding.

## Synthetic, truth by enumeration (`tk_est.py`)

400 suffix values, 8 early states — i.e. the m=2/4 regime. True T = 0.34262.

| M | grouping | singleton cells | error | SNIS | ESS med | error |
|---|---|---|---|---|---|---|
| 64 | 0.03894 | 86% | **−0.30367** | 0.32567 | 42 | −0.01695 |
| 128 | 0.07497 | 84% | −0.26765 | 0.33998 | 80 | −0.00263 |
| 256 | 0.12018 | 70% | −0.22244 | 0.35033 | 160 | +0.00771 |
| 512 | 0.18404 | 48% | **−0.15858** | 0.35042 | 315 | +0.00780 |
| 1024 | 0.25783 | 23% | −0.08479 | 0.34419 | 626 | +0.00157 |

It is **bias, not noise**. Five independent draws at M=512:

- grouping 0.180, 0.196, 0.191, 0.188, 0.181 → mean error **−0.155 (−45%)**
- SNIS 0.336, 0.341, 0.342, 0.340, 0.349 → mean error **−0.001 (−0.3%)**

Grouping still under-reports by 25% at M=1024. SNIS is within 5% at M=64.

## Consequences for the protocol

1. **Occupancy is no longer the binding constraint; ESS is.** ESS degrades
   gracefully (315/512 here) where occupancy degrades catastrophically (a
   singleton gives literally one sample).
2. `w(s)` costs nothing extra — it is the teacher-forced probability of the
   revealed tokens under the sampled prefix, already in the same forward pass.
   This is the *same* machinery as the R_m fix.
3. The outer `E_W` needs no separate treatment: use each anchor's ground-truth
   suffix as w, since that is a draw from p(W) by construction, and average
   across anchors.
4. **Still open:** `min_q` is fitted and evaluated on the same weighted sample,
   which biases T *down* independently of occupancy. Not visible above (SNIS
   errs slightly high at M≥256), so it is second-order here, but a split-sample
   variant is implemented in `tk_est.py` and should be checked on real data.

Grouping is retained as a cross-check at m=1, where occupancy is survivable and
the two should agree.
