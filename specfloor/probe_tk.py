"""T_k^(m): the acceptance floor a realisation-blind drafter cannot beat.

    T_k^(m) = E_{W_m}[ min_q E_{Z|W_m} TV( p(.|X,Z), q ) ],   W_m = (X, Z_{k-m:k-1})

Everything else in this package measures cross-entropy, which needs one number
per position. TOTAL VARIATION needs the whole next-token distribution, so this
probe is the only one that asks for top-K logprobs, and it is the only place the
truncation caveat applies.

Why this quantity. 1 - alpha = TV(p, q) is the per-slot rejection loss under
speculative sampling, so T_k^(m) is a floor on rejection for ANY drafter of ANY
size whose slot-k proposal cannot see the realisation beyond the last m tokens.
Measured against a real drafter's R = E_Z[TV(p_Z, q_D)], the gap R - T is the
part capacity can still remove and T is the part it cannot.

Two rungs are measured here.

  m = 0   one bucket: every sampled path conditions slot k, and the barycentre
          is taken over all M of them. No conditioning problem at all, which is
          why it is the rung to trust first -- and it is the one that gives
          G_pre = R_pre - T^(0).

  m = 1   needs E_{Z | Z_{k-1}}, a POSTERIOR over the earlier segment. Exact-
          suffix grouping is posterior-correct but starves (a singleton bucket
          returns TV = 0 by construction, biasing T down; measured at -45% on a
          synthetic at M=512). Instead: force the realised token at k-1 onto
          every sampled prefix and weight by w(s) = p(z* | X, s), which is
          self-normalised importance sampling and exact in the limit. Every path
          then contributes to the cell, so ESS -- not bucket occupancy -- is the
          binding constraint. See derivations/TK_OCCUPANCY.md.

The estimator of min_q is the common-level weighted quantile:
the TV barycentre is an L1 object, so it is a MEDIAN-type point, not the mixture
E[p_Z]. The mixture is the LOG-LOSS barycentre, which is why CE_B is right for
dCE and wrong here; using it overstates T by up to 24% on bimodal families.

TWO BIASES THAT ARE REPORTED, NOT HIDDEN.
  * Truncation. Restricting to the union of per-path top-K supports drops
    non-negative terms from every |p_i(v) - q(v)|, so TV falls and T is biased
    DOWN -- which FLATTERS the drafter it is compared against. Mean residual
    mass is reported per cell and gated.
  * Winner's curse. q* is fitted and scored on the same weighted sample, which
    also biases T down. --split fits on half the draws and scores on the other
    half; the difference between the two is the size of that bias.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import pathlib
import random
import time

from transformers import AutoTokenizer

from specfloor import config as C
from specfloor import backend_api as BA
from specfloor.corpus import resolve_stops
from specfloor.probe_cheap import anchor_seed, sample_paths


# TK_PROFILE=1 breaks the per-anchor wall clock down by phase. Off by default and
# costs one perf_counter call per phase when on. It exists because the first live
# run of this probe left the GPU at 0% -- the cost was in a pure-Python barycentre,
# not in the model -- and guessing at that a second time would be a waste.
PROF = collections.defaultdict(float) if os.environ.get("TK_PROFILE") else None


def _tick(prof, name, t0):
    if prof is None:
        return t0
    now = time.perf_counter()
    prof[name] += now - t0
    return now


def _report_profile(prof):
    if not prof:
        return
    counts = {k: v for k, v in prof.items() if k.startswith("_")}
    prof = {k: v for k, v in prof.items() if not k.startswith("_")}
    tot = sum(prof.values())
    gpu = sum(v for k, v in prof.items() if k.startswith("GPU"))
    print(f"\n{'phase':>18} {'seconds':>9} {'share':>7}")
    for k in sorted(prof, key=lambda x: -prof[x]):
        print(f"{k:>18} {prof[k]:>9.2f} {100 * prof[k] / tot:>6.1f}%")
    print(f"{'-- GPU':>18} {gpu:>9.2f} {100 * gpu / tot:>6.1f}%")
    print(f"{'-- CPU':>18} {tot - gpu:>9.2f} {100 * (tot - gpu) / tot:>6.1f}%")
    if counts.get("_seqs_requested"):
        r, u = counts["_seqs_requested"], counts["_seqs_scored"]
        print(f"{'dedup':>18} {u:>9.0f} scored of {r:.0f} requested "
              f"({r / max(u, 1):.1f}x)")
    na = counts.get("_anchors", 0)
    if na:
        d = sorted((k, v) for k, v in counts.items() if k.startswith("_distinct@"))
        print(f"{'distinct paths':>18} " + " ".join(
            f"{k.split('@')[1]}:{v / na:.0f}" for k, v in d) +
            "   (mean distinct j-token prefixes per anchor)", flush=True)


# --------------------------------------------------------------- estimator ---
def _prep(ps, ws):
    """Sparse per-coordinate columns, sorted ONCE.

    The dense form -- one length-M column per vocabulary entry, re-sorted at
    every bisection step -- is what made the first live run leave the GPU at 0%
    utilisation: with M=256 paths and top-K=256 the union support runs to tens
    of thousands of tokens, so it was 60 x |support| sorts of 256 values per
    cell. But a token in one path's top-K is absent from most others, so each
    column is (M - n_v) zeros followed by n_v sorted values; only the non-zeros
    need storing. Total work becomes O(nnz log) with nnz = M x K.
    """
    W = sum(ws)
    cols = {}
    for p, w in zip(ps, ws):
        for v, x in p.items():
            cols.setdefault(v, []).append((x, w))
    pre = {}
    for v, lst in cols.items():
        lst.sort()
        vals, cw, acc = [], [], 0.0
        for x, w in lst:
            acc += w
            vals.append(x)
            cw.append(acc)
        pre[v] = (vals, cw, W - acc)          # trailing entry: weight of ABSENT paths
    return pre, W


def _qv(pre_v, beta, W):
    """Weighted beta-quantile of one sparse column (zeros are implicit)."""
    vals, cw, w0 = pre_v
    thr = beta * W - w0
    if thr <= 0:
        return 0.0
    lo, hi = 0, len(cw) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if cw[mid] >= thr:
            hi = mid
        else:
            lo = mid + 1
    return vals[lo]


def tv_barycentre(ps, ws, target_mass=1.0, iters=50):
    """argmin_q sum_i w_i TV(p_i, q) over the simplex of mass `target_mass`.

    KKT gives a COMMON quantile level across coordinates; the
    level is pinned by the mass constraint and the sum is monotone in beta, so
    bisection finds it. The minimiser is not always unique -- see the tie
    handling below, which is load-bearing rather than cosmetic.
    """
    if not ps:
        return {}
    pre, W = _prep(ps, ws)
    if W <= 0:
        return {}

    def qat(beta):
        return {v: _qv(pv, beta, W) for v, pv in pre.items()}

    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        if sum(_qv(pv, mid, W) for pv in pre.values()) < target_mass:
            lo = mid
        else:
            hi = mid

    # The per-coordinate quantile is a STEP function of beta, so sum(q(beta))
    # jumps and generically NEVER equals target_mass exactly -- with M
    # near-deterministic paths on distinct tokens it goes straight from 0 to M.
    # Bisecting alone therefore returns a sub- or super-probability vector, and
    # scoring TV against it silently yields ~0.5 everywhere, which is what the
    # first live run produced. At the critical level the minimiser is a SET;
    # interpolate across the jump to land on the member with the right mass.
    # Any such point attains the same objective, since the subgradient
    # condition holds throughout the tie.
    q_lo, q_hi = qat(lo), qat(hi)
    s_lo, s_hi = sum(q_lo.values()), sum(q_hi.values())
    if s_hi - s_lo < 1e-15:
        base = q_hi if abs(s_hi - target_mass) < abs(s_lo - target_mass) else q_lo
        tot = sum(base.values())
        return {v: x * target_mass / tot for v, x in base.items()} if tot > 0 else {}
    t = min(1.0, max(0.0, (target_mass - s_lo) / (s_hi - s_lo)))
    return {v: q_lo[v] + t * (q_hi[v] - q_lo[v]) for v in pre}


def wloss(ps, ws, q):
    r"""sum_i w_i TV(p_i, q) / sum_i w_i, iterating only over each p's support.

    TV(p,q) = 1/2 [ sum_{v in p} |p(v)-q(v)| + sum_{v in q\p} q(v) ]
            = 1/2 [ sum_{v in p} (|p(v)-q(v)| - q(v)) + mass(q) ]
    so the |q|-sized term is a constant and never has to be walked per path --
    which matters because q's support is the union over all M paths.
    """
    tot = sum(ws)
    if tot <= 0:
        return float("nan")
    mq = sum(q.values())
    acc = 0.0
    for p, w in zip(ps, ws):
        d = 0.0
        for v, x in p.items():
            qv = q.get(v, 0.0)
            d += abs(x - qv) - qv
        acc += w * 0.5 * (d + mq)
    return acc / tot


def ess(ws):
    s = sum(ws)
    return (s * s) / sum(w * w for w in ws) if s > 0 else 0.0


def floor_from(ps, ws, split=False, rng=None):
    """T for one cell, plus the split-sample variant if asked."""
    if len(ps) < 2:
        return None, None
    q = tv_barycentre(ps, ws)
    plain = wloss(ps, ws, q)
    if not split:
        return plain, None
    idx = list(range(len(ps)))
    (rng or random.Random(0)).shuffle(idx)
    a, b = idx[::2], idx[1::2]
    if len(a) < 2 or len(b) < 2:
        return plain, None
    qa = tv_barycentre([ps[i] for i in a], [ws[i] for i in a])
    return plain, wloss([ps[i] for i in b], [ws[i] for i in b], qa)


# ------------------------------------------------------------------ rows -----
def _dist(rows, k_top):
    """top-K (id, logprob) -> (renormalised dict, residual mass).

    Renormalising over the retained support keeps every p_i a proper
    distribution so TV and the barycentre are well defined; the residual is
    reported so the reader can see how much was dropped. Truncation lowers TV,
    hence lowers T, hence flatters the drafter -- so residual is a gate, not a
    footnote.
    """
    top = rows[:k_top]
    m = {int(t): math.exp(lp) for t, lp in top}
    s = sum(m.values())
    if s <= 0:
        return {}, 1.0
    return {t: v / s for t, v in m.items()}, max(0.0, 1.0 - s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=sorted(C.CORPORA))
    ap.add_argument("--corpus-file", required=True)
    ap.add_argument("--cheap", required=True, help="anchor source (a cheap/ladder file)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", default=C.TARGET)
    ap.add_argument("--anchors", type=int, default=64)
    ap.add_argument("--paths", type=int, default=256)
    ap.add_argument("--top-k", type=int, default=256)
    ap.add_argument("--rungs", default="0,1")
    ap.add_argument("--split", action="store_true",
                    help="also report the split-sample floor (winner's-curse size)")
    ap.add_argument("--mem-fraction", type=float, default=0.0)
    ap.add_argument("--context-length", type=int, default=0)
    ap.add_argument("--max-running", type=int, default=C.MAX_RUNNING_AT_HIGH_M)
    ap.add_argument("--scoring-mode", default="append")
    args = ap.parse_args()

    rungs = [int(x) for x in args.rungs.split(",") if x.strip()]
    policy = C.CORPORA[args.corpus]
    K = C.GAMMA

    seqs = {}
    with open(args.corpus_file) as fh:
        for line in fh:
            r = json.loads(line)
            seqs[r["prompt_id"]] = r

    anchors = []
    with open(args.cheap) as fh:
        for line in fh:
            a = json.loads(line)
            if a["prompt_id"] in seqs:
                anchors.append(a)
    rng = random.Random(C.SEED)
    rng.shuffle(anchors)
    anchors = anchors[: args.anchors]
    print(f"== {len(anchors)} anchors from {len({a['prompt_id'] for a in anchors})} "
          f"prompts; M={args.paths}; top-K={args.top_k}; rungs={rungs}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.target)
    stop_ids = resolve_stops(args.target, tok)
    max_ctx = max(seqs[a["prompt_id"]]["prompt_len"] + a["t"] + K
                  for a in anchors) + 8
    args.context_length = args.context_length or max_ctx

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0

    with BA.make_target(args, need_logprobs=True) as eng, out.open("w") as fout:
        for i, a in enumerate(anchors):
            seq = seqs[a["prompt_id"]]
            full = seq["prompt_ids"] + seq["response_ids"]
            cut = seq["prompt_len"] + a["t"]
            prefix, gt = full[:cut], full[cut: cut + K]
            if len(gt) < K:
                continue
            eng.warm_prefix(prefix)
            drawn, _ = sample_paths(eng, prefix, args.paths, policy, K, stop_ids,
                                    anchor_seed(a["prompt_id"], a["t"], C.SEED))
            if not drawn:
                continue

            # Effective support of the path distribution, per prefix length.
            # NOT the unique count: repeats still carry information about mass,
            # which is why ESS keeps improving with M after the unique count has
            # saturated. Simpson/collision form, because the collision
            # probability IS sum_z p(z)^2 and the plug-in sum (n_j/M)^2 is biased
            # up; sum n_j(n_j-1)/(M(M-1)) is the unbiased estimator of it.
            neff = {}
            for j in range(1, K):
                cnt = collections.Counter(tuple(p_[:j]) for p_ in drawn)
                Mn = sum(cnt.values())
                coll = sum(n * (n - 1) for n in cnt.values())
                neff[str(j)] = (Mn * (Mn - 1) / coll) if coll > 0 else float(Mn)
            row_neff = neff

            if PROF is not None:
                # Distinct j-token path prefixes among the M draws -- the raw
                # count behind the dedup ratio. Deliberately NOT interpreted as a
                # sample size: repeated draws still carry information about mass,
                # so a saturating unique count is consistent with ESS continuing
                # to improve in M. Entropy does not bound support size either --
                # a low-entropy law can have unbounded support with a vanishing
                # tail, and new paths keep appearing as M grows. n_eff above is
                # the quantity to read; this one is for the dedup accounting.
                PROF["_anchors"] += 1
                for j in range(1, K):
                    PROF[f"_distinct@{j}"] += len({tuple(p_[:j]) for p_ in drawn})

            row = {q: a[q] for q in ("prompt_id", "t", "context", "stratum", "pi")}
            row.update(corpus=args.corpus, M=len(drawn), top_k=args.top_k,
                       n_eff=row_neff, T={}, resid={}, ess={}, T_split={})

            # ---- rung 0: one bucket. Score prefix+path once; position P+k
            # carries p(Y_k | X, path[:k]) for EVERY k, so one pass does all slots.
            #
            # A path is K-1 tokens, so prefix+path scores positions P..P+K-2 and
            # stops one slot short: the LAST slot, which carries the most floor
            # mass, would be missing. One trailing token creates position P+K-1
            # without being conditioned on by any row read -- the distribution
            # there is conditioned on prefix+path only, so the token's identity
            # is irrelevant and gt[K-1] is used to match score_paths. Without it
            # rung 0 covers slots 0..K-2 while rung m>=1 covers m..K-1, and the
            # paired T^(0) - T^(m) is unavailable exactly at the deepest slot.
            if 0 in rungs:
                s0 = [list(prefix) + list(p) + [gt[K - 1]] for p in drawn]
                tops = eng.teacher_forced_topk(s0, [len(prefix)] * len(s0),
                                               args.top_k)
                for k in range(K):
                    ps, rs = [], []
                    for tp in tops:
                        if k >= len(tp):
                            continue
                        d, r = _dist(tp[k], args.top_k)
                        if d:
                            ps.append(d); rs.append(r)
                    ws = [1.0] * len(ps)
                    T, Ts = floor_from(ps, ws, args.split, rng)
                    row["T"].setdefault("0", {})[str(k)] = T
                    row["resid"].setdefault("0", {})[str(k)] = (
                        sum(rs) / len(rs) if rs else None)
                    row["ess"].setdefault("0", {})[str(k)] = float(len(ps))
                    if Ts is not None:
                        row["T_split"].setdefault("0", {})[str(k)] = Ts

            # ---- rung m>=1: force the realised Z_{k-m:k-1} onto every sampled
            # prefix and weight by w(s) = p(z* | X, s). SNIS, not grouping.
            # Every (m, k) cell is scored in ONE engine call rather than one per
            # cell. The slots are independent given the sampled paths, so there
            # is no reason to serialise them, and sglang batches 6*M sequences
            # sharing a prefix far better than it batches M six times: the
            # measured cost of this stage was round trips and logprob transfer,
            # not arithmetic (84% of wall clock at 0% accelerator utilisation).
            for m in (r for r in rungs if r >= 1):
                _t = time.perf_counter()
                sq, starts, tags = [], [], []
                for k in range(m, K):
                    if k >= len(gt):
                        continue          # no token to hang slot k's row on
                    for p in drawn:
                        # The trailing gt[k] is NOT scored as part of the weight
                        # and its identity does not enter any estimate. It is
                        # there to create the ROW that predicts slot k. The
                        # backend returns one row per token of the sequence and
                        # row i is the distribution that predicted token i, so
                        # without a token at position len(prefix)+k the last row
                        # is the one that predicted the last REVEALED token --
                        # p(.|X, s), the distribution the weight was read from,
                        # not p(.|X, s, z*). See probe_rm.ce_mixed_all, which
                        # has always built the sequence this way.
                        sq.append(list(prefix) + list(p[: k - m])
                                  + list(gt[k - m: k]) + [gt[k]])
                        starts.append(len(prefix) + (k - m))
                        tags.append(k)
                _t = _tick(PROF, "build_seqs", _t)
                if not sq:
                    continue
                # Deduplicate before scoring. Slot k's sequence is
                # prefix + p[:k-m] + gt[k-m:k], so two paths agreeing on their
                # first k-m tokens produce the SAME sequence and the same
                # logprobs; at k = m the free segment is empty and all M
                # sequences are identical. Scoring one representative and
                # scattering the result back is exact, and it keeps the
                # multiplicity in the estimator where it belongs -- duplicate
                # paths are duplicate draws from the posterior and must keep
                # their weight.
                uniq, first = [], {}
                idx = []
                for s_ in sq:
                    key = tuple(s_)
                    j = first.get(key)
                    if j is None:
                        j = first[key] = len(uniq)
                        uniq.append(s_)
                    idx.append(j)
                ustarts = [0] * len(uniq)
                for pos, j in enumerate(idx):
                    ustarts[j] = starts[pos]
                _t = _tick(PROF, "dedup", _t)
                unll, utops = eng.teacher_forced_nll_topk(uniq, ustarts, args.top_k)
                nll = [unll[j] for j in idx]
                tops = [utops[j] for j in idx]
                if PROF is not None:
                    PROF["_seqs_scored"] += len(uniq)
                    PROF["_seqs_requested"] += len(sq)
                _t = _tick(PROF, "GPU_nll_topk", _t)

                by = {}
                for k, v, tp in zip(tags, nll, tops):
                    if not tp:
                        continue
                    d, r = _dist(tp[-1], args.top_k)
                    if not d:
                        continue
                    # v[:m] are the m revealed tokens under THIS sampled
                    # prefix, so their sum is log w(s); v[m] is the trailing
                    # gt[k] and is deliberately unused.
                    b = by.setdefault(k, ([], [], []))
                    b[0].append(d); b[1].append(-sum(v[:m])); b[2].append(r)
                _t = _tick(PROF, "parse_topk", _t)

                for k in sorted(by):
                    ps, ws, rs = by[k]
                    if len(ps) < 2:
                        continue
                    mx = max(ws)
                    ws = [math.exp(x - mx) for x in ws]
                    T, Ts = floor_from(ps, ws, args.split, rng)
                    row["T"].setdefault(str(m), {})[str(k)] = T
                    row["resid"].setdefault(str(m), {})[str(k)] = sum(rs) / len(rs)
                    row["ess"].setdefault(str(m), {})[str(k)] = ess(ws)
                    if Ts is not None:
                        row["T_split"].setdefault(str(m), {})[str(k)] = Ts
                _t = _tick(PROF, "barycentre+wloss", _t)

            fout.write(json.dumps(row) + "\n")
            fout.flush()
            n_written += 1
            if (i + 1) % 8 == 0:
                print(f"  {i+1}/{len(anchors)}", flush=True)

    _report_profile(PROF)
    print(f"== T written to {out} ({n_written} anchors)", flush=True)
    print("   T is a REJECTION-LOSS floor: 1-alpha >= T for any drafter on that rung.",
          flush=True)
    print("   Truncation biases T DOWN (flatters the drafter); check resid.", flush=True)


if __name__ == "__main__":
    main()
