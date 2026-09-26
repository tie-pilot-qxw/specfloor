r"""How much of the floor exists only because a blind drafter must commit to ONE proposal.

$T_k^{(0)}$ is the least risk a single realisation-blind proposal can carry at slot $k$. Allow $K$
of them and an oracle that picks, per realised path, whichever fits best:

    T_k^{(0),K} = min_{q_1..q_K}  E_Z[ min_j TV(p_Z, q_j) ],

the **$K$-median in total variation**. At $K = 1$ it is exactly $T^{(0)}$, so the two are read off
the same rollouts and $T^{(0)} - T^{(0),K}$ is a within-anchor difference: the part of the floor that
is not "no proposal fits this path" but "the paths disagree about which proposal to use".

**This is NOT a tree drafter's acceptance ceiling.** Under speculative sampling a width-$K$ proposal
is accepted when the target's own draw lands in the candidate set -- a UNION event, whose blind
optimum over fixed sets is the top-$K$ mass of the mixture, i.e. Recall@K, computed elsewhere. The
`min_j` here sits INSIDE the expectation, which prices an oracle router: a system told which of its
$K$ proposals suits the path it is on. No drafter can do that. Reading this number as what a
verification tree buys is the one mistake this module exists to prevent.

**Which way the estimate errs.** The $K$-median is solved by Lloyd alternation: assign each path to
its nearest centre, replace each centre by the weighted TV barycentre of its cluster. Neither step
can raise the objective -- the barycentre is by construction the minimiser of its own cluster's
weighted TV loss -- so the iteration converges, to a LOCAL optimum. The value returned is therefore
an UPPER bound on the true $K$-median, and the removed fraction read off it is a LOWER bound on what
$K$ proposals could remove. A large removed fraction is a finding; a small one is equally consistent
with the solver having stopped early and establishes nothing. Restart spread is recorded beside every
cell as the only direct handle on that.

    python -m specfloor.probe_kmedian --corpus C0 --corpus-file <corpus> \
        --cheap <ladder> --out <out> --anchors 32 --paths 256 --widths 1,2,4
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import time

from transformers import AutoTokenizer

from specfloor import config as C
from specfloor import backend_api as BA
from specfloor.corpus import resolve_stops
from specfloor.probe_cheap import anchor_seed, sample_paths
from specfloor.probe_tk import _dist, tv_barycentre, wloss


def tv(p, q, mq=None):
    """TV(p, q) walking only p's support; mq = sum(q.values()) if precomputed.

    Same identity probe_tk.wloss uses: the mass of q outside p's support enters
    as a constant, so q's support -- which is the union over every path in the
    cluster -- never has to be walked per path.
    """
    if mq is None:
        mq = sum(q.values())
    d = 0.0
    for v, x in p.items():
        qv = q.get(v, 0.0)
        d += abs(x - qv) - qv
    return 0.5 * (d + mq)


def _assign(ps, centres):
    """Nearest centre per path, and the resulting mean-of-min objective terms."""
    mqs = [sum(q.values()) for q in centres]
    lab, dist = [], []
    for p in ps:
        best_j, best_d = 0, float("inf")
        for j, q in enumerate(centres):
            d = tv(p, q, mqs[j])
            if d < best_d:
                best_j, best_d = j, d
        lab.append(best_j)
        dist.append(best_d)
    return lab, dist


def _seed_centres(ps, ws, K, rng):
    """k-means++ in TV: first centre is the barycentre, the rest are far points.

    Starting from the barycentre rather than a random path means K=1 reproduces
    T^(0) exactly and larger K starts from the single-proposal solution, so the
    objective is monotone in K by construction rather than by luck.
    """
    centres = [tv_barycentre(ps, ws)]
    while len(centres) < K:
        _, d = _assign(ps, centres)
        tot = sum(wi * di * di for wi, di in zip(ws, d))
        if tot <= 0:                       # every path already exactly covered
            centres.append(dict(ps[rng.randrange(len(ps))]))
            continue
        r, acc = rng.random() * tot, 0.0
        pick = len(ps) - 1
        for i, (wi, di) in enumerate(zip(ws, d)):
            acc += wi * di * di
            if acc >= r:
                pick = i
                break
        centres.append(dict(ps[pick]))
    return centres


def _lloyd(ps, ws, K, iters, rng):
    """One Lloyd run. Returns (objective, centres)."""
    centres = _seed_centres(ps, ws, K, rng)
    prev = float("inf")
    for _ in range(iters):
        lab, d = _assign(ps, centres)
        obj = sum(wi * di for wi, di in zip(ws, d)) / sum(ws)
        # An empty cluster is re-seeded at the worst-covered path rather than
        # dropped: dropping it would silently report a K-1 solution as K.
        for j in range(K):
            mem = [i for i, l in enumerate(lab) if l == j]
            if mem:
                centres[j] = tv_barycentre([ps[i] for i in mem],
                                           [ws[i] for i in mem])
            else:
                centres[j] = dict(ps[max(range(len(ps)), key=lambda i: d[i])])
        if prev - obj < 1e-9:
            break
        prev = obj
    lab, d = _assign(ps, centres)
    return sum(wi * di for wi, di in zip(ws, d)) / sum(ws), centres


def kmedian(ps, ws, K, restarts=3, iters=12, seed=0):
    """min over K proposals of the mean nearest-centre TV. See the module docstring.

    Returns (best, spread) where spread is best-to-worst across restarts -- the
    only honest handle on how far the local optimum might be from the global one.
    """
    if len(ps) < max(2, K):
        return None, None
    if K == 1:
        q = tv_barycentre(ps, ws)
        return wloss(ps, ws, q), 0.0
    rng = random.Random(seed)
    vals = [_lloyd(ps, ws, K, iters, rng)[0] for _ in range(restarts)]
    return min(vals), max(vals) - min(vals)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=sorted(C.CORPORA))
    ap.add_argument("--corpus-file", required=True)
    ap.add_argument("--cheap", required=True, help="anchor source (a cheap/ladder file)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", default=C.TARGET)
    ap.add_argument("--anchors", type=int, default=32)
    ap.add_argument("--paths", type=int, default=256)
    ap.add_argument("--top-k", type=int, default=256)
    ap.add_argument("--widths", default="1,2,4",
                    help="K values; 1 must be present, it is the T^(0) control")
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--mem-fraction", type=float, default=0.0)
    ap.add_argument("--context-length", type=int, default=0)
    ap.add_argument("--max-running", type=int, default=C.MAX_RUNNING_AT_HIGH_M)
    ap.add_argument("--scoring-mode", default="append")
    args = ap.parse_args()

    widths = [int(x) for x in args.widths.split(",") if x.strip()]
    if 1 not in widths:
        raise SystemExit(
            "--widths must include 1. K=1 is the same estimator reduced to one "
            "centre, so it reproduces T^(0) on these paths and is what makes the "
            "gain a within-run difference rather than a comparison across runs.")
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
          f"prompts; M={args.paths}; widths={widths}; "
          f"{args.restarts} restarts x {args.iters} iters", flush=True)

    tok = AutoTokenizer.from_pretrained(args.target)
    stop_ids = resolve_stops(args.target, tok)
    max_ctx = max(seqs[a["prompt_id"]]["prompt_len"] + a["t"] + K
                  for a in anchors) + 8
    args.context_length = args.context_length or max_ctx

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    t0 = time.perf_counter()

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

            row = {q: a[q] for q in ("prompt_id", "t", "context", "stratum", "pi")}
            row.update(corpus=args.corpus, M=len(drawn), top_k=args.top_k,
                       widths=widths, Tb={}, spread={})

            # Same rung-0 construction as probe_tk: one trailing token so the
            # last slot is covered, then p(Y_k | X, path[:k]) is read at k.
            s0 = [list(prefix) + list(p) + [gt[K - 1]] for p in drawn]
            tops = eng.teacher_forced_topk(s0, [len(prefix)] * len(s0), args.top_k)
            for k in range(K):
                ps = []
                for tp in tops:
                    if k >= len(tp):
                        continue
                    d, _r = _dist(tp[k], args.top_k)
                    if d:
                        ps.append(d)
                ws = [1.0] * len(ps)
                for W in widths:
                    v, sp = kmedian(ps, ws, W, args.restarts, args.iters,
                                         seed=C.SEED + k)
                    row["Tb"].setdefault(str(W), {})[str(k)] = v
                    row["spread"].setdefault(str(W), {})[str(k)] = sp

            fout.write(json.dumps(row) + "\n")
            fout.flush()
            n_written += 1
            if (i + 1) % 4 == 0:
                last = row["Tb"]
                print(f"  {i + 1}/{len(anchors)}  {time.perf_counter() - t0:.0f}s  "
                      + "  ".join(f"K={W} T6={last[str(W)]['6']:.4f}"
                                  for W in widths if last[str(W)].get("6") is not None),
                      flush=True)

    print(f"\n== wrote {n_written} rows to {out}", flush=True)
    print("K=1 reproduces T^(0) on these paths, so T^(0) - T^(0),K is a within-run")
    print("difference. Lloyd returns a LOCAL optimum, so the removed fraction is a")
    print("LOWER bound. This prices an ORACLE ROUTER among K proposals -- it is not")
    print("a verification tree's acceptance, which is a union event.", flush=True)


if __name__ == "__main__":
    main()
