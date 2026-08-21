r"""Per-path scalars for the single-slot best-response headroom $\Delta\tau_k^{\mathrm{BR}}$.

**The question.** $G_k = R_k - T_k$ is a per-slot loss under $\mu$, and it says nothing about
whether slot $k$ is *worth* fixing: a slot the block rarely reaches, or one with little left after
it, cannot move the accepted length much however bad its proposal is. The serving-weighted version
holds every other slot at the real drafter and reopens one:

    q_k*  =  argmax_{q_k}  tau(q_0..q_{k-1}, q_k, q_{k+1}..q_{K-1}),
    dtau_k^BR  =  tau(q_{-k}, q_k*) - tau(q_base).

**Why one slot is exactly solvable.** With $a_i = \min(1, q_i(Z_i)/p_i(Z_i))$ and
$\tau - 1 = \mathbb E_\mu[\sum_j \prod_{i\le j} a_i]$ -- which is exact under verification, because a
slot that is reached was fed the target's own realisation (A.3) -- split the sum at $k$:

    tau - 1  =  (terms with no a_k)  +  E[ S_k * a_k * F_k ],
    S_k = prod_{i<k} a_i          the path's probability of REACHING slot k
    F_k = 1 + sum_{j>k} prod_{k<i<=j} a_i     what is still to come after it

Neither $S_k$ nor $F_k$ involves $q_k$, and $Z\sim\mu$ does not either, so with the other slots fixed
the whole $q_k$-dependence is $\mathbb E[c\,a_k]$ with $c = S_k F_k$ known per path. Inside one
information cell $w$ that is

    max_q  sum_{r: W_r = w} c_r * min(1, q(y_r) / p_r)      s.t.  sum_v q(v) = 1,

whose value in $q(v)$ is piecewise linear with slope $g_v(x) = \sum_{r: y_r = v,\ p_r > x} c_r/p_r$ --
non-increasing in $x$, since each path leaves the slope once $q(v)$ reaches its $p_r$. A sum of
concave coordinate functions on the simplex is maximised by pouring each next unit of mass into the
token of highest current slope, so **water filling is the exact optimum of this restricted problem**,
not an approximation to it.

**What this module records, and what it deliberately does not do.** Everything above needs only two
scalars per path per slot -- the target's probability at the realised token, and the drafter's -- so
this probe stores those and nothing else. The full $[M, V]$ simplex that forces `probe_rpre` down to
$M = 256$ is never materialised, which is why this runs at $M = 1024$: the objective is weighted by
$c = S_k F_k$, survival concentrates (§5.8 measures the concentration), and the effective sample size
behind the fit is far below $M$ at deep slots.

The water filling, the cross-fitting and every choice of split live in `br_report`, so a different
split or a different smoothing never costs a GPU pass.

**Fitting and scoring on the same paths would be meaningless**, and more so here than for the floor:
the empirical optimiser sees a rare token carried by one high-$c$ path and pours mass on it, and the
support of the fitted $q$ is confined to tokens the fit half happened to realise. `br_report`
cross-fits for exactly this reason and reports the train-minus-held-out gap as a result in its own
right.

    python -m specfloor.probe_br --corpus C0 --corpus-file <corpus> --cheap <ladder> \
        --drafter <ckpt> --order 0 --out <out> --anchors 96 --paths 1024
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from specfloor import config as C
from specfloor.corpus import resolve_stops
from specfloor.probe_cheap import anchor_seed
from specfloor.probe_rpre import (
    barycentre,
    chain_logits,
    drafter_logits,
    kv_bytes_per_token,
    load_drafter,
    text_cfg,
    warp,
)

from specfloor import _deepspec


@torch.no_grad()
def rollout_scalars(target, draft, dcfg, base_or_q, prefix_ids, M, K, policy,
                    stop_ids, gen, kv_budget_bytes, order, with_tv=False):
    """M free rollouts, recording four scalars per path per slot.

    Returns, per slot k, four aligned 1-D tensors over the paths ALIVE at k:
        gid   global path index, so a running product can be formed along a path
        tok   the token the target realised at slot k
        p     the target's probability of that token
        q     the drafter's probability of that token
    plus the conditioning token per path (the realisation at k-1), which is the
    information cell an order-1 proposal is allowed to depend on.

    The [alive, V] probability rows are consumed and dropped inside the loop
    rather than accumulated, which is the whole reason this can run at M = 1024.
    """
    dev = prefix_ids.device
    S = prefix_ids.shape[1]
    per_tok = kv_bytes_per_token(target.config)
    chunk = max(1, min(M, int(kv_budget_bytes // max(1, per_tok * (S + K)))))
    stop = torch.tensor(sorted(stop_ids), device=dev, dtype=torch.long)

    gids = [[] for _ in range(K)]
    toks = [[] for _ in range(K)]
    ps = [[] for _ in range(K)]
    qs = [[] for _ in range(K)]
    cds = [[] for _ in range(K)]
    # with_tv keeps the [alive, V] rows so the TV barycentre -- the proposal
    # T^(0) selects -- can be scored at the realised tokens alongside the
    # drafter's. It costs M*V*K floats and is off by default for that reason.
    rows_P = [[] for _ in range(K)] if with_tv else None
    anchor = prefix_ids[0, S - 1]
    done = 0
    while done < M:
        c = min(chunk, M - done)
        try:
            out = target(input_ids=prefix_ids, use_cache=True, logits_to_keep=1)
        except TypeError:
            out = target(input_ids=prefix_ids, use_cache=True)
        cache = out.past_key_values
        cache.batch_repeat_interleave(c)
        logits = out.logits[:, -1, :].float().expand(c, -1)
        del out

        alive = torch.ones(c, dtype=torch.bool, device=dev)
        prev = anchor.expand(c)
        gid = torch.arange(done, done + c, device=dev)
        for k in range(K):
            p = warp(logits, policy)
            am = alive.clone()
            nxt = torch.multinomial(p, 1, generator=gen)
            if am.any():
                real = nxt[am, 0]
                rows = torch.arange(int(am.sum()), device=dev)
                p_at = p[am][rows, real]
                # The drafter's proposal at this slot, warped the way serving
                # warps it. Order 0 is one row for every path; order 1 is one
                # row per conditioning token, so it is gathered per path.
                if order == 0:
                    q_at = base_or_q[k][real]
                else:
                    Q = warp(chain_logits(draft, base_or_q, k, prev[am]), policy)
                    q_at = Q[rows, real]
                    del Q
                if rows_P is not None:
                    rows_P[k].append(p[am].clone())
                gids[k].append(gid[am].clone())
                toks[k].append(real.clone())
                ps[k].append(p_at.clone().float())
                qs[k].append(q_at.clone().float())
                cds[k].append(prev[am].clone())
            if stop.numel():
                alive = alive & ~torch.isin(nxt[:, 0], stop)
            del p
            if k == K - 1:
                break
            prev = nxt[:, 0]
            pos = torch.tensor([S + k], device=dev, dtype=torch.long)
            logits = target(input_ids=nxt, past_key_values=cache,
                            cache_position=pos, use_cache=True).logits[:, -1, :].float()
        del cache, logits
        done += c

    cat = lambda ls, dt: (torch.cat(ls, 0) if ls
                          else torch.zeros((0,), device=dev, dtype=dt))
    qtv = qtv_xf = None
    if rows_P is not None:
        # Both are recorded, and the pair matters. qtv is the barycentre of ALL
        # M paths scored at their own realised tokens -- in-sample, the same way
        # T^(0) itself is reported. qtv_xf fits the barycentre on one half and
        # scores it on the other, using the SAME index split br_report uses for
        # the water filling, so the serving-optimal and TV-optimal proposals are
        # compared on equal terms. Comparing a cross-fitted BR against an
        # in-sample TV would flatter the TV side by exactly its own fit-score bias.
        qtv, qtv_xf = [], []
        halves = (lambda g: g < (M // 2), lambda g: g >= (M // 2))
        for k in range(K):
            if not rows_P[k]:
                z = torch.zeros((0,), device=dev, dtype=torch.float32)
                qtv.append(z); qtv_xf.append(z.clone())
                continue
            P = torch.cat(rows_P[k], 0)
            rows_P[k] = None
            t = torch.cat(toks[k], 0)
            g = torch.cat(gids[k], 0)
            b_all = barycentre(P)
            qtv.append(b_all[t].float().clone())
            del b_all
            xf = torch.zeros_like(t, dtype=torch.float32)
            for sel in halves:
                fit_m = sel(g)
                sco_m = ~fit_m
                if int(fit_m.sum()) < 2 or int(sco_m.sum()) == 0:
                    continue
                b = barycentre(P[fit_m])
                xf[sco_m] = b[t[sco_m]].float()
                del b
            qtv_xf.append(xf)
            del P
    return ([cat(x, torch.long) for x in gids],
            [cat(x, torch.long) for x in toks],
            [cat(x, torch.float32) for x in ps],
            [cat(x, torch.float32) for x in qs],
            [cat(x, torch.long) for x in cds], qtv, qtv_xf, chunk)


# The decorator is load-bearing, not hygiene. Every helper below carries its
# own, but the hidden-state pass in the loop is called bare, and a 48-layer
# target at 3k context retains ~16 MiB per token of autograd graph -- 46 GiB
# on one anchor, which is what cost the scale run its long-context domain.
@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="C0", choices=sorted(C.CORPORA))
    ap.add_argument("--corpus-file", required=True)
    ap.add_argument("--cheap", required=True, help="anchor source, same file probe_tk used")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", default=C.TARGET)
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--order", type=int, default=0, choices=(0, 1))
    ap.add_argument("--anchors", type=int, default=96)
    ap.add_argument("--paths", type=int, default=1024)
    ap.add_argument("--kv-budget-gib", type=float, default=12.0)
    ap.add_argument("--with-tv", action="store_true",
                    help="also score the TV barycentre -- the proposal T^(0) "
                         "selects -- at each realised token, so the serving-optimal "
                         "and TV-optimal proposals can be compared under the SAME "
                         "information set. Holds the [M, V] rows, so lower --paths.")
    args = ap.parse_args()

    policy = C.CORPORA[args.corpus]
    K = C.GAMMA
    device = "cuda"

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
          f"prompts; M={args.paths}; K={K}; order={args.order}; scalars only",
          flush=True)

    tok = AutoTokenizer.from_pretrained(args.target)
    stop_ids = resolve_stops(args.target, tok)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    draft, dcfg = load_drafter(args.drafter, device, order=args.order)
    if int(dcfg.block_size) != K:
        raise SystemExit(f"drafter block_size={dcfg.block_size} but GAMMA={K}")
    taps = list(dcfg.target_layer_ids)
    print(f"== drafter {args.drafter}: block={dcfg.block_size} "
          f"markov_rank={dcfg.markov_rank} order={args.order}", flush=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = n_oom = 0
    t0 = time.perf_counter()
    with out.open("w") as fout:
        for i, a in enumerate(anchors):
            seq = seqs[a["prompt_id"]]
            full = seq["prompt_ids"] + seq["response_ids"]
            cut = seq["prompt_len"] + a["t"]
            if len(full) < cut + K:
                continue
            prefix = torch.tensor([full[:cut]], device=device, dtype=torch.long)
            try:
                th = target(input_ids=prefix, output_hidden_states=True,
                            use_cache=False, logits_to_keep=1)
                thid = _deepspec.extract_context_feature(th.hidden_states, taps).to(torch.bfloat16)
                del th
                dl = drafter_logits(draft, dcfg, thid, prefix, K, device)
                del thid
                base_or_q = warp(dl, policy) if args.order == 0 else dl

                seed = anchor_seed(a["prompt_id"], a["t"], C.SEED) % (2 ** 63 - 1)
                gen = torch.Generator(device=device)
                gen.manual_seed(seed)
                gids, toks, ps, qs, cds, qtv, qtv_xf, chunk = rollout_scalars(
                    target, draft, dcfg, base_or_q, prefix, args.paths, K, policy,
                    stop_ids, gen, int(args.kv_budget_gib * (1 << 30)), args.order,
                    with_tv=args.with_tv)
                del base_or_q, dl
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                n_oom += 1
                print(f"   !! OOM at ctx={cut}, anchor skipped ({n_oom} so far)", flush=True)
                continue

            row = {k_: a[k_] for k_ in ("prompt_id", "t", "context", "stratum", "pi")}
            row.update(corpus=args.corpus, M=args.paths, K=K, chunk=chunk,
                       order=args.order, slots=[])
            for k in range(K):
                row["slots"].append(dict(
                    gid=gids[k].tolist(),
                    tok=toks[k].tolist(),
                    p=[round(float(x), 7) for x in ps[k].tolist()],
                    q=[round(float(x), 7) for x in qs[k].tolist()],
                    cond=cds[k].tolist(),
                    **({"qtv": [round(float(x), 7) for x in qtv[k].tolist()],
                        "qtv_xf": [round(float(x), 7) for x in qtv_xf[k].tolist()]}
                       if qtv is not None else {}),
                ))
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            n += 1
            if (i + 1) % 8 == 0:
                print(f"  {i + 1}/{len(anchors)}  {time.perf_counter() - t0:.0f}s  "
                      f"alive@6={len(row['slots'][K - 1]['gid'])}/{args.paths}",
                      flush=True)

    print(f"\n== wrote {n} rows to {out}"
          + (f"; {n_oom} anchors skipped on OOM" if n_oom else ""), flush=True)
    print("Scalars only: (realised token, target p, drafter q) per path per slot.")
    print("The water filling and the cross-fit are in br_report, so re-splitting")
    print("or re-smoothing never costs another GPU pass.", flush=True)


if __name__ == "__main__":
    main()
