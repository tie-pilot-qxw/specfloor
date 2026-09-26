"""R_pre: what the real DFlash achieves, against the floor no rung-0 model can beat.

    R_k = E_Z[ TV(p(.|X,Z_<k), q_k(.|X)) ]        the drafter's actual rejection loss
    T_k = min_q E_Z[ TV(p(.|X,Z_<k), q) ]        the best any rung-0 drafter could do
    G_k = R_k - T_k  >= 0                         all that more parallel capacity can buy

DFlash (`markov_rank: 0`, no head weights at all) is genuinely rung 0: its block
is all-MASK apart from slot 0's anchor token, so q_k depends on X alone and NOT
on any token the drafter itself sampled. That matters more than it sounds. For
DSpark, R - T conflates the head's quality with exposure -- the head conditions
on its OWN sampled token while T conditions on the target's realisation -- so
G_post is not interpretable without a separate oracle-conditioned run. **At rung
0 there is nothing to be exposed to.** G_pre is therefore the one gap in the
whole ladder that is clean by construction.

Three things this probe does differently from probe_tk, all of which make the
comparison tighter rather than merely cheaper:

* PAIRED. T and R are computed from the SAME sampled paths and the same p_Z, so
  G is a within-anchor difference and its bootstrap CI is far narrower than the
  difference of two independently-estimated quantities would be.

* FULL VOCABULARY, no truncation. probe_tk reads top-K logprobs over an API and
  therefore carries a truncation residual. Here the target is local, so p_Z is
  the exact 151936-way distribution and TV is exact. The top-K gate does not
  apply to any number this probe produces.

* EXACT BARYCENTRE, no bisection. With uniform path weights the common-level
  quantile can only sit at an order statistic, so sorting p_Z once along the
  path axis and walking the running coordinate-sums to the crossing gives the
  minimiser directly. The level is pinned by sum_v q(v) = 1, and the crossing is
  guaranteed: the coordinate-wise min sums to <= 1 and the coordinate-wise max
  sums to >= 1.

The independent estimate of T is the point of the last two, not a side effect --
it is a cross-backend and cross-truncation check on the headline T^(0) from
probe_tk, which came from a different engine with a top-256 read.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import pathlib
import random

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from specfloor import config as C
from specfloor.corpus import resolve_stops
from specfloor.probe_cheap import anchor_seed

from specfloor import _deepspec


# --------------------------------------------------------------- estimator ---
def barycentre(P):
    """argmin_q (1/M) sum_i TV(p_i, q) over the probability simplex. P: [M, V].

    KKT gives a COMMON quantile level across coordinates. With uniform weights
    the beta-quantile of column v is an order statistic, so as beta sweeps [0,1]
    the candidate q's are exactly the sorted rows S[0] <= ... <= S[M-1] (sorted
    independently per coordinate). Their coordinate-sums are non-decreasing, so
    the mass constraint picks out one crossing, which is found directly.

    The crossing is generically strict -- sum_v S[j] jumps over 1 rather than
    landing on it -- and at the critical level the minimiser is a SET. Taking
    either bracketing row returns a sub- or super-probability vector, and TV
    against a sub-probability vector is a silent ~0.5 for every cell. Interpolate
    to the member with unit mass; every point of the segment attains the same
    objective because the subgradient condition holds throughout the tie.
    """
    M = P.shape[0]
    S, _ = torch.sort(P, dim=0)
    sums = S.sum(dim=1)                                   # non-decreasing, [M]
    j = int(torch.searchsorted(sums, torch.ones(1, device=P.device, dtype=sums.dtype)).item())
    if j >= M:                                            # numerically flat top
        return S[M - 1] / sums[M - 1]
    lo_v = S[j - 1] if j > 0 else torch.zeros_like(S[0])
    lo_s = sums[j - 1] if j > 0 else torch.zeros((), device=P.device, dtype=sums.dtype)
    span = sums[j] - lo_s
    if span <= 0:
        return S[j] / sums[j].clamp(min=1e-30)
    t = ((1.0 - lo_s) / span).clamp(0.0, 1.0)
    return lo_v + t * (S[j] - lo_v)


def tv_rows(P, q):
    """Per-path TV(p_i, q), exact over the full vocabulary. q is [V] or [M, V]."""
    return 0.5 * (P - (q if q.dim() == 2 else q.unsqueeze(0))).abs().sum(dim=1)


def mean_tv(P, q):
    """(1/M) sum_i TV(p_i, q), exact over the full vocabulary."""
    return float(tv_rows(P, q).mean().item())


def mean_tv_rows(P, Q):
    """(1/M) sum_i TV(p_i, q_i) -- one proposal per path, not one shared q.

    An order-1 proposal is a different distribution on every path because the
    conditioning token differs, so the rejection loss is a mean of matched
    pairs, not a distance to a single point.
    """
    return float(tv_rows(P, Q).mean().item())


def accept_factor(P, q, real):
    r"""P(this slot is ACCEPTED | the target realised `real` here), per path.

    Speculative sampling draws a ~ q and keeps it with probability
    min(1, p(a)/q(a)); the emitted token is distributed exactly as p, and

        P(accept | Y = v) = min(p(v), q(v)) / p(v) = min(1, q(v)/p(v)).

    Conditional on the whole path the accept coins are independent across
    slots, so the product of these along a path is that path's probability of
    surviving -- which is what turns a free rollout into the serving
    population. Every ingredient is already in hand: p_Z per path per slot from
    the rollout, q from the drafter, and the realised token from the sampler.
    """
    idx = torch.arange(P.shape[0], device=P.device)
    p_real = P[idx, real].clamp(min=1e-30)
    q_real = (q[idx, real] if q.dim() == 2 else q[real])
    return (q_real / p_real).clamp(max=1.0)


def cond_floor(P, cond):
    r"""$T^{(1)}$ from the same paths: the floor for order-1 proposals.

        T_k^{(1)} = E_{z}[ min_q E_{Z | Z_{k-1}=z}[ TV(p_Z, q) ] ]

    so the minimisation is run once per value of the conditioning token and the
    results are averaged with that token's own frequency. Two estimators,
    because they fail in opposite directions and the honest number is bracketed
    by them:

    PLUG-IN takes the barycentre of a group and scores the same group. A group
    of size 1 returns exactly 0, so the plug-in is biased DOWN wherever the
    conditioning token is nearly unique -- which at deep slots is most of them.

    SPLIT-HALF fits the barycentre on half the group and scores the other half.
    It is not biased down by singletons because singletons cannot enter it at
    all, but it therefore reports a floor for the sub-population living in
    groups of size >= 2, and `cov` records exactly how much of the path mass
    that is. A low `cov` does not invalidate the split-half number; it says the
    number describes a smaller set of paths than the plug-in does.
    """
    M = P.shape[0]
    vals, inv = torch.unique(cond, return_inverse=True)
    tot_plug, cov_n, tot_split, ngroups2 = 0.0, 0, 0.0, 0
    for g in range(vals.numel()):
        idx = (inv == g).nonzero(as_tuple=True)[0]
        n = idx.numel()
        Pg = P[idx]
        tot_plug += n * mean_tv(Pg, barycentre(Pg))
        if n >= 2:
            h1, h2 = idx[0::2], idx[1::2]
            tot_split += n * mean_tv(P[h2], barycentre(P[h1]))
            cov_n += n
            ngroups2 += 1
    return {
        "plug": tot_plug / M,
        "split": (tot_split / cov_n) if cov_n else None,
        "cov": cov_n / M,
        "groups": int(vals.numel()),
        "groups2": ngroups2,
    }


def recall_floors(P, q, topk=(1, 8, 16)):
    r"""The OTHER rung-0 floor: the one exact-match / Recall@K lives in.

    T^(0) bounds acceptance under speculative sampling, and its minimiser is a
    common-level quantile. Recall is a different functional of the same family
    and has a different minimiser, so the two must not be interchanged -- see
    (4.6). For a candidate set A chosen from X alone,

        P(Y_k in A) = E_Z[ sum_{v in A} p_Z(v) ] = sum_{v in A} pbar(v),
        pbar := E_Z[p_Z]   (the MIXTURE, i.e. the log-loss barycentre)

    so the best any commit-blind drafter can do at Recall@K is the mixture's
    top-K mass. That is a ceiling with no oracle in it: an oracle told the
    correct token and then asked to find it in a candidate set is measuring
    something strictly larger, and the difference is not deployable headroom.

    Two target conventions, because they differ and the literature uses both.
    SAMPLED: Y_k ~ p_Z, so the hit probability of guess a is pbar(a). GREEDY:
    Y_k = argmax p_Z, so the relevant object is the distribution of the target's
    own argmax over paths, not the mixture.
    """
    out = {}
    M = P.shape[0]
    pbar = P.mean(dim=0)
    modes = P.argmax(dim=1)
    mdist = torch.zeros_like(pbar).index_add_(
        0, modes, torch.ones(M, device=P.device, dtype=P.dtype)) / M
    qorder = torch.argsort(q, descending=True)
    for K in topk:
        K = min(K, pbar.numel())
        out[f"ceil_sampled@{K}"] = float(torch.topk(pbar, K).values.sum().item())
        out[f"ceil_greedy@{K}"] = float(torch.topk(mdist, K).values.sum().item())
        sel = qorder[:K]
        out[f"dflash_sampled@{K}"] = float(pbar[sel].sum().item())
        out[f"dflash_greedy@{K}"] = float(mdist[sel].sum().item())
    return out


# ------------------------------------------------------------------ models ---


def load_drafter(path, device, dtype=torch.bfloat16, order=0):
    dcfg = AutoConfig.from_pretrained(path)
    mt = str(getattr(dcfg, "model_type", "") or "")
    cls = _deepspec.drafter_class(mt)
    if cls is None:
        raise SystemExit(
            f"{path} has model_type={mt!r}; this probe knows "
            f"{_deepspec.drafter_families()}. Guessing the draft class would silently "
            "read the wrong backbone, so it refuses instead.")
    dcfg.architectures = [cls.__name__]
    dcfg._attn_implementation = "flex_attention"
    rank = int(getattr(dcfg, "markov_rank", 0))
    if order == 0 and rank != 0:
        raise SystemExit(
            f"{path} has markov_rank={rank}; --order 0 expects a product-measure "
            "drafter. Pass --order 1 to measure a chain head, which requires "
            "choosing --cond oracle or --cond self.")
    if order >= 1 and rank == 0:
        raise SystemExit(
            f"{path} has markov_rank=0 and no chain to condition; --order 1 needs "
            "a drafter with a markov head.")
    if bool(getattr(dcfg, "causal_draft_mask", False)):
        raise SystemExit(f"{path} sets causal_draft_mask; expected the bidirectional "
                         "serving mask for a parallel block drafter.")
    m = cls.from_pretrained(path, config=dcfg, dtype=dtype).to(device).eval()
    return m, dcfg


@torch.no_grad()
def drafter_logits(draft, dcfg, target_hidden, prefix_ids, K, device):
    """DFlash's block logits for slots 0..K-1 given the prefix. [K, V].

    The block input is slot 0 = the anchor (last prefix) token, slots 1..K-1 =
    MASK, which is exactly the all-parallel serving regime. Attention is the
    default DSpark mask: bidirectional within the block, context restricted to
    kv < anchor_pos.
    """
    S = prefix_ids.shape[1]
    ap = torch.tensor([[S - 1]], device=device, dtype=torch.long)
    keep = torch.ones((1, 1), dtype=torch.bool, device=device)

    noise_ids = torch.full((1, K), int(dcfg.mask_token_id), dtype=torch.long, device=device)
    noise_ids[0, 0] = prefix_ids[0, S - 1]
    emb = draft.embed_tokens(noise_ids)

    pos = torch.cat([torch.arange(S, device=device).unsqueeze(0),
                     _deepspec.create_position_ids(ap, K)], dim=1)
    mask = _deepspec.create_dspark_attention_mask(anchor_positions=ap, block_keep_mask=keep,
                                        seq_len=S, block_size=K, device=device)
    h = draft._forward_backbone(position_ids=pos, noise_embedding=emb,
                                target_hidden_states=target_hidden, attention_mask=mask)
    # h [1, K, H] comes back with the logits because an ATTENTION head needs it: its bias
    # is a function of (previous token, THIS SLOT'S drafter hidden, committed prefix), not
    # of the previous token alone. A table head ignores it and its numbers are unchanged.
    return draft.compute_logits(h)[0].float(), h


def head_context(draft, target_hidden, prefix_ids, K):
    """The prefix K/V an attention markov head attends over, or None for a table head.

    WHY THIS EXISTS.  `chain_logits` passes `hidden_states=None` and no context, which is
    exact for the official vanilla head -- `compute_step_bias` discards both.  An AttnHead
    asserts on either being absent, deliberately: a head that silently read zeros instead
    of the committed prefix would not fail, it would score as a worse model, and the probe
    would report that as a model gap.  So the probe feeds it what training feeds it, or it
    does not run.

    Two simplifications hold HERE and are ASSERTED, not assumed:
      * one block, so there is nothing for a mask to separate.  The drafter's rule is
        `kv_idx < anchor_pos` with no window, so slicing the context to `[:anchor_pos]`
        and passing `context_mask=None` is the same visibility rather than an
        approximation -- `tests/test_attn_context.py` checks that numerically.
      * no per-block anchor K/V column (`markov_anchor_kv=False`).  With the column built,
        the mask would need the block-identity strip that `context_mask=None` cannot
        express.
    The fusion is recomputed exactly as the model's own forward does it,
    `hidden_norm(fc(.))`, for the reason stated there: the head must read what the
    drafter's own layers read, not a second summary that can drift from it.
    """
    head = getattr(draft, "markov_head", None)
    if head is None or getattr(head, "markov_head_type", "") != "attn":
        return None
    assert not head.use_anchor_kv, (
        "this drafter builds a per-block anchor K/V column; the single-block context "
        "below passes context_mask=None, which cannot express the anchor strip")
    assert getattr(draft, "context_window", None) is None, (
        f"context_window={draft.context_window} restricts the head's view; the slice "
        f"below implements the unwindowed rule only")
    anchor_pos = int(prefix_ids.shape[1]) - 1
    fused = draft.hidden_norm(draft.fc(target_hidden))[:, :anchor_pos]
    return head.build_context(
        target_hidden=fused,
        anchor_token_ids=prefix_ids[:, anchor_pos:anchor_pos + 1],
        context_mask=None, block_size=K)


def expand_context(ctx, b):
    """Same prefix, b query rows.  `expand` shares storage, so this costs nothing."""
    if ctx is None:
        return None
    assert ctx.anchor_key is None and ctx.context_mask is None, (
        "expanding a context that carries an anchor column or a mask would have to "
        "expand those too; head_context builds neither")
    return _deepspec.AttnHeadContext(
        key_states=ctx.key_states.expand(b, *ctx.key_states.shape[1:]),
        value_states=ctx.value_states.expand(b, *ctx.value_states.shape[1:]),
        anchor_key=None, anchor_value=None,
        context_mask=None, num_blocks=ctx.num_blocks,
    )


@torch.no_grad()
def chain_logits(draft, base_logits, k, prev_ids, dhid=None, ctx=None):
    r"""An order-1 head's slot-k logits, one row per conditioning token.

    The markov head is an additive bias on the SAME realisation-blind base
    logits -- `apply_step_logits(base_logits[k], token_ids=prev)` -- so
    q_k(. | X, z) is a function of z that the head defines for every z. Which z
    to evaluate it at is the whole question:

      --cond oracle   z = Z_{k-1}, the target's realised token on that path.
                      This is the SERVING-relevant reading, not a charitable
                      one: a block is verified left to right and acceptance
                      emits the drafter's own token, so surviving to slot k
                      implies the drafter's own a_{k-1} equalled Z_{k-1}. It is
                      also the only reading comparable to T^(1), whose inner
                      minimisation is over q(. | z) with z the realisation.

      --cond self     z = the head's own sample at k-1, which is the free-rollout
                      reading. R_self - R_oracle is exposure and is reported on
                      its own; it is never folded into R - T.

    `hidden_states=None` is exact for the official vanilla head, whose
    `compute_step_bias` discards that argument -- the bias is a pure function of
    the conditioning token.
    """
    B = prev_ids.shape[0]
    base = base_logits[k].unsqueeze(0).expand(B, -1)
    if ctx is None:
        return draft.markov_head.apply_step_logits(
            base, token_ids=prev_ids.long(), hidden_states=None).float()
    # Only the conditioning token varies across the B rows: this slot's drafter hidden
    # and the prefix are properties of the anchor, so both are broadcast.
    return draft.markov_head.apply_step_logits(
        base, token_ids=prev_ids.long(),
        hidden_states=dhid[0, k].unsqueeze(0).expand(B, -1),
        context=expand_context(ctx, B)).float()


@torch.no_grad()
def chain_rollout(draft, base_logits, K, M, policy, gen, anchor_tok, dhid=None, ctx=None):
    """M free samples of the drafter's OWN chain. Returns [M, K] token ids.

    This is the block DSpark would actually emit if nothing were verified:
    a_0 ~ q(.|X, anchor), a_k ~ q(.|X, a_{k-1}). Its only use here is the
    exposure diagnostic -- under verification a surviving a_{k-1} equals the
    target's realisation, which is what --cond oracle evaluates.
    """
    prev = torch.full((M,), int(anchor_tok), device=base_logits.device, dtype=torch.long)
    toks = torch.empty((M, K), device=base_logits.device, dtype=torch.long)
    for k in range(K):
        q = warp(chain_logits(draft, base_logits, k, prev, dhid, ctx), policy)
        prev = torch.multinomial(q, 1, generator=gen)[:, 0]
        toks[:, k] = prev
        del q
    return toks


def warp(logits, policy):
    r"""The serving law: temperature, then top-k, then top-p, then renormalise.

    THIS IS THE ONE PLACE WHERE mu AND p_verify HAVE TO BE THE SAME OBJECT, and
    the reason the C0/C1 distinction is not cosmetic. Two different
    distributions live behind the symbol "p":

      mu(. | X, z_<k)   the TRAJECTORY law -- what actually generates Z, and
                        therefore what the outer expectation E_Z averages over;
      p_verify          the distribution the rejection test compares against.

    On C0 (T=1, untruncated) they are both the raw softmax and the distinction
    is invisible. On C1 they are both the WARPED distribution -- serving applies
    temperature/top-p/top-k to the target before verifying -- so they coincide
    again, but they coincide at the warped law, not the raw one. Sampling a
    trajectory at C1 and then scoring TV on raw logits mixes the two and is
    simply a different quantity from either. A top-256 residual is meaningful
    for the raw law and meaningless after top-k=20, which is how that mistake
    shows up in a table.

    Order matters and follows the serving stack: temperature scaling first, then
    top-k, then top-p over what top-k left. top-p keeps the first token that
    crosses the threshold (mass strictly below the cut is what is compared), so
    the retained set always carries at least top_p.
    """
    t = float(policy.get("temp", 1.0))
    if t != 1.0:
        logits = logits / t
    p = logits.softmax(-1)
    k = int(policy.get("top_k", 0) or 0)
    if 0 < k < p.shape[-1]:
        v, i = torch.topk(p, k, dim=-1)
        p = torch.zeros_like(p).scatter_(-1, i, v)
    tp = float(policy.get("top_p", 1.0))
    if 0.0 < tp < 1.0:
        s, idx = torch.sort(p, dim=-1, descending=True)
        keep = (s.cumsum(-1) - s) < tp
        p = torch.zeros_like(p).scatter_(-1, idx, s * keep)
    return p / p.sum(-1, keepdim=True).clamp(min=1e-30)


def text_cfg(cfg):
    """The text tower's config, for targets whose top-level config is a wrapper.

    Gemma4's released config is Gemma4UnifiedConfig, a multimodal container that
    carries no hidden_size or vocab_size of its own -- those live one level down
    in .text_config. Reading through the wrapper is not optional: it is not a
    missing-attribute crash everywhere, so an unwrapped read can also silently
    return the wrong number.
    """
    return getattr(cfg, "text_config", None) or cfg


def kv_bytes_per_token(cfg) -> int:
    """bf16 K and V for every layer of the target."""
    cfg = text_cfg(cfg)
    hd = int(getattr(cfg, "head_dim", 0) or
             cfg.hidden_size // cfg.num_attention_heads)
    return int(cfg.num_hidden_layers) * int(cfg.num_key_value_heads) * hd * 2 * 2


def _prefill(target, prefix_ids, c):
    """Prefix cache expanded to batch c, plus the batch-c last-position logits.

    The expansion is why this has to be chunked at all: a shared prefix is
    physically copied c times, so a 2048-token prefix at c=256 is ~75 GiB of KV
    on its own. `logits_to_keep=1` stops the prefill from also materialising a
    [1, S, 151936] logit tensor, which is another gigabyte at long context.
    """
    try:
        out = target(input_ids=prefix_ids, use_cache=True, logits_to_keep=1)
    except TypeError:                                    # older signature
        out = target(input_ids=prefix_ids, use_cache=True)
    cache = out.past_key_values
    cache.batch_repeat_interleave(c)
    return cache, out.logits[:, -1, :].float().expand(c, -1)


@torch.no_grad()
def rollout(target, prefix_ids, M, K, policy, stop_ids, gen, kv_budget_bytes):
    """M free rollouts sharing one prefill. Per slot, all row-aligned:

      parts[k]  [alive, V]  target probs p(.|X, Z_<k)
      conds[k]  [alive]     the realisation Z_{k-1} an order-1 head conditions on
      reals[k]  [alive]     the realisation Z_k here, which the accept test sees
      idxs[k]   [alive]     the path's global index, so a per-slot factor can be
                            multiplied ALONG a path even though rows drop out

    p at step k is read from the SAME forward that samples token k, so no
    teacher-forced rescoring pass is needed and p is exactly the distribution
    the token was drawn from.

    Paths run in chunks sized so the expanded prefix cache fits a FIXED budget
    rather than whatever happens to be free -- a budget that moved with the
    card's other tenants would silently change the chunking, and the chunking
    changes how the sampler's random stream is consumed. Chunk size is therefore
    a deterministic function of (context length, K, budget) and the run
    reproduces.
    """
    dev = prefix_ids.device
    S = prefix_ids.shape[1]
    per_tok = kv_bytes_per_token(target.config)
    chunk = max(1, min(M, int(kv_budget_bytes // max(1, per_tok * (S + K)))))
    stop = torch.tensor(sorted(stop_ids), device=dev, dtype=torch.long)

    parts = [[] for _ in range(K)]
    conds = [[] for _ in range(K)]
    reals = [[] for _ in range(K)]
    idxs = [[] for _ in range(K)]
    anchor = prefix_ids[0, S - 1]
    done = 0
    while done < M:
        c = min(chunk, M - done)
        cache, logits = _prefill(target, prefix_ids, c)
        alive = torch.ones(c, dtype=torch.bool, device=dev)
        prev = anchor.expand(c)                     # slot 0 conditions on the anchor
        gid = torch.arange(done, done + c, device=dev)      # identity across chunks
        for k in range(K):
            p = warp(logits, policy)
            am = alive.clone()
            if am.any():
                parts[k].append(p[am].clone())
                # Filtered by the CURRENT alive mask, so the conditioning token
                # stays row-aligned with parts[k] even though paths die between
                # slots and alive[k] is a strict subset of alive[k-1].
                conds[k].append(prev[am].clone())
                idxs[k].append(gid[am].clone())
            nxt = torch.multinomial(p, 1, generator=gen)                 # [c,1]
            if am.any():
                # The token the TARGET realised HERE, in parts[k]'s row order.
                # The accept test is conditioned on it, so it must be filtered
                # by the pre-update mask, not the post-update one.
                reals[k].append(nxt[am, 0].clone())
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
    V = text_cfg(target.config).vocab_size
    cat = lambda ls, sh, dt: (torch.cat(ls, 0) if ls
                              else torch.zeros(sh, device=dev, dtype=dt))
    return ([cat(pl, (0, V), torch.float32) for pl in parts],
            [cat(cl, (0,), torch.long) for cl in conds],
            [cat(rl, (0,), torch.long) for rl in reals],
            [cat(il, (0,), torch.long) for il in idxs], chunk)


# -------------------------------------------------------------------- main ---
# The decorator is load-bearing, not hygiene. Every helper below carries its
# own, but the hidden-state pass in the loop is called bare, and a 48-layer
# target at 3k context retains ~16 MiB per token of autograd graph -- 46 GiB
# on one anchor, which is what cost the scale run its long-context domain.
@torch.no_grad()
def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--corpus", default="C0", choices=sorted(C.CORPORA))
    ap_.add_argument("--corpus-file", required=True)
    ap_.add_argument("--cheap", required=True, help="anchor source, same file probe_tk used")
    ap_.add_argument("--out", required=True)
    ap_.add_argument("--target", default=C.TARGET)
    ap_.add_argument("--drafter", required=True, help="rung-0 drafter checkpoint (dflash_sgl)")
    ap_.add_argument("--order", type=int, default=0, choices=(0, 1),
                     help="0: product-measure drafter (DFlash) against T^(0). "
                          "1: markov-head drafter (DSpark) against T^(1).")
    ap_.add_argument("--cond", default="both", choices=("oracle", "self", "both"),
                     help="--order 1 only. oracle: condition on the target's "
                          "realisation Z_{k-1}, which is what a slot reached "
                          "under left-to-right verification actually saw. "
                          "self: condition on the head's own free-rollout "
                          "sample, reported separately as exposure. Both are "
                          "cheap on the same paths, so both is the default.")
    ap_.add_argument("--anchors", type=int, default=96)
    ap_.add_argument("--paths", type=int, default=256)
    ap_.add_argument("--split", action="store_true")
    ap_.add_argument("--kv-budget-gib", type=float, default=12.0,
                     help="cap on the expanded prefix KV cache; fixed, not "
                          "free-memory-derived, so chunking is reproducible")
    args = ap_.parse_args()

    device = "cuda"
    policy = C.CORPORA[args.corpus]
    K = C.GAMMA

    seqs = {}
    with open(args.corpus_file) as fh:
        for line in fh:
            r = json.loads(line)
            seqs[r["prompt_id"]] = r

    # Anchor selection is byte-identical to probe_tk so the two runs cover the
    # same cells; G is only a paired quantity if the anchor sets agree.
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
          f"prompts; M={args.paths}; K={K}; full-vocab exact TV", flush=True)

    tok = AutoTokenizer.from_pretrained(args.target)
    stop_ids = resolve_stops(args.target, tok)

    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    draft, dcfg = load_drafter(args.drafter, device, order=args.order)
    if int(dcfg.block_size) != K:
        raise SystemExit(f"drafter block_size={dcfg.block_size} but GAMMA={K}")
    taps = list(dcfg.target_layer_ids)
    print(f"== drafter {args.drafter}: block={dcfg.block_size} markov_rank="
          f"{dcfg.markov_rank} taps={taps} order={args.order}"
          + (f" cond={args.cond}" if args.order else ""), flush=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = n_oom = 0
    with out.open("w") as fout:
        for a in anchors:
            seq = seqs[a["prompt_id"]]
            full = seq["prompt_ids"] + seq["response_ids"]
            cut = seq["prompt_len"] + a["t"]
            if len(full) < cut + K:
                continue
            prefix = torch.tensor([full[:cut]], device=device, dtype=torch.long)

            try:
                # logits_to_keep: the hidden-state pass needs hidden states, not
                # a [1, S, 151936] logit tensor -- 2.4 GiB at 8k context, and it
                # was being built and thrown away once per anchor.
                th = target(input_ids=prefix, output_hidden_states=True,
                            use_cache=False, logits_to_keep=1)
                thid = _deepspec.extract_context_feature(th.hidden_states, taps).to(torch.bfloat16)
                del th
                dl, dhid = drafter_logits(draft, dcfg, thid, prefix, K, device)
                hctx = head_context(draft, thid, prefix, K)
                del thid

                seed = anchor_seed(a["prompt_id"], a["t"], C.SEED) % (2 ** 63 - 1)
                gen = torch.Generator(device=device)
                gen.manual_seed(seed)

                base = q_warp = q_temp = chain = None
                if args.order == 0:
                    q_warp = warp(dl, policy)                   # serving warps q too
                    q_temp = warp(dl, {"temp": policy.get("temp", 1.0),
                                       "top_p": 1.0, "top_k": 0})  # temperature only
                    del dl
                else:
                    base = dl                                   # per-path q needs it
                    if args.cond in ("self", "both"):
                        # Its OWN generator: the target rollout below must draw
                        # the same paths whatever the drafter side is doing, or
                        # order-0 and order-1 numbers stop being paired.
                        gc = torch.Generator(device=device)
                        gc.manual_seed((seed ^ 0x5EED_C4A1) % (2 ** 63 - 1))
                        chain = chain_rollout(draft, base, K, args.paths, policy,
                                              gc, full[cut - 1], dhid, hctx)

                slots, conds, reals, idxs, chunk = rollout(
                    target, prefix, args.paths, K, policy, stop_ids, gen,
                    int(args.kv_budget_gib * (1 << 30)))
            except torch.OutOfMemoryError:
                # One long anchor must not take the domain down with it. Skipped
                # anchors are counted and printed, never silently dropped.
                n_oom += 1
                torch.cuda.empty_cache()
                print(f"   !! OOM at ctx={cut}, anchor skipped ({n_oom} so far)", flush=True)
                continue

            row = {k_: a[k_] for k_ in ("prompt_id", "t", "context", "stratum", "pi")}
            row.update(corpus=args.corpus, M=args.paths, K=K, chunk=chunk,
                       order=args.order, cond=(args.cond if args.order else None),
                       T={}, R={}, G={}, R_temp={}, T_split={}, alive={}, rec={},
                       T1={}, R_self={}, S={}, abar={}, R_serve={})
            # W_{k-1} = prod_{i<k} min(1, q_i(Z_i)/p_i(Z_i)) is this path's
            # probability of REACHING slot k, so it converts the free-rollout
            # population into the serving one. Dense over all M paths: a path
            # that died gets factor 0 and stays dead, which is correct -- the
            # block ends there.
            W = torch.ones(args.paths, device=device, dtype=torch.float64)
            for k in range(K):
                P = slots[k]
                row["alive"][str(k)] = int(P.shape[0])
                if P.shape[0] < 2:
                    continue
                # T^(0) on these paths is reported at BOTH orders: at order 1 it
                # is what the same block would be worth with the chain switched
                # off, so T^(0) - T^(1) is the value of the conditioning token
                # and R^(1) - T^(1) is what is left for the backbone.
                T = mean_tv(P, barycentre(P))
                row["T"][str(k)] = T
                if args.order == 0:
                    R = mean_tv(P, q_warp[k])
                    row["R"][str(k)] = R
                    row["G"][str(k)] = R - T
                    row["R_temp"][str(k)] = mean_tv(P, q_temp[k])
                    row["rec"][str(k)] = recall_floors(P, q_warp[k])
                else:
                    zc = conds[k]
                    row["T1"][str(k)] = cond_floor(P, zc)
                    Q_orc = warp(chain_logits(draft, base, k, zc, dhid, hctx), policy)
                    row["R"][str(k)] = mean_tv_rows(P, Q_orc)
                    if chain is not None:
                        na = P.shape[0]
                        ps = (zc if k == 0 else chain[:na, k - 1])
                        Q = warp(chain_logits(draft, base, k, ps, dhid, hctx), policy)
                        row["R_self"][str(k)] = mean_tv_rows(P, Q)
                        del Q
                # ---- serving reweighting. q_k is whatever this drafter actually
                # proposes at slot k, so the same code serves both orders.
                qk = q_warp[k] if args.order == 0 else Q_orc
                tvs = tv_rows(P, qk).double()
                w = W[idxs[k]]
                sw = float(w.sum().item())
                row["R_serve"][str(k)] = (float((w * tvs).sum().item() / sw)
                                          if sw > 0 else None)
                a_k = accept_factor(P, qk, reals[k]).double()
                row["abar"][str(k)] = float(a_k.mean().item())
                dense = torch.zeros_like(W).index_put_((idxs[k],), a_k)
                W = W * dense
                row["S"][str(k)] = float(W.mean().item())
                del tvs, qk

                if args.split:
                    idx = torch.randperm(P.shape[0], generator=gen, device=device)
                    h1, h2 = idx[0::2], idx[1::2]
                    if h1.numel() >= 2 and h2.numel() >= 2:
                        row["T_split"][str(k)] = mean_tv(P[h2], barycentre(P[h1]))
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            n += 1
            del slots, conds, reals, idxs, q_warp, q_temp, base, chain, dhid, hctx
            if n % 8 == 0:
                torch.cuda.empty_cache()
                print(f"   {n}/{len(anchors)}  last T5={row['T'].get('5')} "
                      f"R5={row['R'].get('5')}", flush=True)

    print(f"\n== wrote {n} rows to {out}" + (f"; {n_oom} anchors skipped on OOM" if n_oom else ""))
    if args.order == 0:
        print("G = R - T is >= 0 by construction (T is a min over ALL q, and q_DFlash")
        print("is one of them). A G near 0 means DFlash has essentially exhausted what")
        print("a realisation-blind proposal can do, so parallel capacity has nothing")
        print("left to buy -- that is the ceiling, not an estimate of the gain.")
    else:
        print("R is measured against T1, not T: the head sees Z_{k-1}, so the floor")
        print("it is allowed to reach is the conditional one. T is carried in the same")
        print("rows so that T - T1 (the value of the conditioning token) and R - T1")
        print("(what the backbone still owes) are read off one set of paths.")


if __name__ == "__main__":
    main()
