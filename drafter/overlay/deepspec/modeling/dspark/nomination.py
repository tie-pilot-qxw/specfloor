"""Candidate-nomination loss on the base logits u (before the order-1 head).

At serving time the head scores transitions only among the top-16 candidates of u,
so u acts as a nominator: a token outside Top-16(u) can never be proposed.  This
term trains u for that role,

    L_nom,k = -sum_{v in S_k} p_k^nom(v) log softmax(u_k)_v,

where S_k is the target's top-K set and p_k^nom the target distribution
renormalised on S_k.  The softmax spans the full vocabulary, so the term is a
competition between the top-K and everything else, which is the event a candidate
set decides.
"""

import os

import torch

_COMPILED_NOM = None


def _nom_impl(unary_logits, target_logits, eval_mask, top_k, top_idx=None):
    """Per-slot sums of CE(u ; p_topK) = logsumexp(u) - sum_b p_topK(b) u_b.

    Returned as per-slot (num, den) so the caller applies the same slot weights as
    the main objective.  The target's top-K is taken on its logits: softmax is
    monotone, and renormalising the K values equals a softmax over those K.
    """
    u = unary_logits.float()
    w = eval_mask.to(torch.float32)                           # [B, N, K]
    if top_idx is None:
        tvals, idx = target_logits.topk(top_k, dim=-1)
    else:
        idx = top_idx
        tvals = torch.gather(target_logits, -1, idx)
    p16 = tvals.float().softmax(-1)
    ce = torch.logsumexp(u, dim=-1) - (p16 * torch.gather(u, -1, idx)).sum(-1)
    return (ce * w).sum(dim=(0, 1)), w.sum(dim=(0, 1))


def nomination_terms(*, unary_logits, target_logits, eval_mask, top_k, top_idx=None):
    """Compiled by default; eagerly this materialises a full-vocabulary fp32 tensor.
    Set DSPARK_NOMINATION_EAGER=1 to run it eagerly."""
    global _COMPILED_NOM
    if os.environ.get("DSPARK_NOMINATION_EAGER", "0") == "1":
        return _nom_impl(unary_logits, target_logits, eval_mask, top_k, top_idx)
    if _COMPILED_NOM is None:
        _COMPILED_NOM = torch.compile(_nom_impl, dynamic=False)
    return _COMPILED_NOM(unary_logits, target_logits, eval_mask, top_k, top_idx)


__all__ = ["nomination_terms"]
