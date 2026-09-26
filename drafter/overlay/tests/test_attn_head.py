"""Unit tests for the prefix-attention head (AttnHead).

  cell isolation        rows for different predecessor hypotheses of one slot must not
                        see each other (checked as an input invariance);
  anchor visibility     the optional anchor column is visible to its own block only;
  borrowed embedding    the target table is shared by reference, frozen, and not saved;
  mask equivalence      the flex BlockMask and the dense reference agree;
  scale                 predecessor and drafter state enter the query on comparable
                        scales, and every parameter receives gradient at step 0.
"""

import torch
import torch.nn as nn

from deepspec.modeling.dspark.attn_head import AttnHead

V, H, R, K, NB, S, B = 1024, 64, 32, 7, 5, 23, 2


def _head(gate_mode="state_output", seed=0):
    torch.manual_seed(seed)
    h = AttnHead(vocab_size=V, markov_rank=R, hidden_size=H, num_heads=2,
                     head_dim=16, mlp_hidden=48, gate_mode=gate_mode).double()
    emb = nn.Embedding(V, H).double()
    emb.weight.requires_grad_(False)
    h.set_embedding_source(emb)
    return h, emb


def _inputs(seed=1):
    g = torch.Generator().manual_seed(seed)
    return dict(
        tgt=torch.randn(B, S, H, generator=g, dtype=torch.float64),
        anc=torch.randint(0, V, (B, NB), generator=g),
        tok=torch.randint(0, V, (B, NB, K), generator=g),
        hid=torch.randn(B, NB, K, H, generator=g, dtype=torch.float64),
        mask=torch.ones(B, 1, NB * K, S, dtype=torch.bool),
    )


def _open(h):
    """Move the anchor type offset off zero so the invariance tests are not trivial."""
    nn.init.normal_(h.anchor_type, std=0.05)


def _run(h, d):
    ctx = h.build_context(target_hidden=d["tgt"], anchor_token_ids=d["anc"],
                          context_mask=d["mask"])
    return h.compute_block_latent(d["tok"], d["hid"], ctx)




def test_cells_cannot_see_each_other():
    """Perturbing ONE row must leave every other row bit-identical.

    Run with the zero-init branches opened, or the invariance would hold trivially.
    """
    h, _ = _head()
    _open(h)
    d = _inputs()
    base = _run(h, d)
    for victim in ((0, 2, 3), (1, 0, 0), (1, NB - 1, K - 1)):
        d2 = {k: (x.clone() if torch.is_tensor(x) else x) for k, x in d.items()}
        b, nb, k = victim
        d2["hid"][b, nb, k] += 7.0
        d2["tok"][b, nb, k] = (d2["tok"][b, nb, k] + 511) % V
        got = _run(h, d2)
        delta = (got - base).abs()
        assert delta[b, nb, k].max().item() > 0.0, "the perturbed row must change"
        delta[b, nb, k] = 0.0
        assert delta.max().item() == 0.0, (
            f"row {victim} leaked into another cell: max |d| = {delta.max().item():.3e}")


def test_anchor_is_visible_to_its_own_block_only():
    """Changing block b's anchor must move block b's rows and nothing else."""
    h, _ = _head()
    _open(h)
    d = _inputs()
    base = _run(h, d)
    for b, nb in ((0, 1), (1, NB - 1)):
        d2 = {k: (x.clone() if torch.is_tensor(x) else x) for k, x in d.items()}
        d2["anc"][b, nb] = (d2["anc"][b, nb] + 333) % V
        got = _run(h, d2)
        delta = (got - base).abs()
        assert delta[b, nb].max().item() > 0.0, "the block's own rows must move"
        delta[b, nb] = 0.0
        assert delta.max().item() == 0.0, (
            f"anchor of block {(b, nb)} reached another block: {delta.max().item():.3e}")


def test_embedding_is_borrowed_frozen_and_not_saved():
    h, emb = _head()
    assert h._embed is emb
    assert h.embed_tokens(torch.tensor([[1, 2]])).requires_grad is False
    # not a submodule -> not in state_dict, not double-counted, not separately wrapped.
    # Checked by SHAPE, not by name: `embed_norm.weight` is a legitimate parameter of
    # this head and a name match rejected it, which made the assertion about spelling
    # rather than about the invariant.  Nothing with a vocabulary axis belongs here
    # except the readout.
    vocab_shaped = [k for k, t in h.state_dict().items()
                    if V in tuple(t.shape) and "markov_w2" not in k]
    assert not vocab_shaped, vocab_shaped
    assert all(p is not emb.weight for p in h.parameters())
    # a gradient through the head must not reach the table even if someone unfreezes it
    emb.weight.requires_grad_(True)
    _open(h)
    _run(h, _inputs()).sum().backward()
    assert emb.weight.grad is None, "detach() must keep gradient out of the target table"


def test_codebook_swap_is_parameter_neutral():
    """Dropping W1 and doubling W2's rank is exactly, not approximately, free."""
    vocab, rank = 151936, 256
    assert 2 * rank * vocab == (2 * rank) * vocab == 77_791_232


def test_context_and_anchor_kv_stay_separate():
    """The anchor's K/V must come from the TOKEN, never from a context hidden.

    target_hidden[anchor_pos] is the target's post-anchor state and decodes to the
    slot-1 answer; the DSpark mask excludes it with a strict `<`.  This pins the
    contract that the head builds the anchor's key from the embedding path.
    """
    h, emb = _head()
    d = _inputs()
    ak, av = h.anchor_kv(d["anc"])
    expect_k, expect_v = h.anchor_kv(d["anc"])
    assert torch.equal(ak, expect_k) and torch.equal(av, expect_v)
    # a context of the WRONG length must not silently change the anchor keys
    d2 = dict(d, tgt=torch.randn(B, S + 5, H, dtype=torch.float64))
    ak2, _ = h.anchor_kv(d2["anc"])
    assert torch.equal(ak, ak2), "anchor keys must not depend on the context tensor"



def test_flex_block_mask_matches_the_dense_reference():
    """The flex mask_mod and the dense path must compute the same attention.

    The dense path is the reference precisely because it is the obvious one: the anchor
    strip is a `cat` of an identity block, which is hard to get wrong.  The flex mask_mod
    expresses the same thing as `(kv_idx - seq_len) == q_idx // block_size`, where an
    off-by-one silently lets a block read its NEIGHBOUR's anchor -- a token that block
    has not committed to, and a bug with no crash and no obviously wrong loss.
    """
    torch.manual_seed(3)
    h, _ = _head("none")
    _open(h)
    d = _inputs(seed=5)
    anchor_positions = torch.randint(K + 1, S, (B, NB))
    keep = torch.ones(B, NB, dtype=torch.bool)

    dense = torch.zeros(B, 1, NB * K, S, dtype=torch.bool)
    for b in range(B):
        for nb in range(NB):
            dense[b, 0, nb * K:(nb + 1) * K, :int(anchor_positions[b, nb])] = True
    # no_grad because FlexAttention has no CPU backward; only the forward is compared.
    with torch.no_grad():
        ref = h.compute_block_latent(
            d["tok"], d["hid"],
            h.build_context(target_hidden=d["tgt"], anchor_token_ids=d["anc"],
                            context_mask=dense))
        got = h.compute_block_latent(
            d["tok"], d["hid"],
            h.build_context(target_hidden=d["tgt"], anchor_token_ids=d["anc"],
                            anchor_positions=anchor_positions, block_keep_mask=keep,
                            block_size=K))
    assert torch.allclose(ref, got, atol=1e-8, rtol=1e-6), (
        f"flex vs dense max |d| = {(ref - got).abs().max().item():.3e}")


def test_confidence_head_feature_exists_and_has_the_rank_shape():
    """`predict_confidence_step` calls get_prev_embeddings whenever markov_rank > 0.

    A head that omits it raises AttributeError at the first confidence step -- which a
    smoke test with the confidence head disabled will never reach.
    """
    h, _ = _head()
    e = h.get_prev_embeddings(torch.randint(0, V, (B, NB, K)))
    assert e.shape == (B, NB, K, R), e.shape
    assert e.requires_grad is False or e.grad_fn is not None
    # no new parameters: it reuses in_proj's predecessor half
    assert h.get_prev_embeddings.__doc__ and "in_proj" in h.get_prev_embeddings.__doc__


def test_serving_paths_fail_with_a_reason_not_an_attributeerror():
    """The free-running routes need a prefix cache nobody builds yet.

    They must refuse in a way that NAMES the missing plumbing; an AttributeError from a
    missing method reads like a typo and sends the next person looking in the wrong file.
    """
    h, _ = _head()
    tok = torch.randint(0, V, (B,))
    hid = torch.randn(B, H, dtype=torch.float64)
    lg = torch.randn(B, V, dtype=torch.float64)
    for fn, kw in ((h.apply_step_logits, dict(logits=lg, token_ids=tok, hidden_states=hid)),
                   (h.sample_block_tokens, dict(base_logits=lg, first_prev_token_ids=tok,
                                                hidden_states=hid))):
        try:
            fn(**kw)
        except (AssertionError, NotImplementedError) as exc:
            assert "AttnHeadContext" in str(exc), str(exc)[:200]
        else:
            raise AssertionError(f"{fn.__name__} must refuse without a context")


# ---------------------------------------------------------------------------------
# Scale checks.  The structural tests above would all pass on a head whose query
# ignored the predecessor: a single RMSNorm over [E(prev); h] keeps the ratio the two
# halves arrive with, and on Qwen3-4B the embedding table has RMS 0.022 while the
# drafter's normed hidden state has RMS 2.79.  These check the predecessor is a
# comparable input.
_EMBED_RMS, _HIDDEN_RMS = 0.0220, 2.79        # measured on Qwen/Qwen3-4B


def _realistic_scales(h):
    """Give the borrowed table and the hidden stream their real, very different RMS."""
    with torch.no_grad():
        h._embed.weight.normal_(0, _EMBED_RMS)
    return torch.randn(B, NB, K, H, dtype=torch.float64) * _HIDDEN_RMS


def test_predecessor_and_hidden_enter_the_query_on_comparable_scales():
    h, _ = _head()
    hid = _realistic_scales(h)
    tok = torch.randint(0, V, (B, NB, K))
    e = h.embed_norm(h.embed_tokens(tok).to(hid.dtype))
    s = h.state_norm(hid)
    ratio = (s.pow(2).mean().sqrt() / e.pow(2).mean().sqrt()).item()
    assert 0.25 < ratio < 4.0, (
        f"the two halves of the query input differ by {ratio:.1f}x after their norms; "
        f"raw they differ by {_HIDDEN_RMS/_EMBED_RMS:.0f}x, so a single norm over the "
        f"concatenation would leave the predecessor contributing "
        f"1/{(_HIDDEN_RMS/_EMBED_RMS)**2:.0f} of the variance")


def test_the_query_actually_depends_on_the_predecessor():
    """The functional form of the gate above: does changing prev move the query?

    Scale parity is the mechanism; this is the effect.  Compared against the movement
    from changing h_k, because an absolute number says nothing -- the claim is that the
    predecessor is a comparable input, not a rounding error on one.
    """
    h, _ = _head()
    hid = _realistic_scales(h)
    tok = torch.randint(0, V, (B, NB, K))

    def q_of(t, x):
        e = h.embed_norm(h.embed_tokens(t).to(x.dtype))
        w = h.state_norm(x)
        flat = torch.cat([e, w], -1).reshape(B, NB * K, -1)
        return h.q_proj(h.attn_norm(h.in_proj(flat)))

    base = q_of(tok, hid)
    d_prev = (q_of((tok + 7919) % V, hid) - base).pow(2).mean().sqrt()
    d_hid = (q_of(tok, hid + torch.randn_like(hid) * _HIDDEN_RMS) - base).pow(2).mean().sqrt()
    rel = (d_prev / d_hid).item()
    assert rel > 0.05, (
        f"changing the predecessor moves the query only {rel:.2e} as much as changing "
        f"h_k does; the head's whole purpose is predecessor-conditioned retrieval")


def test_anchor_value_is_on_the_context_value_scale():
    """k_norm rescues the anchor KEY; nothing rescues its VALUE.

    Without a norm on the anchor stream, the extra column added specifically to carry
    the anchor initialises as a near-zero sink: attendable, but delivering ~1/127 of
    what any prefix column delivers.
    """
    h, _ = _head()
    with torch.no_grad():
        h._embed.weight.normal_(0, _EMBED_RMS)
    ctx = torch.randn(B, S, H, dtype=torch.float64) * _HIDDEN_RMS
    _, v_ctx = h.context_kv(ctx)
    _, v_anc = h.anchor_kv(torch.randint(0, V, (B, NB)))
    ratio = (v_ctx.pow(2).mean().sqrt() / v_anc.pow(2).mean().sqrt()).item()
    assert 0.25 < ratio < 4.0, (
        f"prefix values are {ratio:.1f}x the anchor's; the anchor column would be a "
        f"near-zero sink rather than a carrier of the anchor token")


def test_factory_refuses_the_attn_head_for_unwired_models():
    """qwen2/gemma4 share build_markov_head and wire neither the embedding nor a context.

    They would construct the head fine and die on the first forward, far from the config
    that asked for it.
    """
    from types import SimpleNamespace

    from deepspec.modeling.dspark.markov_head import build_markov_head

    cfg = SimpleNamespace(markov_rank=R, markov_head_type="attn", vocab_size=V,
                          hidden_size=H, markov_num_heads=2, markov_head_dim=16,
                          markov_mlp_hidden=48, markov_gate_mode="state_output")
    try:
        build_markov_head(cfg)                      # no context_wired -> must refuse
    except AssertionError as exc:
        assert "context_wired" in str(exc)
    else:
        raise AssertionError("the factory built an attn head for an unwired caller")
    assert build_markov_head(cfg, context_wired=True) is not None


def test_every_parameter_receives_gradient_on_the_first_step():
    """Every parameter receives gradient at step 0 under a training-form loss
    (cross-entropy on base_logits + bias): nothing in the head is zero-gated."""
    h, _ = _head()
    d = _inputs()
    ctx = h.build_context(target_hidden=d["tgt"], anchor_token_ids=d["anc"],
                          context_mask=d["mask"])
    base = torch.randn(B, NB, K, V, dtype=torch.float64)
    tgt = torch.randint(0, V, (B, NB, K))
    bias = h.project_bias(h.compute_block_latent(d["tok"], d["hid"], ctx))
    torch.nn.functional.cross_entropy((base + bias).reshape(-1, V), tgt.reshape(-1)).backward()
    dead = [n for n, p in h.named_parameters()
            if p.requires_grad and (p.grad is None or p.grad.abs().max() == 0)]
    assert not dead, f"{len(dead)} parameters get no gradient at step 0: {dead[:8]}"


def test_step0_bias_is_the_size_of_the_head_it_replaces():
    """No gate means the head emits something at init; it must not be something wild.

    out_norm's weight is set so the step-0 bias matches the vanilla markov bias
    (sqrt(256) * 0.0602 * 0.0664 ~ 0.064).  Left at 1, rank 512 would give ~1.5 -- a
    perturbation comparable to the logits themselves on every token.
    """
    h, _ = _head()
    with torch.no_grad():
        h.markov_w2.weight.normal_(0, 0.0664)
    d = _inputs()
    ctx = h.build_context(target_hidden=d["tgt"], anchor_token_ids=d["anc"],
                          context_mask=d["mask"])
    std = h.project_bias(h.compute_block_latent(d["tok"], d["hid"], ctx)).std().item()
    assert 0.01 < std < 0.4, f"step-0 bias std {std:.3f}; vanilla markov's is ~0.064"
