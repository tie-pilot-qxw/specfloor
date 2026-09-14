"""Gate for probe_rpre's attention-head wiring: the sliced context must equal the masked one.

probe_rpre scores ONE block, so `head_context` builds the head's context as
`target_hidden[:, :anchor_pos]` with `context_mask=None`, instead of the full prefix behind a
flex mask whose rule is `kv_idx < anchor_pos`.  Those are the same visibility only if nothing
else is read from the masked columns.  That is true by construction -- and it is exactly the
kind of "obviously equivalent" step that has silently ablated a module before, so it is checked
numerically rather than argued.

Second check: the per-row broadcast.  The probe pushes B conditioning tokens through
`apply_step_logits` against one shared context, while training uses `apply_block_logits`; if
expanding the context changed the bias, every order-1 number for such a head would be wrong.

Random weights, no checkpoint and no target model: what is new here is the wiring, not the
values.  Needs CUDA, because the masked path is flex_attention.
"""
import pytest

torch = pytest.importorskip("torch")

from specfloor import _deepspec
from specfloor.probe_rpre import expand_context

pytestmark = pytest.mark.skipif(
    not (_deepspec.available() and torch.cuda.is_available()),
    reason="needs DeepSpec on the path and a CUDA device (flex_attention)",
)

K, S, V, H, B, SLOT = 7, 256, 128, 32, 16, 3


def _head(dev):
    from deepspec.modeling.dspark.attn_head import AttnHead
    head = AttnHead(vocab_size=V, markov_rank=64, hidden_size=H, num_heads=2, head_dim=16,
                    mlp_hidden=64, gate_mode="none", out_scale=0.35,
                    anchor_kv=False).to(dev).eval()
    # The head borrows the target's embedding table by reference rather than owning one,
    # so a bare construction has none and every call asserts. Standing this up is part of
    # the fixture, not of what is being tested.
    head.set_embedding_source(torch.nn.Embedding(V, H).to(dev))
    return head


@torch.no_grad()
def _fixtures(dev):
    torch.manual_seed(0)
    head = _head(dev)
    tgt = torch.randn(1, S, H, device=dev)
    anchor_ids = torch.randint(0, V, (1, 1), device=dev)
    masked = head.build_context(
        target_hidden=tgt, anchor_token_ids=anchor_ids,
        anchor_positions=torch.tensor([[S - 1]], device=dev),
        block_keep_mask=torch.ones(1, 1, dtype=torch.bool, device=dev),
        block_size=K, context_window=None)
    sliced = head.build_context(
        target_hidden=tgt[:, :S - 1], anchor_token_ids=anchor_ids,
        context_mask=None, block_size=K)
    return head, masked, sliced


@torch.no_grad()
def test_sliced_context_equals_masked():
    dev = "cuda"
    head, masked, sliced = _fixtures(dev)
    tok = torch.randint(0, V, (1, 1, K), device=dev)
    hid = torch.randn(1, 1, K, H, device=dev)
    a = head.compute_block_latent(tok, hid, masked)
    b = head.compute_block_latent(tok, hid, sliced)
    assert (a - b).abs().max().item() < 2e-3


@torch.no_grad()
def test_broadcast_context_matches_one_at_a_time():
    dev = "cuda"
    head, _, sliced = _fixtures(dev)
    hid = torch.randn(1, 1, K, H, device=dev)
    base = torch.randn(V, device=dev)
    prev = torch.randint(0, V, (B,), device=dev)
    step = head.apply_step_logits(
        base.unsqueeze(0).expand(B, -1), token_ids=prev,
        hidden_states=hid[0, 0, SLOT].unsqueeze(0).expand(B, -1),
        context=expand_context(sliced, B))
    ref = torch.stack([
        head.apply_block_logits(
            base.view(1, 1, 1, V), token_ids=prev[i].view(1, 1, 1),
            hidden_states=hid[0, 0, SLOT].view(1, 1, 1, H),
            context=sliced).view(V)
        for i in range(B)])
    assert (step - ref).abs().max().item() < 2e-3
