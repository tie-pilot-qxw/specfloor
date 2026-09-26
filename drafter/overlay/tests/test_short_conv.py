"""Tests for the two-tap block-causal short convolution (DSparkShortConv).

The operator must be the identity at initialisation (so adding it does not perturb
the model at step 0), strictly block-local and causal within the block, reachable
by gradient, and the fused CUDA kernel must match the eager definition, including
under torch.compile.
"""
import pytest
import torch

from deepspec.modeling.dspark.qwen3.config import build_draft_config
from deepspec.modeling.dspark.qwen3.modeling import (
    DSparkShortConv,
    Qwen3DSparkDecoderLayer,
)


class _Cfg:
    hidden_size = 64
    num_attention_heads = 8
    num_key_value_heads = 2
    head_dim = 8
    intermediate_size = 128
    rms_norm_eps = 1e-6
    attention_bias = False
    attention_dropout = 0.0
    hidden_act = "silu"
    block_size = 7
    layer_types = ["full_attention"]
    _attn_implementation = "eager"
    short_conv = True
    short_conv_group_size = 16


def _cfg(**over):
    c = _Cfg()
    for k, v in over.items():
        setattr(c, k, v)
    return c


def _inputs(cfg, n_blocks=3, bsz=2):
    torch.manual_seed(0)
    return torch.randn(bsz, n_blocks * cfg.block_size, cfg.hidden_size)


def _rope(total_len, head_dim):
    pos = torch.arange(total_len, dtype=torch.float32).unsqueeze(-1)
    inv = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    ang = pos * inv
    emb = torch.cat([ang, ang], dim=-1).unsqueeze(0)
    return emb.cos(), emb.sin()


def test_conv_is_exactly_identity_at_init():
    cfg = _cfg()
    conv = DSparkShortConv(cfg)
    x = _inputs(cfg)
    assert torch.equal(conv(x), x), "identity init must pass the input through untouched"


def test_decoder_layer_is_bit_exact_vs_no_conv_at_init():
    x = _inputs(_cfg())
    torch.manual_seed(7)
    off = Qwen3DSparkDecoderLayer(_cfg(short_conv=False), 0).eval()
    torch.manual_seed(7)
    on = Qwen3DSparkDecoderLayer(_cfg(short_conv=True), 0).eval()
    on.load_state_dict(off.state_dict(), strict=False)
    kw = dict(
        target_hidden_states=torch.zeros(x.shape[0], 5, _Cfg.hidden_size),
        position_embeddings=_rope(x.shape[1] + 5, _Cfg.head_dim),
        attention_mask=None,
    )
    with torch.no_grad():
        assert torch.equal(off(hidden_states=x, **kw), on(hidden_states=x, **kw)), (
            "four freshly initialized convs must not perturb the checkpoint"
        )


def test_all_four_insertion_points_are_present():
    layer = Qwen3DSparkDecoderLayer(_cfg(short_conv=True), 0)
    for name in ("conv_pre_attn", "conv_post_attn", "conv_pre_mlp", "conv_post_mlp"):
        assert isinstance(getattr(layer, name), DSparkShortConv), f"missing {name}"


def test_k1_receives_gradient_at_init():
    """Identity init must not mean a dead branch: k1 is what has to travel."""
    cfg = _cfg()
    conv = DSparkShortConv(cfg)
    out = conv(_inputs(cfg))
    (out * torch.randn_like(out)).sum().backward()
    assert conv.k_base.grad is not None
    g1 = conv.k_base.grad[1].abs().max()
    assert float(g1) > 0.0, "k1 has no gradient -- the conv could never learn to mix"
    assert conv.corr.weight.grad is not None and float(
        conv.corr.weight.grad.abs().max()
    ) > 0.0, "the dynamic correction is unreachable"


def test_conv_cannot_leak_across_blocks():
    cfg = _cfg()
    conv = DSparkShortConv(cfg)
    with torch.no_grad():
        conv.k_base[1].fill_(0.5)
    K = cfg.block_size
    x = _inputs(cfg, n_blocks=3)
    y = x.clone()
    y[:, K : 2 * K] += 10.0
    with torch.no_grad():
        a, b = conv(x), conv(y)
    assert torch.equal(a[:, :K], b[:, :K])
    assert torch.equal(a[:, 2 * K :], b[:, 2 * K :]), (
        "block 1 reached block 2 -- the conv must not cross a block boundary, which is "
        "what lets serving skip the KV cache entirely"
    )


def test_conv_is_causal_within_the_block():
    cfg = _cfg()
    conv = DSparkShortConv(cfg)
    with torch.no_grad():
        conv.k_base[1].fill_(0.5)
    x = _inputs(cfg, n_blocks=1)
    y = x.clone()
    y[:, -1] += 10.0                      # perturb the LAST slot
    with torch.no_grad():
        a, b = conv(x), conv(y)
    assert torch.equal(a[:, :-1], b[:, :-1]), "two-tap {0,-1} must be strictly causal"


def test_slot_zero_has_no_predecessor():
    """Slot 0 already holds the last verified token; it must not read outside the block."""
    cfg = _cfg()
    conv = DSparkShortConv(cfg)
    with torch.no_grad():
        conv.k_base[0].zero_()            # keep only the x_{t-1} tap
        conv.k_base[1].fill_(1.0)
    x = _inputs(cfg, n_blocks=2)
    with torch.no_grad():
        out = conv(x)
    K = cfg.block_size
    for blk in (0, 1):
        assert torch.equal(out[:, blk * K], torch.zeros_like(out[:, blk * K])), (
            "slot 0's predecessor tap must be zero, not the previous block's last slot"
        )


def test_reset_new_params_restores_identity():
    """The meta-device load returns bare Parameters as garbage; the repair must work."""
    cfg = _cfg()
    conv = DSparkShortConv(cfg)
    with torch.no_grad():
        conv.k_base.normal_(0, 5.0)
        conv.corr.weight.normal_(0, 5.0)
    conv.reset_new_params()
    x = _inputs(cfg)
    assert torch.equal(conv(x), x)


@pytest.mark.parametrize("field,value", [("short_conv", True), ("short_conv_group_size", 32)])
def test_config_flags_reach_the_draft_config(field, value):
    from transformers import Qwen3Config

    from deepspec.utils.config import ConfigNode

    target = Qwen3Config(
        hidden_size=64, num_hidden_layers=8, num_attention_heads=8,
        num_key_value_heads=2, intermediate_size=128, vocab_size=128,
    )
    args = ConfigNode({
        "num_draft_layers": 2, "target_layer_ids": [1, 5], "block_size": 7,
        "mask_token_id": 3, "num_anchors": 4, "confidence_head_alpha": 0.0,
        "markov_rank": 0, field: value,
    })
    assert getattr(build_draft_config(target, args), field) == value


# --------------------------------------------------------------------------- #
# The fused CUDA path must be the same operator, not merely a similar one.
#
# The eager body is transcribed here rather than imported: it is the definition
# the model was specified against, so it serves as a frozen oracle.  Gradients are
# checked alongside the output: an operator can be forward-correct with an
# unreachable parameter.
# --------------------------------------------------------------------------- #
def _eager_original(xb, c_small, kb, group):
    c = c_small.repeat_interleave(group, dim=-1)
    k0 = kb[0] + c[:, :, 0, :]
    k1 = kb[1] + c[:, :, 1, :]
    prev = torch.zeros_like(xb)
    prev[:, 1:] = xb[:, :-1]
    return k0 * xb + k1 * prev


def _spare_cuda_device(need_gib=4.0):
    """Pick a card with room, not cuda:0.

    This is a shared 8-GPU box that is usually full of other people's training
    runs; hardcoding cuda:0 made these tests fail as OOM-dressed-as-"diverged"
    while passing in isolation.  Skipping is the correct outcome when every card
    is busy -- a numerics test that competes for memory tests the wrong thing.

    Queried out of process on purpose: torch.cuda.mem_get_info(i) has to create a
    CUDA context on card i to answer, and on a full card that call is itself the
    OOM we are trying to avoid.
    """
    import os
    import subprocess

    if not torch.cuda.is_available():
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.split()
        free_gib = [int(v) / 1024 for v in out]
    except Exception:
        return None

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    phys = ([int(v) for v in visible.split(",") if v.strip() != ""]
            if visible else list(range(len(free_gib))))
    ranked = sorted(range(len(phys)), key=lambda i: -free_gib[phys[i]])
    if not ranked or free_gib[phys[ranked[0]]] < need_gib:
        return None
    return f"cuda:{ranked[0]}"


DEV = _spare_cuda_device()

cuda_only = pytest.mark.skipif(
    DEV is None, reason="fused path needs a CUDA card with free memory"
)


def _fused_vs_eager(M, K, D, group, dtype):
    from deepspec.modeling.dspark.qwen3.short_conv_kernel import short_conv_apply

    torch.manual_seed(0)
    G = D // group
    dev = DEV
    x = torch.randn(M, K, D, device=dev, dtype=dtype)
    c = torch.randn(M, K, 2, G, device=dev, dtype=dtype) * 0.1
    kb = torch.randn(2, D, device=dev, dtype=dtype)
    gy = torch.randn(M, K, D, device=dev, dtype=dtype)

    def run(fn):
        a, b, k = (t.detach().clone().requires_grad_(True) for t in (x, c, kb))
        fn(a, b, k, group).backward(gy)
        return a.grad, b.grad, k.grad

    y_ref = _eager_original(x, c, kb, group)
    y_fus = short_conv_apply(x, c, kb, group)
    ref = run(_eager_original)
    fus = run(short_conv_apply)
    out = [(y_ref, y_fus)] + list(zip(ref, fus))
    return [
        ((f.float() - r.float()).abs().max() / r.float().abs().max().clamp_min(1e-6)).item()
        for r, f in out
    ]


@cuda_only
@pytest.mark.parametrize(
    "M,K,D,group",
    [
        (512, 7, 2560, 16),   # training
        (1, 7, 2560, 16),     # serving, bs=1
        (37, 7, 2560, 16),    # M not a multiple of the m-tile
        (512, 16, 2560, 16),  # block-16
        (8, 7, 1408, 16),     # D not a multiple of BLOCK_D
        (8, 7, 2560, 32),     # wider channel group
    ],
)
def test_fused_matches_the_eager_operator(M, K, D, group):
    rel = _fused_vs_eager(M, K, D, group, torch.float32)
    names = ["y", "grad_x", "grad_c", "grad_k_base"]
    for n, r in zip(names, rel):
        assert r < 2e-5, f"{n} diverged from the eager operator: rel={r:.3e}"


@cuda_only
def test_fused_is_not_less_accurate_in_bf16():
    """The fused kernel accumulates in fp32 where the eager body multiplied in
    bf16, so it must sit CLOSER to the fp32 answer, never further."""
    from deepspec.modeling.dspark.qwen3.short_conv_kernel import short_conv_apply

    torch.manual_seed(0)
    M, K, D, group = 128, 7, 2560, 16
    x = torch.randn(M, K, D, device=DEV, dtype=torch.bfloat16)
    c = torch.randn(M, K, 2, D // group, device=DEV, dtype=torch.bfloat16) * 0.1
    kb = torch.randn(2, D, device=DEV, dtype=torch.bfloat16)
    gold = _eager_original(x.float(), c.float(), kb.float(), group)
    e = (_eager_original(x, c, kb, group).float() - gold).abs().mean()
    f = (short_conv_apply(x, c, kb, group).float() - gold).abs().mean()
    assert f <= e, f"fused error {f:.3e} worse than eager bf16 {e:.3e}"


@cuda_only
def test_identity_init_is_bit_exact_on_the_fused_path():
    """Identity initialisation must be bit-exact on the fused path."""
    from deepspec.modeling.dspark.qwen3.short_conv_kernel import short_conv_apply

    for dtype in (torch.float32, torch.bfloat16):
        x = torch.randn(64, 7, 2560, device=DEV, dtype=dtype)
        c = torch.zeros(64, 7, 2, 160, device=DEV, dtype=dtype)
        kb = torch.zeros(2, 2560, device=DEV, dtype=dtype)
        kb[0].fill_(1.0)
        assert torch.equal(short_conv_apply(x, c, kb, 16), x)


@cuda_only
def test_fused_path_does_not_break_the_compile_graph():
    """The paper configs train with torch.compile.  A raw triton launch is
    traced into by AOTAutograd -- forward AND backward have to be opaque ops, or
    inductor dies on a FakeTensor data pointer."""
    import torch._dynamo as dynamo

    from deepspec.modeling.dspark.qwen3.short_conv_kernel import short_conv_apply

    D, K, M, group = 512, 7, 16, 16

    class _Mod(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.corr = torch.nn.Linear(D, 2 * (D // group), bias=False)
            self.kb = torch.nn.Parameter(torch.zeros(2, D))
            with torch.no_grad():
                self.kb[0].fill_(1.0)

        def forward(self, t):
            c = self.corr(t).view(t.shape[0], K, 2, D // group)
            return short_conv_apply(t, c, self.kb, group)

    mod = _Mod().to(DEV)
    t = torch.randn(M, K, D, device=DEV, requires_grad=True)
    gy = torch.randn_like(t)

    explained = dynamo.explain(mod)(t)
    assert explained.graph_break_count == 0, explained.break_reasons

    eager_y = mod(t)
    eager_y.backward(gy)
    ref = (t.grad.clone(), mod.kb.grad.clone(), mod.corr.weight.grad.clone())
    t.grad = None
    mod.zero_grad()

    torch.compile(mod, dynamic=True)(t).backward(gy)
    got = (t.grad, mod.kb.grad, mod.corr.weight.grad)
    for name, a, b in zip(("grad_x", "grad_k_base", "grad_corr"), ref, got):
        assert torch.equal(a, b), f"{name} differs under torch.compile"


def _ensure_single_rank_group():
    """metrics.flush() reduces across data-parallel ranks, so it needs a group."""
    import os

    import torch.distributed as dist

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29751")
        dist.init_process_group("gloo", rank=0, world_size=1)


def test_conv_health_panel_reports_mix_gate_and_gradient_separately():
    """The panel has to separate the two things the conv does, or a null is unreadable.

    k1 is within-block communication (the predecessor tap, zero at init); k0-1 is
    per-channel gating, which is all the conv does at slot 0 where prev is zero.
    """
    import torch.nn as nn

    from deepspec.trainer.dspark_online_trainer import OnlineTargetTrainer
    from deepspec.utils import metrics

    cfg = _cfg(short_conv=True)
    cfg.layer_types = ["full_attention"] * 4
    layer = Qwen3DSparkDecoderLayer(cfg, 3)
    names = {c.probe_name for c in layer.modules() if isinstance(c, DSparkShortConv)}
    assert names == {"L3.pre_attn", "L3.post_attn", "L3.pre_mlp", "L3.post_mlp"}, names

    class _Stub:
        model = layer
        depth_controller = None
        world_size = 1
        _log_short_conv_health = OnlineTargetTrainer._log_short_conv_health

    _ensure_single_rank_group()
    metrics.reset()
    _Stub()._log_short_conv_health()
    at_init = metrics.flush()
    assert at_init["train/conv_k1"] == 0.0, "k1 must start at 0 (identity init)"
    assert at_init["train/conv_k0d"] == 0.0, "k0 must start at 1 (identity init)"
    assert set(names) <= {k.split("@", 1)[1] for k in at_init if k.startswith("train/conv_k1@")}

    # move ONLY the gate, and only on one insertion point: the aggregate must move
    # in k0d and stay exactly zero in k1, and the per-point tag must localise it.
    with torch.no_grad():
        layer.conv_post_mlp.k_base[0].fill_(1.5)
        layer.conv_post_mlp.k_base.grad = torch.zeros_like(layer.conv_post_mlp.k_base)
        layer.conv_post_mlp.k_base.grad[1].fill_(0.25)
    _ensure_single_rank_group()
    metrics.reset()
    _Stub()._log_short_conv_health()
    after = metrics.flush()
    assert after["train/conv_k1"] == 0.0, "gating leaked into the mixing readout"
    assert after["train/conv_k0d"] == pytest.approx(0.5 / 4), after["train/conv_k0d"]
    assert after["train/conv_g1"] == pytest.approx(0.25 / 4), after["train/conv_g1"]
    assert after["train/conv_k1@L3.post_mlp"] == 0.0

    with torch.no_grad():
        layer.conv_pre_attn.k_base[1].fill_(0.3)
    _ensure_single_rank_group()
    metrics.reset()
    _Stub()._log_short_conv_health()
    mixed = metrics.flush()
    assert mixed["train/conv_k1"] == pytest.approx(0.3 / 4)
    assert mixed["train/conv_k1@L3.pre_attn"] == pytest.approx(0.3)
    assert mixed["train/conv_k1@L3.post_mlp"] == 0.0, "the panel must localise, not smear"


def test_conv_health_panel_is_a_noop_without_convs():
    """The panel runs on every optimizer step, with or without convs."""
    from deepspec.trainer.dspark_online_trainer import OnlineTargetTrainer
    from deepspec.utils import metrics

    class _Stub:
        model = Qwen3DSparkDecoderLayer(_cfg(short_conv=False), 0)
        depth_controller = None
        world_size = 1
        _log_short_conv_health = OnlineTargetTrainer._log_short_conv_health

    _ensure_single_rank_group()
    metrics.reset()
    _Stub()._log_short_conv_health()
    assert metrics.flush() == {}
