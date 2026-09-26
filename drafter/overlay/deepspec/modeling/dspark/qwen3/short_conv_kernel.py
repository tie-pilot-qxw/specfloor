"""Fused two-tap dynamic depthwise conv (the DFlash2 "short conv").

Semantics, identical to the eager body this replaces:

    xb   : [M, K, D]        M = bsz * n_blocks, K = block_size
    c    : [M, K, 2, G]     G = D // group, produced by a Linear on xb
    kb   : [2, D]           base kernel, identity-init (kb[0]=1, kb[1]=0)
    prev : xb shifted one slot inside the block, prev[:, 0] = 0
    y    = (kb[0] + c[:,:,0].repeat_interleave(group)) * xb
         + (kb[1] + c[:,:,1].repeat_interleave(group)) * prev

Why this exists.  The eager body allocates ~8 tensors of xb's size (two from the
repeat_interleave alone, which is a real copy at 2x width) and launches 8 kernels,
and every one of them is saved for backward.  Measured at the training shape
(M=512, K=7, D=2560, bf16) it costs 695 us of CUDA time per insertion point for
fwd+bwd of which only 43 us is the GEMM -- times 20 insertion points.  At the
serving shape the tensors are 17,920 elements and the cost is entirely per-launch
fixed overhead: 160 launches, +0.446 ms/round CUDA-graphed = 19.4% of the round.

Both problems are the same problem, so one fix: a single kernel forward, a single
kernel backward, and only xb + c (1/8 of xb's size) saved for backward.  The
correction Linear stays outside -- it is a real GEMM and cuBLAS should have it.

Numerics differ from the eager body in one respect, deliberately: all arithmetic
is fp32 internally regardless of the input dtype, where the eager body multiplied
and added in bf16.  Identity init is still bit-exact (1.0*x + 0.0*prev == x in
both), so "step 0 IS the loaded checkpoint" is preserved.
"""

from typing import Tuple

import torch
import torch.nn.functional as F

try:  # triton ships with torch on CUDA builds; keep CPU/import-only paths alive
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False


# --------------------------------------------------------------------------- #
# reference implementation -- CPU path, and the oracle the CUDA tests compare to
# --------------------------------------------------------------------------- #
def short_conv_reference(
    xb: torch.Tensor, c: torch.Tensor, kb: torch.Tensor, group: int
) -> torch.Tensor:
    """xb [M,K,D], c [M,K,2,G], kb [2,D] -> [M,K,D].  fp32 math, input dtype out."""
    dt = xb.dtype
    x32, c32, k32 = xb.float(), c.float(), kb.float()
    k0 = k32[0] + c32[:, :, 0, :].repeat_interleave(group, dim=-1)
    k1 = k32[1] + c32[:, :, 1, :].repeat_interleave(group, dim=-1)
    prev = F.pad(x32[:, :-1], (0, 0, 1, 0))
    return (k0 * x32 + k1 * prev).to(dt)


if _HAS_TRITON:

    @triton.jit
    def _sc_fwd(
        X, C, KB, Y,
        K: tl.constexpr, D: tl.constexpr, G: tl.constexpr,
        GROUP: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_d = tl.program_id(1)
        cols = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = cols < D
        gcols = cols // GROUP

        kb0 = tl.load(KB + cols, mask=mask, other=0.0).to(tl.float32)
        kb1 = tl.load(KB + D + cols, mask=mask, other=0.0).to(tl.float32)

        xbase = X + pid_m * (K * D)
        ybase = Y + pid_m * (K * D)
        cbase = C + pid_m * (K * 2 * G)

        # x_{t-1} carried in registers: one load per element, not two
        xprev = tl.zeros([BLOCK_D], dtype=tl.float32)
        for t in tl.static_range(K):
            xt = tl.load(xbase + t * D + cols, mask=mask, other=0.0).to(tl.float32)
            c0 = tl.load(cbase + t * (2 * G) + gcols, mask=mask, other=0.0).to(tl.float32)
            c1 = tl.load(cbase + t * (2 * G) + G + gcols, mask=mask, other=0.0).to(tl.float32)
            y = (kb0 + c0) * xt + (kb1 + c1) * xprev
            tl.store(ybase + t * D + cols, y.to(Y.dtype.element_ty), mask=mask)
            xprev = xt

    @triton.jit
    def _sc_bwd(
        X, C, KB, GY, GX, GC, GKB,
        M,
        K: tl.constexpr, D: tl.constexpr, G: tl.constexpr,
        GROUP: tl.constexpr, BLOCK_D: tl.constexpr,
        M_TILE: tl.constexpr, NGRP: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_d = tl.program_id(1)
        cols = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = cols < D
        gcols = cols // GROUP
        gids = pid_d * NGRP + tl.arange(0, NGRP)
        gmask = gids < G

        kb0 = tl.load(KB + cols, mask=mask, other=0.0).to(tl.float32)
        kb1 = tl.load(KB + D + cols, mask=mask, other=0.0).to(tl.float32)

        # per-program partials for the base kernel; reduced by a deterministic
        # torch.sum afterwards rather than atomics, so the run stays reproducible
        acc0 = tl.zeros([BLOCK_D], dtype=tl.float32)
        acc1 = tl.zeros([BLOCK_D], dtype=tl.float32)

        for i in tl.static_range(M_TILE):
            m = pid_t * M_TILE + i
            if m < M:
                xbase = X + m * (K * D)
                gybase = GY + m * (K * D)
                gxbase = GX + m * (K * D)
                cbase = C + m * (K * 2 * G)
                gcbase = GC + m * (K * 2 * G)
                xprev = tl.zeros([BLOCK_D], dtype=tl.float32)
                for t in tl.static_range(K):
                    xt = tl.load(xbase + t * D + cols, mask=mask, other=0.0).to(tl.float32)
                    gyt = tl.load(gybase + t * D + cols, mask=mask, other=0.0).to(tl.float32)
                    c0 = tl.load(cbase + t * (2 * G) + gcols, mask=mask, other=0.0).to(tl.float32)

                    gk0 = gyt * xt          # d/d k0  (elementwise)
                    gk1 = gyt * xprev       # d/d k1
                    acc0 += gk0
                    acc1 += gk1

                    # d/d c is the same product summed inside each channel group;
                    # BLOCK_D is a whole number of groups so this is a local reshape
                    s0 = tl.sum(tl.reshape(gk0, (NGRP, GROUP)), axis=1)
                    s1 = tl.sum(tl.reshape(gk1, (NGRP, GROUP)), axis=1)
                    tl.store(gcbase + t * (2 * G) + gids, s0.to(GC.dtype.element_ty), mask=gmask)
                    tl.store(gcbase + t * (2 * G) + G + gids, s1.to(GC.dtype.element_ty), mask=gmask)

                    # x_t is read by its own slot (k0 tap) and by slot t+1 (k1 tap)
                    gx = gyt * (kb0 + c0)
                    if t + 1 < K:
                        gyn = tl.load(gybase + (t + 1) * D + cols, mask=mask, other=0.0).to(tl.float32)
                        c1n = tl.load(cbase + (t + 1) * (2 * G) + G + gcols, mask=mask, other=0.0).to(tl.float32)
                        gx += gyn * (kb1 + c1n)
                    tl.store(gxbase + t * D + cols, gx.to(GX.dtype.element_ty), mask=mask)
                    xprev = xt

        tl.store(GKB + pid_t * (2 * D) + cols, acc0, mask=mask)
        tl.store(GKB + pid_t * (2 * D) + D + cols, acc1, mask=mask)


def _launch_cfg(M: int, D: int, group: int) -> Tuple[int, int, int]:
    """(BLOCK_D, M_TILE, num_warps).

    Two regimes with opposite needs.  Training: M ~ 512, so programs are plentiful
    and wide chunks amortise the c gather.  Serving: M = 1 and the whole tensor is
    17,920 elements, so the only thing that matters is that ONE kernel replaces
    eight -- occupancy is irrelevant, and a narrow chunk just adds programs that
    cost nothing.
    """
    block_d = 256
    while block_d > 64 and M * ((D + block_d - 1) // block_d) < 256:
        block_d //= 2
    block_d = max(block_d, group)
    m_tile = 8 if M >= 64 else 1
    num_warps = 4 if block_d >= 256 else 2
    return block_d, m_tile, num_warps


def _fwd_cuda(xb: torch.Tensor, c: torch.Tensor, kb: torch.Tensor, group: int) -> torch.Tensor:
    M, K, D = xb.shape
    G = c.shape[-1]
    block_d, _, num_warps = _launch_cfg(M, D, group)
    y = torch.empty_like(xb)
    grid = (M, triton.cdiv(D, block_d))
    _sc_fwd[grid](
        xb, c, kb, y,
        K=K, D=D, G=G, GROUP=group, BLOCK_D=block_d, num_warps=num_warps,
    )
    return y


def _bwd_cuda(
    gy: torch.Tensor, xb: torch.Tensor, c: torch.Tensor, kb: torch.Tensor, group: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    M, K, D = xb.shape
    G = c.shape[-1]
    block_d, m_tile, num_warps = _launch_cfg(M, D, group)
    n_tiles = triton.cdiv(M, m_tile)
    gx = torch.empty_like(xb)
    gc = torch.empty_like(c)
    # Programs sharing a pid_t own DISJOINT columns, so [n_tiles, 2, D] is written
    # exactly once per element -- no atomics, and the reduction below is a plain
    # deterministic sum.
    gkb_part = torch.empty(n_tiles, 2, D, device=xb.device, dtype=torch.float32)
    grid = (n_tiles, triton.cdiv(D, block_d))
    _sc_bwd[grid](
        xb, c, kb, gy, gx, gc, gkb_part,
        M,
        K=K, D=D, G=G, GROUP=group, BLOCK_D=block_d,
        M_TILE=m_tile, NGRP=block_d // group, num_warps=num_warps,
    )
    return gx, gc, gkb_part.sum(0).to(kb.dtype)


def _bwd_reference(
    gy: torch.Tensor, xb: torch.Tensor, c: torch.Tensor, kb: torch.Tensor, group: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.enable_grad():
        x_ = xb.detach().requires_grad_(True)
        c_ = c.detach().requires_grad_(True)
        k_ = kb.detach().requires_grad_(True)
        y = short_conv_reference(x_, c_, k_, group)
        return torch.autograd.grad(y, (x_, c_, k_), gy)


# --------------------------------------------------------------------------- #
# custom op: opaque to torch.compile (no graph break) and CUDA-graph capturable
# --------------------------------------------------------------------------- #
@torch.library.custom_op("deepspec_dspark::short_conv", mutates_args=())
def _short_conv_op(
    xb: torch.Tensor, c: torch.Tensor, kb: torch.Tensor, group: int
) -> torch.Tensor:
    return _fwd_cuda(xb.contiguous(), c.contiguous(), kb.contiguous(), group)


@_short_conv_op.register_fake
def _(xb, c, kb, group):
    return torch.empty_like(xb)


def _sc_setup_context(ctx, inputs, output):
    xb, c, kb, group = inputs
    ctx.save_for_backward(xb, c, kb)
    ctx.group = group


# The backward must be an opaque op too, not just the forward: AOTAutograd traces
# the function registered by register_autograd into the backward graph, and inductor
# then hits the raw triton launch with FakeTensors ("Cannot access data pointer").
@torch.library.custom_op("deepspec_dspark::short_conv_bwd", mutates_args=())
def _short_conv_bwd_op(
    gy: torch.Tensor, xb: torch.Tensor, c: torch.Tensor, kb: torch.Tensor, group: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _bwd_cuda(gy.contiguous(), xb.contiguous(), c.contiguous(),
                     kb.contiguous(), group)


@_short_conv_bwd_op.register_fake
def _(gy, xb, c, kb, group):
    return torch.empty_like(xb), torch.empty_like(c), torch.empty_like(kb)


def _sc_backward(ctx, grad):
    xb, c, kb = ctx.saved_tensors
    gx, gc, gkb = torch.ops.deepspec_dspark.short_conv_bwd(grad, xb, c, kb, ctx.group)
    return gx, gc, gkb, None


torch.library.register_autograd(
    "deepspec_dspark::short_conv", _sc_backward, setup_context=_sc_setup_context
)


def short_conv_apply(
    xb: torch.Tensor, c: torch.Tensor, kb: torch.Tensor, group: int
) -> torch.Tensor:
    """Dispatch on device: fused triton on CUDA, the reference elsewhere.

    `.is_cuda` is a static property of the traced graph, so this branch does not
    cost a guard under torch.compile.
    """
    if _HAS_TRITON and xb.is_cuda:
        return torch.ops.deepspec_dspark.short_conv(xb, c, kb, group)
    return short_conv_reference(xb, c, kb, group)
