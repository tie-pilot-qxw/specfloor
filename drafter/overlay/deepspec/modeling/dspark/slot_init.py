"""Initialisation of the learned masked-position (slot) embeddings.

Slots 1..K-1 of a draft block all enter the drafter as the same frozen mask-token
embedding, and Qwen3 has no additive position embedding, so the residual stream
carries no slot identity until attention mixes RoPE-dependent values in.  The slot
embedding s_k is added to those slots.  What the first RMSNorm sees is E(mask) + s_k,
with E(mask) shared, so the initialisation has to keep those vectors apart:
`depth_axis_init_` mixes a symmetric ramp along one shared direction with an
orthogonal random component (depth_frac 0.4, std 0.02 gives a worst-pair cosine of
about 0.52 on Qwen3-4B), and `max_pairwise_cosine` checks the result.
"""

import torch


def depth_axis_init_(
    tensor: torch.Tensor,
    *,
    depth_frac: float,
    std: float,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """In-place ramp-plus-random initialisation of a [K, d] tensor.

    Rows are rescaled to the norm a plain `normal_(0, std)` draw would have
    (std * sqrt(d)), so only the angular structure differs.  depth_frac == 0 falls
    through to `normal_(0, std)`.
    """
    assert tensor.dim() == 2, f"expected [K, d], got {tuple(tensor.shape)}"
    assert 0.0 <= depth_frac <= 1.0, depth_frac
    K, d = tensor.shape
    if depth_frac == 0.0:
        return tensor.normal_(mean=0.0, std=std, generator=generator)

    dev, dt = tensor.device, tensor.dtype
    scale = std * (d ** 0.5)

    u = torch.randn(d, generator=generator, device=dev, dtype=torch.float32)
    u = u / u.norm()
    # Symmetric ramp through zero, normalised to unit mean row energy.
    a = torch.linspace(-1.0, 1.0, K, device=dev, dtype=torch.float32)
    a = a / a.abs().max().clamp_min(1e-9)
    a = a / a.pow(2).mean().sqrt()

    r = torch.randn(K, d, generator=generator, device=dev, dtype=torch.float32)
    r = r - (r @ u).unsqueeze(1) * u.unsqueeze(0)       # strictly orthogonal to u
    r = r / r.norm(dim=-1, keepdim=True).clamp_min(1e-9)

    mixed = (depth_frac ** 0.5) * a.unsqueeze(1) * u.unsqueeze(0) \
          + ((1.0 - depth_frac) ** 0.5) * r
    mixed = mixed / mixed.norm(dim=-1, keepdim=True).clamp_min(1e-9) * scale
    with torch.no_grad():
        tensor.copy_(mixed.to(dt))
    return tensor


def max_pairwise_cosine(tensor: torch.Tensor, offset: torch.Tensor = None) -> float:
    """Largest cosine between two rows, optionally after adding a shared offset."""
    x = tensor.float()
    if offset is not None:
        x = x + offset.float().unsqueeze(0)
    x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    c = x @ x.T
    K = x.shape[0]
    off = c[~torch.eye(K, dtype=torch.bool, device=c.device)]
    return float(off.max())


__all__ = ["depth_axis_init_", "max_pairwise_cosine"]
