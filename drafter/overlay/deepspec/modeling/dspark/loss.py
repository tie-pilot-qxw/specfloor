"""DSpark training objective: slot-weighted CE + full-vocabulary L1 (TV) + confidence BCE.

Relative to upstream DeepSpec this adds three things used by the paper's drafters:

  * global denominators from one early, asynchronous all_reduce of the per-slot
    supervised-token counts (`launch_slot_count_reduction`), instead of blocking
    reductions inside every micro-batch;
  * window normalisation (`window_denominator`): the denominator is applied once per
    optimizer step rather than per micro-batch (used by the final 10-epoch run);
  * an optional compiled L1 term (`compile_l1`) that avoids materialising two fp32
    probability tensors.  Compiled and eager agree to fp32 reduction order only.

It also logs greedy and per-slot acceptance diagnostics.  The candidate-nomination
term lives in `nomination.py` and is added by the trainer.
"""

import os
from typing import Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F

from deepspec.utils.metrics import add_metric
from .common import DSparkForwardOutput


_DENOMINATOR_KEYS = ("ce_loss_den", "l1_loss_den", "confidence_loss_den")


def _all_reduce_loss_denominators(
    loss_terms: dict[str, torch.Tensor],
    *,
    world_size: int,
    prefetch=None,
    slot_w: Optional[torch.Tensor] = None,
    mask_sum_keys: Sequence[str] = (),
) -> dict[str, torch.Tensor]:
    """Global loss denominators.

    With `prefetch` (the early per-slot count reduction), every denominator listed in
    `mask_sum_keys` equals sum(loss_weight_mask) = sum_k w_k n_k and is read off the
    global counts; the others are zero.  Without it, one packed blocking all_reduce.
    """
    keys = _DENOMINATOR_KEYS
    if prefetch is not None:
        handle, counts = prefetch
        handle.wait()
        den = (counts * slot_w.to(counts.dtype)).sum()
        zero = den.new_zeros(())
        return {k: (den if k in mask_sum_keys else zero) for k in keys}
    packed = torch.stack([loss_terms[key].detach() for key in keys])
    if world_size > 1:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    return {key: packed[idx] for idx, key in enumerate(keys)}


def _build_loss_weight_mask(
    *,
    eval_mask: torch.Tensor,
    block_size: int,
    device: torch.device,
    loss_decay_gamma: Optional[float],
) -> torch.Tensor:
    loss_weight_mask = eval_mask.to(torch.float32)
    if loss_decay_gamma is not None and loss_decay_gamma > 0:
        positions = torch.arange(block_size, device=device).view(1, 1, -1)
        decay_weights = torch.exp(-positions.float() / float(loss_decay_gamma))
        loss_weight_mask = loss_weight_mask * decay_weights
    return loss_weight_mask


def slot_depth_weights(
    *,
    block_size: int,
    device: torch.device,
    loss_decay_gamma,
) -> torch.Tensor:
    """The per-slot factor w_k that `_build_loss_weight_mask` multiplies eval_mask by."""
    if loss_decay_gamma is not None and loss_decay_gamma > 0:
        k = torch.arange(block_size, device=device, dtype=torch.float32)
        return torch.exp(-k / float(loss_decay_gamma))
    return torch.ones(block_size, device=device, dtype=torch.float32)


_WINDOW_NORM = False
_ASYNC_DENOM = True


def set_async_denominator(enabled: bool) -> None:
    """Launch the per-slot count all_reduce early inside the forward (default) or not.

    Off falls back to one blocking reduction inside the loss; the global counts are
    the same, only the timing differs.
    """
    global _ASYNC_DENOM
    _ASYNC_DENOM = bool(enabled)


def set_window_normalization(enabled: bool) -> None:
    """Window mode has no per-micro-batch collective, so the early launch is skipped."""
    global _WINDOW_NORM
    _WINDOW_NORM = bool(enabled)


def launch_slot_count_reduction(eval_mask: torch.Tensor):
    """Start the per-slot supervised-token all_reduce as soon as eval_mask exists."""
    if _WINDOW_NORM or not _ASYNC_DENOM:
        return None
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return None
    counts = eval_mask.to(torch.float32).sum(dim=(0, 1)).contiguous()
    handle = dist.all_reduce(counts, op=dist.ReduceOp.SUM, async_op=True)
    return (handle, counts)


def _compute_local_probabilistic_stats(
    *,
    outputs: DSparkForwardOutput,
    accept_rate_3d: Optional[torch.Tensor],
    valid_block_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, _, block_size, _ = outputs.draft_logits.shape
    device = outputs.draft_logits.device
    pos_accept_sums = torch.zeros(block_size, device=device, dtype=torch.float32)
    if accept_rate_3d is None:
        return outputs.draft_logits.new_zeros((), dtype=torch.float32), pos_accept_sums

    valid_accept_rate = accept_rate_3d * outputs.eval_mask.to(torch.float32)
    pos_accept_sums = valid_accept_rate.sum(dim=(0, 1))
    expected_draft_accepted = valid_accept_rate.cumprod(dim=-1).sum(dim=-1)
    tau_prob_per_block = expected_draft_accepted + 1.0
    tau_prob_sum = (tau_prob_per_block * valid_block_weights).sum()
    return tau_prob_sum, pos_accept_sums


def _compute_greedy_stats(
    *,
    draft_logits: torch.Tensor,
    target_ids: torch.Tensor,
    eval_mask: torch.Tensor,
    valid_block_weights: torch.Tensor,
    aligned_target_logits: Optional[torch.Tensor],
) -> dict:
    """Greedy (temperature-0) teacher-forced acceptance diagnostics.

    acc@i / tau_greedy compare with the reference token; acc_tgt@i / tau_greedy_tgt
    with the target's own argmax.  tau_greedy counts draft slots only (max
    block_size), whereas tau_probabilistic includes the bonus token.
    """
    stats = {}
    eval_f = eval_mask.to(torch.float32)
    stats["pos_counts"] = eval_f.sum(dim=(0, 1))

    gt_logit = draft_logits.gather(-1, target_ids.unsqueeze(-1))
    rank_gt = (draft_logits > gt_logit).sum(dim=-1)     # 0 => argmax is correct
    hit = (rank_gt == 0).to(torch.float32)
    stats["acc_pos"] = (hit * eval_f).sum(dim=(0, 1))
    stats["top5_pos"] = ((rank_gt < 5).to(torch.float32) * eval_f).sum(dim=(0, 1))
    stats["top10_pos"] = ((rank_gt < 10).to(torch.float32) * eval_f).sum(dim=(0, 1))

    # Accept length = leading run of correct slots; masked-out slots do not break it.
    run = torch.where(eval_mask, hit, torch.ones_like(hit)).cumprod(dim=-1) * eval_f
    stats["tau_greedy_sum"] = (run.sum(dim=-1) * valid_block_weights).sum()

    if aligned_target_logits is not None:
        tgt_greedy = aligned_target_logits.argmax(dim=-1)
        draft_greedy = draft_logits.argmax(dim=-1)
        hit_t = (draft_greedy == tgt_greedy).to(torch.float32)
        stats["acc_tgt_pos"] = (hit_t * eval_f).sum(dim=(0, 1))
        run_t = torch.where(
            eval_mask, hit_t, torch.ones_like(hit_t)
        ).cumprod(dim=-1) * eval_f
        stats["tau_greedy_tgt_sum"] = (run_t.sum(dim=-1) * valid_block_weights).sum()
        # How often the reference token is the target's argmax (a data property).
        stats["gt_is_tgt_greedy_pos"] = (
            (tgt_greedy == target_ids).to(torch.float32) * eval_f
        ).sum(dim=(0, 1))
    return stats


def _compute_l1_dist_per_token(
    *,
    outputs: DSparkForwardOutput,
    aligned_target_logits: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """Full-vocabulary L1 (= 2 TV), plus the rank-1 / rank>=2 split of the overlap.

    The buckets are diagnostics weighted like accept_rate@k, so
    ovl1@k + ovlge2@k == accept_rate@k.  Ranks are the target's.  When the forward
    already computed the target's top-K for nomination, its prefix is reused.
    """
    draft_probs = torch.softmax(outputs.draft_logits.float(), dim=-1)
    target_probs = torch.softmax(aligned_target_logits.float(), dim=-1)
    l1 = (draft_probs - target_probs).abs().sum(dim=-1)
    with torch.no_grad():
        _shared = getattr(outputs, "target_top_idx", None)
        if _shared is not None:
            assert _shared.shape[:-1] == aligned_target_logits.shape[:-1], (
                "target_top_idx must be the top-K OF aligned_target_logits: "
                f"{tuple(_shared.shape)} vs {tuple(aligned_target_logits.shape)}"
            )
            assert _shared.size(-1) >= 2, "need at least the target's top-2"
            top_idx = _shared[..., :2]
        else:
            top_idx = aligned_target_logits.detach().topk(2, dim=-1).indices
        p_top = target_probs.detach().gather(-1, top_idx)
        q_top = draft_probs.detach().gather(-1, top_idx)
        m_top = torch.minimum(p_top, q_top)
        ovl_all = (1.0 - 0.5 * l1.detach()).clamp_(0.0, 1.0)
        buckets = {
            "rp1": p_top[..., 0], "rp2": p_top[..., 1],
            "rq1": q_top[..., 0], "rq2": q_top[..., 1],
            "ovl1": m_top[..., 0], "ovlge2": ovl_all - m_top[..., 0],
        }
    return l1, buckets


_COMPILED_L1 = None


def _l1_dispatch(*, outputs, aligned_target_logits, compile_l1: bool):
    """Eager by default.  `compile_l1=True` runs the same function under inductor,
    which removes the two fp32 [B, N, K, V] probability tensors; the result agrees
    with eager to fp32 reduction order, not bit for bit."""
    if not compile_l1:
        return _compute_l1_dist_per_token(
            outputs=outputs, aligned_target_logits=aligned_target_logits
        )
    global _COMPILED_L1
    if _COMPILED_L1 is None:
        _COMPILED_L1 = torch.compile(_compute_l1_dist_per_token, dynamic=False)
    return _COMPILED_L1(
        outputs=outputs, aligned_target_logits=aligned_target_logits
    )


def _compute_local_l1_term(
    *,
    l1_dist_per_token: Optional[torch.Tensor],
    loss_weight_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    zero = loss_weight_mask.new_zeros((), dtype=torch.float32)
    if l1_dist_per_token is None:
        return zero, zero
    l1_loss_num = (l1_dist_per_token * loss_weight_mask).sum()
    l1_loss_den = loss_weight_mask.sum()
    return l1_loss_num, l1_loss_den


def _collect_local_terms(
    *,
    outputs: DSparkForwardOutput,
    loss_decay_gamma: Optional[float],
    l1_loss_alpha: float,
    compile_l1: bool = False,
) -> tuple[dict[str, torch.Tensor], bool]:
    draft_logits = outputs.draft_logits
    target_ids = outputs.target_ids
    eval_mask = outputs.eval_mask
    block_keep_mask = outputs.block_keep_mask
    _, _, block_size, vocab_size = draft_logits.shape
    device = draft_logits.device

    loss_weight_mask = _build_loss_weight_mask(
        eval_mask=eval_mask,
        block_size=block_size,
        device=device,
        loss_decay_gamma=loss_decay_gamma,
    )
    flat_logits = draft_logits.reshape(-1, vocab_size)
    flat_targets = target_ids.reshape(-1)
    flat_weights = loss_weight_mask.reshape(-1)
    loss_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    ce_loss_num = (loss_per_token * flat_weights).sum()
    ce_loss_den = flat_weights.sum()
    aligned_target_logits = outputs.aligned_target_logits
    has_confidence = outputs.confidence_pred is not None
    zero = ce_loss_num.new_zeros(())
    assert l1_loss_alpha <= 0 or aligned_target_logits is not None, (
        "aligned_target_logits is required when l1_loss_alpha > 0."
    )
    assert not has_confidence or aligned_target_logits is not None, (
        "aligned_target_logits is required when confidence head is enabled."
    )

    # The L1 term and the acceptance diagnostics share one full-vocabulary L1.
    l1_dist_per_token = None
    rank_buckets = None
    if aligned_target_logits is not None:
        if l1_loss_alpha > 0:
            l1_dist_per_token, rank_buckets = _l1_dispatch(
                outputs=outputs,
                aligned_target_logits=aligned_target_logits,
                compile_l1=compile_l1,
            )
        else:
            with torch.no_grad():
                l1_dist_per_token, rank_buckets = _l1_dispatch(
                    outputs=outputs,
                    aligned_target_logits=aligned_target_logits,
                    compile_l1=compile_l1,
                )
    accept_rate_3d = None
    if l1_dist_per_token is not None:
        accept_rate_3d = (1.0 - 0.5 * l1_dist_per_token.detach()).clamp_(0.0, 1.0)

    if l1_loss_alpha > 0:
        l1_loss_num, l1_loss_den = _compute_local_l1_term(
            l1_dist_per_token=l1_dist_per_token,
            loss_weight_mask=loss_weight_mask,
        )
    else:
        l1_loss_num = zero
        l1_loss_den = zero

    with torch.no_grad():
        eval_mask_f = eval_mask.to(torch.float32)
        pos_total_counts = eval_mask_f.sum(dim=(0, 1))
        valid_pred_tokens = eval_mask.any(dim=-1)
        valid_blocks = block_keep_mask & valid_pred_tokens
        valid_block_weights = valid_blocks.to(torch.float32)
        accept_block_count = valid_block_weights.sum()
        tau_prob_sum, pos_accept_sums = _compute_local_probabilistic_stats(
            outputs=outputs,
            accept_rate_3d=accept_rate_3d,
            valid_block_weights=valid_block_weights,
        )
        greedy = _compute_greedy_stats(
            draft_logits=draft_logits.detach(),
            target_ids=target_ids,
            eval_mask=eval_mask,
            valid_block_weights=valid_block_weights,
            aligned_target_logits=(
                None if aligned_target_logits is None
                else aligned_target_logits.detach()
            ),
        )
        ce_pos_sums = (
            loss_per_token.detach().reshape_as(eval_mask).float() * eval_mask
        ).sum(dim=(0, 1))

    confidence_loss_num = zero
    confidence_loss_den = zero
    confidence_abs_error_num = zero
    confidence_bias_num = zero
    confidence_cumprod_bias_num = zero
    if has_confidence:
        assert accept_rate_3d is not None
        confidence_targets = accept_rate_3d.detach()
        confidence_errors = F.binary_cross_entropy_with_logits(
            outputs.confidence_pred.float(),
            confidence_targets,
            reduction="none",
        ) * loss_weight_mask
        confidence_loss_num = confidence_errors.sum()
        confidence_loss_den = loss_weight_mask.sum()
        with torch.no_grad():
            confidence_probs = outputs.confidence_pred.float().sigmoid()
            confidence_error = confidence_probs - accept_rate_3d
            confidence_abs_error_num = (
                confidence_error.abs() * loss_weight_mask
            ).sum()
            confidence_bias_num = (confidence_error * loss_weight_mask).sum()
            valid_mask = outputs.eval_mask.to(torch.float32)
            confidence_prefix_probs = (
                confidence_probs * valid_mask
            ).cumprod(dim=-1)
            confidence_prefix_targets = (
                accept_rate_3d * valid_mask
            ).cumprod(dim=-1)
            confidence_cumprod_bias_num = (
                (confidence_prefix_probs - confidence_prefix_targets)
                * loss_weight_mask
            ).sum()

    loss_terms = {
        "ce_loss_num": ce_loss_num,
        "ce_loss_den": ce_loss_den,
        "l1_loss_num": l1_loss_num,
        "l1_loss_den": l1_loss_den,
        "confidence_loss_num": confidence_loss_num,
        "confidence_loss_den": confidence_loss_den,
    }

    # accept_rate@i is the temperature-1 acceptance rate 1 - TV; emitted only when
    # the target distribution was computed.
    has_tv = accept_rate_3d is not None
    for pos_idx in range(block_size):
        if has_tv:
            add_metric(
                f"accept_rate@{pos_idx}",
                pos_accept_sums[pos_idx],
                den=pos_total_counts[pos_idx],
                tag="train",
            )
            add_metric(
                f"tv@{pos_idx}",
                pos_total_counts[pos_idx] - pos_accept_sums[pos_idx],
                den=pos_total_counts[pos_idx],
                tag="train",
            )
            if rank_buckets is not None:
                for _bk, _bv in rank_buckets.items():
                    add_metric(
                        f"{_bk}@{pos_idx}",
                        (_bv[:, :, pos_idx] * eval_mask_f[:, :, pos_idx]).sum(),
                        den=pos_total_counts[pos_idx],
                        tag="train",
                    )
        den_pos = greedy["pos_counts"][pos_idx]
        add_metric(f"acc@{pos_idx}", greedy["acc_pos"][pos_idx], den=den_pos, tag="train")
        add_metric(f"top5@{pos_idx}", greedy["top5_pos"][pos_idx], den=den_pos, tag="train")
        add_metric(f"top10@{pos_idx}", greedy["top10_pos"][pos_idx], den=den_pos, tag="train")
        add_metric(f"ce@{pos_idx}", ce_pos_sums[pos_idx], den=den_pos, tag="train")
        if "acc_tgt_pos" in greedy:
            add_metric(
                f"acc_tgt@{pos_idx}", greedy["acc_tgt_pos"][pos_idx],
                den=den_pos, tag="train",
            )
    if has_tv:
        add_metric(
            "tau_probabilistic",
            tau_prob_sum,
            den=accept_block_count,
            tag="train",
        )
    add_metric(
        "tau_greedy", greedy["tau_greedy_sum"], den=accept_block_count, tag="train"
    )
    if "tau_greedy_tgt_sum" in greedy:
        add_metric(
            "tau_greedy_tgt", greedy["tau_greedy_tgt_sum"],
            den=accept_block_count, tag="train",
        )
        add_metric(
            "gt_is_tgt_greedy", greedy["gt_is_tgt_greedy_pos"].sum(),
            den=pos_total_counts.sum(), tag="train",
        )
    if has_confidence:
        add_metric(
            "confidence_abs_error",
            confidence_abs_error_num,
            den=confidence_loss_den,
            tag="train",
        )
        add_metric(
            "confidence_bias",
            confidence_bias_num,
            den=confidence_loss_den,
            tag="train",
        )
        add_metric(
            "confidence_cumprod_bias",
            confidence_cumprod_bias_num,
            den=confidence_loss_den,
            tag="train",
        )
    return loss_terms, has_confidence


def _build_loss(
    *,
    loss_terms: dict[str, torch.Tensor],
    global_denominators: dict[str, torch.Tensor],
    ce_loss_alpha: float,
    l1_loss_alpha: float,
    confidence_head_alpha: float,
    has_confidence: bool,
    world_size: int,
) -> torch.Tensor:
    ce_loss = loss_terms["ce_loss_num"] / (global_denominators["ce_loss_den"] + 1e-6)
    l1_loss = ce_loss.new_zeros(())
    if global_denominators["l1_loss_den"].item() > 0:
        l1_loss = loss_terms["l1_loss_num"] / (
            global_denominators["l1_loss_den"] + 1e-6
        )
    confidence_loss = ce_loss.new_zeros(())
    if has_confidence:
        confidence_loss = loss_terms["confidence_loss_num"] / (
            global_denominators["confidence_loss_den"] + 1e-6
        )
    # FSDP averages rank gradients, so each local numerator over the GLOBAL
    # denominator is multiplied by world_size.
    return (
        ce_loss_alpha * ce_loss
        + l1_loss_alpha * l1_loss
        + confidence_head_alpha * confidence_loss
    ) * world_size


def compute_dspark_loss(
    *,
    outputs: DSparkForwardOutput,
    loss_decay_gamma: Optional[float],
    ce_loss_alpha: float,
    l1_loss_alpha: float,
    confidence_head_alpha: float,
    compile_l1: bool = False,
    window_denominator: Optional[dict] = None,
):
    """Backward loss for one micro-batch.

    `window_denominator` (a dict carrying "loss_scale") switches to window
    normalisation: the loss is returned as numerator / loss_scale, the LOCAL
    denominator is handed back in window_denominator["den_local"], and the trainer
    divides the accumulated gradient by the global window denominator once per
    optimizer step.  Because the denominator does not depend on the parameters,
    grad(sum_j N_j / D) == sum_j grad(N_j) / D exactly.
    """
    loss_terms, has_confidence = _collect_local_terms(
        compile_l1=bool(compile_l1),
        outputs=outputs,
        loss_decay_gamma=loss_decay_gamma,
        l1_loss_alpha=float(l1_loss_alpha),
    )
    world_size = dist.get_world_size()
    # DSPARK_SYNC_DENOM=1 forces the blocking reduction instead of the prefetch.
    _prefetch = (None if os.environ.get("DSPARK_SYNC_DENOM") == "1"
                 else getattr(outputs, "slot_count_reduction", None))
    _mask_keys = ("ce_loss_den",)
    if float(l1_loss_alpha) > 0:
        _mask_keys += ("l1_loss_den",)
    if outputs.confidence_pred is not None:
        _mask_keys += ("confidence_loss_den",)
    if window_denominator is not None:
        den_local = loss_terms["ce_loss_den"].detach()
        window_denominator["den_local"] = den_local
        const = den_local.new_full((), float(window_denominator["loss_scale"]))
        zero = den_local.new_zeros(())
        global_denominators = {
            k: (const if k in _mask_keys else zero) for k in _DENOMINATOR_KEYS
        }
    else:
        global_denominators = _all_reduce_loss_denominators(
            loss_terms,
            world_size=world_size,
            prefetch=_prefetch,
            slot_w=slot_depth_weights(
                block_size=int(outputs.eval_mask.shape[-1]),
                device=outputs.eval_mask.device,
                loss_decay_gamma=loss_decay_gamma,
            ),
            mask_sum_keys=_mask_keys,
        )
    ce_loss_alpha = float(ce_loss_alpha)
    l1_loss_alpha = float(l1_loss_alpha)
    confidence_head_alpha = float(confidence_head_alpha)

    local_ce_loss = loss_terms["ce_loss_num"] / (loss_terms["ce_loss_den"] + 1e-6)
    local_l1_loss = local_ce_loss.new_zeros(())
    if global_denominators["l1_loss_den"].item() > 0:
        local_l1_loss = loss_terms["l1_loss_num"] / (
            loss_terms["l1_loss_den"] + 1e-6
        )
    local_confidence_loss = local_ce_loss.new_zeros(())
    if has_confidence:
        local_confidence_loss = loss_terms["confidence_loss_num"] / (
            loss_terms["confidence_loss_den"] + 1e-6
        )
    local_loss = (
        ce_loss_alpha * local_ce_loss
        + l1_loss_alpha * local_l1_loss
        + confidence_head_alpha * local_confidence_loss
    )

    add_metric(
        "ce_loss",
        loss_terms["ce_loss_num"],
        den=loss_terms["ce_loss_den"],
        tag="train",
    )
    if global_denominators["l1_loss_den"].item() > 0:
        add_metric(
            "l1_loss",
            loss_terms["l1_loss_num"],
            den=loss_terms["l1_loss_den"],
            tag="train",
        )
    if has_confidence:
        add_metric(
            "confidence_loss",
            loss_terms["confidence_loss_num"],
            den=loss_terms["confidence_loss_den"],
            tag="train",
        )
    add_metric(
        "loss",
        local_loss,
        reduction="mean",
        tag="train",
    )
    return _build_loss(
        loss_terms=loss_terms,
        global_denominators=global_denominators,
        ce_loss_alpha=ce_loss_alpha,
        l1_loss_alpha=l1_loss_alpha,
        confidence_head_alpha=confidence_head_alpha,
        has_confidence=has_confidence,
        world_size=world_size,
    )


__all__ = [
    "compute_dspark_loss",
    "launch_slot_count_reduction",
    "set_async_denominator",
    "set_window_normalization",
    "slot_depth_weights",
]
