"""A prefill-only, frozen Qwen3 backbone for the online teacher forward (`fused_target`).

A throughput option used by the final 10-epoch run.  Relative to the HF forward it
changes only kernels and weight layout, not the computation:

  merged QKV        one GEMM instead of three.
  merged gate/up    one GEMM instead of two.
  sgl_kernel        fused_qk_norm_rope, fused_add_rmsnorm, silu_and_mul.
  FA3               flash_attn_varlen_func instead of SDPA.

It is not bit-identical to HF (merged GEMMs change the reduction order).  The output
contract matches `OnlineTargetTrainer._online_target_features`: the tapped hiddens are
the outputs of `layers[lid]`, and the second value is the final hidden state AFTER
`model.norm` (what `Qwen3Model.forward()[0]` returns).  Requires `sgl_kernel`.
"""

from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn

import sgl_kernel
from sgl_kernel.flash_attn import flash_attn_varlen_func


class FusedQwen3Target(nn.Module):
    def __init__(self, hf_model, tap_layer_ids: Sequence[int]):
        super().__init__()
        cfg = hf_model.config
        body = hf_model.model
        self.cfg = cfg
        self.nq = int(cfg.num_attention_heads)
        self.nkv = int(cfg.num_key_value_heads)
        self.hd = int(cfg.head_dim)
        self.eps = float(cfg.rms_norm_eps)
        rs = getattr(cfg, "rope_scaling", None) or {}
        self.rope_theta = float(rs.get("rope_theta", getattr(cfg, "rope_theta", 1e6)))
        assert rs.get("rope_type", "default") == "default", (
            f"fused target implements plain RoPE only; config asks for "
            f"{rs.get('rope_type')}, whose scaling parameters this path does not pass."
        )
        self.tap_layer_ids: List[int] = list(tap_layer_ids)

        self.embed_tokens = body.embed_tokens
        self.final_norm_w = body.norm.weight
        self.layers = nn.ModuleList()
        for lyr in body.layers:
            a, m = lyr.self_attn, lyr.mlp
            blk = nn.Module()
            # [q; k; v] on the output axis, the order fused_qk_norm_rope expects.
            blk.w_qkv = nn.Parameter(
                torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight], dim=0),
                requires_grad=False,
            )
            blk.w_o = a.o_proj.weight
            blk.w_gate_up = nn.Parameter(
                torch.cat([m.gate_proj.weight, m.up_proj.weight], dim=0),
                requires_grad=False,
            )
            blk.w_down = m.down_proj.weight
            blk.q_norm_w = a.q_norm.weight
            blk.k_norm_w = a.k_norm.weight
            blk.in_norm_w = lyr.input_layernorm.weight
            blk.post_norm_w = lyr.post_attention_layernorm.weight
            self.layers.append(blk)
        self.requires_grad_(False)

    @torch.no_grad()
    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert input_ids.shape[0] == 1, (
            "fused target is the local_batch_size == 1 path: a single unpadded sequence "
            "is what lets it skip the mask entirely. Batch > 1 needs cu_seqlens over "
            "real lengths, which the padded collator does not carry."
        )
        S = int(input_ids.shape[1])
        dev = input_ids.device
        h = self.embed_tokens(input_ids).view(S, -1)
        pos = torch.arange(S, device=dev, dtype=torch.int32)
        cu = torch.tensor([0, S], device=dev, dtype=torch.int32)

        taps = {}
        if -1 in self.tap_layer_ids:
            taps[-1] = h.view(1, S, -1).clone()

        # NO FP32 RESIDUAL STREAM.  It was tried and it is the wrong shape of fix: HF
        # carries the residual in BF16 and the drafter was trained on what that produces,
        # so a more accurate teacher is as much a deviation as a less accurate one.  The
        # target is to reproduce this teacher, not to improve it.
        residual = None
        for i, blk in enumerate(self.layers):
            if residual is None:
                residual = h
                x = sgl_kernel.rmsnorm(h, blk.in_norm_w, self.eps)
            else:
                sgl_kernel.fused_add_rmsnorm(h, residual, blk.in_norm_w, self.eps)
                x = h
                # THE TAP IS TAKEN HERE, ONE ITERATION LATE, ON PURPOSE.  Reading it at
                # the end of the producing layer means writing out `mlp_out + residual`,
                # which silently assumes what fused_add_rmsnorm does to its two
                # arguments.  That assumption was wrong and cost a 163x error at a single
                # position of one row -- invisible in the logits, because the forward
                # itself was correct, and invisible to any check that only looks at
                # lm_head.  After the call above, `residual` IS the previous layer's
                # output by construction, whatever the kernel's convention happens to be.
                if (i - 1) in self.tap_layer_ids:
                    taps[i - 1] = residual.view(1, S, -1).clone()

            qkv = x @ blk.w_qkv.t()
            # q_norm and k_norm are per-head over head_dim, then RoPE -- one kernel,
            # in place on qkv. v is left untouched, which is why it must sit last.
            sgl_kernel.fused_qk_norm_rope(
                qkv, self.nq, self.nkv, self.nkv, self.hd, self.eps,
                blk.q_norm_w, blk.k_norm_w, self.rope_theta, True, pos,
                1.0, 0.0, 0.0, 1.0,
            )
            nq, nkv, hd = self.nq, self.nkv, self.hd
            q = qkv[:, : nq * hd].view(S, nq, hd)
            k = qkv[:, nq * hd : (nq + nkv) * hd].view(S, nkv, hd)
            v = qkv[:, (nq + nkv) * hd :].view(S, nkv, hd)
            o = flash_attn_varlen_func(
                q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
                max_seqlen_q=S, max_seqlen_k=S, causal=True,
            )
            attn = o.reshape(S, nq * hd) @ blk.w_o.t()

            sgl_kernel.fused_add_rmsnorm(attn, residual, blk.post_norm_w, self.eps)
            gate_up = attn @ blk.w_gate_up.t()
            h = sgl_kernel.silu_and_mul(gate_up) @ blk.w_down.t()


        sgl_kernel.fused_add_rmsnorm(h, residual, self.final_norm_w, self.eps)
        # Same trick for a tap on the LAST layer: the final norm call leaves the last
        # layer's output in `residual`.
        last = len(self.layers) - 1
        if last in self.tap_layer_ids:
            taps[last] = residual.view(1, S, -1).clone()
        last_hidden = h.view(1, S, -1)
        tapped = torch.cat([taps[lid] for lid in self.tap_layer_ids], dim=-1)
        return tapped, last_hidden
