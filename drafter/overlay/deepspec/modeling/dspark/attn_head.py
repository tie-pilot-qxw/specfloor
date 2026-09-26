"""The prefix-attention order-1 head (`markov_head_type="attn"`).

DSpark's vanilla head adds a low-rank predecessor correction to the base logits,

    logits_k(X, a) = u_k(X) + W2 W1[a],

which is additively separable in the prefix X and the predecessor a: changing a
shifts the relative preference between two successors by the same amount in every
context.  This head lets that shift depend on the prefix.  The predecessor forms a
query together with the drafter state, the query re-reads the committed prefix
through cross-attention, and the result is projected onto the vocabulary:

    x      = W_in [RMSNorm_e(E(a)); RMSNorm_h(h_k(X))]
    x      = x + W_O MHA(RMSNorm_a(x), RMSNorm_c(H_X), RMSNorm_c(H_X))
    x      = x + SwiGLU(RMSNorm_f(x))
    logits = u_k(X) + W2 RMSNorm_o(x)

E is the frozen target embedding table (borrowed, not copied), h_k is the
drafter's final hidden state at slot k, and H_X are the fused target features of
the committed prefix (the same features the drafter's own layers read).  The head
still observes only the prefix and one predecessor, so its information floor is
the order-1 floor T^(1).

Design notes, all reflected in the paper's appendix:
  * There is no W1: the predecessor code is the target embedding.  Dropping W1 and
    doubling W2's rank keeps the vocabulary-facing parameter count of a rank-256
    vanilla head (512 * V == 2 * 256 * V).
  * The two input streams are normalised separately; one RMSNorm over the
    concatenation would let the drafter state drown out the predecessor.
  * Predecessor hypotheses share the projected prefix keys/values and never attend
    to one another, so rows for mutually exclusive predecessors cannot mix.
  * Nothing is zero-initialised.  `out_norm.weight` sets the initial output scale
    (0.35 in the final configuration, `markov_out_scale`).

Two options exist only so the one-epoch component comparison can be rebuilt:
  * `gate_mode="state_output"`: a per-head sigmoid gate on the attention output.
    It stayed near its open initialisation and the final head omits it.
  * `anchor_kv=True`: one extra key/value column per block built from the anchor
    token embedding.  Masking it out of a trained head did not change acceptance,
    and it was where a longer run diverged, so the final head omits it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


_NO_CONTEXT_MSG = (
    "AttnHead.{caller} was reached without an AttnHeadContext.  This head reads the "
    "committed prefix, so it cannot be evaluated from (prev_token, hidden) alone: pass "
    "context=model.markov_head.build_context(...) or use the teacher-forced path."
)


@dataclass
class AttnHeadContext:
    """Per-forward inputs that a per-token head does not need.

    `key_states`/`value_states` are the projected prefix, [b, nq, S, hd], shared by
    every block; the mask decides which columns each block may read.
    """

    key_states: torch.Tensor            # [b, nq, S, hd]
    value_states: torch.Tensor          # [b, nq, S, hd]
    # None unless the head is built with anchor_kv=True.
    anchor_key: Optional[torch.Tensor]  # [b, nb, nq, hd]
    anchor_value: Optional[torch.Tensor]
    context_mask: torch.Tensor          # flex BlockMask, dense mask, or None
    num_blocks: int


class AttnHead(nn.Module):
    """The prefix-attention head.  See the module docstring."""

    markov_head_type = "attn"

    def __init__(self, *, vocab_size: int, markov_rank: int, hidden_size: int,
                 num_heads: int = 4, head_dim: int = 128, mlp_hidden: int = 2048,
                 gate_mode: str = "state_output", out_scale: float = 0.0,
                 rms_norm_eps: float = 1e-6, anchor_kv: bool = True) -> None:
        super().__init__()
        assert gate_mode in ("none", "state_output"), gate_mode
        assert out_scale >= 0.0, out_scale
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.gate_mode = str(gate_mode)
        # 0.0 selects the vanilla-matched initial scale in `reset_new_params`.
        self.out_scale = float(out_scale)
        self.use_anchor_kv = bool(anchor_kv)
        att = self.num_heads * self.head_dim
        # Construction order is load-bearing: it fixes the RNG stream at init and the
        # parameter order seen by the optimizer and FSDP.
        self.embed_norm = nn.RMSNorm(self.hidden_size, eps=float(rms_norm_eps))
        self.state_norm = nn.RMSNorm(self.hidden_size, eps=float(rms_norm_eps))
        self.in_proj = nn.Linear(2 * self.hidden_size, self.markov_rank, bias=False)
        self.attn_norm = nn.RMSNorm(self.markov_rank, eps=float(rms_norm_eps))
        self.q_proj = nn.Linear(self.markov_rank, att, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=float(rms_norm_eps))
        self.k_proj = nn.Linear(self.hidden_size, att, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, att, bias=False)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=float(rms_norm_eps))
        # Marks E(anchor) as a different kind of input from a target feature.
        self.anchor_type = (
            nn.Parameter(torch.zeros(self.hidden_size)) if self.use_anchor_kv else None)
        self.anchor_norm = (
            nn.RMSNorm(self.hidden_size, eps=float(rms_norm_eps))
            if self.use_anchor_kv else None)
        self.ctx_norm = nn.RMSNorm(self.hidden_size, eps=float(rms_norm_eps))
        if self.gate_mode == "state_output":
            self.gate_proj = nn.Linear(self.markov_rank + att, self.num_heads, bias=True)
        self.o_proj = nn.Linear(att, self.markov_rank, bias=False)
        self.mlp_norm = nn.RMSNorm(self.markov_rank, eps=float(rms_norm_eps))
        self.gate_p = nn.Linear(self.markov_rank, int(mlp_hidden), bias=False)
        self.up = nn.Linear(self.markov_rank, int(mlp_hidden), bias=False)
        self.down = nn.Linear(int(mlp_hidden), self.markov_rank, bias=False)
        self.out_norm = nn.RMSNorm(self.markov_rank, eps=float(rms_norm_eps))
        # The vocabulary readout W2.
        self.markov_w2 = nn.Linear(self.markov_rank, self.vocab_size, bias=False)
        # Borrowed by reference, never owned: see set_embedding_source.
        self._embed: Optional[nn.Embedding] = None
        self.reset_new_params()

    # ---- construction -----------------------------------------------------------
    def set_embedding_source(self, embed: nn.Embedding) -> None:
        """Borrow the drafter's (frozen, target-copied) embedding table by reference.

        `object.__setattr__` keeps it out of `_modules`, so it is not saved in this
        head's state_dict, not wrapped again by FSDP, and not counted twice.
        """
        object.__setattr__(self, "_embed", embed)

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        assert self._embed is not None, (
            "AttnHead needs the target's embedding table; call "
            "set_embedding_source() during model construction."
        )
        return self._embed(token_ids.long()).detach()

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """The predecessor's contribution to the head's residual stream.

        Used as the predecessor feature of the confidence head: the E(prev) half of
        `in_proj` applied to the normalised embedding.  Adds no parameters.
        """
        e = self.embed_norm(self.embed_tokens(token_ids))
        w = self.in_proj.weight[:, : self.hidden_size]
        return F.linear(e.to(w.dtype), w)

    def reset_new_params(self) -> None:
        """Set the initial output scale, the anchor type code and the gate bias.

        HF `post_init` runs after the constructor and would undo this, so the model
        calls it again once construction is finished.
        """
        with torch.no_grad():
            if self.out_scale > 0.0:
                self.out_norm.weight.fill_(self.out_scale)
            else:
                # Match the magnitude of a rank-256 vanilla head seeded at
                # std 0.0602 / 0.0664: sqrt(rank) * w * std(W2) == sqrt(256) * 0.0602 * std(W2).
                target = (256 ** 0.5) * 0.0602
                self.out_norm.weight.fill_(target / (self.markov_rank ** 0.5))
            if self.anchor_type is not None:
                self.anchor_type.zero_()
            if self.gate_mode != "none":
                # Start the gate open.
                self.gate_proj.weight.zero_()
                nn.init.constant_(self.gate_proj.bias, 4.0)

    @staticmethod
    def _project_table(table: torch.Tensor, rank: int, std: float,
                       generator: torch.Generator = None) -> torch.Tensor:
        """Row-normalise, randomly project to `rank`, rescale to `std`."""
        e = table.detach().float()
        e = e / e.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        proj = torch.randn(e.shape[1], rank, device=e.device, dtype=torch.float32,
                           generator=generator)
        proj /= proj.norm(dim=0, keepdim=True).clamp_min(1e-9)
        out = e @ proj
        return out / out.std().clamp_min(1e-9) * float(std)

    def seed_codebooks(self, *, embed_weight: torch.Tensor, lm_head_weight: torch.Tensor,
                       std_pred: float, std_succ: float,
                       generator: torch.Generator = None) -> None:
        """Seed W2 by a random projection of the target's lm_head.

        There is no W1 to seed; `embed_weight` and `std_pred` are accepted so the
        model's seeding call does not need to branch on the head type.
        """
        del embed_weight, std_pred
        with torch.no_grad():
            b = self._project_table(lm_head_weight, self.markov_rank, std_succ, generator)
            self.markov_w2.weight.copy_(b.to(self.markov_w2.weight.dtype))

    # ---- the layer --------------------------------------------------------------
    def project_bias(self, latent_states: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(latent_states)

    def context_kv(self, target_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project the committed prefix into this layer's K/V: [b, S, H] -> [b, nq, S, hd]."""
        b, s, _ = target_hidden.shape
        t = self.ctx_norm(target_hidden)
        k = self.k_norm(self.k_proj(t).view(b, s, self.num_heads, self.head_dim))
        v = self.v_proj(t).view(b, s, self.num_heads, self.head_dim)
        return k.transpose(1, 2), v.transpose(1, 2)

    def anchor_kv(self, anchor_token_ids: torch.Tensor):
        """One key/value per block from the anchor token: [b, nb] -> [b, nb, nq, hd]."""
        if not self.use_anchor_kv:
            return None, None
        e = self.anchor_norm(self.embed_tokens(anchor_token_ids)) + self.anchor_type
        b, nb, _ = e.shape
        k = self.k_norm(self.k_proj(e).view(b, nb, self.num_heads, self.head_dim))
        v = self.v_proj(e).view(b, nb, self.num_heads, self.head_dim)
        return k, v

    def build_context(self, *, target_hidden: torch.Tensor, anchor_token_ids: torch.Tensor,
                      context_mask=None, anchor_positions: Optional[torch.Tensor] = None,
                      block_keep_mask: Optional[torch.Tensor] = None,
                      block_size: int = 0) -> AttnHeadContext:
        """Project the prefix (and anchors, if used) and build the attention mask.

        During training every block reads the prefix strictly before its anchor, the
        same visibility rule as the drafter's own layers.  The mask is a flex
        BlockMask; pass `context_mask` directly (dense or None) only for a single
        block, as in tests and serving.
        """
        k, v = self.context_kv(target_hidden)
        ak, av = self.anchor_kv(anchor_token_ids)
        nb = int(anchor_token_ids.shape[1])
        if context_mask is None and anchor_positions is not None:
            from torch.nn.attention.flex_attention import create_block_mask
            seq_len = int(target_hidden.shape[1])
            bs_ = int(anchor_positions.shape[0])

            use_anchor = ak is not None

            def mask_mod(b, h, q_idx, kv_idx):
                del h
                qb = q_idx // block_size
                anchor_pos = anchor_positions[b, qb]
                ctx_ok = kv_idx < anchor_pos
                if not use_anchor:
                    return ctx_ok & block_keep_mask[b, qb]
                # anchor column seq_len + j is visible to block j only
                anchor_ok = (kv_idx >= seq_len) & ((kv_idx - seq_len) == qb)
                return (ctx_ok | anchor_ok) & block_keep_mask[b, qb]

            context_mask = create_block_mask(
                mask_mod, B=bs_, H=None, Q_LEN=nb * block_size,
                KV_LEN=seq_len + (nb if use_anchor else 0),
                device=target_hidden.device)
        return AttnHeadContext(key_states=k, value_states=v, anchor_key=ak,
                               anchor_value=av, context_mask=context_mask,
                               num_blocks=nb)

    def _attend(self, q: torch.Tensor, ctx: AttnHeadContext, block_size: int
                ) -> torch.Tensor:
        """q [b, nq, nb*K, hd] -> [b, nb*K, nq*hd]."""
        b, nq, n_rows, hd = q.shape
        nb = ctx.num_blocks
        use_anchor = ctx.anchor_key is not None
        if use_anchor:
            ak = ctx.anchor_key.permute(0, 2, 1, 3)     # [b, nq, nb, hd]
            av = ctx.anchor_value.permute(0, 2, 1, 3)
            k = torch.cat([ctx.key_states, ak], dim=2)
            v = torch.cat([ctx.value_states, av], dim=2)
        else:
            k, v = ctx.key_states, ctx.value_states

        m = ctx.context_mask
        if m is not None and not torch.is_tensor(m):          # flex BlockMask
            from torch.nn.attention.flex_attention import flex_attention
            o = flex_attention(q, k, v, block_mask=m)
        else:
            if m is not None and use_anchor:
                # Dense path: the caller's mask spans the prefix only, so the
                # block-identity strip for the anchor columns is appended here.
                rows_block = torch.arange(n_rows, device=q.device) // block_size
                strip = rows_block.unsqueeze(-1) == torch.arange(nb, device=q.device)
                while strip.dim() < m.dim():
                    strip = strip.unsqueeze(0)
                strip = strip.expand(*m.shape[:-2], n_rows, nb)
                if m.dtype == torch.bool:
                    m = torch.cat([m, strip], dim=-1)
                else:                                          # additive mask
                    neg = torch.finfo(m.dtype).min
                    m = torch.cat([m, torch.where(strip, 0.0, neg).to(m.dtype)], dim=-1)
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        return o.transpose(1, 2).reshape(b, n_rows, nq * hd)

    def compute_block_latent(self, token_ids: torch.Tensor, hidden_states: torch.Tensor,
                             ctx: AttnHeadContext) -> torch.Tensor:
        """token_ids [b, nb, K], hidden_states [b, nb, K, H] -> latent [b, nb, K, rank]."""
        b, nb, K = token_ids.shape
        e = self.embed_norm(self.embed_tokens(token_ids).to(hidden_states.dtype))
        h = self.state_norm(hidden_states)
        wide = torch.cat([e, h], dim=-1).reshape(b, nb * K, -1)
        x = self.in_proj(wide)                                       # 2H -> rank
        h1 = self.attn_norm(x)
        q = self.q_norm(self.q_proj(h1).view(b, nb * K, self.num_heads, self.head_dim))
        o = self._attend(q.transpose(1, 2), ctx, K)                  # [b, nb*K, nq*hd]
        if self.gate_mode == "state_output":
            g = torch.sigmoid(self.gate_proj(torch.cat([h1, o], dim=-1)))
            o = (o.view(b, nb * K, self.num_heads, self.head_dim)
                 * g.unsqueeze(-1)).reshape(b, nb * K, -1)
        x = x + self.o_proj(o)                                       # attention residual
        h2 = self.mlp_norm(x)
        x = x + self.down(F.silu(self.gate_p(h2)) * self.up(h2))     # SwiGLU residual
        z = self.out_norm(x)
        return z.view(b, nb, K, self.markov_rank)

    # ---- the head interface -----------------------------------------------------
    def compute_step_bias(self, token_ids: torch.Tensor,
                          hidden_states: Optional[torch.Tensor],
                          context: Optional[AttnHeadContext] = None) -> torch.Tensor:
        assert context is not None, _NO_CONTEXT_MSG.format(caller="compute_step_bias")
        assert hidden_states is not None, "AttnHead needs the drafter's hidden state"
        return self.project_bias(self.compute_block_latent(token_ids, hidden_states, context))

    def apply_step_logits(self, logits: torch.Tensor, *, token_ids: torch.Tensor,
                          hidden_states: Optional[torch.Tensor],
                          context: Optional[AttnHeadContext] = None) -> torch.Tensor:
        """One slot's logits; same contract as VanillaMarkov plus the context."""
        assert context is not None, _NO_CONTEXT_MSG.format(caller="apply_step_logits")
        assert hidden_states is not None, "AttnHead needs the drafter's hidden state"
        t = token_ids.reshape(token_ids.shape[0], 1, 1)
        h = hidden_states.reshape(hidden_states.shape[0], 1, 1, -1)
        return logits + self.project_bias(
            self.compute_block_latent(t, h, context)).reshape(logits.shape)

    def sample_block_tokens(self, base_logits: torch.Tensor, *,
                            first_prev_token_ids: torch.Tensor,
                            hidden_states: Optional[torch.Tensor],
                            temperature: float = 0.0,
                            context: Optional[AttnHeadContext] = None):
        """Free-running proposal is served by SGLang (top-16 candidate lattice), which
        keeps the incremental prefix cache this head needs.  It is not implemented in
        the training package."""
        raise NotImplementedError(
            _NO_CONTEXT_MSG.format(caller="sample_block_tokens")
            + "  Use the SGLang runtime in serving/ for free-running proposal."
        )

    def apply_block_logits(self, base_logits: torch.Tensor, *, token_ids: torch.Tensor,
                           hidden_states: Optional[torch.Tensor],
                           context: Optional[AttnHeadContext] = None) -> torch.Tensor:
        if base_logits.size(2) == 0:
            return base_logits
        return base_logits + self.compute_step_bias(token_ids, hidden_states, context)
