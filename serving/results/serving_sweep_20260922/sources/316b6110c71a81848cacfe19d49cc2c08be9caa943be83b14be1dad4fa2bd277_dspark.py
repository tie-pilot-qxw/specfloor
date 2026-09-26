from __future__ import annotations

import logging
from typing import Callable, Iterable, Optional, Tuple

import msgspec
import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.distributed.communication_op import tensor_model_parallel_all_gather
from sglang.srt.environ import envs
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import should_apply_lm_head_quant_method
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.dflash import DFlashDraftModel
from sglang.srt.speculative.dflash_utils import can_dflash_slice_qkv_weight
from sglang.srt.speculative.dspark_components.dspark_config import (
    get_dspark_sample_from_anchor,
    parse_dspark_draft_config,
)
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyMode,
    read_ragged_verify_mode,
)

logger = logging.getLogger(__name__)

StepSampler = Callable[[torch.Tensor, int], torch.Tensor]

_ATTN_HEAD_NO_CONTEXT = (
    "AttnHead reads the committed prefix, so unlike every table-shaped head it cannot "
    "be evaluated from (prev_token, hidden) alone. The caller is on a route that has "
    "not been wired: pass context=model.markov_head.build_serving_context(...)."
)


def gather_and_crop_vocab(
    local_logits: torch.Tensor, lm_head: nn.Module
) -> torch.Tensor:
    full_logits = tensor_model_parallel_all_gather(local_logits, dim=-1)
    return full_logits[..., : int(lm_head.org_vocab_size)]


def project_through_lm_head(hidden: torch.Tensor, lm_head: nn.Module) -> torch.Tensor:
    """Project draft hidden states through the target head; a quantized head
    stores `weight` packed, so it needs its own kernel instead of a matmul."""
    quant_method = lm_head.quant_method
    if should_apply_lm_head_quant_method(lm_head, quant_method):
        return quant_method.apply(lm_head, hidden, None)
    weight = lm_head.weight
    return torch.matmul(hidden.to(weight.dtype), weight.T)


def run_markov_block(
    head: nn.Module,
    base_logits: torch.Tensor,
    *,
    first_prev_tokens: torch.Tensor,
    hidden_states: Optional[torch.Tensor],
    sampler: StepSampler,
    context=None,
    collect_logits: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """`collect_logits=False` returns None for the block logits.

    The all-greedy accept path compares TOKENS and never reads them, but keeping them
    costs a [bs, gamma, V] tensor plus every per-step [bs, V] slice alive until the cat:
    at V=152k, bs=64, gamma=7, bf16 that is ~136 MB of pure garbage per round.  The
    folded path already skipped this (it sets corrected_logits=None when all_greedy);
    the eager walk -- which is the ONLY path the attention head takes -- did not.
    """
    batch_size, proposal_len = base_logits.shape[:2]
    if proposal_len == 0:
        empty = torch.empty(batch_size, 0, dtype=torch.long, device=base_logits.device)
        return empty, (base_logits if collect_logits else None)

    sampled_tokens = []
    corrected_logits = [] if collect_logits else None
    prev_tokens = first_prev_tokens.long()
    for step_idx in range(proposal_len):
        step_hidden = None if hidden_states is None else hidden_states[:, step_idx, ...]
        step_kwargs = {} if context is None else {"context": context}
        step_logits = head.apply_step_logits(
            base_logits[:, step_idx, :],
            token_ids=prev_tokens,
            hidden_states=step_hidden,
            **step_kwargs,
        )
        next_tokens = sampler(step_logits, step_idx)
        sampled_tokens.append(next_tokens)
        if corrected_logits is not None:
            corrected_logits.append(step_logits.unsqueeze(1))
        prev_tokens = next_tokens
    return (
        torch.stack(sampled_tokens, dim=1),
        None if corrected_logits is None else torch.cat(corrected_logits, dim=1),
    )


class VanillaMarkov(nn.Module):

    markov_head_type = "vanilla"

    def __init__(self, *, vocab_size: int, markov_rank: int) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        if self.markov_rank <= 0:
            raise ValueError(
                f"VanillaMarkov requires markov_rank > 0, got {self.markov_rank}."
            )
        self.markov_w1 = nn.Embedding(self.vocab_size, self.markov_rank)
        self.markov_w2 = nn.Linear(self.markov_rank, self.vocab_size, bias=False)

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids.long())

    def project_bias(self, latent_states: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(latent_states)

    def compute_step_latent(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """The rank-dim state `project_bias` turns into a vocabulary bias.

        Split out so the lattice can score a slot against MANY predecessors without
        projecting each one to the full vocabulary: only the candidate rows of
        markov_w2 are ever needed.  compute_step_bias goes through here so the two
        cannot drift into computing different things.
        """
        del hidden_states
        return self.get_prev_embeddings(token_ids)

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return self.project_bias(self.compute_step_latent(token_ids, hidden_states))

    def apply_step_logits(
        self,
        logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return logits + self.compute_step_bias(token_ids, hidden_states)

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if base_logits.size(-2) == 0:
            return base_logits
        return base_logits + self.compute_step_bias(token_ids, hidden_states)

    def sample_block(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_tokens: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        sampler: StepSampler,
        collect_logits: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return run_markov_block(
            self,
            base_logits,
            first_prev_tokens=first_prev_tokens,
            hidden_states=hidden_states,
            sampler=sampler,
            collect_logits=collect_logits,
        )


class Nemotron35VanillaMarkov(VanillaMarkov):
    """Checkpoint-quantized Markov head used only by Nemotron 3.5 DSpark."""

    def __init__(
        self,
        *,
        vocab_size: int,
        markov_rank: int,
        quant_config,
        prefix: str,
    ) -> None:
        nn.Module.__init__(self)
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        if self.markov_rank <= 0:
            raise ValueError(
                "Nemotron35VanillaMarkov requires markov_rank > 0, "
                f"got {self.markov_rank}."
            )
        self.markov_w1 = nn.Embedding(self.vocab_size, self.markov_rank)
        self.markov_w2 = ReplicatedLinear(
            self.markov_rank,
            self.vocab_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.markov_w2" if prefix else "markov_w2",
        )

    def project_bias(self, latent_states: torch.Tensor) -> torch.Tensor:
        bias, _ = self.markov_w2(latent_states)
        return bias


class GatedMarkovHead(VanillaMarkov):

    markov_head_type = "gated"

    def __init__(self, *, vocab_size: int, markov_rank: int, hidden_size: int) -> None:
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.gate_proj = nn.Linear(int(hidden_size) + markov_rank, markov_rank)

    def compute_gate(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if hidden_states is None:
            raise ValueError("GatedMarkovHead requires hidden_states.")
        prev_embeddings = self.get_prev_embeddings(token_ids)
        gate_inputs = torch.cat([hidden_states, prev_embeddings], dim=-1)
        return torch.sigmoid(self.gate_proj(gate_inputs))

    def compute_step_latent(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        prev_embeddings = self.get_prev_embeddings(token_ids)
        gate = self.compute_gate(token_ids, hidden_states).to(
            dtype=prev_embeddings.dtype
        )
        return gate * prev_embeddings


class CondMarkovHead(VanillaMarkov):
    """VanillaMarkov's bias, conditioned on the drafter's hidden state as well.

        e    = W1[prev]
        c    = state_proj(state_norm(h))
        z    = e + MLP([c, e, c * e])
        bias = W2 z

    Trained as z = e at step 0 (the MLP's output layer starts at zero), so the head's
    null hypothesis is exactly VanillaMarkov and it can only lose by training worse.

    The product enters as an MLP INPUT and is deliberately not part of the residual.
    An earlier head carried `c * e` in the residual instead, which hands step 0 a
    randomly per-channel-rescaled predecessor code; a knockout on a converged drafter
    measured a scrambled predecessor as WORSE than deleting the predecessor path
    outright, and that head's predecessor path ended up worth +0.027 accept length
    where a plain Markov head is worth +1.125.

    Module names match the training definition exactly, because the checkpoint loads by
    name -- a rename here is a silently zero-initialised submodule, not a load error.
    """

    markov_head_type = "cond"

    def __init__(
        self,
        *,
        vocab_size: int,
        markov_rank: int,
        hidden_size: int,
        mlp_hidden: int,
        rms_norm_eps: float,
    ) -> None:
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.hidden_size = int(hidden_size)
        self.mlp_hidden = int(mlp_hidden)
        self.state_norm = RMSNorm(self.hidden_size, eps=float(rms_norm_eps))
        self.state_proj = nn.Linear(self.hidden_size, self.markov_rank, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(3 * self.markov_rank, self.mlp_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(self.mlp_hidden, self.markov_rank, bias=False),
        )

    def compute_step_latent(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if hidden_states is None:
            raise ValueError("CondMarkovHead requires hidden_states.")
        prev_embeddings = self.get_prev_embeddings(token_ids)
        state = self.state_proj(self.state_norm(hidden_states))
        if state.shape != prev_embeddings.shape:
            raise ValueError(
                f"hidden_states and token_ids disagree: state {tuple(state.shape)} "
                f"vs prev {tuple(prev_embeddings.shape)}."
            )
        return prev_embeddings + self.mlp(
            torch.cat([state, prev_embeddings, state * prev_embeddings], dim=-1)
        )


class RNNHead(VanillaMarkov):

    markov_head_type = "rnn"

    def __init__(self, *, vocab_size: int, markov_rank: int, hidden_size: int) -> None:
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.hidden_size = int(hidden_size)
        self.state_size = markov_rank
        self.joint_proj = nn.Linear(2 * markov_rank + self.hidden_size, 3 * markov_rank)

    def _rnn_step(
        self,
        state: torch.Tensor,
        prev_embeddings: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.cat([state, prev_embeddings, hidden_states], dim=-1)
        gate_raw, candidate_raw, output_raw = self.joint_proj(z).chunk(3, dim=-1)
        gate = torch.sigmoid(gate_raw)
        candidate = torch.tanh(candidate_raw)
        new_state = gate * state + (1.0 - gate) * candidate
        bias = self.project_bias(torch.tanh(output_raw))
        return new_state, bias

    def compute_step_latent(self, token_ids, hidden_states):
        raise NotImplementedError(
            "RNNHead carries a recurrent state across slots, so its bias at slot k "
            "depends on the whole path, not on the immediate predecessor. An order-1 "
            "(slot, predecessor) lattice cannot represent it; this head has to use the "
            "sequential walk."
        )

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if hidden_states is None:
            raise ValueError("RNNHead requires hidden_states.")
        prev_embeddings = self.get_prev_embeddings(token_ids)
        state = torch.zeros_like(prev_embeddings)
        _, bias = self._rnn_step(state, prev_embeddings, hidden_states)
        return bias

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if hidden_states is None:
            raise ValueError("RNNHead requires hidden_states.")
        block_size = base_logits.size(-2)
        if block_size == 0:
            return base_logits
        leading_shape = base_logits.shape[:-2]
        state = torch.zeros(
            *leading_shape,
            self.markov_rank,
            device=base_logits.device,
            dtype=hidden_states.dtype,
        )
        output_logits = []
        for k in range(block_size):
            prev_emb = self.get_prev_embeddings(token_ids[..., k])
            state, bias = self._rnn_step(state, prev_emb, hidden_states[..., k, :])
            output_logits.append(base_logits[..., k, :] + bias)
        return torch.stack(output_logits, dim=-2)

    def sample_block(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_tokens: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        sampler: StepSampler,
        collect_logits: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if hidden_states is None:
            raise ValueError("RNNHead requires hidden_states.")
        batch_size, proposal_len = base_logits.shape[:2]
        if proposal_len == 0:
            empty = torch.empty(
                batch_size, 0, dtype=torch.long, device=base_logits.device
            )
            return empty, (base_logits if collect_logits else None)

        state = torch.zeros(
            batch_size,
            self.markov_rank,
            device=base_logits.device,
            dtype=hidden_states.dtype,
        )
        sampled_tokens = []
        corrected_logits = [] if collect_logits else None
        prev_tokens = first_prev_tokens.long()
        for step_idx in range(proposal_len):
            prev_emb = self.get_prev_embeddings(prev_tokens)
            state, bias = self._rnn_step(state, prev_emb, hidden_states[:, step_idx, :])
            step_logits = base_logits[:, step_idx, :] + bias
            next_tokens = sampler(step_logits, step_idx)
            sampled_tokens.append(next_tokens)
            if corrected_logits is not None:
                corrected_logits.append(step_logits.unsqueeze(1))
            prev_tokens = next_tokens
        return (
            torch.stack(sampled_tokens, dim=1),
            None if corrected_logits is None else torch.cat(corrected_logits, dim=1),
        )


class AttnHeadContext(msgspec.Struct):
    """What the attention head needs that a table-shaped head does not.

    The prefix K/V live in a pool of THIS head's own, written from the same fused target
    hidden the drafter's layers read and at the same positions.  It cannot share a
    drafter layer's KV slot: those are 8 kv heads wide and this head is 4, so the slot
    shape does not match.  At 4 heads x 128 it costs 2 KB per token.

    `cache_seqlens` is the committed prefix STRICTLY BEFORE the anchor.  The anchor's own
    target hidden is excluded on purpose -- it is the target's POST-anchor
    representation and decodes to the slot-1 answer, so letting the head read it would
    hand it the token it is trying to predict.  The anchor enters as one extra key built
    from its TOKEN embedding, which leaks nothing.
    """

    # [pages, page_size, heads, head_dim] -- the layout the paged kernel demands.
    # sglang runs page_size = 1, so a "page" is one token slot.
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    # THE ANCHOR IS A REAL KEY IN THE CACHE, not a tensor merged in afterwards.  It is
    # written into this head's own pool at the slot for position prefix_len and the page
    # table points there, so cache_seqlens covers prefix + anchor and ONE paged attention
    # call does the whole thing.  Merging it separately is also exact, but costs a
    # log-sum-exp read and ~6 elementwise kernels on every one of the K steps, for the
    # sake of a single key.
    page_table: torch.Tensor  # [bs, max_len]
    cache_seqlens: torch.Tensor  # [bs] int32, = prefix + 1


class AttnHead(nn.Module):
    """The order-1 head as an ordinary attention block over the committed prefix.

    Every other head here is a bias TABLE: `logits_k = base_k(prefix) + W2 W1[prev]` is
    additively separable in (prefix, prev), so none of them can express "in THIS prefix,
    prev=X means something different".  The cond head broke the separability with an MLP,
    but an MLP can only re-weight a fixed summary of the prefix; it cannot go back and
    re-READ it given the realised predecessor.  This can.

    Serving shape: one query per request per step, over that request's prefix plus one
    anchor key -- decode-shaped, which is why it runs on the ordinary paged attention.

    A TRANSCRIPTION of deepspec AttnHead, module names included, because the checkpoint
    loads by name.  The one thing that is NOT a transcription is the attention itself:
    training materialises the prefix as a dense [b, heads, S, dim] tensor and masks it,
    serving reads a paged cache and merges the single anchor key analytically.  Those two
    have to agree, which is what the parity test asserts.
    """

    markov_head_type = "attn"

    def __init__(
        self,
        *,
        vocab_size: int,
        markov_rank: int,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        mlp_hidden: int,
        gate_mode: str,
        rms_norm_eps: float,
        anchor_kv: bool = True,
    ) -> None:
        super().__init__()
        if gate_mode not in ("none", "state", "state_output"):
            raise ValueError(f"unknown markov_gate_mode {gate_mode!r}")
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.gate_mode = str(gate_mode)
        self.scaling = self.head_dim**-0.5
        inner = self.num_heads * self.head_dim
        # THE NARROWING IS AT THE ENTRANCE, so everything after it is an ordinary
        # transformer block at width markov_rank: both sublayers have residuals and
        # pre-norms.  E(prev) and h_k get a norm EACH -- one norm over the concatenation
        # divides by the RMS of the whole vector and preserves the ratio between the
        # halves, which on Qwen3-4B is 126.6x, leaving the predecessor at 1/16000 of the
        # query's variance and the head effectively blind to the thing it exists to read.
        self.embed_norm = RMSNorm(self.hidden_size, eps=rms_norm_eps)
        self.state_norm = RMSNorm(self.hidden_size, eps=rms_norm_eps)
        self.in_proj = nn.Linear(2 * self.hidden_size, self.markov_rank, bias=False)
        self.attn_norm = RMSNorm(self.markov_rank, eps=rms_norm_eps)
        self.q_proj = nn.Linear(self.markov_rank, inner, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_proj = nn.Linear(self.hidden_size, inner, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, inner, bias=False)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        # markov_anchor_kv=False removes the per-block anchor column.  MEASURED worth
        # nothing on a trained checkpoint (1 block in 768 changes accepted length) and
        # measured to be where the 10-epoch training run diverged.  The two tensors are
        # NOT constructed, so a checkpoint trained without them loads exactly -- built
        # unconditionally they would silently serve an anchor column at its INIT value
        # (anchor_type 0, anchor_norm 1) that the weights were never trained with, and
        # nothing in the loader checks the head's backbone tensors for absence.
        self.use_anchor_kv = bool(anchor_kv)
        self.anchor_type = (
            nn.Parameter(torch.zeros(self.hidden_size)) if self.use_anchor_kv else None)
        self.anchor_norm = (
            RMSNorm(self.hidden_size, eps=rms_norm_eps) if self.use_anchor_kv else None)
        self.ctx_norm = RMSNorm(self.hidden_size, eps=rms_norm_eps)
        if self.gate_mode == "state":
            self.gate_proj = nn.Linear(self.markov_rank, self.num_heads, bias=True)
        elif self.gate_mode == "state_output":
            self.gate_proj = nn.Linear(
                self.markov_rank + inner, self.num_heads, bias=True
            )
        self.o_proj = nn.Linear(inner, self.markov_rank, bias=False)
        self.mlp_norm = RMSNorm(self.markov_rank, eps=rms_norm_eps)
        self.gate_p = nn.Linear(self.markov_rank, int(mlp_hidden), bias=False)
        self.up = nn.Linear(self.markov_rank, int(mlp_hidden), bias=False)
        self.down = nn.Linear(int(mlp_hidden), self.markov_rank, bias=False)
        self.out_norm = RMSNorm(self.markov_rank, eps=rms_norm_eps)
        self._gate_up_activation = SiluAndMul()
        # There is no W1.  The predecessor code IS the target's embedding, at full width
        # and for the cost of a gather, so dropping W1 and doubling W2's rank is exactly
        # parameter-neutral: 256*V*2 == 512*V.
        self.markov_w2 = nn.Linear(self.markov_rank, self.vocab_size, bias=False)
        # Kept off _modules and off the state dict: borrowed, and allocated at runtime.
        self._embed_tokens = None
        self._key_cache = None
        self._value_cache = None
        # Derived inference weights are not checkpoint parameters. Allocate them
        # after the draft weights and the shared target embedding have been loaded.
        self.register_buffer("_projected_embedding", None, persistent=False)
        self.register_buffer("_state_projection_weight", None, persistent=False)
        self.register_buffer("_gate_up_weight", None, persistent=False)
        self.register_buffer("_ctx_kv_weight", None, persistent=False)

    def set_embedding_source(self, embed: nn.Module) -> None:
        """Borrow the target's embedding BY REFERENCE, kept out of _modules so it is
        neither saved nor counted twice."""
        object.__setattr__(self, "_embed_tokens", embed)

    @torch.no_grad()
    def prepare_for_inference(self) -> None:
        """Factor in_proj([norm(E[token]), norm(h)]) before graph capture.

        The token half is constant for the lifetime of the loaded weights. The
        state half is shared by all predecessor candidates of a lattice slot.
        FP32 partials preserve the single output rounding of the original GEMM.
        Reuse existing storage on reload so captured graph pointers remain valid.
        """
        def update_buffer(name, value):
            old = getattr(self, name)
            if (
                old is not None
                and old.shape == value.shape
                and old.device == value.device
                and old.dtype == value.dtype
            ):
                old.copy_(value)
            else:
                setattr(self, name, value)

        # Not gated on SGLANG_DSPARK_ATTN_PRECOMPUTE: that flag is about the token
        # table, this concatenation only makes the context write's two projections
        # one GEMM.
        if self.k_proj.weight.dtype == self.v_proj.weight.dtype:
            update_buffer(
                "_ctx_kv_weight", torch.cat([self.k_proj.weight, self.v_proj.weight])
            )

        weight = self.in_proj.weight
        if (
            not envs.SGLANG_DSPARK_ATTN_PRECOMPUTE.get()
            or weight.device.type != "cuda"
            or weight.dtype not in (torch.bfloat16, torch.float16)
            or self._embed_tokens is None
        ):
            return

        shape = (self.vocab_size, self.markov_rank)
        if (
            self._projected_embedding is None
            or self._projected_embedding.shape != shape
            or self._projected_embedding.device != weight.device
            or self._projected_embedding.dtype != torch.float32
        ):
            self._projected_embedding = torch.empty(
                shape, device=weight.device, dtype=torch.float32
            )
        token_weight = weight[:, : self.hidden_size].contiguous()
        for start in range(0, self.vocab_size, 2048):
            stop = min(start + 2048, self.vocab_size)
            ids = torch.arange(start, stop, device=weight.device)
            normalized = self.embed_norm(self.embed_tokens(ids).to(weight.dtype))
            torch.mm(
                normalized,
                token_weight.T,
                out=self._projected_embedding[start:stop],
                out_dtype=torch.float32,
            )
        update_buffer(
            "_state_projection_weight", weight[:, self.hidden_size :].contiguous()
        )
        update_buffer(
            "_gate_up_weight", torch.cat([self.gate_p.weight, self.up.weight])
        )
        logger.info(
            "DSpark attention head: fixed token projection cached (%.1f MiB); "
            "state projection shared across lattice predecessors.",
            self._projected_embedding.numel()
            * self._projected_embedding.element_size()
            / 2**20,
        )

    def _project_inputs(self, token_ids, hidden_states, predecessors=1):
        from sglang.kernels.ops.speculative.dspark.attn_head import (
            combine_input_projection,
        )

        state = torch.mm(
            self.state_norm(hidden_states),
            self._state_projection_weight.T,
            out_dtype=torch.float32,
        )
        return combine_input_projection(
            token_ids,
            self._projected_embedding,
            state,
            predecessors,
            hidden_states.dtype,
        )

    def compute_lattice_latent(self, token_ids, hidden_states, context):
        bs, slots, top_k = token_ids.shape
        if self._projected_embedding is None:
            hidden = hidden_states[:, :, None, :].expand(bs, slots, top_k, -1)
            return self.compute_step_latent(
                token_ids.reshape(-1),
                hidden.reshape(bs * slots * top_k, -1),
                context,
                queries_per_seq=slots * top_k,
            )
        x = self._project_inputs(
            token_ids.reshape(-1), hidden_states.reshape(bs * slots, -1), top_k
        )
        return self._latent_from_input(x, context, slots * top_k)

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self._embed_tokens is None:
            raise RuntimeError(
                "AttnHead has no embedding source; the model must call "
                "set_embedding_source() after construction."
            )
        return self._embed_tokens(token_ids.long())

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """The confidence head's notion of a predecessor code: in_proj's predecessor
        half applied to the normalised embedding.  No new parameters."""
        if self._projected_embedding is not None:
            return F.embedding(token_ids.long(), self._projected_embedding).to(
                self.in_proj.weight.dtype
            )
        embedded = self.embed_norm(self.embed_tokens(token_ids))
        weight = self.in_proj.weight[:, : self.hidden_size]
        return F.linear(embedded.to(weight.dtype), weight)

    def project_bias(self, latent_states: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(latent_states)

    def context_kv(self, ctx_hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """[tokens, hidden] -> two [tokens, heads, head_dim].  Fed the SAME fused target
        hidden the drafter's own layers are given, at the same positions."""
        normed = self.ctx_norm(ctx_hidden)
        shape = (-1, self.num_heads, self.head_dim)
        return (
            self.k_norm(self.k_proj(normed).view(shape)),
            self.v_proj(normed).view(shape),
        )

    def anchor_kv(self, anchor_token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """One key/value per request, from the anchor TOKEN's embedding."""
        embedded = self.anchor_norm(
            self.embed_tokens(anchor_token_ids)
        ) + self.anchor_type.to(self.anchor_type.dtype)
        shape = (-1, self.num_heads, self.head_dim)
        return (
            self.k_norm(self.k_proj(embedded).view(shape)),
            self.v_proj(embedded).view(shape),
        )

    def freeze_context_cache(self) -> None:
        """Called once the cache's address is baked into a captured graph."""
        self._context_cache_frozen = True

    def ensure_context_cache(self, *, num_slots: int, device, dtype) -> None:
        """Allocate this head's own prefix K/V pool.

        It cannot share a drafter layer's KV slot: those are num_key_value_heads wide
        (8 on Qwen3-4B) and this head has 4, so the slot shape does not match.  A
        dedicated pool at 4 heads x 128 costs 2 KB per token, written from the same fused
        target hidden and at the same positions as the layers' own.
        """
        if self._key_cache is not None and self._key_cache.shape[0] >= num_slots:
            return
        if getattr(self, "_context_cache_frozen", False):
            # A folded lattice graph has these addresses baked in.  Swapping the tensor
            # here would leave the captured kernels reading freed memory, and nothing
            # downstream would notice: the head would attend over whatever now occupies
            # those pages and accept length would move without an error.  The pool size
            # is fixed for the server's life, so this can only fire on a real bug.
            raise RuntimeError(
                f"AttnHead context cache is frozen at {self._key_cache.shape[0]} slots "
                f"but {num_slots} were requested; a captured graph holds its address."
            )
        shape = (int(num_slots), 1, self.num_heads, self.head_dim)
        # ALLOCATED OUTSIDE INFERENCE MODE.  This runs inside the draft forward, which is
        # under torch.inference_mode(), and a tensor born there is an inference tensor
        # that can never be written from normal mode again.  The anchor key is written
        # from propose(), which is normal mode, so the cache has to be an ordinary
        # tensor -- writing an ordinary tensor from inside inference mode is fine, the
        # other direction is not.
        with torch.inference_mode(False):
            object.__setattr__(
                self, "_key_cache", torch.zeros(shape, device=device, dtype=dtype)
            )
            object.__setattr__(
                self, "_value_cache", torch.zeros(shape, device=device, dtype=dtype)
            )

    def _fused_ctx_kv_supported(self, ctx_hidden: torch.Tensor) -> bool:
        w = self._ctx_kv_weight
        tail = (1, self.num_heads, self.head_dim)
        return (
            w is not None
            and w.dtype == torch.bfloat16
            and ctx_hidden.dtype == torch.bfloat16
            and ctx_hidden.device.type == "cuda"
            and self._key_cache.dtype == torch.bfloat16
            and self._value_cache.dtype == torch.bfloat16
            and self._key_cache.is_contiguous()
            and self._value_cache.is_contiguous()
            and tuple(self._key_cache.shape[1:]) == tail
            and tuple(self._value_cache.shape[1:]) == tail
            and self.k_norm.weight.numel() == self.head_dim
        )

    def write_context_kv(
        self,
        *,
        ctx_hidden: torch.Tensor,
        cache_loc: torch.Tensor,
        commit_lens: Optional[torch.Tensor] = None,
        locs_row_width: Optional[int] = None,
    ) -> None:
        """Write this head's prefix K/V for the rows `commit_lens` accepts.

        `cache_loc` is the flattened [bs, locs_row_width] verify window when
        commit_lens is given, and one slot per token otherwise.
        """
        if self._key_cache is None:
            raise RuntimeError("AttnHead context cache was never allocated.")
        if self._fused_ctx_kv_supported(ctx_hidden):
            from sglang.kernels.ops.speculative.dspark.fused_ctx_kv_write import (
                fused_ctx_kv_norm_write,
            )

            fused_ctx_kv_norm_write(
                kv=F.linear(self.ctx_norm(ctx_hidden), self._ctx_kv_weight),
                k_norm_weight=self.k_norm.weight,
                locs=cache_loc,
                k_cache=self._key_cache,
                v_cache=self._value_cache,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                eps=self.k_norm.variance_epsilon,
                commit_lens=commit_lens,
                locs_row_width=locs_row_width,
            )
            return

        key, value = self.context_kv(ctx_hidden)
        loc = cache_loc.to(torch.long)
        if commit_lens is not None:
            # Write EVERY row and send the rejected ones to the pool's dummy slot 0,
            # rather than selecting the accepted rows out.  `ctx_hidden[valid]` is
            # boolean-mask indexing, whose output shape depends on the mask's CONTENTS,
            # so torch reads the mask back to the host before it can allocate: an
            # aten::nonzero and a full device-to-host sync, twice, on every commit.
            # Measured on this path at 5.7 ms of blocked CPU per round.  Free pages
            # start at 1, so slot 0 is where the allocator already sends discarded
            # rows and nothing reads it.
            valid = (
                torch.arange(locs_row_width, device=loc.device)
                < commit_lens.to(torch.long).view(-1, 1)
            ).reshape(-1)
            loc = torch.where(valid, loc, torch.zeros_like(loc))
        self._key_cache[loc, 0] = key.to(self._key_cache.dtype)
        self._value_cache[loc, 0] = value.to(self._value_cache.dtype)

    def build_serving_context(
        self,
        *,
        page_table: torch.Tensor,
        prefix_lens: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        anchor_cache_loc: torch.Tensor,
    ) -> AttnHeadContext:
        """`cache_seqlens` counts the committed prefix STRICTLY BEFORE the anchor.

        At draft time the anchor sits at position seq_len and the context was written for
        [0, seq_len), so seq_lens is already that count -- but it is the caller's job and
        an off-by-one here is invisible: the head would read the target's post-anchor
        representation, which decodes to the slot-1 answer, and accept length would go
        UP for the wrong reason.
        """
        if self._key_cache is None:
            raise RuntimeError("AttnHead context cache was never allocated.")
        if self.use_anchor_kv:
            anchor_key, anchor_value = self.anchor_kv(anchor_token_ids)
            loc = anchor_cache_loc.to(torch.long)
            self._key_cache[loc, 0] = anchor_key.to(self._key_cache.dtype)
            self._value_cache[loc, 0] = anchor_value.to(self._value_cache.dtype)

        # `page_table` arrives ALREADY TRIMMED to max(seq_lens) + 1 columns; the caller
        # takes that width from seq_lens_cpu, so nothing here needs a GPU-to-CPU sync.
        # Re-deriving it with prefix.max().item() cost one serialising sync per decode
        # round and re-materialised what the caller had just sized correctly.
        #
        # Copied before the scatter: req_to_token belongs to the scheduler and this must
        # not write through it.  Row-gathering already produced a fresh tensor upstream,
        # but the copy is kept explicit rather than resting on that.
        prefix = prefix_lens.to(torch.long).view(-1, 1)
        table = page_table.to(torch.int32).clone()
        if self.use_anchor_kv:
            table.scatter_(1, prefix, anchor_cache_loc.view(-1, 1).to(torch.int32))
        return AttnHeadContext(
            key_cache=self._key_cache,
            value_cache=self._value_cache,
            page_table=table,
            # +1 for the anchor column only when there IS one; flash_attn reads exactly
            # cache_seqlens entries, so the spare page-table column is never visited.
            cache_seqlens=((prefix_lens + 1) if self.use_anchor_kv
                           else prefix_lens).to(torch.int32),
        )

    def _attend(self, query: torch.Tensor, context: AttnHeadContext,
                queries_per_seq: int = 1) -> torch.Tensor:
        """query [bs, heads, head_dim] -> [bs, heads * head_dim].

        One paged attention over [prefix ; anchor].  The anchor was written into the
        cache by build_serving_context, so there is nothing to merge here.
        """
        from sglang.kernels.ops.attention.flash_attention import (
            flash_attn_with_kvcache,
        )

        rows = query.shape[0]
        bs = rows // queries_per_seq
        out = flash_attn_with_kvcache(
            # [bs, queries_per_seq, heads, dim].  Every query of a request reads the
            # SAME prefix, so page_table stays [bs, width] and cache_seqlens [bs] no
            # matter how many queries there are -- which is what makes the lattice's
            # gamma*K queries one kernel launch instead of gamma of them, and keeps
            # the narrow page table the caller built.
            q=query.view(bs, queries_per_seq, self.num_heads, self.head_dim),
            k_cache=context.key_cache,
            v_cache=context.value_cache,
            page_table=context.page_table,
            cache_seqlens=context.cache_seqlens,
            softmax_scale=self.scaling,
            causal=False,
        )
        if isinstance(out, tuple):
            out = out[0]
        return out.reshape(rows, -1)

    def compute_step_latent(
        self,
        token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        context: AttnHeadContext,
        queries_per_seq: int = 1,
    ) -> torch.Tensor:
        if self._projected_embedding is not None:
            x = self._project_inputs(token_ids, hidden_states)
        else:
            embedded = self.embed_norm(
                self.embed_tokens(token_ids).to(hidden_states.dtype)
            )
            wide = torch.cat([embedded, self.state_norm(hidden_states)], dim=-1)
            x = self.in_proj(wide)
        return self._latent_from_input(x, context, queries_per_seq)

    def _latent_from_input(self, x, context, queries_per_seq):
        normed = self.attn_norm(x)
        query = self.q_norm(self.q_proj(normed).view(-1, self.num_heads, self.head_dim))
        attended = self._attend(query, context, queries_per_seq)
        if self.gate_mode == "state":
            gate = torch.sigmoid(self.gate_proj(normed))
        elif self.gate_mode == "state_output":
            gate = torch.sigmoid(self.gate_proj(torch.cat([normed, attended], dim=-1)))
        if self.gate_mode != "none":
            attended = (
                attended.view(-1, self.num_heads, self.head_dim) * gate.unsqueeze(-1)
            ).reshape(attended.shape)
        x = x + self.o_proj(attended)
        normed = self.mlp_norm(x)
        if self._gate_up_weight is None:
            gated = F.silu(self.gate_p(normed)) * self.up(normed)
        else:
            gated = self._gate_up_activation(F.linear(normed, self._gate_up_weight))
        x = x + self.down(gated)
        return self.out_norm(x)

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        context: Optional[AttnHeadContext] = None,
    ) -> torch.Tensor:
        if context is None:
            raise ValueError(_ATTN_HEAD_NO_CONTEXT)
        if hidden_states is None:
            raise ValueError("AttnHead needs the drafter's hidden state.")
        return self.project_bias(
            self.compute_step_latent(token_ids, hidden_states, context)
        )

    def apply_step_logits(
        self,
        logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        context: Optional[AttnHeadContext] = None,
    ) -> torch.Tensor:
        return logits + self.compute_step_bias(token_ids, hidden_states, context)

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        context: Optional[AttnHeadContext] = None,
    ) -> torch.Tensor:
        # UNREACHABLE, and not implementable for this head either.  Nothing in sglang
        # calls apply_block_logits: it is a TEACHER-FORCED path, where every slot's
        # predecessor is already known and the block can be scored in one shot.  Serving
        # never has that.  The other heads inherit a working version from VanillaMarkov
        # because their bias does not depend on anything a later slot produces; this
        # one's query literally contains the token the previous slot just sampled.
        raise NotImplementedError(
            "AttnHead has no whole-block path: every slot's query contains the token the "
            "previous slot sampled. Use run_markov_block (sample_block)."
        )

    def sample_block(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_tokens: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        sampler: StepSampler,
        context: Optional[AttnHeadContext] = None,
        collect_logits: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if context is None:
            raise ValueError(_ATTN_HEAD_NO_CONTEXT)
        return run_markov_block(
            self,
            base_logits,
            first_prev_tokens=first_prev_tokens,
            hidden_states=hidden_states,
            sampler=sampler,
            context=context,
            collect_logits=collect_logits,
        )


def build_markov_lattice(
    head: nn.Module,
    *,
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    hidden_states: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    context=None,
) -> torch.Tensor:
    """A DSpark markov head's K x K lattice, in the DFlash2 selector's own shape.

    WHY THIS IS THE SERVING FORM, not an optimisation of the sequential walk.  These
    heads are trained with a nomination CE on the UNARY head -- `nominate_alpha > 0`,
    `nominate_topk = 16` -- whose whole content is

        M16 = sum_{b in Top16(u)} p(b),

    i.e. u is trained so that its top-16 candidate set contains the answer, and the
    order-1 head reranks inside that set.  Proposing by a full-vocabulary argmax walk
    searches a set the objective never asked u to be calibrated over, and pays a
    [rank, vocab] projection per slot per step to do it.

    WHY IT IS ALSO PARALLEL.  The predecessor enters every one of these heads through
    the previous token alone, and for the attention head the keys are the committed
    prefix -- which no predecessor changes.  So all slots x all predecessors are
    independent: one batched call, then a walk over the precomputed table.  The
    sequential dependence is real but it is a dependence between TABLE LOOKUPS, not
    between GPU launches.

    Differs from CandidateSelector.build_lattice only in how the per-(slot,
    predecessor) vector is formed; the predecessor indexing and the edge einsum are
    imported from there so the two cannot diverge.
    """
    from sglang.srt.models.dflash import (
        lattice_predecessor_ids,
        score_edges_from_vectors,
    )

    weight = getattr(head.markov_w2, "weight", None)
    if weight is None or weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(
            "The lattice gathers successor rows out of markov_w2.weight, so it needs a "
            f"dense float weight; got {type(head.markov_w2).__name__} with "
            f"{None if weight is None else weight.dtype}."
        )

    top_k = int(candidate_ids.shape[-1])
    predecessor_ids = lattice_predecessor_ids(
        candidate_ids=candidate_ids,
        anchor_token_ids=anchor_token_ids,
        top_k=top_k,
    )
    bs, slots, _ = predecessor_ids.shape
    # Request-major flattening: row = ((b * slots) + l) * top_k + p.  The attention head
    # views these back as [bs, slots * top_k, ...] against a per-REQUEST page table, so
    # this order is load-bearing, not cosmetic.
    kwargs = {}
    if context is not None:
        kwargs = {"context": context, "queries_per_seq": slots * top_k}
    if isinstance(head, AttnHead):
        latent = head.compute_lattice_latent(predecessor_ids, hidden_states, context)
    else:
        hidden = hidden_states[:, :, None, :].expand(bs, slots, top_k, -1)
        latent = head.compute_step_latent(
            predecessor_ids.reshape(-1),
            hidden.reshape(bs * slots * top_k, -1),
            **kwargs,
        )
    return score_edges_from_vectors(
        predecessor_vectors=latent.view(bs, slots, top_k, -1),
        successor_table=weight,
        candidate_ids=candidate_ids,
        unary_logits=unary_logits,
    )


def build_markov_head(config) -> Optional[nn.Module]:
    markov_rank = int(getattr(config, "markov_rank", 0))
    if markov_rank <= 0:
        raise ValueError(
            "DSpark requires markov_rank > 0 (the Markov head is the core of the "
            f"semi-AR draft); got markov_rank={markov_rank}."
        )
    markov_head_type = str(getattr(config, "markov_head_type", "vanilla")).lower()
    vocab_size = int(config.vocab_size)
    hidden_size = int(config.hidden_size)
    if markov_head_type == "vanilla":
        return VanillaMarkov(vocab_size=vocab_size, markov_rank=markov_rank)
    if markov_head_type == "gated":
        return GatedMarkovHead(
            vocab_size=vocab_size, markov_rank=markov_rank, hidden_size=hidden_size
        )
    if markov_head_type == "rnn":
        return RNNHead(
            vocab_size=vocab_size, markov_rank=markov_rank, hidden_size=hidden_size
        )
    if markov_head_type == "attn":
        return AttnHead(
            vocab_size=vocab_size,
            markov_rank=markov_rank,
            hidden_size=hidden_size,
            num_heads=int(config.markov_num_heads),
            head_dim=int(config.markov_head_dim),
            mlp_hidden=int(config.markov_mlp_hidden),
            gate_mode=str(config.markov_gate_mode),
            rms_norm_eps=float(config.rms_norm_eps),
            anchor_kv=bool(getattr(config, "markov_anchor_kv", True)),
        )
    if markov_head_type == "cond":
        return CondMarkovHead(
            vocab_size=vocab_size,
            markov_rank=markov_rank,
            hidden_size=hidden_size,
            mlp_hidden=int(config.markov_mlp_hidden),
            rms_norm_eps=float(config.rms_norm_eps),
        )
    raise ValueError(f"Unsupported DSpark markov_head_type={markov_head_type!r}.")


def build_nemotron_35_markov_head(config, quant_config, prefix: str) -> nn.Module:
    markov_head_type = str(getattr(config, "markov_head_type", "vanilla")).lower()
    if markov_head_type != "vanilla":
        raise ValueError(
            "Nemotron 3.5 DSpark requires markov_head_type='vanilla', "
            f"got {markov_head_type!r}."
        )
    markov_prefix = f"{prefix}.markov_head" if prefix else "markov_head"
    return Nemotron35VanillaMarkov(
        vocab_size=int(config.vocab_size),
        markov_rank=int(config.markov_rank),
        quant_config=quant_config,
        prefix=markov_prefix,
    )


class DSparkConfidenceHead(nn.Module):

    def __init__(
        self,
        *,
        hidden_size: int,
        markov_rank: int,
        with_markov: bool = True,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.with_markov = bool(with_markov)
        input_dim = int(hidden_size) + (int(markov_rank) if self.with_markov else 0)
        self.proj = nn.Linear(input_dim, 1, bias=bias, dtype=dtype)
        self.register_buffer(
            "sts_temperatures", torch.ones((), dtype=torch.float32), persistent=False
        )
        self._last_confidence_raw: Optional[torch.Tensor] = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        markov_embed_stack: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.with_markov:
            if markov_embed_stack is None:
                raise ValueError(
                    "DSparkConfidenceHead(with_markov=True) requires markov_embed_stack."
                )
            features = torch.cat(
                [hidden_states, markov_embed_stack.to(dtype=hidden_states.dtype)],
                dim=-1,
            )
        else:
            features = hidden_states
        features = features.to(dtype=self.proj.weight.dtype)
        return self.proj(features).squeeze(-1)

    def apply_sts(self, confidence_raw: torch.Tensor) -> torch.Tensor:
        self._last_confidence_raw = confidence_raw
        return torch.sigmoid(confidence_raw.float() / self.sts_temperatures)


def build_confidence_head(config) -> Optional[nn.Module]:
    if read_ragged_verify_mode() is RaggedVerifyMode.STATIC:
        return None
    if not hasattr(config, "enable_confidence_head"):
        logger.warning(
            "DSpark draft config has no enable_confidence_head field; treating the "
            "confidence head as enabled."
        )
    hidden_size = int(config.hidden_size)
    markov_rank = int(getattr(config, "markov_rank", 0))
    with_markov = bool(getattr(config, "confidence_head_with_markov", markov_rank > 0))
    if with_markov and markov_rank <= 0:
        raise ValueError(
            "DSpark confidence_head_with_markov requires markov_rank > 0, "
            f"got markov_rank={markov_rank}."
        )
    return DSparkConfidenceHead(
        hidden_size=hidden_size,
        markov_rank=markov_rank,
        with_markov=with_markov,
    )


_DSPARK_SKIPPED_WEIGHT_PREFIXES = ("lm_head.", "rotary_emb.")


class DSparkDraftMixin:

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        self._fused_kv_write_cache = None
        self.logits_mup_width_multiplier = None
        dspark_config = parse_dspark_draft_config(draft_hf_config=config)
        if not dspark_config.require_markov():
            raise ValueError(
                "DSpark draft requires markov_rank > 0, "
                f"got markov_rank={dspark_config.markov_rank}."
            )
        self.gamma = int(dspark_config.resolve_gamma(default=self.block_size))
        self.sample_from_anchor = get_dspark_sample_from_anchor(config)
        if self.is_nemotron_35_draft:
            self.markov_head = build_nemotron_35_markov_head(
                config, quant_config, prefix
            )
        else:
            self.markov_head = build_markov_head(config)
        self.confidence_head = build_confidence_head(config)
        self.lm_head: Optional[nn.Module] = None

    def attach_shared_modules(
        self, *, embed_tokens: nn.Module, lm_head: nn.Module
    ) -> None:
        if not self.is_nemotron_35_draft:
            self.embed_tokens = embed_tokens
        self.lm_head = lm_head
        # The attention head has no W1: its predecessor code IS the target's embedding,
        # borrowed by reference so it is neither saved nor counted twice.  It can only be
        # wired here, because the shared modules do not exist at construction.
        if isinstance(self.markov_head, AttnHead):
            self.markov_head.set_embedding_source(embed_tokens)
            self.markov_head.prepare_for_inference()

    def forward_embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Embeds with the shared target embedding INSIDE the draft graph
        # (the runner skips the eager input_embeds staging when the draft
        # model exposes forward_embed).
        return self.embed_tokens(input_ids)

    def compute_base_logits(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Project the draft's raw final hidden through the target lm_head.

        muP targets (Inkling) train the draft against a FOLDED head (weights
        pre-divided by logits_mup_width_multiplier) while serving attaches the
        target's unfolded head, so the division happens here — exactly once,
        keeping base logits in the scale the markov bias and confidence head
        were trained against. DSparkWorkerV2 wires the multiplier from the
        target config; it stays None for non-muP targets.
        """
        if self.lm_head is None:
            raise ValueError(
                "DSpark dense draft requires the target lm_head "
                "(call attach_shared_modules first)."
            )
        if self.logits_mup_width_multiplier:
            hidden = hidden / self.logits_mup_width_multiplier
        local_logits = project_through_lm_head(hidden, self.lm_head)
        base_logits = gather_and_crop_vocab(local_logits, self.lm_head)
        return base_logits, None

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        markov_weights = []
        confidence_weights = []
        backbone_weights = []
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            normalized_name = name.removeprefix("model.")
            if any(
                normalized_name.startswith(p) for p in _DSPARK_SKIPPED_WEIGHT_PREFIXES
            ):
                continue
            if normalized_name.startswith("embed_tokens.") and not (
                self.is_nemotron_35_draft
            ):
                continue
            if name.startswith("confidence_head."):
                if self.confidence_head is None:
                    continue
                confidence_weights.append((name, loaded_weight))
            elif name.startswith("markov_head."):
                markov_weights.append((name, loaded_weight))
            else:
                backbone_weights.append((name, loaded_weight))

        super().load_weights(backbone_weights)

        for name, loaded_weight in markov_weights:
            if name not in params_dict:
                raise ValueError(
                    f"DSpark unexpected markov weight {name!r} not found in model "
                    f"parameters (known markov params require a {type(self.markov_head).__name__} head)."
                )
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)

        self._load_confidence_weights(
            confidence_weights=confidence_weights, params_dict=params_dict
        )
        if isinstance(self.markov_head, AttnHead):
            self.markov_head.prepare_for_inference()

    def _load_confidence_weights(
        self,
        *,
        confidence_weights: list,
        params_dict: dict,
    ) -> None:
        if self.confidence_head is None:
            return
        loaded_names = set()
        for name, loaded_weight in confidence_weights:
            if name not in params_dict:
                raise ValueError(
                    f"DSpark unexpected confidence weight {name!r} not found in "
                    "model parameters."
                )
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_names.add(name)

        confidence_param_names = {
            name for name in params_dict if name.startswith("confidence_head.")
        }
        missing = confidence_param_names - loaded_names
        if missing:
            raise ValueError(
                f"DSpark confidence head is enabled but the checkpoint is missing "
                f"{sorted(missing)}. Provide a checkpoint with trained confidence weights, "
                f"or disable the confidence head (enable_confidence_head=False)."
            )

    def _fused_kv_write_bundle(self, pool):
        cached = self._fused_kv_write_cache
        if cached is not None and cached[0] == id(pool):
            return cached[1]
        bundle = self._build_fused_kv_write_bundle(pool)
        self._fused_kv_write_cache = (id(pool), bundle)
        return bundle

    def _build_fused_kv_write_bundle(self, pool):
        layers = list(self.layers)
        if not layers:
            return None
        if not (hasattr(pool, "get_key_buffer") and hasattr(pool, "get_value_buffer")):
            return None
        attn0 = layers[0].self_attn
        head_dim = attn0.head_dim
        kv_size = attn0.kv_size
        rotary = attn0.rotary_emb
        if type(rotary).__name__ != "RotaryEmbedding":
            return None
        if not getattr(rotary, "is_neox_style", False):
            return None
        if getattr(rotary, "rotary_dim", None) != head_dim:
            return None
        eps = attn0.k_norm.variance_epsilon
        weights, knws, meta_rows = [], [], []
        for layer in layers:
            attn = layer.self_attn
            ok, _ = can_dflash_slice_qkv_weight(attn.qkv_proj)
            if not ok:
                return None
            if attn.qkv_proj.bias is not None:
                return None
            if attn.attn.k_scale is not None or attn.attn.v_scale is not None:
                return None
            if attn.head_dim != head_dim or attn.kv_size != kv_size:
                return None
            if attn.rotary_emb is not rotary and not torch.equal(
                attn.rotary_emb.cos_sin_cache, rotary.cos_sin_cache
            ):
                return None
            if attn.k_norm.variance_epsilon != eps:
                return None
            k_buf = pool.get_key_buffer(attn.attn.layer_id)
            v_buf = pool.get_value_buffer(attn.attn.layer_id)
            nh = kv_size // head_dim
            for buf in (k_buf, v_buf):
                if buf.dtype != torch.bfloat16:
                    return None
                if buf.shape[1:] != (nh, head_dim):
                    return None
                if buf.stride(1) != head_dim or buf.stride(2) != 1:
                    return None
            kv_slice = slice(attn.q_size, attn.q_size + 2 * attn.kv_size)
            w = attn.qkv_proj.weight[kv_slice]
            if w.dtype != torch.bfloat16:
                return None
            weights.append(w)
            knws.append(attn.k_norm.weight.data)
            meta_rows.append(
                [k_buf.data_ptr(), v_buf.data_ptr(), k_buf.stride(0), v_buf.stride(0)]
            )
        device = weights[0].device
        w_all = torch.cat(weights, dim=0).contiguous()
        knw = torch.stack(knws).to(device)
        meta = torch.tensor(meta_rows, dtype=torch.int64, device=device)
        cos_sin = rotary.cos_sin_cache.to(device)
        return (w_all, meta, knw, cos_sin, eps, len(layers), kv_size, head_dim)

    def _stacked_ctx_kv_params(self) -> Optional[dict]:
        """Stack every layer's KV projection into one weight (exact: the input
        hidden is shared, so concatenating output columns is equivalent).
        Cached; None (per-layer fallback) when a QKV weight cannot be sliced
        (quantized) or layers disagree on norm epsilon / bias presence.
        """
        if not envs.SGLANG_DSPARK_STACKED_CTX_KV.get():
            return None
        cached = getattr(self, "_stacked_ctx_kv_cache", False)
        if cached is not False:
            return cached
        weights, biases, k_norm_weights = [], [], []
        eps = None
        for layer in self.layers:
            attn = layer.self_attn
            can_slice, _ = can_dflash_slice_qkv_weight(attn.qkv_proj)
            if not can_slice or eps not in (None, attn.k_norm.variance_epsilon):
                self._stacked_ctx_kv_cache = None
                return None
            eps = attn.k_norm.variance_epsilon
            kv_slice = slice(attn.q_size, attn.q_size + 2 * attn.kv_size)
            weights.append(attn.qkv_proj.weight[kv_slice])
            biases.append(
                attn.qkv_proj.bias[kv_slice] if attn.qkv_proj.bias is not None else None
            )
            k_norm_weights.append(attn.k_norm.weight)
        has_bias = [b is not None for b in biases]
        if any(has_bias) and not all(has_bias):
            self._stacked_ctx_kv_cache = None
            return None
        self._stacked_ctx_kv_cache = {
            "weight": torch.cat(weights, dim=0),
            "bias": torch.cat(biases, dim=0) if all(has_bias) else None,
            "k_norm_weight": torch.stack(k_norm_weights, dim=0).float(),
            "eps": eps,
        }
        return self._stacked_ctx_kv_cache

    def write_target_hidden_kv(
        self,
        *,
        target_hidden: torch.Tensor,
        pool,
        positions: torch.Tensor,
        cache_loc: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor] = None,
        commit_lens: Optional[torch.Tensor] = None,
    ) -> None:
        ctx_hidden = self.project_target_hidden(target_hidden)
        if isinstance(self.markov_head, AttnHead):
            # SAME ctx_hidden, SAME positions as the layers below.  A second projection
            # or a different tap would give the head a drifting summary of the target
            # instead of what the drafter actually reads.
            self.markov_head.ensure_context_cache(
                num_slots=int(pool.size) + 1,
                device=ctx_hidden.device,
                dtype=ctx_hidden.dtype,
            )
            if cache_loc_2d is not None and commit_lens is not None:
                self.markov_head.write_context_kv(
                    ctx_hidden=ctx_hidden,
                    cache_loc=cache_loc_2d.reshape(-1),
                    commit_lens=commit_lens,
                    locs_row_width=cache_loc_2d.shape[1],
                )
            else:
                self.markov_head.write_context_kv(
                    ctx_hidden=ctx_hidden, cache_loc=cache_loc
                )

        bundle = self._fused_kv_write_bundle(pool)
        if bundle is not None:
            from sglang.kernels.ops.speculative.dspark.fused_kv_write import (
                fused_kv_norm_rope_write,
            )

            w_all, meta, knw, cos_sin, eps, num_layers, kv_size, head_dim = bundle
            kv_all = F.linear(ctx_hidden, w_all)
            if cache_loc_2d is not None and commit_lens is not None:
                locs = cache_loc_2d.reshape(-1)
                write_commit_lens = commit_lens
                locs_row_width = cache_loc_2d.shape[1]
            else:
                locs = cache_loc
                write_commit_lens = None
                locs_row_width = None
            fused_kv_norm_rope_write(
                kv_all,
                meta,
                knw,
                cos_sin,
                positions,
                locs,
                num_layers,
                kv_size,
                head_dim,
                eps,
                commit_lens=write_commit_lens,
                locs_row_width=locs_row_width,
            )
            return

        stacked = self._stacked_ctx_kv_params()
        if stacked is not None:
            k_all, v_all = self._project_ctx_kv_stacked(
                ctx_hidden=ctx_hidden, positions=positions, stacked=stacked
            )
        for i, layer in enumerate(self.layers):
            attn = layer.self_attn
            if stacked is not None:
                k = k_all[i]
                v = v_all[i]
            else:
                k, v = attn.kv_proj_only(ctx_hidden)
                k = attn.apply_k_norm(k)
                k = attn.apply_k_rope(positions, k)
                k = k.view(-1, attn.num_kv_heads, attn.head_dim)
                v = v.view(-1, attn.num_kv_heads, attn.head_dim)
            if cache_loc_2d is not None and commit_lens is not None:
                pool.set_kv_buffer_prefix_valid(
                    attn.attn,
                    cache_loc_2d,
                    commit_lens,
                    k,
                    v,
                    attn.attn.k_scale,
                    attn.attn.v_scale,
                )
            else:
                pool.set_kv_buffer(
                    attn.attn,
                    cache_loc,
                    k,
                    v,
                    attn.attn.k_scale,
                    attn.attn.v_scale,
                )

    def _project_ctx_kv_stacked(
        self,
        *,
        ctx_hidden: torch.Tensor,
        positions: torch.Tensor,
        stacked: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn0 = self.layers[0].self_attn
        num_layers = len(self.layers)
        kv_size = attn0.kv_size
        head_dim = attn0.head_dim
        num_kv_heads = attn0.num_kv_heads
        tokens = ctx_hidden.shape[0]

        kv_all = F.linear(ctx_hidden, stacked["weight"], stacked["bias"])
        kv_all = kv_all.view(tokens, num_layers, 2, kv_size)
        # Batched per-head k-norm across layers (fp32 variance + weight, cast back).
        k32 = (
            kv_all[:, :, 0, :]
            .reshape(tokens, num_layers, num_kv_heads, head_dim)
            .to(torch.float32)
        )
        variance = k32.pow(2).mean(dim=-1, keepdim=True)
        k32 = k32 * torch.rsqrt(variance + stacked["eps"])
        k32 = k32 * stacked["k_norm_weight"].view(1, num_layers, 1, head_dim)
        k_all = k32.to(ctx_hidden.dtype)
        # One RoPE over all layers' heads (shared rotary params + positions).
        k_flat = k_all.reshape(tokens, num_layers * kv_size)
        dummy_q = k_flat.new_empty(k_flat.shape)
        _, k_flat = attn0.rotary_emb(positions, dummy_q, k_flat)
        # [layers, tokens, heads, dim]: per-layer slices are contiguous views.
        k_all = (
            k_flat.view(tokens, num_layers, num_kv_heads, head_dim)
            .permute(1, 0, 2, 3)
            .contiguous()
        )
        v_all = (
            kv_all[:, :, 1, :]
            .view(tokens, num_layers, num_kv_heads, head_dim)
            .permute(1, 0, 2, 3)
            .contiguous()
        )
        return k_all, v_all


class DSparkDraftModel(DSparkDraftMixin, DFlashDraftModel):

    def prune_to_ctx_kv_injection(self) -> None:
        self.markov_head = None
        self.confidence_head = None
        for layer in self.layers:
            layer.mlp = None
            layer.self_attn.o_proj = None
        torch.cuda.empty_cache()


class Qwen3DSparkModel(DSparkDraftModel):
    pass


EntryClass = [Qwen3DSparkModel, DSparkDraftModel]
