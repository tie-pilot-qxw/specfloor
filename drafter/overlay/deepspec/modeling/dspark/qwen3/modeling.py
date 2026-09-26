"""Qwen3 DSpark drafter with the three additions used by the paper's drafter.

  * `short_conv`: a DFlash2-style two-tap, block-causal dynamic convolution applied
    inside the residual branch before and after each attention and MLP sublayer
    (four independent modules per layer, identity-initialised).
  * `slot_embed`: learned embeddings added to the masked slots 1..K-1.
  * `markov_head_type="attn"`: the prefix-attention order-1 head (attn_head.py),
    plus the candidate-nomination loss on the base logits (nomination.py).

With all three off the model is upstream DSpark.
"""

from typing import Callable, Optional

import torch
from torch import nn

from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    GradientCheckpointingLayer,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    eager_attention_forward,
    rotate_half,
)
from typing_extensions import Tuple, Unpack

from deepspec.modeling.dspark.common import (
    AcceptRatePredictor,
    DSparkForwardOutput,
    build_eval_mask,
    create_dspark_attention_mask,
    create_noise_embed,
    create_position_ids,
    sample_anchor_positions,
)
from deepspec.modeling.dspark.loss import launch_slot_count_reduction
from deepspec.modeling.dspark.markov_head import build_markov_head
from deepspec.modeling.dspark.qwen3.short_conv_kernel import short_conv_apply
from deepspec.modeling.dspark.slot_init import depth_axis_init_
from deepspec.utils.sampling import sample_tokens


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


@torch.compiler.disable
def _attn_kernel_no_compile(attn_fn, module, q, k, v, attention_mask, **kw):
    """Graph break around the attention kernel: the flex_attention block-16 backward
    compiles pathologically under torch.compile, so it runs eagerly while the rest of
    the model compiles.  A no-op when torch.compile is not active."""
    return attn_fn(module, q, k, v, attention_mask, **kw)


class Qwen3DSparkAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = (
            self.num_attention_heads // self.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == "sliding_attention"
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        target_hidden_states: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        cos, sin = position_embeddings
        ctx_len = target_hidden_states.shape[1]
        q = self.q_proj(hidden_states).view(
            bsz, q_len, self.num_attention_heads, self.head_dim
        )
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden_states)
        v_ctx = self.v_proj(target_hidden_states)
        k_noise = self.k_proj(hidden_states)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(
            bsz, ctx_len + q_len, self.num_key_value_heads, self.head_dim
        )
        v = torch.cat([v_ctx, v_noise], dim=1).view(
            bsz, ctx_len + q_len, self.num_key_value_heads, self.head_dim
        )
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        if (
            self.config._attn_implementation == "flex_attention"
            and self.num_key_value_groups > 1
        ):
            kv_seq_len = k.shape[-2]
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)
            k = k.reshape(bsz, self.num_attention_heads, kv_seq_len, self.head_dim)
            v = v.reshape(bsz, self.num_attention_heads, kv_seq_len, self.head_dim)
        attn_fn: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_is_causal = bool(kwargs.get("is_causal", False))
        # The SDPA path may consult module.is_causal when dispatching kernels,
        # so keep the per-call value mirrored on the module before invoking it.
        self.is_causal = attn_is_causal
        kwargs["is_causal"] = attn_is_causal
        attn_output, attn_weights = _attn_kernel_no_compile(
            attn_fn,
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(
            bsz, q_len, self.num_attention_heads * self.head_dim
        )
        return self.o_proj(attn_output), attn_weights


class DSparkShortConv(nn.Module):
    """Two-tap block-causal dynamic depthwise convolution (the DFlash2 operator).

        Conv(x)_t = (b_0 + c_0(x_t)) * x_t + (b_1 + c_1(x_t)) * x_{t-1}

    t indexes slots within a draft block and x_{-1} = 0, so the operator never mixes
    blocks.  b_0, b_1 are per-channel base kernels; the corrections c_0, c_1 are a
    linear function of x_t shared within groups of `short_conv_group_size` channels.
    Identity at initialisation (b_0 = 1, b_1 = 0, correction = 0).  The model uses
    four independent instances per layer; the post-sublayer instances compute their
    coefficients from the sublayer output.
    """

    def __init__(self, config, name: str = ""):
        super().__init__()
        d = int(config.hidden_size)
        # Insertion point, e.g. "L3.post_attn", used by the training health metrics.
        self.probe_name = str(name)
        self.block_size = int(config.block_size)
        self.group = int(getattr(config, "short_conv_group_size", 16))
        assert d % self.group == 0, f"hidden_size {d} not divisible by group {self.group}"
        self.n_groups = d // self.group
        self.k_base = nn.Parameter(torch.zeros(2, d))
        self.corr = nn.Linear(d, 2 * self.n_groups, bias=False)
        self.reset_new_params()

    def reset_new_params(self) -> None:
        """Identity: k0 = 1, k1 = 0, correction = 0.  HF post_init re-draws every
        nn.Linear, so the model calls this again after construction."""
        with torch.no_grad():
            self.k_base[0].fill_(1.0)
            self.k_base[1].zero_()
            self.corr.weight.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [bsz, num_blocks * block_size, hidden]."""
        bsz, q_len, d = x.shape
        K = self.block_size
        assert q_len % K == 0, f"short conv needs whole blocks: {q_len} % {K} != 0"
        n = q_len // K
        xb = x.reshape(bsz * n, K, d)
        # The correction Linear is a regular GEMM; the group broadcast, predecessor
        # shift and two-tap combine are one fused kernel (short_conv_kernel.py).
        c = self.corr(xb).view(bsz * n, K, 2, self.n_groups)
        out = short_conv_apply(xb, c, self.k_base, self.group)
        return out.view(bsz, q_len, d)


class Qwen3DSparkDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3DSparkAttention(config=config, layer_idx=layer_idx)
        # One conv before and one after each sublayer, inside the residual branch,
        # so the identity initialisation leaves the residual stream exact.
        self.short_conv = bool(getattr(config, "short_conv", False))
        if self.short_conv:
            self.conv_pre_attn = DSparkShortConv(config, f"L{layer_idx}.pre_attn")
            self.conv_post_attn = DSparkShortConv(config, f"L{layer_idx}.post_attn")
            self.conv_pre_mlp = DSparkShortConv(config, f"L{layer_idx}.pre_mlp")
            self.conv_post_mlp = DSparkShortConv(config, f"L{layer_idx}.post_mlp")
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        target_hidden_states: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.short_conv:
            hidden_states = self.conv_pre_attn(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden_states=target_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        if self.short_conv:
            hidden_states = self.conv_post_attn(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.short_conv:
            hidden_states = self.conv_pre_mlp(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.short_conv:
            hidden_states = self.conv_post_mlp(hidden_states)
        return residual + hidden_states


class Qwen3DSparkModel(Qwen3PreTrainedModel):
    _no_split_modules = ["Qwen3DSparkDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        required_fields = (
            "target_layer_ids",
            "mask_token_id",
            "num_anchors",
            "enable_confidence_head",
            "markov_rank",
        )
        for field in required_fields:
            assert hasattr(config, field), f"config.{field} must be provided."
        if int(config.markov_rank) > 0:
            assert hasattr(config, "markov_head_type"), (
                "config.markov_head_type must be provided when markov_rank > 0."
            )
        if bool(config.enable_confidence_head):
            assert hasattr(config, "confidence_head_with_markov"), (
                "config.confidence_head_with_markov must be provided when "
                "enable_confidence_head is true."
            )
        self.target_layer_ids = config.target_layer_ids

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=getattr(config, "pad_token_id", None),
        )
        self.layers = nn.ModuleList(
            [
                Qwen3DSparkDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.block_size = int(config.block_size)
        self.mask_token_id = config.mask_token_id
        self.num_anchors = int(config.num_anchors)

        # Slot embeddings for the masked slots 1..K-1 (slot 0 is the real anchor
        # token).  Drawn from a dedicated generator so enabling them does not shift
        # the global RNG stream.  See slot_init.py for the initialisation.
        self.slot_embed = None
        if bool(getattr(config, "slot_embed", False)):
            self.slot_embed = nn.Parameter(
                torch.empty(int(config.block_size) - 1, config.hidden_size)
            )
            _g = torch.Generator()
            _g.manual_seed(0x510E_C0DE)
            depth_axis_init_(
                self.slot_embed,
                depth_frac=float(getattr(config, "slot_embed_depth_frac", 0.4)),
                std=float(getattr(config, "slot_embed_std", 0.02)),
                generator=_g,
            )

        # Markov (order-1) head.
        self.markov_head = build_markov_head(config, context_wired=True)
        if getattr(self.markov_head, "markov_head_type", "") == "attn":
            # By reference: the predecessor code is the frozen target embedding.
            self.markov_head.set_embedding_source(self.embed_tokens)

        # Confidence head.
        self.enable_confidence_head = bool(config.enable_confidence_head)
        self.confidence_head_with_markov = False
        if self.enable_confidence_head:
            self.confidence_head_with_markov = bool(config.confidence_head_with_markov)
        if self.enable_confidence_head and self.confidence_head_with_markov:
            assert self.markov_head is not None

        self.confidence_head = None
        if self.enable_confidence_head:
            input_dim = int(config.hidden_size)
            if self.confidence_head_with_markov:
                input_dim += config.markov_rank
            self.confidence_head = AcceptRatePredictor(input_dim=input_dim)
        self.post_init()
        # post_init re-initialises every nn.Linear, which would undo the identity
        # initialisation of the convs and the head's initial scale, so re-apply them.
        for module in self.modules():
            if module is not self and hasattr(module, "reset_new_params"):
                module.reset_new_params()

    def initialize_embeddings_and_head(
        self,
        *,
        embed_tokens: nn.Module,
        lm_head: nn.Module,
        freeze: bool = True,
        target_last_layer: nn.Module = None,
    ):
        # target_last_layer is accepted for call-site compatibility and unused.
        del target_last_layer
        assert self.embed_tokens.weight.shape == embed_tokens.weight.shape
        assert self.lm_head.weight.shape == lm_head.weight.shape
        with torch.no_grad():
            self.embed_tokens.weight.copy_(embed_tokens.weight.detach())
            self.lm_head.weight.copy_(lm_head.weight.detach())
        # The head's vocabulary readout is seeded from the target table when the
        # config asks for it (markov_seed_std_*).  A dedicated generator keeps the
        # global CUDA stream -- which the anchor sampler draws from -- untouched.
        if self.markov_head is not None and hasattr(self.markov_head, "seed_codebooks"):
            sp = float(getattr(self.config, "markov_seed_std_pred", 0.0) or 0.0)
            ss = float(getattr(self.config, "markov_seed_std_succ", 0.0) or 0.0)
            if sp > 0.0 and ss > 0.0:
                gen = torch.Generator(device=self.embed_tokens.weight.device)
                gen.manual_seed(0x5EED_C0DE)
                self.markov_head.seed_codebooks(
                    embed_weight=self.embed_tokens.weight,
                    lm_head_weight=self.lm_head.weight,
                    std_pred=sp, std_succ=ss, generator=gen,
                )
                w2 = self.markov_head.markov_w2.weight
                print(f"[{getattr(self.markov_head, 'markov_head_type', '?')}-head] "
                      f"W2 seeded from the target lm_head: std {float(w2.float().std()):.4f} "
                      f"(target {ss:.4f})")
        if freeze:
            self.set_embedding_head_trainable(False)

    def set_embedding_head_trainable(self, trainable: bool):
        self.embed_tokens.requires_grad_(trainable)
        self.lm_head.requires_grad_(trainable)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def predict_confidence_step(
        self,
        hidden_states: torch.Tensor,
        prev_token_ids: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if self.confidence_head is None:
            return None
        if self.confidence_head_with_markov:
            assert self.markov_head is not None
            assert prev_token_ids is not None
            prev_embeddings = self.markov_head.get_prev_embeddings(prev_token_ids).to(
                dtype=hidden_states.dtype
            )
            features = torch.cat([hidden_states, prev_embeddings], dim=-1)
            return self.confidence_head(features).float()
        return self.confidence_head(hidden_states).float()

    def sample_draft_tokens(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_token_ids: torch.Tensor,
        temperature: float = 0.0,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, proposal_len = base_logits.shape[:2]
        if proposal_len == 0:
            empty_tokens = torch.empty(
                batch_size,
                0,
                dtype=torch.long,
                device=base_logits.device,
            )
            return empty_tokens, base_logits
        if self.markov_head is None:
            return sample_tokens(base_logits, temperature), base_logits
        return self.markov_head.sample_block_tokens(
            base_logits,
            first_prev_token_ids=first_prev_token_ids,
            hidden_states=hidden_states,
            temperature=temperature,
        )

    def sample_draft_token_step(
        self,
        base_logits: torch.Tensor,
        *,
        prev_token_ids: torch.Tensor,
        temperature: float = 0.0,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert base_logits.ndim == 2, (
            "sample_draft_token_step expects base_logits shaped [batch, vocab], "
            f"got {tuple(base_logits.shape)}."
        )
        if self.markov_head is None:
            step_logits = base_logits
        else:
            step_logits = self.markov_head.apply_step_logits(
                base_logits,
                token_ids=prev_token_ids,
                hidden_states=hidden_states,
            )
        sampled_token_ids = sample_tokens(
            step_logits.unsqueeze(1),
            temperature=temperature,
        ).squeeze(1)
        return sampled_token_ids, step_logits

    def _forward_backbone(
        self,
        *,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden_states: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = noise_embedding
        if self.slot_embed is not None and hidden_states is not None:
            # Applied here so every path that drafts a block (training and
            # evaluation) sees it.  Layout is block-major with slot 0 first.
            K = self.block_size
            b, L, d = hidden_states.shape
            assert L % K == 0, (
                f"noise_embedding length {L} is not a multiple of block_size {K}; "
                "slot_embed assumes block-major layout")
            ne = hidden_states.view(b, L // K, K, d)
            hidden_states = torch.cat(
                [ne[:, :, :1], ne[:, :, 1:] + self.slot_embed.to(ne.dtype)], dim=2
            ).reshape(b, L, d)
        target_hidden_states = self.hidden_norm(self.fc(target_hidden_states))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden_states=target_hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)

    def forward(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        target_hidden_states: Optional[torch.Tensor] = None,
        target_last_hidden_states: Optional[torch.Tensor] = None,
    ) -> DSparkForwardOutput:
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        anchor_positions, block_keep_mask = sample_anchor_positions(
            seq_len=seq_len,
            loss_mask=loss_mask,
            num_anchors=self.num_anchors,
            device=device,
        )
        noise_embedding = create_noise_embed(
            self.embed_tokens,
            input_ids,
            anchor_positions,
            block_keep_mask,
            mask_token_id=self.mask_token_id,
            block_size=self.block_size,
        )
        context_position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        draft_position_ids = create_position_ids(anchor_positions, self.block_size)
        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)
        dspark_attn_mask = create_dspark_attention_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            seq_len=seq_len,
            block_size=self.block_size,
            device=device,
        )
        output_hidden = self._forward_backbone(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden_states=target_hidden_states,
            attention_mask=dspark_attn_mask,
        )

        num_blocks = anchor_positions.size(1)
        output_hidden_4d = output_hidden.reshape(bsz, num_blocks, self.block_size, -1)

        label_offsets = torch.arange(1, self.block_size + 1, device=device).view(
            1, 1, -1
        )
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        safe_label_indices = label_indices.clamp(max=seq_len - 1)
        safe_label_indices = torch.where(
            block_keep_mask.unsqueeze(-1),
            safe_label_indices,
            torch.zeros_like(safe_label_indices),
        )
        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )
        aligned_target_logits = None
        if target_last_hidden_states is not None:
            target_pred_indices = (safe_label_indices - 1).clamp(min=0)
            aligned_target_hidden = torch.gather(
                target_last_hidden_states.unsqueeze(1).expand(
                    -1,
                    anchor_positions.size(1),
                    -1,
                    -1,
                ),
                2,
                target_pred_indices.unsqueeze(-1).expand(
                    -1,
                    -1,
                    -1,
                    target_last_hidden_states.size(-1),
                ),
            )
            aligned_target_logits = self.compute_logits(aligned_target_hidden)
        eval_mask = build_eval_mask(
            seq_len=seq_len,
            loss_mask=loss_mask,
            label_indices=label_indices,
            safe_label_indices=safe_label_indices,
            block_keep_mask=block_keep_mask,
        )
        # eval_mask is final here: start the loss-denominator reduction now so the
        # lm_head and the loss numerators overlap with it.
        slot_count_reduction = launch_slot_count_reduction(eval_mask)
        anchor_token_ids = torch.gather(
            input_ids,
            1,
            anchor_positions,
        )
        prev_token_ids = torch.cat(
            [anchor_token_ids.unsqueeze(-1), target_ids[:, :, :-1]],
            dim=-1,
        )
        draft_logits = self.compute_logits(output_hidden).reshape(
            bsz,
            num_blocks,
            self.block_size,
            -1,
        )
        # Candidate nomination trains the base logits u = draft_logits, before the
        # order-1 head, since serving draws its candidate set from Top-K(u).
        nominate_terms = None
        target_top_idx = None
        nominate_alpha = float(getattr(self.config, "nominate_alpha", 0.0) or 0.0)
        if nominate_alpha > 0.0:
            assert aligned_target_logits is not None, (
                "nominate_alpha > 0 but the target distribution was not computed."
            )
            from deepspec.modeling.dspark.nomination import nomination_terms

            nominate_topk = int(getattr(self.config, "nominate_topk", 16) or 16)
            # One top-K of the target logits, shared with the loss's rank metrics.
            target_top_idx = aligned_target_logits.detach().topk(
                nominate_topk, dim=-1
            ).indices
            nominate_terms = nomination_terms(
                unary_logits=draft_logits,
                target_logits=aligned_target_logits.detach(),
                eval_mask=eval_mask,
                top_k=nominate_topk,
                top_idx=target_top_idx,
            )
        if self.markov_head is not None:
            head_kwargs = {}
            if getattr(self.markov_head, "markov_head_type", "") == "attn":
                # The head reads the same fused target features as the drafter's
                # layers (recomputed here; _forward_backbone does not return them)
                # under the drafter's own prefix-visibility rule.
                fused_target = self.hidden_norm(self.fc(target_hidden_states))
                head_kwargs["context"] = self.markov_head.build_context(
                    target_hidden=fused_target,
                    anchor_token_ids=anchor_token_ids,
                    anchor_positions=anchor_positions,
                    block_keep_mask=block_keep_mask,
                    block_size=self.block_size,
                )
            draft_logits = self.markov_head.apply_block_logits(
                draft_logits,
                token_ids=prev_token_ids,
                hidden_states=output_hidden_4d,
                **head_kwargs,
            )

        confidence_pred = None
        if self.confidence_head is not None:
            if self.confidence_head_with_markov:
                prev_embeddings = self.markov_head.get_prev_embeddings(prev_token_ids).to(
                    dtype=output_hidden_4d.dtype
                )
                confidence_features = torch.cat(
                    [output_hidden_4d, prev_embeddings],
                    dim=-1,
                )
                confidence_pred = self.confidence_head(confidence_features).float()
            else:
                confidence_pred = self.confidence_head(output_hidden_4d).float()

        return DSparkForwardOutput(
            draft_logits=draft_logits,
            target_ids=target_ids,
            eval_mask=eval_mask,
            block_keep_mask=block_keep_mask,
            confidence_pred=confidence_pred,
            aligned_target_logits=aligned_target_logits,
            nominate_terms=nominate_terms,
            target_top_idx=target_top_idx,
            slot_count_reduction=slot_count_reduction,
        )


__all__ = [
    "Qwen3DSparkModel",
    "Qwen3DSparkAttention",
    "Qwen3DSparkDecoderLayer",
    "DSparkShortConv",
]
