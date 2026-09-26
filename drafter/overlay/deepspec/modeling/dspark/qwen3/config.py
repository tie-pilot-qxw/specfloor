import copy

from deepspec.modeling.dspark.common import validate_target_layer_ids


TRAIN_ATTN_IMPLEMENTATION = "flex_attention"


def build_draft_config(
    target_config,
    model_args,
):
    num_target_layers = int(target_config.num_hidden_layers)
    num_draft_layers = int(model_args.num_draft_layers)
    layer_types = ["full_attention"] * num_draft_layers
    assert "target_layer_ids" in model_args, "target_layer_ids must be provided."
    target_layer_ids = validate_target_layer_ids(
        model_args.target_layer_ids,
        num_target_layers,
    )

    confidence_head_alpha = float(model_args.confidence_head_alpha)
    assert confidence_head_alpha >= 0.0
    enable_confidence_head = confidence_head_alpha > 0.0
    if enable_confidence_head:
        assert "confidence_head_with_markov" in model_args, (
            "confidence_head_with_markov must be provided when "
            "confidence_head_alpha > 0."
        )
    markov_rank = int(model_args.markov_rank)
    assert markov_rank >= 0, f"markov_rank must be >= 0, got {markov_rank}"
    if markov_rank > 0:
        assert "markov_head_type" in model_args, (
            "markov_head_type must be provided when markov_rank > 0."
        )

    draft_config = copy.deepcopy(target_config)
    draft_config.architectures = ["Qwen3DSparkModel"]
    draft_config.num_target_layers = num_target_layers
    draft_config.num_hidden_layers = num_draft_layers
    draft_config.block_size = int(model_args.block_size)
    draft_config.tie_word_embeddings = False
    draft_config.layer_types = layer_types
    draft_config._attn_implementation = TRAIN_ATTN_IMPLEMENTATION
    draft_config.mask_token_id = int(model_args.mask_token_id)
    draft_config.target_layer_ids = target_layer_ids
    draft_config.num_anchors = int(model_args.num_anchors)
    draft_config.enable_confidence_head = enable_confidence_head
    if enable_confidence_head:
        draft_config.confidence_head_with_markov = bool(
            model_args.confidence_head_with_markov
        )
    draft_config.markov_rank = markov_rank
    if markov_rank > 0:
        draft_config.markov_head_type = str(model_args.markov_head_type)
        if str(model_args.markov_head_type).lower() == "attn":
            # Written out explicitly so the saved config.json carries effective
            # values: SGLang rebuilds the head from it.
            draft_config.markov_num_heads = int(
                getattr(model_args, "markov_num_heads", 4) or 4
            )
            draft_config.markov_head_dim = int(
                getattr(model_args, "markov_head_dim", 128) or 128
            )
            draft_config.markov_mlp_hidden = int(
                getattr(model_args, "markov_mlp_hidden", 2048) or 2048
            )
            draft_config.markov_gate_mode = str(
                getattr(model_args, "markov_gate_mode", "state_output")
            )
            # 0.0 selects the vanilla-matched initial output scale.
            draft_config.markov_out_scale = float(
                getattr(model_args, "markov_out_scale", 0.0) or 0.0
            )
            # Seeding W2 from the target's lm_head; 0.0 keeps a random init.
            draft_config.markov_seed_std_pred = float(
                getattr(model_args, "markov_seed_std_pred", 0.0) or 0.0
            )
            draft_config.markov_seed_std_succ = float(
                getattr(model_args, "markov_seed_std_succ", 0.0) or 0.0
            )
    for field, default in (
        # Within-block attention stays bidirectional; persisted because SGLang reads
        # it to select the serving mask.
        ("causal_draft_mask", False),
        # How often (in micro-batches) anchor-sampler statistics are logged.
        ("sampler_stats_stride", 50),
        # Two-tap block-causal short convolution before and after each sublayer.
        ("short_conv", False),
        ("short_conv_group_size", 16),
        # The prefix-attention head's per-block anchor key/value column.
        ("markov_anchor_kv", True),
        # Learned masked-position (slot) embeddings for slots 1..K-1.
        ("slot_embed", False),
        ("slot_embed_std", 0.02),
        ("slot_embed_depth_frac", 0.4),
    ):
        setattr(
            draft_config,
            field,
            model_args[field] if field in model_args else default,
        )
    draft_config.is_causal = bool(draft_config.causal_draft_mask)
    assert not draft_config.causal_draft_mask, (
        "causal_draft_mask=True is not supported by this release."
    )
    # Candidate-nomination loss on the base logits (0.0 = off).
    draft_config.nominate_alpha = float(
        getattr(model_args, "nominate_alpha", 0.0) or 0.0
    )
    draft_config.nominate_topk = int(
        getattr(model_args, "nominate_topk", 16) or 16
    )
    return draft_config


__all__ = [
    "build_draft_config",
]
