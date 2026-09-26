"""DFlash2 reproduction: speculators' own DFlash2DraftModel and loss on our data.

Used only for the "DFlash2 reproduction" row of the one-epoch component comparison
(tab:solution-components).  The model, its convolution wiring and candidate
selector, and the objective (0.1 soft CE + 0.9 TV on the unary logits plus the
selector's K-way hard CE, fixed exp decay gamma=4) are speculators' code at the
pinned commit below.  Ours are the data (the same Qwen3-4B regenerated corpus as
every other arm), the online target forward (speculators reads hidden states from
disk or a patched vLLM), and the training loop with the published hyperparameters.

`speculators` is an OPTIONAL dependency, imported lazily here and nowhere else:

    pip install "speculators @ git+https://github.com/vllm-project/speculators@0a1b3e0a15d67d551041933529c2c41032f5b28d"

See drafter/README.md for installing it without its data-transfer extras.

One trap this file guards against: speculators applies its own `verifier_norm` to
`verifier_last_hidden_states`, so the target's final hidden state must be passed
PRE-norm (captured from the last decoder layer); `Qwen3Model(...)[0]` is already
normed.  `_assert_prenorm_readout` checks this at startup.
"""

import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from deepspec.utils import print_on_local_main
from deepspec.utils.metrics import add_metric

from .dspark_online_trainer import OnlineTargetTrainer

SPECULATORS_COMMIT = "0a1b3e0a15d67d551041933529c2c41032f5b28d"


def _import_speculators():
    """Import the speculators symbols this trainer needs, or explain how to get them."""
    try:
        from speculators.config import SpeculatorsConfig, VerifierConfig
        from speculators.losses import resolve_loss_config
        from speculators.models.dflash2.config import DFlash2SpeculatorConfig
        from speculators.models.dflash2.core import DFlash2DraftModel
        from speculators.proposals.greedy import GreedyTokenProposalConfig
    except ImportError as exc:
        raise ImportError(
            "The DFlash2 reproduction needs the optional `speculators` package at "
            f"commit {SPECULATORS_COMMIT}:\n"
            "    pip install \"speculators @ git+https://github.com/vllm-project/"
            f"speculators@{SPECULATORS_COMMIT}\"\n"
            "See drafter/README.md (optional dependencies). Nothing else in this "
            "package needs it."
        ) from exc
    return dict(
        SpeculatorsConfig=SpeculatorsConfig,
        VerifierConfig=VerifierConfig,
        resolve_loss_config=resolve_loss_config,
        DFlash2SpeculatorConfig=DFlash2SpeculatorConfig,
        DFlash2DraftModel=DFlash2DraftModel,
        GreedyTokenProposalConfig=GreedyTokenProposalConfig,
    )


def _cfg_get(ns, key, default=None):
    return getattr(ns, key, default)


class OfficialDFlash2Trainer(OnlineTargetTrainer):
    """Our data + our target forward, their model and their objective."""

    def build_models(self):
        spec = _import_speculators()
        SpeculatorsConfig = spec["SpeculatorsConfig"]
        VerifierConfig = spec["VerifierConfig"]
        DFlash2SpeculatorConfig = spec["DFlash2SpeculatorConfig"]
        DFlash2DraftModel = spec["DFlash2DraftModel"]
        GreedyTokenProposalConfig = spec["GreedyTokenProposalConfig"]

        ma = self.args.model
        tokenizer = AutoTokenizer.from_pretrained(ma.target_model_name_or_path)
        target_config = AutoConfig.from_pretrained(ma.target_model_name_or_path)

        # The drafter's own transformer stack: the target's layer config, narrowed to
        # num_draft_layers.  Copied rather than hand-built so every unlisted field
        # (rope theta, norm eps, head_dim) matches the target exactly.
        layer_cfg = AutoConfig.from_pretrained(ma.target_model_name_or_path)
        layer_cfg.num_hidden_layers = int(ma.num_draft_layers)
        cw = _cfg_get(ma, "context_window", None)
        if cw is None:
            layer_cfg.use_sliding_window = False
            layer_cfg.sliding_window = None
            layer_cfg.layer_types = ["full_attention"] * int(ma.num_draft_layers)
        else:
            layer_cfg.use_sliding_window = True
            layer_cfg.sliding_window = int(cw)
            layer_cfg.layer_types = ["sliding_attention"] * int(ma.num_draft_layers)

        draft_config = DFlash2SpeculatorConfig(
            transformer_layer_config=layer_cfg,
            # No vocabulary mapping: we speculate over the target's own vocabulary,
            # so draft_vocab_size == target vocab and their vocab-mapping path is a
            # no-op.  (Their released 27B DOES map 248320 -> a smaller draft vocab;
            # turning that on would be a second change and is not part of this arm.)
            draft_vocab_size=int(target_config.vocab_size),
            block_size=int(ma.block_size),
            target_hidden_size=int(target_config.hidden_size),
            # +1: THEIR convention indexes the hidden_states LIST, where 0 is the
            # embedding output and i is decoder layer i-1.  Ours indexes
            # `target_backbone.layers` directly.  Their own docs make the offset
            # explicit -- convert/entrypoints.py:88 recommends [2,10,18,26,34] for
            # DFlash, exactly our [1,9,17,25,33] plus one -- so the layers we hook and
            # the layers they name are THE SAME layers, and only the stored number
            # differs.  Storing ours unshifted would make anyone loading this
            # checkpoint through vLLM or their loader feed the draft model taps from
            # one decoder layer earlier than the fc was trained on.
            aux_hidden_state_layer_ids=[int(i) + 1 for i in ma.target_layer_ids],
            mask_token_id=int(ma.mask_token_id),
            sliding_window_non_causal=bool(_cfg_get(ma, "sliding_window_non_causal", True)),
            # FALSE = theirs.  The anchor slot is context, not a supervised
            # prediction, so block_size 8 yields 7 speculative tokens -- the same 7
            # our b7 arms produce, but with the anchor as slot 0's predecessor tap.
            sample_from_anchor=bool(_cfg_get(ma, "sample_from_anchor", False)),
            conv_kernel_size=int(_cfg_get(ma, "conv_kernel_size", 2)),
            conv_group_size=int(_cfg_get(ma, "conv_group_size", 16)),
            selector_rank=int(_cfg_get(ma, "selector_rank", 256)),
            selector_top_k=int(_cfg_get(ma, "selector_top_k", 16)),
            # REQUIRED, not decoration.  `SpeculatorModel.load_verifier_weights`
            # opens with `if speculators_config is None: return` -- it fails SILENTLY,
            # loading no verifier weights at all -- and `verifier_norm` /
            # `verifier_lm_head` are excluded from the saved state precisely because
            # they are meant to be restored from here.  Leaving it null produces a
            # checkpoint that their own loader cannot reconstitute.
            # `speculative_tokens` is block_size - 1 exactly when sample_from_anchor is
            # False (their `_build_base_config_kwargs` computes it the same way), i.e.
            # 7 here -- the same 7 our b7 arms emit.
            speculators_config=SpeculatorsConfig(
                algorithm="dflash2",
                proposal_methods=[GreedyTokenProposalConfig(
                    speculative_tokens=int(ma.block_size) - (
                        0 if bool(_cfg_get(ma, "sample_from_anchor", False)) else 1))],
                default_proposal_method="greedy",
                verifier=VerifierConfig.from_pretrained(ma.target_model_name_or_path),
            ),
        )

        draft_model = DFlash2DraftModel(draft_config).to(
            device=self.device, dtype=self.precision_dtype)

        target_model = AutoModelForCausalLM.from_pretrained(
            ma.target_model_name_or_path, dtype=self.precision_dtype
        ).to(self.device).eval()
        target_model.requires_grad_(False)

        # Borrow and FREEZE the vocabulary-facing weights, exactly as their
        # `load_verifier_weights` does: embed_tokens, lm_head, and the verifier's own
        # head+norm which turn target hidden states into teacher logits.
        borrowed = []
        with torch.no_grad():
            src = {
                "embed_tokens": target_model.get_input_embeddings().weight,
                "lm_head": target_model.get_output_embeddings().weight,
                "verifier_lm_head": target_model.get_output_embeddings().weight,
                "verifier_norm": target_model.model.norm.weight,
            }
            for name, w in src.items():
                mod = getattr(draft_model, name, None)
                if mod is None:
                    continue
                mod.weight.copy_(w.detach().to(mod.weight.dtype))
                mod.weight.requires_grad_(False)
                borrowed.append(name)
        assert {"embed_tokens", "lm_head", "verifier_lm_head", "verifier_norm"} <= set(borrowed), (
            f"failed to borrow all verifier-facing weights; got {sorted(borrowed)}")

        self.target_model = target_model
        self.model_target_layer_ids = list(ma.target_layer_ids)
        self.target_backbone = target_model.model
        # Their `_backbone_forward` takes verifier_last_hidden_states PRE-norm.  We
        # capture it from the last decoder layer; see the module docstring.
        self._prenorm_layer_idx = len(self.target_backbone.layers) - 1
        self._readout_probe_model = draft_model
        self._assert_prenorm_readout()

        n_train = sum(p.numel() for p in draft_model.parameters() if p.requires_grad)
        print_on_local_main(
            f"[official-dflash2] speculators DFlash2DraftModel: block={draft_config.block_size} "
            f"sample_from_anchor={draft_config.sample_from_anchor} "
            f"conv(k={draft_config.conv_kernel_size},g={draft_config.conv_group_size}) "
            f"selector(rank={draft_config.selector_rank},top_k={draft_config.selector_top_k}); "
            f"taps: hook layers{self.model_target_layer_ids} "
            f"= their hidden_states{draft_config.aux_hidden_state_layer_ids} (stored); "
            f"borrowed+frozen {sorted(borrowed)}; "
            f"trainable={n_train:,}"
        )
        self.draft_config = draft_config
        return draft_model, tokenizer

    def resume_draft_model(self, resume_checkpoint_dir):
        """Load a checkpoint back into speculators' model.

        The inherited implementation calls `set_embedding_head_trainable`, which only
        exists on Qwen3DSparkModel.  The four verifier-facing tensors are re-frozen
        here instead.
        """
        import glob

        from safetensors.torch import load_file

        shards = sorted(glob.glob(os.path.join(resume_checkpoint_dir, "*.safetensors")))
        assert shards, f"no safetensors in {resume_checkpoint_dir}"
        sd = {}
        for shard in shards:
            sd.update(load_file(shard))
        missing, unexpected = self.draft_model.load_state_dict(sd, strict=False)

        # Same guard as the parent: every trainable tensor must actually be in the
        # checkpoint.  A silent partial resume looks exactly like a bad arm.
        trainable = {n for n, p in self.draft_model.named_parameters() if p.requires_grad}
        missing_trainable = sorted(trainable - set(sd.keys()))
        assert not missing_trainable, (
            f"resume checkpoint {resume_checkpoint_dir} is missing "
            f"{len(missing_trainable)} trainable params: e.g. {missing_trainable[:3]}")

        for name in ("embed_tokens", "lm_head", "verifier_lm_head", "verifier_norm"):
            mod = getattr(self.draft_model, name, None)
            if mod is not None:
                mod.weight.requires_grad_(False)

        print_on_local_main(
            f"[official-dflash2] resumed {resume_checkpoint_dir}: {len(sd)} tensors, "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"(all {len(trainable)} trainable present; verifier-facing re-frozen)")
        return self.draft_model.to(device=self.device, dtype=self.precision_dtype)

    @torch.no_grad()
    def _assert_prenorm_readout(self):
        """Prove the readout is pre-norm, rather than commenting that it is.

        Passing an already-normed hidden state to a module that norms again squares
        the norm weight.  The check that catches it: their
        `verifier_lm_head(verifier_norm(h))` must reproduce the target's OWN logits.
        Feeding the post-norm tensor instead moves the argmax on real text; feeding
        the pre-norm one reproduces it exactly.
        """
        # Deterministic ids, not torch.randint: speculators' anchor sampler draws
        # from the global CUDA generator, so a random draw here would shift every
        # sampled supervision set of the run.  Ordinary-token range, where the
        # post-norm comparison is representative of real text.
        ids = torch.arange(1000, 1064, device=self.device).unsqueeze(0)
        captured = {}

        def hook(_m, _i, out):
            captured["pre"] = out[0] if isinstance(out, (tuple, list)) else out

        h = self.target_backbone.layers[self._prenorm_layer_idx].register_forward_hook(hook)
        try:
            out = self.target_backbone(input_ids=ids, use_cache=False)
        finally:
            h.remove()
        post = out[0]
        pre = captured["pre"]

        inner = self._readout_probe_model
        ours = inner.verifier_lm_head(inner.verifier_norm(pre.to(self.precision_dtype)))
        ref = self.target_model.lm_head(post.to(self.precision_dtype))
        agree = (ours.argmax(-1) == ref.argmax(-1)).float().mean().item()

        wrong = inner.verifier_lm_head(inner.verifier_norm(post.to(self.precision_dtype)))
        agree_wrong = (wrong.argmax(-1) == ref.argmax(-1)).float().mean().item()

        print_on_local_main(
            f"[official-dflash2] readout check: pre-norm agreement {agree:.4f}, "
            f"post-norm (the double-norm bug) {agree_wrong:.4f}")
        assert agree > 0.999, (
            f"pre-norm readout only reproduces {agree:.4f} of the target's argmax; the "
            "hooked layer is not the pre-norm hidden state and every teacher logit "
            "this arm trains against would be wrong")

    def _online_target_features(self, input_ids, attention_mask=None):
        """Tapped hiddens for the draft backbone + the PRE-norm final hidden state.

        Overrides the parent, which returns the POST-norm `Qwen3Model(...)[0]`:
        speculators' model norms again (`verifier_lm_head(verifier_norm(h))`), so it
        needs the PRE-norm tensor.  Keeping the parent's name means any caller of
        `_online_target_features` gets the right tensor for this model.
        """
        _none_mask = os.environ.get("DSPARK_TARGET_NONE_MASK", "1") != "0"
        if _none_mask and attention_mask is not None and attention_mask.shape[0] == 1:
            attention_mask = None
        captured, handles = {}, []

        def mk(key, mod):
            def hook(_m, _i, out):
                captured[key] = out[0] if isinstance(out, (tuple, list)) else out
            return mod.register_forward_hook(hook)

        try:
            for lid in self.model_target_layer_ids:
                mod = (self.target_backbone.embed_tokens if lid == -1
                       else self.target_backbone.layers[lid])
                handles.append(mk(lid, mod))
            if self._prenorm_layer_idx not in captured:
                handles.append(mk("__pre",
                                  self.target_backbone.layers[self._prenorm_layer_idx]))
            self.target_backbone(input_ids=input_ids, attention_mask=attention_mask,
                                 use_cache=False)
            th = torch.cat([captured[lid] for lid in self.model_target_layer_ids], dim=-1)
            pre = captured.get("__pre", captured.get(self._prenorm_layer_idx))
        finally:
            for h in handles:
                h.remove()
        return th.to(self.precision_dtype), pre.to(self.precision_dtype)

    def run_batch(self, batch):
        resolve_loss_config = _import_speculators()["resolve_loss_config"]

        ma = self.args.model
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        loss_mask = batch["loss_mask"].to(self.device)

        with torch.no_grad():
            hidden_states, verifier_last = self._online_target_features(
                input_ids, attention_mask)

        # One document per row in our corpus, so a constant id is correct here.  Their
        # anchor sampler uses this only to keep a block from straddling a boundary.
        document_ids = torch.zeros_like(input_ids)

        if getattr(self, "_loss_config", None) is None:
            spec = str(_cfg_get(ma, "loss_fn", '{"ce": 0.1, "tv": 0.9}'))
            self._loss_config = resolve_loss_config(spec, "fused")
            print_on_local_main(
                f"[official-dflash2] loss_fn={spec} -> "
                f"{ {k: w for k, (_, w) in self._loss_config.items()} }")

        _, loss, metrics = self.model(
            hidden_states,
            input_ids,
            loss_mask,
            verifier_last,
            document_ids,
            None,
            loss_config=self._loss_config,
            gamma=float(ma.loss_decay_gamma),
            max_anchors=int(ma.num_anchors),
            selector_loss_alpha=float(_cfg_get(ma, "selector_loss_alpha", 1.0)),
            per_position_loss_weight=str(
                _cfg_get(ma, "per_position_loss_weight", "fixed-exp-decay")),
        )

        # Their metrics come as *_sum / *_total pairs; add_metric takes exactly that,
        # so the whole panel (accept lengths, recall@16, per-position CE) lands in
        # tensorboard without being re-derived on our side.
        for key, val in metrics.items():
            if key.endswith("_sum"):
                den = metrics.get(key[:-4] + "_total")
                if den is not None:
                    add_metric(key[:-4], val.detach(), den=den, tag="train")
        return loss
