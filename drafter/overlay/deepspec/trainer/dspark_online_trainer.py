"""Online-feature DSpark trainer for a Qwen3 target.

Trains a fresh N-layer DSpark drafter on target features computed ONLINE: the frozen
target runs every micro-batch to produce the tapped hidden states (and, for the
distributional terms, its final hidden state), so no hidden-state cache is needed
and the training data is a JSONL of token trajectories.

`OnlineTargetTrainer` holds the online machinery (token dataset/collator, target
forward, optional fused target, window normalisation, run_batch).
`Qwen3DSparkOnlineTrainer` builds the Qwen3 DSpark drafter and adds the
candidate-nomination term.  Every drafter in the paper's solution section is
trained with `Qwen3DSparkOnlineTrainer`; the DFlash2 reproduction reuses
`OnlineTargetTrainer` (official_dflash2_trainer.py).
"""
import copy
import os

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from deepspec.data.jsonl_dataset import JsonLineDataset
from deepspec.modeling.dspark.loss import (
    compute_dspark_loss,
    set_async_denominator,
    set_window_normalization,
)
from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel
from deepspec.modeling.dspark.qwen3.config import (
    build_draft_config as build_qwen3_draft_config,
)
from deepspec.modeling.dspark.qwen3.modeling import DSparkShortConv
from deepspec.modeling.dspark.slot_init import max_pairwise_cosine
from deepspec.trainer.base_trainer import BaseTrainer
from deepspec.utils import print_on_local_main
from deepspec.utils.metrics import add_metric


def _cfg_get(cfg, key, default):
    return cfg[key] if key in cfg else default


class TokenCollator:
    """Right-pads {"input_ids", "prompt_len" | "loss_mask"} rows into tensors.

    loss_mask marks the positions to draft and score (the assistant turns);
    sequences are truncated to `max_len`, keeping the prefix.
    """

    pad_token_id = 0
    max_len = 4096  # set from config.data.max_length by the trainer

    def __call__(self, batch):
        ml = self.max_len
        lens = [min(len(b["input_ids"]), ml) for b in batch]
        maxlen = max(lens)
        n = len(batch)
        input_ids = torch.full((n, maxlen), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((n, maxlen), dtype=torch.long)
        loss_mask = torch.zeros((n, maxlen), dtype=torch.float32)
        for i, b in enumerate(batch):
            L = lens[i]
            input_ids[i, :L] = torch.tensor(b["input_ids"][:L], dtype=torch.long)
            attention_mask[i, :L] = 1
            if "loss_mask" in b:
                lm = b["loss_mask"][:L]
                loss_mask[i, : len(lm)] = torch.tensor(lm, dtype=torch.float32)
            else:
                p = min(int(b.get("prompt_len", 0)), L)
                loss_mask[i, p:L] = 1.0
        return {"input_ids": input_ids, "attention_mask": attention_mask, "loss_mask": loss_mask}


class OnlineTargetTrainer(BaseTrainer):
    """Online target features + the DSpark objective; subclasses build the models."""

    data_collator_cls = TokenCollator
    fused_target = None
    # Optional loss terms this trainer's combine_loss consumes (see
    # _assert_declared_terms_are_consumed).
    _consumed_loss_terms = ()
    _OPTIONAL_LOSS_TERM_FIELDS = ("nominate_alpha",)

    # ---------------------------------------------------------------- window norm
    # Normalise once per optimizer step instead of once per micro-batch.  The
    # denominator does not depend on the parameters, so each micro-batch backprops
    # its numerator over a fixed constant, FSDP reduces the gradients as usual on
    # the last micro-batch, and on_optimizer_boundary rescales them by the global
    # window denominator before clipping.  This removes a blocking collective from
    # every micro-batch and weights every token by the same 1/D.
    @property
    def window_normalized(self) -> bool:
        return bool(getattr(self.args.train, "window_normalized_denominator", False))

    def _window_state(self):
        if not self.window_normalized:
            return None
        state = getattr(self, "_window_denominator", None)
        if state is None:
            assert str(self.args.train.sharding_strategy) == "no_shard", (
                "window normalization rescales param.grad in place before one SUM "
                "all-reduce; a sharded strategy owns that reduction itself."
            )
            state = self._window_denominator = {
                # A data-independent constant that cancels at the boundary; it only
                # keeps bf16 gradients in range.
                "loss_scale": float(
                    int(self.args.train.local_batch_size)
                    * int(self.args.model.num_anchors)
                    * int(self.args.model.block_size)
                ),
            }
            self._window_den = None
        return state

    def _accumulate_window_denominator(self, state):
        den = state.pop("den_local")
        if self._window_den is None:
            self._window_den = den.detach().float().clone()
            self._window_n = 1
        else:
            self._window_den.add_(den.detach().float())
            self._window_n += 1

    @torch.no_grad()
    def _close_denominator_window(self) -> None:
        """Rescale the accumulated gradient by loss_scale * accumulation / D_window."""
        assert self._window_den is not None, "empty accumulation window"
        if self.world_size > 1:
            dist.all_reduce(self._window_den, op=dist.ReduceOp.SUM)
        assert float(self._window_den) > 0.0, "window has no valid slot positions"
        scale = (
            self._window_denominator["loss_scale"] * self.gradient_accumulation_steps
        ) / self._window_den
        for param in self.optimizer.model_params:
            if param.grad is not None:
                param.grad.mul_(scale)
        self._window_den = None
        self._window_n = 0

    # ----------------------------------------------------------- target features
    def maybe_build_fused_target(self, target_model) -> None:
        """Build the fused (sgl_kernel) teacher forward when `model.fused_target`.

        Not bit-identical to the HF forward (merged QKV / gate-up change reduction
        order); it is a throughput option used by the final run.
        """
        self.fused_target = None
        if not bool(_cfg_get(self.args.model, "fused_target", False)):
            return
        from deepspec.modeling.fused_target import FusedQwen3Target

        arch = type(target_model).__name__
        assert "Qwen3" in arch, (
            f"fused_target implements the Qwen3 block; target model is {arch}."
        )
        assert -1 not in self.model_target_layer_ids, (
            "fused target taps decoder layers; an embed_tokens tap (-1) is not "
            "implemented."
        )
        self.fused_target = FusedQwen3Target(target_model, self.model_target_layer_ids)
        print_on_local_main(
            f"[fused-target] sgl_kernel path ON, taps {list(self.model_target_layer_ids)}"
        )

    def build_train_dataset(self):
        paths = self.args.data.train_data_paths
        if isinstance(paths, str):
            paths = [paths]
        return JsonLineDataset(list(paths))

    def resume_draft_model(self, resume_checkpoint_dir):
        """Load a checkpoint into the model build_models already constructed."""
        import glob
        from safetensors.torch import load_file

        shards = sorted(glob.glob(os.path.join(resume_checkpoint_dir, "*.safetensors")))
        assert shards, f"no safetensors in {resume_checkpoint_dir}"
        sd = {}
        for s in shards:
            sd.update(load_file(s))
        missing, unexpected = self.draft_model.load_state_dict(sd, strict=False)
        # Every trainable parameter must come from the checkpoint.
        trainable_names = {n for n, p in self.draft_model.named_parameters() if p.requires_grad}
        missing_trainable = sorted(trainable_names - set(sd.keys()))
        assert not missing_trainable, (
            f"resume checkpoint {resume_checkpoint_dir} is missing "
            f"{len(missing_trainable)} trainable params: e.g. {missing_trainable[:3]}"
        )
        print_on_local_main(
            f"[dspark-online] resumed {resume_checkpoint_dir}: {len(sd)} tensors, "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"(all {len(trainable_names)} trainable present)"
        )
        self.draft_model.set_embedding_head_trainable(False)
        return self.draft_model.to(device=self.device, dtype=self.precision_dtype)

    @torch.no_grad()
    def _online_target_features(self, input_ids, attention_mask=None):
        """Tapped target hiddens (concatenated) and the target's final hidden state.

        Forward hooks capture `backbone.layers[lid]` outputs (-1 = embeddings), so
        the other layers' activations are never materialised and no lm_head runs.
        """
        # With one unpadded sequence per micro-batch the mask is all ones; passing
        # None avoids a GPU->CPU sync in HF's mask construction.
        if attention_mask is not None and attention_mask.shape[0] == 1:
            attention_mask = None
        if self.fused_target is not None:
            th, last = self.fused_target(input_ids)
            return th.to(self.precision_dtype), last.to(self.precision_dtype)
        captured, handles = {}, []

        def mk(lid, mod):
            def hook(_m, _i, out):
                captured[lid] = out[0] if isinstance(out, (tuple, list)) else out
            return mod.register_forward_hook(hook)

        try:
            for lid in self.model_target_layer_ids:
                mod = self.target_backbone.embed_tokens if lid == -1 else self.target_backbone.layers[lid]
                handles.append(mk(lid, mod))
            target_outputs = self.target_backbone(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
            )
            th = torch.cat([captured[lid] for lid in self.model_target_layer_ids], dim=-1)
            target_last_hidden = target_outputs[0]
        finally:
            for h in handles:
                h.remove()
        return (
            th.to(self.precision_dtype),
            target_last_hidden.to(self.precision_dtype),
        )

    @torch.no_grad()
    def _online_target_hidden(self, input_ids, attention_mask=None):
        target_hidden, _ = self._online_target_features(input_ids, attention_mask)
        return target_hidden

    def _requires_target_distribution(self) -> bool:
        """Whether the objective needs exact target logits at the sampled blocks."""
        return (
            float(self.args.model.l1_loss_alpha) > 0.0
            or float(self.args.model.confidence_head_alpha) > 0.0
            or float(getattr(self.args.model, "nominate_alpha", 0.0) or 0.0) > 0.0
        )

    # ----------------------------------------------------------------- hooks
    def on_optimizer_boundary(self) -> None:
        """Called by BaseTrainer after the gradient reduction, before clipping."""
        if self.window_normalized:
            self._close_denominator_window()
        self._log_short_conv_health()

    @torch.no_grad()
    def _log_short_conv_health(self) -> None:
        """Short-conv diagnostics from parameters alone: the predecessor tap |k1|, the
        self-tap deviation |k0 - 1|, the correction weight scales, and |grad|."""
        convs = [
            m for m in self.model.modules()
            if type(m).__name__ == "DSparkShortConv" and hasattr(m, "k_base")
        ]
        if not convs:
            return
        n = float(len(convs))
        tot = {"k1": 0.0, "k0d": 0.0, "c1": 0.0, "c0": 0.0, "g1": 0.0, "g0": 0.0}
        for c in convs:
            k = c.k_base.detach().float()
            k1 = float(k[1].abs().mean())
            k0d = float((k[0] - 1.0).abs().mean())
            w = c.corr.weight.detach().float()
            g = int(c.n_groups)
            c0 = float(w[:g].pow(2).mean().sqrt())
            c1 = float(w[g:].pow(2).mean().sqrt())
            gk = c.k_base.grad
            g1 = 0.0 if gk is None else float(gk.detach().float()[1].abs().mean())
            g0 = 0.0 if gk is None else float(gk.detach().float()[0].abs().mean())
            add_metric(f"conv_k1@{c.probe_name}", k1, den=1.0, tag="train")
            for key, val in (("k1", k1), ("k0d", k0d), ("c1", c1), ("c0", c0),
                             ("g1", g1), ("g0", g0)):
                tot[key] += val
        for key, val in tot.items():
            add_metric(f"conv_{key}", val, den=n, tag="train")

    def combine_loss(self, loss, outputs):
        """Hook to add terms after compute_dspark_loss (default: identity)."""
        del outputs
        return loss

    def _assert_declared_terms_are_consumed(self):
        """A declared optional term that this trainer's combine_loss would drop is an
        error, not a silent no-op."""
        consumed = getattr(self, "_consumed_loss_terms", ())
        for field in self._OPTIONAL_LOSS_TERM_FIELDS:
            if field in consumed:
                continue
            alpha = float(getattr(self.args.model, field, 0.0) or 0.0)
            assert alpha <= 0.0, (
                f"{field}={alpha} is declared but {type(self).__name__} does not "
                f"consume it."
            )

    # ----------------------------------------------------------------- step
    def run_batch(self, batch):
        if not getattr(self, "_window_norm_wired", False):
            # Before the first forward: in window mode the model must not launch the
            # per-micro-batch count reduction.
            set_window_normalization(self.window_normalized)
            set_async_denominator(
                bool(getattr(self.args.train, "async_denominator", True)))
            self._window_norm_wired = True

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        loss_mask = batch["loss_mask"].to(self.device)
        needs_target_distribution = self._requires_target_distribution()
        # With a CE-only objective the temperature-1 acceptance metrics still need the
        # target logits; `tv_metric_stride` computes them every N steps.
        tv_stride = int(getattr(self.args.train, "tv_metric_stride", 0) or 0)
        want_tv_metric = tv_stride > 0 and (self.global_step % tv_stride == 0)
        target_last_hidden = None
        if needs_target_distribution or want_tv_metric:
            th, target_last_hidden = self._online_target_features(input_ids, attention_mask)
        else:
            th = self._online_target_hidden(input_ids, attention_mask)
        outputs = self.model(
            input_ids=input_ids,
            loss_mask=loss_mask,
            target_hidden_states=th,
            target_last_hidden_states=target_last_hidden,
        )
        self.maybe_log_sampler_stats(
            input_ids=input_ids,
            loss_mask=loss_mask,
            outputs=outputs,
        )
        _window = self._window_state()
        loss = compute_dspark_loss(
            outputs=outputs,
            loss_decay_gamma=self.args.model.loss_decay_gamma,
            ce_loss_alpha=float(self.args.model.ce_loss_alpha),
            l1_loss_alpha=float(self.args.model.l1_loss_alpha),
            confidence_head_alpha=float(self.args.model.confidence_head_alpha),
            compile_l1=bool(getattr(self.args.model, "compile_l1", False)),
            window_denominator=_window,
        )
        if _window is not None:
            self._accumulate_window_denominator(_window)
        if not getattr(self, "_loss_terms_checked", False):
            self._assert_declared_terms_are_consumed()
            self._loss_terms_checked = True
        return self.combine_loss(loss, outputs)


class _NominationCEMixin:
    """Adds the candidate-nomination loss (nomination.py) with weight nominate_alpha.

    It uses the same slot weights exp(-k/gamma) and the same global denominator as
    the main objective (in window mode, the deferred window denominator).
    """

    _consumed_loss_terms = ("nominate_alpha",)

    def combine_loss(self, loss, outputs):
        loss = super().combine_loss(loss, outputs)
        terms = getattr(outputs, "nominate_terms", None)
        alpha = float(getattr(self.args.model, "nominate_alpha", 0.0) or 0.0)
        if terms is None or alpha <= 0.0:
            return loss
        num, den = terms                                     # [K], [K]
        gamma = float(self.args.model.loss_decay_gamma)
        k = torch.arange(num.shape[0], device=num.device, dtype=torch.float32)
        w = torch.exp(-k / gamma) if gamma and gamma > 0 else torch.ones_like(k)
        local_num = (num.float() * w).sum()
        local_den = (den.float() * w).sum()
        # Local numerator over the global denominator, times world_size (FSDP
        # averages rank gradients), as in the main objective.
        world = dist.get_world_size() if dist.is_initialized() else 1
        _win = getattr(self, "_window_denominator", None)
        if _win is not None:
            nom = local_num / float(_win["loss_scale"]) * world
            for i in range(num.shape[0]):
                add_metric(f"nom_ce@{i}", num[i].detach(), den=den[i], tag="train")
            add_metric("nom_ce", local_num.detach(), den=local_den, tag="train")
            return loss + alpha * nom
        # `den` is the per-slot eval_mask count, so the global denominator is the
        # one the loss already reduced early; reuse it when available.
        _pf = (None if os.environ.get("DSPARK_SYNC_DENOM") == "1"
               else getattr(outputs, "slot_count_reduction", None))
        if _pf is not None:
            _handle, _counts = _pf
            _handle.wait()
            global_den = (_counts.to(w.dtype) * w).sum()
        else:
            global_den = local_den.detach().clone()
            if world > 1:
                dist.all_reduce(global_den, op=dist.ReduceOp.SUM)
        nom = local_num / (global_den + 1e-6) * world
        for i in range(num.shape[0]):
            add_metric(f"nom_ce@{i}", num[i].detach(), den=den[i], tag="train")
        add_metric("nom_ce", local_num.detach(), den=local_den, tag="train")
        total = loss + alpha * nom
        add_metric("loss_total", total.detach(), reduction="mean", tag="train")
        return total


# Modules that do not exist in the plain DSpark drafter.  Constructing them draws
# from the global RNG stream (and HF post_init re-draws every nn.Linear), so an arm
# that enables one would otherwise start from a differently initialised backbone
# than the arm it is compared with.  `shared_init_ablate` names them; the shared
# parameters are then copied from a sibling model built without them.
_ADDITIVE_MODULE_FLAGS = ("short_conv", "slot_embed")
# A non-vanilla order-1 head counts as additive too; its sibling uses the vanilla
# head at `shared_init_markov_rank`.
_MARKOV_HEAD_ABLATE_KEY = "markov_head"


def _report_slot_embed_separation(draft_model):
    """Worst-pair cosine of E(mask) + s_k after the target embeddings are loaded."""
    se = getattr(draft_model, "slot_embed", None)
    if se is None:
        return
    off = draft_model.embed_tokens.weight[draft_model.mask_token_id].detach().float()
    mc = max_pairwise_cosine(se.detach().float(), offset=off)
    print_on_local_main(
        f"[dspark-online] slot_embed separation: max pairwise cos(E(mask)+s_k) = "
        f"{mc:.3f}  (||E(mask)|| = {off.norm():.4f}, ||s_k|| = "
        f"{se.detach().float().norm(dim=-1).mean():.4f})")
    assert mc < 0.90, (
        f"slot_embed leaves the worst slot pair at cosine {mc:.3f}: the slots still "
        f"enter near-identical. Raise slot_embed_std.")


def _repair_from_scratch_additive_modules(draft_model, draft_config, model_args, *, seed):
    from deepspec.utils import seed_all

    active = [k for k in _ADDITIVE_MODULE_FLAGS if bool(getattr(draft_config, k, False))]
    head_type = str(getattr(draft_config, "markov_head_type", "vanilla") or "vanilla").lower()
    if int(getattr(draft_config, "markov_rank", 0) or 0) > 0 and head_type != "vanilla":
        active.append(_MARKOV_HEAD_ABLATE_KEY)
    ablate = list(_cfg_get(model_args, "shared_init_ablate", []) or [])
    assert not (active and not ablate), (
        f"config enables additive module(s) {active} but does not declare "
        f"`shared_init_ablate`; the shared parameters would not match the arm without "
        f"them.  Set shared_init_ablate={active!r} in model=dict(...)."
    )
    for k in ablate:
        assert k in _ADDITIVE_MODULE_FLAGS + (_MARKOV_HEAD_ABLATE_KEY,), (
            f"unknown shared_init_ablate entry {k!r}"
        )

    if ablate:
        base_config = copy.deepcopy(draft_config)
        for k in ablate:
            if k == _MARKOV_HEAD_ABLATE_KEY:
                # The sibling must carry the BASELINE's rank: the rank changes how
                # much RNG the head consumes before post_init.
                _r = _cfg_get(model_args, "shared_init_markov_rank", None)
                assert _r is not None, (
                    "`shared_init_ablate` includes 'markov_head', so set "
                    "`shared_init_markov_rank` to the markov_rank of the paired arm."
                )
                base_config.markov_rank = int(_r)
                base_config.markov_head_type = "vanilla"
            else:
                setattr(base_config, k, False)
        seed_all(int(seed))
        base = Qwen3DSparkModel(base_config)
        base_params = dict(base.named_parameters())
        n = 0
        with torch.no_grad():
            for name, p in draft_model.named_parameters():
                q = base_params.get(name)
                if q is not None and q.shape == p.shape:
                    p.copy_(q.to(device=p.device, dtype=p.dtype))
                    n += 1
        del base, base_params
        print_on_local_main(
            f"[dspark-online] shared init drawn from a sibling with {ablate} ablated: "
            f"{n} tensors copied"
        )

    if getattr(draft_model, "slot_embed", None) is not None:
        se = draft_model.slot_embed
        assert torch.isfinite(se).all() and se.abs().sum() > 0, (
            "slot_embed is zero/degenerate at step 0")
    convs = [m for m in draft_model.modules() if isinstance(m, DSparkShortConv)]
    for c in convs:
        c.reset_new_params()
    if convs:
        print_on_local_main(
            f"[dspark-online] {len(convs)} short convs re-initialised to identity"
        )
    for i, c in enumerate(convs):
        assert torch.equal(c.k_base[0], torch.ones_like(c.k_base[0])), f"conv {i}: k0 != 1"
        assert torch.equal(c.k_base[1], torch.zeros_like(c.k_base[1])), f"conv {i}: k1 != 0"
        assert torch.equal(c.corr.weight, torch.zeros_like(c.corr.weight)), (
            f"conv {i}: corr != 0 at step 0"
        )


class Qwen3DSparkOnlineTrainer(_NominationCEMixin, OnlineTargetTrainer):
    """Qwen3 DSpark drafter trained from scratch on online Qwen3 target features."""

    def build_models(self):
        ma = self.args.model
        tokenizer = AutoTokenizer.from_pretrained(ma.target_model_name_or_path)
        target_config = AutoConfig.from_pretrained(ma.target_model_name_or_path)

        draft_config = build_qwen3_draft_config(target_config=target_config, model_args=ma)
        draft_model = Qwen3DSparkModel(draft_config).to(
            device=self.device, dtype=self.precision_dtype
        )
        _repair_from_scratch_additive_modules(
            draft_model, draft_config, ma, seed=int(self.args.seed)
        )

        target_model = (
            AutoModelForCausalLM.from_pretrained(
                ma.target_model_name_or_path, dtype=self.precision_dtype
            )
            .to(device=self.device)
            .eval()
        )
        target_model.requires_grad_(False)
        draft_model.initialize_embeddings_and_head(
            embed_tokens=target_model.get_input_embeddings(),
            lm_head=target_model.get_output_embeddings(),
            freeze=True,
        )
        _report_slot_embed_separation(draft_model)

        self.target_model = target_model
        self.model_target_layer_ids = list(draft_config.target_layer_ids)
        self.target_backbone = target_model.model
        self.maybe_build_fused_target(target_model)
        TokenCollator.max_len = int(_cfg_get(self.args.data, "max_length", 4096))

        trainable = sum(p.numel() for p in draft_model.parameters() if p.requires_grad)
        print_on_local_main(
            f"[dspark-online] {draft_config.num_hidden_layers}-layer Qwen3 DSpark "
            f"(online target features @ {self.model_target_layer_ids}); "
            f"trainable={trainable:,}"
        )
        return draft_model, tokenizer


__all__ = [
    "OnlineTargetTrainer",
    "Qwen3DSparkOnlineTrainer",
    "TokenCollator",
]
