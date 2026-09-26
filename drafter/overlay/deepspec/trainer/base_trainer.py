from contextlib import nullcontext
import math
import os
import time

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from deepspec.data import CacheDataset, validate_train_cache
from deepspec.data.cuda_prefetcher import CUDAPrefetcher
from deepspec.modeling.dspark.common import log_sampler_stats
from deepspec.utils.metrics import add_metric
from deepspec.utils import (
    BF16Optimizer,
    StatelessResumableDistributedSampler,
    ensure_dir,
    init_dist,
    is_global_main_process,
    print_on_global_main,
    print_on_local_main,
)
from deepspec.trainer.ckpt_manager import (
    discover_latest_checkpoint,
    load_resume_draft_model,
    load_training_state,
    save_checkpoint,
    validate_resume_parallelism,
)
import deepspec.utils.training_logger as training_logger
from deepspec.utils.hfai_suspend import SuspendController


_PRECISION_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

_SHARDING_STRATEGIES = {
    "full_shard": ShardingStrategy.FULL_SHARD,
    "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
    "no_shard": ShardingStrategy.NO_SHARD,
    "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
    "hybrid_shard_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
    "_hybrid_shard_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
}

_HYBRID_STRATEGIES = (
    ShardingStrategy.HYBRID_SHARD,
    ShardingStrategy._HYBRID_SHARD_ZERO2,
)


def _build_fsdp_kwargs(
    *, sharding_strategy_name: str, precision_dtype, world_size: int
) -> dict:
    sharding_strategy = _SHARDING_STRATEGIES[sharding_strategy_name]
    fsdp_kwargs = dict(
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=precision_dtype,
            buffer_dtype=precision_dtype,
        ),
        sharding_strategy=sharding_strategy,
    )
    if sharding_strategy in _HYBRID_STRATEGIES:
        devices_per_node = torch.cuda.device_count()
        fsdp_kwargs["device_mesh"] = init_device_mesh(
            "cuda",
            (world_size // devices_per_node, devices_per_node),
            mesh_dim_names=("replicate", "shard"),
        )
    return fsdp_kwargs


def _compute_gradient_accumulation_steps(
    *, world_size: int, local_batch_size: int, global_batch_size: int
) -> int:
    denom = world_size * local_batch_size
    assert global_batch_size % denom == 0, (
        "global_batch_size must be divisible by world_size * local_batch_size: "
        f"global_batch_size={global_batch_size}, world_size={world_size}, "
        f"local_batch_size={local_batch_size}"
    )
    return global_batch_size // denom


def _compute_samples_per_epoch(*, dataset_size: int, global_batch_size: int) -> int:
    samples_per_epoch = (dataset_size // global_batch_size) * global_batch_size
    assert samples_per_epoch > 0, (
        "train dataset is too small to form one full global batch: "
        f"dataset_size={dataset_size}, global_batch_size={global_batch_size}"
    )
    return samples_per_epoch


def _compute_training_schedule(
    *,
    world_size: int,
    dataset_size: int,
    local_batch_size: int,
    global_batch_size: int,
    num_train_epochs: int,
    max_train_steps=None,
) -> tuple[int, int, int, int, int, int, int]:
    gradient_accumulation_steps = _compute_gradient_accumulation_steps(
        world_size=world_size,
        local_batch_size=local_batch_size,
        global_batch_size=global_batch_size,
    )
    samples_per_epoch = _compute_samples_per_epoch(
        dataset_size=dataset_size,
        global_batch_size=global_batch_size,
    )
    per_rank_samples_per_epoch = samples_per_epoch // world_size
    micro_batches_per_epoch = per_rank_samples_per_epoch // local_batch_size
    steps_per_epoch = micro_batches_per_epoch // gradient_accumulation_steps
    if max_train_steps is None:
        resolved_max_train_steps = int(num_train_epochs) * steps_per_epoch
        resolved_num_train_epochs = int(num_train_epochs)
    else:
        resolved_max_train_steps = int(max_train_steps)
        resolved_num_train_epochs = math.ceil(
            resolved_max_train_steps / steps_per_epoch
        )
    return (
        gradient_accumulation_steps,
        samples_per_epoch,
        per_rank_samples_per_epoch,
        micro_batches_per_epoch,
        steps_per_epoch,
        resolved_max_train_steps,
        resolved_num_train_epochs,
    )


def _launch_eval(
    *,
    target_model_name_or_path: str,
    checkpoint_dir: str,
    step: int,
    tensorboard_dir: str,
    exp_name: str,
) -> None:
    print("You can use this function to launch your auto eval script!")

class BaseTrainer:
    data_collator_cls = None

    def __init__(self, local_rank, args):
        self.args = args
        self.device, self.global_rank, self.world_size = init_dist(local_rank)
        self.precision_dtype = _PRECISION_DTYPES[self.args.train.precision]
        self.checkpoint_dir_root = self.args.logging.checkpoint_dir
        self.resume_checkpoint_dir = discover_latest_checkpoint(
            self.checkpoint_dir_root
        )
        self.suspend_controller = SuspendController(device=self.device)
        self.next_micro_step = 0

        if is_global_main_process(): ensure_dir(self.checkpoint_dir_root)
        training_logger.init(
            logging_steps=int(self.args.logging.logging_steps),
            tensorboard_dir=self.args.logging.tensorboard_dir,
        )

        if self.resume_checkpoint_dir is not None:
            validate_resume_parallelism(
                resume_checkpoint_dir=self.resume_checkpoint_dir,
                global_rank=self.global_rank,
                world_size=self.world_size,
                sharding_strategy=str(self.args.train.sharding_strategy),
            )

        self.draft_model, self.tokenizer = self.build_models()
        if self.resume_checkpoint_dir is not None:
            self.draft_model = self.resume_draft_model(self.resume_checkpoint_dir)
        self.model = self.draft_model
        if self.args.train.torch_compile:
            print_on_local_main("Compiling training model with torch.compile...")
            self.model = torch.compile(self.model, dynamic=True)
        self.model = self._wrap_with_fsdp(self.model)

        self.train_dataset = self.build_train_dataset()

        (
            self.gradient_accumulation_steps,
            self.samples_per_epoch,
            self.per_rank_samples_per_epoch,
            self.micro_batches_per_epoch,
            self.steps_per_epoch,
            self.max_train_steps,
            self.args.train.num_train_epochs,
        ) = _compute_training_schedule(
            world_size=self.world_size,
            dataset_size=len(self.train_dataset),
            local_batch_size=int(self.args.train.local_batch_size),
            global_batch_size=int(self.args.train.global_batch_size),
            num_train_epochs=int(self.args.train.num_train_epochs),
            max_train_steps=self.args.train.max_train_steps,
        )

        # The cosine anneals to zero at `lr_schedule_steps`, which may exceed
        # max_train_steps (a run stopped early on a longer schedule).  Defaults to
        # max_train_steps.
        _sched_steps = getattr(self.args.train, "lr_schedule_steps", None)
        _sched_steps = self.max_train_steps if _sched_steps is None else int(_sched_steps)
        assert _sched_steps >= self.max_train_steps, (
            f"lr_schedule_steps {_sched_steps} < max_train_steps "
            f"{self.max_train_steps}: the schedule would end before the run does."
        )
        self.lr_schedule_steps = _sched_steps
        self.optimizer = BF16Optimizer(
            self.draft_model,
            lr=float(self.args.train.lr),
            total_steps=_sched_steps,
            warmup_ratio=float(self.args.train.warmup_ratio),
            weight_decay=float(self.args.train.weight_decay),
        )
        if self.resume_checkpoint_dir is not None:
            resume_state = load_training_state(
                resume_checkpoint_dir=self.resume_checkpoint_dir,
                optimizer=self.optimizer,
                global_rank=self.global_rank,
                world_size=self.world_size,
                local_batch_size=int(self.args.train.local_batch_size),
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                micro_batches_per_epoch=self.micro_batches_per_epoch,
                sharding_strategy=str(self.args.train.sharding_strategy),
            )
            self.next_micro_step = resume_state.next_micro_step
            self._resumed_extra_state = dict(resume_state.extra_state)
            # Opt-in: turn the remaining budget into a cosine anneal to zero without
            # touching the optimizer moments (the checkpoint's own schedule otherwise
            # wins on resume).  Off for an ordinary crash-resume.
            if bool(getattr(self.args.train, "lr_cooldown_on_resume", False)):
                remaining = int(self.max_train_steps) - int(self.global_step)
                assert remaining > 0, (
                    f"lr_cooldown_on_resume with {remaining} steps left: the resume "
                    f"point ({self.global_step}) is at or past max_train_steps "
                    f"({self.max_train_steps})"
                )
                print_on_local_main(
                    f"[cooldown] resumed at step {self.global_step}; annealing the "
                    f"remaining {remaining} steps to lr 0"
                )
                self.optimizer.restart_cosine(remaining)
        else:
            print_on_local_main("Training from scratch.")
        self.info_board()

    @property
    def global_step(self):
        return self.next_micro_step // self.gradient_accumulation_steps

    def maybe_log_sampler_stats(self, *, input_ids, loss_mask, outputs) -> None:
        """Anchor-sampler diagnostics, outside the compiled forward, every
        `sampler_stats_stride` micro-batches."""
        stride = int(getattr(self.draft_model.config, "sampler_stats_stride", 1))
        if stride > 1 and self.next_micro_step % stride != 0:
            return
        log_sampler_stats(
            seq_len=int(input_ids.shape[1]),
            loss_mask=loss_mask,
            eval_mask=outputs.eval_mask,
            block_keep_mask=outputs.block_keep_mask,
            block_size=int(self.draft_model.config.block_size),
            num_anchors=int(self.draft_model.config.num_anchors),
        )

    def info_board(self):
        print_on_local_main("***** Running training *****")
        print_on_local_main(f"  Train dataset size = {len(self.train_dataset)}")
        print_on_local_main(f"  Num train epochs = {self.args.train.num_train_epochs}")
        print_on_local_main(f"  Samples per epoch = {self.samples_per_epoch}")
        print_on_local_main(f"  Local batch size = {self.args.train.local_batch_size}")
        print_on_local_main(f"  Global batch size = {self.args.train.global_batch_size}")
        print_on_local_main(f"  Gradient accumulation steps = {self.gradient_accumulation_steps}")
        print_on_local_main(f"  Steps per epoch = {self.steps_per_epoch}")
        print_on_local_main(f"  Max train steps = {self.max_train_steps}")
        print_on_local_main(
            f"  LR schedule steps = {getattr(self, 'lr_schedule_steps', self.max_train_steps)}"
        )

    def build_models(self):
        model_args = self.args.model

        tokenizer = AutoTokenizer.from_pretrained(
            model_args.target_model_name_or_path,
        )
        target_config = AutoConfig.from_pretrained(
            model_args.target_model_name_or_path,
        )

        draft_model = self._build_draft_model(
            target_config=target_config,
            model_args=model_args,
        )
        draft_model = draft_model.to(device=self.device, dtype=self.precision_dtype)

        # Training only uses the target checkpoint to initialize frozen draft
        # embeddings and lm_head weights.
        target_model = AutoModelForCausalLM.from_pretrained(
            model_args.target_model_name_or_path,
            dtype=self.precision_dtype,
        ).to(device="cpu").eval()
        target_embed_tokens = target_model.get_input_embeddings()
        target_lm_head = target_model.get_output_embeddings()
        assert (target_lm_head is not None) and (target_embed_tokens is not None)
        draft_model.initialize_embeddings_and_head(
            embed_tokens=target_embed_tokens,
            lm_head=target_lm_head,
            freeze=True,
        )
        del target_model
        return draft_model, tokenizer

    def _build_draft_model(self, *, target_config, model_args):
        raise NotImplementedError

    def build_train_dataset(self):
        """Default: the precomputed target-hidden cache.  Online-feature trainers
        override this with a token dataset."""
        ds = CacheDataset(cache_dir=self.args.data.target_cache_path)
        validate_train_cache(
            train_dataset=ds,
            draft_model=self.draft_model,
            target_model_name_or_path=self.args.model.target_model_name_or_path,
        )
        return ds

    def resume_draft_model(self, resume_checkpoint_dir):
        """Default: rebuild the draft model with from_pretrained."""
        return load_resume_draft_model(
            resume_checkpoint_dir=resume_checkpoint_dir,
            draft_model=self.draft_model,
            device=self.device,
            precision_dtype=self.precision_dtype,
            global_rank=self.global_rank,
        )

    def _frozen_modules_to_ignore(self):
        """Top-level submodules whose parameters are all frozen."""
        root = getattr(self, "draft_model", None)
        if root is None:
            return []
        out = []
        for _, child in root.named_children():
            params = list(child.parameters())
            if params and not any(p.requires_grad for p in params):
                out.append(child)
        return out

    def _wrap_with_fsdp(self, model):
        fsdp_kwargs = _build_fsdp_kwargs(
            sharding_strategy_name=self.args.train.sharding_strategy,
            precision_dtype=self.precision_dtype,
            world_size=self.world_size,
        )
        # Opt-in: keep the frozen embedding and lm_head out of FSDP's flat
        # parameter.  FSDP flattens the whole model into one FlatParameter that
        # requires grad if any member does, so otherwise the frozen tables take part
        # in every gradient operation.  Under NO_SHARD ignored parameters are
        # replicated, as every parameter already is, so this costs no memory.
        if getattr(self.args.train, "fsdp_ignore_frozen", False):
            assert str(self.args.train.sharding_strategy) == "no_shard", (
                "fsdp_ignore_frozen replicates the ignored parameters instead of "
                "sharding them; only NO_SHARD already does that. Got "
                f"{self.args.train.sharding_strategy!r}"
            )
            ignored = self._frozen_modules_to_ignore()
            if ignored:
                n = sum(p.numel() for m in ignored for p in m.parameters())
                total = sum(p.numel() for p in self.draft_model.parameters())
                print_on_local_main(
                    f"[fsdp] ignoring {len(ignored)} fully-frozen submodules "
                    f"({n:,} of {total:,} params = {100 * n / max(total, 1):.1f}%): "
                    + ", ".join(type(m).__name__ for m in ignored)
                )
                fsdp_kwargs["ignored_modules"] = ignored
        return FSDP(model, **fsdp_kwargs)

    def _build_train_dataloader(self, start_offset_samples=0, num_samples=None):
        sampler = StatelessResumableDistributedSampler(
            dataset=self.train_dataset,
            num_replicas=self.world_size,
            rank=self.global_rank,
            total_size=self.samples_per_epoch,
            start_global_offset_samples=start_offset_samples,
            num_samples=num_samples,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.args.train.local_batch_size),
            sampler=sampler,
            collate_fn=self.data_collator_cls(),
            num_workers=int(self.args.data.num_workers),
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=4,
        )

    def run_batch(self, batch):
        raise NotImplementedError

    def _checkpoint_kwargs(self):
        return dict(
            model=self.model,
            draft_model=self.draft_model,
            optimizer=self.optimizer,
            checkpoint_dir_root=self.checkpoint_dir_root,
            train_config=self.args,
            next_micro_step=self.next_micro_step,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            global_rank=self.global_rank,
            world_size=self.world_size,
            local_batch_size=int(self.args.train.local_batch_size),
            extra_state=self.on_save_extra_state(),
        )

    def on_save_extra_state(self) -> dict:
        """Objective-owned state to persist alongside the optimizer (default: none)."""
        return {}

    def save_and_eval_checkpoint(self):
        checkpoint_dir = save_checkpoint(**self._checkpoint_kwargs())
        if is_global_main_process():
            _launch_eval(
                target_model_name_or_path=self.args.model.target_model_name_or_path,
                checkpoint_dir=checkpoint_dir,
                step=self.global_step,
                tensorboard_dir=self.args.logging.tensorboard_dir,
                exp_name=self.args.exp_name,
            )
        dist.barrier()
        return checkpoint_dir

    def _save_and_suspend(self):
        print_on_global_main("Saving checkpoint before suspending...")
        save_checkpoint(**self._checkpoint_kwargs())
        dist.barrier()
        if is_global_main_process():
            print_on_global_main("Going to suspend...")
            self.suspend_controller.go_suspend()
        dist.barrier()

    def train(self):
        self.model.train()
        if self.global_step >= self.max_train_steps:
            return

        local_batch_size = int(self.args.train.local_batch_size)
        total_micro_steps = self.max_train_steps * self.gradient_accumulation_steps
        remaining_micro_steps = total_micro_steps - self.next_micro_step
        remaining_samples = remaining_micro_steps * local_batch_size

        dataloader = self._build_train_dataloader(
            start_offset_samples=self.next_micro_step * local_batch_size,
            num_samples=remaining_samples,
        )
        prefetcher = CUDAPrefetcher(dataloader, self.device)
        training_logger.start_session(global_step=self.global_step)

        with self.suspend_controller.monitoring():
            for batch in prefetcher:
                should_sync = (
                    (self.next_micro_step + 1) % self.gradient_accumulation_steps == 0
                )
                sync_context = nullcontext() if should_sync else self.model.no_sync()
                with sync_context:
                    loss = self.run_batch(batch) / self.gradient_accumulation_steps
                    loss.backward()
                self.next_micro_step += 1

                if not should_sync:
                    continue

                # Objectives with a per-optimizer-step statistic (window
                # normalisation) close it here: after FSDP's gradient reduction on
                # the last micro-batch, before clipping and the update.
                boundary_hook = getattr(self, "on_optimizer_boundary", None)
                if callable(boundary_hook):
                    boundary_hook()

                grad_norm = FSDP.clip_grad_norm_(
                    self.model,
                    float(self.args.train.max_grad_norm),
                )
                self.optimizer.step()
                _now = time.perf_counter()
                _prev = getattr(self, "_last_step_wall", None)
                self._last_step_wall = _now
                if _prev is not None:
                    add_metric("step_seconds", _now - _prev, reduction="last",
                               tag="train")
                training_logger.on_optimizer_step(
                    global_step=self.global_step,
                    next_micro_step=self.next_micro_step,
                    micro_batches_per_epoch=self.micro_batches_per_epoch,
                    max_train_steps=self.max_train_steps,
                    learning_rate=self.optimizer.get_learning_rate(),
                    grad_norm=grad_norm.item(),
                )

                if self.global_step % int(self.args.logging.checkpointing_steps) == 0:
                    self.save_and_eval_checkpoint()

                if self.suspend_controller.requested():
                    self._save_and_suspend()
                    return

        _every = int(self.args.logging.checkpointing_steps)
        if _every > 0 and self.global_step % _every == 0:
            # The loop above already saved this step.
            print_on_local_main(f"[dspark] step {self.global_step} was already saved by "
                                f"the periodic checkpoint; skipping the end-of-run save")
        else:
            self.save_and_eval_checkpoint()

    def clean_up(self):
        training_logger.close()
        dist.barrier()
        dist.destroy_process_group()
