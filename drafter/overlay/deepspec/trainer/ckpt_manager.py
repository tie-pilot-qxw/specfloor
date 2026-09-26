import ast
import json
import os
import random
import shutil
from typing import Optional

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType

from deepspec.utils import (
    ensure_dir,
    is_global_main_process,
    print_on_global_main,
    print_on_local_main,
    safe_symlink,
)


TRAIN_CONFIG_FILE_NAME = "train_config.py"
TRAINING_METADATA_FILE_NAME = "training_metadata.json"


def discover_latest_checkpoint(checkpoint_dir):
    latest_link = os.path.join(checkpoint_dir, "step_latest")
    if not (os.path.islink(latest_link) or os.path.isdir(latest_link)):
        return None
    return os.path.realpath(latest_link)


def save_train_config(*, train_config, checkpoint_dir: str) -> str:
    dest_path = os.path.join(checkpoint_dir, TRAIN_CONFIG_FILE_NAME)
    if not is_global_main_process():
        return dest_path

    ensure_dir(checkpoint_dir)
    shutil.copy(train_config._origin_config_path, dest_path)
    opts = train_config._origin_opts
    if opts:
        with open(dest_path, "a", encoding="utf-8") as handle:
            handle.write("\n\n# --opts overrides applied at save time\n")
            for opt in opts:
                handle.write(_render_opt_assignment(opt) + "\n")
    return dest_path


def _render_opt_assignment(opt: str) -> str:
    key, raw_value = opt.split("=", 1)
    head, *rest = key.split(".")
    accessors = "".join(f"[{part!r}]" for part in rest)
    value = yaml.safe_load(raw_value)
    return f"{head}{accessors} = {value!r}"


@dataclass(frozen=True)
class TrainingResumeState:
    # next_micro_step is the single source of truth for training progress;
    # global_step and current_epoch are derived from it together with
    # gradient_accumulation_steps / micro_batches_per_epoch.
    next_micro_step: int
    # Objective-owned state that must survive a resume, keyed by owner.  Empty
    # for every checkpoint written before this existed, and empty for arms that
    # do not carry any -- so a missing key is a normal resume, not an error.
    extra_state: dict = field(default_factory=dict)


def _resolve_training_state_path(
    resume_checkpoint_dir: str,
    global_rank: int,
) -> str:
    """Resolve replicated no-shard state for existing and newly added ranks."""
    rank_path = _rank_training_state_path(resume_checkpoint_dir, global_rank)
    if os.path.exists(rank_path):
        return rank_path
    rank0_path = _rank_training_state_path(resume_checkpoint_dir, 0)
    if os.path.exists(rank0_path):
        return rank0_path
    raise FileNotFoundError(
        "resume checkpoint has no training state for the requested rank and no "
        f"rank-0 fallback: requested={rank_path}, fallback={rank0_path}"
    )


def _validate_cross_world_resume(
    *,
    saved_world_size: int,
    current_world_size: int,
    saved_sharding_strategy,
    current_sharding_strategy: str,
) -> None:
    if int(saved_world_size) == int(current_world_size):
        return
    saved_strategy = (
        None
        if saved_sharding_strategy is None
        else str(saved_sharding_strategy).lower()
    )
    current_strategy = str(current_sharding_strategy).lower()
    if saved_strategy != "no_shard" or current_strategy != "no_shard":
        raise RuntimeError(
            "resume with a different world size is supported only when both "
            "the saved and current sharding strategies are confirmed as "
            f"'no_shard'; saved_world_size={saved_world_size}, "
            f"current_world_size={current_world_size}, "
            f"saved_sharding_strategy={saved_strategy!r}, "
            f"current_sharding_strategy={current_strategy!r}. Legacy "
            "checkpoints without sharding metadata cannot be safely resharded."
        )


def _validate_rank0_state_fallback(
    *,
    used_rank0_fallback: bool,
    saved_sharding_strategy,
    current_sharding_strategy: str,
) -> None:
    if not used_rank0_fallback:
        return
    saved_strategy = (
        None
        if saved_sharding_strategy is None
        else str(saved_sharding_strategy).lower()
    )
    current_strategy = str(current_sharding_strategy).lower()
    if saved_strategy != "no_shard" or current_strategy != "no_shard":
        raise RuntimeError(
            "rank-0 training-state fallback is safe only for confirmed "
            "replicated no_shard optimizer state; "
            f"saved_sharding_strategy={saved_strategy!r}, "
            f"current_sharding_strategy={current_strategy!r}."
        )


def _extract_legacy_sharding_strategy(resume_checkpoint_dir: str):
    """Read a literal train.sharding_strategy without executing saved config."""
    config_path = os.path.join(resume_checkpoint_dir, TRAIN_CONFIG_FILE_NAME)
    try:
        with open(config_path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=config_path)
    except (OSError, SyntaxError):
        return None

    strategy = None
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        for target in targets:
            if isinstance(target, ast.Name) and target.id == "train":
                if (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Name)
                    and value.func.id == "dict"
                ):
                    for keyword in value.keywords:
                        if keyword.arg == "sharding_strategy":
                            try:
                                strategy = ast.literal_eval(keyword.value)
                            except (ValueError, TypeError):
                                strategy = None
                elif isinstance(value, ast.Dict):
                    for key, item in zip(value.keys, value.values):
                        try:
                            key_value = ast.literal_eval(key)
                        except (ValueError, TypeError):
                            continue
                        if key_value == "sharding_strategy":
                            try:
                                strategy = ast.literal_eval(item)
                            except (ValueError, TypeError):
                                strategy = None
            elif (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "train"
            ):
                try:
                    key_value = ast.literal_eval(target.slice)
                except (ValueError, TypeError):
                    continue
                if key_value == "sharding_strategy":
                    try:
                        strategy = ast.literal_eval(value)
                    except (ValueError, TypeError):
                        strategy = None
    return None if strategy is None else str(strategy).lower()


def _saved_sharding_strategy(resume_checkpoint_dir: str, metadata):
    strategy = metadata.get("sharding_strategy")
    if strategy is not None:
        return str(strategy).lower()
    return _extract_legacy_sharding_strategy(resume_checkpoint_dir)


def validate_resume_parallelism(
    *,
    resume_checkpoint_dir: str,
    global_rank: int,
    world_size: int,
    sharding_strategy: str,
) -> None:
    """Fail before model/FSDP construction on an unsafe cross-world resume."""
    requested_state_path = _rank_training_state_path(
        resume_checkpoint_dir,
        global_rank,
    )
    state_path = _resolve_training_state_path(
        resume_checkpoint_dir,
        global_rank,
    )
    metadata_path = os.path.join(
        resume_checkpoint_dir,
        TRAINING_METADATA_FILE_NAME,
    )
    if os.path.exists(metadata_path):
        with open(metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
    else:
        # Legacy fallback. mmap prevents eagerly copying the potentially large
        # optimizer tensors merely to inspect the small resume metadata.
        metadata = torch.load(
            state_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    saved_sharding_strategy = _saved_sharding_strategy(
        resume_checkpoint_dir,
        metadata,
    )
    _validate_cross_world_resume(
        saved_world_size=int(metadata["world_size"]),
        current_world_size=int(world_size),
        saved_sharding_strategy=saved_sharding_strategy,
        current_sharding_strategy=sharding_strategy,
    )
    _validate_rank0_state_fallback(
        used_rank0_fallback=state_path != requested_state_path,
        saved_sharding_strategy=saved_sharding_strategy,
        current_sharding_strategy=sharding_strategy,
    )


def load_resume_draft_model(
    *,
    resume_checkpoint_dir: str,
    draft_model,
    device,
    precision_dtype,
    global_rank: int,
):
    # A newly added rank in a no_shard scale-up legitimately has no matching
    # state file; rank 0's replicated state is the fallback used later by the
    # optimizer/data-state loader as well.
    _resolve_training_state_path(resume_checkpoint_dir, global_rank)
    resumed_model = type(draft_model).from_pretrained(
        resume_checkpoint_dir,
        dtype=precision_dtype,
        attn_implementation=str(draft_model.config._attn_implementation),
    )
    resumed_model = resumed_model.to(device=device, dtype=precision_dtype)
    resumed_model.set_embedding_head_trainable(False)
    return resumed_model


def load_training_state(
    *,
    resume_checkpoint_dir: str,
    optimizer,
    global_rank: int,
    world_size: int,
    local_batch_size: int,
    gradient_accumulation_steps: int,
    micro_batches_per_epoch: int,
    sharding_strategy: str,
) -> TrainingResumeState:
    # Per-rank training_state files are REPLICATED under no_shard DDP (optimizer state is
    # all-reduced -> identical across ranks), so a rank beyond the saved world can load
    # rank 0's file. That, plus the global-cursor conversion below, is what lets us resume
    # with a DIFFERENT world_size / local_batch (e.g. scale 2 -> 4 GPUs).
    requested_state_path = _rank_training_state_path(
        resume_checkpoint_dir,
        global_rank,
    )
    state_path = _resolve_training_state_path(
        resume_checkpoint_dir,
        global_rank,
    )

    checkpoint = torch.load(state_path, map_location="cpu", weights_only=False)
    saved_next_micro_step = int(checkpoint["next_micro_step"])
    saved_world_size = int(checkpoint["world_size"])
    saved_local_batch_size = int(checkpoint["local_batch_size"])
    saved_rank = int(checkpoint["global_rank"])
    saved_sharding_strategy = _saved_sharding_strategy(
        resume_checkpoint_dir,
        checkpoint,
    )
    _validate_cross_world_resume(
        saved_world_size=saved_world_size,
        current_world_size=world_size,
        saved_sharding_strategy=saved_sharding_strategy,
        current_sharding_strategy=sharding_strategy,
    )
    _validate_rank0_state_fallback(
        used_rank0_fallback=state_path != requested_state_path,
        saved_sharding_strategy=saved_sharding_strategy,
        current_sharding_strategy=sharding_strategy,
    )
    optimizer.load_state_dict(checkpoint["optimizer"])

    # Progress is stored as a per-rank micro-step under the SAVED parallelism. Convert it to a
    # parallelism-invariant GLOBAL sample cursor, then re-derive the per-rank micro-step for the
    # CURRENT world_size/local_batch. The schedule (grad-accum, max steps) is already recomputed
    # for the new parallelism by the caller, so global_step and the data offset stay correct
    # across a change of WORLD x LOCAL_BATCH. Fail loud (never silently mis-train) if the saved
    # cursor cannot land on a clean optimizer-step boundary under the new parallelism.
    global_samples = saved_next_micro_step * saved_world_size * saved_local_batch_size
    denom = int(world_size) * int(local_batch_size)
    assert global_samples % denom == 0, (
        "resume: global sample cursor not divisible by new world_size*local_batch_size "
        f"(global_samples={global_samples}, world_size={world_size}, "
        f"local_batch_size={local_batch_size})."
    )
    next_micro_step = global_samples // denom
    assert next_micro_step % gradient_accumulation_steps == 0, (
        "resume: cursor not aligned to the new gradient-accumulation window "
        f"(next_micro_step={next_micro_step}, "
        f"gradient_accumulation_steps={gradient_accumulation_steps}); choose a parallelism "
        "whose optimizer-step boundary divides the saved global sample cursor."
    )

    torch.set_rng_state(checkpoint["torch_rng"])
    torch.cuda.set_rng_state(checkpoint["torch_cuda_rng"])
    np.random.set_state(checkpoint["numpy_rng"])
    random.setstate(checkpoint["python_rng"])
    if (saved_world_size, saved_local_batch_size, saved_rank) != (
        int(world_size), int(local_batch_size), int(global_rank),
    ):
        # Parallelism changed: several new ranks may share one saved RNG stream (rank-0
        # fallback). Re-seed per rank so corrupt/reveal augmentation is decorrelated. Only
        # augmentation noise depends on this -- the data cursor and optimizer step are
        # deterministic in the global cursor above. (Streams necessarily differ from a
        # same-parallelism resume; acceptable per design.)
        _rank_seed = (next_micro_step * 1_000_003 + int(global_rank)) % (2**63 - 1)
        torch.manual_seed(_rank_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_rank_seed)

    global_step = next_micro_step // gradient_accumulation_steps
    current_epoch = next_micro_step // micro_batches_per_epoch + 1
    print_on_global_main(
        (
            "AUTO-RESUME from "
            f"{resume_checkpoint_dir}, next_micro_step={next_micro_step}, "
            "to force fresh run change exp_name or remove step_latest"
        )
    )
    print_on_local_main(
        f"Resumed from {resume_checkpoint_dir}: "
        f"next_micro_step={next_micro_step}, global_step={global_step}, "
        f"epoch={current_epoch}"
    )
    return TrainingResumeState(
        next_micro_step=next_micro_step,
        extra_state=dict(checkpoint.get("extra_state") or {}),
    )


def save_checkpoint(
    *,
    model,
    draft_model,
    optimizer,
    checkpoint_dir_root: str,
    train_config,
    next_micro_step: int,
    gradient_accumulation_steps: int,
    global_rank: int,
    world_size: int,
    local_batch_size: int,
    extra_state: Optional[dict] = None,
) -> str:
    assert next_micro_step % gradient_accumulation_steps == 0, (
        "next_micro_step must be aligned with gradient_accumulation_steps at "
        f"checkpoint time: next_micro_step={next_micro_step}, "
        f"gradient_accumulation_steps={gradient_accumulation_steps}"
    )
    global_step = next_micro_step // gradient_accumulation_steps
    checkpoint_dir = os.path.join(checkpoint_dir_root, f"step_{global_step}")
    # Optional checkpoint thinning (default off):
    #   logging.keep_last_n_checkpoints -> drop optimizer state of older step_* dirs
    keep_last_n = _ckpt_flag(train_config, "logging", "keep_last_n_checkpoints", None)
    #   logging.keep_optimizer_every  -> never thin a checkpoint whose step is a
    #                                    multiple of this (i.e. epoch boundaries)
    keep_opt_every = _ckpt_flag(train_config, "logging", "keep_optimizer_every", None)
    if is_global_main_process():
        ensure_dir(checkpoint_dir)
        save_train_config(train_config=train_config, checkpoint_dir=checkpoint_dir)
        with open(
            os.path.join(checkpoint_dir, TRAINING_METADATA_FILE_NAME),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                {
                    "world_size": int(world_size),
                    "local_batch_size": int(local_batch_size),
                    "sharding_strategy": str(
                        _ckpt_flag(
                            train_config,
                            "train",
                            "sharding_strategy",
                            "unknown",
                        )
                    ).lower(),
                },
                handle,
            )
    dist.barrier()
    _save_model_checkpoint(
        model=model,
        draft_model=draft_model,
        checkpoint_dir=checkpoint_dir,
    )
    training_state = _serialize_training_state(
        optimizer=optimizer,
        next_micro_step=next_micro_step,
        gradient_accumulation_steps=gradient_accumulation_steps,
        global_rank=global_rank,
        world_size=world_size,
        local_batch_size=local_batch_size,
        extra_state=extra_state,
        sharding_strategy=str(
            _ckpt_flag(
                train_config,
                "train",
                "sharding_strategy",
                "unknown",
            )
        ).lower(),
    )
    # One file, not one per rank, when the state is replicated: under no_shard every
    # rank's optimizer state is identical, and the loader falls back to rank 0's file
    # (only when both the saved and the current strategy are no_shard).  Sharded
    # strategies write every rank's file.
    state_is_replicated = str(
        _ckpt_flag(train_config, "train", "sharding_strategy", "unknown")
    ).lower() == "no_shard"
    if not state_is_replicated or global_rank == 0:
        torch.save(
            training_state,
            _rank_training_state_path(checkpoint_dir, global_rank),
        )
    dist.barrier()
    if is_global_main_process():
        safe_symlink(
            checkpoint_dir,
            os.path.join(checkpoint_dir_root, "step_latest"),
        )
        print_on_global_main(f"Saved checkpoint to {checkpoint_dir}")
        if keep_last_n:
            _prune_old_checkpoints(checkpoint_dir_root, int(keep_last_n),
                                   keep_optimizer_every=keep_opt_every)
    dist.barrier()
    return checkpoint_dir


def _ckpt_flag(train_config, section, key, default):
    try:
        node = train_config[section] if section in train_config else getattr(train_config, section, None)
        if node is None:
            return default
        return node[key] if key in node else default
    except Exception:
        return default


def _prune_old_checkpoints(checkpoint_dir_root: str, keep_last_n: int,
                           keep_optimizer_every: Optional[int] = None) -> None:
    """Keep the last `keep_last_n` checkpoints resumable.  Older step dirs keep their
    model weights (still evaluable) but lose the optimizer/training state, which is
    only needed to resume.  Steps that are multiples of `keep_optimizer_every` (e.g.
    epoch boundaries) stay resumable."""
    import re
    import glob

    steps = []
    for name in os.listdir(checkpoint_dir_root):
        m = re.fullmatch(r"step_(\d+)", name)
        if m:
            steps.append((int(m.group(1)), os.path.join(checkpoint_dir_root, name)))
    steps.sort(reverse=True)
    every = int(keep_optimizer_every) if keep_optimizer_every else 0
    for step, path in steps[keep_last_n:]:
        if every > 0 and step % every == 0:
            continue
        removed = 0
        for state_file in glob.glob(os.path.join(path, "training_state.rank*.pt")):
            try:
                os.remove(state_file)
                removed += 1
            except OSError:
                pass
        if removed:
            print_on_global_main(
                f"Thinned checkpoint {path}: removed {removed} optimizer-state file(s), "
                f"kept model weights (still eval-able, not resumable)"
            )


def _rank_training_state_path(checkpoint_dir: str, global_rank: int) -> str:
    return os.path.join(
        checkpoint_dir,
        f"training_state.rank{int(global_rank)}.pt",
    )


def _serialize_training_state(
    *,
    optimizer,
    next_micro_step: int,
    gradient_accumulation_steps: int,
    global_rank: int,
    world_size: int,
    local_batch_size: int,
    sharding_strategy: str,
    extra_state: Optional[dict] = None,
):
    assert next_micro_step % gradient_accumulation_steps == 0, (
        "next_micro_step must be aligned with gradient_accumulation_steps at "
        f"checkpoint time: next_micro_step={next_micro_step}, "
        f"gradient_accumulation_steps={gradient_accumulation_steps}"
    )
    return {
        "next_micro_step": int(next_micro_step),
        "optimizer": optimizer.state_dict(),
        "global_rank": int(global_rank),
        "world_size": int(world_size),
        "local_batch_size": int(local_batch_size),
        "sharding_strategy": str(sharding_strategy).lower(),
        "extra_state": dict(extra_state or {}),
        "torch_rng": torch.get_rng_state(),
        "torch_cuda_rng": torch.cuda.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }


def _full_model_state_dict(model):
    assert isinstance(model, FSDP), "training model must be wrapped in FSDP"
    state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        state_dict_config,
    ):
        return model.state_dict()


def _save_model_checkpoint(*, model, draft_model, checkpoint_dir: str):
    state_dict = _full_model_state_dict(model)
    if is_global_main_process():
        draft_state_dict = {}
        for key, value in state_dict.items():
            normalized_key = key
            if normalized_key.startswith("_orig_mod."):
                normalized_key = normalized_key[len("_orig_mod.") :]
            draft_state_dict[normalized_key] = value
        assert draft_state_dict, "Failed to extract draft model state_dict from checkpoint."
        draft_model.save_pretrained(checkpoint_dir, state_dict=draft_state_dict)
