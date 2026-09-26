import json
from types import SimpleNamespace

import pytest
import torch

from deepspec.trainer.ckpt_manager import (
    TRAINING_METADATA_FILE_NAME,
    load_resume_draft_model,
    validate_resume_parallelism,
)


def _write_checkpoint_metadata(path, *, world_size, sharding_strategy):
    torch.save(
        {
            "world_size": world_size,
            "sharding_strategy": sharding_strategy,
        },
        path / "training_state.rank0.pt",
    )
    with open(path / TRAINING_METADATA_FILE_NAME, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "world_size": world_size,
                "local_batch_size": 1,
                "sharding_strategy": sharding_strategy,
            },
            handle,
        )


class _FakeDraftModel:
    loaded_from = None

    def __init__(self):
        self.config = SimpleNamespace(_attn_implementation="eager")
        self.embedding_head_trainable = True

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.loaded_from = (path, kwargs)
        return cls()

    def to(self, **kwargs):
        return self

    def set_embedding_head_trainable(self, value):
        self.embedding_head_trainable = value


def test_new_rank_uses_rank0_state_before_model_load(tmp_path):
    _write_checkpoint_metadata(
        tmp_path,
        world_size=2,
        sharding_strategy="no_shard",
    )

    validate_resume_parallelism(
        resume_checkpoint_dir=str(tmp_path),
        global_rank=3,
        world_size=4,
        sharding_strategy="no_shard",
    )
    resumed = load_resume_draft_model(
        resume_checkpoint_dir=str(tmp_path),
        draft_model=_FakeDraftModel(),
        device="cpu",
        precision_dtype=torch.float32,
        global_rank=3,
    )

    assert _FakeDraftModel.loaded_from[0] == str(tmp_path)
    assert resumed.embedding_head_trainable is False


def test_legacy_config_can_confirm_no_shard_scale_up(tmp_path):
    torch.save(
        {"world_size": 2},
        tmp_path / "training_state.rank0.pt",
    )
    (tmp_path / "train_config.py").write_text(
        "train = dict(trainer_cls=UnknownTrainer, sharding_strategy='no_shard')\n",
        encoding="utf-8",
    )

    validate_resume_parallelism(
        resume_checkpoint_dir=str(tmp_path),
        global_rank=3,
        world_size=4,
        sharding_strategy="no_shard",
    )


@pytest.mark.parametrize(
    ("saved_strategy", "current_strategy"),
    [
        ("full_shard", "full_shard"),
        ("shard_grad_op", "no_shard"),
        ("no_shard", "hybrid_shard"),
        (None, "no_shard"),
    ],
)
def test_cross_world_resume_rejects_unconfirmed_or_sharded_state(
    tmp_path,
    saved_strategy,
    current_strategy,
):
    _write_checkpoint_metadata(
        tmp_path,
        world_size=2,
        sharding_strategy=saved_strategy,
    )

    with pytest.raises(RuntimeError, match="different world size"):
        validate_resume_parallelism(
            resume_checkpoint_dir=str(tmp_path),
            global_rank=3,
            world_size=4,
            sharding_strategy=current_strategy,
        )


def test_missing_rank_state_rejects_sharded_rank0_fallback(tmp_path):
    _write_checkpoint_metadata(
        tmp_path,
        world_size=2,
        sharding_strategy="full_shard",
    )

    with pytest.raises(RuntimeError, match="rank-0 training-state fallback"):
        validate_resume_parallelism(
            resume_checkpoint_dir=str(tmp_path),
            global_rank=1,
            world_size=2,
            sharding_strategy="full_shard",
        )
