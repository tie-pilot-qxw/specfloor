"""Under no_shard only rank 0's optimizer state is written (every rank's copy is
identical); these tests pin the rank-0 fallback on resume and its guard."""
import json
import os

import pytest

from deepspec.trainer.ckpt_manager import (
    _rank_training_state_path,
    _resolve_training_state_path,
    _validate_rank0_state_fallback,
)


def _ckpt(tmp_path, ranks, strategy="no_shard"):
    d = tmp_path / "step_100"
    d.mkdir()
    for r in ranks:
        (d / f"training_state.rank{r}.pt").write_bytes(b"x" * 16)
    (d / "training_metadata.json").write_text(json.dumps(
        {"world_size": max(ranks) + 1, "local_batch_size": 1,
         "sharding_strategy": strategy}))
    return str(d)


def test_a_missing_rank_falls_back_to_rank0(tmp_path):
    """Deleting rank1..N is only safe because every rank can still find a file."""
    d = _ckpt(tmp_path, [0])
    for r in (0, 1, 3, 7):
        got = _resolve_training_state_path(d, r)
        assert got == _rank_training_state_path(d, 0 if r else r), r


def test_a_rank_with_its_own_file_does_not_fall_back(tmp_path):
    d = _ckpt(tmp_path, [0, 1])
    assert _resolve_training_state_path(d, 1) == _rank_training_state_path(d, 1)


def test_no_rank0_and_no_own_file_is_an_error_not_a_guess(tmp_path):
    d = _ckpt(tmp_path, [1])
    with pytest.raises(FileNotFoundError):
        _resolve_training_state_path(d, 3)


@pytest.mark.parametrize("saved,current,ok", [
    ("no_shard", "no_shard", True),
    ("full_shard", "no_shard", False),
    ("no_shard", "full_shard", False),
    (None, "no_shard", False),          # legacy checkpoint: unknown is not yes
])
def test_the_fallback_is_refused_unless_both_sides_are_no_shard(saved, current, ok):
    """A sharded checkpoint's rank files hold DIFFERENT shards, so reading rank0 in
    place of rank3 would silently load the wrong parameters' optimizer moments."""
    call = dict(used_rank0_fallback=True, saved_sharding_strategy=saved,
                current_sharding_strategy=current)
    if ok:
        _validate_rank0_state_fallback(**call)
    else:
        with pytest.raises(RuntimeError):
            _validate_rank0_state_fallback(**call)
