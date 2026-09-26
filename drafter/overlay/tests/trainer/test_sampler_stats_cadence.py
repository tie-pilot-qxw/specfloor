from types import SimpleNamespace
from unittest.mock import patch

import torch

from deepspec.trainer.base_trainer import BaseTrainer


def _trainer(*, next_micro_step: int, stride: int) -> BaseTrainer:
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.next_micro_step = next_micro_step
    trainer.draft_model = SimpleNamespace(
        config=SimpleNamespace(
            sampler_stats_stride=stride,
            block_size=16,
            num_anchors=32,
        )
    )
    return trainer


def test_sampler_stats_cadence_is_owned_by_the_trainer():
    outputs = SimpleNamespace(
        eval_mask=torch.ones(1, 2, 16, dtype=torch.bool),
        block_keep_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    input_ids = torch.ones(1, 64, dtype=torch.long)
    loss_mask = torch.ones(1, 64)

    with patch("deepspec.trainer.base_trainer.log_sampler_stats") as log_stats:
        _trainer(next_micro_step=49, stride=50).maybe_log_sampler_stats(
            input_ids=input_ids,
            loss_mask=loss_mask,
            outputs=outputs,
        )
        log_stats.assert_not_called()

        _trainer(next_micro_step=50, stride=50).maybe_log_sampler_stats(
            input_ids=input_ids,
            loss_mask=loss_mask,
            outputs=outputs,
        )
        log_stats.assert_called_once_with(
            seq_len=64,
            loss_mask=loss_mask,
            eval_mask=outputs.eval_mask,
            block_keep_mask=outputs.block_keep_mask,
            block_size=16,
            num_anchors=32,
        )
