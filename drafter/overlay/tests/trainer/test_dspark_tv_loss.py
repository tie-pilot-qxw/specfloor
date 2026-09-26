import torch
from torch import nn

from deepspec.modeling.dspark.common import DSparkForwardOutput
from deepspec.modeling.dspark.loss import (
    _collect_local_terms,
    _compute_l1_dist_per_token,
)
from deepspec.trainer.dspark_online_trainer import OnlineTargetTrainer


def _outputs(draft_logits, target_logits):
    return DSparkForwardOutput(
        draft_logits=draft_logits,
        target_ids=torch.tensor([[[0, 1]]]),
        eval_mask=torch.ones((1, 1, 2), dtype=torch.bool),
        block_keep_mask=torch.ones((1, 1), dtype=torch.bool),
        aligned_target_logits=target_logits,
    )


def test_l1_term_is_full_vocab_distribution_distance_and_has_gradient():
    draft_logits = torch.tensor(
        [[[[2.0, 0.0, -1.0], [0.0, 1.0, -1.0]]]],
        requires_grad=True,
    )
    target_logits = torch.tensor(
        [[[[1.0, 0.5, -0.5], [0.0, -1.0, 1.0]]]]
    )
    outputs = _outputs(draft_logits, target_logits)

    expected = (
        draft_logits.softmax(-1) - target_logits.softmax(-1)
    ).abs().sum(-1)
    actual, buckets = _compute_l1_dist_per_token(
        outputs=outputs,
        aligned_target_logits=target_logits,
    )
    torch.testing.assert_close(actual, expected)

    # The rank buckets ride along on the same probability tensors.  Three things
    # are pinned because a wrong one would look plausible on a curve:
    #   * ranks are the TARGET's, so rq2 is the drafter's mass on p's RUNNER-UP,
    #     not on the drafter's own second choice.  Getting this backwards would
    #     silently measure the wrong axis -- the whole point is the coordinate
    #     hard CE drains (gradient +q_v on every non-label token).
    #   * ovl1 + ovlge2 == 1 - l1/2 == accept_rate exactly, so the split is a
    #     partition of the logged acceptance and not an independent estimate.
    #   * the buckets are detached; they must never add a gradient path.
    p = target_logits.softmax(-1)
    q = draft_logits.softmax(-1)
    p_top, idx = p.topk(2, dim=-1)
    torch.testing.assert_close(buckets["rp1"], p_top[..., 0])
    torch.testing.assert_close(buckets["rp2"], p_top[..., 1])
    torch.testing.assert_close(buckets["rq1"], q.gather(-1, idx)[..., 0].detach())
    torch.testing.assert_close(buckets["rq2"], q.gather(-1, idx)[..., 1].detach())
    torch.testing.assert_close(
        buckets["ovl1"] + buckets["ovlge2"], (1.0 - 0.5 * actual).detach()
    )
    torch.testing.assert_close(
        buckets["ovl1"], torch.minimum(p_top[..., 0], q.gather(-1, idx)[..., 0]).detach()
    )
    for name, value in buckets.items():
        assert not value.requires_grad, f"bucket {name} must not carry a gradient"

    terms, _ = _collect_local_terms(
        outputs=outputs,
        loss_decay_gamma=None,
        l1_loss_alpha=0.9,
    )
    torch.testing.assert_close(terms["l1_loss_num"], expected.sum())
    terms["l1_loss_num"].backward()
    assert draft_logits.grad is not None
    assert torch.isfinite(draft_logits.grad).all()


def test_positive_l1_weight_requires_teacher_logits():
    outputs = _outputs(torch.zeros(1, 1, 2, 3), None)
    try:
        _collect_local_terms(
            outputs=outputs,
            loss_decay_gamma=None,
            l1_loss_alpha=0.9,
        )
    except AssertionError as error:
        assert "aligned_target_logits" in str(error)
    else:
        raise AssertionError("missing teacher logits should fail loudly")


class _AddLayer(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = float(value)

    def forward(self, hidden_states):
        return (hidden_states + self.value,)


class _TinyTargetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 2)
        self.layers = nn.ModuleList([_AddLayer(1.0), _AddLayer(2.0)])

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)[0]
        return (hidden,)


def test_online_target_features_reuses_one_forward_for_taps_and_final_hidden():
    trainer = object.__new__(OnlineTargetTrainer)
    trainer.target_backbone = _TinyTargetBackbone()
    trainer.model_target_layer_ids = [-1, 0, 1]
    trainer.precision_dtype = torch.float32
    input_ids = torch.tensor([[1, 2, 3]])

    tapped, final_hidden = trainer._online_target_features(
        input_ids,
        torch.ones_like(input_ids),
    )

    embedded = trainer.target_backbone.embed_tokens(input_ids)
    torch.testing.assert_close(final_hidden, embedded + 3.0)
    torch.testing.assert_close(
        tapped,
        torch.cat([embedded, embedded + 1.0, embedded + 3.0], dim=-1),
    )
