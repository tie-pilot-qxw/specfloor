"""Correctness of the greedy (temp-0) training metrics.

These tags are what future architecture arms will be compared on EARLY, before
any of them is trained long enough for an offline accept-length diagnostic.  A
silently wrong tag would not crash anything -- it would just make two
architectures look different (or identical) for a reason that is not the
architecture.  So the accept-length recurrence, the rank trick and the mask
handling are each pinned against hand-computed values.
"""
import torch

from deepspec.modeling.dspark.loss import _compute_greedy_stats


def _call(logits, target_ids, eval_mask, tgt_logits=None, block_weights=None):
    if block_weights is None:
        block_weights = torch.ones(logits.shape[:2]).reshape(logits.shape[0], -1)
    return _compute_greedy_stats(
        draft_logits=logits,
        target_ids=target_ids,
        eval_mask=eval_mask,
        valid_block_weights=block_weights,
        aligned_target_logits=tgt_logits,
    )


def _logits_where(correct, vocab=5, k_rank=0):
    """Build logits whose ground-truth token has rank `k_rank` when not correct."""
    B, NB, K = correct.shape
    logits = torch.zeros(B, NB, K, vocab)
    target = torch.zeros(B, NB, K, dtype=torch.long)
    for b in range(B):
        for n in range(NB):
            for k in range(K):
                target[b, n, k] = 1
                if correct[b, n, k]:
                    logits[b, n, k, 1] = 10.0
                else:
                    # put `k_rank` other tokens above the ground truth
                    logits[b, n, k, 1] = 1.0
                    for j in range(k_rank):
                        logits[b, n, k, 2 + j] = 5.0 + j
    return logits, target


def test_accept_length_is_the_leading_run():
    correct = torch.tensor([[[True, True, False, True]]])
    logits, target = _logits_where(correct, k_rank=1)
    mask = torch.ones_like(correct)
    s = _call(logits, target, mask)
    assert float(s["tau_greedy_sum"]) == 2.0, "run must stop at the first miss"


def test_all_correct_gives_full_block():
    correct = torch.ones(1, 2, 4, dtype=torch.bool)
    logits, target = _logits_where(correct)
    s = _call(logits, target, torch.ones_like(correct))
    assert float(s["tau_greedy_sum"]) == 8.0, "2 blocks x 4 slots"


def test_first_slot_wrong_gives_zero():
    correct = torch.tensor([[[False, True, True, True]]])
    logits, target = _logits_where(correct, k_rank=1)
    s = _call(logits, target, torch.ones_like(correct))
    assert float(s["tau_greedy_sum"]) == 0.0


def test_masked_slots_do_not_break_the_run():
    """A slot outside eval_mask is unsupervised, not a miss."""
    correct = torch.tensor([[[True, False, True, True]]])
    logits, target = _logits_where(correct, k_rank=1)
    mask = torch.tensor([[[True, False, True, True]]])
    s = _call(logits, target, mask)
    # slot 1 is unsupervised => the run continues through it, counting slots 0,2,3
    assert float(s["tau_greedy_sum"]) == 3.0
    assert float(s["pos_counts"][1]) == 0.0, "masked slot must not enter the denominator"


def test_rank_trick_matches_topk():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 4, 50)
    target = torch.randint(0, 50, (2, 3, 4))
    mask = torch.ones(2, 3, 4, dtype=torch.bool)
    s = _call(logits, target, mask)
    for k, key in ((1, "acc_pos"), (5, "top5_pos"), (10, "top10_pos")):
        topk = logits.topk(k, dim=-1).indices
        ref = (topk == target.unsqueeze(-1)).any(-1).float().sum(dim=(0, 1))
        assert torch.allclose(s[key], ref), f"top-{k} disagrees with torch.topk"


def test_target_ruler_differs_from_gt_ruler():
    """acc_tgt must follow the TARGET's argmax, not the reference token."""
    logits = torch.zeros(1, 1, 2, 4)
    logits[0, 0, :, 3] = 10.0                     # drafter always says token 3
    target_ids = torch.zeros(1, 1, 2, dtype=torch.long)   # reference says token 0
    tgt_logits = torch.zeros(1, 1, 2, 4)
    tgt_logits[0, 0, :, 3] = 10.0                 # target would also say token 3
    mask = torch.ones(1, 1, 2, dtype=torch.bool)
    s = _call(logits, target_ids, mask, tgt_logits=tgt_logits)
    assert float(s["tau_greedy_sum"]) == 0.0, "wrong against the reference token"
    assert float(s["tau_greedy_tgt_sum"]) == 2.0, "right against the target's argmax"
    assert float(s["gt_is_tgt_greedy_pos"].sum()) == 0.0, "target disagrees with the ref"


def test_target_ruler_absent_when_no_target_logits():
    correct = torch.ones(1, 1, 3, dtype=torch.bool)
    logits, target = _logits_where(correct)
    s = _call(logits, target, torch.ones_like(correct))
    assert "acc_tgt_pos" not in s and "tau_greedy_tgt_sum" not in s


def test_dropped_blocks_are_excluded_from_tau():
    correct = torch.ones(1, 2, 3, dtype=torch.bool)
    logits, target = _logits_where(correct)
    weights = torch.tensor([[1.0, 0.0]])          # second block not kept
    s = _call(logits, target, torch.ones_like(correct), block_weights=weights)
    assert float(s["tau_greedy_sum"]) == 3.0
