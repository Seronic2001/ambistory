"""Loss functions, vectorised and with the sign convention fixed."""
import torch
import torch.nn.functional as F


def within_setup_ranking_loss(preds, targets, group_ids, margin=0.3, min_gap=0.1):
    """Hinge loss on pairs from the same group (setup or story) whose labels differ.

    For every pair (i, j) in the same group with targets[i] > targets[j] + min_gap:
        max(0, margin - (preds[i] - preds[j]))

    Uses the sign of the label gap, not its size. Model 6 used
    `margin - dy * dp`, which can never be satisfied when dy is small and pushes
    predictions apart without bound.

    group_ids: LongTensor [B]. Returns (loss, n_pairs). Log n_pairs - if it is
    ~0, the batches are not grouped (see StoryBatchSampler).
    """
    same = group_ids.unsqueeze(1) == group_ids.unsqueeze(0)
    higher = (targets.unsqueeze(1) - targets.unsqueeze(0)) > min_gap
    mask = same & higher
    n_pairs = int(mask.sum())
    if n_pairs == 0:
        return preds.sum() * 0.0, 0
    gap = preds.unsqueeze(1) - preds.unsqueeze(0)
    return F.relu(margin - gap)[mask].mean(), n_pairs


def annotator_kl(logits, target_dist):
    """KL(target || model) over the 5 rating classes, from raw logits (numerically stable)."""
    return F.kl_div(F.log_softmax(logits, dim=-1), target_dist, reduction="batchmean")


def expected_score(logits):
    probs = torch.softmax(logits, dim=-1)
    return (probs * torch.arange(1, 6, device=logits.device, dtype=probs.dtype)).sum(-1)
