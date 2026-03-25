# sasrec/evaluation/metrics.py
"""NDCG@K and HR@K metrics for sequential recommendation evaluation.

Evaluation modes:
- sampled: candidate set = {positive} + {num_neg_eval negatives}. Fast.
- full:    candidate set = {positive} + {all items not in user's history except positive}.

In both modes the positive item is always at index 0 in the candidates tensor.
"""
import math
import torch


def hit_at_k(scores: torch.Tensor, label_idx: torch.Tensor, k: int) -> float:
    """Hit Rate @ K averaged over a batch.

    Args:
        scores:    [B, num_candidates] — higher = more relevant
        label_idx: [B] — index of the positive item in each row of scores
        k:         cutoff

    Returns:
        Mean HR@K over the batch.
    """
    _, top_k_indices = scores.topk(k, dim=1)  # [B, k]
    label_idx_expanded = label_idx.unsqueeze(1).expand_as(top_k_indices)
    hits = (top_k_indices == label_idx_expanded).any(dim=1).float()
    return hits.mean().item()


def ndcg_at_k(scores: torch.Tensor, label_idx: torch.Tensor, k: int) -> float:
    """NDCG @ K averaged over a batch.

    Args:
        scores:    [B, num_candidates]
        label_idx: [B]
        k:         cutoff

    Returns:
        Mean NDCG@K over the batch.
    """
    B = scores.shape[0]
    sorted_indices = scores.argsort(dim=1, descending=True)  # [B, num_candidates]

    ndcg_values = []
    for b in range(B):
        pos_idx = label_idx[b].item()
        rank_positions = (sorted_indices[b] == pos_idx).nonzero(as_tuple=True)[0]
        rank = rank_positions[0].item()  # 0-indexed rank
        if rank < k:
            ndcg_values.append(1.0 / math.log2(rank + 2))  # rank+2: 1-indexed rank → log2(rank+1)
        else:
            ndcg_values.append(0.0)

    return sum(ndcg_values) / B


def evaluate_batch(
    scores: torch.Tensor,
    label_idx: torch.Tensor,
    k_values: list[int],
) -> dict[str, float]:
    """Compute NDCG@K and HR@K for multiple K values.

    Args:
        scores:    [B, num_candidates]
        label_idx: [B]
        k_values:  list of K values to evaluate

    Returns:
        Dict like {"NDCG@10": 0.42, "HR@10": 0.65, ...}
    """
    results: dict[str, float] = {}
    for k in k_values:
        results[f"NDCG@{k}"] = ndcg_at_k(scores, label_idx, k)
        results[f"HR@{k}"] = hit_at_k(scores, label_idx, k)
    return results
