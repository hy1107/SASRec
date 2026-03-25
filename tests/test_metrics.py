import math
import torch
from sasrec.evaluation.metrics import ndcg_at_k, hit_at_k, evaluate_batch


def test_hit_at_k_positive_ranked_first():
    scores = torch.tensor([[10.0, 1.0, 2.0, 3.0]])
    label_idx = torch.tensor([0])
    assert hit_at_k(scores, label_idx, k=3) == 1.0


def test_hit_at_k_positive_not_in_top_k():
    scores = torch.tensor([[1.0, 5.0, 6.0, 7.0]])
    label_idx = torch.tensor([0])
    assert hit_at_k(scores, label_idx, k=2) == 0.0


def test_ndcg_at_k_positive_ranked_first():
    scores = torch.tensor([[10.0, 1.0, 2.0, 3.0]])
    label_idx = torch.tensor([0])
    result = ndcg_at_k(scores, label_idx, k=10)
    assert abs(result - 1.0) < 1e-5


def test_ndcg_at_k_positive_ranked_second():
    scores = torch.tensor([[5.0, 10.0, 1.0, 2.0]])
    label_idx = torch.tensor([0])
    # Positive at rank 2 (0-indexed rank=1): NDCG = 1/log2(3)
    result = ndcg_at_k(scores, label_idx, k=10)
    assert abs(result - 1.0 / math.log2(3)) < 1e-5


def test_ndcg_at_k_positive_outside_top_k():
    scores = torch.tensor([[1.0, 10.0, 9.0, 8.0, 7.0]])
    label_idx = torch.tensor([0])
    assert ndcg_at_k(scores, label_idx, k=2) == 0.0


def test_evaluate_batch_returns_dict():
    scores = torch.randn(8, 101)
    label_idx = torch.zeros(8, dtype=torch.long)
    results = evaluate_batch(scores, label_idx, k_values=[5, 10, 20])
    for k in [5, 10, 20]:
        assert f"NDCG@{k}" in results
        assert f"HR@{k}" in results


def test_evaluate_batch_values_in_range():
    scores = torch.randn(16, 101)
    label_idx = torch.zeros(16, dtype=torch.long)
    results = evaluate_batch(scores, label_idx, k_values=[10])
    assert 0.0 <= results["NDCG@10"] <= 1.0
    assert 0.0 <= results["HR@10"] <= 1.0
