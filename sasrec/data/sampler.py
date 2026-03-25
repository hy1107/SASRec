# sasrec/data/sampler.py
"""Negative sampling utilities for training and evaluation."""
import numpy as np


def sample_negative(
    item_count: int,
    history: set[int],
    rng: np.random.Generator,
) -> int:
    """Sample one negative item (1-indexed) not in history via rejection sampling."""
    while True:
        neg = int(rng.integers(1, item_count + 1))
        if neg not in history:
            return neg


def build_eval_negatives(
    train_data: dict[int, list[int]],
    valid_data: dict[int, int],
    test_data: dict[int, int],
    item_count: int,
    num_neg: int,
    seed: int,
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """Build fixed negative sets for validation and test evaluation.

    val negatives use `seed`, test negatives use `seed + 1`.
    All negatives exclude the user's full interaction history (train + val + test).

    Returns:
        val_negatives:  {user_id: [neg_item, ...]}  length = num_neg per user
        test_negatives: {user_id: [neg_item, ...]}  length = num_neg per user
    """
    val_negs: dict[int, list[int]] = {}
    test_negs: dict[int, list[int]] = {}

    val_rng = np.random.default_rng(seed)
    test_rng = np.random.default_rng(seed + 1)

    for uid in train_data:
        full_history = (
            set(train_data[uid])
            | {valid_data[uid]}
            | {test_data[uid]}
        )
        val_negs[uid] = [
            sample_negative(item_count, full_history, val_rng)
            for _ in range(num_neg)
        ]
        test_negs[uid] = [
            sample_negative(item_count, full_history, test_rng)
            for _ in range(num_neg)
        ]

    return val_negs, test_negs
