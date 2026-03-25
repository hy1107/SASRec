import numpy as np
from sasrec.data.sampler import sample_negative, build_eval_negatives


def test_sample_negative_avoids_history():
    history = {1, 2, 3}
    item_count = 10
    for _ in range(50):
        neg = sample_negative(item_count, history, rng=np.random.default_rng(0))
        assert neg not in history
        assert 1 <= neg <= item_count


def test_build_eval_negatives_shape_and_exclusion():
    train_data = {1: [1, 2, 3], 2: [4, 5, 6]}
    valid_data = {1: 4, 2: 7}
    test_data = {1: 5, 2: 8}
    item_count = 20
    num_neg = 10

    val_negs, test_negs = build_eval_negatives(
        train_data, valid_data, test_data, item_count, num_neg, seed=42
    )

    for uid in [1, 2]:
        full_history = set(train_data[uid]) | {valid_data[uid]} | {test_data[uid]}
        assert len(val_negs[uid]) == num_neg
        assert len(test_negs[uid]) == num_neg
        for neg in val_negs[uid]:
            assert neg not in full_history
        for neg in test_negs[uid]:
            assert neg not in full_history


def test_build_eval_negatives_reproducible():
    train_data = {1: [1, 2, 3]}
    valid_data = {1: 4}
    test_data = {1: 5}
    item_count = 100
    a_val, a_test = build_eval_negatives(train_data, valid_data, test_data, item_count, 10, seed=42)
    b_val, b_test = build_eval_negatives(train_data, valid_data, test_data, item_count, 10, seed=42)
    assert a_val[1] == b_val[1]
    assert a_test[1] == b_test[1]


def test_val_and_test_negatives_use_independent_seeds():
    train_data = {1: [1, 2, 3]}
    valid_data = {1: 4}
    test_data = {1: 5}
    item_count = 100
    val_negs, test_negs = build_eval_negatives(
        train_data, valid_data, test_data, item_count, 10, seed=42
    )
    assert val_negs[1] != test_negs[1]
