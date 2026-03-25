import torch
import pytest
from sasrec.data.dataset import SASRecDataset, SASRecEvalDataset


@pytest.fixture
def small_data():
    train_data = {
        1: [11, 12, 13, 14, 15],
        2: [21, 22, 23, 24, 25],
        3: [31, 32, 33, 34, 35],
    }
    valid_data = {1: 16, 2: 26, 3: 36}
    test_data  = {1: 17, 2: 27, 3: 37}
    return train_data, valid_data, test_data


def test_train_dataset_length(small_data):
    train_data, _, _ = small_data
    ds = SASRecDataset(train_data, item_count=40, maxlen=10)
    assert len(ds) == 3


def test_train_dataset_shapes(small_data):
    train_data, _, _ = small_data
    ds = SASRecDataset(train_data, item_count=40, maxlen=10)
    seq, pos, neg = ds[0]
    assert seq.shape == (10,)
    assert pos.shape == (10,)
    assert neg.shape == (10,)


def test_train_seq_left_padded(small_data):
    train_data, _, _ = small_data
    ds = SASRecDataset(train_data, item_count=40, maxlen=10)
    seq, pos, _ = ds[0]
    # 5 train items → seq uses items[:-1] = 4 items, pos uses items[1:] = 4 items
    assert (seq == 0).sum() == 6
    assert (seq != 0).sum() == 4
    assert (pos == 0).sum() == 6
    assert (pos != 0).sum() == 4


def test_pos_is_seq_shifted_left(small_data):
    train_data, _, _ = small_data
    ds = SASRecDataset(train_data, item_count=40, maxlen=10)
    seq, pos, _ = ds[0]
    for i in range(9):
        if seq[i] != 0 and seq[i + 1] != 0:
            assert pos[i] == seq[i + 1]
    # The last non-zero pos slot holds the last training item (target-only)
    last_pos_item = pos[pos != 0][-1].item()
    last_train_item = list(train_data.values())[0][-1]
    assert last_pos_item == last_train_item


def test_neg_not_in_user_history(small_data):
    train_data, _, _ = small_data
    ds = SASRecDataset(train_data, item_count=40, maxlen=10)
    for idx in range(len(ds)):
        seq, _, neg = ds[idx]
        user_items = set(seq.tolist()) - {0}
        for n in neg.tolist():
            if n != 0:
                assert n not in user_items


def test_eval_dataset_length(small_data):
    train_data, valid_data, test_data = small_data
    val_negs = {1: [2, 3], 2: [4, 5], 3: [6, 7]}
    ds = SASRecEvalDataset(train_data, valid_data, val_negs, item_count=40, maxlen=10)
    assert len(ds) == 3


def test_eval_dataset_candidate_count(small_data):
    train_data, valid_data, _ = small_data
    val_negs = {uid: list(range(1, 11)) for uid in train_data}  # 10 negs
    ds = SASRecEvalDataset(train_data, valid_data, val_negs, item_count=40, maxlen=10)
    seq, candidates, label_idx = ds[0]
    # candidates = 10 negs + 1 positive = 11
    assert candidates.shape == (11,)
    # label_idx points to the positive item in candidates (always index 0)
    assert label_idx.item() == 0
    assert candidates[0].item() == valid_data[1]


def test_eval_dataset_test_includes_val_item(small_data):
    """Test evaluation input should include the val item as context."""
    train_data, valid_data, test_data = small_data
    test_negs = {uid: list(range(1, 11)) for uid in train_data}
    # Without prefix: input = train items only
    ds_no_prefix = SASRecEvalDataset(train_data, test_data, test_negs, item_count=40, maxlen=10)
    seq_no_prefix, _, _ = ds_no_prefix[0]
    # With prefix: input = train items + val item
    ds_with_prefix = SASRecEvalDataset(
        train_data, test_data, test_negs, item_count=40, maxlen=10, prefix_data=valid_data
    )
    seq_with_prefix, _, _ = ds_with_prefix[0]
    # The prefix version should have one more non-zero element
    assert (seq_with_prefix != 0).sum() > (seq_no_prefix != 0).sum()
