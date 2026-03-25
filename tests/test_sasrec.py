import torch
import pytest
from sasrec.model.sasrec import SASRec


@pytest.fixture
def model():
    return SASRec(
        item_num=100, hidden_units=32, maxlen=10,
        num_blocks=2, num_heads=2, dropout_rate=0.0
    )


def test_forward_shape(model):
    B, L = 4, 10
    seq = torch.randint(0, 101, (B, L))
    logits = model(seq)
    assert logits.shape == (B, 100)


def test_no_nan_in_output(model):
    seq = torch.randint(0, 101, (4, 10))
    logits = model(seq)
    assert torch.isfinite(logits).all()


def test_padding_sequence_output_finite(model):
    """A fully padded sequence should not produce NaN."""
    seq = torch.zeros(2, 10, dtype=torch.long)
    logits = model(seq)
    assert torch.isfinite(logits).all()


def test_score_positive_negative(model):
    """score() method returns dot product with given item embeddings."""
    seq = torch.randint(1, 100, (2, 10))
    pos_items = torch.randint(1, 100, (2, 10))
    neg_items = torch.randint(1, 100, (2, 10))
    pos_logits, neg_logits = model.score(seq, pos_items, neg_items)
    assert pos_logits.shape == (2, 10)
    assert neg_logits.shape == (2, 10)


def test_weight_tying(model):
    """Changing item_emb weights affects score output."""
    # Use item 1 in seq/pos so modifying weight[1] is guaranteed to change the output
    seq = torch.ones(1, 10, dtype=torch.long)
    pos = torch.ones(1, 10, dtype=torch.long)
    neg = torch.full((1, 10), 51, dtype=torch.long)
    p1, n1 = model.score(seq, pos, neg)
    with torch.no_grad():
        model.item_emb.weight[1] += 100.0
    p2, n2 = model.score(seq, pos, neg)
    assert not torch.allclose(p1, p2)
