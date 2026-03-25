import torch
from sasrec.model.modules import MultiHeadAttention, PointwiseFeedForward


def test_multihead_attention_output_shape():
    B, L, D = 4, 10, 32
    mha = MultiHeadAttention(hidden_units=D, num_heads=2, dropout_rate=0.0)
    x = torch.randn(B, L, D)
    token_ids = torch.randint(1, 20, (B, L))
    out = mha(x, token_ids)
    assert out.shape == (B, L, D)


def test_multihead_attention_causal_mask():
    """Output at position t must not depend on positions t+1, ..., L-1."""
    B, L, D = 1, 6, 16
    mha = MultiHeadAttention(hidden_units=D, num_heads=1, dropout_rate=0.0)
    mha.eval()
    x = torch.randn(B, L, D)
    token_ids = torch.ones(B, L, dtype=torch.long)
    out1 = mha(x, token_ids)

    x2 = x.clone()
    x2[:, 3:, :] += 10.0
    out2 = mha(x2, token_ids)
    assert torch.allclose(out1[:, :3, :], out2[:, :3, :], atol=1e-5)


def test_multihead_attention_padding_mask():
    """Padding positions (token_id=0) are masked as keys."""
    B, L, D = 2, 8, 16
    mha = MultiHeadAttention(hidden_units=D, num_heads=1, dropout_rate=0.0)
    mha.eval()
    x = torch.randn(B, L, D)
    token_ids = torch.ones(B, L, dtype=torch.long)
    token_ids[:, :4] = 0  # first 4 positions are padding

    out = mha(x, token_ids)
    assert torch.isfinite(out).all()


def test_feedforward_output_shape():
    B, L, D = 4, 10, 32
    ffn = PointwiseFeedForward(hidden_units=D, dropout_rate=0.0)
    x = torch.randn(B, L, D)
    out = ffn(x)
    assert out.shape == (B, L, D)
