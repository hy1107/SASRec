# sasrec/model/modules.py
"""Reusable Transformer primitives for SASRec.

Architecture follows the original SASRec codebase:
- Pre-LN: LayerNorm applied BEFORE each sublayer (not after).
- ReLU activation in the feed-forward network.
- Causal (lower-triangular) self-attention mask.
- Key padding mask derived from token ids (id == 0 → padding), not from embedding vectors.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Causally-masked multi-head self-attention with Pre-LN and key padding mask.

    Padding positions are identified by token_ids == 0 (passed in from the caller),
    which is unambiguous regardless of embedding weight magnitude.
    """

    def __init__(self, hidden_units: int, num_heads: int, dropout_rate: float) -> None:
        super().__init__()
        assert hidden_units % num_heads == 0, (
            f"hidden_units ({hidden_units}) must be divisible by num_heads ({num_heads})"
        )
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads

        self.q_proj = nn.Linear(hidden_units, hidden_units, bias=False)
        self.k_proj = nn.Linear(hidden_units, hidden_units, bias=False)
        self.v_proj = nn.Linear(hidden_units, hidden_units, bias=False)
        self.out_proj = nn.Linear(hidden_units, hidden_units, bias=False)

        self.layer_norm = nn.LayerNorm(hidden_units)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:          [B, L, D] input sequence
            token_ids:  [B, L] original token indices; 0 = padding

        Returns:
            [B, L, D] output after Pre-LN → attention → residual
        """
        residual = x
        x = self.layer_norm(x)

        B, L, D = x.shape
        H = self.num_heads
        Dh = self.head_dim

        Q = self.q_proj(x).view(B, L, H, Dh).transpose(1, 2)  # [B, H, L, Dh]
        K = self.k_proj(x).view(B, L, H, Dh).transpose(1, 2)
        V = self.v_proj(x).view(B, L, H, Dh).transpose(1, 2)

        scale = math.sqrt(Dh)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # [B, H, L, L]

        # Causal mask: each position can only attend to itself and earlier positions
        causal_mask = torch.triu(
            torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Key padding mask: mask out key positions where token_id == 0
        key_padding_mask = (token_ids == 0)  # [B, L], True = padding
        scores = scores.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
        )

        # Replace any all-masked rows with 0 to avoid NaN from softmax
        all_masked = scores.isinf().all(dim=-1, keepdim=True)
        scores = scores.masked_fill(all_masked, 0.0)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # [B, H, L, Dh]
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)

        return residual + out


class PointwiseFeedForward(nn.Module):
    """Position-wise feed-forward network with Pre-LN.

    Uses ReLU activation to match the original SASRec codebase.
    """

    def __init__(self, hidden_units: int, dropout_rate: float) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_units)
        self.fc1 = nn.Linear(hidden_units, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return residual + x
