# sasrec/model/sasrec.py
"""SASRec: Self-Attentive Sequential Recommendation (Kang & McAuley, ICDM 2018).

Architecture:
- Learned item embeddings (padding_idx=0, weight-tied with output)
- Learned 1-indexed positional embeddings (padding_idx=0, maxlen+1 rows)
- N stacked Transformer blocks: Pre-LN → CausalMHA → Residual → Pre-LN → FFN → Residual
- Prediction: last non-padding hidden state dot-producted with item embeddings
"""
import torch
import torch.nn as nn
from sasrec.model.modules import MultiHeadAttention, PointwiseFeedForward


class SASRec(nn.Module):
    def __init__(
        self,
        item_num: int,
        hidden_units: int,
        maxlen: int,
        num_blocks: int,
        num_heads: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.item_num = item_num
        self.maxlen = maxlen

        # Item embedding: index 0 = padding (always zero, no gradient)
        self.item_emb = nn.Embedding(item_num + 1, hidden_units, padding_idx=0)

        # Positional embedding: 1-indexed, table has maxlen+1 rows.
        # Position 0 = padding (suppressed by padding_idx=0).
        # Position i corresponds to the i-th slot in the fixed-length window
        # (1=oldest, maxlen=newest). Padding slots receive position index 0.
        self.pos_emb = nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)

        self.emb_dropout = nn.Dropout(dropout_rate)

        self.blocks = nn.ModuleList([
            _TransformerBlock(hidden_units, num_heads, dropout_rate)
            for _ in range(num_blocks)
        ])

        self.final_norm = nn.LayerNorm(hidden_units)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding) and module.padding_idx is not None:
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.weight[module.padding_idx])

    def _encode(self, seq: torch.Tensor) -> torch.Tensor:
        """Encode an input sequence.

        Args:
            seq: [B, L] token id tensor (0 = padding)

        Returns:
            [B, L, D] hidden states
        """
        B, L = seq.shape
        # 1-indexed position ids assigned to fixed-length padded window slots.
        # Padding slots (seq == 0) are assigned position index 0 (suppressed by padding_idx).
        pos_ids = torch.arange(1, L + 1, device=seq.device).unsqueeze(0).expand(B, -1)
        pos_ids = pos_ids * (seq != 0).long()

        x = self.item_emb(seq) + self.pos_emb(pos_ids)
        x = self.emb_dropout(x)

        for block in self.blocks:
            x = block(x, seq)

        x = self.final_norm(x)
        return x

    def _get_last_hidden(self, seq: torch.Tensor) -> torch.Tensor:
        """Extract the hidden state at the last position (rightmost in left-padded seq).

        For left-padded sequences, the last position always holds the most recent item.

        Returns:
            [B, D]
        """
        x = self._encode(seq)
        return x[:, -1, :]  # [B, D]

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """Compute logits over all items for the next-item prediction task.

        Args:
            seq: [B, L] input sequence (left-padded with 0)

        Returns:
            [B, item_num] logit scores.
            logits[:, i] corresponds to item id (i+1).
            i.e. logits[:, 0] = score for item 1, logits[:, item_num-1] = score for item_num.
        """
        h = self._get_last_hidden(seq)  # [B, D]
        # Weight-tied output: rows 1..item_num of item_emb (exclude padding row 0).
        item_embeddings = self.item_emb.weight[1:]  # [item_num, D]
        logits = torch.matmul(h, item_embeddings.T)  # [B, item_num]
        return logits

    def score(
        self,
        seq: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-position scores for training (BCE loss).

        Args:
            seq:       [B, L] input sequence
            pos_items: [B, L] positive target items per position
            neg_items: [B, L] negative sample items per position

        Returns:
            pos_logits: [B, L] dot product scores for positive items
            neg_logits: [B, L] dot product scores for negative items
        """
        h = self._encode(seq)  # [B, L, D]
        pos_emb = self.item_emb(pos_items)  # [B, L, D]
        neg_emb = self.item_emb(neg_items)  # [B, L, D]
        pos_logits = (h * pos_emb).sum(dim=-1)  # [B, L]
        neg_logits = (h * neg_emb).sum(dim=-1)  # [B, L]
        return pos_logits, neg_logits


class _TransformerBlock(nn.Module):
    """Single SASRec Transformer block: Pre-LN MHA + Pre-LN FFN."""

    def __init__(self, hidden_units: int, num_heads: int, dropout_rate: float) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(hidden_units, num_heads, dropout_rate)
        self.ffn = PointwiseFeedForward(hidden_units, dropout_rate)

    def forward(self, x: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.attention(x, token_ids)
        x = self.ffn(x)
        return x
