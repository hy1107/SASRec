# sasrec/data/dataset.py
"""PyTorch Dataset classes for SASRec training and evaluation."""
import numpy as np
import torch
from torch.utils.data import Dataset
from sasrec.data.sampler import sample_negative


class SASRecDataset(Dataset):
    """Training dataset. Each sample is one user's full sequence.

    Matches the original SASRec training protocol:
        seq = training_items[:-1]  (input: all but last training item)
        pos = training_items[1:]   (targets: all but first training item)
        neg = one sampled negative per non-padding position in pos

    This ensures pos[-1] = last training item ≠ 0, so ALL non-padding
    positions contribute to the BCE loss.

    Returns (seq, pos, neg) tensors of shape [maxlen], left-padded with 0.
    """

    def __init__(
        self,
        train_data: dict[int, list[int]],
        item_count: int,
        maxlen: int,
        seed: int = 42,
    ) -> None:
        self.users = sorted(train_data.keys())
        self.train_data = train_data
        self.item_count = item_count
        self.maxlen = maxlen
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        uid = self.users[idx]
        seq_full = self.train_data[uid]
        history = set(seq_full)

        # Match the original SASRec training protocol exactly:
        #   seq = training_items[:-1]  (all but the last training item, as input)
        #   pos = training_items[1:]   (all but the first training item, as targets)
        # pos[-1] = last training item ≠ 0, so ALL non-padding positions contribute to loss.
        seq_raw = seq_full[:-1]
        pos_raw = seq_full[1:]

        # Truncate from the right to fit maxlen
        seq_raw = seq_raw[-(self.maxlen):]
        pos_raw = pos_raw[-(self.maxlen):]
        seq_len = len(seq_raw)

        # Left-pad both to maxlen with 0
        seq = np.zeros(self.maxlen, dtype=np.int64)
        seq[-seq_len:] = seq_raw
        pos = np.zeros(self.maxlen, dtype=np.int64)
        pos[-seq_len:] = pos_raw

        # Build neg: one sampled negative per position where pos is non-zero
        neg = np.zeros(self.maxlen, dtype=np.int64)
        for i in range(self.maxlen):
            if pos[i] != 0:
                neg[i] = sample_negative(self.item_count, history, self.rng)

        return (
            torch.from_numpy(seq),
            torch.from_numpy(pos),
            torch.from_numpy(neg),
        )


class SASRecEvalDataset(Dataset):
    """Evaluation dataset (val or test). Each sample is one user.

    For validation: input sequence = training items only (excludes val and test).
    For test:       input sequence = training items + val item (excludes test only).
                    Pass `prefix_data=valid_data` to include the val item.

    Returns (seq, candidates, label_idx):
        seq:        input sequence tensor [maxlen], left-padded with 0.
        candidates: item ids tensor [num_neg+1], positive item first (index 0).
        label_idx:  index of the positive item in candidates (always 0).
    """

    def __init__(
        self,
        train_data: dict[int, list[int]],
        target_data: dict[int, int],
        neg_samples: dict[int, list[int]],
        item_count: int,
        maxlen: int,
        prefix_data: dict[int, int] | None = None,
    ) -> None:
        self.users = sorted(target_data.keys())
        self.train_data = train_data
        self.target_data = target_data
        self.neg_samples = neg_samples
        self.maxlen = maxlen
        self.prefix_data = prefix_data

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        uid = self.users[idx]
        seq_raw = list(self.train_data[uid])
        if self.prefix_data is not None and uid in self.prefix_data:
            seq_raw = seq_raw + [self.prefix_data[uid]]
        seq_raw = seq_raw[-self.maxlen:]
        seq_len = len(seq_raw)

        seq = np.zeros(self.maxlen, dtype=np.int64)
        seq[-seq_len:] = seq_raw

        target = self.target_data[uid]
        negs = self.neg_samples[uid]
        candidates = np.array([target] + negs, dtype=np.int64)

        return (
            torch.from_numpy(seq),
            torch.from_numpy(candidates),
            torch.tensor(0, dtype=torch.long),  # positive is always index 0
        )
