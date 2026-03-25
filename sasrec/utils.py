# sasrec/utils.py
"""Shared utilities."""
import random
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility.

    Sets Python, NumPy, PyTorch CPU and GPU seeds.
    Enables deterministic cuDNN behavior.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        logger.warning("torch.use_deterministic_algorithms(True) not available; skipping.")


def worker_init_fn(worker_id: int) -> None:
    """Seed each DataLoader worker independently."""
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
