"""
adr.utils.seed – Cố định random seed cho tái lập thực nghiệm.

Ghi chú: Torch và numpy có trạng thái global riêng.
Gọi set_seed(seed) ở đầu mỗi script/training loop.
"""

import random


def set_seed(seed: int) -> None:
    """Cố định seed cho random, numpy và torch (nếu có)."""
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
