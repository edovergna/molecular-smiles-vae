"""
Seeding utilities for reproducibility.

Notes:
- torch.manual_seed() is the primary seed and applies to CPU and, in practice,
  influences tensor creation on all devices.
- CUDA has extra per-device RNGs; we seed them when available.
- MPS does not expose a separate manual_seed API; rely on torch.manual_seed and
  deterministic settings where possible.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Determinism can reduce performance and may error if an op has no deterministic variant.
        torch.use_deterministic_algorithms(True)
        
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    set_seed(42, deterministic=True)
    print("Random seeds set for reproducibility.")