"""
Device configuration module.

Handles device selection (CPU/MPS/CUDA) for training and inference.
"""

import torch


def get_device() -> torch.device:
    """Return the best available torch.device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")