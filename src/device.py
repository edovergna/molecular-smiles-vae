"""
Device configuration module for PyTorch.

Handles device selection (CPU/MPS/CUDA) for training and inference.
"""

import torch


def get_device() -> torch.device:
    """
    Determine and return the appropriate device for PyTorch operations.
    
    Prioritizes MPS (Apple Silicon) if available, otherwise falls back to CPU.
    
    Returns:
        torch.device: The device to use for computations.
    """
    # TODO: Check for MPS (Apple Silicon) availability
    # TODO: Fall back to CPU if MPS is not available
    # TODO: Add logging for which device is being used
    
    if torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
