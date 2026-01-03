"""
Utility functions for the SMILES VAE project.

Contains helper functions for logging, saving/loading models, and visualization.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch


def save_checkpoint(
    filepath: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a training checkpoint.

    New (recommended) checkpoint format:
      - model_state_dict
      - optimizer_state_dict (optional)
      - epoch
      - metrics (optional)
      - extra (optional)

    Args:
        filepath: Where to save checkpoint (.pt).
        model: Model to save.
        optimizer: Optional optimizer.
        epoch: Current epoch number.
        metrics: Optional dict of scalar metrics (e.g. {"train_loss":..., "val_loss":...}).
        extra: Optional dict for any additional info (history, best_val, cfg, vocab, etc.).
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    ckpt: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": int(epoch),
    }

    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()

    if metrics is not None:
        # ensure JSON-ish scalars where possible
        ckpt["metrics"] = {k: float(v) for k, v in metrics.items()}

    if extra is not None:
        ckpt["extra"] = extra

    torch.save(ckpt, filepath)



def load_checkpoint(
    filepath: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Optional[Union[str, torch.device]] = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load a checkpoint into model (and optimizer optionally).

    Supports BOTH:
      - new format keys: model_state_dict / optimizer_state_dict
      - old format keys: model_state / optimizer_state

    Returns:
        Dict with at least: epoch, metrics, extra (if present)
    """
    filepath = Path(filepath)
    ckpt = torch.load(filepath, map_location=map_location)

    # Model state (new or old)
    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif "model_state" in ckpt:  # backward compatibility
        state = ckpt["model_state"]
    else:
        raise KeyError(f"Checkpoint missing model state keys. Found: {list(ckpt.keys())}")

    model.load_state_dict(state, strict=strict)

    # Optimizer state (new or old)
    if optimizer is not None:
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        elif "optimizer_state" in ckpt:  # backward compatibility
            optimizer.load_state_dict(ckpt["optimizer_state"])

    info = {
        "epoch": ckpt.get("epoch", None),
        "metrics": ckpt.get("metrics", None),
        "extra": ckpt.get("extra", None),
    }

    # Old format had "loss"
    if "loss" in ckpt and (info["metrics"] is None):
        info["metrics"] = {"loss": ckpt["loss"]}

    return info


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Number of trainable parameters.
    """
    
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
