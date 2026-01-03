"""
Configuration module for the SMILES VAE project.

Contains hyperparameters, model configuration, and training settings.
"""

from dataclasses import dataclass
from typing import Optional


from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Paths
    raw_smiles_path: str = "data/raw/smiles.txt"
    outputs_dir: str = "outputs"

    # Data / tokenization
    max_len: int = 120  # includes <bos>/<eos>; dataset uses max_len-1 for x and t

    # Model
    emb_dim: int = 64
    hidden_dim: int = 96
    latent_dim: int = 32
    num_layers: int = 1
    dropout: float = 0.3
    word_dropout: float = 0.15

    # Training
    batch_size: int = 128
    lr: float = 2e-3
    epochs: int = 30
    beta_kl: float = 1.0
    capacity: Optional[float] = 3.0
    warmup_epochs: int = 28
    grad_clip: float = 1.0
    seed: int = 42

    # Sampling / generation
    sample_n: int = 5000
    temperature: float = 1.1
    top_p: float = 0.95
    greedy: bool = False


if __name__ == "__main__":
    cfg = Config()
    print(cfg)
