"""
PyTorch Dataset classes for SMILES data.

Handles loading and batching of SMILES sequences.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Tuple


class SMILESDataset(Dataset):
    """PyTorch Dataset for SMILES strings."""
    
    def __init__(self, smiles_list: List[str], tokenizer, max_length: int = 100):
        """
        Initialize SMILES dataset.
        
        Args:
            smiles_list: List of SMILES strings.
            tokenizer: SMILES tokenizer instance.
            max_length: Maximum sequence length for padding/truncation.
        """
        
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _pad_truncate(self, token_ids: List[int]) -> List[int]:
        """
        Pad or truncate token ID sequence to max_length.
        
        Args:
            token_ids: List of token IDs.
        
        Returns:
            Padded or truncated list of token IDs.
        """

        token_ids = token_ids[: self.max_length]

        if len(token_ids) < self.max_length:
            token_ids = token_ids + [self.tokenizer.pad_id] * (self.max_length - len(token_ids))

        return token_ids
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.smiles_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample.
            
        Returns:
            Tuple of (tokenized input tensor, tokenized target tensor).
        """

        smiles = self.smiles_list[idx]

        token_ids = self.tokenizer.encode(smiles)
        token_ids = self._pad_truncate(token_ids)

        x = torch.tensor(token_ids[:-1], dtype=torch.long)
        t = torch.tensor(token_ids[1:], dtype=torch.long)

        return x, t
