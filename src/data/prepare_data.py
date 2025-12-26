"""
Data preparation utilities.

Functions for loading, preprocessing, and splitting SMILES data.
"""

from pathlib import Path
from typing import List, Tuple
import pandas as pd


def load_smiles_from_file(filepath: Path) -> List[str]:
    """
    Load SMILES strings from a text file.
    
    Args:
        filepath: Path to the SMILES file (one SMILES per line).
        
    Returns:
        List of SMILES strings.
    """
    # TODO: Read file and extract SMILES
    # TODO: Remove empty lines and comments
    # TODO: Return cleaned SMILES list
    pass


def validate_smiles(smiles: str) -> bool:
    """
    Validate a SMILES string using RDKit.
    
    Args:
        smiles: SMILES string to validate.
        
    Returns:
        True if valid, False otherwise.
    """
    # TODO: Use RDKit to check if SMILES is valid
    # TODO: Handle exceptions gracefully
    pass


def split_data(smiles_list: List[str], train_ratio: float = 0.8, 
               val_ratio: float = 0.1) -> Tuple[List[str], List[str], List[str]]:
    """
    Split SMILES data into train, validation, and test sets.
    
    Args:
        smiles_list: List of SMILES strings.
        train_ratio: Proportion of data for training.
        val_ratio: Proportion of data for validation.
        
    Returns:
        Tuple of (train_smiles, val_smiles, test_smiles).
    """
    # TODO: Implement train/val/test split
    # TODO: Ensure reproducible splitting (use seed)
    pass


def preprocess_smiles(smiles_list: List[str]) -> List[str]:
    """
    Preprocess SMILES strings.
    
    Args:
        smiles_list: Raw SMILES strings.
        
    Returns:
        Preprocessed SMILES strings.
    """
    # TODO: Apply canonicalization using RDKit
    # TODO: Filter out invalid SMILES
    # TODO: Apply any additional preprocessing
    pass
