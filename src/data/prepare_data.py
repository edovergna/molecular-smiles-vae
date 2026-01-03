"""
Data preparation utilities.

Functions for loading, preprocessing, and splitting SMILES data.
"""

from pathlib import Path
import random
from typing import List, Tuple
import pandas as pd
from rdkit import Chem


def load_smiles_from_file(filepath: Path) -> List[str]:
    """
    Load SMILES strings from a text file.
    
    Args:
        filepath: Path to the SMILES file (one SMILES per line).
        
    Returns:
        List of SMILES strings.
    """

    with filepath.open("r", encoding="utf-8") as f:
        smiles_list = [line.strip() for line in f if line.strip()]

    return smiles_list


def validate_smiles(smiles: str) -> Tuple[bool, Chem.Mol]:
    """
    Validate a SMILES string using RDKit.
    
    Args:
        smiles: SMILES string to validate.
        
    Returns:
        A tuple (is_valid, mol), where:
        - is_valid is True if the SMILES is valid
        - mol is the RDKit Mol object (None if invalid)
    """
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None, mol


def split_data(smiles_list: List[str], train_frac: float = 0.8, 
               val_frac: float = 0.1, seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """
    Split SMILES data into train, validation, and test sets.
    
    Args:
        smiles_list: List of SMILES strings.
        train_frac: Proportion of data for training.
        val_frac: Proportion of data for validation.
        
    Returns:
        Tuple of (train_smiles, val_smiles, test_smiles).
    """
    assert 0 < train_frac < 1
    assert 0 <= val_frac < 1
    assert train_frac + val_frac < 1

    smiles = list(smiles_list)  # copy
    random.Random(seed).shuffle(smiles)

    total = len(smiles)
    train_end = int(total * train_frac)
    val_end = train_end + int(total * val_frac)

    train_smiles = smiles[:train_end]
    val_smiles = smiles[train_end:val_end]
    test_smiles = smiles[val_end:]

    return train_smiles, val_smiles, test_smiles


def preprocess_smiles(smiles_list: List[str]) -> List[str]:
    """
    Preprocess SMILES strings.
    
    Args:
        smiles_list: Raw SMILES strings.
        
    Returns:
        Preprocessed SMILES strings.
    """
    
    preprocessed = []

    for smi in smiles_list:

        is_valid, mol = validate_smiles(smi)

        if is_valid:
            canon_smi = Chem.MolToSmiles(mol, canonical=True)
            preprocessed.append(canon_smi)
    
    return preprocessed