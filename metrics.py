# metrics.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Set, Tuple, Dict, Optional
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, QED

def canonicalize_smiles(smi: str) -> Optional[str]:
    """Return canonical SMILES or None if invalid."""
    if not smi or not isinstance(smi, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=True)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None

def compute_basic_props(mol: Chem.Mol) -> Dict[str, float]:
    return {
        "MolWt": float(Descriptors.MolWt(mol)),
        "LogP": float(Crippen.MolLogP(mol)),
        "HBD": float(Descriptors.NumHDonors(mol)),
        "HBA": float(Descriptors.NumHAcceptors(mol)),
        "TPSA": float(Descriptors.TPSA(mol)),
        "Rings": float(Descriptors.RingCount(mol)),
        "QED": float(QED.qed(mol)),
    }

@dataclass
class QualityReport:
    n_total: int
    n_valid: int
    n_unique_valid: int
    n_novel_valid: int
    validity: float
    uniqueness: float
    novelty: float
    df_valid: pd.DataFrame

def quality_report(
    smiles: List[str],
    train_smiles_set: Set[str],
    max_props: int = 20000
) -> QualityReport:
    """
    smiles: generated SMILES (raw)
    train_smiles_set: canonical SMILES set from training data
    """
    canon = [canonicalize_smiles(s) for s in smiles]
    valid = [c for c in canon if c is not None]
    valid_set = set(valid)

    n_total = len(smiles)
    n_valid = len(valid)
    n_unique_valid = len(valid_set)

    novel = [s for s in valid_set if s not in train_smiles_set]
    n_novel_valid = len(novel)

    validity = n_valid / n_total if n_total else 0.0
    uniqueness = n_unique_valid / n_valid if n_valid else 0.0
    novelty = n_novel_valid / n_unique_valid if n_unique_valid else 0.0

    # properties on a sample of valid molecules
    props_rows = []
    for s in list(valid_set)[:max_props]:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        row = {"smiles": s}
        row.update(compute_basic_props(mol))
        props_rows.append(row)
    df = pd.DataFrame(props_rows)

    return QualityReport(
        n_total=n_total,
        n_valid=n_valid,
        n_unique_valid=n_unique_valid,
        n_novel_valid=n_novel_valid,
        validity=validity,
        uniqueness=uniqueness,
        novelty=novelty,
        df_valid=df,
    )

def print_report(name: str, rep: QualityReport) -> None:
    print(f"\n=== {name} ===")
    print(f"Total samples:       {rep.n_total}")
    print(f"Valid:               {rep.n_valid} ({rep.validity:.3f})")
    print(f"Unique (of valid):   {rep.n_unique_valid} ({rep.uniqueness:.3f})")
    print(f"Novel (of unique):   {rep.n_novel_valid} ({rep.novelty:.3f})")
    if not rep.df_valid.empty:
        desc = rep.df_valid.drop(columns=["smiles"]).describe(percentiles=[0.05,0.5,0.95]).T
        print("\nProperty summary (valid unique):")
        print(desc[["mean","std","5%","50%","95%"]].to_string(float_format=lambda x: f"{x:0.3f}"))
