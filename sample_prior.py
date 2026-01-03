import json
import csv
from pathlib import Path
from collections import Counter

import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, QED

from src.config import Config
from src.data.tokenize import SMILESTokenizer
from src.models.smiles_vae import SMILESVAE
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


def canonicalize(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def compute_props(mol: Chem.Mol):
    return {
        "MolWt": float(Descriptors.MolWt(mol)),
        "LogP": float(Crippen.MolLogP(mol)),
        "HBD": float(Descriptors.NumHDonors(mol)),
        "HBA": float(Descriptors.NumHAcceptors(mol)),
        "TPSA": float(Descriptors.TPSA(mol)),
        "Rings": float(Descriptors.RingCount(mol)),
        "QED": float(QED.qed(mol)),
    }


def summarize_numeric(values):
    # basic summary without numpy/pandas
    if not values:
        return None
    values = sorted(values)
    n = len(values)
    def pct(p):
        i = int(round((p/100) * (n-1)))
        return values[max(0, min(n-1, i))]
    mean = sum(values) / n
    var = sum((x-mean)**2 for x in values) / max(n-1, 1)
    return {
        "n": n,
        "mean": mean,
        "std": var**0.5,
        "p05": pct(5),
        "p50": pct(50),
        "p95": pct(95),
    }


def main():
    cfg = Config()
    out_dir = Path(cfg.outputs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_dir = out_dir / "sample_prior"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- tokenizer ----
    vocab = json.loads((out_dir / "vocab.json").read_text(encoding="utf-8"))
    tok = SMILESTokenizer(vocab=vocab)

    # ---- model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SMILESVAE(
        vocab_size=tok.vocab_size,
        pad_idx=tok.pad_id,
        embedding_dim=cfg.emb_dim,
        hidden_dim=getattr(cfg, "hidden_dim", cfg.hidden_dim),
        latent_dim=cfg.latent_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        greedy=cfg.greedy,
    ).to(device)

    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ---- load training set for novelty ----
    train_path = Path("outputs/splits/train.txt")
    train_raw = [l.strip() for l in train_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    train_canon = set(filter(None, (canonicalize(s) for s in train_raw)))

    # ---- sample (PRIOR) ----
    with torch.no_grad():
        token_seqs = model.sample(
            num_samples=cfg.sample_n,
            bos_id=tok.bos_id,
            eos_id=tok.eos_id,
            max_len=cfg.max_len,
            device=device,
        )

    smiles = [tok.decode(seq.tolist()) for seq in token_seqs]
    canon = [canonicalize(s) for s in smiles]
    valid = [c for c in canon if c is not None]
    valid_set = set(valid)

    validity = len(valid) / max(len(smiles), 1)
    uniqueness = len(valid_set) / max(len(valid), 1)
    novel = [s for s in valid_set if s not in train_canon]
    novelty = len(novel) / max(len(valid_set), 1)

    # ---- properties on valid unique ----
    props = {k: [] for k in ["MolWt", "LogP", "HBD", "HBA", "TPSA", "Rings", "QED"]}
    for s in list(valid_set):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        d = compute_props(m)
        for k, v in d.items():
            props[k].append(v)

    prop_summ = {k: summarize_numeric(v) for k, v in props.items()}

    # ---- console ----
    print("\n" + "=" * 60)
    print("SMILES VAE — PRIOR Sampling Report")
    print("=" * 60)
    print(f"Checkpoint:      {out_dir / 'best.pt'}")
    print(f"Train set (canon): {len(train_canon)}")
    print(f"Samples:         {len(smiles)}")
    print(f"Valid:           {len(valid)} ({validity:.2%})")
    print(f"Unique (valid):  {len(valid_set)} / {len(valid)} ({uniqueness:.2%})")
    print(f"Novel (unique):  {len(novel)} / {len(valid_set)} ({novelty:.2%})")
    print("-" * 60)

    print("Property summary (valid unique):")
    for k in ["MolWt", "LogP", "HBD", "HBA", "TPSA", "Rings", "QED"]:
        s = prop_summ.get(k)
        if not s:
            print(f"  {k}: (none)")
        else:
            print(f"  {k}: mean={s['mean']:.3f} std={s['std']:.3f} p05={s['p05']:.3f} p50={s['p50']:.3f} p95={s['p95']:.3f}")

    print("=" * 60 + "\n")

    # ---- write .smi + csv ----
    (results_dir / "samples_prior.smi").write_text("\n".join(smiles) + "\n", encoding="utf-8")

    csv_path = results_dir / "generated_smiles_prior.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "smiles_raw", "smiles_canon", "valid", "unique_valid", "novel"])
        for i, (raw, c) in enumerate(zip(smiles, canon)):
            ok = int(c is not None)
            is_unique = int(c is not None and c in valid_set)  # always 1 for valid after set, kept for clarity
            is_novel = int(c is not None and c not in train_canon)
            w.writerow([i, raw, c or "", ok, is_unique if ok else 0, is_novel if ok else 0])

    print(f"[OK] Wrote: {out_dir/'samples_prior.smi'}")
    print(f"[OK] Wrote: {csv_path}")


if __name__ == "__main__":
    main()
