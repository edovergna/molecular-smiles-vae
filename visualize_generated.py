# generate_and_grid.py
import json
import csv
from pathlib import Path

import torch
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import RDLogger

from src.config import Config
from src.data.tokenize import SMILESTokenizer
from src.models.smiles_vae import SMILESVAE

RDLogger.DisableLog("rdApp.*")


def canonicalize(smiles: str):
    if not smiles or not smiles.strip():
        return None
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    return Chem.MolToSmiles(m, canonical=True)


def visualize_smiles_grid(smiles, mols_per_row=5, size=(260, 260), max_mols=20):
    mols = []
    legends = []
    for s in smiles[:max_mols]:
        m = Chem.MolFromSmiles(s)
        mols.append(m)
        legends.append(s if len(s) <= 24 else s[:24] + "…")

    return Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        subImgSize=size,
        legends=legends,
        useSVG=False,
    )


def main():
    cfg = Config()
    out_dir = Path(cfg.outputs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_dir = out_dir / "generation"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- tokenizer ----
    vocab_path = out_dir / "vocab.json"
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    tok = SMILESTokenizer(vocab=vocab)

    # ---- model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # If your SMILESVAE signature differs, remove args that don't exist.
    model = SMILESVAE(
        vocab_size=tok.vocab_size,
        pad_idx=tok.pad_id,
        embedding_dim=cfg.emb_dim,
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        top_p=cfg.top_p,
        temperature=cfg.temperature,
        greedy=cfg.greedy,
        bos_idx=tok.bos_id,
        eos_idx=tok.eos_id,
        word_dropout=getattr(cfg, "word_dropout", 0.0),
    ).to(device)

    ckpt_path = out_dir / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)

    # supports either {"model_state_dict": ...} or raw state dict
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    # ---- sample ----
    with torch.no_grad():
        token_seqs = model.sample(
            num_samples=cfg.sample_n,
            bos_id=tok.bos_id,
            eos_id=tok.eos_id,
            max_len=cfg.max_len,
            device=device,
        )

    raw_smiles = [tok.decode(seq.tolist()) for seq in token_seqs]

    # ---- validity + canonicalization ----
    canon_smiles = []
    valid_flags = []
    for s in raw_smiles:
        c = canonicalize(s)
        if c is None:
            valid_flags.append(0)
        else:
            valid_flags.append(1)
            canon_smiles.append(c)

    valid_smiles = canon_smiles
    validity = sum(valid_flags) / max(len(valid_flags), 1)
    uniqueness = len(set(valid_smiles)) / max(len(valid_smiles), 1)

    print("\n" + "=" * 60)
    print("SMILES VAE — Generation Report")
    print("=" * 60)
    print(f"Checkpoint:      {ckpt_path}")
    print(f"Samples:         {len(raw_smiles)}")
    print(f"Valid:           {len(valid_smiles)}  ({validity:.2%})")
    print(f"Unique (valid):  {len(set(valid_smiles))} / {len(valid_smiles)}  ({uniqueness:.2%})")
    print("-" * 60)
    print("Example valid canonical SMILES:")
    for s in list(dict.fromkeys(valid_smiles))[:10]:  # stable unique order
        print("  ", s)
    print("=" * 60 + "\n")

    # ---- save CSV ----
    csv_path = results_dir / "generated_smiles.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "smiles_raw", "valid", "smiles_canon"])
        canon_iter = iter(valid_smiles)
        for i, (s, ok) in enumerate(zip(raw_smiles, valid_flags)):
            w.writerow([i, s, ok, next(canon_iter) if ok else ""])

    # ---- save report ----
    report_path = results_dir / "generation_report.txt"
    unique_valid = list(dict.fromkeys(valid_smiles))
    report_path.write_text(
        "\n".join(
            [
                "SMILES VAE — Generation Report",
                f"Checkpoint: {ckpt_path}",
                f"Samples: {len(raw_smiles)}",
                f"Valid: {len(valid_smiles)} ({validity:.2%})",
                f"Unique (valid): {len(unique_valid)}/{len(valid_smiles)} ({len(unique_valid)/max(len(valid_smiles),1):.2%})",
                "",
                "First 20 UNIQUE valid canonical SMILES:",
                *unique_valid[:20],
                "",
            ]
        ),
        encoding="utf-8",
    )

    # ---- save grids ----
    grid_valid_path = results_dir / "generated_grid_valid.png"
    img_valid = visualize_smiles_grid(unique_valid, mols_per_row=5, max_mols=20)
    img_valid.save(grid_valid_path)

    print(f"[OK] Wrote: {csv_path}")
    print(f"[OK] Wrote: {report_path}")
    print(f"[OK] Wrote: {grid_valid_path}")


if __name__ == "__main__":
    main()
