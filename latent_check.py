import json
import random
import csv
from pathlib import Path
from typing import List, Tuple

import torch
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from src.config import Config
from src.data.tokenize import SMILESTokenizer
from src.models.smiles_vae import SMILESVAE


# ----------------------------
# Helpers
# ----------------------------
def is_valid_smiles(s: str) -> bool:
    if not s or not s.strip():
        return False
    return Chem.MolFromSmiles(s) is not None


def load_smiles_lines(path: str) -> List[str]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(s)
    return out


@torch.no_grad()
def decode_from_z(
    model: SMILESVAE,
    tok: SMILESTokenizer,
    z: torch.Tensor,
    max_len: int,
    device: torch.device,
) -> List[str]:
    toks = model.sample_from_z(
        z=z.to(device),
        bos_id=tok.bos_id,
        eos_id=tok.eos_id,
        max_len=max_len,
        device=device,
    )
    # sample_from_z returns [B, <=max_len] without BOS (per your model docstring)
    return [tok.decode(seq.tolist()) for seq in toks]


def grid_image(smiles: List[str], mols_per_row: int = 5, sub_img_size=(260, 200)):
    mols = []
    legends = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        mols.append(m if m is not None else None)
        legends.append(s if len(s) <= 22 else s[:22] + "…")
    return Draw.MolsToGridImage(
        mols, molsPerRow=mols_per_row, subImgSize=sub_img_size, legends=legends
    )


# ----------------------------
# Latent sanity checks (keep)
# ----------------------------
@torch.no_grad()
def interpolation_check(
    model: SMILESVAE,
    tok: SMILESTokenizer,
    s1: str,
    s2: str,
    train_smiles: List[str],  # not strictly needed, but kept for consistency
    max_len: int,
    device: torch.device,
    steps: int = 12,
) -> List[str]:
    # Encode s1,s2 -> mu1,mu2 -> linearly interpolate -> decode each point.
    def encode_and_pad(smiles: str) -> torch.Tensor:
        ids = tok.encode(smiles)  # includes BOS/EOS already
        if len(ids) > max_len:
            ids = ids[:max_len]
            ids[-1] = tok.eos_id
        if len(ids) < max_len:
            ids = ids + [tok.pad_id] * (max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    model.eval()
    x1 = encode_and_pad(s1).unsqueeze(0).to(device)
    x2 = encode_and_pad(s2).unsqueeze(0).to(device)
    mu1, _ = model.encode(x1)
    mu2, _ = model.encode(x2)

    alphas = torch.linspace(0.0, 1.0, steps, device=device).unsqueeze(1)  # [steps,1]
    z = (1 - alphas) * mu1 + alphas * mu2  # [steps, Z]
    return decode_from_z(model, tok, z, max_len=max_len, device=device)


@torch.no_grad()
def local_perturbation_check(
    model: SMILESVAE,
    tok: SMILESTokenizer,
    seed_smiles: str,
    max_len: int,
    device: torch.device,
    n: int = 24,
    sigma: float = 0.5,
) -> List[str]:
    def encode_and_pad(smiles: str) -> torch.Tensor:
        ids = tok.encode(smiles)  # includes BOS/EOS already
        if len(ids) > max_len:
            ids = ids[:max_len]
            ids[-1] = tok.eos_id
        if len(ids) < max_len:
            ids = ids + [tok.pad_id] * (max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    model.eval()
    x = encode_and_pad(seed_smiles).unsqueeze(0).to(device)
    mu, _ = model.encode(x)

    eps = torch.randn(n, mu.size(1), device=device) * sigma
    z = mu.repeat(n, 1) + eps
    return decode_from_z(model, tok, z, max_len=max_len, device=device)


# ----------------------------
# Main
# ----------------------------
def main():
    cfg = Config()
    random.seed(cfg.seed)

    out_dir = Path(cfg.outputs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_dir = out_dir / "latent_checks"
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- load tokenizer ---
    vocab = json.loads((out_dir / "vocab.json").read_text(encoding="utf-8"))
    tok = SMILESTokenizer(vocab=vocab)

    # --- device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- load model ---
    model = SMILESVAE(
        vocab_size=tok.vocab_size,
        pad_idx=tok.pad_id,
        bos_idx=tok.bos_id,
        eos_idx=tok.eos_id,
        embedding_dim=cfg.emb_dim,
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        top_p=cfg.top_p,
        temperature=cfg.temperature,
        greedy=cfg.greedy,
        word_dropout=cfg.word_dropout,
    ).to(device)

    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # --- load training smiles (only for picking endpoints/seeds) ---
    train_path = out_dir / "splits" / "train.txt"
    train_smiles = load_smiles_lines(str(train_path))

    # ============================================================
    # 1) PRIOR generation (this is now the “main event”)
    # ============================================================
    with torch.no_grad():
        token_seqs = model.sample(
            num_samples=cfg.sample_n,
            bos_id=tok.bos_id,
            eos_id=tok.eos_id,
            max_len=cfg.max_len,
            device=device,
        )

    gen_smiles = [tok.decode(seq.tolist()) for seq in token_seqs]
    valid_flags = [is_valid_smiles(s) for s in gen_smiles]
    valid_smiles = [s for s, ok in zip(gen_smiles, valid_flags) if ok]

    validity = len(valid_smiles) / max(len(gen_smiles), 1)
    uniqueness = len(set(valid_smiles)) / max(len(valid_smiles), 1)

    print("\n" + "=" * 70)
    print("1) PRIOR generation (z ~ N(0,I) -> decode)")
    print("=" * 70)
    print(f"Checkpoint:      {out_dir / 'best.pt'}")
    print(f"Samples:         {len(gen_smiles)}")
    print(f"Valid:           {len(valid_smiles)} ({validity:.3f})")
    print(f"Unique (valid):  {len(set(valid_smiles))} / {len(valid_smiles)} ({uniqueness:.3f})")
    print("-" * 70)
    print("Examples (valid):")
    for s in valid_smiles[:12]:
        print(" ", s)
    if not valid_smiles:
        print("  (none)")
    print("=" * 70)

    # Save CSV
    prior_csv = results_dir / "generated_smiles_prior.csv"
    with prior_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "smiles", "valid"])
        for i, (s, ok) in enumerate(zip(gen_smiles, valid_flags)):
            w.writerow([i, s, int(ok)])
    print(f"[OK] Wrote: {prior_csv}")

    # Save image grids
    prior_grid_all = results_dir / "prior_grid_all.png"
    grid_image(gen_smiles[:20], mols_per_row=5).save(prior_grid_all)
    print(f"[OK] Wrote: {prior_grid_all}")

    if valid_smiles:
        prior_grid_valid = results_dir / "prior_grid_valid.png"
        grid_image(valid_smiles[:20], mols_per_row=5).save(prior_grid_valid)
        print(f"[OK] Wrote: {prior_grid_valid}")

    # ============================================================
    # 2) Latent interpolation (mu1 -> mu2)
    # ============================================================
    s1, s2 = random.sample(train_smiles, k=2)
    inter_smiles = interpolation_check(
        model, tok, s1, s2, train_smiles, max_len=cfg.max_len, device=device, steps=12
    )

    print("\n" + "=" * 70)
    print("2) Latent interpolation (mu1 -> mu2)")
    print("=" * 70)
    print("Endpoints:")
    print("  s1:", s1)
    print("  s2:", s2)
    print("-" * 70)
    for i, s in enumerate(inter_smiles):
        print(f"{i:02d}: {s}  | valid={is_valid_smiles(s)}")
    print("=" * 70)

    inter_path = results_dir / "latent_interpolation.png"
    grid_image(inter_smiles, mols_per_row=6).save(inter_path)
    print(f"[OK] Wrote: {inter_path}")

    # ============================================================
    # 3) Local perturbation (mu + eps)
    # ============================================================
    seed_smiles = random.choice(train_smiles)
    local_smiles = local_perturbation_check(
        model, tok, seed_smiles, max_len=cfg.max_len, device=device, n=24, sigma=0.5
    )
    valid_local = [s for s in local_smiles if is_valid_smiles(s)]
    uniq_local = len(set(valid_local))

    print("\n" + "=" * 70)
    print("3) Local perturbation (mu + eps)")
    print("=" * 70)
    print("Seed:", seed_smiles)
    print(f"Valid: {len(valid_local)}/{len(local_smiles)}  | Unique valid: {uniq_local}")
    print("-" * 70)
    for s in local_smiles[:12]:
        print(f"{s}  | valid={is_valid_smiles(s)}")
    print("=" * 70)

    local_path = results_dir / "latent_local_perturbations.png"
    grid_image([seed_smiles] + local_smiles[:19], mols_per_row=5).save(local_path)
    print(f"[OK] Wrote: {local_path}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
