# posterior_compare.py
import json
from pathlib import Path
from typing import List, Optional

import torch
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from metrics import quality_report, print_report, canonicalize_smiles
from src.config import Config
from src.data.tokenize import SMILESTokenizer
from src.models.smiles_vae import SMILESVAE


# -----------------------
# Paths (match your repo)
# -----------------------
OUT_DIR = Path("outputs")
TRAIN_PATH = OUT_DIR / "splits" / "train.txt"
CKPT_PATH = OUT_DIR / "best.pt"
VOCAB_PATH = OUT_DIR / "vocab.json"

N_CONDITION = 5000          # how many train molecules to encode for posterior sampling
N_GENERATE = 5000          # how many samples to generate for each mode
MAX_LEN = None             # if None -> use cfg.max_len

# output files
PRIOR_SMI = OUT_DIR / "compare" / "samples_prior.smi"
POST_MU_SMI = OUT_DIR / "compare" / "samples_post_mu.smi"
POST_SAMPLE_SMI = OUT_DIR / "compare" / "samples_post_sample.smi"


def load_smiles_lines(path: Path, limit: Optional[int] = None) -> List[str]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(s)
            if limit is not None and len(out) >= limit:
                break
    return out


def pad_batch(seqs: List[List[int]], pad_id: int, device: torch.device) -> torch.Tensor:
    """Pad list of variable-length token-id lists to [B, T]."""
    max_len = max(len(x) for x in seqs)
    B = len(seqs)
    x = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        x[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
    return x


def decode_batch(tok: SMILESTokenizer, token_seqs: torch.Tensor) -> List[str]:
    """token_seqs: [B, T]"""
    return [tok.decode(seq.tolist()) for seq in token_seqs]


def write_smi(path: Path, smiles: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in smiles:
            s = s.strip()
            if s:
                f.write(s + "\n")


def build_train_set(train_path: Path) -> set[str]:
    train = load_smiles_lines(train_path, limit=None)
    canon = [canonicalize_smiles(s) for s in train]
    canon = [s for s in canon if s is not None]
    return set(canon)


def main():
    cfg = Config()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    max_len = cfg.max_len if MAX_LEN is None else MAX_LEN

    results_dir = OUT_DIR / "compare"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- tokenizer ----
    vocab = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    tok = SMILESTokenizer(vocab=vocab)

    # ---- model ----
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

    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ---- load conditioning SMILES (train) ----
    cond_smiles = load_smiles_lines(TRAIN_PATH, limit=N_CONDITION)

    # tokenize + pad
    cond_ids = [tok.encode(s) for s in cond_smiles]  # your encode: SMILES -> List[int]
    x = pad_batch(cond_ids, pad_id=tok.pad_id, device=device)  # [B, T]

    # ---- encode -> mu/logvar ----
    with torch.no_grad():
        mu, logvar = model.encoder(x)  # [B, Z], [B, Z]

    # choose how many to generate (can differ from N_CONDITION)
    # if you request more generations than conditioning points, we'll tile mu/logvar
    def tile_latent(z: torch.Tensor, n: int) -> torch.Tensor:
        if z.size(0) == n:
            return z
        reps = (n + z.size(0) - 1) // z.size(0)
        zt = z.repeat(reps, 1)[:n]
        return zt

    mu_g = tile_latent(mu, N_GENERATE)
    logvar_g = tile_latent(logvar, N_GENERATE)

    # ---- generate 3 ways ----
    with torch.no_grad():
        # PRIOR
        toks_prior = model.sample(
            num_samples=N_GENERATE,
            bos_id=tok.bos_id,
            eos_id=tok.eos_id,
            max_len=max_len,
            device=device,
        )

        # POSTERIOR_MU (deterministic)
        toks_post_mu = model.sample_from_z(
            mu_g,
            bos_id=tok.bos_id,
            eos_id=tok.eos_id,
            max_len=max_len,
            device=device,
        )

        # POSTERIOR_SAMPLE (stochastic)
        z_post = model.reparameterize(mu_g, logvar_g)
        toks_post = model.sample_from_z(
            z_post,
            bos_id=tok.bos_id,
            eos_id=tok.eos_id,
            max_len=max_len,
            device=device,
        )

    smiles_prior = decode_batch(tok, toks_prior)
    smiles_post_mu = decode_batch(tok, toks_post_mu)
    smiles_post = decode_batch(tok, toks_post)

    # ---- write files ----
    write_smi(PRIOR_SMI, smiles_prior)
    write_smi(POST_MU_SMI, smiles_post_mu)
    write_smi(POST_SAMPLE_SMI, smiles_post)

    print(f"[OK] wrote {PRIOR_SMI} ({len(smiles_prior)})")
    print(f"[OK] wrote {POST_MU_SMI} ({len(smiles_post_mu)})")
    print(f"[OK] wrote {POST_SAMPLE_SMI} ({len(smiles_post)})")

    # ---- quality report (uses your metrics.py functions) ----
    train_set = build_train_set(TRAIN_PATH)

    rep_prior = quality_report(smiles_prior, train_set)
    print_report("PRIOR", rep_prior)

    rep_post_mu = quality_report(smiles_post_mu, train_set)
    print_report("POSTERIOR_MU", rep_post_mu)

    rep_post = quality_report(smiles_post, train_set)
    print_report("POSTERIOR_SAMPLE", rep_post)


if __name__ == "__main__":
    main()
