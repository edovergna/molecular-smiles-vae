"""
Training script for SMILES VAE.

Trains the variational autoencoder on SMILES data.
"""

import json
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import argparse

from src.config import Config
from src.device import get_device
from src.seed import set_seed
from src.data.tokenize import SMILESTokenizer
from src.data.dataset import SMILESDataset
from src.data.prepare_data import load_smiles_from_file, preprocess_smiles, split_data
from src.models.smiles_vae import SMILESVAE, vae_loss
from src.utils import load_checkpoint, save_checkpoint, count_parameters


def train_epoch(model, dataloader, optimizer, device, pad_id, beta: float = 1.0, grad_clip: float = 1.0, capacity: float = 1.5):
    """
    Train for one epoch.
    
    Args:
        model: SMILES VAE model.
        dataloader: Training data loader.
        optimizer: Optimizer.
        device: Device to train on.
        beta: KL divergence weight.
        
    Returns:
        Average loss for the epoch.
    """

    model.train()
    total_loss = 0.0

    for x, t in dataloader:
        x = x.to(device)   # [B, T]
        t = t.to(device)   # [B, T]

        logits, mean, log_var = model(x)

        loss, _, _ = vae_loss(
            logits,
            t,
            mean,
            log_var,
            pad_id=pad_id,
            beta=beta,
            capacity=capacity,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)
    

@torch.no_grad()
def validate(model, dataloader, device, pad_id, beta: float = 1.0, capacity: float = 1.5):
    """
    Validate the model.
    
    Args:
        model: SMILES VAE model.
        dataloader: Validation data loader.
        device: Device to validate on.
        beta: KL divergence weight.
        
    Returns:
        Average validation loss.
    """

    model.eval()
    total_loss = 0.0
    total_kl = 0.0

    for x, t in dataloader:
        x = x.to(device)
        t = t.to(device)

        logits, mean, log_var = model(x)

        loss, _, kl_loss = vae_loss(
            logits,
            t,
            mean,
            log_var,
            pad_id=pad_id,
            beta=beta,
            capacity=capacity,
        )

        total_kl += kl_loss.item()
        total_loss += loss.item()

    return (
        total_loss / max(len(dataloader), 1),
        total_kl / max(len(dataloader), 1),
    )

def capacity_for_epoch(epoch, warmup_epochs, c_start=0.2, c_end=1.5):
    progress = min(1.0, epoch / float(max(1, warmup_epochs)))
    return c_start + (c_end - c_start) * progress


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train SMILES VAE")
    parser.add_argument("--smiles_path", type=str, default=None, help="Path to raw SMILES txt file.")
    parser.add_argument("--outputs_dir", type=str, default=None, help="Output directory for checkpoints/logs.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--beta_kl", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--greedy", type=bool, default=False)
    parser.add_argument("--capacity", type=float, default=1.5, help="KL capacity for loss calculation.")
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Number of epochs to hold KL capacity at 0.")
    parser.add_argument("--word_dropout", type=float, default=0.2, help="Word dropout rate for decoder input.")
    args = parser.parse_args()

    # Load config and allow CLI overrides
    cfg = Config()

    raw_smiles_path = Path(args.smiles_path) if args.smiles_path else Path(cfg.raw_smiles_path)
    outputs_dir = Path(args.outputs_dir) if args.outputs_dir else Path(cfg.outputs_dir)

    epochs = args.epochs if args.epochs is not None else cfg.epochs
    batch_size = args.batch_size if args.batch_size is not None else cfg.batch_size
    lr = args.lr if args.lr is not None else cfg.lr
    beta_kl = args.beta_kl if args.beta_kl is not None else cfg.beta_kl
    seed = args.seed if args.seed is not None else cfg.seed
    top_p = args.top_p if args.top_p is not None else cfg.top_p
    temperature = args.temperature if args.temperature is not None else cfg.temperature
    greedy = args.greedy if args.greedy is not None else cfg.greedy
    capacity = args.capacity if args.capacity is not None else cfg.capacity
    warmup_epochs = args.warmup_epochs if args.warmup_epochs is not None else cfg.warmup_epochs
    word_dropout = args.word_dropout if args.word_dropout is not None else cfg.word_dropout

    outputs_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    device = get_device()

    print(f"Device: {device}")
    print(f"Raw SMILES: {raw_smiles_path}")
    print(f"Outputs: {outputs_dir}")

    # -------------------------
    # Load + preprocess data
    # -------------------------
    smiles = load_smiles_from_file(raw_smiles_path)
    smiles = preprocess_smiles(smiles)

    train_smiles, val_smiles, test_smiles = split_data(
        smiles,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=seed,
    )

    # Save splits (handy for reproducibility)
    (outputs_dir / "splits").mkdir(exist_ok=True)
    (outputs_dir / "splits" / "train.txt").write_text("\n".join(train_smiles) + "\n", encoding="utf-8")
    (outputs_dir / "splits" / "val.txt").write_text("\n".join(val_smiles) + "\n", encoding="utf-8")
    (outputs_dir / "splits" / "test.txt").write_text("\n".join(test_smiles) + "\n", encoding="utf-8")

    # -------------------------
    # Tokenizer + datasets
    # -------------------------
   
    vocab_path = outputs_dir / "vocab.json"

    if args.resume and vocab_path.exists():
        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        tokenizer = SMILESTokenizer(vocab=vocab)
        print(f"[OK] Loaded vocab from {vocab_path} (size={tokenizer.vocab_size})")
    else:
        tokenizer = SMILESTokenizer()
        tokenizer.fit(train_smiles)
        vocab_path.write_text(json.dumps(tokenizer.vocab, indent=2), encoding="utf-8")
        print(f"[OK] Saved vocab to {vocab_path} (size={tokenizer.vocab_size})")
    
    train_ds = SMILESDataset(train_smiles, tokenizer, max_length=cfg.max_len)
    val_ds = SMILESDataset(val_smiles, tokenizer, max_length=cfg.max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )

    # -------------------------
    # Model + optimizer
    # -------------------------

    model = SMILESVAE(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=cfg.emb_dim,
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
        num_layers=cfg.num_layers,

        pad_idx=tokenizer.pad_id,
        bos_idx=tokenizer.bos_id,
        eos_idx=tokenizer.eos_id,
        unk_idx=tokenizer.unk_id,

        dropout=cfg.dropout,
        word_dropout=word_dropout,

        temperature=temperature,
        top_p=top_p,
        greedy=greedy,
    ).to(device)


    print(f"Parameters: {count_parameters(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_epoch = 1
    best_val = float("inf")

    if args.resume:
        info = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            map_location=device,
            strict=True,
        )
        if info.get("epoch") is not None:
            start_epoch = int(info["epoch"]) + 1

        # If you stored metrics/best_val in checkpoint, restore it
        metrics = info.get("metrics") or {}
        if "best_val" in metrics:
            best_val = float(metrics["best_val"])

        print(f"Resumed from {args.resume} @ epoch {start_epoch}")

    # Save run config
    run_meta = {
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "beta_kl": beta_kl,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "raw_smiles_path": str(raw_smiles_path),
    }
    (outputs_dir / "run_config.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    # -------------------------
    # Train loop
    # -------------------------
    history = []

    for epoch in range(start_epoch, epochs + 1):
        capacity = capacity_for_epoch(epoch, cfg.warmup_epochs, c_start=0.2, c_end=cfg.capacity)

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            pad_id=tokenizer.pad_id,
            beta=beta_kl,
            grad_clip=cfg.grad_clip,
            capacity=capacity,
        )

        val_loss, val_kl = validate(
            model,
            val_loader,
            device,
            pad_id=tokenizer.pad_id,
            beta=beta_kl,
            capacity=capacity,
        )

        row = {"epoch": epoch, "train_loss": float(train_loss), "val_loss": float(val_loss), "val_kl": float(val_kl), "beta": float(beta_kl)}
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train {train_loss:.4f} | "
            f"val {val_loss:.4f} | "
            f"beta {beta_kl:.3f} | "
            f"kl {val_kl:.6f}"
        )

        # Save best
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss

        # Save checkpoint
        if (epoch % args.save_every == 0) or is_best or (epoch == epochs):
            ckpt_path = outputs_dir / ("best.pt" if is_best else f"epoch_{epoch:03d}.pt")
            save_checkpoint(
                ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={
                    "val_loss": float(val_loss),
                    "best_val": float(best_val),
                },
                extra={
                    "history": history,
                    "run_meta": run_meta,
                },
            )

    # Save training curve data
    (outputs_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[OK] Done. Best val loss: {best_val:.4f}")
    print(f"[OK] Artifacts saved to: {outputs_dir}")




if __name__ == "__main__":
    main()
