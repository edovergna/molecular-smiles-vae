"""
SMILES Variational Autoencoder model.

Implements the encoder, decoder, and VAE architecture for SMILES generation.
"""

from doctest import debug
import torch
import torch.nn as nn
from typing import Optional, Tuple
import torch.nn.functional as F

class Encoder(nn.Module):
    """Encoder network for SMILES VAE."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, 
                 latent_dim: int, num_layers: int = 1, pad_idx: int = 0, dropout: float = 0.0):
        """
        Initialize the encoder.
        
        Args:
            vocab_size: Size of the vocabulary.
            embedding_dim: Dimension of token embeddings.
            hidden_dim: Dimension of hidden states.
            latent_dim: Dimension of latent space.
            num_layers: Number of recurrent layers.
        """
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder.
        
        Args:
            x: Input token sequences [batch_size, seq_len].
            
        Returns:
            Tuple of (mean, log_variance) for latent distribution.
        """
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        h_n = h_n[-1]  # Take the last layer's hidden state

        mean = self.fc_mean(h_n)
        log_var = self.fc_log_var(h_n)
        
        return mean, log_var


class Decoder(nn.Module):
    """Decoder network for SMILES VAE."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 latent_dim: int, num_layers: int = 1, pad_idx: int = 0, bos_idx: int = 1, 
                 eos_idx: int = 2, unk_idx: int = 3, dropout: float = 0.0, word_dropout_p: float = 0.2):
        """
        Initialize the decoder.
        
        Args:
            vocab_size: Size of the vocabulary.
            embedding_dim: Dimension of token embeddings.
            hidden_dim: Dimension of hidden states.
            latent_dim: Dimension of latent space.
            num_layers: Number of recurrent layers.
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.word_dropout_p = word_dropout_p

        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.unk_idx = unk_idx

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.z_to_step = nn.Linear(latent_dim, embedding_dim)

        # Map latent -> initial hidden for all layers
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim * num_layers)

        self.lstm = nn.LSTM(
            embedding_dim + embedding_dim,  # concat z at each step
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_projection = nn.Linear(hidden_dim, vocab_size)


    def apply_word_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T] token indices
        """
        if not self.training or self.word_dropout_p <= 0.0:
            return x

        # sample mask
        drop_mask = torch.rand_like(x.float()) < self.word_dropout_p

        # never drop special tokens
        drop_mask &= (x != self.bos_idx)
        drop_mask &= (x != self.eos_idx)
        drop_mask &= (x != self.pad_idx)

        x = x.clone()
        x[drop_mask] = self.unk_idx
        return x

        
    def forward(self, z: torch.Tensor, x_inp: torch.Tensor) -> torch.Tensor:
        """
        Teacher forcing decoder.

        Args:
            z:     [B, Z] latent code
            x_inp: [B, T] input tokens (e.g., tokens[:-1])

        Returns:
            logits: [B, T, V]
        """

        batch_size = z.size(0)

        h0 = self.latent_to_hidden(z).view(self.num_layers, batch_size, self.hidden_dim)
        c0 = torch.zeros_like(h0)

        x_inp = self.apply_word_dropout(x_inp)

        emb = self.embedding(x_inp)              # [B, T, E]

        z_step = self.z_to_step(z).unsqueeze(1)          # [B, 1, E]
        z_step = z_step.expand(-1, emb.size(1), -1)      # [B, T, E]
        
        dec_in = torch.cat([emb, z_step], dim=-1)        # [B, T, 2E]

        out, _ = self.lstm(dec_in, (h0, c0))             # [B, T, H]
        logits = self.output_projection(out)             # [B, T, V]

        return logits


class SMILESVAE(nn.Module):
    """Variational Autoencoder for SMILES strings."""
    
    def __init__(self, vocab_size: int, pad_idx: int,  bos_idx: int = 1, eos_idx: int = 2, unk_idx: int = 3, embedding_dim: int = 256, hidden_dim: int = 512,
                 latent_dim: int = 128, num_layers: int = 2, dropout: float = 0.1, top_p: float = 0.9, 
                 temperature: float = 1.0, greedy: bool = False, word_dropout: float = 0.2):
        """
        Initialize the SMILES VAE.
        
        Args:
            vocab_size: Size of the vocabulary.
            embedding_dim: Dimension of token embeddings.
            hidden_dim: Dimension of hidden states.
            latent_dim: Dimension of latent space.
            num_layers: Number of recurrent layers.
        """
        super().__init__()

        self.pad_id = pad_idx
        self.latent_dim = latent_dim
        self.top_p = top_p
        self.temperature = temperature
        self.greedy = greedy
        
        self.encoder = Encoder(vocab_size, embedding_dim, hidden_dim, latent_dim, num_layers, pad_idx, dropout)
        self.decoder = Decoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            pad_idx=pad_idx,
            bos_idx=bos_idx,
            eos_idx=eos_idx,
            unk_idx=unk_idx,      
            dropout=dropout,
            word_dropout_p=word_dropout,
        )

        
    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE.
        
        Args:
            mean: Mean of latent distribution.
            log_var: Log variance of latent distribution.
            
        Returns:
            Sampled latent code.
        """
        
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mean + eps * std
    
    def forward(self, x_inp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.
        
        Args:
            x: Input token sequences [batch_size, seq_len].
            
        Returns:
            Tuple of (reconstruction, mean, log_variance).
        """

        mean, log_var = self.encoder(x_inp)

        z = self.reparameterize(mean, log_var)

        recon_x = self.decoder(z, x_inp)  # Teacher forcing with x_inp

        return recon_x, mean, log_var
    

    @torch.no_grad()
    def encode(self, x: torch.Tensor):
        """
        Convenience wrapper.
        x: [B, T] token ids
        returns: (mu, log_var) both [B, Z]
        """
        self.eval()
        return self.encoder(x)
    

    @torch.no_grad()
    def sample_from_z(
        self,
        z: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation conditioned on provided latent codes z.

        Args:
            z: [B, Z]
            bos_id/eos_id: special token ids
            max_len: max generation steps (not counting BOS)
            device: optional (defaults to z.device)

        Returns:
            token ids: [B, <=max_len]  (does NOT include BOS, matches your current sample())
        """
        self.eval()

        if device is None:
            device = z.device
        z = z.to(device)

        B = z.size(0)

        x = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros((B, 1), dtype=torch.bool, device=device)

        generated = []

        for _ in range(max_len):
            logits = self.decoder(z, x)        # [B, T, V]
            next_logits = logits[:, -1, :]     # [B, V]

            if self.greedy:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                next_token = self.sample_top_p(
                    next_logits,
                    top_p=self.top_p,
                    temperature=self.temperature,
                    min_tokens_to_keep=1,
                )

            # if already finished, keep emitting EOS
            next_token = torch.where(
                finished,
                torch.full_like(next_token, eos_id),
                next_token,
            )

            generated.append(next_token)
            x = torch.cat([x, next_token], dim=1)

            finished |= (next_token == eos_id)
            if finished.all():
                break

        return torch.cat(generated, dim=1)

    
    @staticmethod
    def sample_top_p(
        logits: torch.Tensor,
        min_tokens_to_keep: int = 1,
        temperature: float = 1.0,
        top_p: float = 0.9,
        ) -> torch.Tensor:
        """
        Nucleus (top-p) sampling for a batch of logits.

        Args:
            logits: [B, V] unnormalized logits for next token.
            min_tokens_to_keep: ensure at least this many tokens are kept.
            temperature: sampling temperature (>0).
            top_p: cumulative probability threshold.

        Returns:
            next_token: [B, 1] sampled token ids
        """
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        logits = logits / temperature

        # Sort by probability
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)

        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Mask tokens with cumulative prob above top_p
        sorted_indices_to_remove = cumulative_probs > top_p

        # Keep at least min_tokens_to_keep
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[:, :min_tokens_to_keep] = False
        else:
            sorted_indices_to_remove[:, :1] = False

        # Set removed logits to -inf
        sorted_logits = sorted_logits.masked_fill(sorted_indices_to_remove, float("-inf"))

        # Sample from the filtered distribution
        probs = torch.softmax(sorted_logits, dim=-1)
        next_in_sorted = torch.multinomial(probs, num_samples=1)  # [B,1]

        # Map back to original token ids
        next_token = sorted_indices.gather(-1, next_in_sorted)    # [B,1]
        
        return next_token

    
    def sample(
        self,
        num_samples: int,
        bos_id: int,
        eos_id: int,
        max_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Generate samples from the latent space.

        Args:
            num_samples: number of sequences to generate
            bos_id: BOS token id
            eos_id: EOS token id
            max_len: max generation steps (not counting BOS)
            device: torch device

        Returns:
            token ids: [B, <=max_len]
        """

        self.eval()
        z = torch.randn(num_samples, self.latent_dim, device=device)

        return self.sample_from_z(z, bos_id=bos_id, eos_id=eos_id, max_len=max_len, device=device)

def vae_loss(logits: torch.Tensor, y_tgt: torch.Tensor, mean: torch.Tensor, 
             log_var: torch.Tensor, pad_id: int, beta: float = 1.0, capacity: float = 1.5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute VAE loss (reconstruction + KL divergence).
    
    Args:
        logits: Reconstruction logits [B, T, V].
        y_tgt: Target tokens [B, T].
        mean: Mean of latent distribution.
        log_var: Log variance of latent distribution.
        pad_id: Padding token ID.
        beta: Weight for KL divergence term.
        capacity: Target KL divergence (for capacity objective).
        
    Returns:
        Tuple of (total_loss, reconstruction_loss, kl_divergence).
    """
    
    B, T, V = logits.shape

    recon = F.cross_entropy(
        logits.reshape(B * T, V),
        y_tgt.reshape(B * T),
        ignore_index=pad_id,
    )

    kl_per_dim = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp())

    kl = kl_per_dim.sum(dim=1).mean()

    if capacity is None:
        # classic beta-VAE
        total = recon + beta * kl
    else:
        # capacity (target KL) objective
        total = recon + beta * torch.abs(kl - capacity)

    return total, recon, kl
