"""
SMILES tokenization module.

Handles conversion between SMILES strings and token sequences.
"""

from typing import List, Dict


class SMILESTokenizer:
    """Tokenizer for SMILES molecular representations."""
    
    def __init__(self, vocab: Dict[str, int] = None):
        """
        Initialize the SMILES tokenizer.

        Args:
            vocab: Optional pre-defined vocabulary mapping tokens to indices.
        """

        # Define special tokens with fixed indices
        self.special_tokens = {
            "<pad>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "<unk>": 3,
        }

        if vocab is None:
            # Fresh vocabulary: start with special tokens
            self.vocab = dict(self.special_tokens)
        else:
            # Copy provided vocab and ensure special tokens are present & consistent
            self.vocab = dict(vocab)
            for tok, idx in self.special_tokens.items():
                if tok not in self.vocab:
                    self.vocab[tok] = idx
                else:
                    assert self.vocab[tok] == idx, (
                        f"Special token {tok} has index {self.vocab[tok]}, expected {idx}"
                    )

        # Build reverse vocabulary
        self.reverse_vocab = {idx: tok for tok, idx in self.vocab.items()}

        # Cache ids for convenience
        self.pad_id = self.vocab["<pad>"]
        self.bos_id = self.vocab["<bos>"]
        self.eos_id = self.vocab["<eos>"]
        self.unk_id = self.vocab["<unk>"]
        
    def fit(self, smiles_list: List[str]) -> None:
        """
        Build vocabulary from a list of SMILES strings.

        Args:
            smiles_list: List of SMILES strings.
        """

        # Collect unique characters from the dataset
        chars = set()
        for smiles in smiles_list:
            s = smiles.strip()
            if not s:
                continue
            chars.update(s)

        # Add characters to vocab in sorted (deterministic) order
        # This allows consistent indexing across runs
        last_idx = max(self.vocab.values(), default=-1) # Default to -1 if vocab is empty
        for ch in sorted(chars):
            if ch not in self.vocab:
                last_idx += 1
                self.vocab[ch] = last_idx

        # Rebuild reverse vocabulary
        self.reverse_vocab = {idx: tok for tok, idx in self.vocab.items()}

    
    def encode(self, smiles: str) -> List[int]:
        """
        Convert SMILES string to token indices.
        
        Args:
            smiles: SMILES string to encode.
            
        Returns:
            List of token indices.
        """

        encoded = [self.bos_id]

        for ch in smiles.strip():
            encoded.append(self.vocab.get(ch, self.unk_id))

        encoded.append(self.eos_id)

        return encoded

    
    def decode(self, tokens: List[int]) -> str:
        """
        Convert token indices back to SMILES string.
        
        Args:
            tokens: List of token indices.
            
        Returns:
            Decoded SMILES string.
        """
        smiles = []

        for idx in tokens:

            if idx == self.eos_id:
                break
            if idx in (self.bos_id, self.pad_id):
                continue

            tok = self.reverse_vocab.get(idx, "<unk>")
            # Handle unknown tokens with a placeholder
            if tok == "<unk>":
                smiles.append("?")
            else:
                smiles.append(tok)

        return "".join(smiles)
    

    def pad_or_truncate(self, sequence: List[int], max_length: int) -> List[int]:
        """
        Pad or truncate a sequence to a fixed length.
        
        Args:
            sequence: List of token indices.
            max_length: Desired sequence length.
            pad_id: Token index used for padding.
            
        Returns:
            Padded or truncated sequence.
        """
        if len(sequence) > max_length:
            return sequence[:max_length]
        else:
            return sequence + [self.pad_id] * (max_length - len(sequence))
    

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        return len(self.vocab)
