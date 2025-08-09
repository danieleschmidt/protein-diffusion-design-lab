"""
SELFIES-based tokenizer for protein sequence representation.

This module implements a SELFIES (SELF-referencIng Embedded Strings) tokenizer
specifically adapted for protein sequences, providing grammar-constrained
molecular representation that ensures valid protein generation.
"""

import json
import pickle
import re
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    class MockNumpy:
        @staticmethod
        def mean(arr):
            return sum(arr)/len(arr) if arr else 0.5
        @staticmethod
        def array(data):
            return data
        @staticmethod
        def std(arr):
            return 1.0
        @staticmethod
        def min(arr):
            return min(arr) if arr else 0
        @staticmethod
        def max(arr):
            return max(arr) if arr else 1
    np = MockNumpy()
    NUMPY_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    from .. import mock_torch as torch
    TORCH_AVAILABLE = False
from dataclasses import dataclass

# Standard amino acid alphabet
STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
EXTENDED_AA = STANDARD_AA + "UOX"  # Include non-standard amino acids


@dataclass
class TokenizerConfig:
    """Configuration for SELFIES tokenizer."""
    vocab_size: int = 50000
    max_length: int = 512
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"
    bos_token: str = "<BOS>"
    eos_token: str = "<EOS>"
    mask_token: str = "<MASK>"
    special_tokens: List[str] = None
    
    def __post_init__(self):
        if self.special_tokens is None:
            self.special_tokens = [
                self.pad_token, self.unk_token, self.bos_token, 
                self.eos_token, self.mask_token
            ]


class SELFIESTokenizer:
    """
    SELFIES-based tokenizer for protein sequences.
    
    This tokenizer converts protein sequences into SELFIES representation,
    which provides grammatical constraints ensuring valid molecular structures.
    """
    
    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.vocab_to_id: Dict[str, int] = {}
        self.id_to_vocab: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Initialize with special tokens
        self._build_base_vocab()
    
    def _build_base_vocab(self):
        """Build initial vocabulary with special tokens and amino acids."""
        # Add special tokens first
        for token in self.config.special_tokens:
            self._add_token(token)
        
        # Add standard amino acids
        for aa in EXTENDED_AA:
            self._add_token(aa)
        
        # Add common protein motifs and structures
        common_motifs = [
            "HELIX", "SHEET", "LOOP", "TURN", "COIL",
            "ALPHA", "BETA", "GAMMA", "PHI", "PSI",
        ]
        for motif in common_motifs:
            self._add_token(f"[{motif}]")
        
        # Add SELFIES-style tokens for structural elements
        structural_tokens = [
            "[C]", "[N]", "[O]", "[S]", "[P]",  # Atomic elements
            "[Ring1]", "[Ring2]", "[Ring3]",    # Ring closures
            "[Branch1]", "[Branch2]",           # Branching
            "[=C]", "[=N]", "[=O]",            # Double bonds
            "[#C]", "[#N]",                    # Triple bonds
        ]
        for token in structural_tokens:
            self._add_token(token)
    
    def _add_token(self, token: str) -> int:
        """Add a token to the vocabulary."""
        if token not in self.vocab_to_id:
            token_id = len(self.vocab_to_id)
            self.vocab_to_id[token] = token_id
            self.id_to_vocab[token_id] = token
            self.vocab_size += 1
        return self.vocab_to_id[token]
    
    def sequence_to_selfies(self, sequence: str) -> str:
        """
        Convert protein sequence to SELFIES representation.
        
        Args:
            sequence: Protein sequence string
            
        Returns:
            SELFIES representation of the sequence
        """
        # Clean sequence
        sequence = sequence.upper().strip()
        
        # Convert to SELFIES-like representation
        selfies_tokens = []
        
        i = 0
        while i < len(sequence):
            # Check for multi-character motifs first
            found_motif = False
            
            # Look for common dipeptide patterns
            if i < len(sequence) - 1:
                dipeptide = sequence[i:i+2]
                if dipeptide in ["GP", "PG", "GG", "PP"]:  # Common flexible motifs
                    selfies_tokens.append(f"[{dipeptide}]")
                    i += 2
                    found_motif = True
            
            if not found_motif:
                # Single amino acid
                aa = sequence[i]
                if aa in EXTENDED_AA:
                    selfies_tokens.append(f"[{aa}]")
                else:
                    selfies_tokens.append(f"[{self.config.unk_token}]")
                i += 1
        
        return "".join(selfies_tokens)
    
    def selfies_to_sequence(self, selfies: str) -> str:
        """
        Convert SELFIES representation back to protein sequence.
        
        Args:
            selfies: SELFIES string
            
        Returns:
            Protein sequence
        """
        # Extract tokens from SELFIES string
        tokens = re.findall(r'\[([^\]]+)\]', selfies)
        
        sequence = []
        for token in tokens:
            if len(token) == 1 and token in EXTENDED_AA:
                sequence.append(token)
            elif len(token) == 2 and all(aa in EXTENDED_AA for aa in token):
                sequence.extend(list(token))
            elif token not in self.config.special_tokens:
                # Skip structural tokens that don't map to amino acids
                continue
        
        return "".join(sequence)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize input text into SELFIES tokens.
        
        Args:
            text: Input text (protein sequence or SELFIES)
            
        Returns:
            List of tokens
        """
        if text.startswith("[") and "]" in text:
            # Already in SELFIES format
            selfies = text
        else:
            # Convert sequence to SELFIES
            selfies = self.sequence_to_selfies(text)
        
        # Extract individual tokens
        tokens = re.findall(r'\[([^\]]+)\]', selfies)
        return [f"[{token}]" for token in tokens]
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
    ) -> Dict[str, List[int]]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Add BOS/EOS tokens
            max_length: Maximum sequence length
            padding: Pad to max_length
            truncation: Truncate if too long
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        if max_length is None:
            max_length = self.config.max_length
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Add special tokens
        if add_special_tokens:
            tokens = [self.config.bos_token] + tokens + [self.config.eos_token]
        
        # Truncate if necessary
        if truncation and len(tokens) > max_length:
            tokens = tokens[:max_length]
            if add_special_tokens:
                tokens[-1] = self.config.eos_token
        
        # Convert to IDs
        input_ids = []
        for token in tokens:
            if token in self.vocab_to_id:
                input_ids.append(self.vocab_to_id[token])
            else:
                input_ids.append(self.vocab_to_id[self.config.unk_token])
        
        # Create attention mask
        attention_mask = [1] * len(input_ids)
        
        # Pad if necessary
        if padding and len(input_ids) < max_length:
            pad_length = max_length - len(input_ids)
            pad_id = self.vocab_to_id[self.config.pad_token]
            
            input_ids.extend([pad_id] * pad_length)
            attention_mask.extend([0] * pad_length)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    
    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs or tensor
            skip_special_tokens: Skip special tokens in output
            
        Returns:
            Decoded text
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_vocab:
                token = self.id_to_vocab[token_id]
                if skip_special_tokens and token in self.config.special_tokens:
                    continue
                tokens.append(token)
        
        # Join tokens and convert back to sequence
        selfies = "".join(tokens)
        sequence = self.selfies_to_sequence(selfies)
        
        return sequence
    
    def batch_encode(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: Optional[str] = None,
    ) -> Dict[str, Union[List[List[int]], torch.Tensor]]:
        """
        Encode a batch of texts.
        
        Args:
            texts: List of input texts
            add_special_tokens: Add BOS/EOS tokens
            max_length: Maximum sequence length
            padding: Pad sequences
            truncation: Truncate long sequences
            return_tensors: Return format ("pt" for PyTorch tensors)
            
        Returns:
            Dictionary with batch encodings
        """
        batch_encodings = {
            "input_ids": [],
            "attention_mask": [],
        }
        
        for text in texts:
            encoding = self.encode(
                text,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
            )
            batch_encodings["input_ids"].append(encoding["input_ids"])
            batch_encodings["attention_mask"].append(encoding["attention_mask"])
        
        if return_tensors == "pt":
            batch_encodings = {
                key: torch.tensor(value, dtype=torch.long)
                for key, value in batch_encodings.items()
            }
        
        return batch_encodings
    
    def build_vocab_from_sequences(
        self,
        sequences: List[str],
        min_frequency: int = 2,
    ):
        """
        Build vocabulary from a collection of protein sequences.
        
        Args:
            sequences: List of protein sequences
            min_frequency: Minimum frequency for token inclusion
        """
        # Count token frequencies
        token_counts = {}
        
        for sequence in sequences:
            tokens = self.tokenize(sequence)
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
        
        # Add frequent tokens to vocabulary
        for token, count in sorted(token_counts.items(), key=lambda x: x[1], reverse=True):
            if count >= min_frequency and token not in self.vocab_to_id:
                if len(self.vocab_to_id) >= self.config.vocab_size:
                    break
                self._add_token(token)
    
    def save_pretrained(self, save_path: Union[str, Path]):
        """Save tokenizer to directory."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_path = save_path / "tokenizer_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        # Save vocabulary
        vocab_path = save_path / "vocab.json"
        with open(vocab_path, 'w') as f:
            json.dump(self.vocab_to_id, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, load_path: Union[str, Path]) -> 'SELFIESTokenizer':
        """Load tokenizer from directory."""
        load_path = Path(load_path)
        
        # Load config
        config_path = load_path / "tokenizer_config.json"
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = TokenizerConfig(**config_dict)
        
        # Create tokenizer
        tokenizer = cls(config)
        
        # Load vocabulary
        vocab_path = load_path / "vocab.json"
        with open(vocab_path, 'r') as f:
            vocab_to_id = json.load(f)
        
        tokenizer.vocab_to_id = vocab_to_id
        tokenizer.id_to_vocab = {int(v): k for k, v in vocab_to_id.items()}
        tokenizer.vocab_size = len(vocab_to_id)
        
        return tokenizer
    
    def get_special_token_ids(self) -> Dict[str, int]:
        """Get IDs for special tokens."""
        return {
            token: self.vocab_to_id[token]
            for token in self.config.special_tokens
            if token in self.vocab_to_id
        }
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size
    
    def __call__(self, text: str, **kwargs) -> Dict[str, List[int]]:
        """Make tokenizer callable."""
        return self.encode(text, **kwargs)