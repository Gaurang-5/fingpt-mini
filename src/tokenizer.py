import json
import logging
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Tokenizer:
    """
    A tokenizer that converts text into token IDs based on character or word level mappings.

    Attributes
    ----------
    mode : str
        The tokenization mode ('char' or 'word').
    vocab_path : str
        The path to the JSON vocabulary file.
    vocab : dict
        A dictionary mapping strings to integer IDs.
    inverse_vocab : dict
        A dictionary mapping integer IDs to strings.
    """
    def __init__(self, vocab_path: str, mode: str = 'char', max_vocab_size: int = None):
        self.mode = mode
        self.vocab_path = vocab_path
        
        with open(vocab_path, 'r', encoding='utf-8') as f:
            full_vocab = json.load(f)
            
        if max_vocab_size is not None:
            # We assume the vocab dict is insertion-ordered by frequency 
            # (which is true in Python 3.7+ and build_vocab.py preserves this)
            self.vocab = dict(list(full_vocab.items())[:max_vocab_size])
            # Ensure UNK is always in the truncated vocab
            if "<UNK>" not in self.vocab:
                self.vocab["<UNK>"] = full_vocab.get("<UNK>", 1)
        else:
            self.vocab = full_vocab
            
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Use existing <UNK> and <PAD> tokens if they exist, otherwise default to 1 and 0
        self.unk_id = self.vocab.get("<UNK>", 1)
        self.pad_id = self.vocab.get("<PAD>", 0)

    def encode(self, text: str) -> List[int]:
        """
        Encodes a given string into a list of integer token IDs.

        Parameters
        ----------
        text : str
            The input string to encode.

        Returns
        -------
        List[int]
            The list of encoded integer token IDs.
        """
        if self.mode == 'char':
            # Split into individual characters
            return [self.vocab.get(c, self.unk_id) for c in text]
        elif self.mode == 'word':
            # Split by whitespace, lowercase, map to id, use <UNK> for OOV
            words = text.lower().split()
            return [self.vocab.get(w, self.unk_id) for w in words]
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def decode(self, ids: List[int]) -> str:
        """
        Decodes a list of integer token IDs back into a string.

        Parameters
        ----------
        ids : List[int]
            The list of token IDs to decode.

        Returns
        -------
        str
            The decoded text string.
        """
        if self.mode == 'char':
            # Reverse mapping, join without spaces
            return "".join(self.inverse_vocab.get(i, "<UNK>") for i in ids)
        elif self.mode == 'word':
            # Reverse mapping, join with spaces
            return " ".join(self.inverse_vocab.get(i, "<UNK>") for i in ids)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
