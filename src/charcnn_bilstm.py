"""Shared preprocessing for the CharCNN + BiLSTM + Attention URL model.

This module is deliberately numpy-only so the deployed API can encode URLs
without importing torch.  Training and inference both call
``encode_url_bilstm``, which applies the production ``canonicalize_url`` first,
so the model always sees URLs in exactly the form the API serves.
"""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

from src.url_normalization import canonicalize_url

MAX_SEQUENCE_LENGTH = 256
PAD_INDEX = 0

# 95 printable ASCII characters: space through tilde
PRINTABLE_ASCII = (
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
    "-._~:/?#[]@!$&'()*+,;=%"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    " \"\\^`{|}<>"
)
UNK_INDEX = len(PRINTABLE_ASCII) + 1
VOCAB_SIZE = UNK_INDEX + 1  # includes PAD (0) and UNK


def build_enhanced_vocab() -> Dict[str, int]:
    vocab: Dict[str, int] = {"<PAD>": PAD_INDEX}
    for idx, ch in enumerate(PRINTABLE_ASCII, start=1):
        vocab[ch] = idx
    vocab["<UNK>"] = UNK_INDEX
    return vocab


def encode_url_bilstm(url: str, vocab: Dict[str, int], max_len: int = MAX_SEQUENCE_LENGTH) -> np.ndarray:
    # Keep the beginning (scheme, host, leading path) when truncating.
    text = canonicalize_url(url)[:max_len]
    unk = vocab.get("<UNK>", UNK_INDEX)
    encoded = [vocab.get(ch, unk) for ch in text]
    encoded.extend([PAD_INDEX] * (max_len - len(encoded)))
    return np.asarray(encoded, dtype=np.int64)


def encode_urls_bilstm(urls: Iterable[str], vocab: Dict[str, int], max_len: int = MAX_SEQUENCE_LENGTH) -> np.ndarray:
    return np.stack([encode_url_bilstm(u, vocab, max_len=max_len) for u in urls], axis=0)
