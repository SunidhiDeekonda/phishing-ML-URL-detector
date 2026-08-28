"""Character tokenization utilities for CNN-style sequence input.

The reference implementation uses a fixed character alphabet (lowercase URLs only),
with padding and unknown handling defined by index.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np

MAX_SEQUENCE_LENGTH = 200
PAD_INDEX = 0

REFERENCE_CHARACTERS = (
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
    ":/?&=.%-_+#@~"
)
UNK_INDEX = len(REFERENCE_CHARACTERS) + 1


def normalize_url(url: str) -> str:
    return "" if url is None else str(url).strip().lower()


def build_vocab(urls: Iterable[str] | None = None) -> Dict[str, int]:
    # urls parameter is kept for backward compatibility with the existing pipeline.
    del urls

    vocab: Dict[str, int] = {
        "<PAD>": PAD_INDEX,
    }

    for index, char in enumerate(REFERENCE_CHARACTERS, start=1):
        vocab[char] = index

    vocab["<UNK>"] = UNK_INDEX

    return vocab


def save_vocab(vocab: Dict[str, int], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(vocab, ensure_ascii=True, indent=2, sort_keys=True))


def encode_url(url: str, vocab: Dict[str, int], max_len: int = MAX_SEQUENCE_LENGTH) -> np.ndarray:
    normalized = normalize_url(url)
    if len(normalized) > max_len:
        normalized = normalized[-max_len:]

    encoded = [vocab.get(char, UNK_INDEX) for char in normalized]
    if len(encoded) < max_len:
        encoded.extend([PAD_INDEX] * (max_len - len(encoded)))

    return np.array(encoded, dtype=np.int32)


def encode_urls(urls: Iterable[str], vocab: Dict[str, int], max_len: int = MAX_SEQUENCE_LENGTH) -> np.ndarray:
    sequences = [encode_url(url, vocab, max_len=max_len) for url in urls]
    return np.stack(sequences, axis=0)
