"""Modern Character-Level CNN + Bidirectional LSTM + Self-Attention Phishing URL Detector.

Architecture:
1. Character Embedding (dim=64, vocab=97 printable ASCII characters)
2. Multi-Kernel Conv1d (k=3, 5, 7, filters=128 each) with BatchNorm & ReLU
3. MaxPool1d (stride=2) preserving sequential structure (256 -> 128 steps)
4. Bidirectional LSTM (2 layers, hidden_dim=128, bidirectional -> 256 dim)
5. Multi-Head / Additive Self-Attention Pooling + Temporal Max Pooling (512 dim)
6. Calibrated Dense Classification Head with Dropout & LayerNorm
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


def normalize_url_for_encoding(url: str) -> str:
    text = str(url).strip()
    if not text:
        return ""
    has_scheme = "://" in text
    parsed = urlsplit(text if has_scheme else f"//{text}")
    scheme = parsed.scheme.lower() if parsed.scheme else "http"
    netloc = parsed.netloc.lower()
    path = parsed.path
    if path == "/":
        path = ""
    # Reassemble: preserve case of path/query if needed, but lowercase scheme & host
    reassembled = urlunsplit((scheme, netloc, path, parsed.query, parsed.fragment))
    return reassembled


def encode_url_bilstm(url: str, vocab: Dict[str, int], max_len: int = MAX_SEQUENCE_LENGTH) -> np.ndarray:
    normalized = normalize_url_for_encoding(url)
    if len(normalized) > max_len:
        # Keep beginning of URL (scheme, domain, main path) rather than trimming the domain!
        normalized = normalized[:max_len]

    encoded = [vocab.get(ch, UNK_INDEX) for ch in normalized]
    if len(encoded) < max_len:
        encoded.extend([PAD_INDEX] * (max_len - len(encoded)))

    return np.array(encoded, dtype=np.int64)


def encode_urls_bilstm(urls: List[str], vocab: Dict[str, int], max_len: int = MAX_SEQUENCE_LENGTH) -> np.ndarray:
    return np.stack([encode_url_bilstm(u, vocab, max_len=max_len) for u in urls], axis=0)


class CharCNNBiLSTMAttention(nn.Module):
    """Hybrid Char-CNN + BiLSTM + Attention Network for Phishing URL Detection."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embedding_dim: int = 64,
        num_filters: int = 128,
        filter_sizes: Tuple[int, ...] = (3, 5, 7),
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=PAD_INDEX,
        )

        # Multi-scale 1D CNN
        self.conv_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=embedding_dim,
                    out_channels=num_filters,
                    kernel_size=k,
                    padding=k // 2,
                ),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
            )
            for k in filter_sizes
        ])

        total_conv_filters = num_filters * len(filter_sizes)  # 128 * 3 = 384
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)  # 256 -> 128 length
        self.conv_dropout = nn.Dropout(0.2)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=total_conv_filters,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if lstm_layers > 1 else 0.0,
        )
        lstm_out_dim = lstm_hidden_dim * 2  # 256

        # Self-Attention Pooling
        self.attention_score = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Dense Classifier Head
        pooled_dim = lstm_out_dim * 2  # Attention pooled (256) + Max pooled (256) = 512
        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        embedded = embedded.transpose(1, 2)  # (batch_size, embed_dim, seq_len)

        # Parallel Conv branches
        conv_outs = [branch(embedded) for branch in self.conv_branches]
        conv_cat = torch.cat(conv_outs, dim=1)  # (batch_size, 384, seq_len)

        pooled_conv = self.pool(conv_cat)  # (batch_size, 384, seq_len // 2)
        pooled_conv = self.conv_dropout(pooled_conv)

        # LSTM sequence modeling
        batch_size = x.size(0)
        lstm_input = pooled_conv.transpose(1, 2)  # (batch_size, seq_len // 2, 384)
        h0 = torch.zeros(self.lstm.num_layers * 2, batch_size, self.lstm.hidden_size, device=x.device, dtype=torch.float32)
        c0 = torch.zeros(self.lstm.num_layers * 2, batch_size, self.lstm.hidden_size, device=x.device, dtype=torch.float32)
        lstm_out, _ = self.lstm(lstm_input, (h0, c0))  # (batch_size, seq_len // 2, 256)

        # Attention pooling
        attn_scores = self.attention_score(lstm_out)  # (batch_size, seq_len // 2, 1)
        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch_size, 256)

        # Temporal max pooling
        max_pooled, _ = torch.max(lstm_out, dim=1)  # (batch_size, 256)

        # Combined representation
        features = torch.cat([context, max_pooled], dim=1)  # (batch_size, 512)

        # Classification logits
        logits = self.classifier(features)  # (batch_size, 1)
        return logits.squeeze(-1)
