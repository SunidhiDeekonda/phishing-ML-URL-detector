"""CharCNN + Bidirectional LSTM + Self-Attention phishing URL classifier (training only).

Architecture:
1. Character embedding (dim=64, 95 printable ASCII characters + PAD + UNK)
2. Multi-kernel Conv1d (k=3, 5, 7; 128 filters each) with BatchNorm and ReLU
3. MaxPool1d (stride=2): 256 -> 128 time steps
4. 2-layer bidirectional LSTM (hidden 128 per direction -> 256)
5. Additive self-attention pooling concatenated with temporal max pooling (512)
6. Dense head with LayerNorm and dropout producing one phishing logit

The deployed API runs the exported ONNX graph and never imports this module.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.charcnn_bilstm import PAD_INDEX, VOCAB_SIZE


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

        total_conv_filters = num_filters * len(filter_sizes)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv_dropout = nn.Dropout(0.2)

        self.lstm = nn.LSTM(
            input_size=total_conv_filters,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if lstm_layers > 1 else 0.0,
        )
        lstm_out_dim = lstm_hidden_dim * 2

        self.attention_score = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        pooled_dim = lstm_out_dim * 2
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
        embedded = self.embedding(x).transpose(1, 2)
        conv_cat = torch.cat([branch(embedded) for branch in self.conv_branches], dim=1)
        pooled_conv = self.conv_dropout(self.pool(conv_cat))

        batch_size = x.size(0)
        lstm_input = pooled_conv.transpose(1, 2)
        h0 = torch.zeros(self.lstm.num_layers * 2, batch_size, self.lstm.hidden_size, device=x.device, dtype=torch.float32)
        c0 = torch.zeros(self.lstm.num_layers * 2, batch_size, self.lstm.hidden_size, device=x.device, dtype=torch.float32)
        lstm_out, _ = self.lstm(lstm_input, (h0, c0))

        attn_weights = F.softmax(self.attention_score(lstm_out), dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)
        max_pooled, _ = torch.max(lstm_out, dim=1)

        logits = self.classifier(torch.cat([context, max_pooled], dim=1))
        return logits.squeeze(-1)
