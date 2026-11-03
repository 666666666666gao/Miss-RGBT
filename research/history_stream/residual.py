"""Isolated residual composition for a future history-assisted missing stream."""
import torch
from torch import nn


class HistoryResidualAdapter(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128):
        super().__init__()
        self.feature_dim = feature_dim
        self.age_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
        )
        input_dim = 2 * feature_dim + hidden_dim
        self.residual = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, native, history, age_frames, available):
        assert native.ndim == 3 and native.shape == history.shape
        batch, tokens, feature_dim = native.shape
        assert feature_dim == self.feature_dim
        assert age_frames.shape == (batch, 1)
        assert available.shape == (batch, 1)

        age = self.age_embedding(torch.log1p(age_frames))
        age = age.unsqueeze(1).expand(-1, tokens, -1)
        delta = self.residual(torch.cat((native, history, age), dim=-1))
        return native + (1. - available[:, None, :]) * delta
