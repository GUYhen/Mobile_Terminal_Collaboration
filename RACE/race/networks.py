from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch.distributions import Categorical

from .config import NetworkConfig


class TSFEN(nn.Module):
    def __init__(self, N: int, F: int, num_heads: int, lstm_hidden: int, out_dim: int):
        super().__init__()
        self.mhsa = nn.MultiheadAttention(embed_dim=F, num_heads=num_heads, batch_first=True)
        self.lstm = nn.LSTM(input_size=N * F, hidden_size=lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, out_dim)

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        b, M, N, F = S.shape
        tokens = S.reshape(b * M, N, F)
        attended, _ = self.mhsa(tokens, tokens, tokens)
        out, _ = self.lstm(attended.reshape(b, M, N * F))
        h_M = out[:, -1]
        return self.fc(h_M)


class MLP(nn.Module):
    def __init__(self, M: int, N: int, F: int, hidden: Sequence[int], out_dim: int):
        super().__init__()
        layers, width = [], M * N * F
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU()]
            width = h
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        return self.net(S.flatten(start_dim=1))


def build_network(cfg: NetworkConfig, M: int, N: int, F: int, out_dim: int) -> nn.Module:
    if cfg.arch == "tsfen":
        return TSFEN(N, F, cfg.num_heads, cfg.lstm_hidden, out_dim)
    if cfg.arch == "mlp":
        return MLP(M, N, F, cfg.mlp_hidden, out_dim)
    raise ValueError(cfg.arch)


class Actor(nn.Module):
    def __init__(self, cfg: NetworkConfig, M: int, N: int, F: int):
        super().__init__()
        self.f_theta = build_network(cfg, M, N, F, N)

    def distribution(self, S: torch.Tensor, m_tilde: torch.Tensor) -> Categorical:
        return Categorical(logits=self.f_theta(S) * m_tilde)


class Critic(nn.Module):
    def __init__(self, cfg: NetworkConfig, M: int, N: int, F: int):
        super().__init__()
        self.V_psi = build_network(cfg, M, N, F, 1)

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        return self.V_psi(S).squeeze(-1)
