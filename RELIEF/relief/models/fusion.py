from __future__ import annotations

import math
from typing import Dict, Sequence

import torch
import torch.nn as nn


class MDLoRAFusion(nn.Module):
    def __init__(self, d: Sequence[int], d_o: int, rho: int, frozen_base: bool, b_mode: str = "shared"):
        super().__init__()
        if b_mode not in ("shared", "block"):
            raise ValueError(f"unknown b_mode {b_mode}")
        self.d = list(d)
        self.D = sum(self.d)
        self.d_o = d_o
        self.rho = rho
        self.b_mode = b_mode
        self.A = nn.ParameterList([nn.Parameter(torch.empty(rho, d_m)) for d_m in self.d])
        n_B = 1 if b_mode == "shared" else len(self.d)
        self.B = nn.ParameterList([nn.Parameter(torch.empty(d_o, rho)) for _ in range(n_B)])
        if frozen_base:
            self.W0 = nn.Linear(self.D, d_o)
            self.W0.requires_grad_(False)
            self.bias = None
        else:
            self.W0 = None
            self.bias = nn.Parameter(torch.zeros(d_o))
        bound = 1.0 / math.sqrt(self.D)
        for A_m in self.A:
            nn.init.uniform_(A_m, -bound, bound)
        for B in self.B:
            if frozen_base:
                nn.init.zeros_(B)
            else:
                nn.init.uniform_(B, -1.0 / math.sqrt(rho), 1.0 / math.sqrt(rho))

    def forward(self, h: Sequence[torch.Tensor]) -> torch.Tensor:
        A_h = [h_m @ A_m.t() for h_m, A_m in zip(h, self.A)]
        if self.b_mode == "shared":
            out = torch.stack(A_h).sum(0) @ self.B[0].t()
        else:
            out = sum(A_h_m @ B_m.t() for A_h_m, B_m in zip(A_h, self.B))
        if self.W0 is not None:
            out = out + self.W0(torch.cat(list(h), dim=-1))
        if self.bias is not None:
            out = out + self.bias
        return out

    def macs_per_sample(self) -> Dict[str, float]:
        macs = {f"A.{m}": float(self.rho * d_m) for m, d_m in enumerate(self.d)}
        for i in range(len(self.B)):
            macs[f"B.{i}"] = float(self.d_o * self.rho)
        if self.W0 is not None:
            macs["B.0"] += float(self.D * self.d_o)
        return macs
