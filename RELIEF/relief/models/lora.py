from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rho: int):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.rho = rho
        w = base.weight
        self.lora_A = nn.Parameter(torch.empty(rho, base.in_features, device=w.device, dtype=w.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rho, device=w.device, dtype=w.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_A.t()) @ self.lora_B.t()


def inject_lora(root: nn.Module, targets: Iterable[str], rho: int) -> int:
    targets = set(targets)
    found = []
    for _, parent in root.named_modules():
        if isinstance(parent, LoRALinear):
            continue
        for name, child in parent.named_children():
            if name in targets and isinstance(child, nn.Linear):
                found.append((parent, name, child))
    for parent, name, child in found:
        setattr(parent, name, LoRALinear(child, rho))
    return len(found)
