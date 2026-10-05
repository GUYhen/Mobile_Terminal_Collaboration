from __future__ import annotations

import torch

QUANT_DTYPE = {16: torch.float16, 32: torch.float32}


def top_k(g: torch.Tensor, ratio: float) -> torch.Tensor:
    k = max(1, int(round(ratio * g.numel())))
    idx = torch.topk(g.abs(), k).indices
    sparse = torch.zeros_like(g)
    sparse[idx] = g[idx]
    return sparse


def quantize(g: torch.Tensor, bits: int) -> torch.Tensor:
    return g.to(QUANT_DTYPE[bits]).to(g.dtype)


def compress(g: torch.Tensor, ratio: float, bits: int) -> torch.Tensor:
    return quantize(top_k(g, ratio), bits)


def D_bits(num_params: int, ratio: float, bits: int) -> float:
    return float(num_params * ratio * bits)
