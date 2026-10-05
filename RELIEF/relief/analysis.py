from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr

from .algorithm.divergence import Uploads
from .algorithm.groups import GroupRegistry


def flatten_update(update: Mapping[str, torch.Tensor], names: Sequence[str]) -> torch.Tensor:
    return torch.cat([update[n].flatten().float() for n in names])


def blockwise_cosine_similarity(uploads: Uploads, registry: GroupRegistry) -> Dict[str, Dict[Tuple[int, int], float]]:
    out = {}
    for g in registry:
        if g.kind != "A":
            continue
        devices = sorted(n for n in uploads if g.gid in uploads[n])
        vecs = {n: flatten_update(uploads[n][g.gid], g.param_names) for n in devices}
        out[g.name] = {
            (a, b): float(F.cosine_similarity(vecs[a], vecs[b], dim=0))
            for i, a in enumerate(devices)
            for b in devices[i + 1:]
        }
    return out


def leakage_ratio(uploads: Uploads, registry: GroupRegistry, M: Mapping[int, frozenset]) -> Dict[str, float]:
    out = {}
    for g in registry:
        if g.kind != "A":
            continue
        inside = [n for n in uploads if g.gid in uploads[n] and g.modality in M[n]]
        outside = [n for n in uploads if g.gid in uploads[n] and g.modality not in M[n]]
        if not inside or not outside:
            continue
        g_bar = torch.stack([flatten_update(uploads[n][g.gid], g.param_names) for n in inside]).mean(0)
        norm = float(g_bar.norm())
        if norm > 0.0:
            out[g.name] = float(np.mean([float(flatten_update(uploads[n][g.gid], g.param_names).norm()) / norm for n in outside]))
    return out


def divergence_utility_correlation(pairs: Sequence[Tuple[float, float]], top_fraction: float = 0.25) -> Dict[str, float]:
    d = np.asarray([p[0] for p in pairs], dtype=float)
    lift = np.asarray([p[1] for p in pairs], dtype=float)
    out = {}
    if len(d) > 2:
        rho, p = pearsonr(d, lift)
        out["pearson"], out["p_value"] = float(rho), float(p)
        top = d >= np.quantile(d, 1.0 - top_fraction)
        if top.sum() > 2:
            rho_top, p_top = pearsonr(d[top], lift[top])
            out["pearson_top"], out["p_value_top"] = float(rho_top), float(p_top)
    return out
