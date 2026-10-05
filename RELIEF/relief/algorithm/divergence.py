from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

import torch

from .groups import GroupRegistry, ParamGroup

Uploads = Mapping[int, Mapping[int, Mapping[str, torch.Tensor]]]


def cohort(group: ParamGroup, M: Mapping[int, frozenset]) -> List[int]:
    return sorted(n for n, M_n in M.items() if group.modality is None or group.modality in M_n)


def cohort_divergence(group: ParamGroup, members: Sequence[int], uploads: Uploads) -> float:
    C = len(members)
    trained = [n for n in members if group.gid in uploads.get(n, {})]
    if C == 0 or not trained:
        return 0.0
    total = 0.0
    for name in group.param_names:
        deltas = torch.stack([uploads[n][group.gid][name].float() for n in trained])
        mean = deltas.sum(0) / C
        total += float((deltas - mean).pow(2).sum()) + (C - len(trained)) * float(mean.pow(2).sum())
    return total / C


def measure_divergences(uploads: Uploads, registry: GroupRegistry, M: Mapping[int, frozenset]) -> Dict[int, float]:
    d = {}
    for g in registry:
        members = cohort(g, M)
        if members:
            d[g.gid] = cohort_divergence(g, members, uploads)
    return d


class DivergenceTracker:
    def __init__(self, gamma: float):
        if not 0.0 < gamma < 1.0:
            raise ValueError("gamma must lie in (0, 1)")
        self.gamma = gamma
        self.d_bar: Dict[int, float] = {}

    def initialize(self, d_0: Mapping[int, float]) -> None:
        self.d_bar = dict(d_0)

    def update(self, d_r: Mapping[int, float]) -> Dict[int, float]:
        for j, d_j in d_r.items():
            prev = self.d_bar.get(j, d_j)
            self.d_bar[j] = self.gamma * d_j + (1.0 - self.gamma) * prev
        return self.d_bar
