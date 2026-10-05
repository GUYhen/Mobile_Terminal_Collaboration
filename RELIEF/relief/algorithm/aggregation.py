from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import torch

from .divergence import Uploads
from .groups import GroupRegistry, ParamGroup


class CohortAggregator:
    def __init__(self, registry: GroupRegistry, mode: str = "cohort", b_aggregation: str = "modality_count"):
        if mode not in ("cohort", "fedavg"):
            raise ValueError(f"unknown aggregation {mode}")
        if b_aggregation not in ("modality_count", "uniform", "freeze"):
            raise ValueError(f"unknown B aggregation {b_aggregation}")
        self.registry = registry
        self.M = registry.M
        self.mode = mode
        self.b_aggregation = b_aggregation

    def w(self, C_hat: Sequence[int], M: Mapping[int, frozenset]) -> Dict[int, float]:
        total = sum(len(M[k]) / self.M for k in C_hat)
        return {n: (len(M[n]) / self.M) / total for n in C_hat}

    def weights(self, g: ParamGroup, C_hat: Sequence[int], uploads: Uploads, M: Mapping[int, frozenset]) -> Dict[int, float]:
        trained = [n for n in C_hat if g.gid in uploads.get(n, {})]
        if not trained:
            return {}
        if g.kind == "B" and g.modality is None:
            if self.b_aggregation == "uniform":
                return {n: 1.0 / len(C_hat) for n in trained}
            w = self.w(C_hat, M)
            return {n: w[n] for n in trained}
        if self.mode == "fedavg":
            return {n: 1.0 / len(C_hat) for n in trained}
        if g.modality is not None:
            C_tilde = [n for n in trained if g.modality in M[n]]
            return {n: 1.0 / len(C_tilde) for n in C_tilde}
        return {n: 1.0 / len(C_hat) for n in trained}

    @torch.no_grad()
    def aggregate(
        self,
        theta: Mapping[str, torch.Tensor],
        uploads: Uploads,
        M: Mapping[int, frozenset],
        C_hat: Sequence[int],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
        new = dict(theta)
        active_cohort: Dict[str, int] = {}
        for g in self.registry:
            w = self.weights(g, C_hat, uploads, M)
            if g.kind == "A":
                active_cohort[g.name] = len(w)
            if not w:
                continue
            for name in g.tensor_names:
                if torch.is_floating_point(theta[name]):
                    delta = torch.zeros_like(theta[name])
                    for n, w_n in w.items():
                        delta.add_(uploads[n][g.gid][name], alpha=w_n)
                    new[name] = theta[name] + delta
                else:
                    new[name] = theta[name] + torch.stack([uploads[n][g.gid][name] for n in w]).max()
        return new, active_cohort
