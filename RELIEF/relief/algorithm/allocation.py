from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def elastic_budget(T_star: float, T_o: float, tau_n: float, k_min: int) -> int:
    return max(k_min, int(math.floor((T_star - T_o) / tau_n)))


def max_round_time(T_star: float, T_o: float, tau: Mapping[int, float], k_min: Mapping[int, int], n_G: Mapping[int, int]) -> float:
    return max(T_o + min(elastic_budget(T_star, T_o, tau[n], k_min[n]), n_G[n]) * tau[n] for n in tau)


def full_training_target(T_o: float, tau: Mapping[int, float], n_G: Mapping[int, int]) -> float:
    return max(T_o + n_G[n] * tau[n] for n in tau)


def search_T_star(
    T_o: float,
    tau: Mapping[int, float],
    k_min: Mapping[int, int],
    n_G: Mapping[int, int],
    lo_mult: float,
    hi_mult: float,
    iters: int,
) -> float:
    lo = lo_mult * min(tau.values())
    hi = hi_mult * max(tau.values())
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if max_round_time(mid, T_o, tau, k_min, n_G) <= mid:
            hi = mid
        else:
            lo = mid
    return hi


def interpolate_T_star(lam: float, T_a: float, T_f: float) -> float:
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lambda must lie in [0, 1]")
    return (1.0 - lam) * T_a + lam * T_f


def greedy_allocation(G_n: Sequence[int], mandatory: Sequence[int], d_bar: Mapping[int, float], k_n: int) -> List[int]:
    required = set(mandatory)
    S_n = [j for j in G_n if j in required]
    optional = sorted((j for j in G_n if j not in required), key=lambda j: (-d_bar.get(j, 0.0), j))
    return S_n + optional[: max(0, k_n - len(S_n))]


def random_allocation(G_n: Sequence[int], mandatory: Sequence[int], k_n: int, rng: np.random.Generator) -> List[int]:
    required = set(mandatory)
    S_n = [j for j in G_n if j in required]
    optional = [j for j in G_n if j not in required]
    count = min(max(0, k_n - len(S_n)), len(optional))
    picks = rng.choice(len(optional), size=count, replace=False) if count > 0 else []
    return S_n + [optional[i] for i in picks]


def apply_round_robin_floor(
    S_n: Sequence[int],
    G_n: Sequence[int],
    mandatory: Sequence[int],
    last_trained: Mapping[int, int],
    r: int,
    Q: int,
    d_bar: Mapping[int, float],
) -> List[int]:
    S_n = list(S_n)
    chosen = set(S_n)
    required = set(mandatory)
    stale = sorted((j for j in G_n if j not in chosen and r - last_trained.get(j, 0) >= Q), key=lambda j: last_trained.get(j, 0))
    optional = sorted((j for j in S_n if j not in required), key=lambda j: (d_bar.get(j, 0.0), j))
    for j in stale:
        if optional:
            S_n.remove(optional.pop(0))
        S_n.append(j)
    return S_n


class ElasticAllocator:
    def __init__(self, cfg, Q: int, rng: np.random.Generator):
        if cfg.allocation not in ("divergence", "random"):
            raise ValueError(f"unknown allocation {cfg.allocation}")
        self.lam = cfg.lam
        self.allocation = cfg.allocation
        self.mandatory_inclusion = cfg.mandatory_inclusion
        self.search = cfg.tstar
        self.Q = Q
        self.rng = rng
        self.T_a: Optional[float] = None
        self.T_f: Optional[float] = None
        self.T_star: Optional[float] = None
        self.last_trained: Dict[int, Dict[int, int]] = defaultdict(dict)

    def k_min(self, mandatory: Sequence[int]) -> int:
        return len(mandatory) if self.mandatory_inclusion else 1

    def needs_calibration(self, r: int) -> bool:
        return self.T_star is None or r % self.search.refresh_every == 0

    def calibrate(self, T_o: float, tau: Mapping[int, float], k_min: Mapping[int, int], n_G: Mapping[int, int]) -> float:
        self.T_a = full_training_target(T_o, tau, n_G)
        self.T_f = search_T_star(T_o, tau, k_min, n_G, self.search.lo_mult, self.search.hi_mult, self.search.iters)
        self.T_star = interpolate_T_star(self.lam, self.T_a, self.T_f)
        return self.T_star

    def allocate(
        self,
        r: int,
        G: Mapping[int, Sequence[int]],
        mandatory: Mapping[int, Sequence[int]],
        d_bar: Mapping[int, float],
        tau: Mapping[int, float],
        T_o: float,
    ) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
        S: Dict[int, List[int]] = {}
        k: Dict[int, int] = {}
        for n, G_n in G.items():
            if self.lam == 0.0:
                S[n], k[n] = list(G_n), len(G_n)
                continue
            required = list(mandatory[n]) if self.mandatory_inclusion else []
            k[n] = elastic_budget(self.T_star, T_o, tau[n], self.k_min(mandatory[n]))
            if self.allocation == "divergence":
                S_n = greedy_allocation(G_n, required, d_bar, k[n])
            else:
                S_n = random_allocation(G_n, required, k[n], self.rng)
            S[n] = sorted(apply_round_robin_floor(S_n, G_n, required, self.last_trained[n], r, self.Q, d_bar))
        return S, k

    def record(self, r: int, S: Mapping[int, Sequence[int]]) -> None:
        for n, S_n in S.items():
            for j in S_n:
                self.last_trained[n][j] = r
