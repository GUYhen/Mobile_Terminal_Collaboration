from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..algorithm.aggregation import CohortAggregator
from ..algorithm.allocation import ElasticAllocator
from ..algorithm.divergence import DivergenceTracker, measure_divergences
from ..algorithm.groups import GroupRegistry
from ..analysis import blockwise_cosine_similarity, divergence_utility_correlation, leakage_ratio
from ..metrics import evaluate_predictions, time_to_accuracy
from .client import Client
from .system import SystemModel

logger = logging.getLogger("relief")


class ReliefServer:
    def __init__(
        self,
        cfg,
        model: nn.Module,
        registry: GroupRegistry,
        clients: Sequence[Client],
        system: SystemModel,
        device: torch.device,
        test_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        Y: Optional[Dict[str, List[int]]] = None,
    ):
        self.cfg = cfg
        self.model = model
        self.registry = registry
        self.clients = {c.n: c for c in clients}
        self.system = system
        self.device = device
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.Y = Y
        self.rng = np.random.default_rng(cfg.seed)
        self.theta = registry.state(model)
        self.tracker = DivergenceTracker(cfg.relief.gamma)
        self.allocator = ElasticAllocator(cfg.relief, cfg.relief.Q or registry.round_robin_period(), self.rng)
        self.aggregator = CohortAggregator(registry, cfg.relief.aggregation, cfg.relief.b_aggregation)
        self.d_prev: Dict[int, float] = {}
        self.total_params = sum(p.numel() for p in model.parameters())
        self.history: List[Dict[str, Any]] = []
        self.utility_pairs: List[Tuple[float, float]] = []
        self.clock = 0.0
        self.energy = 0.0

    def select(self) -> List[int]:
        ids = sorted(self.clients)
        count = max(1, int(round(self.cfg.fl.participation * len(ids))))
        if count >= len(ids):
            return ids
        return sorted(int(n) for n in self.rng.choice(ids, size=count, replace=False))

    def modalities(self, C_hat: Sequence[int], dynamic: bool) -> Dict[int, frozenset]:
        M = {n: set(self.clients[n].M_n) for n in C_hat}
        frac = self.cfg.system.dynamic_drop_frac
        count = int(round(frac * len(C_hat))) if dynamic else 0
        if count > 0:
            for n in self.rng.choice(list(C_hat), size=count, replace=False):
                n = int(n)
                if M[n]:
                    M[n].discard(int(self.rng.choice(sorted(M[n]))))
        return {n: frozenset(M_n) for n, M_n in M.items()}

    def cohort_modalities(self, M: Mapping[int, frozenset]) -> Dict[int, frozenset]:
        return {n: M.get(n, c.M_n) for n, c in self.clients.items()}

    def group_sets(self, M: Mapping[int, frozenset], r: int) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
        include_B = not (self.cfg.relief.b_aggregation == "freeze" and r > 0)
        G = {n: self.registry.accessible(M_n, self.cfg.relief.train_scope, include_B) for n, M_n in M.items()}
        mandatory = {n: self.registry.mandatory(M_n) for n, M_n in M.items()}
        return G, mandatory

    def train_and_aggregate(self, S: Mapping[int, Sequence[int]], M: Mapping[int, frozenset]):
        fl = self.cfg.fl
        uploads = {
            n: self.clients[n].local_train(self.model, self.registry, self.theta, S_n, M[n], fl.E, fl.lr, self.device)
            for n, S_n in S.items()
        }
        theta, active_cohort = self.aggregator.aggregate(self.theta, uploads, M, sorted(S))
        return uploads, theta, active_cohort

    def initialize(self) -> None:
        C_hat = sorted(self.clients)
        M = self.modalities(C_hat, dynamic=False)
        G, _ = self.group_sets(M, 0)
        tau = {n: self.system.tau(n, G[n]) for n in C_hat}
        uploads, self.theta, active_cohort = self.train_and_aggregate(G, M)
        self.d_prev = measure_divergences(uploads, self.registry, self.cohort_modalities(M))
        self.tracker.initialize(self.d_prev)
        self.allocator.record(0, G)
        self.log(0, G, {n: len(G_n) for n, G_n in G.items()}, tau, uploads, M, active_cohort)

    def run_round(self, r: int) -> None:
        C_hat = self.select()
        M = self.modalities(C_hat, dynamic=True)
        G, mandatory = self.group_sets(M, r)
        tau = {n: self.system.tau(n, G[n]) for n in C_hat}
        T_o = self.system.overhead(G)
        if self.allocator.needs_calibration(r):
            k_min = {n: self.allocator.k_min(mandatory[n]) for n in C_hat}
            self.allocator.calibrate(T_o, tau, k_min, {n: len(G[n]) for n in C_hat})
        self.tracker.update(self.d_prev)
        S, k = self.allocator.allocate(r, G, mandatory, self.tracker.d_bar, tau, T_o)
        if self.cfg.eval.utility_probe:
            self.probe(S, G, M)
        uploads, self.theta, active_cohort = self.train_and_aggregate(S, M)
        self.d_prev = measure_divergences(uploads, self.registry, self.cohort_modalities(M))
        self.allocator.record(r, S)
        self.log(r, S, k, tau, uploads, M, active_cohort)

    def probe(self, S: Mapping[int, Sequence[int]], G: Mapping[int, Sequence[int]], M: Mapping[int, frozenset]) -> None:
        loader = self.val_loader if self.val_loader is not None else self.test_loader
        for g in self.registry:
            include = {n: sorted(set(S_n) | {g.gid}) if g.gid in G[n] else list(S_n) for n, S_n in S.items()}
            exclude = {n: [j for j in S_n if j != g.gid] for n, S_n in S.items()}
            f1 = [self.evaluate(loader, self.train_and_aggregate(S_alt, M)[1])["macro_f1"] for S_alt in (include, exclude)]
            self.utility_pairs.append((self.tracker.d_bar.get(g.gid, 0.0), f1[0] - f1[1]))

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, theta: Optional[Mapping[str, torch.Tensor]] = None) -> Dict[str, Any]:
        self.registry.load_state(self.model, self.theta if theta is None else theta)
        self.model.eval()
        y_true, y_pred = [], []
        for x, y in loader:
            y_pred.append(self.model([x_m.to(self.device) for x_m in x]).argmax(-1).cpu())
            y_true.append(y)
        spec = self.cfg.spec
        return evaluate_predictions(
            torch.cat(y_true).numpy(), torch.cat(y_pred).numpy(), spec.num_classes, self.Y, spec.rare_modalities
        )

    def log(self, r, S, k, tau, uploads, M, active_cohort) -> None:
        cost = self.system.round_cost(S, tau)
        self.clock += cost.T_round
        self.energy += cost.energy
        trained = [sum(self.registry.numel(j) for j in S_n) / self.total_params for S_n in S.values()]
        record: Dict[str, Any] = {
            "round": r,
            "T_round": cost.T_round,
            "wall_clock": self.clock,
            "energy": cost.energy,
            "cum_energy": self.energy,
            "comm_upload_mb": cost.upload_bytes / 1e6,
            "comm_download_mb": cost.download_bytes / 1e6,
            "trained_pct": 100.0 * float(np.mean(trained)),
            "T_star": self.allocator.T_star,
            "k": {str(n): int(k_n) for n, k_n in k.items()},
            "S_size": {str(n): len(S_n) for n, S_n in S.items()},
            "active_cohort": active_cohort,
        }
        ev = self.cfg.eval
        if r % ev.eval_every == 0 or r == self.cfg.fl.R:
            record.update(self.evaluate(self.test_loader))
            if self.val_loader is not None:
                record.update({f"val_{key}": v for key, v in self.evaluate(self.val_loader).items()})
        if ev.diagnostics_every > 0 and r % ev.diagnostics_every == 0:
            record["block_cosine"] = {
                name: {f"{a}-{b}": v for (a, b), v in sims.items()}
                for name, sims in blockwise_cosine_similarity(uploads, self.registry).items()
            }
            record["block_divergence"] = {self.registry[j].name: d for j, d in self.d_prev.items() if self.registry[j].kind == "A"}
            record["leakage_ratio"] = leakage_ratio(uploads, self.registry, M)
        self.history.append(record)
        logger.info(
            "round %d | F1 %s | T_round %.3fs | energy %.1fJ | upload %.3fMB | T* %s",
            r,
            f"{record['macro_f1']:.4f}" if "macro_f1" in record else "-",
            cost.T_round,
            cost.energy,
            record["comm_upload_mb"],
            f"{self.allocator.T_star:.4f}" if self.allocator.T_star is not None else "-",
        )

    def run(self) -> Dict[str, Any]:
        self.initialize()
        for r in range(1, self.cfg.fl.R + 1):
            self.run_round(r)
        result: Dict[str, Any] = {"history": self.history, "TTA": time_to_accuracy(self.history, self.cfg.eval.tta_threshold)}
        if self.utility_pairs:
            result["divergence_utility"] = divergence_utility_correlation(self.utility_pairs)
        return result
