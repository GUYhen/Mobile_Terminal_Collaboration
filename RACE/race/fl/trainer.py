from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader, Dataset

from ..config import FLConfig
from ..flmd import Theta
from .compression import compress
from .model import DeepLabV3Plus


@dataclass
class LocalRound:
    omega: torch.Tensor
    xi: float
    grad_f: List[torch.Tensor]
    grad_f_hat: List[torch.Tensor]
    f: np.ndarray
    Theta: np.ndarray
    grad_F_sq: float


class WFLTrainer:
    def __init__(self, model: DeepLabV3Plus, clients: Sequence[Dataset], zeta: np.ndarray, test_set: Dataset,
                 weights: torch.Tensor, cfg: FLConfig, device: torch.device):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.zeta = zeta
        self.N = len(clients)
        self.params = model.federated_parameters()
        self.omega_0 = parameters_to_vector(self.params).detach().clone()
        self.criterion = nn.CrossEntropyLoss(weight=weights.to(device), ignore_index=cfg.ignore_index)
        self.loaders = [DataLoader(c, batch_size=cfg.batch_size, shuffle=True) for c in clients]
        self.test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False)
        self.reset()

    def reset(self) -> None:
        vector_to_parameters(self.omega_0.clone(), self.params)

    def xi_t(self, t: int) -> float:
        return self.cfg.xi * (1.0 - min(t, self.cfg.epochs) / self.cfg.epochs) ** self.cfg.poly_power

    def omega(self) -> torch.Tensor:
        return parameters_to_vector(self.params).detach().clone()

    def local_gradient(self, n: int):
        self.model.train()
        for p in self.params:
            p.grad = None
        f_n = 0.0
        for x, y in self.loaders[n]:
            x, y = x.to(self.device), y.to(self.device)
            loss = self.criterion(self.model(x), y) * x.shape[0] / self.zeta[n]
            loss.backward()
            f_n += float(loss.item())
        grad = torch.cat([p.grad.reshape(-1) for p in self.params]).detach().clone()
        return grad, f_n

    def local_round(self, t: int) -> LocalRound:
        omega = self.omega()
        xi = self.xi_t(t)
        grad_f, grad_f_hat, f, Th = [], [], np.zeros(self.N), np.zeros(self.N)
        for n in range(self.N):
            g_n, f[n] = self.local_gradient(n)
            omega_n = omega - xi * g_n
            Th[n] = Theta(omega_n, omega)
            grad_f.append(g_n)
            grad_f_hat.append(compress(g_n, self.cfg.topk, self.cfg.quant_bits))
        zeta = torch.tensor(self.zeta, dtype=omega.dtype, device=omega.device)
        grad_F = sum(z * g for z, g in zip(zeta, grad_f)) / zeta.sum()
        return LocalRound(omega=omega, xi=xi, grad_f=grad_f, grad_f_hat=grad_f_hat, f=f, Theta=Th,
                          grad_F_sq=float(torch.sum(grad_F ** 2)))

    def F(self, f: np.ndarray, S: Sequence[int]) -> float:
        S = list(S)
        return float(np.sum(self.zeta[S] * f[S]) / np.sum(self.zeta[S])) if S else float("nan")

    def aggregate(self, S: Sequence[int], local: LocalRound) -> None:
        S = list(S)
        if not S:
            return
        omega_n = {n: local.omega - local.xi * local.grad_f_hat[n] for n in S}
        omega_next = sum(float(self.zeta[n]) * omega_n[n] for n in S) / float(self.zeta[S].sum())
        vector_to_parameters(omega_next, self.params)

    @torch.no_grad()
    def mIoU(self) -> float:
        self.model.eval()
        C = self.cfg.num_classes
        confusion = torch.zeros(C, C, dtype=torch.int64, device=self.device)
        for x, y in self.test_loader:
            pred = self.model(x.to(self.device)).argmax(dim=1)
            y = y.to(self.device)
            valid = y != self.cfg.ignore_index
            confusion += torch.bincount(C * y[valid] + pred[valid], minlength=C * C).view(C, C)
        tp = confusion.diag().float()
        union = confusion.sum(0).float() + confusion.sum(1).float() - tp
        return float((tp / union.clamp_min(1.0)).mean())
