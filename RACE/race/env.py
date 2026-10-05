from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

from .channel import WirelessChannel
from .config import Config
from .fl.trainer import WFLTrainer
from .flmd import E_set, Lambda_Theta, m, m_tilde
from .platoon import Platoon
from .resource_allocation import Allocation
from .system_model import (W, aoi_update, assignment, check_constraints, delta_cp, delta_n, delta_round, e_cp,
                           e_n, objective_term)


class WFLPlatoonEnv:
    def __init__(self, cfg: Config, platoon: Platoon, channel: WirelessChannel, allocator, trainer: WFLTrainer):
        self.cfg = cfg
        self.platoon = platoon
        self.channel = channel
        self.allocator = allocator
        self.trainer = trainer
        self.N = cfg.platoon.N
        self.K = cfg.channel.K
        self.M = cfg.mdp.M
        self.T = cfg.mdp.T
        self.D = cfg.compute.D
        self.zeta = trainer.zeta
        self.Lambda_Theta = cfg.flmd.Lambda_Theta
        self.feature_dim = 4
        self.allocation = None
        self.chi = None
        self.rho = None

    def reset(self) -> np.ndarray:
        self.platoon.reset()
        self.trainer.reset()
        self.t = 0
        self.Delta = np.zeros(self.N)
        self.grad_F_sq_0 = None
        self.W = W(self.zeta)
        if self.allocation is not None:
            self.configure(self.chi, self.rho)
        self.h2_m = self.channel.h2(self.platoon.advance(self.M * self.cfg.platoon.tau, self.M))
        self.calculate_flmd()
        return self.observe()

    def stage1(self) -> Allocation:
        self.allocation = self.allocator.solve(self.zeta, self.h2_m[-1], self.D)
        self.configure(self.allocation.chi, self.allocation.rho)
        return self.allocation

    def configure(self, chi: np.ndarray, rho: np.ndarray) -> None:
        self.chi = np.array(chi, dtype=float)
        self.rho = np.array(rho, dtype=float)
        for n, w in enumerate(self.W):
            w.C_n = {"phi": 0.0, "B": self.channel.B, "rho": float(self.rho[n]), "chi": float(self.chi[n])}

    def calculate_flmd(self) -> None:
        self.local = self.trainer.local_round(self.t)
        self.Theta = self.local.Theta
        if self.grad_F_sq_0 is None:
            self.grad_F_sq_0 = self.local.grad_F_sq
        f = self.cfg.flmd
        self.Lambda_Theta_t = Lambda_Theta(self.local.grad_F_sq, self.grad_F_sq_0, f.Lambda_min, f.Lambda_max,
                                           f.beta_Lambda)
        self.m = m(self.Theta, self.Lambda_Theta_t)
        self.m_tilde = m_tilde(self.Theta, self.Lambda_Theta_t, f.beta_mask, self.cfg.convergence.mu,
                               self.cfg.convergence.L, self.t)

    def observe(self) -> np.ndarray:
        S = np.empty((self.M, self.N, self.feature_dim), dtype=np.float32)
        for mm in range(self.M):
            S[mm, :, 0] = self.Theta
            S[mm, :, 1] = self.h2_m[mm]
            S[mm, :, 2] = self.Delta
            S[mm, :, 3] = self.m
        return S

    def costs(self):
        c = self.cfg.compute
        h2 = self.h2_m[-1]
        d_cp = delta_cp(c.mu, self.zeta, self.chi, c.G_n)
        d_tx = self.channel.delta_tx(self.D, self.rho, h2)
        return d_cp, d_tx, e_cp(c.kappa, c.mu, self.zeta, self.chi, c.G_n), self.channel.e_tx(self.rho, d_tx)

    def reward(self) -> float:
        r = self.cfg.mdp
        return -float(r.alpha * self.Delta.sum() + r.beta * np.sum(self.Theta ** 2)) / (self.M * self.K)

    def step(self, actions: Sequence[int]) -> Tuple[np.ndarray, float, bool, Dict]:
        phi = assignment(actions, self.N)
        served = phi.sum(axis=0)
        for n, w in enumerate(self.W):
            w.C_n["phi"] = float(served[n])
        d_cp, d_tx, e_cp_n, e_tx_n = self.costs()
        delta_max = delta_round(phi, delta_n(d_cp, d_tx))
        S_t = np.intersect1d(np.flatnonzero(served > 0), E_set(self.Theta, self.Lambda_Theta))
        self.trainer.aggregate(S_t, self.local)
        energy = e_n(e_cp_n, served * e_tx_n)
        constraints = check_constraints(phi, energy, self.cfg.compute.e_max, self.chi, self.rho,
                                        self.Theta[S_t], self.Lambda_Theta)
        self.Delta = aoi_update(self.Delta, phi, delta_max)
        R = self.reward()
        info = {
            "t": self.t,
            "reward": R,
            "sum_aoi": float(self.Delta.sum()),
            "flmd": float(self.Theta.mean()),
            "objective": objective_term(self.Delta, self.Theta, self.cfg.mdp.lambda1, self.cfg.mdp.lambda2),
            "delta_max": delta_max,
            "F_S": self.trainer.F(self.local.f, S_t),
            "S": S_t.tolist(),
            "Lambda_Theta_t": self.Lambda_Theta_t,
            "constraints": constraints,
        }
        self.t += 1
        done = self.t >= self.T
        self.h2_m = self.channel.h2(self.platoon.advance(delta_max, self.M))
        if not done:
            self.calculate_flmd()
        return self.observe(), R, done, info
