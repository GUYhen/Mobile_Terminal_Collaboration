from __future__ import annotations

import copy
import random
from collections import deque
from typing import Dict, List

import cvxpy as cp
import numpy as np
import torch
import torch.nn.functional as F_nn
from torch.optim import Adam

from .channel import WirelessChannel
from .config import ComputeConfig, Config
from .networks import build_network
from .resource_allocation import LN2, Allocation
from .system_model import delta_cp, e_cp


class ConvexOptimization:
    def __init__(self, compute: ComputeConfig, channel: WirelessChannel):
        self.mu = compute.mu
        self.G_n = compute.G_n
        self.kappa = compute.kappa
        self.e_max = compute.e_max
        self.B = channel.B
        self.P_n = channel.P_n
        self.channel = channel

    def solve_pf(self, zeta_n: float, h2: float, D: float) -> Dict:
        if LN2 * D >= self.e_max * self.B * h2:
            return dict(chi=1.0, rho=1.0, feasible=False)
        chi = cp.Variable()
        delta_tx = cp.Variable()
        s = cp.Variable()
        objective = cp.Minimize(self.mu * zeta_n / self.G_n * cp.inv_pos(chi) + delta_tx)
        constraints = [
            cp.constraints.ExpCone(cp.Constant(D * LN2 / self.B), delta_tx, s),
            self.kappa * self.mu * zeta_n * self.G_n ** 2 * cp.square(chi) + (s - delta_tx) / h2 <= self.e_max,
            delta_tx >= D / (self.B * np.log2(1.0 + self.P_n * h2)),
            chi >= 0.0,
            chi <= 1.0,
        ]
        cp.Problem(objective, constraints).solve(solver=cp.MOSEK)
        dtx = float(delta_tx.value)
        rho = min((2.0 ** (D / (dtx * self.B)) - 1.0) / (self.P_n * h2), 1.0)
        return dict(chi=float(chi.value), rho=rho, feasible=True)

    def solve(self, zeta: np.ndarray, h2: np.ndarray, D: float) -> Allocation:
        sols = [self.solve_pf(float(z), float(h), D) for z, h in zip(zeta, h2)]
        chi = np.array([s["chi"] for s in sols])
        rho = np.array([s["rho"] for s in sols])
        dtx = self.channel.delta_tx(D, rho, h2)
        return Allocation(
            chi=chi,
            rho=rho,
            delta_cp=delta_cp(self.mu, zeta, chi, self.G_n),
            delta_tx=dtx,
            e_cp=e_cp(self.kappa, self.mu, zeta, chi, self.G_n),
            e_tx=self.channel.e_tx(rho, dtx),
            lambda1=np.full(len(sols), np.nan),
            feasible=np.array([s["feasible"] for s in sols], dtype=bool),
        )


class AoIGreedy:
    def __init__(self, K: int):
        self.K = K

    def select_actions(self, S: np.ndarray, m_tilde: np.ndarray, deterministic: bool = True):
        Delta = S[-1, :, 2]
        return [int(n) for n in np.argsort(-Delta, kind="stable")[: self.K]], None


class MADDQN:
    def __init__(self, cfg: Config, M: int, N: int, K: int, F: int, device: torch.device,
                 rng: np.random.Generator):
        self.cfg = cfg.maddqn
        self.N = N
        self.K = K
        self.device = device
        self.rng = rng
        self.Q = [build_network(cfg.network, M, N, F, N).to(device) for _ in range(K)]
        self.Q_target = [copy.deepcopy(q) for q in self.Q]
        self.optimizers = [Adam(q.parameters(), lr=self.cfg.beta) for q in self.Q]
        self.replay = deque(maxlen=self.cfg.replay_buffer)
        self.steps = 0

    def epsilon(self) -> float:
        frac = min(self.steps / self.cfg.eps_decay_steps, 1.0)
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    @torch.no_grad()
    def select_actions(self, S: np.ndarray, m_tilde: np.ndarray, deterministic: bool = False):
        S_t = torch.as_tensor(S, dtype=torch.float32, device=self.device).unsqueeze(0)
        actions = []
        for k in range(self.K):
            if not deterministic and self.rng.random() < self.epsilon():
                actions.append(int(self.rng.integers(self.N)))
            else:
                actions.append(int(self.Q[k](S_t)[0].argmax().item()))
        return actions, None

    def update(self) -> None:
        c = self.cfg
        if len(self.replay) < c.N_batch:
            return
        batch = random.sample(self.replay, c.N_batch)
        S = torch.as_tensor(np.stack([b[0] for b in batch]), dtype=torch.float32, device=self.device)
        A = torch.as_tensor(np.stack([b[1] for b in batch]), dtype=torch.long, device=self.device)
        R = torch.as_tensor([b[2] for b in batch], dtype=torch.float32, device=self.device)
        S_next = torch.as_tensor(np.stack([b[3] for b in batch]), dtype=torch.float32, device=self.device)
        not_done = 1.0 - torch.as_tensor([float(b[4]) for b in batch], device=self.device)
        for k in range(self.K):
            q = self.Q[k](S).gather(1, A[:, k: k + 1]).squeeze(1)
            with torch.no_grad():
                a_star = self.Q[k](S_next).argmax(dim=1, keepdim=True)
                y = R + c.gamma * not_done * self.Q_target[k](S_next).gather(1, a_star).squeeze(1)
            loss = F_nn.mse_loss(q, y)
            self.optimizers[k].zero_grad()
            loss.backward()
            self.optimizers[k].step()
        if self.steps % c.target_update == 0:
            for q, q_t in zip(self.Q, self.Q_target):
                q_t.load_state_dict(q.state_dict())

    def train(self, env, E: int) -> List[List[Dict]]:
        history = []
        for e in range(E):
            S = env.reset()
            episode = []
            for t in range(env.T):
                env.configure(env.allocation.chi, env.allocation.rho)
                actions, _ = self.select_actions(S, env.m_tilde)
                S_next, R, done, info = env.step(actions)
                self.replay.append((S, np.array(actions), R, S_next, done))
                self.steps += 1
                self.update()
                episode.append(info)
                S = S_next
                if done:
                    break
            history.append(episode)
            print(f"[MADDQN] episode {e + 1}/{E} reward {sum(i['reward'] for i in episode):.4f}")
        return history

    def state_dict(self) -> Dict:
        return {f"Q{k}": q.state_dict() for k, q in enumerate(self.Q)}
