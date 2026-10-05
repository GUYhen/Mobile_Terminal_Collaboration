from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch.optim import Adam

from .config import Config
from .networks import Actor, Critic


@dataclass
class Transition:
    S: np.ndarray
    S_next: np.ndarray
    m_tilde: np.ndarray
    A: int
    log_pi_old: float
    R: float


class MAPPOAgent:
    def __init__(self, cfg: Config, M: int, N: int, F: int, device: torch.device):
        self.actor = Actor(cfg.network, M, N, F).to(device)
        self.critic = Critic(cfg.network, M, N, F).to(device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=cfg.mappo.beta)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=cfg.mappo.beta)
        self.buffer: List[Transition] = []


class MAPPO:
    def __init__(self, cfg: Config, M: int, N: int, K: int, F: int, device: torch.device):
        self.cfg = cfg.mappo
        self.N = N
        self.K = K
        self.device = device
        self.agents = [MAPPOAgent(cfg, M, N, F, device) for _ in range(K)]

    def _tensor(self, x, dtype=torch.float32) -> torch.Tensor:
        return torch.as_tensor(np.asarray(x), dtype=dtype, device=self.device)

    @torch.no_grad()
    def select_actions(self, S: np.ndarray, m_tilde: np.ndarray, deterministic: bool = False):
        S_t = self._tensor(S).unsqueeze(0)
        m_t = self._tensor(m_tilde).unsqueeze(0)
        actions, log_pis = [], []
        for agent in self.agents:
            pi = agent.actor.distribution(S_t, m_t)
            A = pi.probs.argmax(dim=-1) if deterministic else pi.sample()
            actions.append(int(A.item()))
            log_pis.append(float(pi.log_prob(A).item()))
        return actions, log_pis

    def update(self, agent: MAPPOAgent) -> Dict[str, float]:
        c = self.cfg
        batch = agent.buffer[-c.N_batch:]
        S = self._tensor([b.S for b in batch])
        S_next = self._tensor([b.S_next for b in batch])
        m_tilde = self._tensor([b.m_tilde for b in batch])
        A = self._tensor([b.A for b in batch], dtype=torch.long)
        log_pi_old = self._tensor([b.log_pi_old for b in batch])
        R = self._tensor([b.R for b in batch])

        epsilon_k = R + c.gamma * agent.critic(S_next) - agent.critic(S)

        L_psi = 0.5 * torch.mean(epsilon_k ** 2)
        agent.critic_optimizer.zero_grad()
        L_psi.backward()
        agent.critic_optimizer.step()

        epsilon_k = epsilon_k.detach()
        A_hat = torch.zeros_like(epsilon_k)
        running = torch.zeros((), device=self.device)
        for j in reversed(range(len(batch))):
            running = epsilon_k[j] + c.gamma * c.lambda_ * running
            A_hat[j] = running

        pi = agent.actor.distribution(S, m_tilde)
        r_k = torch.exp(pi.log_prob(A) - log_pi_old)
        L_k = torch.clamp(r_k, 1.0 - c.epsilon, 1.0 + c.epsilon)
        J = torch.mean(torch.min(r_k * A_hat, L_k * A_hat))
        agent.actor_optimizer.zero_grad()
        (-J).backward()
        agent.actor_optimizer.step()
        return {"L_psi": float(L_psi.item()), "J": float(J.item())}

    def train(self, env, E: int) -> List[List[Dict]]:
        history = []
        for e in range(E):
            S = env.reset()
            for agent in self.agents:
                agent.buffer.clear()
            episode = []
            for t in range(env.T):
                chi_star, rho_star = env.allocation.chi, env.allocation.rho
                env.configure(chi_star, rho_star)
                m_tilde = env.m_tilde.copy()
                actions, log_pis = self.select_actions(S, m_tilde)
                S_next, R, done, info = env.step(actions)
                for k, agent in enumerate(self.agents):
                    agent.buffer.append(Transition(S, S_next, m_tilde, actions[k], log_pis[k], R))
                    info[f"agent{k}"] = self.update(agent)
                episode.append(info)
                S = S_next
                if done:
                    break
            history.append(episode)
            print(f"[MAPPO] episode {e + 1}/{E} reward {sum(i['reward'] for i in episode):.4f}")
        return history

    def state_dict(self) -> Dict:
        return {f"agent{k}": {"actor": a.actor.state_dict(), "critic": a.critic.state_dict()}
                for k, a in enumerate(self.agents)}
