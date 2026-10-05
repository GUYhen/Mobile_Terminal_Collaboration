import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Beta, Categorical


class PolicyNetwork(nn.Module):
    def __init__(self, dim_H, M, G_l, hidden_dim):
        super().__init__()
        self.M = M
        self.G_l = G_l
        self.body = nn.Sequential(nn.Linear(dim_H, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.layer_head = nn.Linear(hidden_dim, M)
        self.group_head = nn.Linear(hidden_dim, M * G_l)
        self.ratio_head = nn.Linear(hidden_dim, 2)

    def forward(self, H):
        h = self.body(H)
        conc = F.softplus(self.ratio_head(h)) + 1.0
        return self.layer_head(h), self.group_head(h).view(-1, self.M, self.G_l), conc[:, 0], conc[:, 1]

    def log_prob(self, H, ell, G_mask, s):
        layer_logits, group_logits, a, b = self(H)
        rows = torch.arange(H.shape[0], device=H.device)
        pi_ell = Categorical(logits=layer_logits)
        pi_G = Bernoulli(logits=group_logits[rows, ell])
        pi_s = Beta(a, b)
        return pi_ell.log_prob(ell) + pi_G.log_prob(G_mask).sum(-1) + pi_s.log_prob(s)


class ValueNetwork(nn.Module):
    def __init__(self, dim_H, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim_H, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                                 nn.Linear(hidden_dim, 1))

    def forward(self, H):
        return self.net(H).squeeze(-1)


class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.H, self.ell, self.G, self.s, self.logp, self.R, self.H_next = [], [], [], [], [], [], []

    def store(self, H, ell, G_mask, s, logp, R, H_next):
        self.H.append(H)
        self.ell.append(ell)
        self.G.append(G_mask)
        self.s.append(s)
        self.logp.append(logp)
        self.R.append(R)
        self.H_next.append(H_next)

    def __len__(self):
        return len(self.R)


class PPO:
    def __init__(self, dim_H, M, G_l, cfg, device):
        self.cfg = cfg
        self.device = device
        self.pi_theta = PolicyNetwork(dim_H, M, G_l, cfg.hidden_dim).to(device)
        self.V_phi = ValueNetwork(dim_H, cfg.hidden_dim).to(device)
        self.opt_theta = torch.optim.SGD(self.pi_theta.parameters(), lr=cfg.eta_a)
        self.opt_phi = torch.optim.SGD(self.V_phi.parameters(), lr=cfg.eta_c)

    @torch.no_grad()
    def act(self, H):
        H = H.to(self.device).unsqueeze(0)
        layer_logits, group_logits, a, b = self.pi_theta(H)
        ell = Categorical(logits=layer_logits).sample()
        G_mask = Bernoulli(logits=group_logits[0, ell]).sample()
        s = Beta(a, b).sample().clamp(1e-6, 1.0 - 1e-6)
        logp = self.pi_theta.log_prob(H, ell, G_mask, s)
        return int(ell.item()), G_mask[0].cpu(), float(s.item()), float(logp.item())

    def tensors(self, B):
        H = torch.stack(B.H).to(self.device)
        ell = torch.tensor(B.ell, dtype=torch.long, device=self.device)
        G = torch.stack(B.G).to(self.device)
        s = torch.tensor(B.s, device=self.device)
        logp = torch.tensor(B.logp, device=self.device)
        R = torch.tensor(B.R, device=self.device)
        H_next = torch.stack(B.H_next).to(self.device)
        return H, ell, G, s, logp, R, H_next

    @torch.no_grad()
    def advantages(self, R, H, H_next):
        V = self.V_phi(H)
        V_next = self.V_phi(H_next)
        delta = R + self.cfg.gamma * V_next - V
        A_hat = torch.zeros_like(R)
        running = 0.0
        for k in reversed(range(len(R))):
            running = delta[k] + self.cfg.gamma * self.cfg.lambda_gae * running
            A_hat[k] = running
        return A_hat, A_hat + V

    def update(self, B):
        H, ell, G, s, logp_old, R, H_next = self.tensors(B)
        A_hat, R_hat = self.advantages(R, H, H_next)
        eps = self.cfg.clip_eps
        for _ in range(self.cfg.K_epoch):
            r_theta = torch.exp(self.pi_theta.log_prob(H, ell, G, s) - logp_old)
            L_clip = torch.min(r_theta * A_hat, torch.clamp(r_theta, 1.0 - eps, 1.0 + eps) * A_hat).mean()
            self.opt_theta.zero_grad()
            (-L_clip).backward()
            self.opt_theta.step()
            L_V = ((self.V_phi(H) - R_hat) ** 2).mean()
            self.opt_phi.zero_grad()
            L_V.backward()
            self.opt_phi.step()
        return float(L_clip.item()), float(L_V.item())
