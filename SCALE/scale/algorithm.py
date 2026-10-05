import torch

from .aoi import avg_aoi, global_system_aoi, sum_aoi
from .env import SparsificationMDP
from .ppo import PPO, RolloutBuffer
from .sensitivity import identify_sensitive_layers


def run_scale(Theta, W_clients, M_sizes, n, layers, groups, aoi, t_fl, cfg, device):
    c = cfg.scale

    L_s, sensitivity = identify_sensitive_layers(Theta, W_clients, M_sizes, n, layers, c.lam, c.M, c.eps)

    aoi = aoi.copy()
    aoi.rebase(t_fl)
    mdp = SparsificationMDP(Theta, L_s, sensitivity["S"], groups, aoi)
    agent = PPO(mdp.dim_H, len(L_s), groups.G_l, c, device)
    B = RolloutBuffer()
    log = []

    for t in range(1, c.T_total + 1):
        R_sum, s_sum = 0.0, 0.0
        for step in range(1, c.T_collect + 1):
            H_t = mdp.state(t)
            ell, G_mask, s, logp = agent.act(H_t)
            G = torch.nonzero(G_mask).flatten().tolist()
            R, R_f, R_c = mdp.reward(ell, G, s, t, c.w_f, c.w_c)
            mdp.sparsify(ell, G, s, t)
            H_next = mdp.state(t if step < c.T_collect else t + 1)
            B.store(H_t, ell, G_mask, s, logp, R, H_next)
            R_sum += R
            s_sum += s

        A_t = mdp.aoi.A(t)
        record = {
            "t": t,
            "reward": R_sum,
            "s": s_sum / c.T_collect,
            "sum_aoi": sum_aoi(A_t),
            "avg_aoi": avg_aoi(A_t),
            "A_g": global_system_aoi(A_t, len(L_s)),
            "C_t": mdp.C_t,
        }

        if len(B) >= c.batch_size:
            record["L_clip"], record["L_V"] = agent.update(B)
            B.clear()
        log.append(record)

    return mdp.unlearned_model(), L_s, sensitivity, mdp, log
