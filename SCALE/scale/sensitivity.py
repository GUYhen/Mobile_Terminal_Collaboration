import math

import torch


def pearson_correlation(W_ln, W_l):
    x = W_ln.reshape(-1).double()
    y = W_l.reshape(-1).double()
    x = x - x.mean()
    y = y - y.mean()
    return float((x * y).sum() / torch.sqrt((x * x).sum() * (y * y).sum()))


def alignment_sensitivity(rho_ln, eps):
    return -0.5 * math.log(max(1.0 - rho_ln ** 2, eps))


def leave_one_out(W_clients, M_sizes, n, l):
    others = [j for j in W_clients if j != n]
    total = float(sum(M_sizes[j] for j in others))
    return sum(M_sizes[j] * W_clients[j][l].double() for j in others) / total


def normalized_distribution(W, eps):
    p = W.reshape(-1).double().abs() + eps
    return p / p.sum()


def distributional_sensitivity(W_l, W_l_minus_n, eps):
    P = normalized_distribution(W_l, eps)
    Q = normalized_distribution(W_l_minus_n, eps)
    return float((P * torch.log(P / Q)).sum())


def combined_sensitivity(S_a, S_d, lam):
    return lam * S_a + (1.0 - lam) * S_d


def select_sensitive_layers(S, M):
    return sorted(S, key=lambda l: S[l], reverse=True)[:M]


def identify_sensitive_layers(Theta, W_clients, M_sizes, n, layers, lam, M, eps):
    rho, S_a, S_d, S = {}, {}, {}, {}
    for l in layers:
        rho[l] = pearson_correlation(W_clients[n][l], Theta[l])
        S_a[l] = alignment_sensitivity(rho[l], eps)
        W_l_minus_n = leave_one_out(W_clients, M_sizes, n, l)
        S_d[l] = distributional_sensitivity(Theta[l], W_l_minus_n, eps)
        S[l] = combined_sensitivity(S_a[l], S_d[l], lam)
    L_s = select_sensitive_layers(S, M)
    return L_s, {"rho": rho, "S_a": S_a, "S_d": S_d, "S": S}
