import math

import torch

from .model import group_of
from .utils import pixel_ce


def task_forgetting(loss, t, t_prime):
    return loss[t][t_prime] - loss[t_prime][t_prime]


def cumulative_forgetting(loss, t):
    if t == 0:
        return 0.0
    return sum(task_forgetting(loss, t, t_prime) for t_prime in range(0, t)) / t


def memory_coverage(M, stream, t):
    return M / sum(len(stream.Y[i]) for i in range(0, t))


def replace_layer(Theta_base, Theta_new, l, prefixes):
    Theta = {key: value.clone() for key, value in Theta_base.items()}
    for key, value in Theta_new.items():
        if group_of(key, prefixes) == l:
            Theta[key] = value[: Theta_base[key].shape[0]].clone() if value.shape != Theta_base[key].shape else value.clone()
    return Theta


@torch.no_grad()
def layer_replacement(model, Theta_0, Theta_t, prefixes, mIoU):
    model.load_state_dict(Theta_0)
    base = mIoU(model)
    change = {}
    for l in prefixes:
        model.load_state_dict(replace_layer(Theta_0, Theta_t, l, prefixes))
        change[l] = base - mIoU(model)
    total = sum(change.values())
    contribution = {l: (value / total if total else 0.0) for l, value in change.items()}
    return {"mIoU_0": base, "drop": change, "contribution": contribution}


def kappa(model, Theta_t, batches, prefixes, ignore_index, device):
    model.eval()
    model.zero_grad(set_to_none=True)
    for x, y, _ in batches:
        pixel_ce(model(x.to(device)), y.to(device), ignore_index).backward()
    grad_sq = {l: 0.0 for l in prefixes}
    shift_sq = {l: 0.0 for l in prefixes}
    for name, theta_l in model.named_parameters():
        l = group_of(name, prefixes)
        if theta_l.grad is not None:
            grad_sq[l] += float(theta_l.grad.pow(2).sum())
        theta_l_t = Theta_t[name].to(theta_l.device)[: theta_l.shape[0]]
        shift_sq[l] += float((theta_l_t - theta_l.detach()).pow(2).sum())
    model.zero_grad(set_to_none=True)
    score = {l: math.sqrt(grad_sq[l]) * math.sqrt(shift_sq[l]) for l in prefixes}
    total = sum(score.values())
    return {l: (value / total if total else 0.0) for l, value in score.items()}


def gradient_conflict(model, groups, batch_new, batch_old, ignore_index, device):
    def gradients(batch):
        model.zero_grad(set_to_none=True)
        x, y, _ = batch
        pixel_ce(model(x.to(device)), y.to(device), ignore_index).backward()
        return {l: torch.cat([param.grad.reshape(-1) for _, param in params]) for l, params in groups.items()}

    grad_n, grad_o = gradients(batch_new), gradients(batch_old)
    model.zero_grad(set_to_none=True)
    return {l: float((grad_n[l] - grad_o[l]).pow(2).sum()) for l in groups}


def classifier_conflict_lower_bound(L_c, C_t, C_o, phi_o_mean, phi_n_mean):
    return (L_c ** 2 / 4) * (C_t / C_o) ** 2 * float((phi_o_mean - phi_n_mean).pow(2).sum())


def deep_conflict_upper_bound(L_d, d_d, x_shift_sq):
    return L_d ** 2 / d_d * x_shift_sq


def heterogeneity(model, client_batches, ignore_index, device):
    gradients = []
    for batches in client_batches:
        model.zero_grad(set_to_none=True)
        for x, y, _ in batches:
            (pixel_ce(model(x.to(device)), y.to(device), ignore_index) / len(batches)).backward()
        gradients.append(torch.cat([param.grad.reshape(-1) for param in model.parameters()]))
    model.zero_grad(set_to_none=True)
    G = torch.stack(gradients)
    return float((G - G.mean(0)).pow(2).sum(1).mean())


def uniform_rehearsal_ratio(kappa_l, alpha_l, alpha):
    return 1 - (kappa_l["c"] - kappa_l["d"]) * (alpha_l["c"] - alpha_l["d"]) / (4 * alpha)


def aggregation_drift_bound(drift, eta, L, gamma_sq, num_selected):
    return (1 - eta / (2 * L)) * drift + 2 * eta ** 2 * gamma_sq / num_selected


def longterm_degradation_order(gamma_sq, T, R, C_o, M):
    return gamma_sq * T * R * (C_o / M) ** 0.5


def longterm_degradation_order_T(gamma_0_sq, T, R):
    return gamma_0_sq * T ** 1.5 * R


def recovery_sample_complexity(C_o, d_c, epsilon):
    return C_o * d_c / epsilon ** 2


def recovery_finetune_epochs(epsilon, mu):
    return math.ceil(math.log(epsilon) / math.log(1 - mu))


def retraining_epochs(C_o):
    return C_o
