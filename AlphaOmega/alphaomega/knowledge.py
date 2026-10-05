import torch
import torch.nn.functional as F

from .utils import loader, resize_labels


def class_feature_sums(f, y, C, ignore_index):
    d = f.shape[1]
    y = resize_labels(y, f.shape[-2:]).reshape(-1)
    f = f.permute(0, 2, 3, 1).reshape(-1, d)
    valid = (y != ignore_index) & (y < C)
    sums = f.new_zeros(C, d).index_add_(0, y[valid], f[valid])
    counts = torch.bincount(y[valid], minlength=C)[:C].to(f.dtype)
    return sums, counts


@torch.no_grad()
def class_feature_means(model, dataset, batch_size, ignore_index, device):
    model.eval()
    C, d = model.num_classes, model.d
    sums = torch.zeros(C, d, device=device)
    counts = torch.zeros(C, device=device)
    for x, y, _ in loader(dataset, batch_size, False):
        s, n = class_feature_sums(model.features(x.to(device)), y.to(device), C, ignore_index)
        sums += s
        counts += n
    means = sums / counts.clamp_min(1).unsqueeze(1)
    h_m = sums.sum(0) / counts.sum().clamp_min(1)
    return means, counts, h_m


def prototype_matrix(prototypes, C, d, device):
    p = torch.zeros(C, d, device=device)
    for c, p_c in prototypes.items():
        if c < C:
            p[c] = p_c.to(device)
    return p


def angular_features(p, d_a):
    p_hat = F.normalize(p, dim=1)
    a = p.new_zeros(p.shape[0], d_a)
    a[:, : p.shape[0]] = p_hat @ p_hat.t()
    return a


def head_rows(W, b):
    return torch.cat([W.reshape(W.shape[0], -1), b.unsqueeze(1)], dim=1)


def split_rows(theta_c, W_shape):
    return theta_c[:, :-1].reshape(W_shape), theta_c[:, -1]
