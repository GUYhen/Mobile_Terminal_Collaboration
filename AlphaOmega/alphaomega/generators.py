import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(dims, out_act=False):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or out_act:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class LayerGenerator(nn.Module):
    def __init__(self, block, d_m, hidden):
        super().__init__()
        self.block = block
        self.mlp = mlp([2 * block + d_m, *hidden, block])

    def forward(self, theta_l, grad_l, h_m):
        n = theta_l.numel()
        pad = (-n) % self.block
        theta_rows = F.pad(theta_l, (0, pad)).view(-1, self.block)
        grad_rows = F.pad(grad_l, (0, pad)).view(-1, self.block)
        h_rows = h_m.view(1, -1).expand(theta_rows.shape[0], -1)
        return self.mlp(torch.cat([theta_rows, grad_rows, h_rows], dim=1)).reshape(-1)[:n]


class HeadGenerator(nn.Module):
    def __init__(self, d, d_a, hidden):
        super().__init__()
        self.E_theta = mlp([d + 1, hidden, hidden], out_act=True)
        self.E_grad = mlp([d + 1, hidden, hidden], out_act=True)
        self.E_p = mlp([d, hidden, hidden], out_act=True)
        self.E_M = mlp([d, hidden, hidden], out_act=True)
        self.E_a = mlp([d_a, hidden, hidden], out_act=True)
        self.fusion = nn.Linear(5 * hidden, d + 1)

    def forward(self, theta_c, grad_c, p, M_k, a):
        z = torch.cat([self.E_theta(theta_c), self.E_grad(grad_c), self.E_p(p), self.E_M(M_k), self.E_a(a)], dim=1)
        return self.fusion(z)
