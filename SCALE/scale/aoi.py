import numpy as np
import torch


class ParameterGroups:
    def __init__(self, shapes, G_l):
        self.G_l = G_l
        self.layers = list(shapes.keys())
        self.shapes = dict(shapes)
        self.bounds = {}
        for l, shape in self.shapes.items():
            d_l = int(np.prod(shape))
            edges = np.linspace(0, d_l, G_l + 1).round().astype(int)
            self.bounds[l] = [(int(edges[j]), int(edges[j + 1])) for j in range(G_l)]

    def group(self, W_l, l, j):
        a, b = self.bounds[l][j]
        return W_l[a:b]

    def size(self, l, j):
        a, b = self.bounds[l][j]
        return b - a


class AgeOfInformation:
    def __init__(self, groups):
        self.groups = groups
        self.T_lj = {l: np.zeros(groups.G_l) for l in groups.layers}

    def copy(self):
        other = AgeOfInformation(self.groups)
        other.T_lj = {l: v.copy() for l, v in self.T_lj.items()}
        return other

    def A(self, t, layers=None):
        layers = self.groups.layers if layers is None else layers
        return {l: t - self.T_lj[l] for l in layers}

    def update(self, l, G, t):
        for j in G:
            self.T_lj[l][j] = t

    @torch.no_grad()
    def record(self, Theta_prev, Theta_next, t):
        for l in self.groups.layers:
            W_prev = Theta_prev[l].reshape(-1)
            W_next = Theta_next[l].reshape(-1)
            for j in range(self.groups.G_l):
                if not torch.equal(self.groups.group(W_prev, l, j), self.groups.group(W_next, l, j)):
                    self.T_lj[l][j] = t

    def rebase(self, t0):
        for l in self.T_lj:
            self.T_lj[l] = self.T_lj[l] - t0


def sum_aoi(A):
    return float(sum(A_l.sum() for A_l in A.values()))


def avg_aoi(A):
    return float(np.concatenate(list(A.values())).mean())


def global_system_aoi(A, M):
    return sum_aoi(A) / (M * sum(len(A_l) for A_l in A.values()))


def communication_overhead(C_t, E_A_g, alpha, beta):
    return alpha * C_t + beta * E_A_g
