import numpy as np
import torch

from .utils import clone_state


def fedavg(states):
    Theta = {}
    for key, value in states[0].items():
        if value.is_floating_point():
            Theta[key] = torch.stack([state[key] for state in states]).mean(0)
        else:
            Theta[key] = value.clone()
    return Theta


class Server:
    def __init__(self, Theta, clients_per_round, seed):
        self.Theta = clone_state(Theta)
        self.clients_per_round = clients_per_round
        self.rng = np.random.RandomState(seed)

    def select(self, K):
        if self.clients_per_round >= K:
            return list(range(K))
        return sorted(self.rng.choice(K, self.clients_per_round, replace=False).tolist())

    def aggregate(self, Thetas):
        self.Theta = fedavg(Thetas)
        return self.Theta
