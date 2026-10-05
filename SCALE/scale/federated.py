import numpy as np
import torch
import torch.nn.functional as F

from .aoi import AgeOfInformation, ParameterGroups
from .data import make_loader


def per_sample_loss(model, x, y):
    return F.cross_entropy(model(x), y, reduction="none")


@torch.no_grad()
def local_loss(model, Theta, client):
    model.load_state_dict(Theta)
    model.eval()
    total = 0.0
    for x, y in client.loader:
        total += per_sample_loss(model, x.to(client.device), y.to(client.device)).sum().item()
    return total / client.M_n


def global_loss(model, Theta, clients):
    M = float(sum(c.M_n for c in clients))
    return sum(c.M_n / M * local_loss(model, Theta, c) for c in clients)


class Client:
    def __init__(self, n, dataset, D_n, batch_size, device):
        self.n = n
        self.D_n = D_n
        self.M_n = len(D_n)
        self.loader = make_loader(dataset, D_n, batch_size)
        self.device = device

    def local_update(self, model, Theta_t, E, eta):
        model.load_state_dict(Theta_t)
        model.train()
        for e in range(E):
            model.zero_grad()
            for x, y in self.loader:
                F_n = per_sample_loss(model, x.to(self.device), y.to(self.device)).sum() / self.M_n
                F_n.backward()
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.sub_(eta * p.grad)
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def fed_avg(Thetas, M_list):
    total = float(sum(M_list))
    Theta = {}
    for k, v in Thetas[0].items():
        if torch.is_floating_point(v):
            Theta[k] = sum(Theta_n[k] * (M_n / total) for Theta_n, M_n in zip(Thetas, M_list))
        else:
            Theta[k] = v.clone()
    return Theta


class EdgeServer:
    def __init__(self, model, clients, layers, cfg_fl, G_l, rng):
        self.model = model
        self.clients = {c.n: c for c in clients}
        self.layers = layers
        self.cfg = cfg_fl
        self.rng = rng
        self.Theta = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        self.W_clients = {}
        self.groups = ParameterGroups({l: tuple(self.Theta[l].shape) for l in layers}, G_l)
        self.aoi = AgeOfInformation(self.groups)
        self.t = 0

    @property
    def M_sizes(self):
        return {n: c.M_n for n, c in self.clients.items()}

    def select_clients(self):
        ids = np.array(sorted(self.clients))
        size = min(self.cfg.clients_per_round, len(ids))
        return sorted(self.rng.choice(ids, size=size, replace=False).tolist())

    def round(self):
        C_t = self.select_clients()
        Thetas = []
        for n in C_t:
            Theta_nE = self.clients[n].local_update(self.model, self.Theta, self.cfg.E, self.cfg.eta)
            self.W_clients[n] = Theta_nE
            Thetas.append(Theta_nE)
        Theta_next = fed_avg(Thetas, [self.clients[n].M_n for n in C_t])
        self.t += 1
        self.aoi.record(self.Theta, Theta_next, self.t)
        self.Theta = Theta_next
        return C_t

    def train(self):
        for _ in range(self.cfg.T):
            self.round()
        return self.Theta


def retrain(model_fn, dataset, D_r, layers, cfg, device, rng):
    clients = [Client(n, dataset, D_rn, cfg.fl.batch_size, device) for n, D_rn in enumerate(D_r) if len(D_rn) > 0]
    server = EdgeServer(model_fn().to(device), clients, layers, cfg.fl, cfg.scale.G_l, rng)
    return server.train(), clients
