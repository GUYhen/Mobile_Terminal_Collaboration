import torch
import torch.nn as nn

from .generators import mlp
from .knowledge import angular_features, head_rows, split_rows
from .model import head_logits
from .utils import cycle, loader, pixel_ce, restrict_labels


class RecoveryFunction(nn.Module):
    def __init__(self, d, d_a, hidden):
        super().__init__()
        self.E_s = mlp([d + 1, hidden, hidden], out_act=True)
        self.E_p = mlp([d, hidden, hidden], out_act=True)
        self.E_m = mlp([d, hidden, hidden], out_act=True)
        self.E_g = mlp([d_a, hidden, hidden], out_act=True)
        self.q = nn.Parameter(torch.randn(hidden) / hidden ** 0.5)
        self.F = mlp([hidden, hidden, d + 1])

    def forward(self, theta_c_star, p, M_k, a):
        E = torch.stack([self.E_s(theta_c_star), self.E_p(p), self.E_m(M_k), self.E_g(a)], dim=1)
        attention = torch.softmax(E @ self.q, dim=1)
        return self.F((attention.unsqueeze(-1) * E).sum(dim=1))


class EpisodicMetaLearning:
    def __init__(self, psi, cfg, d_a, batch_size, momentum, ignore_index, device, rng):
        self.psi = psi
        self.cfg = cfg
        self.d_a = d_a
        self.batch_size = batch_size
        self.momentum = momentum
        self.ignore_index = ignore_index
        self.device = device
        self.rng = rng

    def head_loss(self, model, batch, W, b, Y):
        x, y, _ = batch
        with torch.no_grad():
            f = model.features(x.to(self.device))
        y = restrict_labels(y.to(self.device), Y, self.ignore_index)
        return pixel_ce(head_logits(f, W, b), y, self.ignore_index)

    def degrade(self, model, batches, theta_W, theta_b, Y_minus):
        W = theta_W.clone().requires_grad_(True)
        b = theta_b.clone().requires_grad_(True)
        optimizer = torch.optim.SGD([W, b], lr=self.cfg.inner_lr, momentum=self.momentum)
        for _ in range(self.cfg.inner_steps):
            loss = self.head_loss(model, next(batches), W, b, Y_minus)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return W.detach(), b.detach()

    def episode(self, model, batches, theta_W, theta_b, Y_0, p, M_k):
        order = [Y_0[i] for i in self.rng.permutation(len(Y_0))]
        n_minus = int(self.cfg.minus_ratio * len(order))
        Y_minus = order[:n_minus]
        Y_plus = [c for c in Y_0 if c not in Y_minus]
        W_star, b_star = self.degrade(model, batches, theta_W, theta_b, Y_minus)
        Delta_theta_c = self.psi(head_rows(W_star, b_star), p, M_k, angular_features(p, self.d_a))
        dW, db = split_rows(Delta_theta_c, W_star.shape)
        return self.head_loss(model, next(batches), W_star + dW, b_star + db, Y_plus)

    def train(self, model, T_k_0, Y_0, p, M_k):
        model.eval()
        theta_W = model.classifier.weight.detach().flatten(1)
        theta_b = model.classifier.bias.detach()
        batches = cycle(loader(T_k_0, self.batch_size, True))
        optimizer = torch.optim.SGD(self.psi.parameters(), lr=self.cfg.meta_lr, momentum=self.momentum)
        for _ in range(self.cfg.episodes):
            loss = self.episode(model, batches, theta_W, theta_b, Y_0, p, M_k)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


@torch.no_grad()
def recover(model, psi, p, M_k, d_a):
    W, b = model.classifier.weight, model.classifier.bias
    theta_c_star = head_rows(W, b)
    Delta_theta_c = psi(theta_c_star, p, M_k, angular_features(p, d_a))
    dW, db = split_rows(Delta_theta_c, W.shape)
    W.add_(dW)
    b.add_(db)


def finetune(model, M_k, epochs, lr, momentum, batch_size, ignore_index, device):
    model.eval()
    optimizer = torch.optim.SGD(model.classifier.parameters(), lr=lr, momentum=momentum)
    for _ in range(epochs):
        for x, y, _ in loader(M_k, batch_size, True):
            with torch.no_grad():
                f = model.features(x.to(device))
            loss = pixel_ce(model.classifier(f), y.to(device), ignore_index)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
