import torch
import torch.nn as nn

from .generators import HeadGenerator, LayerGenerator
from .knowledge import angular_features, head_rows, split_rows


def flatten(params):
    return torch.cat([param.detach().reshape(-1) for param in params])


def flatten_grad(params):
    return torch.cat([param.grad.detach().reshape(-1) for param in params])


class LayerSelectiveRehearsal(nn.Module):
    def __init__(self, cfg, d, d_a):
        super().__init__()
        self.alpha = {l: float(cfg.alpha[l]) for l in ("s", "d", "c")}
        self.phi_s = LayerGenerator(cfg.phi_s.block, d, cfg.phi_s.hidden)
        self.phi_d = LayerGenerator(cfg.phi_d.block, d, cfg.phi_d.hidden)
        self.phi_c = HeadGenerator(d, d_a, cfg.phi_c.hidden)
        self.d_a = d_a
        self.lr = cfg.lr
        self.momentum = cfg.momentum
        self.optimizer = None

    def setup(self, device):
        self.to(device)
        if self.optimizer is None:
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)

    def corrections(self, groups, h_m, p, M_k):
        theta = {l: [param for _, param in groups[l]] for l in ("s", "d", "c")}
        Delta_theta_s = self.phi_s(flatten(theta["s"]), flatten_grad(theta["s"]), h_m)
        Delta_theta_d = self.phi_d(flatten(theta["d"]), flatten_grad(theta["d"]), h_m)
        W, b = theta["c"]
        theta_c = head_rows(W.detach(), b.detach())
        grad_c = head_rows(W.grad.detach(), b.grad.detach())
        Delta_rows = self.phi_c(theta_c, grad_c, p, M_k, angular_features(p, self.d_a))
        dW, db = split_rows(Delta_rows, W.shape)
        Delta_theta_c = torch.cat([dW.reshape(-1), db])
        return {"s": Delta_theta_s, "d": Delta_theta_d, "c": Delta_theta_c}

    @torch.no_grad()
    def update_layers(self, groups, Delta_theta, eta):
        for l, Delta in Delta_theta.items():
            offset = 0
            for _, theta_l in groups[l]:
                n = theta_l.numel()
                theta_l.add_(-eta * theta_l.grad + self.alpha[l] * Delta[offset: offset + n].reshape(theta_l.shape))
                offset += n

    def update_generators(self, groups, Delta_theta):
        self.optimizer.zero_grad(set_to_none=True)
        outputs, upstream = [], []
        for l, Delta in Delta_theta.items():
            outputs.append(Delta)
            upstream.append(self.alpha[l] * flatten_grad([param for _, param in groups[l]]))
        torch.autograd.backward(outputs, upstream)
        self.optimizer.step()
