import numpy as np
import torch
from torch.utils.data import ConcatDataset

from .analysis import memory_coverage
from .data import Empty, TaskView
from .knowledge import class_feature_means, prototype_matrix
from .lsr import LayerSelectiveRehearsal
from .memory import ExemplarMemory
from .metrics import evaluate
from .model import layer_groups
from .rkr import EpisodicMetaLearning, RecoveryFunction, finetune, recover
from .utils import clone_state, loader, pixel_ce


def selective_rehearsal_loss(model, x, y, is_memory, lam, ignore_index):
    logits = model(x)
    L_s = logits.sum() * 0.0
    if bool((~is_memory).any()):
        L_s = L_s + pixel_ce(logits[~is_memory], y[~is_memory], ignore_index)
    if bool(is_memory.any()):
        L_s = L_s + lam * pixel_ce(logits[is_memory], y[is_memory], ignore_index)
    return L_s


class Rover:
    def __init__(self, k, cfg, stream, base, hist, indices, capacity, d, device):
        self.k = k
        self.cfg = cfg
        self.stream = stream
        self.base = base
        self.hist = hist
        self.indices = np.asarray(indices, dtype=np.int64)
        self.d = d
        self.d_a = stream.num_classes
        self.device = device
        self.ignore_index = cfg.data.ignore_index
        self.batch_size = cfg.train.batch_size
        self.M_k = ExemplarMemory(capacity)
        self.p = {}
        self.lsr = LayerSelectiveRehearsal(cfg.lsr, d, self.d_a)
        self.psi = None
        self.Theta_k = None
        self.rng = np.random.RandomState(cfg.seed + k)

    def T_k(self, t):
        Y_t = self.stream.Y[t]
        if len(self.indices):
            idx = self.indices[self.hist[self.indices][:, Y_t].sum(1) > 0]
        else:
            idx = self.indices
        return TaskView(self.base, idx.tolist(), Y_t, self.ignore_index)

    def memory(self):
        return self.M_k.dataset(self.base, self.stream, self.ignore_index)

    def memory_features(self, model):
        M_feat, _, h_m = class_feature_means(model, self.memory(), self.batch_size, self.ignore_index, self.device)
        return M_feat, h_m

    def prototypes(self, model):
        return prototype_matrix(self.p, model.num_classes, self.d, self.device)

    def local_train(self, model, Theta, t):
        train = self.cfg.train
        model.load_state_dict(Theta)
        data = ConcatDataset([self.T_k(t), self.memory() if t > 0 else Empty()])
        if len(data) == 0:
            return clone_state(Theta)
        use_lsr = t > 0 and self.cfg.method == "lifecycle_aware"
        if use_lsr:
            groups = layer_groups(model, self.cfg.model.layer_groups)
            self.lsr.setup(self.device)
            p = self.prototypes(model)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=train.lr, momentum=train.momentum)
        for _ in range(train.local_epochs):
            if use_lsr:
                M_feat, h_m = self.memory_features(model)
            model.train()
            for x, y, is_memory in loader(data, self.batch_size, True):
                x, y, is_memory = x.to(self.device), y.to(self.device), is_memory.to(self.device)
                L_s = selective_rehearsal_loss(model, x, y, is_memory, train.lam, self.ignore_index)
                model.zero_grad(set_to_none=True)
                L_s.backward()
                if not use_lsr:
                    optimizer.step()
                    continue
                Delta_theta = self.lsr.corrections(groups, h_m, p, M_feat)
                self.lsr.update_layers(groups, Delta_theta, train.lr)
                model.zero_grad(set_to_none=True)
                selective_rehearsal_loss(model, x, y, is_memory, train.lam, self.ignore_index).backward()
                self.lsr.update_generators(groups, Delta_theta)
        return clone_state(model.state_dict())

    @torch.no_grad()
    def end_task(self, model, Theta, t):
        model.load_state_dict(Theta)
        T_k_t = self.T_k(t)
        means, counts, _ = class_feature_means(model, T_k_t, self.batch_size, self.ignore_index, self.device)
        for c in self.stream.Y[t]:
            if counts[c] > 0:
                self.p[c] = means[c].cpu()
        m = int(memory_coverage(self.M_k.capacity, self.stream, t + 1))
        self.M_k.reduce(m)
        self.M_k.add(model, self.base, self.hist, T_k_t.indices, self.stream.Y[t], t, m, self.batch_size, self.ignore_index, self.device)

    def meta_train(self, model, Theta, xi):
        model.load_state_dict(Theta)
        T_k_0 = self.T_k(0)
        if len(T_k_0) == 0:
            return None
        rkr = self.cfg.rkr
        psi = RecoveryFunction(self.d, self.d_a, rkr.hidden).to(self.device)
        psi.load_state_dict(xi)
        M_feat, _ = self.memory_features(model)
        meta = EpisodicMetaLearning(psi, rkr, self.d_a, self.batch_size, self.cfg.train.momentum, self.ignore_index, self.device, self.rng)
        meta.train(model, T_k_0, self.stream.Y[0], self.prototypes(model), M_feat)
        return clone_state(psi.state_dict())

    def receive_psi(self, xi):
        self.psi = RecoveryFunction(self.d, self.d_a, self.cfg.rkr.hidden).to(self.device)
        self.psi.load_state_dict(xi)
        self.psi.requires_grad_(False)

    def rapid_knowledge_recovery(self, model, Theta, t, force):
        rkr = self.cfg.rkr
        model.load_state_dict(Theta)
        M_k = self.memory()
        learned = self.stream.learned(t)
        I_k_c = evaluate(model, ConcatDataset([self.T_k(t), M_k]), learned, self.batch_size, self.ignore_index, self.device)["mIoU"]
        triggered = force or I_k_c < rkr.tau
        if triggered:
            M_feat, _ = self.memory_features(model)
            recover(model, self.psi, self.prototypes(model), M_feat, self.d_a)
            finetune(model, M_k, rkr.finetune_epochs, self.cfg.train.lr, self.cfg.train.momentum, self.batch_size, self.ignore_index, self.device)
        self.Theta_k = clone_state(model.state_dict())
        return {"rover": self.k, "I_k_c": I_k_c, "triggered": bool(triggered)}
