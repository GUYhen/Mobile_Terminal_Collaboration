import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset

from .data import Empty, TaskView
from .utils import loader, resize_labels


def herding(features, m):
    phi = F.normalize(features, dim=1)
    mu = phi.mean(0)
    selected, running = [], torch.zeros_like(mu)
    available = torch.ones(phi.shape[0], dtype=torch.bool, device=phi.device)
    for k in range(1, min(m, phi.shape[0]) + 1):
        distance = (mu.unsqueeze(0) - (running.unsqueeze(0) + phi) / k).norm(dim=1)
        distance[~available] = float("inf")
        i = int(distance.argmin())
        selected.append(i)
        available[i] = False
        running += phi[i]
    return selected


class ExemplarMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.exemplars = {}

    def __len__(self):
        return sum(len(v) for v in self.exemplars.values())

    def reduce(self, m):
        for c in self.exemplars:
            self.exemplars[c] = self.exemplars[c][:m]

    @torch.no_grad()
    def add(self, model, base, hist, candidates, Y_t, t, m, batch_size, ignore_index, device):
        model.eval()
        candidates = np.asarray(candidates, dtype=np.int64)
        for c in Y_t:
            idx = candidates[hist[candidates, c] > 0] if len(candidates) else candidates
            if len(idx) == 0 or m == 0:
                self.exemplars[c] = []
                continue
            features = []
            for x, y, _ in loader(TaskView(base, idx, [c], ignore_index), batch_size, False):
                f = model.features(x.to(device))
                mask = (resize_labels(y.to(device), f.shape[-2:]) == c).unsqueeze(1).float()
                features.append((f * mask).sum((2, 3)) / mask.sum((2, 3)).clamp_min(1))
            self.exemplars[c] = [(int(idx[i]), t) for i in herding(torch.cat(features), m)]

    def dataset(self, base, stream, ignore_index):
        by_task = {}
        for exemplars in self.exemplars.values():
            for index, t in exemplars:
                by_task.setdefault(t, set()).add(index)
        if not by_task:
            return Empty()
        return ConcatDataset(
            [TaskView(base, sorted(ix), stream.Y[t], ignore_index, is_memory=True) for t, ix in sorted(by_task.items())]
        )
