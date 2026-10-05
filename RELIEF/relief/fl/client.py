from __future__ import annotations

from typing import Dict, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..algorithm.groups import GroupRegistry
from ..data.har import ModalityWindowDataset
from ..data.partition import ClientSplit


class Client:
    def __init__(
        self,
        split: ClientSplit,
        modalities: Sequence[str],
        channels: Mapping[str, int],
        window: int,
        num_classes: int,
        batch_size: int,
        class_reweighting: bool = False,
    ):
        self.n = split.client_id
        self.subject = split.subject
        self.device_type = split.device_type
        index = {m: i for i, m in enumerate(modalities)}
        self.M_n = frozenset(index[m] for m in split.modalities)
        X, y = split.train
        self.dataset = ModalityWindowDataset(X, y, modalities, dict(channels), window, split.modalities)
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, drop_last=len(self.dataset) > batch_size)
        self.class_weight = None
        if class_reweighting:
            counts = torch.bincount(self.dataset.y, minlength=num_classes).float()
            self.class_weight = torch.where(counts > 0, counts.sum() / (num_classes * counts.clamp(min=1.0)), torch.zeros_like(counts))

    @property
    def num_samples(self) -> int:
        return len(self.dataset)

    def local_train(
        self,
        model: nn.Module,
        registry: GroupRegistry,
        theta: Mapping[str, torch.Tensor],
        S_n: Sequence[int],
        M_n: frozenset,
        E: int,
        lr: float,
        device: torch.device,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        if not S_n:
            return {}
        registry.load_state(model, theta)
        optimizer = torch.optim.Adam(registry.set_trainable(model, S_n), lr=lr)
        weight = self.class_weight.to(device) if self.class_weight is not None else None
        model.train()
        for _ in range(E):
            for x, y in self.loader:
                x = [x_m.to(device) if m in M_n else torch.zeros_like(x_m, device=device) for m, x_m in enumerate(x)]
                loss = F.cross_entropy(model(x), y.to(device), weight=weight)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        local = registry.state(model, S_n)
        return {j: {name: local[name] - theta[name] for name in registry[j].tensor_names} for j in S_n}
