import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def loader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle and len(dataset) > batch_size)


def cycle(iterable):
    while True:
        for item in iterable:
            yield item


def clone_state(state):
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def resize_labels(y, size):
    if tuple(y.shape[-2:]) == tuple(size):
        return y
    return F.interpolate(y[:, None].float(), size=tuple(size), mode="nearest")[:, 0].long()


def restrict_labels(y, Y, ignore_index):
    keep = torch.as_tensor(list(Y), dtype=y.dtype, device=y.device)
    return torch.where(torch.isin(y, keep), y, torch.full_like(y, ignore_index))


def pixel_ce(logits, y, ignore_index):
    if logits.shape[-2:] != y.shape[-2:]:
        logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
    if not bool((y != ignore_index).any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits, y, ignore_index=ignore_index)
