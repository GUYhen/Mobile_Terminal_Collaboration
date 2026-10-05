from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


DATASETS = {
    "fashionmnist": (datasets.FashionMNIST, 1, 10, 28, (0.2860,), (0.3530,)),
    "cifar100": (datasets.CIFAR100, 3, 100, 32, (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}


def load_dataset(name, root, train=True):
    cls, _, _, _, mean, std = DATASETS[name]
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return cls(root, train=train, download=False, transform=tf)


def dataset_info(name):
    _, in_channels, num_classes, img_size, _, _ = DATASETS[name]
    return in_channels, num_classes, img_size


def labels_of(dataset):
    return np.asarray(dataset.targets, dtype=np.int64)


def dirichlet_partition(labels, N, alpha, rng):
    D = [[] for _ in range(N)]
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        p = rng.dirichlet(alpha * np.ones(N))
        cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
        for n, part in enumerate(np.split(idx, cuts)):
            D[n].extend(part.tolist())
    return [np.array(sorted(D_n), dtype=np.int64) for D_n in D]


@dataclass
class UnlearningRequest:
    C_u: List[int]
    D_u: Dict[int, np.ndarray]
    tau: str

    def forget_indices(self):
        return np.concatenate([self.D_u[n] for n in self.C_u])


def make_request(tau, n, D, labels, Y_u, sample_fraction, rng):
    D_n = D[n]
    if tau == "client":
        D_un = D_n
    elif tau == "class":
        D_un = D_n[np.isin(labels[D_n], list(Y_u))]
    elif tau == "sample":
        D_un = np.sort(rng.choice(D_n, size=max(1, int(round(sample_fraction * len(D_n)))), replace=False))
    else:
        raise ValueError(tau)
    return UnlearningRequest(C_u=[n], D_u={n: D_un}, tau=tau)


def remaining_data(D, U):
    return [np.setdiff1d(D_n, U.D_u[n], assume_unique=True) if n in U.D_u else D_n for n, D_n in enumerate(D)]


def make_loader(dataset, indices, batch_size, shuffle=False):
    return DataLoader(Subset(dataset, np.asarray(indices).tolist()), batch_size=batch_size, shuffle=shuffle)
