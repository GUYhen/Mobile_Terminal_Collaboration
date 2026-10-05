from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision.transforms import ColorJitter, InterpolationMode

from ..config import FLConfig

TERRAIN = ("soil", "bedrock", "sand", "big_rock")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class AI4MARS(Dataset):
    def __init__(self, cfg: FLConfig, label_dir: str, train: bool):
        self.image_dir = os.path.join(cfg.root, cfg.image_dir)
        self.label_dir = os.path.join(cfg.root, label_dir)
        self.labels = sorted(f for f in os.listdir(self.label_dir) if f.lower().endswith(".png"))
        self.size = [cfg.image_size, cfg.image_size]
        self.train = train
        self.num_classes = cfg.num_classes
        self.ignore_index = cfg.ignore_index
        self.jitter = ColorJitter(brightness=cfg.color_jitter[0], contrast=cfg.color_jitter[1])

    def __len__(self) -> int:
        return len(self.labels)

    def image_path(self, label_file: str) -> str:
        stem = os.path.splitext(label_file)[0].replace("_merged", "")
        return os.path.join(self.image_dir, stem + ".JPG")

    def read_label(self, i: int) -> np.ndarray:
        return np.array(Image.open(os.path.join(self.label_dir, self.labels[i])), dtype=np.int64)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.image_path(self.labels[i])).convert("RGB")
        label = Image.open(os.path.join(self.label_dir, self.labels[i]))
        image = TF.resize(image, self.size, interpolation=InterpolationMode.BILINEAR)
        label = TF.resize(label, self.size, interpolation=InterpolationMode.NEAREST)
        if self.train:
            if torch.rand(1).item() < 0.5:
                image, label = TF.hflip(image), TF.hflip(label)
            if torch.rand(1).item() < 0.5:
                image, label = TF.vflip(image), TF.vflip(label)
            image = self.jitter(image)
        x = TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
        y = torch.from_numpy(np.array(label, dtype=np.int64))
        return x, y


def class_pixel_counts(dataset: AI4MARS) -> np.ndarray:
    counts = np.zeros((len(dataset), dataset.num_classes), dtype=np.int64)
    for i in range(len(dataset)):
        y = dataset.read_label(i)
        y = y[y != dataset.ignore_index]
        counts[i] = np.bincount(y, minlength=dataset.num_classes)[: dataset.num_classes]
    return counts


def stratified_sample(terrain: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    chosen = []
    for c in np.unique(terrain):
        idx = np.flatnonzero(terrain == c)
        chosen.append(rng.choice(idx, size=int(round(frac * len(idx))), replace=False))
    return np.sort(np.concatenate(chosen))


def dirichlet_partition(terrain: np.ndarray, N: int, phi: float, num_classes: int,
                        rng: np.random.Generator) -> List[np.ndarray]:
    parts: List[list] = [[] for _ in range(N)]
    for c in range(num_classes):
        idx = rng.permutation(np.flatnonzero(terrain == c))
        p = rng.dirichlet(phi * np.ones(N))
        cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
        for n, chunk in enumerate(np.split(idx, cuts)):
            parts[n].extend(chunk.tolist())
    return [np.array(sorted(p), dtype=np.int64) for p in parts]


def class_weights(pixel_counts: np.ndarray) -> torch.Tensor:
    composition = pixel_counts.sum(axis=0) / pixel_counts.sum()
    return torch.tensor(1.0 - composition, dtype=torch.float32)


def build_federated_dataset(cfg: FLConfig, N: int, rng: np.random.Generator):
    train_set = AI4MARS(cfg, cfg.train_label_dir, train=True)
    test_set = AI4MARS(cfg, cfg.test_label_dir, train=False)
    counts = class_pixel_counts(train_set)
    terrain = counts.argmax(axis=1)
    sampled = stratified_sample(terrain, cfg.sample_frac, rng)
    parts = dirichlet_partition(terrain[sampled], N, cfg.phi, cfg.num_classes, rng)
    clients = [Subset(train_set, sampled[p].tolist()) for p in parts]
    zeta = np.array([len(c) for c in clients], dtype=np.float64)
    return clients, zeta, test_set, class_weights(counts[sampled])
