import os

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_RESAMPLE = getattr(Image, "Resampling", Image)


class TaskStream:
    def __init__(self, class_names, task_split):
        self.class_names = list(class_names)
        self.Y = []
        start = 0
        for C_t in task_split:
            self.Y.append(list(range(start, start + C_t)))
            start += C_t

    @property
    def T(self):
        return len(self.Y)

    @property
    def num_classes(self):
        return len(self.class_names)

    def C(self, t):
        return len(self.Y[t])

    def C_o(self, t):
        return sum(len(self.Y[i]) for i in range(1, t))

    def learned(self, t):
        return [c for i in range(t + 1) for c in self.Y[i]]


def build_lut(raw_ids, class_names, ignore_index):
    lut = torch.full((256,), ignore_index, dtype=torch.long)
    for c, name in enumerate(class_names):
        lut[int(raw_ids[name])] = c
    return lut


def split_ids(root, max_images, test_fraction, seed):
    rng = np.random.RandomState(seed)
    ids = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, "images")))
    ids = [ids[i] for i in rng.permutation(len(ids))][:max_images]
    n_test = int(round(test_fraction * len(ids)))
    return ids[n_test:], ids[:n_test]


class MarsTerrain(Dataset):
    def __init__(self, root, ids, lut, img_size):
        self.root = root
        self.ids = list(ids)
        self.lut = lut
        self.img_size = img_size
        self.files = {os.path.splitext(f)[0]: f for f in os.listdir(os.path.join(root, "images"))}

    def __len__(self):
        return len(self.ids)

    def label(self, i):
        mask = Image.open(os.path.join(self.root, "labels", self.ids[i] + ".png"))
        mask = mask.resize((self.img_size, self.img_size), _RESAMPLE.NEAREST)
        raw = torch.from_numpy(np.array(mask, dtype=np.uint8)).long()
        if raw.ndim == 3:
            raw = raw[..., 0]
        return self.lut[raw]

    def __getitem__(self, i):
        image = Image.open(os.path.join(self.root, "images", self.files[self.ids[i]])).convert("RGB")
        image = image.resize((self.img_size, self.img_size), _RESAMPLE.BILINEAR)
        x = TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
        return x, self.label(i)

    def class_pixels(self, num_classes, ignore_index):
        hist = np.zeros((len(self), num_classes), dtype=np.int64)
        for i in range(len(self)):
            y = self.label(i)
            hist[i] = np.bincount(y[y != ignore_index].numpy(), minlength=num_classes)[:num_classes]
        return hist


class TaskView(Dataset):
    def __init__(self, base, indices, Y, ignore_index, is_memory=False):
        self.base = base
        self.indices = [int(i) for i in indices]
        self.keep = torch.zeros(256, dtype=torch.bool)
        self.keep[list(Y)] = True
        self.ignore_index = ignore_index
        self.is_memory = is_memory

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.base[self.indices[i]]
        y = torch.where(self.keep[y], y, torch.full_like(y, self.ignore_index))
        return x, y, torch.tensor(self.is_memory)


class Empty(Dataset):
    def __len__(self):
        return 0

    def __getitem__(self, i):
        raise IndexError(i)


def dirichlet_partition(hist, K, beta, seed):
    rng = np.random.RandomState(seed)
    labeled = hist.sum(1) > 0
    dominant = hist.argmax(1)
    parts = [[] for _ in range(K)]
    for c in range(hist.shape[1]):
        idx = np.where(labeled & (dominant == c))[0]
        rng.shuffle(idx)
        q = rng.dirichlet(np.full(K, beta))
        cuts = (np.cumsum(q) * len(idx)).astype(int)[:-1]
        for k, chunk in enumerate(np.split(idx, cuts)):
            parts[k].extend(chunk.tolist())
    return [sorted(part) for part in parts]
