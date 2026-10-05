from __future__ import annotations

import os
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

RawSubject = Tuple[Dict[str, np.ndarray], np.ndarray]
Windows = Tuple[Dict[str, np.ndarray], np.ndarray]

PAMAP2_SUBJECTS = tuple(range(101, 110))
PAMAP2_ACTIVITIES = (1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24)
PAMAP2_IMU_OFFSETS = (3, 20, 37)
PAMAP2_COLUMNS = {
    "acc": [o + 1 + i for o in PAMAP2_IMU_OFFSETS for i in range(3)],
    "gyro": [o + 7 + i for o in PAMAP2_IMU_OFFSETS for i in range(3)],
    "mag": [o + 10 + i for o in PAMAP2_IMU_OFFSETS for i in range(3)],
    "hr": [2],
}
PAMAP2_LABEL_COL = 1

MHEALTH_SUBJECTS = tuple(range(1, 11))
MHEALTH_ACTIVITIES = tuple(range(1, 13))
MHEALTH_COLUMNS = {
    "acc": [0, 1, 2, 5, 6, 7, 14, 15, 16],
    "gyro": [8, 9, 10, 17, 18, 19],
    "mag": [11, 12, 13, 20, 21, 22],
    "ecg": [3, 4],
}
MHEALTH_LABEL_COL = 23


def fill_nan(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=True)
    idx = np.arange(x.shape[0])
    for c in range(x.shape[1]):
        col = x[:, c]
        valid = ~np.isnan(col)
        if not valid.any():
            x[:, c] = 0.0
            continue
        last = np.maximum.accumulate(np.where(valid, idx, 0))
        first = int(np.argmax(valid))
        last[:first] = first
        x[:, c] = col[last]
    return x


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(0, keepdims=True)) / (x.std(0, keepdims=True) + 1e-6)


def load_pamap2(root: str, modalities: Sequence[str], excluded: Iterable[int], raw_hz: int, target_hz: int) -> Dict[int, RawSubject]:
    step = max(1, raw_hz // target_hz)
    out = {}
    for sid in PAMAP2_SUBJECTS:
        if sid in set(excluded):
            continue
        arr = np.loadtxt(os.path.join(root, f"subject{sid}.dat"), dtype=np.float32)[::step]
        labels = arr[:, PAMAP2_LABEL_COL].astype(np.int64)
        out[sid] = ({m: zscore(fill_nan(arr[:, PAMAP2_COLUMNS[m]])) for m in modalities}, labels)
    return out


def load_mhealth(root: str, modalities: Sequence[str], excluded: Iterable[int], raw_hz: int, target_hz: int) -> Dict[int, RawSubject]:
    step = max(1, raw_hz // target_hz)
    out = {}
    for sid in MHEALTH_SUBJECTS:
        if sid in set(excluded):
            continue
        arr = np.loadtxt(os.path.join(root, f"mHealth_subject{sid}.log"), dtype=np.float32)[::step]
        labels = arr[:, MHEALTH_LABEL_COL].astype(np.int64)
        out[sid] = ({m: zscore(fill_nan(arr[:, MHEALTH_COLUMNS[m]])) for m in modalities}, labels)
    return out


def segment(signals: Dict[str, np.ndarray], labels: np.ndarray, activities: Sequence[int], window: int, stride: int) -> Windows:
    label_map = {a: i for i, a in enumerate(activities)}
    T = labels.shape[0]
    if T < window:
        return {m: np.zeros((0, s.shape[1], window), np.float32) for m, s in signals.items()}, np.zeros(0, np.int64)
    starts = np.arange(0, T - window + 1, stride)
    idx = starts[:, None] + np.arange(window)[None, :]
    keep, y = [], []
    for i, row in enumerate(labels[idx]):
        values, counts = np.unique(row, return_counts=True)
        majority = int(values[np.argmax(counts)])
        if majority in label_map:
            keep.append(i)
            y.append(label_map[majority])
    keep = np.asarray(keep, dtype=np.int64)
    X = {m: np.ascontiguousarray(s[idx[keep]].transpose(0, 2, 1)) for m, s in signals.items()}
    return X, np.asarray(y, dtype=np.int64)


def load_subject_windows(data_cfg, spec) -> Dict[int, Windows]:
    name = data_cfg.dataset.lower()
    if name == "pamap2":
        raw = load_pamap2(data_cfg.root, spec.modalities, spec.excluded_subjects, spec.raw_sample_rate_hz, data_cfg.sample_rate_hz)
        activities = PAMAP2_ACTIVITIES
    elif name == "mhealth":
        raw = load_mhealth(data_cfg.root, spec.modalities, spec.excluded_subjects, spec.raw_sample_rate_hz, data_cfg.sample_rate_hz)
        activities = MHEALTH_ACTIVITIES
    else:
        raise ValueError(f"unknown dataset {data_cfg.dataset}")
    out = {}
    for sid, (signals, labels) in raw.items():
        X, y = segment(signals, labels, activities, data_cfg.window_size, data_cfg.stride)
        if len(y):
            out[sid] = (X, y)
    return out


class ModalityWindowDataset(Dataset):
    def __init__(self, X: Dict[str, np.ndarray], y: np.ndarray, modalities: Sequence[str], channels: Dict[str, int], window: int, available: Sequence[str]):
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.modalities = list(modalities)
        self.x = [torch.as_tensor(X[m], dtype=torch.float32) if m in set(available) else None for m in self.modalities]
        self.zeros = [torch.zeros(channels[m], window) for m in self.modalities]

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, i: int):
        x = [x_m[i] if x_m is not None else z_m for x_m, z_m in zip(self.x, self.zeros)]
        return x, self.y[i]
