from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .har import Windows


@dataclass
class ClientSplit:
    client_id: int
    subject: int
    device_type: str
    modalities: List[str]
    train: Windows


def assign_device_types(subjects: Sequence[int], counts: Mapping[str, int], force: Optional[str] = None) -> Dict[int, str]:
    order = [t for t, c in counts.items() for _ in range(c)]
    if force is None and len(order) < len(subjects):
        raise ValueError("device_type_counts do not cover all subjects")
    return {s: force if force is not None else order[i] for i, s in enumerate(sorted(subjects))}


def dirichlet_shards(y: np.ndarray, n_shards: int, beta: float, rng: np.random.Generator) -> List[np.ndarray]:
    if n_shards <= 1:
        return [np.arange(len(y))]
    shards: List[List[int]] = [[] for _ in range(n_shards)]
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        p = rng.dirichlet(np.full(n_shards, beta))
        cuts = (np.cumsum(p)[:-1] * len(idx)).astype(int)
        for s, part in enumerate(np.split(idx, cuts)):
            shards[s].extend(part.tolist())
    return [np.sort(np.asarray(s, dtype=np.int64)) for s in shards]


def shard_counts(n_subjects: int, N: int) -> List[int]:
    base, extra = divmod(N, n_subjects)
    return [base + (1 if i < extra else 0) for i in range(n_subjects)]


def take(w: Windows, idx: np.ndarray) -> Windows:
    X, y = w
    return {m: x[idx] for m, x in X.items()}, y[idx]


def concat(parts: Sequence[Windows], modalities: Sequence[str]) -> Windows:
    return {m: np.concatenate([p[0][m] for p in parts]) for m in modalities}, np.concatenate([p[1] for p in parts])


def build_federated_splits(
    subject_windows: Dict[int, Windows],
    spec,
    data_cfg,
    type_modalities: Mapping[str, List[str]],
    rng: np.random.Generator,
) -> Tuple[List[ClientSplit], Windows, Optional[Windows]]:
    val_ids = [s for s in sorted(subject_windows) if s in set(data_cfg.val_subjects)]
    subjects = [s for s in sorted(subject_windows) if s not in set(data_cfg.val_subjects)]
    N = data_cfg.num_clients or len(subjects)
    if N < len(subjects):
        raise ValueError("num_clients must be at least the number of subjects")
    types = assign_device_types(subjects, spec.device_type_counts, data_cfg.force_device_type)
    clients: List[ClientSplit] = []
    test_parts: List[Windows] = []
    for s, n_shards in zip(subjects, shard_counts(len(subjects), N)):
        mods = type_modalities[types[s]]
        for shard in dirichlet_shards(subject_windows[s][1], n_shards, data_cfg.dirichlet_beta, rng):
            if len(shard) < 2:
                continue
            perm = rng.permutation(shard)
            n_test = min(int(round(len(perm) * data_cfg.test_ratio)), len(perm) - 1)
            X_tr, y_tr = take(subject_windows[s], np.sort(perm[n_test:]))
            clients.append(ClientSplit(len(clients), s, types[s], list(mods), ({m: X_tr[m] for m in mods}, y_tr)))
            test_parts.append(take(subject_windows[s], np.sort(perm[:n_test])))
    test = concat(test_parts, spec.modalities)
    val = concat([subject_windows[s] for s in val_ids], spec.modalities) if val_ids else None
    return clients, test, val
