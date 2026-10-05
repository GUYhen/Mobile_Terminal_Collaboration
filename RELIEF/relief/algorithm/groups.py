from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn

KINDS = ("A", "B", "enc", "head")
SCOPES = ("accessible", "full_model")


@dataclass(frozen=True)
class ParamGroup:
    gid: int
    name: str
    kind: str
    modality: Optional[int]
    param_names: Tuple[str, ...]
    buffer_names: Tuple[str, ...] = ()
    module_prefix: Optional[str] = None

    @property
    def tensor_names(self) -> Tuple[str, ...]:
        return self.param_names + self.buffer_names


def named_tensors(model: nn.Module) -> Dict[str, torch.Tensor]:
    out = dict(model.named_parameters())
    out.update(dict(model.named_buffers()))
    return out


class GroupRegistry:
    def __init__(self, groups: Iterable[ParamGroup], model: nn.Module, M: int):
        self.groups = list(groups)
        for i, g in enumerate(self.groups):
            if g.gid != i or g.kind not in KINDS:
                raise ValueError(f"invalid parameter group {g.name}")
        self.M = M
        tensors = named_tensors(model)
        self._numel = [sum(tensors[n].numel() for n in g.tensor_names) for g in self.groups]

    def __len__(self) -> int:
        return len(self.groups)

    def __iter__(self) -> Iterator[ParamGroup]:
        return iter(self.groups)

    def __getitem__(self, gid: int) -> ParamGroup:
        return self.groups[gid]

    @property
    def G(self) -> int:
        return len(self.groups)

    def numel(self, gid: int) -> int:
        return self._numel[gid]

    def payload_bytes(self, gids: Iterable[int], bytes_per_param: int) -> int:
        return bytes_per_param * sum(self._numel[j] for j in gids)

    def L_m(self) -> List[int]:
        return [sum(1 for g in self.groups if g.kind == "enc" and g.modality == m) for m in range(self.M)]

    def L_H(self) -> int:
        return sum(1 for g in self.groups if g.kind == "head")

    def round_robin_period(self) -> int:
        return self.M + max(self.L_m(), default=0) + self.L_H() + 1

    def accessible(self, M_n: Iterable[int], scope: str = "accessible", include_B: bool = True) -> List[int]:
        if scope not in SCOPES:
            raise ValueError(f"unknown train scope {scope}")
        mods = set(M_n)
        G_n = []
        for g in self.groups:
            if g.kind == "B" and not include_B:
                continue
            if scope == "accessible" and g.modality is not None and g.modality not in mods:
                continue
            G_n.append(g.gid)
        return G_n

    def mandatory(self, M_n: Iterable[int]) -> List[int]:
        mods = set(M_n)
        return [g.gid for g in self.groups if g.kind == "A" and g.modality in mods]

    def state(self, model: nn.Module, gids: Optional[Iterable[int]] = None) -> Dict[str, torch.Tensor]:
        tensors = named_tensors(model)
        selected = self.groups if gids is None else [self.groups[j] for j in gids]
        return {n: tensors[n].detach().clone() for g in selected for n in g.tensor_names}

    @torch.no_grad()
    def load_state(self, model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
        tensors = named_tensors(model)
        for n, value in state.items():
            tensors[n].copy_(value)

    def set_trainable(self, model: nn.Module, gids: Iterable[int]) -> List[nn.Parameter]:
        params = dict(model.named_parameters())
        selected = set(gids)
        trainable = []
        for g in self.groups:
            flag = g.gid in selected
            for n in g.param_names:
                params[n].requires_grad_(flag)
                if flag:
                    trainable.append(params[n])
        return trainable
