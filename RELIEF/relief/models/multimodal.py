from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..algorithm.groups import ParamGroup
from .encoders import build_encoder
from .fusion import MDLoRAFusion
from .lora import LoRALinear


class TaskHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_classes: int, L_H: int, dropout: float, lora: bool, rho: int):
        super().__init__()
        dims = [in_dim] + [hidden] * (L_H - 1) + [num_classes]
        layers = []
        for i in range(L_H):
            linear = nn.Linear(dims[i], dims[i + 1])
            layers.append(LoRALinear(linear, rho) if lora else linear)
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)

    def layer_modules(self) -> List[nn.Module]:
        return list(self.layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            z = layer(z)
            if i < len(self.layers) - 1:
                z = self.dropout(F.relu(z))
        return z


class MultimodalModel(nn.Module):
    def __init__(self, modalities: Sequence[str], encoders: Dict[str, nn.Module], fusion: MDLoRAFusion, head: TaskHead, dropout: float):
        super().__init__()
        self.modalities = list(modalities)
        self.encoders = nn.ModuleDict(encoders)
        self.fusion = fusion
        self.head = head
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Sequence[torch.Tensor]) -> torch.Tensor:
        h = [self.encoders[m](x_m) for m, x_m in zip(self.modalities, x)]
        return self.head(self.dropout(F.relu(self.fusion(h))))

    def parameter_groups(self) -> List[ParamGroup]:
        param_name = {id(p): n for n, p in self.named_parameters()}
        buffer_name = {id(b): n for n, b in self.named_buffers()}
        module_name = {id(mod): n for n, mod in self.named_modules()}
        groups: List[ParamGroup] = []

        def add(name, kind, modality, params, buffers=(), prefix=None):
            groups.append(ParamGroup(len(groups), name, kind, modality, tuple(params), tuple(buffers), prefix))

        def trainable(module: nn.Module) -> List[str]:
            return [param_name[id(p)] for p in module.parameters() if p.requires_grad]

        def buffers(module: nn.Module) -> List[str]:
            return [buffer_name[id(b)] for b in module.buffers()]

        bias = [param_name[id(self.fusion.bias)]] if self.fusion.bias is not None else []
        for m, name in enumerate(self.modalities):
            add(f"A.{name}", "A", m, [param_name[id(self.fusion.A[m])]])
        if self.fusion.b_mode == "shared":
            add("B", "B", None, [param_name[id(self.fusion.B[0])]] + bias)
            head_extra = []
        else:
            for m, name in enumerate(self.modalities):
                add(f"B.{name}", "B", m, [param_name[id(self.fusion.B[m])]])
            head_extra = bias
        for m, name in enumerate(self.modalities):
            for l, layer in enumerate(self.encoders[name].layer_modules()):
                add(f"E.{name}.{l}", "enc", m, trainable(layer), buffers(layer), module_name[id(layer)])
        for l, layer in enumerate(self.head.layer_modules()):
            add(f"H.{l}", "head", None, trainable(layer) + (head_extra if l == 0 else []), buffers(layer), module_name[id(layer)])
        return groups


def build_model(cfg, modalities: Sequence[str], channels: Dict[str, int], num_classes: int, seq_len: int) -> MultimodalModel:
    encoders = {m: build_encoder(cfg, channels[m], seq_len) for m in modalities}
    lora = cfg.backbone != "cnn"
    fusion = MDLoRAFusion([encoders[m].out_dim for m in modalities], cfg.fusion_dim, cfg.rho, frozen_base=lora, b_mode=cfg.b_mode)
    head = TaskHead(cfg.fusion_dim, cfg.head_hidden, num_classes, cfg.L_H, cfg.dropout, lora, cfg.rho)
    return MultimodalModel(modalities, encoders, fusion, head, cfg.dropout)
