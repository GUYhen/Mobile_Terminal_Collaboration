from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence

import torch
import torch.nn as nn

from ..algorithm.groups import GroupRegistry
from ..models.lora import LoRALinear


@dataclass
class DeviceProfile:
    n: int
    device_type: str
    flops_per_s: float
    P_active: float
    P_comm: float
    P_idle: float
    bandwidth_bps: float
    num_samples: int


@dataclass
class RoundCost:
    T_round: float
    energy: float
    upload_bytes: int
    download_bytes: int
    per_device: Dict[int, Dict[str, float]] = field(default_factory=dict)


def module_macs(module: nn.Module, x: torch.Tensor, out: torch.Tensor) -> float:
    if isinstance(module, LoRALinear):
        tokens = x.numel() // module.in_features
        return float(tokens * module.rho * (module.in_features + module.out_features))
    if isinstance(module, nn.Linear):
        return float(out.numel() * module.in_features)
    if isinstance(module, nn.Conv1d):
        return float(out.numel() * (module.in_channels // module.groups) * module.kernel_size[0])
    return 0.0


@torch.no_grad()
def profile_group_flops(model: nn.Module, registry: GroupRegistry, x: Sequence[torch.Tensor]) -> Dict[int, float]:
    was_training = model.training
    model.eval()
    prefixes = [(g.module_prefix, g.gid) for g in registry if g.module_prefix]
    owner = {}
    for name, module in model.named_modules():
        for prefix, gid in prefixes:
            if name == prefix or name.startswith(prefix + "."):
                owner[id(module)] = gid
                break
    macs = {g.gid: 0.0 for g in registry}

    def hook(module, inputs, output):
        macs[owner[id(module)]] += module_macs(module, inputs[0], output)

    handles = [
        module.register_forward_hook(hook)
        for module in model.modules()
        if id(module) in owner and isinstance(module, (nn.Linear, nn.Conv1d, LoRALinear))
    ]
    model(list(x))
    for handle in handles:
        handle.remove()
    batch = x[0].shape[0]
    fusion_macs = model.fusion.macs_per_sample()
    for g in registry:
        if g.kind in ("A", "B"):
            local = [p.split(".", 1)[1] for p in g.param_names if p.startswith("fusion.")]
            macs[g.gid] += batch * sum(fusion_macs.get(p, 0.0) for p in local)
    model.train(was_training)
    return {gid: 6.0 * v / batch for gid, v in macs.items()}


class SystemModel:
    def __init__(self, cfg, registry: GroupRegistry, group_flops: Mapping[int, float], E: int):
        self.cfg = cfg
        self.registry = registry
        self.group_flops = dict(group_flops)
        self.E = E
        self.devices: Dict[int, DeviceProfile] = {}
        self.download_bytes = registry.payload_bytes(range(registry.G), cfg.bytes_per_param)

    def add_device(self, n: int, device_type: str, num_samples: int) -> DeviceProfile:
        t = self.cfg.device_types[device_type]
        self.devices[n] = DeviceProfile(
            n=n,
            device_type=device_type,
            flops_per_s=t.tops * 1e12,
            P_active=t.active_power_w,
            P_comm=self.cfg.comm_power_ratio * t.active_power_w,
            P_idle=self.cfg.idle_power_ratio * t.active_power_w,
            bandwidth_bps=t.bandwidth_mbps * 1e6,
            num_samples=num_samples,
        )
        return self.devices[n]

    def group_time(self, n: int, j: int) -> float:
        dev = self.devices[n]
        return self.group_flops[j] * self.E * dev.num_samples / dev.flops_per_s

    def tau(self, n: int, G_n: Sequence[int]) -> float:
        return sum(self.group_time(n, j) for j in G_n) / max(len(G_n), 1)

    def comm_time(self, n: int, upload_bytes: int) -> float:
        return 8.0 * (self.download_bytes + upload_bytes) / self.devices[n].bandwidth_bps + self.cfg.sync_latency_s

    def overhead(self, G: Mapping[int, Sequence[int]]) -> float:
        return max(self.comm_time(n, self.registry.payload_bytes(G_n, self.cfg.bytes_per_param)) for n, G_n in G.items())

    def round_cost(self, S: Mapping[int, Sequence[int]], tau: Mapping[int, float]) -> RoundCost:
        per: Dict[int, Dict[str, float]] = {}
        for n, S_n in S.items():
            up = self.registry.payload_bytes(S_n, self.cfg.bytes_per_param)
            per[n] = {"compute_s": len(S_n) * tau[n], "comm_s": self.comm_time(n, up), "upload_bytes": float(up)}
        T_round = max(p["compute_s"] + p["comm_s"] for p in per.values())
        energy = 0.0
        for n, p in per.items():
            dev = self.devices[n]
            p["idle_s"] = max(T_round - p["compute_s"] - p["comm_s"], 0.0)
            p["energy_j"] = dev.P_active * p["compute_s"] + dev.P_comm * p["comm_s"] + dev.P_idle * p["idle_s"]
            energy += p["energy_j"]
        upload = int(sum(p["upload_bytes"] for p in per.values()))
        return RoundCost(T_round, energy, upload, self.download_bytes * len(per), per)
