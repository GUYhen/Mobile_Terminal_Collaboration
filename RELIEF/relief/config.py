from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional, Sequence, Union, get_args, get_origin, get_type_hints

import yaml


@dataclass
class DataConfig:
    dataset: str = "pamap2"
    root: str = "./data/PAMAP2_Dataset/Protocol"
    window_size: int = 256
    stride: int = 50
    sample_rate_hz: int = 50
    test_ratio: float = 0.2
    val_subjects: List[int] = field(default_factory=list)
    num_clients: Optional[int] = None
    dirichlet_beta: float = 0.3
    class_reweighting: bool = False
    force_device_type: Optional[str] = None


@dataclass
class DatasetSpec:
    modalities: List[str] = field(default_factory=list)
    num_classes: int = 12
    raw_sample_rate_hz: int = 50
    excluded_subjects: List[int] = field(default_factory=list)
    device_type_counts: Dict[str, int] = field(default_factory=dict)
    rare_modalities: List[str] = field(default_factory=list)


@dataclass
class CNNConfig:
    channels: List[int] = field(default_factory=lambda: [64, 128])
    kernel_size: int = 5
    pool: int = 2


@dataclass
class MomentConfig:
    name: str = "AutonLab/MOMENT-1-small"
    seq_len: int = 512


@dataclass
class PatchTSTConfig:
    d_model: int = 128
    n_layers: int = 3
    n_heads: int = 4
    d_ff: int = 256
    patch_len: int = 16
    patch_stride: int = 8
    checkpoint: Optional[str] = None


@dataclass
class ModelConfig:
    backbone: str = "cnn"
    cnn: CNNConfig = field(default_factory=CNNConfig)
    moment: MomentConfig = field(default_factory=MomentConfig)
    patchtst: PatchTSTConfig = field(default_factory=PatchTSTConfig)
    fusion_dim: int = 128
    rho: int = 8
    lora_targets: List[str] = field(default_factory=lambda: ["q", "v", "wi", "wi_0", "wi_1", "wo"])
    head_hidden: int = 64
    L_H: int = 2
    dropout: float = 0.2
    b_mode: str = "shared"


@dataclass
class FLConfig:
    R: int = 200
    E: int = 5
    batch_size: int = 32
    lr: float = 1e-3
    participation: float = 1.0


@dataclass
class TStarSearchConfig:
    lo_mult: float = 0.5
    hi_mult: float = 2.0
    iters: int = 4
    refresh_every: int = 50


@dataclass
class ReliefConfig:
    gamma: float = 0.9
    lam: float = 1.0
    aggregation: str = "cohort"
    b_aggregation: str = "modality_count"
    allocation: str = "divergence"
    mandatory_inclusion: bool = True
    Q: Optional[int] = None
    train_scope: str = "accessible"
    tstar: TStarSearchConfig = field(default_factory=TStarSearchConfig)


@dataclass
class DeviceTypeConfig:
    modalities: List[str] = field(default_factory=lambda: ["all"])
    tops: float = 1.0
    active_power_w: float = 5.0
    bandwidth_mbps: float = 20.0


@dataclass
class SystemConfig:
    device_types: Dict[str, DeviceTypeConfig] = field(default_factory=dict)
    idle_power_ratio: float = 0.2
    comm_power_ratio: float = 0.5
    sync_latency_s: float = 0.05
    bytes_per_param: int = 4
    dynamic_drop_frac: float = 0.0


@dataclass
class EvalConfig:
    tta_threshold: float = 0.85
    eval_every: int = 1
    batch_size: int = 256
    critical_f1_threshold: float = 0.5
    critical_classes_file: Optional[str] = None
    diagnostics_every: int = 0
    utility_probe: bool = False


@dataclass
class Config:
    seed: int = 0
    device: str = "cuda"
    out_dir: str = "runs/relief"
    method: str = "relief"
    data: DataConfig = field(default_factory=DataConfig)
    datasets: Dict[str, DatasetSpec] = field(default_factory=dict)
    model: ModelConfig = field(default_factory=ModelConfig)
    fl: FLConfig = field(default_factory=FLConfig)
    relief: ReliefConfig = field(default_factory=ReliefConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @property
    def spec(self) -> DatasetSpec:
        return self.datasets[self.data.dataset]

    def device_modalities(self, device_type: str) -> List[str]:
        mods = self.system.device_types[device_type].modalities
        if "all" in mods:
            return list(self.spec.modalities)
        return [m for m in self.spec.modalities if m in mods]


METHOD_PRESETS: Dict[str, Dict[str, Any]] = {
    "relief": {},
    "relief-acc": {"relief.lam": 0.0},
    "relief-fast": {"relief.lam": 1.0},
    "fedavg": {
        "relief.lam": 0.0,
        "relief.train_scope": "full_model",
        "relief.aggregation": "fedavg",
        "relief.b_aggregation": "uniform",
    },
    "cohort-fedavg": {
        "relief.lam": 0.0,
        "relief.train_scope": "full_model",
        "relief.aggregation": "cohort",
        "relief.b_aggregation": "uniform",
    },
    "ablation-v2": {
        "relief.lam": 1.0,
        "relief.aggregation": "fedavg",
        "relief.mandatory_inclusion": False,
    },
    "ablation-v3": {
        "relief.lam": 1.0,
        "relief.aggregation": "fedavg",
        "relief.allocation": "random",
        "relief.mandatory_inclusion": False,
    },
    "b-uniform": {"relief.b_aggregation": "uniform"},
    "b-freeze": {"relief.b_aggregation": "freeze"},
    "b-block": {"model.b_mode": "block"},
}


def build(tp: Any, value: Any) -> Any:
    if value is None:
        return None
    if isinstance(tp, type) and is_dataclass(tp):
        hints = get_type_hints(tp)
        names = {f.name for f in fields(tp)}
        unknown = set(value) - names
        if unknown:
            raise KeyError(f"unknown keys for {tp.__name__}: {sorted(unknown)}")
        return tp(**{k: build(hints[k], v) for k, v in value.items()})
    origin = get_origin(tp)
    if origin is Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        return build(args[0], value) if len(args) == 1 else value
    if origin in (list, List):
        (item,) = get_args(tp)
        return [build(item, v) for v in value]
    if origin in (dict, Dict):
        _, item = get_args(tp)
        return {k: build(item, v) for k, v in value.items()}
    if tp is float and isinstance(value, int):
        return float(value)
    return value


def set_path(d: Dict[str, Any], key: str, value: Any) -> None:
    *parents, leaf = key.split(".")
    for p in parents:
        d = d.setdefault(p, {})
    d[leaf] = value


def load_config(path: str, method: Optional[str] = None, overrides: Sequence[str] = ()) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw["method"] = method or raw.get("method", "relief")
    if raw["method"] not in METHOD_PRESETS:
        raise KeyError(f"unknown method {raw['method']}")
    for key, value in METHOD_PRESETS[raw["method"]].items():
        set_path(raw, key, value)
    for item in overrides:
        key, _, value = item.partition("=")
        set_path(raw, key.strip(), yaml.safe_load(value))
    return build(Config, raw)
