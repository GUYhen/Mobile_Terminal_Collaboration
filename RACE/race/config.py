from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, List

import yaml


@dataclass
class PlatoonConfig:
    N: int
    v_init: List[float]
    gap_init: List[float]
    d: float
    a_max: float
    b_max: float
    d_min: float
    t_min: float
    v_des: float
    delta: float
    tau: float


@dataclass
class ChannelConfig:
    K: int
    B: float
    alpha: float
    eta: float
    sigma2_dbm: float
    P_n_dbm: float
    P: float


@dataclass
class ComputeConfig:
    mu: float
    G_n: float
    kappa: float
    e_max: float
    D: float


@dataclass
class ResourceConfig:
    solver: str
    tol: float
    max_iter: int


@dataclass
class FLMDConfig:
    Lambda_Theta: float
    Lambda_min: float
    Lambda_max: float
    beta_Lambda: float
    beta_mask: float


@dataclass
class ConvergenceConfig:
    mu: float
    L: float


@dataclass
class MDPConfig:
    M: int
    T: int
    alpha: float
    beta: float
    lambda1: float
    lambda2: float


@dataclass
class NetworkConfig:
    arch: str
    num_heads: int
    lstm_hidden: int
    mlp_hidden: List[int]


@dataclass
class MAPPOConfig:
    E: int
    beta: float
    gamma: float
    lambda_: float
    epsilon: float
    N_batch: int


@dataclass
class MADDQNConfig:
    E: int
    beta: float
    gamma: float
    replay_buffer: int
    N_batch: int
    eps_start: float
    eps_end: float
    eps_decay_steps: int
    target_update: int


@dataclass
class FLConfig:
    root: str
    image_dir: str
    train_label_dir: str
    test_label_dir: str
    sample_frac: float
    phi: float
    num_classes: int
    ignore_index: int
    image_size: int
    backbone: str
    aspp_rates: List[int]
    xi: float
    poly_power: float
    epochs: int
    batch_size: int
    color_jitter: List[float]
    topk: float
    quant_bits: int


@dataclass
class Config:
    seed: int
    device: str
    platoon: PlatoonConfig
    channel: ChannelConfig
    compute: ComputeConfig
    resource: ResourceConfig
    flmd: FLMDConfig
    convergence: ConvergenceConfig
    mdp: MDPConfig
    network: NetworkConfig
    mappo: MAPPOConfig
    maddqn: MADDQNConfig
    fl: FLConfig


_SECTIONS = {
    "PlatoonConfig": PlatoonConfig,
    "ChannelConfig": ChannelConfig,
    "ComputeConfig": ComputeConfig,
    "ResourceConfig": ResourceConfig,
    "FLMDConfig": FLMDConfig,
    "ConvergenceConfig": ConvergenceConfig,
    "MDPConfig": MDPConfig,
    "NetworkConfig": NetworkConfig,
    "MAPPOConfig": MAPPOConfig,
    "MADDQNConfig": MADDQNConfig,
    "FLConfig": FLConfig,
}


def _build(cls, raw: Dict[str, Any]):
    kwargs = {}
    for f in fields(cls):
        value = raw[f.name]
        type_name = f.type if isinstance(f.type, str) else getattr(f.type, "__name__", "")
        if type_name in _SECTIONS:
            value = _build(_SECTIONS[type_name], value)
        kwargs[f.name] = value
    return cls(**kwargs)


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    cfg = _build(Config, raw)
    if cfg.platoon.N < cfg.channel.K:
        raise ValueError("N >= K is required")
    return cfg
