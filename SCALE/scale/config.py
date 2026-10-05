from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class FLConfig:
    N: int = 100
    E: int = 10
    T: int = 100
    eta: float = 1e-3
    dirichlet_alpha: float = 1.0
    clients_per_round: int = 10
    batch_size: int = 64


@dataclass
class UnlearnConfig:
    tau: str = "client"
    n: int = 0
    Y_u: Tuple[int, ...] = (0,)
    sample_fraction: float = 0.1


@dataclass
class SCALEConfig:
    lam: float = 0.5
    M: int = 3
    G_l: int = 8
    w_f: float = 0.7
    w_c: float = 0.3
    T_total: int = 800
    T_collect: int = 32
    batch_size: int = 128
    K_epoch: int = 10
    clip_eps: float = 0.2
    gamma: float = 0.99
    lambda_gae: float = 0.95
    eta_a: float = 3e-4
    eta_c: float = 3e-4
    hidden_dim: int = 128
    eps: float = 1e-8


@dataclass
class OverheadConfig:
    alpha: float = 1.0
    beta: float = 1.0


@dataclass
class TheoryConfig:
    gamma0: float = 1.0
    gamma1: float = 0.1


@dataclass
class ExperimentConfig:
    dataset: str = "fashionmnist"
    model: str = "lenet"
    data_root: str = "./data"
    output_dir: str = "./outputs"
    seed: int = 0
    device: str = "cuda"


@dataclass
class Config:
    fl: FLConfig = field(default_factory=FLConfig)
    unlearn: UnlearnConfig = field(default_factory=UnlearnConfig)
    scale: SCALEConfig = field(default_factory=SCALEConfig)
    overhead: OverheadConfig = field(default_factory=OverheadConfig)
    theory: TheoryConfig = field(default_factory=TheoryConfig)
    exp: ExperimentConfig = field(default_factory=ExperimentConfig)
