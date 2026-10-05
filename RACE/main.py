from __future__ import annotations

import argparse

import numpy as np
import torch

from race.baselines import MADDQN, AoIGreedy, ConvexOptimization
from race.channel import WirelessChannel
from race.config import Config, load_config
from race.env import WFLPlatoonEnv
from race.fl import DeepLabV3Plus, WFLTrainer, build_federated_dataset
from race.mappo import MAPPO
from race.platoon import Platoon
from race.resource_allocation import LagrangianDualDecomposition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--method", choices=["race", "maddqn", "convex_greedy"], default="race")
    parser.add_argument("--arch", choices=["tsfen", "mlp"], default=None)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--K", type=int, default=None)
    return parser.parse_args()


def build_env(cfg: Config, method: str, rng: np.random.Generator, device: torch.device) -> WFLPlatoonEnv:
    clients, zeta, test_set, weights = build_federated_dataset(cfg.fl, cfg.platoon.N, rng)
    model = DeepLabV3Plus(cfg.fl.num_classes, cfg.fl.backbone, cfg.fl.aspp_rates)
    trainer = WFLTrainer(model, clients, zeta, test_set, weights, cfg.fl, device)
    channel = WirelessChannel(cfg.channel, rng)
    if method == "convex_greedy":
        allocator = ConvexOptimization(cfg.compute, channel)
    else:
        allocator = LagrangianDualDecomposition(cfg.compute, channel, cfg.resource)
    return WFLPlatoonEnv(cfg, Platoon(cfg.platoon, rng), channel, allocator, trainer)


def initialize_agents(cfg: Config, method: str, env: WFLPlatoonEnv, rng: np.random.Generator,
                      device: torch.device):
    M, N, K, F = cfg.mdp.M, cfg.platoon.N, cfg.channel.K, env.feature_dim
    if method == "race":
        agents = MAPPO(cfg, M, N, K, F, device)
        agents.train(env, cfg.mappo.E)
    elif method == "maddqn":
        agents = MADDQN(cfg, M, N, K, F, device, rng)
        agents.train(env, cfg.maddqn.E)
    else:
        agents = AoIGreedy(K)
    return agents


def race(cfg: Config, method: str) -> list:
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    env = build_env(cfg, method, rng, device)

    env.reset()
    env.stage1()

    agents = initialize_agents(cfg, method, env, rng, device)

    S = env.reset()
    records = []
    for t in range(1, env.T + 1):
        actions, _ = agents.select_actions(S, env.m_tilde, deterministic=True)
        S, R, done, info = env.step(actions)
        info["mIoU"] = env.trainer.mIoU()
        records.append(info)
        print(f"round {t:4d} | reward {info['reward']:.4f} | sum AoI {info['sum_aoi']:.2f} | "
              f"FLMD {info['flmd']:.4f} | mIoU {info['mIoU']:.4f}")
        if done:
            break
    return records


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.arch is not None:
        cfg.network.arch = args.arch
    if args.N is not None:
        cfg.platoon.N = args.N
    if args.K is not None:
        cfg.channel.K = args.K
    if cfg.platoon.N < cfg.channel.K:
        raise ValueError("N >= K is required")
    race(cfg, args.method)


if __name__ == "__main__":
    main()
