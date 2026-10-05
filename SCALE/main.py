import argparse
import json
import os

import numpy as np
import torch

from scale.algorithm import run_scale
from scale.aoi import avg_aoi, communication_overhead
from scale.config import Config
from scale.data import (dataset_info, dirichlet_partition, labels_of, load_dataset, make_loader, make_request,
                        remaining_data)
from scale.federated import Client, EdgeServer, global_loss, retrain
from scale.metrics import forgetting_accuracy, forgetting_rate, remaining_accuracy
from scale.models import build_model, model_layers
from scale.theory import client_proportion, corollary1, theorem1_lower_bound, theorem2_bound, theorem3_bound


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["fashionmnist", "cifar100"], default=None)
    parser.add_argument("--model", choices=["lenet", "mobilenetv3", "resnet18"], default=None)
    parser.add_argument("--tau", choices=["client", "class", "sample"], default=None)
    parser.add_argument("--n", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    if args.dataset is not None:
        cfg.exp.dataset = args.dataset
    if args.model is not None:
        cfg.exp.model = args.model
    if args.tau is not None:
        cfg.unlearn.tau = args.tau
    if args.n is not None:
        cfg.unlearn.n = args.n

    torch.manual_seed(cfg.exp.seed)
    rng = np.random.default_rng(cfg.exp.seed)
    device = torch.device(cfg.exp.device if torch.cuda.is_available() else "cpu")

    dataset = load_dataset(cfg.exp.dataset, cfg.exp.data_root, train=True)
    labels = labels_of(dataset)
    in_channels, num_classes, img_size = dataset_info(cfg.exp.dataset)

    def model_fn():
        return build_model(cfg.exp.model, in_channels, num_classes, img_size)

    model = model_fn().to(device)
    layers = model_layers(model)

    D = dirichlet_partition(labels, cfg.fl.N, cfg.fl.dirichlet_alpha, rng)
    clients = [Client(n, dataset, D[n], cfg.fl.batch_size, device) for n in range(cfg.fl.N)]
    server = EdgeServer(model, clients, layers, cfg.fl, cfg.scale.G_l, rng)
    Theta = server.train()
    F_Theta = global_loss(model, Theta, clients)

    U = make_request(cfg.unlearn.tau, cfg.unlearn.n, D, labels, cfg.unlearn.Y_u, cfg.unlearn.sample_fraction, rng)
    n = U.C_u[0]

    Theta_prime, L_s, sensitivity, mdp, log = run_scale(
        Theta, server.W_clients, server.M_sizes, n, layers, server.groups, server.aoi, server.t, cfg, device)

    D_r = remaining_data(D, U)
    loader_r = make_loader(dataset, np.concatenate(D_r), cfg.fl.batch_size)
    loader_u = make_loader(dataset, U.forget_indices(), cfg.fl.batch_size)
    Theta_star, clients_r = retrain(model_fn, dataset, D_r, layers, cfg, device, rng)

    results = {}
    for name, Theta_u in (("SCALE", Theta_prime), ("Retrain", Theta_star)):
        results[name] = {
            "RA": remaining_accuracy(model, Theta_u, loader_r, device),
            "FA": forgetting_accuracy(model, Theta_u, loader_u, device),
            "FR": forgetting_rate(model, Theta, Theta_u, loader_u, device),
            "F_r": global_loss(model, Theta_u, clients_r),
        }
    E_A_g = float(np.mean([r["A_g"] for r in log]))
    results["SCALE"]["avg_aoi"] = float(np.mean([r["avg_aoi"] for r in log]))
    results["SCALE"]["C_t"] = mdp.C_t
    results["SCALE"]["E_A_g"] = E_A_g
    results["SCALE"]["overhead"] = communication_overhead(mdp.C_t, E_A_g, cfg.overhead.alpha, cfg.overhead.beta)
    results["delta"] = {k: abs(results["SCALE"][k] - results["Retrain"][k]) for k in ("RA", "FA", "FR")}

    alpha_ln = client_proportion(server.M_sizes, n)
    A_final = mdp.aoi.A(cfg.scale.T_total)
    A_bar = avg_aoi(A_final)
    A_bar_sen = avg_aoi({l: A_final[l] for l in L_s})
    theory = {
        "theorem1": {l: {"S": sensitivity["S"][l],
                         "lower_bound": theorem1_lower_bound(alpha_ln, sensitivity["S_d"][l], cfg.scale.lam)}
                     for l in layers},
        "corollary1": corollary1({l: alpha_ln for l in layers}, L_s, len(layers), cfg.scale.M),
        "theorem2": theorem2_bound(log[-1]["s"], {l: A_final[l] for l in L_s}, cfg.theory.gamma0,
                                   cfg.theory.gamma1, max(sensitivity["S"].values())),
        "theorem3": theorem3_bound(len(layers), len(L_s), A_bar, A_bar_sen) if A_bar_sen > 0 else float("inf"),
    }

    os.makedirs(cfg.exp.output_dir, exist_ok=True)
    tag = f"{cfg.exp.dataset}_{cfg.exp.model}_{cfg.unlearn.tau}"
    torch.save(Theta_prime, os.path.join(cfg.exp.output_dir, f"{tag}_theta_prime.pt"))
    with open(os.path.join(cfg.exp.output_dir, f"{tag}.json"), "w") as f:
        json.dump({"F_Theta": F_Theta, "L_s": L_s, "sensitivity": sensitivity, "results": results,
                   "theory": theory, "log": log}, f, indent=2)


if __name__ == "__main__":
    main()
