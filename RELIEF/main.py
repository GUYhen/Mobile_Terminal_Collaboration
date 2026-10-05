import argparse
import json
import logging
import os
import random
from dataclasses import asdict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from relief.algorithm import GroupRegistry
from relief.config import load_config
from relief.data import ModalityWindowDataset, build_federated_splits, load_subject_windows
from relief.fl import Client, ReliefServer, SystemModel, profile_group_flops
from relief.metrics import critical_classes
from relief.models import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--method", default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_Y(cfg):
    if not cfg.eval.critical_classes_file:
        return None
    with open(cfg.eval.critical_classes_file, "r", encoding="utf-8") as f:
        reference = json.load(f)
    per_class = reference.get("val_per_class_f1", reference["per_class_f1"])
    Y = critical_classes(per_class, cfg.eval.critical_f1_threshold)
    return {m: Y for m in cfg.spec.modalities}


def main():
    args = parse_args()
    cfg = load_config(args.config, args.method, args.set)
    set_seed(cfg.seed)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    device = torch.device(cfg.device if cfg.device == "cpu" or torch.cuda.is_available() else "cpu")
    spec = cfg.spec
    T = cfg.data.window_size
    rng = np.random.default_rng(cfg.seed)

    windows = load_subject_windows(cfg.data, spec)
    type_modalities = {t: cfg.device_modalities(t) for t in cfg.system.device_types}
    splits, test, val = build_federated_splits(windows, spec, cfg.data, type_modalities, rng)
    channels = {m: test[0][m].shape[1] for m in spec.modalities}

    model = build_model(cfg.model, spec.modalities, channels, spec.num_classes, T).to(device)
    registry = GroupRegistry(model.parameter_groups(), model, len(spec.modalities))

    clients = [
        Client(s, spec.modalities, channels, T, spec.num_classes, cfg.fl.batch_size, cfg.data.class_reweighting)
        for s in splits
    ]
    probe = [torch.zeros(2, channels[m], T, device=device) for m in spec.modalities]
    system = SystemModel(cfg.system, registry, profile_group_flops(model, registry, probe), cfg.fl.E)
    for c in clients:
        system.add_device(c.n, c.device_type, c.num_samples)

    def loader(w):
        dataset = ModalityWindowDataset(w[0], w[1], spec.modalities, channels, T, spec.modalities)
        return DataLoader(dataset, batch_size=cfg.eval.batch_size, shuffle=False)

    server = ReliefServer(
        cfg,
        model,
        registry,
        clients,
        system,
        device,
        loader(test),
        loader(val) if val is not None else None,
        load_Y(cfg),
    )
    result = server.run()

    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    final = next((h for h in reversed(result["history"]) if "per_class_f1" in h), {})
    with open(os.path.join(cfg.out_dir, "final_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(final, f, indent=1)
    with open(os.path.join(cfg.out_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)


if __name__ == "__main__":
    main()
