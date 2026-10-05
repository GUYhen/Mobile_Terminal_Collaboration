import json
import os

import numpy as np
import torch

from .analysis import cumulative_forgetting, layer_replacement
from .client import Rover
from .data import MarsTerrain, TaskStream, TaskView, build_lut, dirichlet_partition, split_ids
from .metrics import evaluate
from .model import build_model, model_like
from .rkr import RecoveryFunction
from .server import Server, fedavg
from .utils import clone_state, set_seed


class Lifecycle:
    def __init__(self, cfg):
        set_seed(cfg.seed)
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        data = cfg.data
        self.ignore_index = data.ignore_index
        self.batch_size = cfg.train.batch_size
        self.stream = TaskStream(data.classes, data.task_split)
        lut = build_lut(data.raw_ids, data.classes, self.ignore_index)
        train_ids, test_ids = split_ids(data.root, data.max_images, data.test_fraction, cfg.seed)
        self.train_set = MarsTerrain(data.root, train_ids, lut, data.img_size)
        self.test_set = MarsTerrain(data.root, test_ids, lut, data.img_size)
        C = self.stream.num_classes
        self.train_hist = self.train_set.class_pixels(C, self.ignore_index)
        self.test_hist = self.test_set.class_pixels(C, self.ignore_index)
        K = cfg.fl.K
        parts = dirichlet_partition(self.train_hist, K, cfg.fl.beta, cfg.seed)
        capacities = cfg.memory.capacities
        if not isinstance(capacities, list):
            capacities = [capacities] * K
        self.model = build_model(cfg.model, len(self.stream.Y[0])).to(self.device)
        self.rovers = [
            Rover(k, cfg, self.stream, self.train_set, self.train_hist, parts[k], capacities[k], self.model.d, self.device)
            for k in range(K)
        ]
        self.server = Server(self.model.state_dict(), cfg.fl.clients_per_round, cfg.seed)
        self.loss = {}
        self.rounds = []
        self.tasks = []
        os.makedirs(cfg.output_dir, exist_ok=True)

    def test_view(self, Y):
        idx = np.nonzero(self.test_hist[:, Y].sum(1) > 0)[0].tolist()
        return TaskView(self.test_set, idx, Y, self.ignore_index)

    def evaluate(self, Theta, t):
        self.model.load_state_dict(Theta)
        learned = self.stream.learned(t)
        return evaluate(self.model, self.test_view(learned), learned, self.batch_size, self.ignore_index, self.device)

    def forgetting(self, t):
        self.model.load_state_dict(self.server.Theta)
        for t_prime in range(t + 1):
            Y = self.stream.Y[t_prime]
            result = evaluate(self.model, self.test_view(Y), Y, self.batch_size, self.ignore_index, self.device)
            self.loss.setdefault(t, {})[t_prime] = result["loss"]
        return cumulative_forgetting(self.loss, t)

    def communication_round(self, t, r):
        C_hat = self.server.select(len(self.rovers))
        Thetas = [self.rovers[k].local_train(self.model, self.server.Theta, t) for k in C_hat]
        self.server.aggregate(Thetas)
        self.rounds.append({"task": t, "round": r, "mIoU": self.evaluate(self.server.Theta, t)["mIoU"]})

    def train_recovery_function(self):
        xi = clone_state(RecoveryFunction(self.model.d, self.stream.num_classes, self.cfg.rkr.hidden).state_dict())
        xis = [rover.meta_train(self.model, self.server.Theta, xi) for rover in self.rovers]
        xi = fedavg([x for x in xis if x is not None])
        for rover in self.rovers:
            rover.receive_psi(xi)

    def task_zero_initialization(self):
        for r in range(1, self.cfg.fl.R + 1):
            self.communication_round(0, r)
        for rover in self.rovers:
            rover.end_task(self.model, self.server.Theta, 0)
        if self.cfg.method == "lifecycle_aware":
            self.train_recovery_function()
        self.save(0, {"mIoU": self.evaluate(self.server.Theta, 0)["mIoU"], "Delta": self.forgetting(0)})

    def run(self):
        self.task_zero_initialization()
        T = self.stream.T
        for t in range(1, T + 1):
            self.model.load_state_dict(self.server.Theta)
            self.model.add_classes(self.stream.C(t))
            self.server.Theta = clone_state(self.model.state_dict())
            for r in range(1, self.cfg.fl.R + 1):
                self.communication_round(t, r)
            for rover in self.rovers:
                rover.end_task(self.model, self.server.Theta, t)
            record = {"mIoU": self.evaluate(self.server.Theta, t)["mIoU"], "Delta": self.forgetting(t)}
            if self.cfg.method == "lifecycle_aware":
                force = self.cfg.rkr.trigger_after_final_task and t == T
                record["recovery"] = [
                    rover.rapid_knowledge_recovery(self.model, self.server.Theta, t, force) for rover in self.rovers
                ]
                record["mIoU_recovered"] = float(np.nanmean([self.evaluate(rover.Theta_k, t)["mIoU"] for rover in self.rovers]))
            self.save(t, record)
        return self.server.Theta

    def save(self, t, record):
        torch.save(self.server.Theta, os.path.join(self.cfg.output_dir, f"Theta_task{t}.pt"))
        record["task"] = t
        self.tasks.append(record)
        with open(os.path.join(self.cfg.output_dir, "lifecycle.json"), "w", encoding="utf-8") as f:
            json.dump({"rounds": self.rounds, "tasks": self.tasks}, f, indent=2)

    def layer_replacement(self, Theta_0, Theta_t):
        model = model_like(self.model, Theta_0)
        Y_0 = self.stream.Y[0]
        view = self.test_view(Y_0)

        def mIoU(m):
            return evaluate(m, view, Y_0, self.batch_size, self.ignore_index, self.device)["mIoU"]

        return layer_replacement(model, Theta_0, Theta_t, self.cfg.model.layer_groups, mIoU)
