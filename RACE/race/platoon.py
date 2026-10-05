from __future__ import annotations

import numpy as np

from .config import PlatoonConfig


class Platoon:
    def __init__(self, cfg: PlatoonConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.N = cfg.N
        self.d = np.full(self.N + 1, cfg.d)
        self.x = np.zeros(self.N + 1)
        self.v = np.zeros(self.N + 1)
        self.a = np.zeros(self.N + 1)

    def reset(self) -> None:
        c = self.cfg
        gaps = self.rng.uniform(c.gap_init[0], c.gap_init[1], size=self.N)
        self.v = self.rng.uniform(c.v_init[0], c.v_init[1], size=self.N + 1)
        self.x = np.zeros(self.N + 1)
        for n in range(1, self.N + 1):
            self.x[n] = self.x[n - 1] - self.d[n - 1] - gaps[n - 1]
        self.a = self.acceleration()

    def relative_position(self) -> np.ndarray:
        return self.x[:-1] - self.x[1:] - self.d[:-1]

    def relative_speed(self) -> np.ndarray:
        return self.v[1:] - self.v[:-1]

    def H(self, v: np.ndarray, dv: np.ndarray) -> np.ndarray:
        c = self.cfg
        return c.d_min + c.t_min * v + v * dv / (2.0 * np.sqrt(c.a_max * c.b_max))

    def acceleration(self) -> np.ndarray:
        c = self.cfg
        a = np.empty(self.N + 1)
        a[0] = c.a_max * (1.0 - (self.v[0] / c.v_des) ** c.delta)
        dx = self.relative_position()
        dv = self.relative_speed()
        vn = self.v[1:]
        a[1:] = c.a_max * (1.0 - (vn / c.v_des) ** c.delta - (self.H(vn, dv) / dx) ** 2)
        return a

    def step(self) -> None:
        tau = self.cfg.tau
        x_prev, v_prev, a_prev = self.x.copy(), self.v.copy(), self.a.copy()
        self.v = v_prev + a_prev * tau
        self.x = x_prev + v_prev * tau + 0.5 * a_prev * tau ** 2
        self.a = self.acceleration()

    def distance_to_leader(self) -> np.ndarray:
        return self.x[0] - self.x[1:]

    def advance(self, duration: float, M: int) -> np.ndarray:
        steps = max(M, int(np.ceil(duration / self.cfg.tau)))
        marks = np.linspace(steps / M, steps, M).round().astype(int)
        snapshots = []
        for s in range(1, steps + 1):
            self.step()
            while len(snapshots) < M and s >= marks[len(snapshots)]:
                snapshots.append(self.distance_to_leader())
        return np.stack(snapshots, axis=0)
