from __future__ import annotations

import numpy as np

from .config import ChannelConfig


def dbm_to_watt(dbm: float) -> float:
    return 10.0 ** ((dbm - 30.0) / 10.0)


class WirelessChannel:
    def __init__(self, cfg: ChannelConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.B = cfg.B
        self.alpha = cfg.alpha
        self.eta = cfg.eta
        self.P = cfg.P
        self.P_n = dbm_to_watt(cfg.P_n_dbm)
        self.sigma2 = dbm_to_watt(cfg.sigma2_dbm)

    def h_hat2(self, d_n: np.ndarray) -> np.ndarray:
        g_hat = (self.rng.standard_normal(d_n.shape) + 1j * self.rng.standard_normal(d_n.shape)) / np.sqrt(2.0)
        return np.abs(g_hat) ** 2 * self.eta * d_n ** (-self.alpha)

    def h_tilde2(self, d_n: np.ndarray) -> np.ndarray:
        g_tilde = (self.rng.standard_normal(d_n.shape) + 1j * self.rng.standard_normal(d_n.shape)) / np.sqrt(2.0)
        return np.abs(g_tilde) ** 2 * self.eta * d_n ** (-self.alpha)

    def h2(self, d_n: np.ndarray) -> np.ndarray:
        d_n = np.asarray(d_n, dtype=float)
        composite = np.sqrt(1.0 - self.P) * self.h_hat2(d_n) + np.sqrt(self.P) * self.h_tilde2(d_n)
        return composite / self.sigma2

    def r(self, rho: np.ndarray, h2: np.ndarray) -> np.ndarray:
        return self.B * np.log2(1.0 + rho * self.P_n * h2)

    def delta_tx(self, D: float, rho: np.ndarray, h2: np.ndarray) -> np.ndarray:
        return D / self.r(rho, h2)

    def e_tx(self, rho: np.ndarray, delta_tx: np.ndarray) -> np.ndarray:
        return rho * self.P_n * delta_tx
