from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import brentq

from .channel import WirelessChannel
from .config import ComputeConfig, ResourceConfig
from .system_model import delta_cp, delta_n, e_cp, e_n

LN2 = np.log(2.0)


@dataclass
class Allocation:
    chi: np.ndarray
    rho: np.ndarray
    delta_cp: np.ndarray
    delta_tx: np.ndarray
    e_cp: np.ndarray
    e_tx: np.ndarray
    lambda1: np.ndarray
    feasible: np.ndarray

    @property
    def delta(self) -> np.ndarray:
        return delta_n(self.delta_cp, self.delta_tx)

    @property
    def energy(self) -> np.ndarray:
        return e_n(self.e_cp, self.e_tx)


class LagrangianDualDecomposition:
    def __init__(self, compute: ComputeConfig, channel: WirelessChannel, cfg: ResourceConfig):
        self.mu = compute.mu
        self.G_n = compute.G_n
        self.kappa = compute.kappa
        self.e_max = compute.e_max
        self.B = channel.B
        self.P_n = channel.P_n
        self.channel = channel
        self.cfg = cfg

    def e_cp(self, chi: float, zeta_n: float) -> float:
        return float(e_cp(self.kappa, self.mu, zeta_n, chi, self.G_n))

    def e_tx(self, delta_tx: float, D: float, h2: float) -> float:
        return float(delta_tx * (2.0 ** (D / (delta_tx * self.B)) - 1.0) / h2)

    def rho(self, delta_tx: float, D: float, h2: float) -> float:
        return float((2.0 ** (D / (delta_tx * self.B)) - 1.0) / (self.P_n * h2))

    def delta_tx_min(self, D: float, h2: float) -> float:
        return float(D / (self.B * np.log2(1.0 + self.P_n * h2)))

    def feasible(self, D: float, h2: float) -> bool:
        return not (LN2 * D >= self.e_max * self.B * h2)

    def chi_star(self, lambda1: float, zeta_n: float) -> float:
        if lambda1 == 0.0:
            return 1.0
        ratio = (self.mu * zeta_n) / (2.0 * lambda1 * self.kappa * self.mu * zeta_n * self.G_n ** 3)
        return float(min(ratio ** (1.0 / 3.0), 1.0))

    def dL_ddelta_tx(self, delta_tx: float, lambda1: float, lambda2: float, D: float, h2: float) -> float:
        x = D / (delta_tx * self.B)
        energy_term = (2.0 ** x - 1.0) / h2 - (D * LN2) / (self.B * h2 * delta_tx) * 2.0 ** x
        power_term = (D * LN2) / (self.B * self.P_n * h2 * delta_tx ** 2) * 2.0 ** x
        return 1.0 + lambda1 * energy_term - lambda2 * power_term

    def delta_tx_star_given_lambda1(self, lambda1: float, D: float, h2: float) -> float:
        lo = self.delta_tx_min(D, h2)
        if lambda1 == 0.0 or self.dL_ddelta_tx(lo, lambda1, 0.0, D, h2) >= 0.0:
            return lo
        hi = 2.0 * lo
        while self.dL_ddelta_tx(hi, lambda1, 0.0, D, h2) < 0.0:
            hi *= 2.0
        return float(brentq(lambda t: self.dL_ddelta_tx(t, lambda1, 0.0, D, h2), lo, hi,
                            rtol=self.cfg.tol, maxiter=self.cfg.max_iter))

    def dual_energy(self, lambda1: float, zeta_n: float, D: float, h2: float) -> float:
        chi = self.chi_star(lambda1, zeta_n)
        dtx = self.delta_tx_star_given_lambda1(lambda1, D, h2)
        return self.e_cp(chi, zeta_n) + self.e_tx(dtx, D, h2)

    def lambda1_star(self, zeta_n: float, D: float, h2: float) -> float:
        lo, hi = 0.0, 1.0
        while self.dual_energy(hi, zeta_n, D, h2) > self.e_max:
            lo, hi = hi, 10.0 * hi
        for _ in range(self.cfg.max_iter):
            if hi - lo <= self.cfg.tol * hi:
                break
            mid = np.sqrt(lo * hi) if lo > 0.0 else 0.5 * hi
            if self.dual_energy(mid, zeta_n, D, h2) > self.e_max:
                lo = mid
            else:
                hi = mid
        return float(hi)

    def delta_tx_star_numerical(self, chi: float, zeta_n: float, D: float, h2: float) -> float:
        lo = self.delta_tx_min(D, h2)
        residual = lambda t: self.e_cp(chi, zeta_n) + self.e_tx(t, D, h2) - self.e_max
        if residual(lo) <= 0.0:
            return lo
        hi = 2.0 * lo
        while residual(hi) > 0.0:
            hi *= 2.0
        return float(brentq(residual, lo, hi, rtol=self.cfg.tol, maxiter=self.cfg.max_iter))

    def delta_tx_star_large_model(self, chi: float, zeta_n: float, D: float, h2: float) -> Optional[float]:
        arg = (self.e_max - self.e_cp(chi, zeta_n)) * h2 / (D * LN2)
        if arg <= 1.0:
            return None
        dtx = D * LN2 / (self.B * np.log(arg))
        return float(dtx) if D / (dtx * self.B) >= 3.0 else None

    def delta_tx_star_high_snr(self, chi: float, zeta_n: float, D: float, h2: float) -> Optional[float]:
        if self.P_n * h2 < 10.0:
            return None
        return float(D / (self.B * np.log2(1.0 + self.e_max * h2 / self.e_cp(chi, zeta_n))))

    def solve_pf(self, zeta_n: float, h2: float, D: float) -> dict:
        if not self.feasible(D, h2):
            return dict(chi=1.0, rho=1.0, lambda1=np.nan, feasible=False)
        dtx_min = self.delta_tx_min(D, h2)
        if self.e_cp(1.0, zeta_n) + self.e_tx(dtx_min, D, h2) <= self.e_max:
            return dict(chi=1.0, rho=1.0, lambda1=0.0, feasible=True)
        lambda1 = self.lambda1_star(zeta_n, D, h2)
        chi = self.chi_star(lambda1, zeta_n)
        dtx = None
        if self.cfg.solver == "large_model":
            dtx = self.delta_tx_star_large_model(chi, zeta_n, D, h2)
        elif self.cfg.solver == "high_snr":
            dtx = self.delta_tx_star_high_snr(chi, zeta_n, D, h2)
        if dtx is None:
            dtx = self.delta_tx_star_numerical(chi, zeta_n, D, h2)
        rho = min(self.rho(dtx, D, h2), 1.0)
        return dict(chi=chi, rho=rho, lambda1=lambda1, feasible=True)

    def solve(self, zeta: np.ndarray, h2: np.ndarray, D: float) -> Allocation:
        sols = [self.solve_pf(float(z), float(h), D) for z, h in zip(zeta, h2)]
        chi = np.array([s["chi"] for s in sols])
        rho = np.array([s["rho"] for s in sols])
        dtx = self.channel.delta_tx(D, rho, h2)
        return Allocation(
            chi=chi,
            rho=rho,
            delta_cp=delta_cp(self.mu, zeta, chi, self.G_n),
            delta_tx=dtx,
            e_cp=e_cp(self.kappa, self.mu, zeta, chi, self.G_n),
            e_tx=self.channel.e_tx(rho, dtx),
            lambda1=np.array([s["lambda1"] for s in sols]),
            feasible=np.array([s["feasible"] for s in sols], dtype=bool),
        )
