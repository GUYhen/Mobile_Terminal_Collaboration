from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence

import numpy as np


@dataclass
class PlatoonFollower:
    zeta_n: int
    C_n: Dict[str, float] = field(default_factory=dict)


def W(zeta: np.ndarray) -> list:
    return [PlatoonFollower(zeta_n=int(z)) for z in zeta]


def delta_cp(mu: float, zeta: np.ndarray, chi: np.ndarray, G_n: float) -> np.ndarray:
    return mu * zeta * (chi * G_n) ** -1.0


def e_cp(kappa: float, mu: float, zeta: np.ndarray, chi: np.ndarray, G_n: float) -> np.ndarray:
    return kappa * mu * zeta * (chi * G_n) ** 2


def delta_n(delta_cp_n: np.ndarray, delta_tx_n: np.ndarray) -> np.ndarray:
    return delta_cp_n + delta_tx_n


def delta_round(phi: np.ndarray, delta: np.ndarray) -> float:
    return float(np.max(phi.sum(axis=0) * delta))


def e_n(e_cp_n: np.ndarray, e_tx_n: np.ndarray) -> np.ndarray:
    return e_cp_n + e_tx_n


def aoi_update(Delta_prev: np.ndarray, phi: np.ndarray, delta_max: float) -> np.ndarray:
    return (1.0 - phi.sum(axis=0)) * (Delta_prev + delta_max)


def assignment(actions: Sequence[int], N: int) -> np.ndarray:
    phi = np.zeros((len(actions), N))
    for k, n in enumerate(actions):
        phi[k, int(n)] = 1.0
    return phi


def check_constraints(phi: np.ndarray, energy: np.ndarray, e_max: float, chi: np.ndarray,
                      rho: np.ndarray, Theta_S: np.ndarray, Lambda_Theta: float) -> Dict[str, bool]:
    return {
        "23a": bool(np.all(energy <= e_max)),
        "23b": bool(np.all((chi >= 0.0) & (chi <= 1.0))),
        "23c": bool(np.all((rho >= 0.0) & (rho <= 1.0))),
        "23d": bool(np.isin(phi, (0.0, 1.0)).all()),
        "23e": bool(np.all(phi.sum(axis=1) == 1.0)),
        "23f": bool(np.isin(phi.sum(axis=0), (0.0, 1.0)).all()),
        "23g": bool(np.all(Theta_S <= Lambda_Theta)),
    }


def objective_term(Delta: np.ndarray, Theta: np.ndarray, lambda1: float, lambda2: float) -> float:
    return float(lambda1 * Delta.sum() + lambda2 * Theta.sum())
