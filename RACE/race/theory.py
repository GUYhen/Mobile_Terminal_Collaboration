from __future__ import annotations

from typing import Sequence

import numpy as np


def lipschitz_smooth(grad_prev: np.ndarray, grad_curr: np.ndarray, omega_prev: np.ndarray,
                     omega_curr: np.ndarray, L: float) -> bool:
    return bool(np.linalg.norm(grad_prev - grad_curr) <= L * np.linalg.norm(omega_prev - omega_curr))


def polyak_lojasiewicz(grad_F: np.ndarray, F_gap: float, eta: float) -> bool:
    return bool(np.linalg.norm(grad_F) ** 2 >= 2.0 * eta * F_gap)


def lemma1_bound(grad_f: np.ndarray, grad_F: np.ndarray, zeta: np.ndarray, K: int) -> float:
    N = len(zeta)
    divergence = np.sum(zeta ** 2 * np.linalg.norm(grad_f - grad_F[None, :], axis=1) ** 2)
    return float((1.0 - K / N) * divergence / (K * (N - 1) * np.mean(zeta) ** 2))


def theorem1_bound(t: int, F1_gap: float, e_sq: Sequence[float], mu: float, L: float) -> float:
    c = 1.0 - mu / L
    drift = sum(c ** (t - i) * e_sq[i - 1] for i in range(1, t + 1))
    return float(c ** t * F1_gap + drift / (2.0 * L))


def gamma2(grad_f: np.ndarray, grad_F: np.ndarray, zeta: np.ndarray) -> float:
    return float(np.max(zeta ** 2 * np.linalg.norm(grad_f - grad_F[None, :], axis=1) ** 2))


def theorem2_bound(t: int, F1_gap: float, gamma_sq: float, K: int, N: int, zeta_bar: float,
                   mu: float, L: float) -> float:
    c = 1.0 - mu / L
    geometric = sum(c ** (t - i) for i in range(1, t + 1))
    return float(c ** t * F1_gap + gamma_sq * (1.0 - K / N) / (2.0 * L * K * zeta_bar ** 2) * geometric)


def rho_adaptive(Theta_t: np.ndarray, Lambda_adaptive: float, Lambda_fixed: float) -> float:
    M_a = np.count_nonzero(Theta_t <= Lambda_adaptive)
    M_f = np.count_nonzero(Theta_t <= Lambda_fixed)
    return float(M_a / max(M_f, 1))


def theorem3_bound(t: int, F1_gap: float, e_sq: Sequence[float], rho: Sequence[float], mu: float,
                   L: float) -> float:
    e_a_sq = [r * e for r, e in zip(rho, e_sq)]
    return theorem1_bound(t, F1_gap, e_a_sq, mu, L)


def theorem4_step_size(epsilon: float, L: float, T: int, delta: float) -> float:
    return float(epsilon / (2.0 * L * T * np.sqrt(np.log(1.0 / delta))))


def theorem4_initialization(omega_0: np.ndarray, omega_star: np.ndarray, R: float, epsilon: float) -> bool:
    return bool(np.linalg.norm(omega_0 - omega_star) <= R - epsilon)


def theorem5_bound(F0_gap: float, e_sq: Sequence[float], L: float, T: int) -> float:
    return float((2.0 * L * F0_gap + float(np.sum(e_sq[:T]))) / T)
