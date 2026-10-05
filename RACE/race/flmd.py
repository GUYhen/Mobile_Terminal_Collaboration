from __future__ import annotations

import numpy as np
import torch


def Theta(omega_n: torch.Tensor, omega: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(omega_n - omega) / torch.linalg.vector_norm(omega))


def Theta_from_gradient(xi: float, grad_f_n: torch.Tensor, omega: torch.Tensor) -> float:
    return float(xi * torch.linalg.vector_norm(grad_f_n) / torch.linalg.vector_norm(omega))


def Theta_upper_bound(xi: float, L: float, omega_dist_to_opt: float, grad_f_n_opt_norm: float,
                      omega_norm: float) -> float:
    return xi * (L * omega_dist_to_opt + grad_f_n_opt_norm) / omega_norm


def E_set(Theta_t: np.ndarray, Lambda_Theta: float) -> np.ndarray:
    return np.flatnonzero(Theta_t <= Lambda_Theta)


def Lambda_Theta(grad_F_sq_t: float, grad_F_sq_0: float, Lambda_min: float, Lambda_max: float,
                 beta: float) -> float:
    return Lambda_min + (Lambda_max - Lambda_min) * float(np.exp(-beta * grad_F_sq_t / grad_F_sq_0))


def m(Theta_t: np.ndarray, Lambda: float) -> np.ndarray:
    return (Theta_t <= Lambda).astype(np.float64)


def m_tilde(Theta_t: np.ndarray, Lambda: float, beta: float, mu: float, L: float, t: int) -> np.ndarray:
    soft = np.exp(-beta * Theta_t * (1.0 - mu / L) ** (-t))
    return np.where(Theta_t <= Lambda, 1.0, soft)
