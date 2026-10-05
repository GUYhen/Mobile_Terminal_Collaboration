import math

import numpy as np


def client_proportion(M_sizes, n):
    return M_sizes[n] / float(sum(M_sizes.values()))


def unlearning_effectiveness(A_lj, gamma0, gamma1):
    return gamma0 + gamma1 * A_lj


def theorem1_lower_bound(alpha_ln, D_KL, lam):
    return lam * alpha_ln ** 2 / (2.0 * (1.0 - alpha_ln ** 2)) + (1.0 - lam) * D_KL


def corollary1(alpha, L_s, L, M):
    delta = (L - M) / L
    lhs = sum(alpha[l] for l in L_s)
    rhs = (1.0 - delta) * sum(alpha.values())
    return lhs, rhs, lhs >= rhs


def theorem2_bound(s, A_sen, gamma0, gamma1, S_max):
    C_1 = 2.0 * S_max
    inner = sum(float(np.mean(s ** 2 / unlearning_effectiveness(A_l, gamma0, gamma1))) for A_l in A_sen.values())
    return C_1 / len(A_sen) * inner


def theorem3_bound(L, L_s_size, A_bar, A_bar_sen):
    return math.sqrt(L / L_s_size) * A_bar / A_bar_sen
