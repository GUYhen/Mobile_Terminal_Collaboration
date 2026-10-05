import torch


class SparsificationMDP:
    def __init__(self, Theta, L_s, S, groups, aoi):
        self.Theta = Theta
        self.L_s = list(L_s)
        self.S = S
        self.S_max = max(S[l] for l in self.L_s)
        self.groups = groups
        self.G_l = groups.G_l
        self.aoi = aoi
        self.W = {l: Theta[l].detach().reshape(-1).clone().float() for l in self.L_s}
        self.C_t = 0

    @property
    def dim_H(self):
        return len(self.L_s) * self.G_l * 3

    def state(self, t):
        A = self.aoi.A(t, self.L_s)
        H = torch.zeros(len(self.L_s), self.G_l, 3)
        for i, l in enumerate(self.L_s):
            for j in range(self.G_l):
                W_lj = self.groups.group(self.W[l], l, j)
                mu_lj = W_lj.mean()
                sigma_lj = torch.sqrt(((W_lj - mu_lj) ** 2).mean())
                H[i, j, 0] = float(A[l][j])
                H[i, j, 1] = mu_lj
                H[i, j, 2] = sigma_lj
        return H.reshape(-1)

    def reward(self, ell, G, s, t, w_f, w_c):
        if len(G) == 0:
            return 0.0, 0.0, 0.0
        l = self.L_s[ell]
        A = self.aoi.A(t)
        A_max = max(float(A_l.max()) for A_l in A.values())
        R_f = sum(self.S[l] / self.S_max * s for _ in G)
        R_c = sum(float(A[l][j]) / A_max * s for j in G) / len(G) if A_max > 0 else 0.0
        R = w_f * R_f + w_c * R_c
        return R, R_f, R_c

    @torch.no_grad()
    def sparsify(self, ell, G, s, t):
        l = self.L_s[ell]
        for j in G:
            W_lj = self.groups.group(self.W[l], l, j)
            k = int(s * W_lj.numel())
            if k > 0:
                idx = torch.topk(W_lj.abs(), k, largest=False).indices
                W_lj[idx] = 0.0
            self.C_t += W_lj.numel()
        self.aoi.update(l, G, t)

    def unlearned_model(self):
        Theta_prime = {k: v.clone() for k, v in self.Theta.items()}
        for l in self.L_s:
            Theta_prime[l] = self.W[l].reshape(self.Theta[l].shape).to(self.Theta[l].dtype)
        return Theta_prime
