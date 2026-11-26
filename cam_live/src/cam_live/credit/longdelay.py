"""Long-delay key-to-door with closed-form credit quantities for delays of 16 ... 1024+ decisions.

Provenance: copied from superlearner/src/superlearner/cam_unified/credit/longdelay.py at commit 3968583 (R-HCA terms
removed; nothing else changed).

Structure (Meulemans et al. 2023's linear key-to-door with noisy distractors): at t = 0 the agent takes the key
(a = 0) or not; t = 1 .. T-2 are distractor steps where action 0 yields an apple with probability p_hi, action 1 with
p_lo (reward ``apple`` +- ``noise``); at t = T-1 the door pays ``door`` iff the key was taken. The key decision's only
consequence is the delayed door reward. Therefore, for the key decision:
  * Q_0(a) = door * [a = key] + E[apples], exact gradient = sum_a grad pi_0(a) Q_0(a)
  * COCOA coefficients: w(a, door) = [a = key]/pi_0(key) - 1, w(a, apple) = 0
Per-trajectory key-decision gradient samples of REINFORCE, advantage (exact V_0), COCOA and RUDDER (exact
decomposition, immediate redistributed reward) give bias, variance and SNR as a function of the delay T.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

DT = torch.float64


@dataclass
class LongDelay:
    T: int = 64
    p_hi: float = 0.7
    p_lo: float = 0.3
    apple: float = 0.2
    noise: float = 0.5
    door: float = 1.0
    pi_key: float = 0.5           # policy probability of taking the key at t = 0
    pi_d0: float = 0.5            # distractor policy probability of action 0
    noise0: float = 0.0           # two-point noise on the key step's own immediate reward

    def exact_grad_key(self) -> torch.Tensor:
        """d J / d theta_0 for the two key logits (softmax): grad pi(a) = pi(a)(e_a - pi)."""
        p = torch.tensor([self.pi_key, 1 - self.pi_key], dtype=DT)
        Q = torch.tensor([self.door, 0.0], dtype=DT)       # apples cancel (independent of a_0)
        return p * (Q - (p * Q).sum())

    def sample(self, n: int, gen: torch.Generator) -> dict:
        a0 = (torch.rand(n, generator=gen, dtype=DT) >= self.pi_key).long()          # 0 = key
        D = max(self.T - 2, 0)
        ad = (torch.rand(n, D, generator=gen, dtype=DT) >= self.pi_d0).long()
        p = torch.where(ad == 0, torch.tensor(self.p_hi, dtype=DT), torch.tensor(self.p_lo, dtype=DT))
        apple = torch.rand(n, D, generator=gen, dtype=DT) < p
        sgn = torch.where(torch.rand(n, D, generator=gen, dtype=DT) < 0.5, -1.0, 1.0).to(DT)
        r_d = apple.to(DT) * self.apple + sgn * self.noise
        r_door = (a0 == 0).to(DT) * self.door
        r0 = torch.where(torch.rand(n, generator=gen, dtype=DT) < 0.5, -1.0, 1.0).to(DT) * self.noise0
        return {"a0": a0, "r0": r0, "r_d": r_d, "r_door": r_door}

    def t0_gradients(self, s: dict, w_door_key=None) -> dict:
        """Per-trajectory gradient w.r.t. the two key logits [N, 2] for each estimator. An optional learned
        coefficient replaces the exact w(key, door)."""
        p = torch.tensor([self.pi_key, 1 - self.pi_key], dtype=DT)
        eye = torch.eye(2, dtype=DT)
        glogp = eye[s["a0"]] - p                                           # [N, 2]
        G = s["r0"] + s["r_d"].sum(1) + s["r_door"]
        dpi = p[:, None] * (eye - p[None, :])                              # [a, theta]: d pi(a) / d theta
        Ed = (self.pi_d0 * self.p_hi + (1 - self.pi_d0) * self.p_lo) * self.apple * max(self.T - 2, 0)
        out = {"reinforce": glogp * G[:, None],
               "advantage": glogp * (G - (p[0] * self.door + Ed))[:, None]}
        w_key = (1 / self.pi_key - 1) if w_door_key is None else w_door_key
        w = torch.tensor([w_key, -1.0], dtype=DT)                          # w(a, door) for a = key, no key
        door_occ = s["r_door"]
        # COCOA: grad log pi(A_0) R_0 + sum_a grad pi(a) sum_k w(a, U_k) R_k ; apples have w = 0
        out["cocoa"] = glogp * s["r0"][:, None] + (door_occ[:, None, None] * w[None, :, None] * dpi[None]).sum(1)
        # RUDDER with the exact decomposition g_0 = Q_0(A_0): the credit of A_0 is Q_0(A_0) - V_0
        q_taken = torch.where(s["a0"] == 0, torch.tensor(self.door, dtype=DT), torch.tensor(0.0, dtype=DT))
        out["rudder"] = glogp * (q_taken - p[0] * self.door)[:, None]
        return out

    def stats(self, n: int = 20000, seed: int = 0, **kw) -> dict:
        g = torch.Generator().manual_seed(seed)
        s = self.sample(n, g)
        ex = self.exact_grad_key()
        res = {}
        for k, v in self.t0_gradients(s, **kw).items():
            m = v.mean(0)
            var = float(v.var(0, unbiased=True).sum())
            res[k] = {"bias_norm": float((m - ex).norm()), "se": math.sqrt(var / n), "var": var,
                      "snr": float(ex.norm() ** 2) / max(var, 1e-300), "exact_norm": float(ex.norm())}
        return res
