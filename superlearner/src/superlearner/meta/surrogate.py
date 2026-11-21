"""The Meta-Researcher's trainable research policy: a learned model of which proposals are worth an experiment.

`Surrogate` is an ensemble of small neural networks trained by gradient descent on the discovery ledger. Input:
the frozen language model's embedding of the proposal (hypothesis + change summary) and structural features
(where the change is, how large, which proposal source). Outputs: the expected difference in the primary measure
against the champion, and whether the candidate will pass verification. Both heads are trained on real
experimental outcomes; before there are any, every proposal scores the same and selection is uniform.

`select` ranks proposals by an upper confidence bound (mean + beta * ensemble spread, weighted by the predicted
probability of being valid). `fit` is called after every experiment, so the next choice depends on what the
previous experiments measured. `loo_report` says how well it predicts held-out experiments (leave-one-out), which
is the honest measure of whether the meta-level has learned anything yet.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

SOURCES = ("rule_space", "llm", "learned_rule", "manual")
SLOTS = ("advantages", "policy_loss", "value_loss", "entropy_loss", "aux_loss", "total_loss", "replay_loss", "model",
         "curriculum", "other")
N_STRUCT = len(SOURCES) + len(SLOTS) + 4


def structural_features(p: dict) -> np.ndarray:
    v = np.zeros(N_STRUCT, np.float32)
    if p.get("source") in SOURCES:
        v[SOURCES.index(p["source"])] = 1.0
    for s in p.get("slots", []) or ["other"]:
        v[len(SOURCES) + (SLOTS.index(s) if s in SLOTS else len(SLOTS) - 1)] = 1.0
    o = len(SOURCES) + len(SLOTS)
    v[o] = np.log1p(p.get("lines_changed", 0)) / 6.0
    v[o + 1] = len(p.get("slots", []) or []) / 4.0
    v[o + 2] = float(p.get("generation", 0)) / 5.0
    v[o + 3] = 1.0
    return v


class _Net(nn.Module):
    def __init__(self, d_in: int, hid: int = 48):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d_in, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh())
        self.effect = nn.Linear(hid, 1)
        self.valid = nn.Linear(hid, 1)

    def forward(self, x):
        h = self.body(x)
        return self.effect(h).squeeze(-1), self.valid(h).squeeze(-1)


class Surrogate:
    def __init__(self, embed=None, n_models: int = 5, seed: int = 0, emb_dim: int = 512, pca: int = 16):
        self.embed = embed                 # callable: list[str] -> [n, emb_dim]; None = structural features only
        self.n_models, self.seed, self.emb_dim, self.pca = n_models, seed, emb_dim, pca
        self.models: list[_Net] = []
        self.X: list[np.ndarray] = []
        self.y: list[float] = []
        self.valid: list[float] = []
        self.proj: np.ndarray | None = None
        self.scale = 1.0
        self.updates = 0

    def features(self, proposals: list[dict]) -> np.ndarray:
        s = np.stack([structural_features(p) for p in proposals])
        if self.embed is None:
            return s
        e = np.asarray(self.embed([(p.get("hypothesis", "") + " | " + p.get("summary", ""))[:1500] for p in proposals]), np.float32)
        if self.proj is None:
            rng = np.random.default_rng(self.seed)
            self.proj = (rng.normal(size=(e.shape[1], self.pca)) / np.sqrt(self.pca)).astype(np.float32)
        return np.concatenate([s, e @ self.proj], 1)

    def add(self, proposal: dict, effect: float | None, valid: bool) -> None:
        self.X.append(self.features([proposal])[0])
        self.y.append(0.0 if effect is None else float(effect))
        self.valid.append(float(valid))

    def fit(self, steps: int = 300, lr: float = 3e-3) -> dict:
        n = len(self.X)
        if n < 3:
            self.models = []
            return {"n": n, "fitted": False}
        X = torch.tensor(np.stack(self.X))
        y = torch.tensor(self.y, dtype=torch.float32)
        ok = torch.tensor(self.valid, dtype=torch.float32)
        self.scale = float(y[ok > 0].abs().max().clamp(min=1e-3)) if (ok > 0).any() else 1.0
        self.models = []
        losses = []
        for k in range(self.n_models):
            torch.manual_seed(self.seed * 1000 + k)
            net = _Net(X.shape[1])
            opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-3)
            g = torch.Generator().manual_seed(self.seed * 7919 + k)
            idx = torch.randint(0, n, (n,), generator=g)          # bootstrap resample per member
            for _ in range(steps):
                e, v = net(X[idx])
                w = ok[idx]
                loss = (((e - y[idx] / self.scale) ** 2) * w).sum() / w.sum().clamp(min=1) \
                    + nn.functional.binary_cross_entropy_with_logits(v, ok[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
            losses.append(float(loss.detach()))
            self.models.append(net)
        self.updates += 1
        return {"n": n, "fitted": True, "loss": float(np.mean(losses)), "updates": self.updates}

    def predict(self, proposals: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(mean effect, ensemble spread, probability of passing verification) per proposal."""
        if not self.models:
            n = len(proposals)
            return np.zeros(n), np.ones(n), np.full(n, 0.5)
        X = torch.tensor(self.features(proposals))
        with torch.no_grad():
            outs = [m(X) for m in self.models]
        e = np.stack([o[0].numpy() for o in outs]) * self.scale
        v = torch.sigmoid(torch.stack([o[1] for o in outs])).numpy()
        return e.mean(0), e.std(0), v.mean(0)

    def select(self, proposals: list[dict], k: int = 1, beta: float = 1.0, rng: np.random.Generator | None = None) -> list[int]:
        rng = rng or np.random.default_rng(self.seed + len(self.X))
        if not self.models:
            return [int(i) for i in rng.permutation(len(proposals))[:k]]
        mu, sd, pv = self.predict(proposals)
        score = pv * (mu + beta * sd) + 1e-6 * rng.random(len(proposals))
        return [int(i) for i in np.argsort(-score)[:k]]

    def loo_report(self) -> dict:
        """Leave-one-out: train on all experiments but one, predict it. Rank correlation over valid experiments."""
        n = len(self.X)
        idx = [i for i in range(n) if self.valid[i] > 0]
        if len(idx) < 5:
            return {"n": n, "n_valid": len(idx), "spearman": None}
        preds = []
        for i in idx:
            s = Surrogate(None, 3, self.seed)
            s.X = [x for j, x in enumerate(self.X) if j != i]
            s.y = [v for j, v in enumerate(self.y) if j != i]
            s.valid = [v for j, v in enumerate(self.valid) if j != i]
            s.fit(150)
            X = torch.tensor(self.X[i][None])
            with torch.no_grad():
                preds.append(float(np.mean([m(X)[0].item() for m in s.models])) * s.scale if s.models else 0.0)
        y = np.array([self.y[i] for i in idx])
        p = np.array(preds)
        ry, rp = np.argsort(np.argsort(y)), np.argsort(np.argsort(p))
        rho = float(np.corrcoef(ry, rp)[0, 1]) if y.std() > 0 and p.std() > 0 else 0.0
        return {"n": n, "n_valid": len(idx), "spearman": round(rho, 3)}

    def state_dict(self) -> dict:
        return {"X": [x.tolist() for x in self.X], "y": self.y, "valid": self.valid, "updates": self.updates,
                "proj": None if self.proj is None else self.proj.tolist()}

    def load_state_dict(self, s: dict) -> None:
        self.X = [np.asarray(x, np.float32) for x in s.get("X", [])]
        self.y, self.valid = list(s.get("y", [])), list(s.get("valid", []))
        self.updates = s.get("updates", 0)
        self.proj = None if s.get("proj") is None else np.asarray(s["proj"], np.float32)
        self.fit()
