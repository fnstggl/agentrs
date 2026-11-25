"""Multistep forecasting baselines and a shared evaluation API (spec 6.4 / 6.6(5)).

Every forecaster answers the same question: at ``start`` = t, given the observations O_{<=t}, the outcomes
Y_{<t} of earlier actions and the *recorded* actions A_{t..t+h-1} with their recorded durations, predict the
outcomes Y_{t+h-1} (the consequences of the h-th action) for every horizon h. None may read O_{>t} or Y_{>=t}.

* ``NoChangeBaseline``        -- the last observed value of the same outcome key persists.
* ``TrendPersistenceBaseline``-- the last observed per-step change of the key persists (linear extrapolation).
* ``KeyMeanBaseline``         -- the training mean of the key (climatology).
* ``LinearRegressionBaseline``-- per (key, horizon) ridge regression on the last two values of the key, the mean
  ACTION token of every recorded action up to the horizon, log-durations and the previous action.
* ``WorldModelForecaster``    -- the world model's open-loop rollout (``WorldModel.forecast``).

``evaluate_multistep`` scores any set of forecasters on the same held-out sequences and the same (start, h)
pairs, so error-vs-horizon curves are directly comparable.
"""
from __future__ import annotations

from typing import Dict, Optional, Protocol, Sequence

import torch

from ..interfaces import Likelihood, SequenceBatch

CONTINUOUS = (int(Likelihood.GAUSSIAN), int(Likelihood.STUDENT_T), int(Likelihood.COUNT))


class Forecaster(Protocol):
    def fit(self, batches: Sequence[SequenceBatch]) -> "Forecaster": ...
    def predict(self, batch: SequenceBatch, start: int, horizons: Sequence[int]) -> Dict[int, torch.Tensor]: ...


# ------------------------------------------------------------------------------------------------ history tensors

class KeyVocab:
    """Outcome-key strings -> integer ids (keys missing -> 'q<position>')."""

    def __init__(self):
        self.ids: dict = {}

    def id(self, key) -> int:
        if key not in self.ids:
            self.ids[key] = len(self.ids)
        return self.ids[key]

    def matrix(self, keys: Optional[list], B: int, Q: int) -> torch.Tensor:
        out = torch.full((B, Q), -1, dtype=torch.long)
        for b in range(B):
            for q in range(Q):
                k = keys[b][q] if keys is not None else f"q{q}"
                if k is not None:
                    out[b, q] = self.id(k)
        return out


class _Hist:
    """Per-sequence tensors on CPU: y/m/lik/key [T,B,Q], valid/done [T,B], act pooled [T,B,D], dt [T,B]."""

    def __init__(self, batch: SequenceBatch, vocab: KeyVocab):
        steps = batch.steps
        self.T, self.B = len(steps), steps[0].obs.B
        Q = max(s.outcomes.q.N for s in steps)
        self.Q = Q
        y = torch.zeros(self.T, self.B, Q)
        m = torch.zeros(self.T, self.B, Q, dtype=torch.bool)
        lik = torch.zeros(self.T, self.B, Q, dtype=torch.long)
        key = torch.full((self.T, self.B, Q), -1, dtype=torch.long)
        acts, valid = [], []
        for t, s in enumerate(steps):
            o = s.outcomes
            q = o.q.N
            y[t, :, :q] = o.y.detach().float().cpu()
            m[t, :, :q] = o.y_mask.bool().cpu()
            lik[t, :, :q] = o.lik.long().cpu()
            key[t, :, :q] = vocab.matrix(o.keys, self.B, q)
            am = s.act.mask.float().cpu().unsqueeze(-1)
            acts.append((s.act.x.detach().float().cpu() * am).sum(1) / am.sum(1).clamp(min=1.0))
            valid.append(s.obs.mask.any(-1).cpu() | s.act.mask.any(-1).cpu())
        self.y, self.m, self.lik, self.key = y, m & torch.stack(valid).unsqueeze(-1), lik, key
        self.act = torch.stack(acts)
        self.valid = torch.stack(valid)
        self.done = torch.stack([torch.as_tensor(s.done).bool().cpu().reshape(self.B) for s in steps])
        self.dt = torch.stack([torch.as_tensor(s.dt).float().cpu().reshape(self.B) for s in steps])

    def episode_start(self, start: int) -> torch.Tensor:
        """[B] first step index of the episode containing ``start``."""
        es = torch.zeros(self.B, dtype=torch.long)
        for s in range(start):
            es = torch.where(self.done[s], torch.full_like(es, s + 1), es)
        return es

    def lags(self, start: int, target_key: torch.Tensor, n: int = 2):
        """Last n observed values of ``target_key`` [B,Q] before ``start`` in the current episode.
        -> vals [B,Q,n], step [B,Q,n] (float), ok [B,Q,n] (most recent first)."""
        es = self.episode_start(start)
        B, Q = target_key.shape
        vals = torch.zeros(B, Q, n)
        stp = torch.zeros(B, Q, n)
        ok = torch.zeros(B, Q, n, dtype=torch.bool)
        cnt = torch.zeros(B, Q, dtype=torch.long)
        for s in range(start - 1, -1, -1):
            in_ep = (s >= es).view(B, 1)
            # match target key against step s keys
            km = self.key[s].unsqueeze(1) == target_key.unsqueeze(2)            # [B,Q,Qs]
            km = km & self.m[s].unsqueeze(1) & in_ep.unsqueeze(2) & (target_key >= 0).unsqueeze(2)
            has = km.any(2)
            v = (self.y[s].unsqueeze(1) * km.float()).sum(2) / km.float().sum(2).clamp(min=1.0)
            take = has & (cnt < n)
            for j in range(n):
                put = take & (cnt == j)
                vals[..., j] = torch.where(put, v, vals[..., j])
                stp[..., j] = torch.where(put, torch.full_like(v, float(s)), stp[..., j])
                ok[..., j] = ok[..., j] | put
            cnt = cnt + take.long()
            if bool((cnt >= n).all()):
                break
        return vals, stp, ok

    def target_valid(self, start: int, h: int) -> torch.Tensor:
        """[B,Q] whether Y_{start+h-1} is a valid, observed, continuous target reachable without an episode end."""
        t = start + h - 1
        if t >= self.T:
            return torch.zeros(self.B, self.Q, dtype=torch.bool)
        ok = torch.ones(self.B, dtype=torch.bool)
        for s in range(start, t + 1):
            ok = ok & self.valid[s]
        for s in range(start, t):
            ok = ok & ~self.done[s]
        cont = torch.isin(self.lik[t], torch.tensor(CONTINUOUS))
        return ok.view(-1, 1) & self.m[t] & cont


# ------------------------------------------------------------------------------------------------ baselines

class KeyMeanBaseline:
    """Per-key training mean."""

    def __init__(self):
        self.vocab = KeyVocab()
        self.sum: dict = {}
        self.n: dict = {}

    def fit(self, batches):
        for b in batches:
            h = _Hist(b, self.vocab)
            for k in torch.unique(h.key[h.m]).tolist():
                sel = h.m & (h.key == k)
                self.sum[k] = self.sum.get(k, 0.0) + float(h.y[sel].sum())
                self.n[k] = self.n.get(k, 0) + int(sel.sum())
        return self

    def mean_of(self, key: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(key.shape)
        for k, s in self.sum.items():
            out = torch.where(key == k, torch.tensor(s / max(1, self.n[k])), out)
        return out

    def predict(self, batch, start, horizons):
        h = _Hist(batch, self.vocab)
        return {k: self.mean_of(h.key[start + k - 1]) for k in horizons if start + k - 1 < h.T}


class NoChangeBaseline:
    """Last observed value of the same key persists (training key mean when nothing was observed yet)."""

    def __init__(self):
        self.mean = KeyMeanBaseline()

    @property
    def vocab(self):
        return self.mean.vocab

    def fit(self, batches):
        self.mean.fit(batches)
        return self

    def predict(self, batch, start, horizons):
        h = _Hist(batch, self.vocab)
        out = {}
        for k in horizons:
            t = start + k - 1
            if t >= h.T:
                continue
            vals, _, ok = h.lags(start, h.key[t], 1)
            out[k] = torch.where(ok[..., 0], vals[..., 0], self.mean.mean_of(h.key[t]))
        return out


class TrendPersistenceBaseline:
    """The last observed per-step change of the key persists: y_{t+h-1} = y_last + slope * (t+h-1 - s_last)."""

    def __init__(self):
        self.mean = KeyMeanBaseline()

    @property
    def vocab(self):
        return self.mean.vocab

    def fit(self, batches):
        self.mean.fit(batches)
        return self

    def predict(self, batch, start, horizons):
        h = _Hist(batch, self.vocab)
        out = {}
        for k in horizons:
            t = start + k - 1
            if t >= h.T:
                continue
            vals, stp, ok = h.lags(start, h.key[t], 2)
            slope = torch.where(ok[..., 1], (vals[..., 0] - vals[..., 1]) / (stp[..., 0] - stp[..., 1]).clamp(min=1),
                                torch.zeros_like(vals[..., 0]))
            pred = vals[..., 0] + slope * (t - stp[..., 0])
            out[k] = torch.where(ok[..., 0], pred, self.mean.mean_of(h.key[t]))
        return out


class LinearRegressionBaseline:
    """Per (outcome key, horizon) ridge regression, fitted from streamed sufficient statistics (float64).

    Features for target Y_{t+h-1}: [1, last value, second-last value, their presence flags, mean ACTION token of
    A_{t-1}, then for j < h: mean ACTION token of A_{t+j} and log1p(dt_{t+j} in days)]."""

    def __init__(self, ridge: float = 1.0, horizons: Sequence[int] = (1, 2, 4, 8, 16, 32)):
        self.ridge = ridge
        self.horizons = tuple(horizons)
        self.vocab = KeyVocab()
        self.stats: dict = {}      # (key, h) -> [XtX, Xty]
        self.w: dict = {}
        self.mean = KeyMeanBaseline()

    def _features(self, h: _Hist, start: int, k: int, target_key: torch.Tensor) -> torch.Tensor:
        B, Q = target_key.shape
        vals, _, ok = h.lags(start, target_key, 2)
        prev = h.act[start - 1] if start > 0 else torch.zeros_like(h.act[0])
        if start > 0:
            prev = torch.where(h.episode_start(start).view(-1, 1) <= start - 1, prev, torch.zeros_like(prev))
        fut = [torch.cat([h.act[start + j], torch.log1p(h.dt[start + j]).view(-1, 1)], 1) for j in range(k)]
        rowf = torch.cat([prev] + fut, 1)                                          # [B, F_row]
        f = torch.cat([torch.ones(B, Q, 1), vals, ok.float(), rowf.unsqueeze(1).expand(-1, Q, -1)], 2)
        return f.double()

    def fit(self, batches):
        self.mean.fit(batches)
        for b in batches:
            h = _Hist(b, self.vocab)
            for start in range(h.T):
                for k in self.horizons:
                    t = start + k - 1
                    if t >= h.T:
                        continue
                    tv = h.target_valid(start, k)
                    if not bool(tv.any()):
                        continue
                    X = self._features(h, start, k, h.key[t])
                    for key in torch.unique(h.key[t][tv]).tolist():
                        sel = tv & (h.key[t] == key)
                        Xs, ys = X[sel], h.y[t][sel].double()
                        st = self.stats.get((key, k))
                        if st is None:
                            st = [torch.zeros(X.shape[-1], X.shape[-1], dtype=torch.float64),
                                  torch.zeros(X.shape[-1], dtype=torch.float64)]
                            self.stats[(key, k)] = st
                        st[0] += Xs.T @ Xs
                        st[1] += Xs.T @ ys
        for (key, k), (A, c) in self.stats.items():
            reg = self.ridge * torch.eye(A.shape[0], dtype=torch.float64)
            reg[0, 0] = 1e-6                                                     # do not shrink the intercept
            self.w[(key, k)] = torch.linalg.solve(A + reg, c)
        return self

    def predict(self, batch, start, horizons):
        h = _Hist(batch, self.vocab)
        out = {}
        for k in horizons:
            t = start + k - 1
            if t >= h.T:
                continue
            X = self._features(h, start, k, h.key[t])
            pred = self.mean.mean_of(h.key[t]).double()
            for key in torch.unique(h.key[t]).tolist():
                w = self.w.get((key, k))
                if w is None:
                    continue
                pred = torch.where(h.key[t] == key, X @ w, pred)
            out[k] = pred.float()
        return out


class WorldModelForecaster:
    """Adapter: the world model's open-loop rollout under the shared forecasting API (mixture mean)."""

    def __init__(self, wm, n_samples: int = 8, beliefs_fn=None, seed: int = 0):
        self.wm, self.n_samples, self.beliefs_fn, self.seed = wm, n_samples, beliefs_fn, seed

    def fit(self, batches):          # trained elsewhere (WorldModel.loss); nothing to fit here
        return self

    def predict(self, batch, start, horizons):
        dev = batch.steps[0].obs.x.device
        g = torch.Generator(device=dev).manual_seed(self.seed)
        beliefs = None if self.beliefs_fn is None else self.beliefs_fn(batch)
        f = self.wm.forecast(batch, start, horizons, beliefs=beliefs, n_samples=self.n_samples, generator=g)
        return {k: v["mean"].detach().float().cpu() for k, v in f.items()}


def evaluate_multistep(forecasters: dict, batches: Sequence[SequenceBatch], horizons: Sequence[int],
                       starts: Optional[Sequence[int]] = None) -> dict:
    """Mean squared / absolute error of each forecaster's point prediction per horizon on continuous observed
    outcomes, over identical (sequence, start, h, row, query) targets. Returns {name: {h: {"mse","mae","n"}}}."""
    acc = {n: {h: [0.0, 0.0, 0] for h in horizons} for n in forecasters}
    vocab = KeyVocab()
    for b in batches:
        hist = _Hist(b, vocab)
        st_list = range(hist.T) if starts is None else starts
        for start in st_list:
            hs = [h for h in horizons if start + h - 1 < hist.T]
            if not hs:
                continue
            masks = {h: hist.target_valid(start, h) for h in hs}
            if not any(bool(m.any()) for m in masks.values()):
                continue
            for name, fc in forecasters.items():
                pred = fc.predict(b, start, hs)
                for h in hs:
                    m = masks[h]
                    if not bool(m.any()):
                        continue
                    err = pred[h].float().cpu() - hist.y[start + h - 1]
                    acc[name][h][0] += float((err[m] ** 2).sum())
                    acc[name][h][1] += float(err[m].abs().sum())
                    acc[name][h][2] += int(m.sum())
    return {n: {h: {"mse": v[0] / max(1, v[2]), "mae": v[1] / max(1, v[2]), "n": v[2]} for h, v in d.items()}
            for n, d in acc.items()}
