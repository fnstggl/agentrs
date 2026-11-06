"""Held-out world-model evaluation against explicit baselines.

Baselines (all fit on the same training transitions):
* ``marginal``      - P(event) = training frequency, ignoring state and action.
* ``action_type``   - P(event | action type), Laplace-smoothed frequencies.
* ``persistence``   - next observation = current observation (for state prediction / drift).

Metrics: Bernoulli NLL, Brier, AUROC and ECE per output (12 events, reward, terminal);
diff exact match and next-observation content F1 from greedy generation;
open-loop multi-step drift; and whether ensemble disagreement flags errors.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Sequence

import numpy as np
import torch

from agentrs.actions.schema import Action, try_parse_action
from agentrs.data.diff import EVENT_NAMES, StateDiff, observation_f1
from agentrs.data.schemas import Transition
from agentrs.environments.base import Observation
from agentrs.evaluation.metrics import auroc, bernoulli_nll, brier, expected_calibration_error, reliability_table
from agentrs.models.world_model import N_OUT, WorldModel
from agentrs.training.world_model import WMExample, make_wm_examples

OUT_NAMES = EVENT_NAMES + ["reward", "terminal"]


class FrequencyBaseline:
    def __init__(self, train: Sequence[WMExample], by_action_type: bool):
        self.by = by_action_type
        cnt: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(N_OUT))
        tot: dict[str, int] = defaultdict(int)
        for e in train:
            k = e.action_type if by_action_type else "*"
            cnt[k] += np.asarray(e.targets)
            tot[k] += 1
        self.p = {k: (cnt[k] + 1) / (tot[k] + 2) for k in cnt}
        self.glob = (sum(cnt.values()) + 1) / (sum(tot.values()) + 2)

    def predict(self, e: WMExample) -> np.ndarray:
        k = e.action_type if self.by else "*"
        return self.p.get(k, self.glob)


def _metrics(P: np.ndarray, Y: np.ndarray) -> dict:
    per = {}
    for j, name in enumerate(OUT_NAMES):
        per[name] = {"nll": bernoulli_nll(P[:, j], Y[:, j]), "brier": brier(P[:, j], Y[:, j]),
                     "auroc": auroc(P[:, j], Y[:, j]), "pos_rate": float(Y[:, j].mean())}
    ev = slice(0, len(EVENT_NAMES))
    aurocs = [per[n]["auroc"] for n in EVENT_NAMES if per[n]["auroc"] is not None]
    return {
        "event_nll": float(np.mean([per[n]["nll"] for n in EVENT_NAMES])),
        "event_brier": float(np.mean([per[n]["brier"] for n in EVENT_NAMES])),
        "event_auroc_mean": float(np.mean(aurocs)) if aurocs else None,
        "event_ece": expected_calibration_error(P[:, ev].ravel(), Y[:, ev].ravel()),
        "reward_nll": per["reward"]["nll"], "reward_auroc": per["reward"]["auroc"],
        "terminal_nll": per["terminal"]["nll"], "terminal_auroc": per["terminal"]["auroc"],
        "per_output": per,
    }


def remap_action(action: str, real_obs: Observation, imagined: Observation) -> str:
    """Map an action's target id from the real observation onto the imagined one by content."""
    a = try_parse_action(action)
    if a is None or a.target is None or imagined.element(a.target) is not None:
        return action
    src = real_obs.element(a.target)
    if src is None:
        return action
    for e in imagined.elements:
        if e.role == src.role and e.name == src.name:
            return Action(**{**a.model_dump(), "target": e.eid}).to_str()
    return action


@torch.no_grad()
def evaluate_world_model(wm: WorldModel, train: Sequence[Transition], val: Sequence[Transition],
                         n_generate: int = 60, drift_k: int = 3, n_drift: int = 25, seed: int = 0) -> dict:
    wm.eval()
    rng = random.Random(seed)
    tr_ex = make_wm_examples(wm, train)
    va_ex = make_wm_examples(wm, val)
    Y = np.asarray([e.targets for e in va_ex])
    # --- heads
    P_list, S_list = [], []
    for i in range(0, len(va_ex), 32):
        m, s = wm.predict_heads([e.prompt for e in va_ex[i : i + 32]])
        P_list.append(m.cpu().numpy())
        S_list.append(s.cpu().numpy())
    P, S = np.concatenate(P_list), np.concatenate(S_list)
    res: dict = {"n_val": len(va_ex), "model": _metrics(P, Y)}
    for name, by in (("marginal", False), ("action_type", True)):
        b = FrequencyBaseline(tr_ex, by)
        res[name] = _metrics(np.stack([b.predict(e) for e in va_ex]), Y)
    res["model"]["reliability_events"] = reliability_table(P[:, : len(EVENT_NAMES)].ravel(),
                                                           Y[:, : len(EVENT_NAMES)].ravel())
    # uncertainty quality: does ensemble std flag wrong predictions?
    err = (np.abs(P - Y) > 0.5).any(1).astype(float)
    res["uncertainty_flags_error_auroc"] = auroc(S.mean(1), err)
    res["mean_epistemic_std"] = float(S.mean())
    # --- teacher-forced diff NLL
    nlls = []
    for i in range(0, len(va_ex), 32):
        b = va_ex[i : i + 32]
        nll, cnt = wm.backbone.token_nll([e.prompt for e in b], [wm.diff_target(e.diff_text) for e in b])
        nlls += (nll / cnt).tolist()
    res["diff_nll_per_token"] = float(np.mean(nlls))
    # --- generation: next-state prediction
    sub = rng.sample(va_ex, min(n_generate, len(va_ex)))
    exact, f1_m, f1_p = [], [], []
    by_type: dict[str, list[float]] = defaultdict(list)
    for e in sub:
        t = e.transition
        diff, _ = wm.sample_diffs(e.prompt, n=1, greedy=True)[0]
        pred_obs = StateDiff.from_text(diff).apply(t.observation)
        exact.append(float(diff.strip() == (t.diff_text or "(no change)").strip()))
        f1 = observation_f1(pred_obs, t.next_observation)
        f1_m.append(f1)
        f1_p.append(observation_f1(t.observation, t.next_observation))
        by_type[t.action_type].append(f1)
    res["generation"] = {"n": len(sub), "diff_exact_match": float(np.mean(exact)),
                         "next_obs_f1_model": float(np.mean(f1_m)), "next_obs_f1_persistence": float(np.mean(f1_p)),
                         "next_obs_f1_by_action_type": {k: float(np.mean(v)) for k, v in by_type.items()}}
    # --- multi-step open-loop drift
    res["drift"] = drift_eval(wm, val, k=drift_k, n=n_drift, seed=seed)
    wm.train()
    return res


@torch.no_grad()
def drift_eval(wm: WorldModel, val: Sequence[Transition], k: int = 3, n: int = 25, seed: int = 0) -> dict:
    by_ep: dict[str, list[Transition]] = defaultdict(list)
    for t in val:
        by_ep[t.episode_id].append(t)
    eps = [sorted(v, key=lambda t: t.step) for v in by_ep.values() if len(v) >= k]
    random.Random(seed).shuffle(eps)
    eps = eps[:n]
    model_f1 = np.zeros(k)
    pers_f1 = np.zeros(k)
    for ep in eps:
        imag = ep[0].observation
        hist: list[tuple[str, str]] = []
        for j in range(k):
            t = ep[j]
            act = remap_action(t.action, t.observation, imag)
            diff, _ = wm.sample_diffs(wm.prompt(t.goal, imag, act, hist), n=1, greedy=True)[0]
            imag = StateDiff.from_text(diff).apply(imag)
            hist.append((act, "error: ..." if "error:" in diff else "ok"))
            model_f1[j] += observation_f1(imag, t.next_observation)
            pers_f1[j] += observation_f1(ep[0].observation, t.next_observation)
    m = max(1, len(eps))
    return {"n_episodes": len(eps), "k": k, "f1_model_by_step": (model_f1 / m).tolist(),
            "f1_persistence_by_step": (pers_f1 / m).tolist()}

