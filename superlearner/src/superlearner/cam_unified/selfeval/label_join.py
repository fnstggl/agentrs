"""Join released verified labels (trainer-only LabelStore) to archived decisions (redesign plan §3 W8; spec 4.1, 11.2).

    join(label_store, archive_events) -> LabelBatch

* A label (envs/labels.LabelRecord, or envs/experience.OutcomeLabel) resolves one EPISODE of one environment:
  key (env_id, episode). Every archived decision of that episode (a REAL EventRecord with an action and
  extra["kind"] in (None, "decision")) becomes one training example: ALL prefixes of the resolved episode are
  supervised by its final label, weighted 1 / (number of prefixes) so long episodes do not dominate.
* PENDING outcomes are never labels: OutcomeLabel(status="pending") and any status outside
  (success, failure, uncertain) are dropped; episodes without a released label contribute nothing (they are not
  failures, SE-03). UNCERTAIN labels keep their rows with the UNCERTAIN status code (no success / time target).
* Episode identity. Archived records carry the agent's episode_id (a string); labels carry the operator's integer
  episode counter. The join resolves a record's label episode in this order:
    1. ``episode_of(record)`` when given;
    2. ``record.extra["label_episode"]`` when the canonical loop stamped it (change request to W1/W2);
    3. the ordinal of the record's episode_id among the env's archived episodes (ordered by first appearance,
       i.e. archive order / t_sim) plus ``episode_base`` (ExperienceOperator counts the first episode after its
       initial adapter reset as 1, so pass episode_base=1 there).
* Targets per row: status code (selfeval.evaluator PENDING/SUCCESS/FAILURE/UNCERTAIN), days until resolution
  (t_resolved - t_sim, >= 0; censored rows use the elapsed episode time), the episode's normalised verified utility,
  the verified remaining utility of the episode from this decision on (V_L target: utilities of the episode's labels
  of the joined kinds that resolve at or after the decision), the verified value, the weight.
* `LabelBatch.with_features(R, goal)` attaches CAM features (R [N,S,D] reasoned latents of the decisions, goal
  TokenSet) for `RewardModel.losses`. Nothing in a LabelBatch ever reaches an actor input: it is built and consumed by
  the learner only.

`split_by_episode` gives held-out episodes (calibration is measured on episodes never trained on).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from typing import Callable, Iterable, Optional, Sequence

import torch
from torch import Tensor

from .evaluator import FAILURE, PENDING, SUCCESS, UNCERTAIN, ProspectTargets

STATUS_CODE = {"success": SUCCESS, "failure": FAILURE, "uncertain": UNCERTAIN}


@dataclass
class LabelBatch:
    event_ids: list
    env_ids: list
    episodes: list                # label episode (int) of each row
    episode_ids: list             # archived episode id (str) of each row
    steps: list
    t_sim: Tensor                 # [N] decision time (days)
    status: Tensor                # [N] long status code
    time_days: Tensor             # [N] days from the decision to resolution (or to the censoring time)
    utility: Tensor               # [N] episode's normalised verified utility (nan = unknown)
    remaining_utility: Tensor     # [N] verified utility still to come after the decision (V_L target; nan = unknown)
    value: Tensor                 # [N] verified objective quantity (nan = unknown)
    weight: Tensor                # [N] 1 / prefixes of the episode
    horizon_days: Tensor          # [N] horizon input of Phi (0 = "by the objective's resolution")
    lifetime_frac: Tensor         # [N] fraction of the episode's decisions already taken (V_L context)
    label_index: list             # [N] index into `labels`
    labels: list                  # the joined label records (learner-side only)
    R: Optional[Tensor] = None    # [N,S,D] features (attached by with_features)
    goal: object = None           # TokenSet [N,Ng,D] | None
    stats: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.event_ids)

    def prospect_targets(self, device=None) -> ProspectTargets:
        d = device or self.status.device
        return ProspectTargets(self.status.to(d), self.time_days.to(d), magnitude=self.value.to(d),
                               weight=self.weight.to(d))

    def lifetime_ctx(self, device=None) -> Tensor:
        d = device or self.t_sim.device
        f = self.lifetime_frac.to(d)
        return torch.stack([f, 1 - f, torch.log1p(self.t_sim.to(d).clamp_min(0)), torch.ones_like(f)], -1)

    def with_features(self, R: Tensor, goal=None) -> "LabelBatch":
        if R.shape[0] != len(self):
            raise ValueError(f"features for {R.shape[0]} rows, batch has {len(self)}")
        return replace(self, R=R, goal=goal)

    def select(self, idx: Sequence[int]) -> "LabelBatch":
        idx = list(idx)
        it = torch.tensor(idx, dtype=torch.long)
        pick = lambda xs: [xs[i] for i in idx]
        R = None if self.R is None else self.R[it.to(self.R.device)]
        goal = None
        if self.goal is not None:
            goal = self.goal.select(it.to(self.goal.x.device))
        return LabelBatch(pick(self.event_ids), pick(self.env_ids), pick(self.episodes), pick(self.episode_ids),
                          pick(self.steps), self.t_sim[it], self.status[it], self.time_days[it], self.utility[it],
                          self.remaining_utility[it], self.value[it], self.weight[it], self.horizon_days[it],
                          self.lifetime_frac[it], pick(self.label_index), self.labels, R, goal, dict(self.stats))

    def episode_keys(self) -> list:
        return sorted({(e, k) for e, k in zip(self.env_ids, self.episodes)})


def _is_decision(r) -> bool:
    if getattr(r, "action", None) is None:
        return False
    if (r.extra or {}).get("kind") not in (None, "decision"):
        return False
    prov = getattr(r, "provenance", None)
    return prov is None or getattr(prov, "value", prov) == "real"


def _iter_events(archive_events) -> list:
    if hasattr(archive_events, "scan") and callable(archive_events.scan):
        out = []
        for chunk in archive_events.scan():
            out.extend(chunk)
        return out
    return list(archive_events)


def _iter_labels(label_store) -> list:
    if hasattr(label_store, "iterate") and callable(label_store.iterate):
        return list(label_store.iterate())
    return list(label_store)


def _status(lab) -> Optional[str]:
    s = str(getattr(lab, "status", "")).lower()
    return s if s in STATUS_CODE else None            # "pending" and anything unknown are not labels


def join(label_store, archive_events, *, episode_of: Optional[Callable] = None, episode_base: int = 0,
         kinds: Sequence[str] = ("terminal",), only_before_resolution: bool = True,
         device=None) -> LabelBatch:
    """Join labels to archived decisions by (env_id, episode). See the module docstring."""
    labels = [l for l in _iter_labels(label_store) if _status(l) is not None
              and getattr(l, "kind", "terminal") in tuple(kinds)]
    events = [r for r in _iter_events(archive_events) if _is_decision(r)]
    # archived episodes per env, in order of first appearance
    order: dict = {}
    first_t: dict = {}
    by_ep: dict = {}
    for i, r in enumerate(events):
        key = (r.env_id, r.episode_id)
        if key not in by_ep:
            by_ep[key] = []
            first_t[key] = (float(r.t_sim), i)
            order.setdefault(r.env_id, []).append(r.episode_id)
        by_ep[key].append(r)
    ep_num: dict = {}
    for env, eps in order.items():
        eps = sorted(eps, key=lambda e: first_t[(env, e)])
        for k, e in enumerate(eps):
            ep_num[(env, e)] = k + int(episode_base)

    def label_episode(r) -> int:
        if episode_of is not None:
            return int(episode_of(r))
        le = (r.extra or {}).get("label_episode")
        if le is not None:
            return int(le)
        return ep_num[(r.env_id, r.episode_id)]

    rows_of: dict = {}
    for key, recs in by_ep.items():
        for r in recs:
            rows_of.setdefault((r.env_id, label_episode(r)), []).append(r)
    lab_by: dict = {}
    for li, l in enumerate(labels):
        lab_by.setdefault((l.env_id, int(l.episode)), []).append(li)
    cols = {k: [] for k in ("event_ids", "env_ids", "episodes", "episode_ids", "steps", "t_sim", "status",
                            "time_days", "utility", "remaining", "value", "weight", "frac", "label_index")}
    n_lab_joined, n_lab_unmatched, n_after = 0, 0, 0
    nan = float("nan")
    for key, lis in sorted(lab_by.items()):
        recs = sorted(rows_of.get(key, []), key=lambda r: (float(r.t_sim), int(r.step)))
        if not recs:
            n_lab_unmatched += len(lis)
            continue
        term = [li for li in lis if getattr(labels[li], "kind", "terminal") == "terminal"]
        main = term[0] if term else lis[0]
        L = labels[main]
        t_res = getattr(L, "t_resolved_days", None)
        if only_before_resolution and t_res is not None:
            keep = [r for r in recs if float(r.t_sim) <= float(t_res) + 1e-9]
            n_after += len(recs) - len(keep)
            recs = keep
        if not recs:
            continue
        n_lab_joined += 1
        t_end = max(float(r.t_sim) for r in recs)
        n = len(recs)
        code = STATUS_CODE[_status(L)]
        for j, r in enumerate(recs):
            ts = float(r.t_sim)
            if t_res is not None:
                td = max(0.0, float(t_res) - ts)
            else:
                td = max(0.0, t_end - ts)              # censoring time: the episode's last decision
            rem = 0.0
            known = False
            for li in lis:
                l2 = labels[li]
                u2 = getattr(l2, "utility", None)
                tr2 = getattr(l2, "t_resolved_days", None)
                if u2 is None or (isinstance(u2, float) and not math.isfinite(u2)):
                    continue
                if tr2 is None or float(tr2) + 1e-9 >= ts:
                    rem += float(u2)
                    known = True
            u = getattr(L, "utility", None)
            v = getattr(L, "value", None)
            cols["event_ids"].append(r.event_id)
            cols["env_ids"].append(r.env_id)
            cols["episodes"].append(int(key[1]))
            cols["episode_ids"].append(r.episode_id)
            cols["steps"].append(int(r.step))
            cols["t_sim"].append(ts)
            cols["status"].append(code)
            cols["time_days"].append(td)
            cols["utility"].append(nan if u is None else float(u))
            cols["remaining"].append(rem if known else nan)
            cols["value"].append(nan if v is None else float(v))
            cols["weight"].append(1.0 / n)
            cols["frac"].append(j / max(n, 1))
            cols["label_index"].append(main)
    T = lambda xs, dt=torch.float32: torch.tensor(xs, dtype=dt, device=device)
    N = len(cols["event_ids"])
    return LabelBatch(cols["event_ids"], cols["env_ids"], cols["episodes"], cols["episode_ids"], cols["steps"],
                      T(cols["t_sim"]), T(cols["status"], torch.long), T(cols["time_days"]), T(cols["utility"]),
                      T(cols["remaining"]), T(cols["value"]), T(cols["weight"]), torch.zeros(N, device=device),
                      T(cols["frac"]), cols["label_index"], labels,
                      stats={"labels": len(labels), "labels_joined": n_lab_joined,
                             "labels_unmatched": n_lab_unmatched, "rows": N, "rows_after_resolution": n_after,
                             "decisions_archived": len(events), "episodes_archived": len(by_ep)})


def split_by_episode(lb: LabelBatch, held_out_frac: float = 0.3, seed: int = 0) -> tuple:
    """(train, held_out) LabelBatches with disjoint (env_id, episode) keys."""
    keys = lb.episode_keys()
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_h = max(1, int(round(held_out_frac * len(keys)))) if len(keys) > 1 else 0
    held = set(keys[:n_h])
    tr = [i for i, k in enumerate(zip(lb.env_ids, lb.episodes)) if k not in held]
    ho = [i for i, k in enumerate(zip(lb.env_ids, lb.episodes)) if k in held]
    return lb.select(tr), lb.select(ho)
