"""cam_live.credit.learner — the continual learner: three timescales, event-driven (spec Part 2, 10.2; NA-08, CL-02,
CL-03, CL-08).

:class:`ContinualLearner` owns the optimizers (one per parameter group: ``experiential``, ``heads``,
``semantic_adapters`` — never ``reward``), per-group clipping, learning-rate schedules, replay sampling, the held-out
validation split, loss balancing and the learning records. It contains no loss mathematics: it iterates over the
``learning_api.LossProvider`` objects registered with it.

Timescales
    fast       :meth:`fast_update` — the Bayesian belief update of ``nn.fast.FastBeliefState``; no gradient, no
               persistent parameter changes.
    frequent   experiential + heads. Triggered by *matured informative outcomes*: :meth:`poll` counts the independent
               outcome units waiting in the store's re-training queue (one unit per top-level decision, however many
               correlated rows it produced) and returns an ``UpdatePlan`` only when enough have accumulated. Nothing
               here looks at a wall clock.
    slow       semantic adapters, every ``adapter_every`` applied frequent updates, on batches encoded with a graph
               into the adapters (newest ``grad_last`` context items, under the backbone's gradient checkpointing).
               A slow step that makes held-out validation worse beyond a tolerance is reverted.

Design decisions that answer failures of earlier projects
    * Gradients are taken with ``torch.autograd.grad`` with respect to the learner's own parameter lists, one
      provider at a time. The ``reward`` group therefore cannot receive a gradient from a learner step, and each
      provider's share of the experiential gradient norm is *measured*, logged, and capped (``max_share``).
    * Clipping is per parameter group (a single global clip scaled core updates by 1e-4 before).
    * A learning record states which groups actually changed (hashes), by how much (delta norms), and whether
      behaviour on a fixed probe batch changed. An update that changed weights but not behaviour is flagged.
"""
from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from ..config import CamLiveConfig
from ..contracts import Clock, ExperienceRecord, LearningUpdateRecord, WallClock, new_id
from ..learning_api import (TIMESCALE_FREQUENT, TIMESCALE_SLOW, DecisionBatch, LossProvider, UpdatePlan,
                            encode_decisions)
from ..nn.api import GROUP_ADAPTERS, GROUP_BACKBONE_FROZEN, GROUP_EXPERIENTIAL, GROUP_HEADS, GROUP_REWARD, LatentModelAPI
from .replay import ANY_LOSS

LEARNER_GROUPS = (GROUP_EXPERIENTIAL, GROUP_HEADS, GROUP_ADAPTERS)
TIMESCALE_FAST = "fast"
VAL_PREFIX = "val:"
#: test seam: called with ("frequent" | "slow", phase) inside an update (a subprocess test kills the process there)
FAULT_HOOK: Optional[Callable[[str, str], None]] = None

ProbeFn = Callable[[LatentModelAPI, DecisionBatch, Any], Dict[str, Tensor]]


@dataclass
class LearnerSettings:
    """Learner section of the configuration (``cfg.section("credit")["learner"]``); optimisation defaults come from
    ``cfg.optim``."""
    batch_size: int = 16
    slow_batch_size: int = 4
    grad_last: int = 8                   # newest context items that get an adapter graph in a slow batch
    adapter_every: int = 8
    max_share: float = 0.9               # cap on one provider's share of the experiential gradient norm
    min_independent_outcomes: int = 4    # matured independent outcome units needed to trigger frequent learning
    min_surprise: float = 0.0            # an outcome with a registered prediction counts only above this NLL
    steps_per_trigger: int = 1
    max_steps: int = 8
    fresh_fraction: float = 0.25         # share of a batch reserved for the experience that triggered the update
    val_fraction: float = 0.15
    val_split: str = "objective"         # "objective": whole objectives are held out; "record": single records
    val_batch: int = 32
    probe_size: int = 8
    probe_tol: float = 1e-7
    strata: Optional[Dict[str, float]] = None      # replay strata weights (None = the store's default mix)
    provenance: Optional[str] = None
    warmup_steps: int = 0
    decay_steps: int = 0                 # 0 = constant after warm-up; else cosine decay over this many steps
    min_lr_factor: float = 0.1
    with_next: bool = False              # also encode H_{t+1} (providers that need b_next)
    slow_guard: bool = True
    slow_guard_tol: float = 0.05         # tolerated relative increase of the mean validation loss after a slow step
    betas: Tuple[float, float] = (0.9, 0.999)
    seed: int = 0

    @classmethod
    def from_cfg(cls, cfg: CamLiveConfig, **over: Any) -> "LearnerSettings":
        d = dict(batch_size=cfg.optim.batch_size, adapter_every=cfg.optim.adapter_every, seed=cfg.seed)
        d.update(cfg.section("credit").get("learner", {}))
        d.update(over)
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise KeyError(f"unknown learner settings {sorted(unknown)}")
        if "betas" in d:
            d["betas"] = tuple(d["betas"])
        return cls(**d)


def take(batch: DecisionBatch, rows: Sequence[int]) -> DecisionBatch:
    """Rows of an encoded batch (shares the graph of the full batch)."""
    idx = torch.tensor(list(rows), dtype=torch.long, device=batch.b.device)

    def sel(t: Optional[Tensor]) -> Optional[Tensor]:
        return None if t is None else t.index_select(0, idx)

    return DecisionBatch(records=[batch.records[i] for i in rows], b=sel(batch.b), h=sel(batch.h),
                         pad_mask=sel(batch.pad_mask), e_a=sel(batch.e_a), a_mask=sel(batch.a_mask), e_g=sel(batch.e_g),
                         z=sel(batch.z), chosen=sel(batch.chosen), propensity=sel(batch.propensity),
                         behaviour_probs=sel(batch.behaviour_probs), b_next=sel(batch.b_next),
                         has_next=sel(batch.has_next), duration_s=sel(batch.duration_s), ts=sel(batch.ts))


def encode_with_adapter_grad(model: LatentModelAPI, store: Any, records: Sequence[ExperienceRecord], *,
                             grad_last: Optional[int] = None, with_next: bool = False) -> DecisionBatch:
    """``learning_api.encode_decisions(..., with_grad_semantic=True)`` with the adapter graph restricted to the
    newest ``grad_last`` context items when the model supports it (``CamLiveModel.tokenize(grad_last=)``; change
    request A-3 / D-8). Candidate and goal texts always carry the adapter graph."""
    if grad_last is None or "grad_last" not in inspect.signature(model.tokenize).parameters:
        return encode_decisions(model, store, records, with_next=with_next, with_grad_semantic=True)
    records = list(records)
    with torch.enable_grad():
        tokens = model.tokenize([store.items(r.context_item_refs) for r in records], with_grad=True, grad_last=grad_last)
        out = model.core_forward(tokens, None)
        e_a, a_mask = model.encode_actions([[c.action for c in r.decision.candidates] or [r.decision.chosen]
                                            for r in records], with_grad=True)
        e_g = model.encode_goals([r.goal_text for r in records], with_grad=True)
        z = model.decision_latent(out.b, e_a, e_g)
        dev, n = out.b.device, e_a.shape[1]
        chosen = torch.tensor([min(r.decision.chosen_index, n - 1) for r in records], device=dev, dtype=torch.long)
        probs = torch.zeros(len(records), n, device=dev)
        for i, r in enumerate(records):
            p = r.decision.probabilities or [1.0]
            probs[i, :len(p)] = torch.tensor(p[:n], device=dev)
        b_next = has_next = None
        if with_next:
            has = [bool(r.post_item_refs) for r in records]
            nxt = model.tokenize([store.items(r.post_item_refs if h else r.context_item_refs)
                                  for r, h in zip(records, has)], with_grad=True, grad_last=grad_last)
            b_next = model.core_forward(nxt, None).b
            has_next = torch.tensor(has, device=dev)
    return DecisionBatch(records=records, b=out.b, h=out.h, pad_mask=tokens.pad_mask, e_a=e_a, a_mask=a_mask, e_g=e_g,
                         z=z, chosen=chosen, propensity=probs.gather(1, chosen[:, None]).squeeze(1),
                         behaviour_probs=probs, b_next=b_next, has_next=has_next,
                         duration_s=torch.tensor([r.duration_s for r in records], dtype=torch.float64, device=dev),
                         ts=torch.tensor([r.decision.ts for r in records], dtype=torch.float64, device=dev))


def records_only_batch(model: LatentModelAPI, records: Sequence[ExperienceRecord]) -> DecisionBatch:
    """A batch that carries the records but no latents (zero-width tensors), for providers that declare
    ``needs_latents = False`` because they encode whole trajectories themselves."""
    n, dev = len(records), model.device
    e = torch.zeros(n, 0, device=dev)
    return DecisionBatch(records=list(records), b=e, h=torch.zeros(n, 0, 0, device=dev),
                         pad_mask=torch.zeros(n, 0, dtype=torch.bool, device=dev), e_a=torch.zeros(n, 0, 0, device=dev),
                         a_mask=torch.zeros(n, 0, dtype=torch.bool, device=dev), e_g=e, z=torch.zeros(n, 0, 0, device=dev),
                         chosen=torch.zeros(n, dtype=torch.long, device=dev), propensity=torch.ones(n, device=dev),
                         behaviour_probs=e,
                         duration_s=torch.tensor([r.duration_s for r in records], dtype=torch.float64, device=dev),
                         ts=torch.tensor([r.decision.ts for r in records], dtype=torch.float64, device=dev))


def _norm(ts: Sequence[Optional[Tensor]]) -> float:
    sq = [t.detach().float().pow(2).sum() for t in ts if t is not None]
    return float(torch.stack(sq).sum().sqrt()) if sq else 0.0


def latent_probe(model: LatentModelAPI, batch: DecisionBatch, store: Any) -> Dict[str, Tensor]:
    """Built-in probe: the belief and the decision latent of the executed candidate."""
    return {"latent_b": batch.b, "latent_z": batch.z_chosen}


class ContinualLearner:
    """See the module docstring. ``credit`` (a ``providers.CreditAssigner``) is optional: when given, credit is
    re-assigned for the experience that triggered an update before the update, and the change is recorded."""

    def __init__(self, model: LatentModelAPI, store: Any, cfg: CamLiveConfig, *, clock: Optional[Clock] = None,
                 settings: Optional[LearnerSettings] = None, log_path: Optional[str] = None, credit: Any = None):
        self.model, self.store, self.cfg = model, store, cfg
        self.clock = clock or WallClock()
        self.settings = settings or LearnerSettings.from_cfg(cfg)
        self.log_path, self.credit = log_path, credit
        self.providers: List[LossProvider] = []
        self.probes: Dict[str, ProbeFn] = {"latent": latent_probe}
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
        self._opt_ids: Dict[str, set] = {g: set() for g in LEARNER_GROUPS}
        self.lr = {GROUP_EXPERIENTIAL: cfg.optim.lr_experiential, GROUP_HEADS: cfg.optim.lr_heads,
                   GROUP_ADAPTERS: cfg.optim.lr_adapters}
        self.clip = {GROUP_EXPERIENTIAL: cfg.optim.clip_experiential, GROUP_HEADS: cfg.optim.clip_heads,
                     GROUP_ADAPTERS: cfg.optim.clip_adapters}
        self.counters = {"frequent": 0, "slow": 0, "fast": 0, "skipped": 0, "slow_reverted": 0, "triggers": 0}
        self.sched_step = {g: 0 for g in LEARNER_GROUPS}
        self.probe_ids: List[str] = []
        self.elig_seq: Dict[str, Any] = {}
        self.trigger_state: Dict[str, Any] = {}
        self.history: List[LearningUpdateRecord] = []
        self.share_log: List[Dict[str, float]] = []
        self.fast_state = model.new_fast_state() if hasattr(model, "new_fast_state") else None
        self._hash_cache: Tuple[Any, Dict[str, str]] = (None, {})
        self._probe_cache: Tuple[Any, Dict[str, Tensor]] = (None, {})
        self._sync_optimizers()

    # ------------------------------------------------------------------------------------------ registration
    def register(self, provider: LossProvider) -> LossProvider:
        """Add a loss. Heads the provider registered on the model are picked up by the ``heads`` optimizer."""
        for attr in ("name", "timescale", "weight", "eligible", "loss"):
            if not hasattr(provider, attr):
                raise TypeError(f"{type(provider).__name__} is not a LossProvider (missing {attr})")
        if provider.timescale not in (TIMESCALE_FREQUENT, TIMESCALE_SLOW):
            raise ValueError(f"unknown timescale {provider.timescale!r}")
        if any(p.name == provider.name for p in self.providers):
            raise ValueError(f"a provider named {provider.name!r} is already registered")
        self.providers.append(provider)
        self._sync_optimizers()
        return provider

    def add_probe(self, name: str, fn: ProbeFn) -> None:
        """A behavioural probe: named tensors (predictions, action probabilities ...) on the fixed probe batch."""
        self.probes[name] = fn
        self._probe_cache = (None, {})

    def provider(self, name: str) -> LossProvider:
        return next(p for p in self.providers if p.name == name)

    # ------------------------------------------------------------------------------------------ optimizers
    def _params(self, group: str) -> List[torch.nn.Parameter]:
        return [p for p in self.model.param_groups().get(group, []) if p.requires_grad]

    def _sync_optimizers(self) -> None:
        """One AdamW per learner group; parameters registered later (new heads) are added, never the reward group."""
        o = self.cfg.optim
        reward = {id(p) for p in self.model.param_groups().get(GROUP_REWARD, [])}
        for g in LEARNER_GROUPS:
            new = [p for p in self._params(g) if id(p) not in self._opt_ids[g]]
            if any(id(p) in reward for p in new):
                raise RuntimeError("a reward parameter was offered to the continual learner")
            if not new:
                continue
            for decay, ps in ((o.weight_decay, [p for p in new if p.dim() >= 2]), (0.0, [p for p in new if p.dim() < 2])):
                if not ps:
                    continue
                grp = {"params": ps, "lr": self.lr[g], "weight_decay": decay}
                if g not in self.optimizers:
                    self.optimizers[g] = torch.optim.AdamW([grp], lr=self.lr[g], betas=self.settings.betas)
                else:
                    self.optimizers[g].add_param_group(grp)
            self._opt_ids[g] |= {id(p) for p in new}

    def lr_factor(self, step: int) -> float:
        s = self.settings
        if s.warmup_steps > 0 and step < s.warmup_steps:
            return (step + 1) / s.warmup_steps
        if s.decay_steps > 0:
            t = min(1.0, (step - s.warmup_steps) / s.decay_steps)
            return s.min_lr_factor + (1 - s.min_lr_factor) * 0.5 * (1 + math.cos(math.pi * t))
        return 1.0

    def _set_lr(self, group: str) -> float:
        lr = self.lr[group] * self.lr_factor(self.sched_step[group])
        for pg in self.optimizers[group].param_groups:
            pg["lr"] = lr
        return lr

    @property
    def model_version(self) -> str:
        return f"m{self.counters['frequent']}.{self.counters['slow']}"

    # ------------------------------------------------------------------------------------------ split / eligibility
    def is_validation(self, rec: ExperienceRecord) -> bool:
        key = rec.objective_id if (self.settings.val_split == "objective" and rec.objective_id) else rec.exp_id
        h = int(hashlib.sha1(f"{self.settings.seed}|{key}".encode()).hexdigest()[:8], 16) / 2.0 ** 32
        return h < self.settings.val_fraction

    def refresh_eligibility(self, exp_ids: Optional[Sequence[str]] = None, *, full: bool = False) -> Dict[str, int]:
        """Store, per provider, which records may be trained on (``provider.eligible`` and not held out) and which
        form its validation set (``"val:<name>"``). Incremental: records appended since the last call, the given
        records and every record of their objectives; ``full=True`` (or a provider seen for the first time)
        re-evaluates everything."""
        st = self.store
        head = st.head_seq()
        known = set(self.elig_seq.get("providers", []))
        names = [p.name for p in self.providers]
        if any(n not in known for n in names):
            full = True
        recs = {r.exp_id: r for r in st.iter_records(since_seq=0 if full else int(self.elig_seq.get("head", 0)))}
        if not full and exp_ids:
            for o in set(st.objective_of(exp_ids).values()):
                for r in st.trajectory(o):
                    recs.setdefault(r.exp_id, r)
        changed = n_train = n_val = 0
        with st.batch():
            for rec in recs.values():
                val, dirty = self.is_validation(rec), False
                for p in self.providers:
                    ok = bool(p.eligible(rec))
                    for name, flag in ((p.name, ok and not val), (VAL_PREFIX + p.name, ok and val)):
                        if rec.eligible.get(name) is not flag:
                            rec.eligible[name], dirty = flag, True
                    n_train += ok and not val
                    n_val += ok and val
                if dirty:
                    st.update(rec)
                    changed += 1
        self.elig_seq = {"head": head, "providers": names}
        return {"checked": len(recs), "changed": changed, "train_flags": int(n_train), "val_flags": int(n_val)}

    # ------------------------------------------------------------------------------------------ trigger
    def _unit(self, rec: ExperienceRecord) -> str:
        """Independence unit of a matured outcome: the top-level decision it belongs to."""
        cur, hops = rec, 0
        while cur.decision.chosen.parent_decision_id and hops < 16:
            parent = self.store.by_decision(cur.decision.chosen.parent_decision_id)
            if parent is None:
                break
            cur, hops = parent, hops + 1
        return cur.decision.decision_id

    def _informative(self, rec: ExperienceRecord) -> bool:
        lab = self.store.labels(rec.exp_id)
        if lab.get("surprise", 0.0) > 0.0:
            return lab["surprise"] >= self.settings.min_surprise
        return True                      # nothing was predicted about it: unexplained evidence is informative

    def poll(self) -> Optional[UpdatePlan]:
        """An ``UpdatePlan`` if enough independent informative outcomes matured since the last update, else
        ``None``. Reads the store's re-training queue; time is not an input."""
        queue = self.store.retrain_queue()
        matured = [q for q in queue if any(r in ("outcome_matured", "event_linked") for r in q["reasons"])]
        recs = {r.exp_id: r for r in self.store.get_many([q["exp_id"] for q in matured])}
        units: Dict[str, List[str]] = {}
        for q in matured:
            rec = recs.get(q["exp_id"])
            if rec is not None and self._informative(rec):
                units.setdefault(self._unit(rec), []).append(rec.exp_id)
        s = self.settings
        self.trigger_state = {"queued_rows": len(queue), "matured_rows": len(matured), "independent_units": len(units),
                              "threshold": s.min_independent_outcomes}
        if len(units) < s.min_independent_outcomes:
            return None
        steps = min(s.max_steps, max(1, (len(units) // s.min_independent_outcomes) * s.steps_per_trigger))
        self.counters["triggers"] += 1
        return UpdatePlan(trigger=f"matured_outcomes:{len(units)}_independent/{len(matured)}_rows",
                          timescales=(TIMESCALE_FREQUENT,), max_steps=steps, exp_ids=[q["exp_id"] for q in queue])

    # ------------------------------------------------------------------------------------------ sampling
    def _eligible_rows(self, records: Sequence[ExperienceRecord], name: str) -> List[int]:
        return [i for i, r in enumerate(records) if r.eligible.get(name, False)]

    def _sample(self, providers: Sequence[LossProvider], n: int, fresh_ids: Sequence[str] = ()
                ) -> Tuple[List[ExperienceRecord], Dict[str, int]]:
        """A replay batch shared by ``providers``: an equal quota of records eligible for each, drawn from the
        store's strata, plus (first) a share of the experience that triggered the update."""
        s, st = self.settings, self.store
        ids: List[str] = []
        origin: Dict[str, int] = {}
        names = {p.name for p in providers}
        if fresh_ids and s.fresh_fraction > 0:
            k = max(1, int(n * s.fresh_fraction))
            fresh = [r.exp_id for r in st.get_many(list(fresh_ids)[-4 * k:]) if any(r.eligible.get(x, False) for x in names)]
            ids += fresh[-k:]
            origin["fresh"] = len(ids)
        quota = max(1, math.ceil((n - len(ids)) / max(1, len(providers))))
        for p in providers:
            got = st.sample_ids(quota, loss=p.name, strata=s.strata, provenance=s.provenance)
            for k_, v in (getattr(st, "last_sample_info", {}).get("strata", {}) or {}).items():
                origin[k_] = origin.get(k_, 0) + v
            ids += [e for e in got if e not in ids]
        return st.get_many(ids[:max(n, 1)]), origin

    # ------------------------------------------------------------------------------------------ probe / hashes
    def _fingerprint(self) -> Tuple:
        groups = self.model.param_groups()
        return tuple(p._version for g, ps in groups.items() if g != GROUP_BACKBONE_FROZEN for p in ps) + \
            (sum(len(ps) for ps in groups.values()),)

    def group_hashes(self) -> Dict[str, str]:
        fp = self._fingerprint()
        if self._hash_cache[0] != fp:
            self._hash_cache = (fp, dict(self.model.group_hashes()))
        return dict(self._hash_cache[1])

    def set_probe(self, exp_ids: Optional[Sequence[str]] = None) -> List[str]:
        """Fix the probe batch (default: a seeded draw over the whole store, held-out records first)."""
        if exp_ids is None:
            pool = self.store.get_many(self.store.sample_ids(8 * self.settings.probe_size, loss=ANY_LOSS,
                                                             seed=self.settings.seed, provenance=self.settings.provenance))
            pool.sort(key=lambda r: (not self.is_validation(r),))
            exp_ids = [r.exp_id for r in pool[:self.settings.probe_size]]
        self.probe_ids = list(exp_ids)
        self._probe_cache = (None, {})
        return self.probe_ids

    @torch.no_grad()
    def probe(self) -> Dict[str, Tensor]:
        """Named behavioural outputs on the fixed probe batch (CPU tensors); cached while no parameter changed."""
        if not self.probe_ids and self.store.count():
            self.set_probe()
        if not self.probe_ids:
            return {}
        fp = self._fingerprint()
        if self._probe_cache[0] == fp:
            return self._probe_cache[1]
        recs = self.store.get_many(self.probe_ids)
        was = self.model.training
        self.model.eval()
        try:
            batch = encode_decisions(self.model, self.store, recs, with_next=self.settings.with_next, grad=False)
            out: Dict[str, Tensor] = {}
            for name, fn in self.probes.items():
                for k, v in fn(self.model, batch, self.store).items():
                    out[k if k.startswith(name) else f"{name}.{k}"] = v.detach().float().cpu()
        finally:
            self.model.train(was)
        self._probe_cache = (fp, out)
        return out

    @staticmethod
    def _probe_change(before: Dict[str, Tensor], after: Dict[str, Tensor]) -> Dict[str, float]:
        out = {}
        for k, a in after.items():
            b = before.get(k)
            if b is not None and b.shape == a.shape:
                d = torch.nan_to_num(a - b, nan=0.0, posinf=0.0, neginf=0.0).abs()
                out[k] = float(d.mean()) if d.numel() else 0.0
        out["max_abs"] = max(out.values(), default=0.0)
        return out

    # ------------------------------------------------------------------------------------------ validation
    @torch.no_grad()
    def validate(self, providers: Optional[Sequence[str]] = None, n: Optional[int] = None) -> Dict[str, float]:
        """Held-out loss of every provider on its validation records (never trained on). Deterministic sample."""
        out: Dict[str, float] = {}
        was = self.model.training
        self.model.eval()
        try:
            for p in self.providers:
                if providers is not None and p.name not in providers:
                    continue
                recs = self.store.sample(n or self.settings.val_batch, loss=VAL_PREFIX + p.name, seed=self.settings.seed,
                                         strata={"family": 1.0}, provenance=self.settings.provenance)
                if not recs:
                    continue
                if getattr(p, "needs_latents", True):
                    batch = encode_decisions(self.model, self.store, recs, with_next=self.settings.with_next, grad=False)
                else:
                    batch = records_only_batch(self.model, recs)
                loss, _ = p.loss(self.model, batch, self.store)
                if loss is not None and math.isfinite(float(loss)):
                    out[p.name] = float(loss)
        finally:
            self.model.train(was)
        return out

    # ------------------------------------------------------------------------------------------ fast timescale
    def reset_fast(self, kind: Optional[str] = None) -> None:
        """New objective: task-specific fast beliefs return to the (current, learned) prior (spec 10.4)."""
        if self.fast_state is not None:
            prior = getattr(getattr(self.model, "exp", None), "fast_prior", None)
            self.fast_state.reset(kind, prior=prior) if kind is None else self.fast_state.reset(kind)

    def fast_update(self, phi: Tensor, y: Any, kind: str, *, exposure: Any = None, censored: Any = None,
                    exp_id: str = "", trigger: str = "outcome_observed") -> LearningUpdateRecord:
        """Immediate belief update on new evidence: no gradient, no persistent parameter changes."""
        if self.fast_state is None:
            raise RuntimeError("the model has no fast belief state")
        fp = self._fingerprint()
        with torch.no_grad():
            info = self.fast_state.update(phi.detach(), y, kind, exposure=exposure, censored=censored)
        if self._fingerprint() != fp:
            raise RuntimeError("a fast update modified persistent parameters")
        self.counters["fast"] += 1
        stats = {k: float(v) for k, v in info.items() if isinstance(v, (int, float, bool))}
        rec = LearningUpdateRecord(update_id=new_id("upd"), ts=self.clock.now(), trigger=trigger, timescale=TIMESCALE_FAST,
                                   losses={"surprise": stats.get("surprise", 0.0)}, param_groups=[],
                                   n_experiences=1 if exp_id else 0, exp_ids=[exp_id] if exp_id else [],
                                   model_version=self.model_version,
                                   notes=json.dumps({"kind": kind, "fast": stats}, sort_keys=True))
        self._log(rec)
        return rec

    # ------------------------------------------------------------------------------------------ updates
    def learn(self, plan: UpdatePlan) -> List[LearningUpdateRecord]:
        """Carry out ``plan``: frequent updates (and the slow adapter update whenever it falls due), or a slow update
        when the plan asks for one explicitly. Returns one record per update attempted."""
        t0 = time.perf_counter()
        out: List[LearningUpdateRecord] = []
        self._sync_optimizers()
        credit_changes: Dict[str, float] = {}
        if self.credit is not None and plan.exp_ids:
            up = self.credit.assign(plan.exp_ids, reason=plan.trigger)
            credit_changes = dict(up.changes)
            credit_changes["n_records_changed"] = float(len(up.exp_ids))
        self.refresh_eligibility(plan.exp_ids)
        if TIMESCALE_FREQUENT in plan.timescales:
            for i in range(max(1, plan.max_steps)):
                rec = self._frequent_update(plan, credit_changes if i == 0 else {})
                out.append(rec)
                applied = bool(rec.param_groups)
                if applied and self.counters["frequent"] % max(1, self.settings.adapter_every) == 0 \
                        and GROUP_ADAPTERS in self.optimizers:
                    out.append(self._slow_update(plan, forced=False))
                if not applied or time.perf_counter() - t0 > plan.budget_s:
                    break
        if TIMESCALE_SLOW in plan.timescales:
            out.append(self._slow_update(plan, forced=True))
        if plan.exp_ids and hasattr(self.store, "pop_retrain"):
            self.store.pop_retrain(plan.exp_ids)
        return out

    def _collect(self, providers: Sequence[LossProvider], batch: DecisionBatch, params: Sequence[torch.nn.Parameter],
                 ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], Dict[str, List[Optional[Tensor]]], Dict[str, int]]:
        """Loss, metrics and gradient (w.r.t. ``params``) of each provider on its eligible rows of ``batch``."""
        losses, metrics, grads, rows_used = {}, {}, {}, {}
        live = []
        for p in providers:
            if getattr(p, "uses_batch", True):
                rows = self._eligible_rows(batch.records, p.name)
                if not rows:
                    continue
                sub = take(batch, rows)
            else:                                    # e.g. retention: it reads its own anchors
                rows, sub = [], batch
            loss, m = p.loss(self.model, sub, self.store)
            metrics[p.name] = {k: float(v) for k, v in (m or {}).items()}
            rows_used[p.name] = len(rows)
            if loss is None or not loss.requires_grad:
                continue
            losses[p.name] = float(loss.detach())
            live.append((p, loss))
        for i, (p, loss) in enumerate(live):
            grads[p.name] = list(torch.autograd.grad(p.weight * loss, params, retain_graph=i < len(live) - 1,
                                                     allow_unused=True))
        return losses, metrics, grads, rows_used

    def _frequent_update(self, plan: UpdatePlan, credit_changes: Dict[str, float]) -> LearningUpdateRecord:
        s, model = self.settings, self.model
        providers = [p for p in self.providers if p.timescale == TIMESCALE_FREQUENT]
        rec = LearningUpdateRecord(update_id=new_id("upd"), ts=self.clock.now(), trigger=plan.trigger,
                                   timescale=TIMESCALE_FREQUENT, credit_changes=dict(credit_changes),
                                   model_version=self.model_version)
        notes: Dict[str, Any] = {}
        sampled = [p for p in providers if getattr(p, "uses_batch", True)]
        records, origin = self._sample(sampled, s.batch_size, plan.exp_ids) if sampled else ([], {})
        if not records:
            return self._skip(rec, notes, "no eligible experience")
        hash_before = self.group_hashes()
        probe_before = self.probe()
        exp_p, head_p = self._params(GROUP_EXPERIENTIAL), self._params(GROUP_HEADS)
        params = exp_p + head_p
        was = model.training
        model.train()
        try:
            if any(getattr(p, "needs_latents", True) for p in sampled):
                batch = encode_decisions(model, self.store, records, with_next=s.with_next)
            else:
                batch = records_only_batch(model, records)
            losses, metrics, grads, rows_used = self._collect(providers, batch, params)
        finally:
            model.train(was)
        if FAULT_HOOK is not None:
            FAULT_HOOK(TIMESCALE_FREQUENT, "after_backward")
        rec.losses, rec.n_experiences, rec.exp_ids = losses, len(records), [r.exp_id for r in records]
        notes.update({"strata": origin, "rows": rows_used, "metrics": metrics})
        if not grads:
            return self._skip(rec, notes, "no provider produced a loss")
        ne = len(exp_p)
        exp_norm = {k: _norm(g[:ne]) for k, g in grads.items()}
        head_norm = {k: _norm(g[ne:]) for k, g in grads.items()}
        tot = sum(exp_norm.values())
        shares = {k: (v / tot if tot > 0 else 0.0) for k, v in exp_norm.items()}
        scale = {k: 1.0 for k in grads}
        nonzero = [k for k, v in exp_norm.items() if v > 0]
        if len(nonzero) > 1:
            top = max(nonzero, key=lambda k: exp_norm[k])
            rest = tot - exp_norm[top]
            if shares[top] > s.max_share:
                scale[top] = s.max_share * rest / ((1.0 - s.max_share) * exp_norm[top])
        elif len(nonzero) == 1:
            notes["balance"] = "single provider reaches the experiential group"
        tot2 = sum(exp_norm[k] * scale[k] for k in grads)
        shares_after = {k: (exp_norm[k] * scale[k] / tot2 if tot2 > 0 else 0.0) for k in grads}
        notes.update({"experiential_share": shares, "experiential_share_applied": shares_after, "balance_scale": scale,
                      "experiential_grad_norm_by_loss": exp_norm, "heads_grad_norm_by_loss": head_norm})
        self.share_log.append(dict(shares_after))
        for j, p in enumerate(params):
            g = None
            for k, gs in grads.items():
                if gs[j] is not None:
                    part = gs[j] * scale[k] if j < ne else gs[j]
                    g = part if g is None else g + part
            p.grad = g
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in params) or \
                any(not math.isfinite(v) for v in losses.values()):
            for p in params:
                p.grad = None
            return self._skip(rec, notes, "non-finite loss or gradient")
        before = [p.detach().clone() for p in params]
        for g, ps in ((GROUP_EXPERIENTIAL, exp_p), (GROUP_HEADS, head_p)):
            with_grad = [p for p in ps if p.grad is not None]
            rec.grad_norms[g] = float(torch.nn.utils.clip_grad_norm_(with_grad, self.clip[g])) if with_grad else 0.0
            if with_grad and g in self.optimizers:
                notes.setdefault("lr", {})[g] = self._set_lr(g)
                self.optimizers[g].step()
                self.sched_step[g] += 1
        for p in params:
            p.grad = None
        self.counters["frequent"] += 1
        with torch.no_grad():
            rec.weight_delta = {g: _norm([p - b for p, b in zip(ps, bs)]) for g, ps, bs in
                                ((GROUP_EXPERIENTIAL, exp_p, before[:ne]), (GROUP_HEADS, head_p, before[ne:]))}
        del before
        return self._finish(rec, notes, hash_before, probe_before)

    def _slow_update(self, plan: UpdatePlan, forced: bool) -> LearningUpdateRecord:
        s, model = self.settings, self.model
        rec = LearningUpdateRecord(update_id=new_id("upd"), ts=self.clock.now(), timescale=TIMESCALE_SLOW,
                                   trigger=plan.trigger if forced else f"{self.counters['frequent']} frequent updates "
                                   f"(adapter_every={s.adapter_every})", model_version=self.model_version)
        notes: Dict[str, Any] = {"forced": forced}
        params = self._params(GROUP_ADAPTERS)
        if not params or GROUP_ADAPTERS not in self.optimizers:
            return self._skip(rec, notes, "the model has no trainable semantic adapters")
        providers = [p for p in self.providers if getattr(p, "uses_batch", True)
                     and (p.timescale == TIMESCALE_SLOW or getattr(p, "adapter_grad", True))]
        records, origin = self._sample(providers, s.slow_batch_size) if providers else ([], {})
        if not records:
            return self._skip(rec, notes, "no eligible experience")
        val_before = self.validate() if s.slow_guard else {}
        hash_before = self.group_hashes()
        probe_before = self.probe()
        was = model.training
        model.train()
        try:
            batch = encode_with_adapter_grad(model, self.store, records, grad_last=s.grad_last, with_next=s.with_next)
            losses, metrics, grads, rows_used = self._collect(providers, batch, params)
        finally:
            model.train(was)
        if FAULT_HOOK is not None:
            FAULT_HOOK(TIMESCALE_SLOW, "after_backward")
        rec.losses, rec.n_experiences, rec.exp_ids = losses, len(records), [r.exp_id for r in records]
        notes.update({"strata": origin, "rows": rows_used, "metrics": metrics, "grad_last": s.grad_last,
                      "adapter_grad_norm_by_loss": {k: _norm(g) for k, g in grads.items()}})
        if not grads:
            return self._skip(rec, notes, "no provider produced a loss")
        for j, p in enumerate(params):
            g = None
            for gs in grads.values():
                if gs[j] is not None:
                    g = gs[j] if g is None else g + gs[j]
            p.grad = g
        if all(p.grad is None for p in params):
            return self._skip(rec, notes, "no loss reached the adapters")
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in params):
            for p in params:
                p.grad = None
            return self._skip(rec, notes, "non-finite adapter gradient")
        bb = getattr(model, "backbone", None)
        if bb is not None and hasattr(bb, "lora_parameters"):
            lora = {id(p) for p in bb.lora_parameters()}
            notes["adapter_grad_norm"] = {"lora": _norm([p.grad for p in params if id(p) in lora]),
                                          "bridge": _norm([p.grad for p in params if id(p) not in lora])}
        before = [p.detach().clone() for p in params]
        with_grad = [p for p in params if p.grad is not None]
        rec.grad_norms[GROUP_ADAPTERS] = float(torch.nn.utils.clip_grad_norm_(with_grad, self.clip[GROUP_ADAPTERS]))
        notes.setdefault("lr", {})[GROUP_ADAPTERS] = self._set_lr(GROUP_ADAPTERS)
        self.optimizers[GROUP_ADAPTERS].step()
        self.sched_step[GROUP_ADAPTERS] += 1
        for p in params:
            p.grad = None
        if hasattr(model, "mark_adapters_updated"):
            model.mark_adapters_updated()
        with torch.no_grad():
            rec.weight_delta = {GROUP_ADAPTERS: _norm([p - b for p, b in zip(params, before)])}
            if bb is not None and hasattr(bb, "lora_parameters"):
                notes["adapter_delta_norm"] = {
                    "lora": _norm([p - b for p, b in zip(params, before) if id(p) in lora]),
                    "bridge": _norm([p - b for p, b in zip(params, before) if id(p) not in lora])}
        if s.slow_guard and val_before:
            val_after = self.validate()
            common = [k for k in val_before if k in val_after]
            if common:
                mb = sum(val_before[k] for k in common) / len(common)
                ma = sum(val_after[k] for k in common) / len(common)
                notes["guard"] = {"validation_before": mb, "validation_after": ma}
                if not math.isfinite(ma) or ma > mb + s.slow_guard_tol * max(abs(mb), 1e-6):
                    with torch.no_grad():
                        for p, b in zip(params, before):
                            p.copy_(b)
                    if hasattr(model, "mark_adapters_updated"):
                        model.mark_adapters_updated()
                    self.counters["slow_reverted"] += 1
                    notes["guard"]["reverted"] = True
                    rec.weight_delta = {GROUP_ADAPTERS: 0.0}
        del before
        self.counters["slow"] += 1
        return self._finish(rec, notes, hash_before, probe_before)

    # ------------------------------------------------------------------------------------------ records
    def _skip(self, rec: LearningUpdateRecord, notes: Dict[str, Any], why: str) -> LearningUpdateRecord:
        self.counters["skipped"] += 1
        notes["skipped"] = why
        rec.notes = json.dumps(notes, sort_keys=True, default=str)
        self._log(rec)
        return rec

    def _finish(self, rec: LearningUpdateRecord, notes: Dict[str, Any], hash_before: Dict[str, str],
                probe_before: Dict[str, Tensor]) -> LearningUpdateRecord:
        hash_after = self.group_hashes()
        rec.hash_before = {g: h for g, h in hash_before.items() if g != GROUP_BACKBONE_FROZEN}
        rec.hash_after = {g: h for g, h in hash_after.items() if g != GROUP_BACKBONE_FROZEN}
        rec.param_groups = [g for g in rec.hash_after if rec.hash_before.get(g) != rec.hash_after[g]]
        if GROUP_REWARD in rec.param_groups:
            raise RuntimeError("a continual-learner update changed the reward parameters")
        rec.prediction_change = self._probe_change(probe_before, self.probe())
        changed = rec.prediction_change.get("max_abs", 0.0) > self.settings.probe_tol
        notes["behaviour_changed"] = changed
        if rec.param_groups and not changed:
            notes["flag"] = "weights changed but probe predictions and action probabilities did not"
        rec.model_version = self.model_version
        rec.notes = json.dumps(notes, sort_keys=True, default=str)
        self._log(rec)
        return rec

    def _log(self, rec: LearningUpdateRecord) -> None:
        self.history.append(rec)
        if len(self.history) > 2048:
            del self.history[:1024]
        if self.log_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
            with open(self.log_path, "a") as fh:
                fh.write(rec.to_json() + "\n")

    # ------------------------------------------------------------------------------------------ state
    def state_dict(self) -> Dict[str, Any]:
        """Schedules and counters (JSON-native). Weights, optimizer and RNG states are captured by
        ``credit.checkpoint.capture_learner``."""
        return {"version": 1, "counters": dict(self.counters), "sched_step": dict(self.sched_step),
                "probe_ids": list(self.probe_ids), "elig_seq": dict(self.elig_seq),
                "settings": dataclasses.asdict(self.settings), "lr": dict(self.lr), "clip": dict(self.clip),
                "providers": [p.name for p in self.providers], "model_version": self.model_version}

    def load_state_dict(self, st: Dict[str, Any]) -> Dict[str, Any]:
        self.counters.update(st.get("counters", {}))
        self.sched_step.update(st.get("sched_step", {}))
        self.probe_ids = list(st.get("probe_ids", []))
        self.elig_seq = dict(st.get("elig_seq", {}))
        self._probe_cache, self._hash_cache = (None, {}), (None, {})
        saved = st.get("providers", [])
        mine = [p.name for p in self.providers]
        return {"providers_missing": [n for n in saved if n not in mine],
                "providers_new": [n for n in mine if n not in saved]}

    def status(self) -> Dict[str, Any]:
        last = self.history[-1] if self.history else None
        return {"counters": dict(self.counters), "model_version": self.model_version, "trigger": dict(self.trigger_state),
                "providers": [{"name": p.name, "timescale": p.timescale, "weight": p.weight} for p in self.providers],
                "last_update": None if last is None else {"timescale": last.timescale, "trigger": last.trigger,
                                                          "param_groups": last.param_groups, "losses": last.losses},
                "lr": {g: self.lr[g] * self.lr_factor(self.sched_step[g]) for g in self.optimizers}}
