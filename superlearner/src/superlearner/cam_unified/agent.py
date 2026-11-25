"""CamUnifiedAgent: per-environment state around CamUnifiedModel (spec Parts 11, 12, 14, 18.1, 20 A22).

AgentP shape (used by runtime.run_steps, workstream F's actors and - through `stage1.harness_agent.HarnessAgent` -
G's OpenWorld lifetime harness `envs.harness.run_lifetime`):

    act(obs_batch: ObservationBatch, tools_batch: [B] list[ToolSchema]) -> [B] Action | None   (None = done env)
    observe(results: [B] ExecutionResult, rewards: [B] float, next_obs: ObservationBatch) -> list[EventRecord]
    observe(transition: envs.agent_protocol.Transition) -> list[EventRecord]      (harness call shape)
    update() -> dict                                    (one learner step; no-op when cfg.continual_updates is False)
    reset(env_ids: list[str] | None) -> None
    flush() -> list[EventRecord]                        (end of a run: archive undelivered post-action observations)

Per environment the agent keeps Z_t, the FastState, an EntityRegistry (episode-local slots), the outcome tracker, the
active option, the last executed action, the goal and the remaining budget (days to the goal deadline, funds, decisions).
Every executed decision becomes one `EventRecord` (provenance REAL, behaviour log-probability and its provenance,
predictions + uncertainty made before acting, model_version, realised outcome specs) appended to the injected archive,
and one `DecisionTrace` sent to the injected recorder. Observations delivered after an action that are not part of a
later decision's input (episode end, reset, end of the run) are archived as world-event records (action None,
attributed_to = the action's event id), so every delivered observation reaches the archive.

`update()` builds the loss terms from (a) sequences sampled by the injected replay and tokenised on the fly from
archived EventRecords (world model, support/query adaptation on separate longer sequences, memory), filtered with
the acting context, and (b) the fresh on-policy buffer (actor + value with GAE under the semi-Markov discount
gamma = exp(-kappa * dt), plus the reasoner's compute cost): PPO-clip for decoder-sampled decisions (exact behaviour
probability), advantage-weighted regression for planner-selected ones (the argmax selection has no tractable
behaviour probability), then hands them to the injected trainer, which weights, back-propagates and steps.
With workstream E's classes: the Consolidator runs every update (archived events -> memory segments in the
MemoryIndex; its first segments make the index serve their version), the ReplaySampler is synced from the archive
before sampling, every consumed event (fresh decisions and replayed records) gets the update id in its archived
training provenance (`EventArchive.mark_used` -> `EventRecord.training_provenance["used_in_updates"]`), memory
consumed by fresh decisions is linked to the update (`Consolidator.link_weights`), and events of an abandoned branch
(`exclude_events`, set after a resume) are never drawn. With workstream F's TraceRecorder, one DecisionTrace per
decision is recorded and update refs are patched in by event id per lifetime. `loss_cap` bounds the magnitude of any
one objective before the trainer's global gradient clip (off by default).

Checkpoint attribute contract (the duck-typed contract of workstream E's memory/checkpoint.py:CheckpointOps):
    model, config, weights_exclude, optimizers (+ setter; make_optimizers when the trainer has it), schedulers,
    fast_state / init_fast_state(), latent_state / init_latent_state(), registries, archive, index, replay, rng,
    version (setter invalidates weight-dependent caches), counters; plus reset_fast_state, reset_optimizer,
    reset_experience_access, reset_working_state.
"""
from __future__ import annotations

import inspect
import math
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Sequence

import torch

from .config import CamConfig
from .interfaces import (DecisionTrace, EntityRegistry, EventRecord, EvidenceKind, ExecutionResult, FastState,
                         GoalSpec, Observation, ObservationBatch, ObsKind, Provenance, SequenceBatch, StepBatch,
                         TokenSet, new_id)
from .model import BudgetState, CamState, CamUnifiedModel, ReplayAux, StepInputs, check_tokens, fast_scatter
from .outcome_spec import CONTINUOUS, OutcomeSpec, build_outcome_batch

SUPPORT_SIZES = (0, 1, 2, 4, 8, 16, 32, 64)
OBS_REF = re.compile(r"^obs:(\d+)")
SLOT_REF = re.compile(r"^entity(?:_slot)?:(\d+)$")             # B's EntityEncoder: "entity:<slot>"
REL_REF = re.compile(r"^relation:(-?\d+)->(-?\d+)$")           # B's EntityEncoder: "relation:<i>-><j>" (slots)
FUNDS_NAME = re.compile(r"balance|funds|cash", re.IGNORECASE)


# ============================================================================ protocols (not yet in interfaces.py)

class AgentP(Protocol):
    """What runtimes, actors (F) and the OpenWorld harness (G) drive."""
    def act(self, obs_batch: ObservationBatch, tools_batch: Sequence[Sequence]) -> list: ...   # [B] Action | None
    def observe(self, results: Sequence, rewards: Sequence[float], next_obs: ObservationBatch) -> list: ...
    def update(self) -> dict: ...
    def reset(self, env_ids: Optional[Sequence[str]] = None) -> None: ...
    def set_goals(self, goals: Sequence[GoalSpec]) -> None: ...
    def flush(self) -> list: ...


class ReplayP(Protocol):                      # workstream E: memory/replay.py:ReplaySampler
    def sample(self, n_seqs: int, seq_len: int) -> list: ...   # list[list[EventRecord]] or list[ReplaySlice(.records)]
    # optional: add(records: list[EventRecord]) -> None (notification of newly archived records)


class TraceRecorderP(Protocol):               # a list recorder (doubles); F's TraceRecorder is detected by shape:
    def record(self, traces: list) -> None: ...                # list[DecisionTrace]
    # optional: link_updates(decision_ids: list[str], update_id: str) -> None
    # F's observe/trace.py:TraceRecorder: record(trace) one DecisionTrace per call and
    # add_update_refs(event_ids, update_id, env_id) (patch lines keyed by event id); see _record / _link_traces


class TrainerP(Protocol):                     # workstream D: learn/trainer.py:CombinedTrainer
    def step(self, losses: dict) -> object: ...                 # weights, backward, per-objective core grads, clip,
                                                                # step; returns a dict or an object with to_dict()
                                                                # (its "update_id", when present, links the traces)
    # optional: optimizers (dict) or opt, reset_optimizer(), make_optimizers(), schedulers


class MemoryIndexP(Protocol):                 # a records-level memory index (doubles); E's MemoryIndex is instead
    def add(self, records: list) -> None: ...  # fed by E's Consolidator: step() / link_weights(ids, update_id, v)


# ============================================================================ fresh on-policy buffer

@dataclass
class FreshStep:
    """One decision of the fresh on-policy buffer (sub-batch of live rows)."""
    rows: list                      # agent rows
    episode_ids: list
    inputs: StepInputs
    actions: list
    logp_old: torch.Tensor          # [b] recorded behaviour logp (generator proposal logp for planner choices)
    value_old: torch.Tensor         # [b]
    reward: torch.Tensor            # [b] verified extrinsic return increment
    dt: torch.Tensor                # [b] simulated days until the next decision
    done: torch.Tensor              # [b] bool
    version: int
    decision_ids: list
    lp: torch.Tensor = None         # [b] hindsight learning progress (filled one decision later)
    planner: torch.Tensor = None    # [b] bool: chosen by the planner's argmax (no exact behaviour probability)
    event_ids: list = None          # [b] archived EventRecord ids of the decisions (training provenance)
    env_ids: list = None            # [b] environment ids (trace patches are keyed per lifetime)

    def subset(self, js: Sequence[int]) -> "FreshStep":
        idx = torch.tensor(list(js), dtype=torch.long, device=self.logp_old.device)
        return FreshStep([self.rows[j] for j in js], [self.episode_ids[j] for j in js],
                         None if self.inputs is None else self.inputs.select(js), [self.actions[j] for j in js],
                         self.logp_old[idx], self.value_old[idx], self.reward[idx], self.dt[idx], self.done[idx],
                         self.version, [self.decision_ids[j] for j in js], self.lp[idx],
                         None if self.planner is None else self.planner[idx],
                         None if self.event_ids is None else [self.event_ids[j] for j in js],
                         None if self.env_ids is None else [self.env_ids[j] for j in js])


@dataclass
class UpdateBatch:
    """Data of one learner update. `seq`/`aux`: tokenised replay sequences for the world / memory objectives;
    `adapt_seq`/`adapt_aux`: longer sequences for the support/query objective; `fresh`: FreshSteps."""
    seq: Optional[SequenceBatch]
    aux: Optional[ReplayAux]
    fresh: list
    model: Optional[CamUnifiedModel] = None
    adapt_seq: Optional[SequenceBatch] = None
    adapt_aux: Optional[ReplayAux] = None
    _filt: object = None
    _adapt_filt: object = None

    def filtered(self):
        """Acting-consistent filter over the replay sequences (computed once, shared by the objectives)."""
        if self._filt is None:
            self._filt = self.model.filter_sequence(self.seq, self.aux)
        return self._filt

    def adapt_filtered(self):
        if self.adapt_seq is None:
            return self.filtered()
        if self._adapt_filt is None:
            self._adapt_filt = self.model.filter_sequence(self.adapt_seq, self.adapt_aux)
        return self._adapt_filt


# ============================================================================ default objectives (replaceable)

def objective_world(agent: "CamUnifiedAgent", batch: UpdateBatch) -> dict:
    """L_world from WorldModelP.loss(SequenceBatch[, beliefs, memory]): expects key "world"; scalars -> metrics."""
    if batch.seq is None:
        return {}
    out = agent.model.world_loss(batch.seq, batch.filtered())
    if "world" not in out:
        raise KeyError(f"WorldModelP.loss must return the total under 'world'; got keys {sorted(out)}")
    mets = {f"world/{k}": float(v.detach()) for k, v in out.items()
            if torch.is_tensor(v) and v.numel() == 1 and k != "world"}
    return {"losses": {"world": out["world"]}, "metrics": mets}


def objective_adapt(agent: "CamUnifiedAgent", batch: UpdateBatch) -> dict:
    """L_adapt with a support size k drawn uniformly from SUPPORT_SIZES restricted to k < T (at least one query
    step remains), on the longer adaptation sequences when available. The drawn k are logged (agent.adapt_k_hist)."""
    cfg = agent.config
    seq = batch.adapt_seq if batch.adapt_seq is not None else batch.seq
    if seq is None or not cfg.use_fast_adaptation:
        return {}
    T = len(seq.steps)
    ks = [k for k in SUPPORT_SIZES if k < T]
    k = agent.py_rng.choice(ks)
    agent.adapt_k_hist[k] += 1
    loss, mets = agent.model.adapt_loss(seq, batch.adapt_filtered(), k)
    return {"losses": {"adapt": loss}, "metrics": {f"adapt/{n}": v for n, v in mets.items()}}


def objective_memory(agent: "CamUnifiedAgent", batch: UpdateBatch) -> dict:
    """L_memory from the retriever's own objective (RetrieverP.loss(seq, Zs)), when it defines one."""
    r = agent.model.retriever
    if batch.seq is None or not agent.config.use_memory or r is None or not hasattr(r, "loss"):
        return {}
    return {"losses": {"memory": r.loss(batch.seq, batch.filtered().Zs)}}


def gae_semi_markov(entries: list, kappa: float, lam: float, lp_coef: float) -> dict:
    """GAE over one environment's chronological fresh entries [(FreshStep, j)], one episode at a time.
    gamma_t = exp(-kappa * dt_t) (semi-Markov); r_t = reward + lp_coef * LP_t. The last entry of the current episode
    (no successor yet, not done) only bootstraps; a non-final episode's undone last entry (forced reset) is dropped.
    Returns {(id(step), j): (advantage, return)}."""
    out = {}
    episodes: list = []
    for st, j in entries:
        if not episodes or episodes[-1][0] != st.episode_ids[j]:
            episodes.append((st.episode_ids[j], []))
        episodes[-1][1].append((st, j))
    for _, seq in episodes:
        gae_next, v_next = 0.0, 0.0
        for idx in range(len(seq) - 1, -1, -1):
            st, j = seq[idx]
            done = bool(st.done[j])
            v = float(st.value_old[j])
            if idx == len(seq) - 1 and not done:
                gae_next, v_next = 0.0, v       # bootstrap-only (or truncated) entry
                continue
            gamma = math.exp(-kappa * float(st.dt[j]))
            r = float(st.reward[j]) + lp_coef * float(st.lp[j])
            nonterm = 0.0 if done else 1.0
            delta = r + gamma * v_next * nonterm - v
            gae = delta + gamma * lam * nonterm * gae_next
            out[(id(st), j)] = (gae, gae + v)
            gae_next, v_next = gae, v
    return out


def objective_policy(agent: "CamUnifiedAgent", batch: UpdateBatch) -> dict:
    """Actor + value on the fresh on-policy buffer (spec 12.3), with the reasoner's compute cost (architecture §4) and
    the entropy bonus inside the actor term. Behaviour log-probabilities are recomputed through the full step (tokenizer -> core -> reasoner ->
    option policy -> action generator) from stored inputs, so gradients reach every one of them.
    Routing by behaviour provenance: decoder-sampled decisions use PPO-clip ratios against their exact behaviour logp;
    planner-selected decisions (argmax over imagined scores; the selection probability is not tractable) use
    advantage-weighted regression, -exp(A / beta) log pi(a), which needs no behaviour probability."""
    cfg = agent.config
    per_env: dict = {}
    for st in batch.fresh:
        for j, r in enumerate(st.rows):
            per_env.setdefault(r, []).append((st, j))
    lp_coef = cfg.lp_coef if cfg.use_learned_exploration else 0.0
    targets = {}
    for r, entries in per_env.items():
        targets.update(gae_semi_markov(entries, cfg.kappa, cfg.gae_lambda, lp_coef))
    chosen = [(st, [j for j in range(len(st.rows)) if (id(st), j) in targets]) for st in batch.fresh]
    chosen = [(st, js) for st, js in chosen if js]
    if not chosen:
        return {}
    subs = [st.subset(js) for st, js in chosen]
    inp = StepInputs.cat([s.inputs for s in subs])
    actions = [a for s in subs for a in s.actions]
    dev = agent.device
    adv = torch.tensor([targets[(id(st), j)][0] for st, js in chosen for j in js], device=dev)
    ret = torch.tensor([targets[(id(st), j)][1] for st, js in chosen for j in js], device=dev)
    logp_old = torch.cat([s.logp_old for s in subs]).to(dev)
    planner = torch.cat([s.planner for s in subs]).to(dev).bool()
    lag = torch.tensor([agent.version - s.version for s in subs for _ in s.rows], dtype=torch.float, device=dev)
    out = agent.model.policy_evaluate(inp, actions)
    logp, ent, value = out["logp"], out["entropy"], out["value"]
    a_n = (adv - adv.mean()) / (adv.std() + 1e-8) if adv.numel() > 1 else adv
    ratio = torch.exp(logp - logp_old)
    clipped = torch.clamp(ratio, 1 - cfg.ppo_clip, 1 + cfg.ppo_clip)
    ppo = -torch.min(ratio * a_n, clipped * a_n)
    w = torch.exp(torch.clamp(a_n / agent.awr_beta, max=math.log(agent.awr_max_weight))).detach()
    awr = -w * logp
    actor = torch.where(planner, awr, ppo).mean() - cfg.entropy_coef * ent.mean()
    vh = agent.model.value_head
    if out.get("value_logits") is not None and hasattr(vh, "loss"):
        vloss = vh.loss(out["value_logits"], ret).mean()
    else:
        vloss = ((value - ret) ** 2).mean()
    reason_cost = None
    if out.get("ponder") is not None and cfg.use_reasoning and cfg.adaptive_reasoning:
        # the reasoner's compute penalty is part of the policy objective (keys stay those of cfg.loss_weights,
        # which D's CombinedTrainer requires); reported separately as policy/reason_compute
        reason_cost = cfg.reason_compute_cost * out["ponder"].mean()
        actor = actor + reason_cost
    losses = {"actor": actor, "value": vloss}
    dev_lp = (logp.detach() - logp_old).abs()
    lag0 = lag == 0
    ppo_rows = ~planner
    mets = {"policy/n": int(adv.numel()), "policy/n_ppo": int(ppo_rows.sum()), "policy/n_awr": int(planner.sum()),
            "policy/ratio_mean": float(ratio.detach()[ppo_rows].mean()) if bool(ppo_rows.any()) else float("nan"),
            "policy/ratio_maxdev": float((ratio.detach()[ppo_rows] - 1).abs().max()) if bool(ppo_rows.any())
            else float("nan"),
            "policy/logp_maxdev_lag0": float(dev_lp[lag0].max()) if bool(lag0.any()) else float("nan"),
            "policy/awr_w_mean": float(w[planner].mean()) if bool(planner.any()) else float("nan"),
            "policy/approx_kl": float((logp_old - logp.detach()).mean()), "policy/entropy": float(ent.detach().mean()),
            "policy/adv_mean": float(adv.mean()), "policy/return_mean": float(ret.mean()),
            "policy/max_version_lag": float(lag.max()) if lag.numel() else 0.0,
            "policy/memory_rebuilt": bool(out.get("memory_rebuilt"))}
    if reason_cost is not None:
        mets["policy/reason_compute"] = float(reason_cost.detach())
        mets["policy/ponder_mean"] = float(out["ponder"].detach().mean())
    return {"losses": losses, "metrics": mets, "consumed": [d for s in subs for d in s.decision_ids],
            "consumed_events": [(e, v) for s in subs for e, v in zip(s.event_ids or [], s.env_ids or [])]}


DEFAULT_OBJECTIVES = {"world": objective_world, "adapt": objective_adapt, "memory": objective_memory,
                      "policy": objective_policy}


# ============================================================================ helpers

def public_goal(g: Optional[GoalSpec]) -> GoalSpec:
    """The agent-visible part of a goal (family and meta are bookkeeping, never model inputs)."""
    if g is None:
        return GoalSpec("none", "")
    return GoalSpec(g.goal_id, g.text, dict(g.metrics), g.deadline_days)


def _goal_dict(g: GoalSpec) -> dict:
    return {"goal_id": g.goal_id, "text": g.text, "metrics": dict(g.metrics), "deadline_days": g.deadline_days}


def _obs_key(o: Observation) -> tuple:
    if o.seq is not None:
        return ("seq", int(o.seq))
    return ("obs", o.kind.value, o.source, o.entity, o.t, o.value, (o.content or "")[:256], o.artifact_ref)


def _takes_one_arg(fn) -> bool:
    if not callable(fn):
        return False
    try:
        ps = [p for p in inspect.signature(fn).parameters.values()
              if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty]
    except (TypeError, ValueError):
        return False
    return len(ps) == 1


def default_funds(observations: Sequence[Observation], money_unit: str) -> dict:
    """{key: value} of numeric observations that report the agent's money (unit == the world's money unit and a
    balance / funds / cash metric or field); the remaining funds are the sum of the latest values."""
    out = {}
    for o in observations:
        if o.kind != ObsKind.NUMERIC or o.value is None or (o.unit or "") != money_unit:
            continue
        meta = o.meta if isinstance(o.meta, dict) else {}
        name = str(meta.get("metric") or meta.get("field") or o.source or "")
        if FUNDS_NAME.search(name):
            out[f"{name}@{o.entity}"] = float(o.value)
    return out


# ============================================================================ the agent

class CamUnifiedAgent:
    """Batched agent over B environments (see module docstring). All collaborators are injected; `archive`,
    `recorder`, `replay`, `index` and `trainer` may be None (then the corresponding path is skipped and reported)."""

    def __init__(self, model: CamUnifiedModel, *, archive=None, recorder=None, replay=None, trainer=None, index=None,
                 consolidator=None, actor_id: str = "actor0", seed: int = 0, seq_len: int = 8, n_seqs: int = 4, adapt_seq_len: int = 65,
                 adapt_n_seqs: int = 2, greedy: bool = False, objectives: Optional[dict] = None,
                 fresh_max_steps: int = 256, decision_budget: Optional[int] = None,
                 funds_fn: Optional[Callable] = None, awr_beta: float = 1.0, awr_max_weight: float = 20.0,
                 loss_cap: Optional[float] = None, loss_cap_beta: float = 0.9):
        self.model = model
        self.config: CamConfig = model.cfg
        self.archive = archive
        self.recorder = recorder
        self.replay = replay
        self.trainer = trainer
        self.index = index
        self.consolidator = consolidator
        self.actor_id = actor_id
        self.seq_len = seq_len
        self.n_seqs = n_seqs
        self.adapt_seq_len = adapt_seq_len
        self.adapt_n_seqs = adapt_n_seqs
        self.greedy = greedy
        self.objectives = dict(DEFAULT_OBJECTIVES if objectives is None else objectives)
        self.fresh_max_steps = fresh_max_steps
        self.decision_budget = decision_budget
        self.funds_fn = funds_fn or (lambda obs: default_funds(obs, model.money_unit))
        self.awr_beta = awr_beta
        self.awr_max_weight = awr_max_weight
        # magnitude cap per objective (spec 12.5 "do not let a large reconstruction loss dominate decision-relevant
        # learning"): an objective whose running magnitude m_k (EMA of |L_k|, detached) exceeds `loss_cap` enters
        # the combined loss as L_k * loss_cap / m_k (same gradient direction, bounded scale). D's trainer clips the
        # *global* gradient norm, so without it one objective of magnitude 1e3 shrinks every other objective's step
        # by the same factor. None = off. The scales are logged per update (balance/<objective>).
        self.loss_cap = loss_cap
        self.loss_cap_beta = float(loss_cap_beta)
        self.loss_mag: dict = {}
        self.py_rng = random.Random(seed)
        self.torch_rng = torch.Generator().manual_seed(seed)
        self.seed = seed
        self.state: Optional[CamState] = None
        self.env_ids: list = []
        self.goals: list = []
        self.episode_ids: list = []
        self.steps: list = []
        self.fresh: list = []
        self.update_log: list = []
        self.n_decisions = 0
        self.adapt_k_hist: Counter = Counter()
        self.dropped = {"fresh_steps": 0, "fresh_decisions": 0}
        self._pending = None
        self._post: dict = {}           # row -> undelivered post-action observations of the last action
        self._last_eid: dict = {}       # row -> event id of the last executed action (context refs)
        self._last_fresh: dict = {}
        self._lp_valid: list = []
        self._t0: list = []
        self._funds: list = []
        self._funds0: list = []
        self._ep_decisions: list = []
        self.timing = {"act_s": 0.0, "observe_s": 0.0, "update_s": 0.0}
        # archived events of an abandoned branch (written after the checkpoint a lifetime was resumed from): real
        # experience of a world state that no longer exists; never drawn for training (see `_sample`)
        self.exclude_events: set = set()
        self.provenance_stats = {"marked_events": 0, "replay_records_excluded": 0}
        if trainer is not None and callable(getattr(trainer, "make_optimizers", None)):
            self.make_optimizers = trainer.make_optimizers

    # ---------------------------------------------------------------- checkpoint attribute contract (E)
    @property
    def rng(self) -> dict:
        return {"python": self.py_rng, "torch": self.torch_rng}

    @property
    def version(self) -> int:
        return self.model.version

    @version.setter
    def version(self, v: int) -> None:
        self.model.version = int(v)          # also bumps the model's weights epoch (query-embedding cache)

    @property
    def optimizers(self) -> dict:
        t = self.trainer
        if t is None:
            return {}
        if isinstance(getattr(t, "optimizers", None), dict):
            return dict(t.optimizers)
        if getattr(t, "opt", None) is not None:
            return {"trainer": t.opt}
        return {}

    @optimizers.setter
    def optimizers(self, v: dict) -> None:
        t = self.trainer
        if t is None:
            return
        if isinstance(getattr(t, "optimizers", None), dict):
            t.optimizers = dict(v)
        elif "trainer" in v:
            t.opt = v["trainer"]

    @property
    def schedulers(self) -> dict:
        return dict(getattr(self.trainer, "schedulers", None) or {})

    @property
    def weights_exclude(self) -> tuple:
        """State-dict keys of frozen pretrained semantic weights (re-loaded from cfg.sem_model, never saved)."""
        return self.model.frozen_semantic_keys()

    @property
    def fast_state(self) -> Optional[FastState]:
        return None if self.state is None else self.state.fast

    @fast_state.setter
    def fast_state(self, f: Optional[FastState]) -> None:
        if self.state is None:
            return
        if f is None:
            f = self.init_fast_state()
        elif f.logw.shape[0] != self.B:
            raise ValueError(f"fast_state has {f.logw.shape[0]} rows for {self.B} environments")
        self.state.fast = None if f is None else FastState(
            *(getattr(f, n).to(self.device) for n in ("particles", "logw", "pscale", "bayes_mu", "bayes_P",
                                                       "bayes_noise", "bayes_n", "surprise", "steps")),
            [dict(c) for c in f.channels])
        self.state.fast_prev = None
        self._lp_valid = [False] * self.B

    @property
    def fast(self) -> Optional[FastState]:
        return self.fast_state

    def init_fast_state(self) -> Optional[FastState]:
        return None if not self.env_ids else self.model.init_fast(self.B, self.device)

    @property
    def latent_state(self) -> Optional[torch.Tensor]:
        return None if self.state is None else self.state.Z

    @latent_state.setter
    def latent_state(self, Z: Optional[torch.Tensor]) -> None:
        if self.state is None:
            return
        if Z is None:
            Z = self.init_latent_state()
        cfg = self.config
        if tuple(Z.shape) != (self.B, cfg.n_state, cfg.d_model):
            raise ValueError(f"latent_state {tuple(Z.shape)} != {(self.B, cfg.n_state, cfg.d_model)}")
        self.state.Z = Z.detach().to(self.device).float()

    def init_latent_state(self) -> Optional[torch.Tensor]:
        if not self.env_ids:
            return None
        with torch.no_grad():
            return self.model.core.init_state(self.B, device=self.device).float()

    @property
    def registries(self) -> list:
        return [] if self.state is None else self.state.registries

    @registries.setter
    def registries(self, regs) -> None:
        """A list of B EntityRegistry, or [] / None for fresh episode-local registries (E's reset_fast_state)."""
        if self.state is None:
            return
        if not regs:
            regs = [EntityRegistry(self.config.max_entities, seed=self.py_rng.randrange(2**31)) for _ in range(self.B)]
        if len(regs) != self.B:
            raise ValueError(f"{len(regs)} registries for {self.B} environments")
        self.state.registries = list(regs)

    @property
    def counters(self) -> dict:
        return {"n_decisions": self.n_decisions, "steps": list(self.steps), "episode_ids": list(self.episode_ids),
                "env_ids": list(self.env_ids), "updates": len(self.update_log),
                "adapt_k_hist": {str(k): v for k, v in self.adapt_k_hist.items()}, "dropped": dict(self.dropped)}

    @counters.setter
    def counters(self, c: dict) -> None:
        """{} = fresh counters (weights-only load); a dict from `counters` restores them."""
        c = dict(c or {})
        self.n_decisions = int(c.get("n_decisions", 0))
        if c.get("steps") is not None and len(c["steps"]) == self.B:
            self.steps = [int(x) for x in c["steps"]]
        else:
            self.steps = [0] * self.B
        if c.get("episode_ids") is not None and len(c["episode_ids"]) == self.B:
            self.episode_ids = list(c["episode_ids"])
        elif not c:
            self.episode_ids = [new_id("ep_") for _ in range(self.B)]
        self.adapt_k_hist = Counter({int(k): int(v) for k, v in (c.get("adapt_k_hist") or {}).items()})
        self.dropped = {"fresh_steps": 0, "fresh_decisions": 0, **(c.get("dropped") or {})}

    @property
    def device(self) -> torch.device:
        return self.model.device

    @property
    def B(self) -> int:
        return len(self.env_ids)

    def reset_fast_state(self) -> None:
        """Reinitialise every environment's FastState (particles, weights, Kalman statistics, channel tables)."""
        if self.state is None or self.state.fast is None:
            return
        self.fast_state = None

    def reset_working_state(self) -> None:
        """Reinitialise Z, entity registries, outcome trackers, options and last actions (no context carry-over)."""
        if self.env_ids:
            self.reset(list(self.env_ids))

    def reset_optimizer(self) -> None:
        """Drop optimiser moments (the trainer's own reset when it has one)."""
        if hasattr(self.trainer, "reset_optimizer"):
            self.trainer.reset_optimizer()
            return
        for opt in self.optimizers.values():
            opt.state.clear()

    def reset_experience_access(self) -> None:
        """Detach the archive, index, replay, the retriever's memory and the fresh buffer (weights-only evaluation)."""
        self.archive = None
        self.index = None
        self.replay = None
        self.consolidator = None
        self.fresh = []
        self._last_fresh = {}
        r = self.model.retriever
        if r is not None and hasattr(r, "reset_index"):
            r.reset_index()
        if r is not None and hasattr(r, "detach_index"):
            r.detach_index()

    def drop_report(self) -> dict:
        """Every budget that left data out (never silent): context tokens by kind, outcome targets / queries beyond
        the per-step budgets, Kalman channel evictions and keys that did not fit, fresh-buffer truncation."""
        calls = self.model.calls
        ctx = {k.split("/", 1)[1]: v for k, v in calls.items() if k.startswith("ctx_dropped/")}
        out = {"ctx_tokens": ctx, "channels_evicted": calls.get("channels.evicted", 0),
               "channels_unbound": calls.get("channels.unbound", 0), **self.dropped}
        tot = Counter()
        for tr in (self.state.trackers if self.state is not None else []):
            tot.update(tr.dropped)
        out["outcomes"] = {**{k: calls.get(f"outcome_dropped/{k}", 0) for k in ("numeric_targets",
                                                                                  "event_type_targets")},
                           "numeric_queries_live": tot.get("numeric_queries", 0),
                           "event_type_queries_live": tot.get("event_type_queries", 0)}
        return out

    # ---------------------------------------------------------------- episode control
    def set_goals(self, goals: Sequence[GoalSpec]) -> None:
        if self.env_ids and len(goals) != self.B:
            raise ValueError(f"set_goals: {len(goals)} goals for {self.B} environments")
        self.goals = list(goals)

    def reset(self, env_ids: Optional[Sequence[str]] = None) -> None:
        """Start new episodes. Unknown env ids (first call) define the batch; known ids reset their rows only.
        Undelivered post-action observations of the reset rows are archived first."""
        env_ids = list(env_ids) if env_ids is not None else list(self.env_ids)
        if self.state is None or any(e not in self.env_ids for e in env_ids):
            if self.state is not None and set(env_ids) != set(self.env_ids):
                raise ValueError("reset: a new batch must replace all environments at once")
            self.flush("reset")
            self.env_ids = env_ids
            B = len(env_ids)
            self.state = self.model.init_state(B, device=self.device, seed=self.py_rng.randrange(2**31))
            self.episode_ids = [new_id("ep_") for _ in range(B)]
            self.steps = [0] * B
            self._lp_valid = [False] * B
            self._t0 = [None] * B
            self._funds = [dict() for _ in range(B)]
            self._funds0 = [None] * B
            self._ep_decisions = [0] * B
            if len(self.goals) != B:
                self.goals = [GoalSpec("none", "") for _ in range(B)]
            self._pending = None
            return
        rows = [self.env_ids.index(e) for e in env_ids]
        self.flush("reset", rows)
        new = self.model.init_state(len(rows), device=self.device, seed=self.py_rng.randrange(2**31))
        self.state.scatter(rows, new)
        if self.state.fast_prev is not None and new.fast is not None:
            fast_scatter(self.state.fast_prev, rows, new.fast)
        for r in rows:
            self.episode_ids[r] = new_id("ep_")
            self.steps[r] = 0
            self._lp_valid[r] = False
            self._last_fresh.pop(r, None)
            self._last_eid.pop(r, None)
            self._t0[r] = None
            self._funds[r] = {}
            self._funds0[r] = None
            self._ep_decisions[r] = 0
        self._pending = None

    def _world_event(self, b: int, obs: list, t: float, attributed_to: Optional[str], done: bool, reason: str,
                     version: int) -> EventRecord:
        rec = EventRecord(event_id=new_id("w_"), env_id=self.env_ids[b], actor_id=self.actor_id,
                          episode_id=self.episode_ids[b], step=self.steps[b], model_version=version, t_sim=float(t),
                          observations=list(obs), action=None, result=None, goal_id=self.goals[b].goal_id,
                          done=done, provenance=Provenance.REAL, evidence=EvidenceKind.OBSERVATIONAL,
                          attributed_to=attributed_to,
                          extra={"kind": "post_action_observations", "reason": reason})
        self.steps[b] += 1
        return rec

    # ---------------------------------------------------------------- recorder adapters (list doubles / F's recorder)
    def _per_trace_recorder(self) -> bool:
        return self.recorder is not None and hasattr(self.recorder, "add_update_refs")

    def _record(self, traces: list) -> None:
        if self.recorder is None or not traces:
            return
        if self._per_trace_recorder():                 # F's TraceRecorder: one DecisionTrace per call
            for t in traces:
                self.recorder.record(t)
        else:
            self.recorder.record(traces)

    def _link_traces(self, decision_ids: list, events: list, update_id: str) -> None:
        """Link the traces of consumed decisions to a learner update: F's recorder by event id per lifetime
        (`events` = [(event_id, env_id)]), a list recorder by decision id."""
        if self.recorder is None:
            return
        if self._per_trace_recorder():
            by_env: dict = {}
            for e, env in events:
                by_env.setdefault(env, []).append(e)
            for env, eids in sorted(by_env.items(), key=lambda kv: str(kv[0])):
                self.recorder.add_update_refs(sorted(set(eids)), update_id, env_id=env)
        elif hasattr(self.recorder, "link_updates") and decision_ids:
            self.recorder.link_updates(decision_ids, update_id)

    def _archive(self, records: list) -> None:
        if not records or self.archive is None:
            return
        self.archive.append(records)
        if self.replay is not None and hasattr(self.replay, "add"):
            self.replay.add(records)
        # a records-level index is notified directly; E's MemoryIndex is fed by its Consolidator (update())
        if self.config.use_memory and self.index is not None and _takes_one_arg(getattr(self.index, "add", None)):
            self.index.add(records)

    def flush(self, reason: str = "end_of_run", rows: Optional[Sequence[int]] = None) -> list:
        """Archive the pending post-action observations (of `rows`, default all) as world-event records."""
        recs = []
        for b in sorted(self._post if rows is None else [r for r in rows if r in self._post]):
            obs, t, eid, done = self._post.pop(b)
            if obs:
                recs.append(self._world_event(b, obs, t, eid, done, reason, self.version))
        self._archive(recs)
        return recs

    # ---------------------------------------------------------------- budget
    def _absorb(self, b: int, obs: Sequence[Observation]) -> None:
        f = self.funds_fn(obs)
        if f:
            self._funds[b].update(f)
            if self._funds0[b] is None:
                self._funds0[b] = sum(self._funds[b].values())

    def _budget(self, rows: Sequence[int], now: Sequence[float]) -> BudgetState:
        out = []
        for j, b in enumerate(rows):
            g = self.goals[b]
            t0 = self._t0[b] if self._t0[b] is not None else now[j]
            dl = None if g is None or g.deadline_days is None else float(g.deadline_days)
            funds = sum(self._funds[b].values()) if self._funds[b] else None
            db = self.decision_budget
            out.append({"days_left": None if dl is None else dl - (now[j] - t0), "days_total": dl,
                        "funds_left": funds, "funds_total": self._funds0[b],
                        "decisions_left": None if db is None else db - self._ep_decisions[b],
                        "decisions_total": db})
        return BudgetState.from_rows(out, device=self.device)

    # ---------------------------------------------------------------- acting
    def act(self, obs_batch: ObservationBatch, tools_batch: Sequence[Sequence]) -> list:
        """One decision for every live environment (rows with obs_batch.done are skipped and get None)."""
        t0 = time.perf_counter()
        if self.state is None:
            self.reset(obs_batch.env_ids)
        if list(obs_batch.env_ids) != self.env_ids:
            raise ValueError("act: obs_batch.env_ids differ from the agent's environments (call reset first)")
        B = self.B
        if len(tools_batch) != B:
            raise ValueError(f"act: {len(tools_batch)} tool lists for {B} environments")
        rows = [b for b in range(B) if not obs_batch.done[b]]
        actions: list = [None] * B
        world = []
        for b in rows:                  # post-action observations not handed to this decision are archived as-is
            if b in self._post:
                obs, t, eid, done = self._post.pop(b)
                given = {_obs_key(o) for o in obs_batch.observations[b]}
                rest = [o for o in obs if _obs_key(o) not in given]
                if rest:
                    world.append(self._world_event(b, rest, t, eid, False, "not_in_next_decision", self.version))
        self._archive(world)
        if not rows:
            self._pending = None
            return actions
        now = [float(obs_batch.t[b]) for b in rows]
        for j, b in enumerate(rows):
            if self._t0[b] is None:
                self._t0[b] = now[j]
            self._absorb(b, obs_batch.observations[b])
        budget = self._budget(rows, now)
        sub = self.state.select(rows)
        obs_lists = [list(obs_batch.observations[b]) for b in rows]
        tools = [list(tools_batch[b]) for b in rows]
        goals = [public_goal(self.goals[b]) for b in rows]
        self.model.eval()
        dec = self.model.decide(sub, obs_lists, now, tools, goals, budget=budget, greedy=self.greedy,
                                generator=self.torch_rng)
        self.state.scatter(rows, sub)
        for j, b in enumerate(rows):
            actions[b] = dec.actions[j]
        # event id = the executed action's id: observations the environment attributes to the action
        # (Observation.caused_by = action_id) then resolve to its archived record
        ids = [(dec.actions[j].action_id or new_id("e_"), new_id("d_")) for j in range(len(rows))]
        regs = [list(self.state.registries[b].slot.items()) for b in rows]
        self._pending = (rows, dec, now, ids, self.version, regs)
        self.timing["act_s"] += time.perf_counter() - t0
        return actions

    def _context_refs(self, refs: Sequence, obs: list, eid: str, b: int, reg_items: list) -> list:
        """Map the context's token refs to resolvable references: observation i of this record -> "<event_id>#obs:i"
        (+ its artifact ref / causing event id), memory tokens -> their source event ids, the previous action token
        -> the previous action's event id, entity / relation slots -> "entity:<persistent id>" / "relation:<id>-><id>";
        other refs (goal ids) as-is."""
        slot2id = {s: e for e, s in reg_items}
        out = []
        for r in refs:
            if r is None:
                continue
            if isinstance(r, (tuple, list)):
                out.extend(str(x) for x in r)
                continue
            r = str(r)
            m = OBS_REF.match(r)
            if m:
                i = int(m.group(1))
                out.append(f"{eid}#obs:{i}")
                if i < len(obs):
                    if obs[i].artifact_ref:
                        out.append(obs[i].artifact_ref)
                    if obs[i].caused_by:
                        out.append(obs[i].caused_by)
                continue
            if r.startswith("mem:"):
                out.append(r[4:])
                continue
            if r.startswith("seg:"):                   # E's memory segment: the segment and its source event ids
                out.append(r)
                ret = self.model.retriever
                if ret is not None and callable(getattr(ret, "resolve_ref", None)):
                    try:
                        out.extend(str(e) for e in ret.resolve_ref(r))
                    except (KeyError, RuntimeError, ValueError):
                        pass
                continue
            if r.startswith("act:"):
                if self._last_eid.get(b):
                    out.append(self._last_eid[b])
                continue
            m = SLOT_REF.match(r)
            if m:
                e = slot2id.get(int(m.group(1)))
                if e is not None:
                    out.append(f"entity:{e}")
                continue
            m = REL_REF.match(r)
            if m:
                ei, ej = slot2id.get(int(m.group(1))), slot2id.get(int(m.group(2)))
                if ei is not None and ej is not None:
                    out.append(f"relation:{ei}->{ej}")
                continue
            out.append(r)
        return list(dict.fromkeys(out))

    def observe(self, results, rewards: Optional[Sequence[float]] = None,
                next_obs: Optional[ObservationBatch] = None) -> list:
        """Consequences of the last act(): outcomes -> beliefs/Bayes updates, EventRecords -> archive (+ index,
        replay notification), DecisionTraces -> recorder, fresh buffer for the policy objective. Returns the records
        (including world-event records of post-action observations at episode ends).

        Two call shapes: observe(results [B], rewards [B], next_obs) (runtime.run_steps), or observe(transition) with
        a lifetime-harness `envs.agent_protocol.Transition` (records [B] carrying the ExecutionResults, next_obs,
        rewards = verified return increments). Only those fields are read: the harness's own records are its
        transcript; the agent archives its own records (with its predictions, behaviour provenance, outcomes)."""
        if rewards is None and next_obs is None and hasattr(results, "records") and hasattr(results, "next_obs"):
            tr = results
            results = [None if r is None else r.result for r in tr.records]
            rewards, next_obs = list(tr.rewards), tr.next_obs
        t0 = time.perf_counter()
        if self._pending is None:
            return []
        rows, dec, now, ids, version, regs = self._pending
        self._pending = None
        sub = self.state.select(rows)
        res = [results[b] for b in rows]
        if any(r is None for r in res):
            raise ValueError("observe: a live environment returned no ExecutionResult")
        rew = [float(rewards[b]) for b in rows]
        nxt = [list(next_obs.observations[b]) for b in rows]
        done = [bool(next_obs.done[b]) for b in rows]
        info = self.model.observe_outcomes(sub, dec, res, nxt, rew)
        self.state.scatter(rows, sub)
        lp = info["lp"]
        dt_actual = [max(0.0, float(next_obs.t[b]) - now[j]) for j, b in enumerate(rows)]
        records, traces = [], []
        lp_prev = [0.0] * len(rows)
        for j, b in enumerate(rows):
            if lp is not None and self._lp_valid[b]:
                lp_prev[j] = float(lp[j])
                prev = self._last_fresh.get(b)
                if prev is not None:
                    prev[0].lp[prev[1]] = lp_prev[j]
            self._lp_valid[b] = self.config.use_fast_adaptation
        bud = dec.inputs.budget
        for j, b in enumerate(rows):
            a, r = dec.actions[j], res[j]
            eid, did = ids[j]
            ents = sorted({e for o in dec.inputs.obs_lists[j] for e in (o.entity, o.target) if e}
                          | {e for e in a.entities if isinstance(e, str)})
            outcome = {s.key: s.y for s in info["specs"][j] if s.observed}
            dropped = {**dec.dropped[j], "outcome_targets": dict(info["dropped"][j])}
            g = public_goal(self.goals[b])
            extra = {"outcomes": [s.to_dict() for s in info["specs"][j]], "dt": dt_actual[j],
                     "dt_pred": float(dec.dt[j]), "goal": _goal_dict(g), "value": float(dec.value[j]),
                     "decision_id": did, "lp_prev": lp_prev[j], "retrieved": [x["id"] for x in dec.retrieved[j]],
                     "reasoning_iters": dec.reasoning[j]["iters"], "option": a.option_id,
                     "option_steps": int(dec.inputs.option_steps[j]), "option_new": bool(dec.inputs.option_new[j]),
                     "behavior": dict(dec.behavior[j]), "channels": dict(info["channels"][j]), "dropped": dropped,
                     "budget": None if bud is None else bud.row(j),
                     "belief_loglik": None if info["loglik"] is None else float(info["loglik"][j])}
            rec = EventRecord(event_id=eid, env_id=self.env_ids[b], actor_id=self.actor_id,
                              episode_id=self.episode_ids[b], step=self.steps[b], model_version=version, t_sim=now[j],
                              observations=list(dec.inputs.obs_lists[j]), action=a, result=r,
                              predicted=dict(dec.predicted[j]), uncertainty=dict(dec.uncertainty[j]),
                              goal_id=g.goal_id, reward=rew[j], cost=r.cost, done=done[j], entity_links=ents,
                              behavior_logp=float(dec.logp[j]), provenance=Provenance.REAL,
                              evidence=EvidenceKind.CONTROLLED, extra=extra)
            records.append(rec)
            traces.append(DecisionTrace(
                decision_id=did, event_id=eid, env_id=self.env_ids[b], policy_version=version, t_sim=now[j],
                objective=g.text,
                context_refs=self._context_refs(dec.ctx.refs[j] if dec.ctx.refs else [], dec.inputs.obs_lists[j],
                                                eid, b, regs[j]),
                retrieved=list(dec.retrieved[j]), candidates=list(dec.candidates[j]),
                predicted=dict(dec.predicted[j]), uncertainty=dict(dec.uncertainty[j]),
                imagined=list(dec.imagined[j]), selected=a.to_dict(), reasoning=dict(dec.reasoning[j]),
                belief_before=dict(dec.belief_before[j]) if dec.belief_before else {},
                belief_after=dict(info["belief_after"][j]) if info["belief_after"] else {}, result_ref=eid,
                outcome={"ok": r.ok, "error": r.error, "reward": rew[j], "values": outcome, "lp_prev": lp_prev[j],
                         "dt": dt_actual[j], "dt_pred": float(dec.dt[j]), "behavior": dict(dec.behavior[j]),
                         "channels": dict(info["channels"][j]), "dropped": dropped},
                update_refs=[], source=a.source.value))
            self.steps[b] += 1
            self._ep_decisions[b] += 1
            self._last_eid[b] = eid
            self._absorb(b, nxt[j])
            if done[j]:                 # no later decision of this episode will carry these observations
                if nxt[j]:
                    records.append(self._world_event(b, nxt[j], float(next_obs.t[b]), eid, True, "episode_end",
                                                     version))
            else:
                self._post[b] = (nxt[j], float(next_obs.t[b]), eid, False)
        self._archive(records)
        self._record(traces)
        dev = self.device
        planner = torch.tensor([str(dec.behavior[j]["policy"]).startswith("planner") for j in range(len(rows))],
                               device=dev)
        fs = FreshStep(list(rows), [self.episode_ids[b] for b in rows], dec.inputs, list(dec.actions),
                       dec.logp.detach().float(), dec.value.detach().float(), torch.tensor(rew, device=dev),
                       torch.tensor(dt_actual, device=dev), torch.tensor(done, device=dev), version,
                       [ids[j][1] for j in range(len(rows))], torch.zeros(len(rows), device=dev), planner,
                       [ids[j][0] for j in range(len(rows))], [self.env_ids[b] for b in rows])
        self.fresh.append(fs)
        for j, b in enumerate(rows):
            self._last_fresh[b] = (fs, j)
        if len(self.fresh) > self.fresh_max_steps:
            lost = self.fresh[: len(self.fresh) - self.fresh_max_steps]
            self.dropped["fresh_steps"] += len(lost)
            self.dropped["fresh_decisions"] += sum(len(s.rows) for s in lost)
            self.fresh = self.fresh[-self.fresh_max_steps:]
        self.n_decisions += len(rows)
        self.timing["observe_s"] += time.perf_counter() - t0
        return records

    # ---------------------------------------------------------------- learning
    def tokenise_sequences(self, seqs: Sequence[Sequence[EventRecord]]) -> tuple:
        """Archived decision EventRecords -> (SequenceBatch, ReplayAux) (aligned; shorter sequences padded with
        masked steps marked done). Observations are re-tokenised with the current tokenizer (fresh episode-local
        registries); outcomes come from the records' realised outcome specs; goals from their agent-visible goal
        fields; the aux carries the stored retrieved ids and option decisions for the acting-consistent filter.
        The `channel` field binds observed continuous keys to slots in first-seen order (as the Bayes layer would);
        aux.tables holds the binding."""
        m = self.model
        cfg = self.config
        B = len(seqs)
        T = max(len(s) for s in seqs)
        dev = self.device
        regs = [EntityRegistry(cfg.max_entities, seed=self.py_rng.randrange(2**31)) for _ in range(B)]
        C = m.n_channels
        tables = [dict() for _ in range(B)]
        last_goal = [GoalSpec("none", "") for _ in range(B)]
        steps = []
        aux = ReplayAux([], [], [], [])
        prev_opt = [-1] * B
        prev_steps = [0] * B
        for t in range(T):
            recs = [s[t] if t < len(s) else None for s in seqs]
            obs_lists = [list(r.observations) if r is not None else [] for r in recs]
            now = [float(r.t_sim) if r is not None else 0.0 for r in recs]
            O = check_tokens(m.tokenizer.observations(obs_lists, regs, now), B, cfg.d_model, "replay observations")
            if m.entity_encoder is not None:
                O = TokenSet.cat(O, m.entity_encoder(O))
            act = m.tokenizer.action([r.action if r is not None else None for r in recs])
            specs = [[OutcomeSpec.from_dict(d) for d in (r.extra.get("outcomes", []) if r is not None else [])]
                     for r in recs]
            chans = []
            for b in range(B):
                row = []
                for s in specs[b]:
                    if C and int(s.lik) in CONTINUOUS and s.observed and s.key not in tables[b] \
                            and len(tables[b]) < C:
                        tables[b][s.key] = len(tables[b])
                    row.append(tables[b].get(s.key, -1) if int(s.lik) in CONTINUOUS else -1)
                chans.append(row)
            outcomes = build_outcome_batch(specs, m.query_encoder, regs, chans, dev)
            for b, r in enumerate(recs):
                if r is not None and r.extra.get("goal"):
                    g = r.extra["goal"]
                    last_goal[b] = GoalSpec(g["goal_id"], g["text"], dict(g.get("metrics") or {}),
                                            g.get("deadline_days"))
            goal = m.tokenizer.goal(list(last_goal))
            aux.retrieved.append([list(r.extra.get("retrieved") or []) if r is not None else [] for r in recs])
            aux.now.append(now)
            aux.option_prev.append(list(prev_opt))
            aux.option_steps_prev.append(list(prev_steps))
            for b, r in enumerate(recs):
                o = r.extra.get("option") if r is not None else None
                prev_opt[b] = -1 if o is None or (r is not None and r.done) else int(o)
                prev_steps[b] = int(r.extra.get("option_steps", 0)) if r is not None and o is not None else 0
            steps.append(StepBatch(
                obs=O, act=act, outcomes=outcomes,
                dt=torch.tensor([float(r.extra.get("dt", 0.0)) if r is not None else 0.0 for r in recs], device=dev),
                reward=torch.tensor([float(r.reward) if r is not None else 0.0 for r in recs], device=dev),
                done=torch.tensor([bool(r.done) if r is not None else True for r in recs], device=dev),
                behavior_logp=torch.tensor([float(r.behavior_logp or 0.0) if r is not None else 0.0 for r in recs],
                                           device=dev),
                model_version=torch.tensor([int(r.model_version) if r is not None else -1 for r in recs],
                                           dtype=torch.long, device=dev),
                goal=goal, event_ids=[r.event_id if r is not None else None for r in recs]))
        aux.tables = tables
        return SequenceBatch(steps), aux

    def _sample(self, n: int, length: int) -> list:
        """Replay draws as lists of decision records (world-event records are archived but not trained on)."""
        out = []
        for s in self.replay.sample(n, length):
            recs = s.records if hasattr(s, "records") else s
            recs = [r for r in (recs or []) if r.action is not None]
            if self.exclude_events:
                keep = [r for r in recs if r.event_id not in self.exclude_events]
                self.provenance_stats["replay_records_excluded"] += len(recs) - len(keep)
                recs = keep
            if recs:
                out.append(recs)
        return out

    def _lora_trained(self) -> bool:
        """True when the semantic model's trainable (LoRA) tensors are among the optimised parameters."""
        sem = self.model.semantic
        lora = list(sem.lora_parameters()) if callable(getattr(sem, "lora_parameters", None)) else []
        if not lora:
            return False
        ids = {id(p) for opt in self.optimizers.values() for g in opt.param_groups for p in g["params"]}
        return any(id(p) in ids for p in lora)

    def _trainer_step(self, losses: dict, consumed: list) -> dict:
        fn = self.trainer.step
        names = set()
        try:
            names = set(inspect.signature(fn).parameters)
        except (TypeError, ValueError):
            pass
        kw = {}
        if "model" in names:
            kw["model"] = self.model
        # `consumed` is not passed: D's CombinedTrainer mutates record objects there, while this agent links the
        # consumed decisions itself (recorder.link_updates with the trainer's update id, update_log)
        out = fn(losses, **kw)
        if hasattr(out, "to_dict"):
            out = out.to_dict()
        return dict(out or {})

    def update(self) -> dict:
        """One learner update (spec 12.5). Returns metrics (losses, per-objective grad contributions from the
        trainer, consumed event/decision ids, the support sizes drawn so far)."""
        cfg = self.config
        if not cfg.continual_updates:
            return {"updated": False, "reason": "continual_updates=False", "version": self.version}
        if self.trainer is None:
            raise RuntimeError("CamUnifiedAgent.update() needs an injected trainer (workstream D's CombinedTrainer)")
        t0 = time.perf_counter()
        seq = aux = aseq = aaux = None
        seqs, aseqs = [], []
        replay_ids, replay_decisions = [], []
        consolidated = None
        if self.config.use_memory and self.consolidator is not None and hasattr(self.consolidator, "step"):
            consolidated = self.consolidator.step()     # new archived events -> memory segments (E)
            if hasattr(self.consolidator, "ensure_current"):
                self.consolidator.ensure_current()      # first segments: the index starts serving their version
        if self.replay is not None and self.archive is not None and self.archive.count() > 0:
            if hasattr(self.replay, "sync"):
                self.replay.sync()                      # E's sampler: membership from the archive (CDC)
            seqs = self._sample(self.n_seqs, self.seq_len)
            if seqs:
                seq, aux = self.tokenise_sequences(seqs)
            aseqs = self._sample(self.adapt_n_seqs, self.adapt_seq_len) if cfg.use_fast_adaptation else []
            if aseqs:
                aseq, aaux = self.tokenise_sequences(aseqs)
            for s in seqs + aseqs:
                for r in s:
                    replay_ids.append(r.event_id)
                    if r.extra.get("decision_id"):
                        replay_decisions.append(r.extra["decision_id"])
            replay_ids, replay_decisions = sorted(set(replay_ids)), sorted(set(replay_decisions))
        self.model.train()
        batch = UpdateBatch(seq, aux, list(self.fresh), self.model, aseq, aaux)
        losses, metrics, consumed, consumed_events = {}, {}, [], []
        for name, fn in self.objectives.items():
            out = fn(self, batch)
            if not out:
                continue
            for k, v in out.get("losses", {}).items():
                if not torch.is_tensor(v) or not torch.isfinite(v).all():
                    raise FloatingPointError(f"objective {name!r} produced a non-finite loss {k!r}: {v}")
                if not v.requires_grad:          # e.g. a replay window too short for the objective: no signal
                    metrics[f"skipped/{k}"] = metrics.get(f"skipped/{k}", 0) + 1
                    continue
                losses[k] = v
            metrics.update(out.get("metrics", {}))
            consumed += out.get("consumed", [])
            consumed_events += out.get("consumed_events", [])
        if not losses:
            self.model.eval()
            return {"updated": False, "reason": "no training data yet", "version": self.version}
        raw_losses = {k: float(v.detach()) for k, v in losses.items()}
        if self.loss_cap is not None:
            for k, v in list(losses.items()):
                m = abs(raw_losses[k])
                prev = self.loss_mag.get(k)
                self.loss_mag[k] = m if prev is None else self.loss_cap_beta * prev + (1 - self.loss_cap_beta) * m
                sc = min(1.0, float(self.loss_cap) / max(self.loss_mag[k], 1e-12))
                metrics[f"balance/{k}"] = sc
                if sc < 1.0:
                    losses[k] = v * sc
        tm = self._trainer_step(losses, consumed)
        self.model.eval()
        if tm.get("skipped"):                         # D's trainer refused a non-finite loss / gradient: no step
            self.timing["update_s"] += time.perf_counter() - t0
            return {"updated": False, "reason": "trainer skipped the step (non-finite)", "version": self.version,
                    **{f"trainer/{k}": v for k, v in tm.items()}}
        self.version = self.version + 1               # bumps the weights epoch (query-embedding cache invalid)
        sem = self.model.semantic
        if sem is not None and callable(getattr(sem, "mark_updated", None)) and self._lora_trained():
            sem.mark_updated()                        # A's decoder_version (and encoder cache key with encoder LoRA)
        update_id = str(tm.get("update_id") or f"u_{self.actor_id}_{self.version}")
        linked = sorted(set(consumed) | set(replay_decisions))
        replay_env = {}
        for s_ in list(seqs) + list(aseqs):
            for r in s_:
                replay_env[r.event_id] = r.env_id
        events = sorted(set(consumed_events) | set(replay_env.items()))
        self._link_traces(linked, events, update_id)
        if self.archive is not None and hasattr(self.archive, "mark_used") and events:
            # training provenance: EventRecord.training_provenance["used_in_updates"] of every consumed event
            self.provenance_stats["marked_events"] += int(self.archive.mark_used([e for e, _ in events], update_id))
        mem_used = sorted({str(i) for st in batch.fresh for ids in (st.inputs.memory_ids if st.inputs else [])
                           for i in ids})
        if mem_used and self.consolidator is not None and hasattr(self.consolidator, "link_weights"):
            self.consolidator.link_weights(mem_used, update_id, self.version)   # parametric-knowledge provenance
        # keep each environment's latest undone decision: it bootstraps the next update's advantages
        keep: dict = {}
        for st in self.fresh:
            for j, r in enumerate(st.rows):
                keep[r] = (st, j)
        survivors = []
        for r, (st, j) in sorted(keep.items()):
            if not bool(st.done[j]):
                sub = st.subset([j])
                survivors.append(sub)
                self._last_fresh[r] = (sub, 0)
            else:
                self._last_fresh.pop(r, None)
        self.fresh = survivors
        self.timing["update_s"] += time.perf_counter() - t0
        entry = {"update_id": update_id, "version": self.version, "losses": raw_losses,
                 "scaled_losses": {k: float(v.detach()) for k, v in losses.items()},
                 "consumed_decisions": consumed, "consumed_events": sorted({e for e, _ in consumed_events}),
                 "replay_events": replay_ids, "replay_decisions": replay_decisions,
                 "adapt_seq_T": None if aseq is None else len(aseq.steps), "memory_linked": mem_used,
                 "consolidated": consolidated,
                 "adapt_k_hist": {int(k): v for k, v in sorted(self.adapt_k_hist.items())}}
        self.update_log.append(entry)
        return {"updated": True, **entry, **metrics, **{f"trainer/{k}": v for k, v in (tm or {}).items()}}
