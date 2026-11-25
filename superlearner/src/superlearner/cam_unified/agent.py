"""CamUnifiedAgent: per-environment state around CamUnifiedModel (spec Parts 11, 12, 14, 18.1, 20 A22).

AgentP shape (used by runtime.run_steps and, later, workstream F's actors and G's OpenWorld harness):

    act(obs_batch: ObservationBatch, tools_batch: [B] list[ToolSchema]) -> [B] Action | None   (None = done env)
    observe(results: [B] ExecutionResult, rewards: [B] float, next_obs: ObservationBatch) -> None
    update() -> dict                                    (one learner step; no-op when cfg.continual_updates is False)
    reset(env_ids: list[str] | None) -> None

Per environment the agent keeps Z_t, the FastState, an EntityRegistry (episode-local slots), the outcome tracker, the
active option, the last executed action and the goal. Every executed decision becomes one `EventRecord`
(provenance REAL, behaviour log-probability, predictions + uncertainty made before acting, model_version, realised
outcome specs) appended to the injected archive, and one `DecisionTrace` sent to the injected recorder.

`update()` builds the loss terms from (a) sequences sampled by the injected replay and tokenised on the fly from
archived EventRecords (world model, support/query adaptation, memory) and (b) the fresh on-policy buffer (PPO-clip
actor + value with GAE under the semi-Markov discount gamma = exp(-kappa * dt), plus the reasoner's compute cost),
then hands them to the injected trainer, which weights, back-propagates and steps the optimisers.

Attribute contract for checkpoint operations (interfaces.CheckpointOpsP, workstream E):
    model, config, optimizers, fast, archive, index, replay, rng, version  (+ reset_fast_state, reset_optimizer,
    reset_experience_access, reset_working_state).
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

import torch

from .config import CamConfig
from .interfaces import (DecisionTrace, EntityRegistry, EventRecord, EvidenceKind, ExecutionResult, GoalSpec,
                         ObservationBatch, Provenance, SequenceBatch, StepBatch, TokenSet, new_id)
from .model import CamState, CamUnifiedModel, StepInputs, check_tokens, fast_scatter
from .outcome_spec import OutcomeSpec, build_outcome_batch

SUPPORT_SIZES = (0, 1, 2, 4, 8, 16, 32, 64)


# ============================================================================ protocols (not yet in interfaces.py)

class AgentP(Protocol):
    """What runtimes, actors (F) and the OpenWorld harness (G) drive."""
    def act(self, obs_batch: ObservationBatch, tools_batch: Sequence[Sequence]) -> list: ...   # [B] Action | None
    def observe(self, results: Sequence, rewards: Sequence[float], next_obs: ObservationBatch) -> list: ...
    def update(self) -> dict: ...
    def reset(self, env_ids: Optional[Sequence[str]] = None) -> None: ...
    def set_goals(self, goals: Sequence[GoalSpec]) -> None: ...

class ReplayP(Protocol):                      # workstream E: cam_unified/memory/replay.py
    def sample(self, n_seqs: int, seq_len: int) -> list: ...   # -> list[list[EventRecord]], consecutive steps of one episode
    # optional: add(records: list[EventRecord]) -> None (notification of newly archived records)


class TraceRecorderP(Protocol):               # workstream F: cam_unified/dist/trace.py
    def record(self, traces: list) -> None: ...                # list[DecisionTrace]
    # optional: link_updates(decision_ids: list[str], update_id: str) -> None


class TrainerP(Protocol):                     # workstream D: cam_unified/learn/trainer.py
    optimizers: dict                                           # name -> torch.optim.Optimizer
    def step(self, losses: dict, model) -> dict: ...           # weights (cfg.loss_weights), backward, per-objective
                                                               # grad contributions into the core, clip, optimiser step
    # optional: reset_optimizer() -> None


class MemoryIndexP(Protocol):                 # workstream E: cam_unified/memory/index.py
    def add(self, records: list) -> None: ...


# ============================================================================ fresh on-policy buffer

@dataclass
class FreshStep:
    """One decision of the fresh on-policy buffer (sub-batch of live rows)."""
    rows: list                      # agent rows
    episode_ids: list
    inputs: StepInputs
    actions: list
    logp_old: torch.Tensor          # [b]
    value_old: torch.Tensor         # [b]
    reward: torch.Tensor            # [b] verified extrinsic return increment
    dt: torch.Tensor                # [b] simulated days until the next decision
    done: torch.Tensor              # [b] bool
    version: int
    decision_ids: list
    lp: torch.Tensor = None         # [b] hindsight learning progress (filled one decision later)

    def subset(self, js: Sequence[int]) -> "FreshStep":
        idx = torch.tensor(list(js), dtype=torch.long, device=self.logp_old.device)
        return FreshStep([self.rows[j] for j in js], [self.episode_ids[j] for j in js],
                         select_inputs(self.inputs, js), [self.actions[j] for j in js], self.logp_old[idx],
                         self.value_old[idx], self.reward[idx], self.dt[idx], self.done[idx], self.version,
                         [self.decision_ids[j] for j in js], self.lp[idx])


def select_inputs(inp: StepInputs, js: Sequence[int]) -> StepInputs:
    js = list(js)
    idx = torch.tensor(js, dtype=torch.long, device=inp.Z_prev.device)
    return StepInputs(inp.Z_prev[idx], None if inp.fast is None else inp.fast.select(idx),
                      [inp.registries[j] for j in js], [inp.obs_lists[j] for j in js], [inp.now[j] for j in js],
                      [inp.last_actions[j] for j in js], [inp.goals[j] for j in js], [inp.tools[j] for j in js],
                      None if inp.memory is None else inp.memory.select(idx), inp.option_prev[idx], inp.option[idx],
                      inp.option_switched[idx], inp.option_evaluated[idx], inp.option_terminated[idx])


@dataclass
class UpdateBatch:
    """Data of one learner update. `seq`: tokenised replay sequences (or None); `fresh`: FreshSteps."""
    seq: Optional[SequenceBatch]
    fresh: list
    channel_tables: Optional[list] = None
    _Zs: Optional[list] = None
    model: Optional[CamUnifiedModel] = None

    def filtered(self) -> list:
        """Z_t over the replay sequences (computed once, shared by the adapt and memory objectives)."""
        if self._Zs is None:
            self._Zs = self.model.filter_sequence(self.seq)
        return self._Zs


# ============================================================================ default objectives (replaceable)

def objective_world(agent: "CamUnifiedAgent", batch: UpdateBatch) -> dict:
    """L_world from WorldModelP.loss(SequenceBatch): expects key "world" (the total); scalar entries become metrics."""
    if batch.seq is None:
        return {}
    out = agent.model.world_model.loss(batch.seq)
    if "world" not in out:
        raise KeyError(f"WorldModelP.loss must return the total under 'world'; got keys {sorted(out)}")
    mets = {f"world/{k}": float(v.detach()) for k, v in out.items()
            if torch.is_tensor(v) and v.numel() == 1 and k != "world"}
    return {"losses": {"world": out["world"]}, "metrics": mets}


def objective_adapt(agent: "CamUnifiedAgent", batch: UpdateBatch) -> dict:
    """L_adapt with a support size k drawn from SUPPORT_SIZES (k < T, so at least one query step remains)."""
    cfg = agent.config
    if batch.seq is None or not cfg.use_fast_adaptation:
        return {}
    T = len(batch.seq.steps)
    ks = [k for k in SUPPORT_SIZES if k < T]
    k = agent.py_rng.choice(ks)
    loss, mets = agent.model.adapt_loss(batch.seq, batch.filtered(), k, batch.channel_tables)
    return {"losses": {"adapt": loss}, "metrics": {f"adapt/{n}": v for n, v in mets.items()}}


def objective_memory(agent: "CamUnifiedAgent", batch: UpdateBatch) -> dict:
    """L_memory from the retriever's own objective (RetrieverP.loss(seq, Zs)), when it defines one."""
    r = agent.model.retriever
    if batch.seq is None or not agent.config.use_memory or r is None or not hasattr(r, "loss"):
        return {}
    return {"losses": {"memory": r.loss(batch.seq, batch.filtered())}}


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
    """PPO-clip actor + value on the fresh on-policy buffer (spec 12.3), the reasoner's compute cost (architecture
    §4) and the entropy bonus. Behaviour log-probabilities are recomputed through the full step (tokenizer -> core ->
    reasoner -> option policy -> action generator) from stored inputs, so gradients reach every one of them."""
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
    adv = torch.tensor([targets[(id(st), j)][0] for st, js in chosen for j in js], device=agent.device)
    ret = torch.tensor([targets[(id(st), j)][1] for st, js in chosen for j in js], device=agent.device)
    logp_old = torch.cat([s.logp_old for s in subs]).to(agent.device)
    lag = torch.tensor([agent.version - s.version for s in subs for _ in s.rows], dtype=torch.float)
    out = agent.model.policy_evaluate(inp, actions)
    logp, ent, value = out["logp"], out["entropy"], out["value"]
    a_n = (adv - adv.mean()) / (adv.std() + 1e-8) if adv.numel() > 1 else adv
    ratio = torch.exp(logp - logp_old)
    clipped = torch.clamp(ratio, 1 - cfg.ppo_clip, 1 + cfg.ppo_clip)
    actor = -torch.min(ratio * a_n, clipped * a_n).mean() - cfg.entropy_coef * ent.mean()
    vloss = ((value - ret) ** 2).mean()
    losses = {"actor": actor, "value": vloss}
    if out.get("ponder") is not None and cfg.use_reasoning and cfg.adaptive_reasoning:
        losses["reason_compute"] = cfg.reason_compute_cost * out["ponder"].mean()
    mets = {"policy/n": int(adv.numel()), "policy/ratio_mean": float(ratio.detach().mean()),
            "policy/ratio_maxdev": float((ratio.detach() - 1).abs().max()),
            "policy/approx_kl": float((logp_old - logp.detach()).mean()), "policy/entropy": float(ent.detach().mean()),
            "policy/adv_mean": float(adv.mean()), "policy/return_mean": float(ret.mean()),
            "policy/max_version_lag": float(lag.max()) if lag.numel() else 0.0}
    return {"losses": losses, "metrics": mets, "consumed": [d for s in subs for d in s.decision_ids]}


DEFAULT_OBJECTIVES = {"world": objective_world, "adapt": objective_adapt, "memory": objective_memory,
                      "policy": objective_policy}


# ============================================================================ the agent

def _public_goal(g: Optional[GoalSpec]) -> Optional[dict]:
    """Agent-visible goal fields only (family and meta are bookkeeping, never model inputs)."""
    if g is None:
        return None
    return {"goal_id": g.goal_id, "text": g.text, "metrics": dict(g.metrics), "deadline_days": g.deadline_days}


class CamUnifiedAgent:
    """Batched agent over B environments (see module docstring). All collaborators are injected; `archive`,
    `recorder`, `replay`, `index` and `trainer` may be None (then the corresponding path is skipped and reported)."""

    def __init__(self, model: CamUnifiedModel, *, archive=None, recorder=None, replay=None, trainer=None, index=None,
                 actor_id: str = "actor0", seed: int = 0, seq_len: int = 8, n_seqs: int = 4, greedy: bool = False,
                 objectives: Optional[dict] = None, fresh_max_steps: int = 256):
        self.model = model
        self.config: CamConfig = model.cfg
        self.archive = archive
        self.recorder = recorder
        self.replay = replay
        self.trainer = trainer
        self.index = index
        self.actor_id = actor_id
        self.seq_len = seq_len
        self.n_seqs = n_seqs
        self.greedy = greedy
        self.objectives = dict(DEFAULT_OBJECTIVES if objectives is None else objectives)
        self.fresh_max_steps = fresh_max_steps
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
        self._pending = None
        self._last_fresh: dict = {}
        self._lp_valid: list = []
        self.timing = {"act_s": 0.0, "observe_s": 0.0, "update_s": 0.0}

    # ---------------------------------------------------------------- checkpoint attribute contract
    @property
    def rng(self) -> dict:
        return {"python": self.py_rng, "torch": self.torch_rng}

    @property
    def version(self) -> int:
        return self.model.version

    @version.setter
    def version(self, v: int) -> None:
        self.model.version = int(v)

    @property
    def optimizers(self) -> dict:
        return dict(getattr(self.trainer, "optimizers", {}) or {})

    @property
    def fast(self):
        return None if self.state is None else self.state.fast

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
        self.state.fast = self.model.beliefs.init_fast(self.B, device=self.device)
        self.state.fast_prev = None
        self._lp_valid = [False] * self.B

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
        self.fresh = []
        self._last_fresh = {}
        r = self.model.retriever
        if r is not None and hasattr(r, "reset_index"):
            r.reset_index()

    # ---------------------------------------------------------------- episode control
    def set_goals(self, goals: Sequence[GoalSpec]) -> None:
        if self.env_ids and len(goals) != self.B:
            raise ValueError(f"set_goals: {len(goals)} goals for {self.B} environments")
        self.goals = list(goals)

    def reset(self, env_ids: Optional[Sequence[str]] = None) -> None:
        """Start new episodes. Unknown env ids (first call) define the batch; known ids reset their rows only."""
        env_ids = list(env_ids) if env_ids is not None else list(self.env_ids)
        if self.state is None or any(e not in self.env_ids for e in env_ids):
            if self.state is not None and set(env_ids) != set(self.env_ids):
                raise ValueError("reset: a new batch must replace all environments at once")
            self.env_ids = env_ids
            B = len(env_ids)
            self.state = self.model.init_state(B, device=self.device, seed=self.py_rng.randrange(2**31))
            self.episode_ids = [new_id("ep_") for _ in range(B)]
            self.steps = [0] * B
            self._lp_valid = [False] * B
            if len(self.goals) != B:
                self.goals = [GoalSpec("none", "") for _ in range(B)]
            self._pending = None
            return
        rows = [self.env_ids.index(e) for e in env_ids]
        new = self.model.init_state(len(rows), device=self.device, seed=self.py_rng.randrange(2**31))
        self.state.scatter(rows, new)
        if self.state.fast_prev is not None and new.fast is not None:
            fast_scatter(self.state.fast_prev, rows, new.fast)
        for r in rows:
            self.episode_ids[r] = new_id("ep_")
            self.steps[r] = 0
            self._lp_valid[r] = False
            self._last_fresh.pop(r, None)
        self._pending = None

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
        if not rows:
            self._pending = None
            return actions
        sub = self.state.select(rows)
        obs_lists = [list(obs_batch.observations[b]) for b in rows]
        now = [float(obs_batch.t[b]) for b in rows]
        tools = [list(tools_batch[b]) for b in rows]
        goals = [self.goals[b] for b in rows]
        self.model.eval()
        dec = self.model.decide(sub, obs_lists, now, tools, goals, greedy=self.greedy, generator=self.torch_rng)
        self.state.scatter(rows, sub)
        for j, b in enumerate(rows):
            actions[b] = dec.actions[j]
        ids = [(new_id("e_"), new_id("d_")) for _ in rows]
        self._pending = (rows, dec, now, ids, self.version)
        self.timing["act_s"] += time.perf_counter() - t0
        return actions

    def observe(self, results: Sequence[Optional[ExecutionResult]], rewards: Sequence[float],
                next_obs: ObservationBatch) -> list:
        """Consequences of the last act(): outcomes -> beliefs/Bayes updates, EventRecords -> archive (+ index,
        replay notification), DecisionTraces -> recorder, fresh buffer for the policy objective. Returns the records."""
        t0 = time.perf_counter()
        if self._pending is None:
            return []
        rows, dec, now, ids, version = self._pending
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
        for j, b in enumerate(rows):
            a, r = dec.actions[j], res[j]
            eid, did = ids[j]
            ents = sorted({e for o in dec.inputs.obs_lists[j] for e in (o.entity, o.target) if e}
                          | {e for e in a.entities if isinstance(e, str)})
            outcome = {s.key: s.y for s in info["specs"][j] if s.observed}
            extra = {"outcomes": [s.to_dict() for s in info["specs"][j]], "dt": dt_actual[j],
                     "goal": _public_goal(self.goals[b]), "value": float(dec.value[j]), "decision_id": did,
                     "lp_prev": lp_prev[j], "retrieved": [x["id"] for x in dec.retrieved[j]],
                     "reasoning_iters": dec.reasoning[j]["iters"], "option": a.option_id,
                     "belief_loglik": None if info["loglik"] is None else float(info["loglik"][j])}
            rec = EventRecord(event_id=eid, env_id=self.env_ids[b], actor_id=self.actor_id,
                              episode_id=self.episode_ids[b], step=self.steps[b], model_version=version, t_sim=now[j],
                              observations=list(dec.inputs.obs_lists[j]), action=a, result=r,
                              predicted=dict(dec.predicted[j]), uncertainty=dict(dec.uncertainty[j]),
                              goal_id=self.goals[b].goal_id if self.goals[b] else None, reward=rew[j], cost=r.cost,
                              done=done[j], entity_links=ents, behavior_logp=float(dec.logp[j]),
                              provenance=Provenance.REAL, evidence=EvidenceKind.CONTROLLED, extra=extra)
            records.append(rec)
            traces.append(DecisionTrace(
                decision_id=did, event_id=eid, env_id=self.env_ids[b], policy_version=version, t_sim=now[j],
                objective=self.goals[b].text if self.goals[b] else "",
                context_refs=[x for x in (dec.ctx.refs[j] if dec.ctx.refs else []) if x is not None],
                retrieved=list(dec.retrieved[j]), candidates=list(dec.candidates[j]),
                predicted=dict(dec.predicted[j]), uncertainty=dict(dec.uncertainty[j]),
                imagined=list(dec.imagined[j]), selected=a.to_dict(), reasoning=dict(dec.reasoning[j]),
                belief_before=dict(dec.belief_before[j]) if dec.belief_before else {},
                belief_after=dict(info["belief_after"][j]) if info["belief_after"] else {}, result_ref=eid,
                outcome={"ok": r.ok, "error": r.error, "reward": rew[j], "values": outcome, "lp_prev": lp_prev[j]},
                update_refs=[], source=a.source.value))
            self.steps[b] += 1
        if self.archive is not None:
            self.archive.append(records)
            if self.replay is not None and hasattr(self.replay, "add"):
                self.replay.add(records)
            if self.config.use_memory and self.index is not None and hasattr(self.index, "add"):
                self.index.add(records)
        if self.recorder is not None:
            self.recorder.record(traces)
        dev = self.device
        fs = FreshStep(list(rows), [self.episode_ids[b] for b in rows], dec.inputs, list(dec.actions),
                       dec.logp.detach().float(), dec.value.detach().float(), torch.tensor(rew, device=dev),
                       torch.tensor(dt_actual, device=dev), torch.tensor(done, device=dev), version,
                       [ids[j][1] for j in range(len(rows))], torch.zeros(len(rows), device=dev))
        self.fresh.append(fs)
        for j, b in enumerate(rows):
            self._last_fresh[b] = (fs, j)
        if len(self.fresh) > self.fresh_max_steps:
            self.fresh = self.fresh[-self.fresh_max_steps:]
        self.n_decisions += len(rows)
        self.timing["observe_s"] += time.perf_counter() - t0
        return records

    # ---------------------------------------------------------------- learning
    def tokenise_sequences(self, seqs: Sequence[Sequence[EventRecord]]) -> tuple:
        """Archived EventRecords -> SequenceBatch (aligned; shorter sequences padded with masked steps marked done).
        Observations are re-tokenised with the current tokenizer (fresh episode-local registries); outcomes come
        from the records' realised outcome specs; goals from their agent-visible goal fields.
        Returns (SequenceBatch, channel tables [B] dict key -> slot for the support/query objective)."""
        m = self.model
        cfg = self.config
        B = len(seqs)
        T = max(len(s) for s in seqs)
        dev = self.device
        regs = [EntityRegistry(cfg.max_entities, seed=self.py_rng.randrange(2**31)) for _ in range(B)]
        C = None
        if cfg.use_fast_adaptation:
            C = int(m.beliefs.init_fast(1, device=dev).bayes_mu.shape[1])
        tables = [dict() for _ in range(B)]
        last_goal = [GoalSpec("none", "") for _ in range(B)]
        steps = []
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
                    if C is None:
                        row.append(-1)
                        continue
                    if s.key not in tables[b] and len(tables[b]) < C:
                        tables[b][s.key] = len(tables[b])
                    row.append(tables[b].get(s.key, -1))
                chans.append(row)
            outcomes = build_outcome_batch(specs, m.query_encoder, regs, chans, dev)
            for b, r in enumerate(recs):
                if r is not None and r.extra.get("goal"):
                    g = r.extra["goal"]
                    last_goal[b] = GoalSpec(g["goal_id"], g["text"], dict(g.get("metrics") or {}),
                                            g.get("deadline_days"))
            goal = m.tokenizer.goal(list(last_goal))
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
        return SequenceBatch(steps), tables

    def update(self) -> dict:
        """One learner update (spec 12.5). Returns metrics (losses, per-objective grad contributions from the
        trainer, consumed event/decision ids)."""
        cfg = self.config
        if not cfg.continual_updates:
            return {"updated": False, "reason": "continual_updates=False", "version": self.version}
        if self.trainer is None:
            raise RuntimeError("CamUnifiedAgent.update() needs an injected trainer (workstream D's Trainer)")
        t0 = time.perf_counter()
        seq, tables, replay_ids, replay_decisions = None, None, [], []
        if self.replay is not None and self.archive is not None and self.archive.count() > 0:
            seqs = [s for s in self.replay.sample(self.n_seqs, self.seq_len) if s]
            if seqs:
                seq, tables = self.tokenise_sequences(seqs)
                replay_ids = sorted({e for st in seq.steps for e in (st.event_ids or []) if e})
                replay_decisions = sorted({r.extra.get("decision_id") for s in seqs for r in s
                                           if r.extra.get("decision_id")})
        self.model.train()
        batch = UpdateBatch(seq, list(self.fresh), tables, model=self.model)
        losses, metrics, consumed = {}, {}, []
        for name, fn in self.objectives.items():
            out = fn(self, batch)
            if not out:
                continue
            for k, v in out.get("losses", {}).items():
                if not torch.is_tensor(v) or not torch.isfinite(v).all():
                    raise FloatingPointError(f"objective {name!r} produced a non-finite loss {k!r}: {v}")
                losses[k] = v
            metrics.update(out.get("metrics", {}))
            consumed += out.get("consumed", [])
        if not losses:
            self.model.eval()
            return {"updated": False, "reason": "no training data yet", "version": self.version}
        tm = self.trainer.step(losses, self.model)
        self.model.eval()
        self.version = self.version + 1
        update_id = f"u_{self.actor_id}_{self.version}"
        linked = sorted(set(consumed) | set(replay_decisions))
        if self.recorder is not None and hasattr(self.recorder, "link_updates") and linked:
            self.recorder.link_updates(linked, update_id)
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
        entry = {"update_id": update_id, "version": self.version, "losses": {k: float(v.detach())
                                                                             for k, v in losses.items()},
                 "consumed_decisions": consumed, "replay_events": replay_ids, "replay_decisions": replay_decisions}
        self.update_log.append(entry)
        return {"updated": True, **entry, **metrics, **{f"trainer/{k}": v for k, v in (tm or {}).items()}}
