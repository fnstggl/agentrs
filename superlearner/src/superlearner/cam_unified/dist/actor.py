"""Actor processes: independent environments, goals, seeds, latent and fast state, and a versioned copy of the
shared weights; they write experience to the durable queue and never optimise (spec 14.1, manifest A16).

An actor is described by an `ActorSpec` whose environment and policy are built from factories named by module path
(``"package.module:callable"``) so the spec crosses the process boundary as plain data:

* ``env_factory(seed=..., **env_kwargs)`` → ``EnvAdapter`` or ``(EnvAdapter, GoalEvaluator | None)``;
* ``policy_factory(seed=..., **policy_kwargs)`` (plus ``semantic=<SemanticClient>`` when a shared semantic server is
  configured) → an object implementing `ActorPolicyP`.

Per decision the actor runs the SMDP harness of architecture decision 5: ``act`` → ``execute`` → ``collect_events``
→ ``advance_time(duration + think)`` → ``collect_events``; the verified reward is the increment of the goal
evaluator's score; consequences delivered later with ``caused_by`` become separate EventRecords with
``attributed_to`` = the responsible decision's event id (written before the decision record of the step in which
they arrive, so a trajectory's final decision record closes it). Every record carries ``model_version`` (the weights
version that chose the action) and ``behavior_logp``; every decision also yields a `DecisionTrace`.

Weights are refreshed only at safe boundaries through `PolicyClient` (trajectory boundaries, or decision boundaries
with ``refresh_at="decision"``). After a decision-boundary refresh whose relative parameter change since the latent
state was built exceeds ``rebuild_threshold``, the latent state is reconstructed by `ActorPolicyP.rebuild_state` from
the raw records of the running episode (version-independent memory) instead of being reused. Every decision record
carries ``extra["state_version"]`` (the weights version the latent state was built under; it differs from
``model_version`` after a refresh without rebuild) and ``extra["deterministic"]`` (the policy acted greedily, so
``behavior_logp`` is 0 = log 1), which the learner's router needs to decide what is exactly on-policy.

Crash recovery: environment exceptions close the running episodes with an explicit ``episode_end`` record
(``truncated: true``), rebuild the environment (a new world, hence new environment ids and seeds) and continue (up to
``max_env_failures`` in a row). A killed process is restarted by the launcher under the same actor id with
``incarnation`` + 1: its `SpoolWriter` recovers sequence numbers, leftover temporary segments and every record the
dead process had buffered (write-ahead log), and the actor closes the episodes the dead incarnation left open (it
persists their ids at every episode start) with ``episode_end`` records (``truncated: true``). Seeds of the
environment, the policy and the episode sampler are derived from (seed, incarnation, environment rebuilds), so a
restarted actor never replays the experience of an earlier incarnation and its worlds are new lifetimes with new ids.
"""
from __future__ import annotations

import importlib
import inspect
import os
import random
import time
import traceback
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

from ..interfaces import (DecisionTrace, EventRecord, EvidenceKind, ObservationBatch, Provenance, new_id)
from ..observe.accounting import ComputeMeter, atomic_write_json
from .publish import PolicyClient
from .queue import ExperienceQueue, action_dict


def derive_seed(seed: int, *salt) -> int:
    """``seed`` itself when every salt is 0/empty (the first incarnation's first world), else a deterministic
    31-bit hash of (seed, salt...)."""
    if not any(salt):
        return int(seed)
    return random.Random("/".join(str(x) for x in (seed,) + salt)).randrange(2 ** 31)


def resolve(path: str):
    """"package.module:attr" -> the attribute."""
    mod, _, attr = path.partition(":")
    if not attr:
        raise ValueError(f"factory path must be 'module:callable', got {path!r}")
    obj = importlib.import_module(mod)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


@runtime_checkable
class ActorPolicyP(Protocol):
    """What an actor needs from a policy (CAM's integrated agent implements this after integration)."""

    def reset(self, obs: ObservationBatch, goals: list, tools: list) -> None: ...      # new latent + fast state
    def act(self, obs_lists: list, tools: list, goals: list) -> tuple: ...             # -> ([B] Action|None, [B] info)
    def load_weights(self, state_dict: dict, version: int) -> None: ...                # weights-only swap
    def rebuild_state(self, history: list) -> None: ...                                # [B] list[EventRecord]


@dataclass
class ActorSpec:
    actor_id: str
    run_dir: str
    env_factory: str = "superlearner.cam_unified.dist.toyworld:make_toy_env"
    env_kwargs: dict = field(default_factory=dict)
    policy_factory: str = "superlearner.cam_unified.dist.toyworld:make_tiny_policy"
    policy_kwargs: dict = field(default_factory=dict)
    seed: int = 0
    refresh_at: str = "trajectory"
    rebuild_threshold: float = 0.0
    batch_records: int = 256
    max_decisions: Optional[int] = None
    max_episodes: Optional[int] = None
    max_seconds: Optional[float] = None
    think_days: float = 0.0
    torch_threads: int = 1
    record_traces: bool = True
    status_every_s: float = 0.5
    max_env_failures: int = 3
    wait_version_s: float = 120.0
    queue_kwargs: dict = field(default_factory=dict)
    incarnation: int = 0            # process restarts under this actor id (set by the launcher)
    cuda: bool = False              # False: CUDA_VISIBLE_DEVICES="" in the actor process (no GPU context per actor;
                                    # GPU inference goes through the shared semantic server or an explicit cuda=True)

    def to_dict(self) -> dict:
        return asdict(self)


def run_paths(run_dir: str) -> dict:
    return {k: os.path.join(run_dir, k) for k in ("spool", "ckpt", "archive", "traces", "learner", "status",
                                                  "compute", "logs", "semserve")}


class ActorProcess:
    def __init__(self, spec: ActorSpec, stop_event=None, semantic=None):
        self.spec = spec
        self.stop_event = stop_event
        self.semantic = semantic
        p = run_paths(spec.run_dir)
        self.paths = p
        for d in (p["status"], p["compute"]):
            os.makedirs(d, exist_ok=True)
        self.queue = ExperienceQueue(p["spool"], **spec.queue_kwargs)
        self.writer = self.queue.writer(spec.actor_id, batch_records=spec.batch_records)
        self.client = PolicyClient(p["ckpt"], refresh_at=spec.refresh_at, actor_id=spec.actor_id)
        self.meter = ComputeMeter(spec.actor_id, role="actor")
        self.incarnation = int(spec.incarnation)
        self.rng = random.Random(derive_seed(spec.seed, self.incarnation, "episodes") if self.incarnation
                                 else spec.seed)
        self.env_builds = 0
        self.stats = {"episodes": 0, "decisions": 0, "records": 0, "delayed": 0, "refreshes": 0, "rebuilds": 0,
                      "env_failures": 0, "truncated_episodes": 0, "versions_used": [],
                      "incarnation": self.incarnation, "closed_from_previous_incarnation": 0,
                      "wal_recovered_items": self.writer.stats.get("wal_recovered_items", 0)}
        self.recent_scores: deque = deque(maxlen=50)
        self.episode_log: list = []           # (episode, version, mean score) per finished episode
        self._action_to_event: OrderedDict = OrderedDict()
        self._last_status = 0.0
        self.t_start = time.time()
        self.state = "starting"
        self._make_env()
        kw = dict(spec.policy_kwargs)
        if semantic is not None:
            kw["semantic"] = semantic
        self.policy = resolve(spec.policy_factory)(seed=derive_seed(spec.seed, self.incarnation), **kw)
        try:
            self._act_takes_active = "active" in inspect.signature(self.policy.act).parameters
        except (TypeError, ValueError):
            self._act_takes_active = False

    # ---------------------------------------------------------------- helpers
    def _make_env(self) -> None:
        seed = derive_seed(self.spec.seed, self.incarnation, self.env_builds)
        self.env_builds += 1
        made = resolve(self.spec.env_factory)(seed=seed, **self.spec.env_kwargs)
        if isinstance(made, tuple):
            self.env, self.evaluator = made[0], (made[1] if len(made) > 1 else None)
        else:
            self.env, self.evaluator = made, getattr(made, "evaluator", None)

    def _stop(self) -> bool:
        s = self.spec
        if self.stop_event is not None and self.stop_event.is_set():
            return True
        if s.max_decisions is not None and self.stats["decisions"] >= s.max_decisions:
            return True
        if s.max_seconds is not None and time.time() - self.t_start >= s.max_seconds:
            return True
        return False

    def _scores(self, B: int) -> list:
        if self.evaluator is None:
            return [0.0] * B
        return [float(self.evaluator.evaluate_goal(b).score) for b in range(B)]

    def _remember_action(self, action_id: str, event_id: str) -> None:
        self._action_to_event[action_id] = event_id
        if len(self._action_to_event) > 100_000:
            self._action_to_event.popitem(last=False)

    def status(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_status < self.spec.status_every_s:
            return
        self._last_status = now
        env_ids = getattr(self, "_env_ids", [])
        goals = getattr(self, "_goals", [])
        st = {"actor_id": self.spec.actor_id, "pid": os.getpid(), "state": self.state, "t": now,
              "t_start": self.t_start, "version": self.client.version, "drift": self.client.drift,
              "env_ids": env_ids, "goals": [g.text for g in goals], "goal_ids": [g.goal_id for g in goals],
              "recent_score": (sum(self.recent_scores) / len(self.recent_scores)) if self.recent_scores else None,
              "episode_log": self.episode_log[-200:], "writer": dict(self.writer.stats), **self.stats,
              "versions_used": self.stats["versions_used"][-50:], "refresh_at": self.spec.refresh_at,
              "seed": self.spec.seed, "incarnation": self.incarnation}
        atomic_write_json(os.path.join(self.paths["status"], f"{self.spec.actor_id}.json"), st, fsync=False)
        self.meter.write(os.path.join(self.paths["compute"], f"{self.spec.actor_id}.{os.getpid()}.json"))

    # ---------------------------------------------------------------- main loop
    def run(self) -> dict:
        import torch
        torch.set_num_threads(self.spec.torch_threads)
        self.state = "waiting_for_weights"
        self.status(force=True)
        t_wait = time.time()
        self.client.wait_for_version(self.policy, 0, timeout_s=self.spec.wait_version_s)
        self.stats["wait_weights_s"] = time.time() - t_wait
        self.stats["t_running"] = time.time()
        self.stats["versions_used"].append(self.client.version)
        self._close_previous_incarnation()
        self.state = "running"
        failures = 0
        try:
            while not self._stop():
                if self.spec.max_episodes is not None and self.stats["episodes"] >= self.spec.max_episodes:
                    break
                try:
                    self._episode()
                    failures = 0
                except Exception as e:                         # noqa: BLE001 — environment/policy failure
                    failures += 1
                    self.stats["env_failures"] += 1
                    self.stats["last_error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}"
                    self._close_truncated(f"{type(e).__name__}: {e}")
                    if failures > self.spec.max_env_failures:
                        self.state = "failed"
                        raise
                    try:
                        self.env.close()
                    except Exception:                          # noqa: BLE001
                        pass
                    self._make_env()
            self.writer.flush()
            self.state = "finished"
        finally:
            try:
                self.writer.flush()
            finally:
                self.status(force=True)
        return dict(self.stats)

    def _episode_end(self, ep: str, env_id: str, step: int, reason: str, **extra) -> None:
        self.writer.put(EventRecord(event_id=new_id("e_"), env_id=env_id, actor_id=self.spec.actor_id,
                                    episode_id=ep, step=step, model_version=self.client.version, t_sim=0.0,
                                    observations=[], done=True, evidence=EvidenceKind.OBSERVATIONAL,
                                    extra={"kind": "episode_end", "truncated": True, "reason": reason,
                                           "incarnation": self.incarnation, **extra}))

    def _close_truncated(self, reason: str) -> None:
        open_eps = getattr(self, "_open_episodes", {})
        for b, (ep, env_id, step) in open_eps.items():
            self._episode_end(ep, env_id, step, reason)
            self.stats["truncated_episodes"] += 1
        self._open_episodes = {}
        self.writer.note_open_episodes([])
        self.writer.flush()

    def _close_previous_incarnation(self) -> None:
        """Episodes a killed earlier incarnation left open (persisted at their start) get an explicit end."""
        prev = self.writer.previous_open_episodes()
        for ep, env_id, inc in prev:
            self._episode_end(ep, env_id, -1, "actor process died", dead_incarnation=inc)
        self.stats["closed_from_previous_incarnation"] += len(prev)
        self.writer.note_open_episodes([])
        if prev:
            self.writer.flush()

    def _episode(self) -> None:
        s, env = self.spec, self.env
        if self.client.maybe_refresh(self.policy, "trajectory"):
            self.stats["refreshes"] += 1
            self.stats["versions_used"].append(self.client.version)
        self.client.state_rebuilt()             # a new trajectory starts from a fresh latent state
        # independent per-actor, per-episode environment seeds (the actor's rng is seeded by spec.seed)
        ep_seed = self.rng.randrange(2**31)
        n_env = len(getattr(env, "env_ids", [])) or getattr(env, "B", None)
        obs = env.reset(seeds=[ep_seed + b for b in range(int(n_env))] if n_env else None)
        B = len(obs.env_ids)
        goals = env.goals()
        tools = env.available_tools()
        self.policy.reset(obs, goals, tools)
        self._env_ids, self._goals = list(obs.env_ids), goals
        cursors = list(getattr(env, "initial_cursors", lambda: [0] * B)())
        view = [list(o) for o in obs.observations]
        ep_ids = [new_id("ep_") for _ in range(B)]
        history = [[] for _ in range(B)]
        prev_event = [None] * B
        prev_score = self._scores(B)
        ep_return = [0.0] * B
        done = list(obs.done)
        t_now = list(obs.t)
        step = 0
        self._open_episodes = {b: (ep_ids[b], obs.env_ids[b], 0) for b in range(B)}
        self.writer.note_open_episodes([(ep_ids[b], obs.env_ids[b], self.incarnation) for b in range(B)])
        version_at_start = self.client.version
        while not all(done):
            if self._stop():
                break
            tools = env.available_tools()
            active = [not d for d in done]
            if self._act_takes_active:
                actions, infos = self.policy.act(view, tools, goals, active=active)
            else:
                actions, infos = self.policy.act(view, tools, goals)
            actions = [a if active[b] else None for b, a in enumerate(actions)]
            ev_ids = [new_id("e_") if actions[b] is not None else None for b in range(B)]
            for b, a in enumerate(actions):
                if a is not None:
                    self._remember_action(a.action_id, ev_ids[b])
            results = env.execute(actions)
            ev1, cursors = env.collect_events(cursors)
            durations = [(results[b].duration + s.think_days) if actions[b] is not None else 0.0 for b in range(B)]
            adv = env.advance_time(durations)
            ev2, cursors = env.collect_events(cursors)
            scores = self._scores(B)
            n_active = 0
            for b in range(B):
                if actions[b] is None:
                    continue
                n_active += 1
                a, res, info = actions[b], results[b], infos[b] or {}
                reward = scores[b] - prev_score[b]
                ep_return[b] += reward
                arrived = list(ev1[b]) + list(ev2[b])
                delayed_ids = []
                for o in arrived:
                    cause = None
                    if o.caused_by is not None:
                        cause = self._action_to_event.get(o.caused_by, o.caused_by if
                                                          str(o.caused_by).startswith("e_") else None)
                    if cause is None:
                        continue
                    did = new_id("e_")
                    delayed_ids.append(did)
                    self.writer.put(EventRecord(
                        event_id=did, env_id=obs.env_ids[b], actor_id=s.actor_id, episode_id=ep_ids[b], step=step,
                        model_version=self.client.version, t_sim=float(adv.t[b]), observations=[o],
                        evidence=EvidenceKind.CONTROLLED, attributed_to=cause,
                        extra={"kind": "delayed_outcome"}))
                    self.stats["delayed"] += 1
                    self.stats["records"] += 1
                rec = EventRecord(
                    event_id=ev_ids[b], env_id=obs.env_ids[b], actor_id=s.actor_id, episode_id=ep_ids[b], step=step,
                    model_version=self.client.version, t_sim=float(t_now[b]), observations=view[b], action=a,
                    result=res, predicted=dict(info.get("predicted", {})),
                    uncertainty=dict(info.get("uncertainty", {})), goal_id=goals[b].goal_id, reward=float(reward),
                    cost=float(res.cost), done=bool(adv.done[b]), behavior_logp=a.logp,
                    provenance=Provenance.REAL, evidence=EvidenceKind.CONTROLLED,
                    extra={"dt": durations[b], "drift": self.client.drift, "kind": "decision",
                           "state_version": self.client.state_version,
                           "deterministic": bool(info.get("deterministic", False)),
                           "incarnation": self.incarnation})
                trace = None
                if s.record_traces:
                    trace = DecisionTrace(
                        decision_id=new_id("d_"), event_id=rec.event_id, env_id=rec.env_id,
                        policy_version=self.client.version, t_sim=rec.t_sim, objective=goals[b].text,
                        context_refs=[x for x in [prev_event[b]] if x] + list(info.get("context_refs", [])),
                        retrieved=list(info.get("retrieved", [])), candidates=list(info.get("candidates", [])),
                        predicted=dict(rec.predicted), uncertainty=dict(rec.uncertainty),
                        imagined=list(info.get("imagined", [])), selected=action_dict(a),
                        reasoning=dict(info.get("reasoning", {})), belief_before=dict(info.get("belief_before", {})),
                        belief_after=dict(info.get("belief_after", {})),
                        outcome={"ok": res.ok, "reward": float(reward), "cost": float(res.cost),
                                 "duration": float(res.duration), "error": res.error,
                                 "observed_next": len(arrived)},
                        source=a.source.value if hasattr(a.source, "value") else str(a.source))
                self.writer.put(rec, trace)
                self.stats["records"] += 1
                history[b].append(rec)
                prev_event[b] = rec.event_id
                view[b] = list(res.observations) + list(ev1[b]) + list(adv.observations[b]) + list(ev2[b])
                self._open_episodes[b] = (ep_ids[b], obs.env_ids[b], step + 1)
                if adv.done[b]:
                    self._open_episodes.pop(b, None)
            if hasattr(self.policy, "observe"):
                self.policy.observe(results, view)
            prev_score = scores
            done = [bool(d) for d in adv.done]
            t_now = list(adv.t)
            step += 1
            self.stats["decisions"] += n_active
            self.meter.add(env_steps=n_active, decisions=n_active, sim_days=sum(durations),
                           env_cost=sum(r.cost for r in results), records=n_active)
            if s.refresh_at == "decision" and not all(done):
                if self.client.maybe_refresh(self.policy, "decision"):
                    self.stats["refreshes"] += 1
                    self.stats["versions_used"].append(self.client.version)
                    if self.client.drift > s.rebuild_threshold:
                        self.policy.rebuild_state(history)
                        self.client.state_rebuilt()
                        self.stats["rebuilds"] += 1
            self.status()
        if all(done):
            self.stats["episodes"] += 1
            mean_ret = sum(ep_return) / max(1, B)
            self.recent_scores.append(mean_ret)
            self.episode_log.append([self.stats["episodes"], version_at_start, self.client.version, mean_ret,
                                     time.time()])
        else:                                   # stopped mid-episode: close it explicitly
            self._close_truncated("actor stopped")
        self._open_episodes = {}
        self.writer.note_open_episodes([])
        self.writer.flush()
        self.status()


def actor_main(spec: dict, stop_event=None, sem_conn: Optional[tuple] = None) -> None:
    """Process entry point (spawn): build the actor from a plain-dict spec and run it."""
    if not spec.get("cuda", False):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""          # before any CUDA query in this process
    sem = None
    if sem_conn is not None:
        from .semserve import SemanticClient
        sem = SemanticClient(*sem_conn)
    ActorProcess(ActorSpec(**spec), stop_event=stop_event, semantic=sem).run()
