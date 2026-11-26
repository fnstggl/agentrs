"""CAM Unified over the Crucible 5.0 universal experience interface, observation-only by default (meta-reward spec
Part 2; manifest OB-01..OB-05, OB-08).

Three channels are kept apart by construction (spec 2.4):

* **Actor observations** — `ExperienceAdapter` (implements `interfaces.EnvAdapter`). It holds only bound
  `observe / tools / execute / observe_since / advance` callables of each world; it has no reference to the world
  object, the verifier, the oracle or the label store, so nothing evaluator-side is reachable from the actor runtime
  (tested). Its reward stream does not exist: `reward_mode` must be ``observation_only`` unless an experiment
  declares a legacy baseline condition (`allow_reward_modes`).
* **Training outcome labels** — `ExperienceOperator.release_labels()` returns `OutcomeLabel`s only for episodes
  that have resolved (horizon reached / deadline passed / verified success), computed by the independent verifier
  (`env.verify()`) and normalised with evaluator-side references. Pending episodes are reported as PENDING, never
  as failures. The trainer receives labels through this channel only; the actor never does.
* **Private evaluation state** — `ExperienceOperator.private_verify(i)` (full verifier record, oracle values,
  world family and seed). Used by the evaluator and by experiment reports; never by the learner.

World families come from `crucible.experience.make(tier, name, seed, reward_mode)`: Tier 1 ``rapid:<family>`` (74
structurally diverse families: queues, networks, hidden causes, markets, science, delayed returns, POMDPs, search,
cooperation, program execution, labs, planning, teams, strategy, compositional causes, changing mechanisms) and
``mech:<family>`` (Crucible 4.0 hidden-mechanism facilities), Tier 0 kernel worlds (the full OpenWorld economy,
``ow_*`` families). Observations keep exact numbers (money in integer cents) and their provenance: an event caused by
one of CAM's own actions carries the `event_id` of that action (`Observation.caused_by`), joined through Crucible's
`action_seq`, so delayed consequences link back to the decision that caused them.

Verified utility of an episode (evaluator side): for objectives with a declared target, success is the verifier's
own `success`; for maximisation objectives the operator normalises the verified value against a null policy and the
private oracle evaluated on forks of the same world at episode start, U = (v - v_null) / (v_oracle - v_null), and
success := U >= `success_threshold`. These references are private and are never shown to the actor.
"""
from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from ..interfaces import (Action, EvidenceKind, ExecutionResult, GoalSpec, ObsKind, Observation, ObservationBatch,
                          ToolSchema, new_id)

OBSERVATION_ONLY = "observation_only"
LEGACY = "legacy_shaped_reward"
VERIFIED = "verified_terminal_outcome"

# Payload keys that identify the world family / seed / evaluator state. Crucible 5.0's mech tier still emits
# rig.episode events with {"task": "<family>:<seed>"} (found 2026-11-26): the family name and seed regenerate every
# hidden rule, so the adapter removes such keys from every payload before CAM sees it (OB-02).
PRIVATE_PAYLOAD_KEYS = frozenset({"task", "family", "seed", "mech_seed", "world_seed", "world_info", "success",
                                  "failed", "progress", "streak", "success_t", "initially_true", "default_net",
                                  "reference_net", "random_net", "solvable", "rejected", "score_first_half",
                                  "score_second_half", "private", "objective_quantity", "oracle"})


def sanitize(x):
    if isinstance(x, dict):
        return {k: sanitize(v) for k, v in x.items() if str(k) not in PRIVATE_PAYLOAD_KEYS}
    if isinstance(x, list):
        return [sanitize(v) for v in x]
    return x


_MONEY = re.compile(r"(cents|balance|revenue|cost|cash|price|net_cents|fee|spend|income)", re.I)


def _unit_of(path: str) -> str:
    return "cents" if _MONEY.search(path) else ""


# ============================================================================================ label channel

PENDING, SUCCESS, FAILURE, UNCERTAIN = "pending", "success", "failure", "uncertain"


@dataclass
class OutcomeLabel:
    """A released training outcome label (spec 2.4, 4.1): only after the outcome occurred."""
    env_id: str
    episode: int
    status: str                       # PENDING / SUCCESS / FAILURE / UNCERTAIN
    utility: Optional[float]          # normalised verified utility (None when pending / unverifiable)
    value: Optional[float]            # verified objective quantity in the world's unit (e.g. net cents)
    t_resolved_days: Optional[float]
    verifier: str = ""
    objective: str = ""
    meta: dict = field(default_factory=dict)   # label provenance only (no hidden mechanism data)


# ============================================================================================ conversion

def obs_to_records(obs: dict, env_id: str, seq_to_event: dict, *, first: bool = False,
                   text_seen: Optional[set] = None, sec_per_t: float = 1.0) -> list:
    """Crucible `Observation` dict -> list[cam Observation]. Exact numbers stay numbers; text keeps its source;
    events keep seq, actor, payload and the causing action's event id (when it was CAM's own action)."""
    t_days = float(obs.get("t", 0)) * sec_per_t / 86400.0
    out: list[Observation] = []
    seen = text_seen if text_seen is not None else set()
    for item in obs.get("text") or []:
        s = str(item)
        if s in seen:
            continue
        seen.add(s)
        src, _, body = s.partition(": ") if ": " in s[:80] else ("", "", s)
        out.append(Observation(ObsKind.TEXT, content=body or s, t=t_days, source=src or "world"))
    obj = obs.get("objective")
    if obj and ("objective:" + obj) not in seen:
        seen.add("objective:" + obj)
        out.append(Observation(ObsKind.GOAL, content=str(obj), t=t_days, source="objective"))
    for path, v in sorted((obs.get("numbers") or {}).items()):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or (isinstance(v, float) and not math.isfinite(v)):
            continue
        ent = path.rsplit(".", 1)[0] if "." in path else path
        out.append(Observation(ObsKind.NUMERIC, value=float(v), unit=_unit_of(path), entity=ent, t=t_days,
                               source=path, meta={"path": path}))
    for path, arr in sorted((obs.get("arrays") or {}).items()):
        if path.startswith("semantic."):
            continue
        vals = [float(x) for x in arr if isinstance(x, (int, float)) and not isinstance(x, bool)][:64]
        out.append(Observation(ObsKind.TOOL_RESULT, content=json.dumps({path: vals}), t=t_days, source=path,
                               meta={"array": path}))
    for k, v in sorted((obs.get("resources") or {}).items()):
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
            out.append(Observation(ObsKind.NUMERIC, value=float(v), unit=_unit_of(k), entity="self", t=t_days,
                                   source=f"resources.{k}", meta={"path": f"resources.{k}"}))
    for k, v in sorted((obs.get("analytics") or {}).items()):
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
            out.append(Observation(ObsKind.NUMERIC, value=float(v), unit=_unit_of(k), entity="public", t=t_days,
                                   source=f"analytics.{k}", meta={"path": f"analytics.{k}"}))
    for e in obs.get("entities") or []:
        for rel in e.get("relations") or []:
            out.append(Observation(ObsKind.RELATION, entity=str(e.get("id")), entity_type=str(e.get("kind", "")),
                                   target=str(rel.get("to")), relation=str(rel.get("rel", "")), t=t_days,
                                   source="entities"))
    for a in obs.get("artifacts") or []:
        out.append(Observation(ObsKind.FILE, content=str(a.get("title") or a.get("kind") or ""),
                               artifact_ref=str(a.get("ref")), entity=a.get("owner"), t=float(a.get("t", 0)) / 86400.0,
                               source="artifacts"))
    for m in obs.get("missing") or []:
        out.append(Observation(ObsKind.CONSTRAINT, content=str(m), t=t_days, source="missing"))
    for ev in obs.get("events") or []:
        out.append(event_to_record(ev, seq_to_event, sec_per_t))
    return out


def event_to_record(ev: dict, seq_to_event: dict, sec_per_t: float = 1.0) -> Observation:
    cause = ev.get("cause") or None
    caused_by = None
    evidence = EvidenceKind.OBSERVATIONAL
    meta = {"type": ev.get("type"), "actor": ev.get("actor")}
    if cause:
        caused_by = seq_to_event.get(int(cause.get("action_seq", -1)))
        evidence = EvidenceKind.CONTROLLED
        meta.update({"via": cause.get("via"), "delay_s": cause.get("delay_s"), "action_seq": cause.get("action_seq")})
    typ = str(ev.get("type", ""))
    kind = ObsKind.TRANSACTION if any(k in typ for k in ("bank", "pay", "transfer", "charge", "invoice")) else \
        ObsKind.MESSAGE if any(k in typ for k in ("mail", "message", "chat", "post")) else ObsKind.EVENT
    return Observation(kind, content=json.dumps({"type": typ, "payload": sanitize(ev.get("payload"))}, sort_keys=True,
                                                default=str)[:4000],
                       t=float(ev.get("t", 0)) * sec_per_t / 86400.0, seq=int(ev.get("seq", 0)),
                       source=str(ev.get("actor", "")),
                       evidence=evidence, caused_by=caused_by, meta=meta)


def tool_to_schema(t: dict) -> ToolSchema:
    ann = t.get("annotations") or {}
    cs = ann.get("costSeconds")
    return ToolSchema(name=str(t["name"]), description=str(t.get("description", "")),
                      parameters=t.get("inputSchema") or {"type": "object", "properties": {}},
                      duration_hint=(float(cs) / 86400.0) if isinstance(cs, (int, float)) else None,
                      meta={"category": ann.get("category"), "readOnly": ann.get("readOnlyHint"),
                            "host": ann.get("host")})


# ============================================================================================ worlds

@dataclass(frozen=True)
class WorldSpec:
    tier: int
    name: str                 # "rapid:<family>", "mech:<family>", or a kernel world family
    seed: int = 0
    kw: tuple = ()            # extra make() kwargs as ((key, value), ...)

    @property
    def family(self) -> str:
        return self.name


class _World:
    """Operator-side handle of one Crucible world (never given to the actor)."""

    def __init__(self, spec: WorldSpec, reward_mode: str):
        from crucible.experience import make
        self.spec = spec
        self.reward_mode = reward_mode
        self.env = make(spec.tier, spec.name, seed=spec.seed, reward_mode=reward_mode, **dict(spec.kw))
        self.agent = self.env.agents()[0]
        self.episode = 0
        self.refs: Optional[dict] = None
        self.env_id = f"{spec.name}#{spec.seed}"
        # Tier 1 rapid worlds count time in ticks (one tick per decision / advance); 4.0-derived worlds in seconds.
        self.sec_per_t = 3600.0 if spec.name.startswith("rapid:") else 1.0


class ExperienceOperator:
    """Owns the worlds, the verifier and the label store. Hands the actor an `ExperienceAdapter`."""

    def __init__(self, specs: Sequence[WorldSpec], reward_mode: str = OBSERVATION_ONLY,
                 allow_reward_modes: Sequence[str] = (OBSERVATION_ONLY,), success_threshold: float = 0.5,
                 compute_references: bool = True):
        if reward_mode not in allow_reward_modes:
            raise ValueError(f"reward_mode {reward_mode!r} not allowed (declare it in allow_reward_modes for a "
                             f"baseline condition); new CAM runs use {OBSERVATION_ONLY!r}")
        import crucible
        crucible.load_all_modules()
        self.reward_mode = reward_mode
        self.success_threshold = float(success_threshold)
        self.compute_references = compute_references
        self.worlds = [_World(s, reward_mode) for s in specs]
        self._released: dict = {}
        self._ref_cache: dict = {}
        if compute_references:
            for w in self.worlds:
                w.refs = self._references(w)

    # ---------------------------------------------------------------------------------------- actor view
    def adapter(self) -> "ExperienceAdapter":
        fns = []
        for i, w in enumerate(self.worlds):
            env, agent = w.env, w.agent
            fns.append(_ActorFns(env_id=w.env_id, sec_per_t=w.sec_per_t,
                                 observe=(lambda env=env, agent=agent: env.observe(agent)),
                                 tools=(lambda env=env, agent=agent: env.tools(agent)),
                                 execute=(lambda act, env=env, agent=agent: env.execute(agent, act)),
                                 observe_since=(lambda cur, env=env, agent=agent: env.observe_since(agent, cur)),
                                 advance=(lambda s, env=env: env.advance(s)),
                                 done=(lambda env=env: bool(getattr(env, "done", False))),
                                 reset=(lambda i=i: self.reset_world(i))))
        return ExperienceAdapter(fns)

    # ---------------------------------------------------------------------------------------- operator side
    def reset_world(self, i: int) -> None:
        w = self.worlds[i]
        if hasattr(w.env, "reset"):
            w.env.reset()
        else:
            w.env = _World(w.spec, self.reward_mode).env
        w.episode += 1

    def private_verify(self, i: int) -> dict:
        """Full evaluator record (PRIVATE: never to the learner)."""
        w = self.worlds[i]
        return {"verify": w.env.verify(), "references": w.refs, "family": w.spec.name, "seed": w.spec.seed,
                "episode": w.episode}

    def resolved(self, i: int) -> bool:
        return bool(getattr(self.worlds[i].env, "done", False))

    def release_labels(self, indices: Optional[Sequence[int]] = None) -> list:
        """Training outcome labels of resolved episodes (once each); PENDING for unresolved ones."""
        out = []
        for i in (range(len(self.worlds)) if indices is None else indices):
            w = self.worlds[i]
            key = (i, w.episode)
            if not self.resolved(i):
                out.append(OutcomeLabel(w.env_id, w.episode, PENDING, None, None, None))
                continue
            if key in self._released:
                out.append(self._released[key])
                continue
            lab = self._label(w)
            self._released[key] = lab
            out.append(lab)
        return out

    def _label(self, w: _World) -> OutcomeLabel:
        rec = w.env.verify()
        o = rec.get("outcome") or {}
        t = float(o.get("t", 0))
        if "outcome" in rec:
            value = o.get("value")
            succ = o.get("success")
            util = None
            if value is not None and w.refs and w.refs.get("oracle") is not None:
                lo, hi = w.refs["null"], w.refs["oracle"]
                util = (float(value) - lo) / max(hi - lo, 1e-9) if hi > lo else None
            if succ is None and util is not None:
                succ = util >= self.success_threshold
            status = UNCERTAIN if succ is None else (SUCCESS if succ else FAILURE)
            return OutcomeLabel(w.env_id, w.episode, status, util, None if value is None else float(value),
                                t * w.sec_per_t / 86400.0, rec.get("verifier", ""), str(o.get("objective", "")))
        # mech / kernel verifiers: agents' success flags
        ag = (rec.get("agents") or {}).get(w.agent, {})
        cur = rec.get("current") or {}
        succ = ag.get("success")
        if succ is None:
            succ = cur.get("success_so_far")
        status = UNCERTAIN if succ is None else (SUCCESS if succ else FAILURE)
        value = cur.get("net_cents")
        return OutcomeLabel(w.env_id, w.episode, status, 1.0 if succ else (0.0 if succ is not None else None),
                            None if value is None else float(value), None, rec.get("verifier", ""))

    def _references(self, w: _World) -> Optional[dict]:
        """Null-policy and oracle values on forks of the world at episode start (rapid tier). Private."""
        key = (w.spec.name, w.spec.seed)
        if key in self._ref_cache:
            return self._ref_cache[key]
        refs = None
        inner = getattr(w.env, "env", None)
        if w.spec.name.startswith("rapid:") and inner is not None:
            try:
                from crucible.rapid import oracles
                null = oracles.run_fn(oracles.fork(inner), lambda e: None)
                best = oracles.oracle(oracles.fork(inner))
                refs = {"null": float(null["value"]), "oracle": float(best["value"]), "how": best.get("how")}
            except Exception as e:                       # a family without an applicable oracle
                refs = {"null": None, "oracle": None, "error": repr(e)[:200]}
        self._ref_cache[key] = refs
        return refs


@dataclass
class _ActorFns:
    env_id: str
    sec_per_t: float
    observe: Callable
    tools: Callable
    execute: Callable
    observe_since: Callable
    advance: Callable
    done: Callable
    reset: Callable


class ExperienceAdapter:
    """Actor-side EnvAdapter over Crucible 5.0 worlds (observation-only). Exposes no reward and no verifier."""

    def __init__(self, fns: Sequence[_ActorFns]):
        self._fns = list(fns)
        self._seq_to_event: list = [dict() for _ in self._fns]
        self._cursor = [0 for _ in self._fns]
        self._text_seen: list = [set() for _ in self._fns]
        self.env_ids = [f.env_id for f in self._fns]

    @property
    def B(self) -> int:
        return len(self._fns)

    def _batch(self, first: bool = False) -> ObservationBatch:
        obs_l, t_l, done_l = [], [], []
        for i, f in enumerate(self._fns):
            ob = f.observe()
            self._cursor[i] = max(self._cursor[i], int(ob.get("cursor", 0) or 0))
            obs_l.append(obs_to_records(ob, f.env_id, self._seq_to_event[i], first=first,
                                        text_seen=self._text_seen[i], sec_per_t=f.sec_per_t))
            t_l.append(float(ob.get("t", 0)) * f.sec_per_t / 86400.0)
            done_l.append(f.done())
        return ObservationBatch(list(self.env_ids), obs_l, t_l, done_l)

    def reset(self, seeds: Sequence[int] | None = None) -> ObservationBatch:
        for i, f in enumerate(self._fns):
            f.reset()
            self._seq_to_event[i] = {}
            self._cursor[i] = 0
            self._text_seen[i] = set()
        return self._batch(first=True)

    def observe(self) -> ObservationBatch:
        return self._batch()

    def available_tools(self) -> list:
        return [[tool_to_schema(t) for t in f.tools()] for f in self._fns]

    def execute(self, actions: Sequence[Optional[Action]]) -> list:
        out = []
        for i, (f, a) in enumerate(zip(self._fns, actions)):
            if a is None:
                out.append(None)
                continue
            r = f.execute({"tool": a.tool, "args": a.args})
            seq = r.get("action_seq")
            if seq is not None:
                self._seq_to_event[i][int(seq)] = a.action_id
            t = float(r.get("t", 0)) * f.sec_per_t / 86400.0
            dur = float(r.get("cost_s", 0) or 0) / 86400.0
            body = sanitize(r.get("result") if r.get("ok") else r.get("error"))
            res_obs = Observation(ObsKind.TOOL_RESULT if r.get("ok") else ObsKind.ERROR,
                                  content=json.dumps(body, sort_keys=True, default=str)[:8000], t=t, source=a.tool,
                                  evidence=EvidenceKind.CONTROLLED, caused_by=a.action_id,
                                  meta={"action_seq": seq})
            err = None if r.get("ok") else json.dumps(r.get("error"), default=str)[:500]
            out.append(ExecutionResult(ok=bool(r.get("ok")), observations=[res_obs], error=err, duration=dur,
                                       started_t=t, finished_t=t + dur, meta={"action_seq": seq}))
        return out

    def collect_events(self, since: Sequence[int]) -> tuple:
        obs_l, cur_l = [], []
        for i, (f, c) in enumerate(zip(self._fns, since)):
            evs, cur = f.observe_since(int(c))
            obs_l.append([event_to_record(e, self._seq_to_event[i], f.sec_per_t) for e in evs])
            cur_l.append(int(cur))
        return obs_l, cur_l

    def advance_time(self, duration: Sequence[float]) -> ObservationBatch:
        for f, d in zip(self._fns, duration):
            f.advance(max(0, int(round(float(d) * 86400))))
        return self._batch()

    def goals(self) -> list:
        out = []
        for i, f in enumerate(self._fns):
            ob = f.observe()
            txt = ob.get("objective") or ""
            out.append(GoalSpec(goal_id=f"{f.env_id}:{i}", text=str(txt)))
        return out

    def close(self) -> None:
        self._fns = []


def leakage_scan(batch: ObservationBatch, forbidden: Sequence[str]) -> list:
    """Return every (env, observation) whose serialised form contains a forbidden token (family names, seeds as
    'seed', 'success', 'reward', oracle values...). Used by the OB-02 test over all families CAM trains on."""
    hits = []
    for i, obs in enumerate(batch.observations):
        for o in obs:
            s = json.dumps(o.to_dict(), default=str).lower()
            for tok in forbidden:
                if tok.lower() in s:
                    hits.append((batch.env_ids[i], tok, s[:200]))
    return hits
