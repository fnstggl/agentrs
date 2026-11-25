"""The real Crucible OpenWorld behind the CAM Unified `EnvAdapter` contract.

Two objects are created together by `open_world(tasks, ...)`:

* `OpenWorldOperator` (operator side, **never handed to the learner**): owns the `crucible.api.WorldServer`, the
  worlds, the actor credentials, the clock (`advance`), genesis replay, the operator log used for provenance,
  snapshots / forks / checkpoints, optional independently controlled participants, and the hooks through which the
  goal evaluator (`openworld_eval.OpenWorldEvaluator`) runs its daily checks.
* `OpenWorldAdapter` (agent side, implements `interfaces.EnvAdapter`): B actors, each in its own world (default) or
  all in one world (``shared_world=True``). Every call goes through `WorldServer` **actor operations only**
  (`observe`, `available_tools`, `execute_action`, actor-filtered `replay_events`) and returns contract records.
  The adapter holds no reference to a kernel, a server, a goal contract or any hidden state: it talks to the
  operator through a closure-based port, so attribute traversal of the adapter (what a learner could do with it)
  reaches none of them (tests walk the object graph). Its records never contain the world family, seed or the
  kernel's world id (the id is a hash of the spec).

Semantics (contract v2):

* **Observations.** `reset` delivers, once, the actor-visible events from before the actor existed (genesis public
  events: need-format documents, listings, providers, persons; `observe` never shows them), a self-identity record,
  the join events and the module summaries. Afterwards `observe` / `advance_time` return exactly the events that are
  new since the previous delivery plus fresh summaries: every actor-visible event is delivered exactly once.
* **Provenance.** Events emitted by the actor's own command (read from the operator log, filtered with
  ``World.visible(e, World.principals_of(p))``) carry ``caused_by = action_id`` and evidence CONTROLLED.
* **Cursors.** `collect_events(since)` is stateless and lossless: all actor-visible events with seq > since
  (adapter-side buffer, falling back to actor-filtered `replay_events` for older cursors).
* **Clock (SMDP).** Executing never moves time; `advance_time(days)` is the operator clock (the harness calls it
  with cost_s + think_s, or the agent's own ``wait``). It is split at the evaluator's daily check times so sustain
  streaks are counted exactly (critic: one jump over several days would run every due check on the end state).
* **wait tool.** A harness tool ``{"seconds": int <= max_wait_s}``: no world call, its duration is the wait.
* **Money / time units** as given (cents, seconds -> days only for record times and durations).
"""
from __future__ import annotations

import bisect
import copy
import json
import os
import pickle
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from ..interfaces import (Action, EvidenceKind, ExecutionResult, GoalSpec, Observation, ObservationBatch, ObsKind,
                          ToolSchema, new_id)
from . import ow_convert as C

DAY = 86400
WAIT_TOOL = "wait"
_SYLL_C, _SYLL_V = "bdfgklmnprstvz", "aeiou"


def neutral_handle(rng: random.Random) -> str:
    """An opaque actor handle such as 'kavo-27' (matches Crucible's [a-z][a-z0-9-]{1,30})."""
    s = "".join(rng.choice(_SYLL_C) + rng.choice(_SYLL_V) for _ in range(2))
    return f"{s}-{rng.randrange(10, 100)}"


def wait_tool(max_wait_s: int) -> ToolSchema:
    return ToolSchema(WAIT_TOOL, "Let simulated time pass without acting, e.g. to await the delayed consequences of "
                      "earlier actions. Nothing in the world is called.",
                      {"type": "object", "properties": {"seconds": {"type": "integer", "minimum": 1,
                                                                    "maximum": int(max_wait_s),
                                                                    "description": "simulated seconds to wait"}},
                       "required": ["seconds"], "additionalProperties": False},
                      returns="{waited_s}", duration_hint=None, meta={"harness": True, "read_only": True})


def _ensure_crucible() -> None:
    import crucible
    crucible.load_all_modules()        # registers the ow_* / mech_* families and every module (required)


# ============================================================================================== operator side
@dataclass
class _Slot:
    """One world on the server."""
    wid: str
    config: dict                       # {"family", "seed", "overrides", "warmup_s"} - evaluator-only
    envs: list = field(default_factory=list)
    participants: list = field(default_factory=list)     # [(principal, policy, ident)]
    next_participant_t: int = 0


@dataclass
class _Env:
    """Operator-side state of one actor."""
    index: int
    slot: int
    task: Any                          # GoalContract-like (world, endowment, horizon_s, description)
    handle: str = ""
    principal: str = ""
    ident: Any = None                  # ActorIdentity (bearer token: operator secret)
    env_id: str = ""
    episode_id: str = ""
    t0: int = 0
    horizon_s: int = 0
    pulled: int = 0                    # world seq up to which visible events are in the buffer
    buffer: list = field(default_factory=list)           # converted actor-visible events, seq order
    buffer_base: int = 0               # every visible event with seq > buffer_base is in the buffer
    delivered: int = 0                 # observe-stream cursor
    caused: dict = field(default_factory=dict)          # event seq -> action_id of the own command that emitted it
    own_principals: set = field(default_factory=set)
    own_accounts: set = field(default_factory=set)
    summary: dict = field(default_factory=dict)
    first: list = field(default_factory=list)           # observations still to deliver at the first observation
    tools: tuple = ((), [])           # (names, ToolSchemas) cache
    goal: Optional[GoalSpec] = None
    stats: dict = field(default_factory=lambda: {"actions": 0, "rejected": 0, "waits": 0, "events": 0})


class OpenWorldOperator:
    """Operator side of a batch of OpenWorld environments. See the module docstring. Construct through
    `open_world`. Holds the `WorldServer`; keep this object (and anything holding it, e.g. the evaluator) on the
    harness side."""

    def __init__(self, tasks: Sequence[Any] | Callable[[int, int], Any], n_envs: int | None = None, *,
                 shared_world: bool = False, store_url: str = "memory://", max_wait_s: int = 86400,
                 handles: Sequence[str] | None = None, seed: int = 0, participants: Sequence[tuple] = (),
                 participant_dt: int = 3600, buffer_limit: int = 500_000):
        _ensure_crucible()
        from crucible.api import WorldServer
        if callable(tasks):
            if n_envs is None:
                raise ValueError("n_envs is required with a task factory")
            self._task_fn = tasks
            self.n = int(n_envs)
        else:
            tasks = list(tasks)
            self.n = len(tasks) if n_envs is None else int(n_envs)
            if len(tasks) not in (1, self.n):
                raise ValueError("give one task per environment (or one for all)")
            self._task_fn = (lambda b, seed, _t=tasks: _with_seed(_t[b if len(_t) > 1 else 0], seed))
        self.shared_world = bool(shared_world)
        self.store_url = store_url
        self.max_wait_s = int(max_wait_s)
        self.handles = list(handles) if handles else None
        self.participant_specs = list(participants)
        self.participant_dt = int(participant_dt)
        self.buffer_limit = int(buffer_limit)
        self.rng = random.Random(seed)
        self.srv = WorldServer(store_url, lease_ttl_s=None, max_events=10 ** 9, max_scan=10 ** 12,
                               renderer=lambda o: "")
        self.slots: list[_Slot] = []
        self.envs: list[_Env] = [_Env(i, 0, None) for i in range(self.n)]
        self.episode = 0
        self.clock_hooks: list[Callable[[int], None]] = []       # hook(env_index) at every clock stop
        self.episode_hooks: list[Callable[[int], None]] = []     # hook(env_index) after a reset (before acting)
        self._closed = False

    # ------------------------------------------------------------------ worlds and episodes
    def world(self, b: int):
        """The live kernel World of environment b (operator / evaluator access only)."""
        return self.srv.world(self.slots[self.envs[b].slot].wid)

    def principal(self, b: int) -> str:
        return self.envs[b].principal

    def task(self, b: int):
        return self.envs[b].task

    def env_index(self, env_id: str) -> int:
        for e in self.envs:
            if e.env_id == env_id:
                return e.index
        raise KeyError(env_id)

    def _close_slots(self) -> None:
        for s in self.slots:
            try:
                self.srv.close_world(s.wid)
            except Exception:  # noqa: BLE001 - already closed / superseded
                pass
        self.slots = []

    def reset(self, seeds: Sequence[int] | None = None) -> list[list[Observation]]:
        """New episode: fresh worlds (one per env, or one shared), actors admitted with neutral handles, warmup run,
        genesis events replayed for the actor. Returns the first observations per env."""
        self._close_slots()
        self.episode += 1
        tasks = []
        for b in range(self.n):
            s = None if seeds is None else int(seeds[b] if len(seeds) > 1 else seeds[0])
            tasks.append(self._task_fn(b, s))
        if self.shared_world:
            w0 = _world_key(tasks[0])
            if any(_world_key(t) != w0 for t in tasks):
                raise ValueError("shared_world: every actor's task must name the same world (family, seed, overrides)")
            groups = [list(range(self.n))]
        else:
            groups = [[b] for b in range(self.n)]
        for g in groups:
            t = tasks[g[0]]
            cfg = {"family": t.world["family"], "seed": int(t.world["seed"]),
                   "overrides": copy.deepcopy(t.world.get("overrides", {})), "warmup_s": int(t.world.get("warmup_s", 0))}
            wid = self.srv.create_world({"family": cfg["family"], "overrides": cfg["overrides"]}, cfg["seed"],
                                        world_id=f"cw-{new_id()[:16]}")
            slot = _Slot(wid, cfg, list(g))
            self.slots.append(slot)
            si = len(self.slots) - 1
            w = self.srv.world(wid)
            used = set()
            for b in g:
                e = _Env(b, si, tasks[b])
                h = self.handles[b] if self.handles else neutral_handle(self.rng)
                while h in used:
                    h = neutral_handle(self.rng)
                used.add(h)
                seq0 = w.seq
                e.ident = self.srv.join_world(wid, h, endowment=int(getattr(tasks[b], "endowment", 0)))
                e.handle, e.principal = h, e.ident.principal
                e.own_principals = {e.principal}
                e.env_id = f"ow{b}-{new_id()[:10]}"
                e.episode_id = new_id("ep_")
                e.horizon_s = int(getattr(tasks[b], "horizon_s", 30 * DAY))
                genesis = [d for d in self.srv.replay_events(wid, 0, seq0, actor=e.ident)
                           if not (d["type"] in ("cmd", "cmd.rejected") and d["actor"] == e.principal)]
                e.first = [C.event_observation(d, e.own_accounts, None, e.own_principals) for d in genesis]
                e.first.insert(0, Observation(ObsKind.TEXT, content=f"You are {e.principal}.", entity=e.principal,
                                              entity_type="person", t=w.now / DAY, source="identity",
                                              evidence=EvidenceKind.OBSERVATIONAL, meta={"self": True}))
                e.buffer, e.caused = list(e.first[1:]), {}
                e.pulled = e.delivered = seq0
                e.buffer_base = 0             # the genesis replay covers every visible event from the world's start
                self.envs[b] = e
            for (h, pol, endow) in self.participant_specs:
                import numpy as np
                ident = self.srv.join_world(wid, h, endowment=int(endow))
                pol.reset(ident.principal, np.random.default_rng(self.rng.randrange(2 ** 31)))
                slot.participants.append([ident.principal, pol, ident])
            warm = cfg["warmup_s"]
            if warm:                      # the world runs (actors idle) until the opportunity exists
                self._advance_slot(si, w.now + warm, hooks=False)
            slot.next_participant_t = w.now
            for b in g:
                e = self.envs[b]
                e.t0 = w.now
                self._pull(e)
                e.goal = public_goal(tasks[b], e.horizon_s)
        for b in range(self.n):
            for h in self.episode_hooks:
                h(b)
        return [self._deliver(e, first=True) for e in self.envs]

    # ------------------------------------------------------------------ event stream
    def _ingest(self, e: _Env, events: list[dict]) -> None:
        for d in events:
            et, p = d["type"], d.get("payload")
            if isinstance(p, dict):
                if et == "id.member" and p.get("person") == e.principal and p.get("org"):
                    e.own_principals.add(p["org"])
                elif et == "bank.account" and p.get("owner") in e.own_principals and p.get("id"):
                    e.own_accounts.add(p["id"])
            e.buffer.append(C.event_observation(d, e.own_accounts, e.caused.get(int(d["seq"])), e.own_principals))
        if len(e.buffer) > self.buffer_limit:
            drop = len(e.buffer) - self.buffer_limit
            e.buffer_base = e.buffer[drop - 1].seq
            e.buffer = e.buffer[drop:]

    def _pull(self, e: _Env) -> None:
        """Bring the buffer up to the world's current seq through the actor's own observation (actor op)."""
        wid = self.slots[e.slot].wid
        o = self.srv.observe(wid, e.ident, max_events=10 ** 9, include_tools="never")
        evs = o.events
        if o.dropped or o.unscanned:      # never expected with these limits; recover losslessly from the log
            evs = [d for d in self.srv.replay_events(wid, e.pulled, o.seq, actor=e.ident)
                   if not (d["type"] in ("cmd", "cmd.rejected") and d["actor"] == e.principal)]
        self._ingest(e, [d for d in evs if d["seq"] > e.pulled])
        e.stats["events"] += len(evs)
        e.pulled = o.seq
        e.summary = o.summary
        for org, _ in (o.summary.get("identity", {}).get("orgs", {}) or {}).items():
            e.own_principals.add(org)
        for acct in (o.summary.get("bank", {}).get("balances", {}) or {}):
            e.own_accounts.add(acct)

    @staticmethod
    def _after(e: _Env, seq: int) -> list[Observation]:
        return e.buffer[bisect.bisect_right(e.buffer, seq, key=lambda o: o.seq):]

    def _deliver(self, e: _Env, first: bool = False) -> list[Observation]:
        new = self._after(e, e.delivered)
        if first:
            out = [e.first[0]] + [o for o in e.first[1:]] + new
            e.first = []
        else:
            out = new
        e.delivered = e.pulled
        now = self.srv.world(self.slots[e.slot].wid).now
        return out + C.summary_observations(e.summary, now, e.principal)

    def observe(self, b: int) -> list[Observation]:
        e = self.envs[b]
        self._pull(e)
        if e.first:
            return self._deliver(e, first=True)
        return self._deliver(e)

    def collect(self, b: int, since: int) -> tuple[list[Observation], int]:
        e = self.envs[b]
        self._pull(e)
        since = int(since)
        if since >= e.buffer_base:
            return self._after(e, since), e.pulled
        wid = self.slots[e.slot].wid
        old = [C.event_observation(d, e.own_accounts, e.caused.get(int(d["seq"])), e.own_principals)
               for d in self.srv.replay_events(wid, since, e.buffer_base, actor=e.ident)
               if not (d["type"] in ("cmd", "cmd.rejected") and d["actor"] == e.principal)]
        return old + list(e.buffer), e.pulled

    # ------------------------------------------------------------------ tools and actions
    def tools(self, b: int) -> list[ToolSchema]:
        e = self.envs[b]
        specs = self.srv.available_tools(self.slots[e.slot].wid, e.ident)
        names = tuple(s["name"] for s in specs)
        if names != e.tools[0]:
            e.tools = (names, [C.tool_schema(s) for s in specs] + [wait_tool(self.max_wait_s)])
        return list(e.tools[1])

    def execute(self, b: int, action: Action) -> ExecutionResult:
        e = self.envs[b]
        wid = self.slots[e.slot].wid
        w = self.srv.world(wid)
        t_s = w.now
        if action.tool == WAIT_TOOL:
            return self._wait(e, action, t_s)
        s0 = w.seq
        r = self.srv.execute_action(wid, e.ident, {"tool": action.tool, "args": dict(action.args or {})})
        ps = w.principals_of(e.principal)
        cmd_seqs = [ev.seq for ev in w.events_range(s0, w.seq)
                    if w.visible(ev, ps) and ev.type not in ("cmd", "cmd.rejected")]
        for s in cmd_seqs:
            e.caused[s] = action.action_id
        self._pull(e)
        mine = [o for o in self._after(e, s0) if o.caused_by == action.action_id]
        delta = sum(o.value for o in mine if o.kind is ObsKind.TRANSACTION and o.value is not None)
        cost_s = int(r.get("cost_s", 0) or 0)
        obs = C.result_observations(r, action.tool, t_s, action.action_id)
        e.stats["actions"] += 1
        err = None if r.get("ok") else str((r.get("error") or {}).get("code", "error"))
        if err:
            e.stats["rejected"] += 1
        return ExecutionResult(ok=bool(r.get("ok")), observations=obs, error=err, cost=float(max(0.0, -delta)),
                               duration=cost_s / DAY, started_t=t_s / DAY, finished_t=(t_s + cost_s) / DAY,
                               meta={"tool": r.get("tool") or action.tool, "cost_s": cost_s, "event_seqs": cmd_seqs,
                                     "cash_delta_cents": float(delta)})

    def _wait(self, e: _Env, action: Action, t_s: int) -> ExecutionResult:
        from crucible.core import schema as S
        errs = S.validate(wait_tool(self.max_wait_s).parameters, action.args or {})
        e.stats["waits"] += 1
        if errs:
            obs = [Observation(ObsKind.ERROR, content="invalid_action: " + "; ".join(errs[:4]), t=t_s / DAY,
                               source=WAIT_TOOL, evidence=EvidenceKind.CONTROLLED, caused_by=action.action_id,
                               meta={"tool": WAIT_TOOL, "code": "invalid_action", "message": "; ".join(errs[:4]),
                                     "details": None})]
            return ExecutionResult(False, obs, "invalid_action", started_t=t_s / DAY, finished_t=t_s / DAY,
                                   meta={"tool": WAIT_TOOL, "harness": True})
        s = int(action.args["seconds"])
        obs = [Observation(ObsKind.TOOL_RESULT, content=json.dumps({"waited_s": s}), t=t_s / DAY, source=WAIT_TOOL,
                           evidence=EvidenceKind.CONTROLLED, caused_by=action.action_id, meta={"tool": WAIT_TOOL})]
        return ExecutionResult(True, obs, None, 0.0, s / DAY, t_s / DAY, (t_s + s) / DAY,
                               meta={"tool": WAIT_TOOL, "harness": True, "cost_s": 0})

    # ------------------------------------------------------------------ the clock
    def _advance_slot(self, si: int, until: int, hooks: bool = True) -> None:
        """Advance one world to `until`, stopping at every daily check time of its actors (t0 + k*DAY) and at
        participant turns; runs the clock hooks (evaluator) at each stop."""
        slot = self.slots[si]
        w = self.srv.world(slot.wid)
        while w.now < until:
            stops = [until]
            for b in slot.envs:
                t0 = self.envs[b].t0
                if hooks and t0 <= w.now:
                    stops.append(t0 + ((w.now - t0) // DAY + 1) * DAY)
            if slot.participants:
                stops.append(max(slot.next_participant_t, w.now + 1))
            nxt = min(s for s in stops if s > w.now)
            self.srv.advance_time(slot.wid, until=int(nxt))
            if slot.participants and w.now >= slot.next_participant_t:
                self._participants_turn(slot)
                slot.next_participant_t = w.now + self.participant_dt
            if hooks:
                for b in slot.envs:
                    for h in self.clock_hooks:
                        h(b)

    def _participants_turn(self, slot: _Slot) -> None:
        """Independently controlled participants (scripted controls, frozen LLMs, historical checkpoints wrapped as
        Crucible policies) act once, through actor operations, in a seeded order."""
        order = list(range(len(slot.participants)))
        self.rng.shuffle(order)
        for i in order:
            principal, pol, ident = slot.participants[i]
            o = self.srv.observe(slot.wid, ident, max_events=200, include_tools="never")
            obs = {**o.summary, "time": o.time, "principal": principal, "events": o.events, "events_dropped": o.dropped}
            a = pol.act(obs, self.srv.available_tools(slot.wid, ident))
            if a is not None:
                pol.observe_result(a, self.srv.execute_action(slot.wid, ident, a))

    def advance(self, durations_days: Sequence[float]) -> list[list[Observation]]:
        """Operator clock. Independent worlds: each advances by its own duration. Shared world: synchronous rounds,
        the world advances by the largest requested duration of the round."""
        secs = [max(0, int(round(float(d) * DAY))) for d in durations_days]
        for si, slot in enumerate(self.slots):
            w = self.srv.world(slot.wid)
            dt = max(secs[b] for b in slot.envs)
            if dt > 0:
                self._advance_slot(si, w.now + dt)
            else:
                for b in slot.envs:
                    for h in self.clock_hooks:
                        h(b)
        out = []
        for e in self.envs:
            self._pull(e)
            out.append(self._deliver(e, first=bool(e.first)))
        return out

    def now_days(self, b: int) -> float:
        return self.srv.world(self.slots[self.envs[b].slot].wid).now / DAY

    def done(self, b: int) -> bool:
        e = self.envs[b]
        return self.srv.world(self.slots[e.slot].wid).now - e.t0 >= e.horizon_s

    # ------------------------------------------------------------------ snapshots, forks, checkpoints
    def snapshot(self, b: int) -> str:
        return self.srv.snapshot_world(self.slots[self.envs[b].slot].wid)

    def fork_world(self, b: int, ephemeral: bool = True):
        """Operator: an isolated fork of env b's world (for evaluator-only counterfactual checks). Returns the
        forked World; close it with `close_fork`."""
        sid = self.snapshot(b)
        wid = self.srv.fork_world(sid, ephemeral=ephemeral)
        return self.srv.world(wid)

    def close_fork(self, world) -> None:
        self.srv.close_world(world.world_id)

    def checkpoint(self, path: str) -> dict:
        """Persist the operator state at the current simulated time: world snapshots in the store (use a sqlite
        ``store_url`` for durability across processes) plus the bookkeeping (cursors, buffers, provenance map,
        credentials) in `path`. Participants with generator-based scripts cannot be checkpointed."""
        if any(s.participants for s in self.slots):
            raise ValueError("checkpoint with independently controlled participants is not supported")
        for e in self.envs:
            self._pull(e)
        snaps = [self.srv.snapshot_world(s.wid) for s in self.slots]
        state = {"snaps": snaps, "slots": [(s.wid, s.config, s.envs) for s in self.slots], "episode": self.episode,
                 "rng": self.rng.getstate(),
                 "envs": [{k: v for k, v in e.__dict__.items() if k not in ("ident",)} |
                          {"ident": (e.ident.world_id, e.ident.principal, e.ident.token, e.ident.account)}
                          for e in self.envs]}
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return {"snapshots": snaps, "path": path}

    def restore(self, path: str) -> None:
        """Continue from a checkpoint written by `checkpoint` (possibly by another process on the same store): each
        world is restored from its snapshot (a fork that keeps the actors' tokens), the bookkeeping is reloaded."""
        from crucible.api import ActorIdentity
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.slots = []
        for sid, (wid, cfg, envs) in zip(state["snaps"], state["slots"]):
            new = self.srv.restore_world(sid, name=f"{wid}")
            self.slots.append(_Slot(new, cfg, list(envs)))
        self.episode = state["episode"]
        self.rng.setstate(state["rng"])
        for i, d in enumerate(state["envs"]):
            e = _Env(i, 0, None)
            for k, v in d.items():
                if k != "ident":
                    setattr(e, k, v)
            _, principal, token, account = d["ident"]
            e.ident = ActorIdentity(self.slots[e.slot].wid, principal, token, account)
            self.envs[i] = e
            # the restored world continues at the snapshot: the session cursor of the fork starts there
            w = self.srv.world(self.slots[e.slot].wid)
            if w.seq != e.pulled:
                raise RuntimeError(f"restored world at seq {w.seq}, bookkeeping at {e.pulled}")

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.srv.close()


def _with_seed(task: Any, seed: int | None) -> Any:
    if seed is None:
        return task
    t = copy.copy(task)
    t.world = dict(task.world, seed=int(seed))
    return t


def _world_key(task: Any) -> str:
    w = task.world
    return json.dumps([w["family"], int(w["seed"]), w.get("overrides", {}), int(w.get("warmup_s", 0))], sort_keys=True)


def public_goal(task: Any, horizon_s: int) -> GoalSpec:
    """The agent-visible goal: the task's public text and numeric targets of its success predicate. Never the
    world family, seed or contract internals (bindings, verifier state)."""
    metrics: dict = {}
    pred = getattr(task, "success", None)
    if isinstance(pred, dict):
        from crucible.goals.metrics import predicate_leaves
        for lf in predicate_leaves(pred):
            if isinstance(lf.get("value"), (int, float)):
                metrics[lf["metric"]] = lf["value"]
                win = (lf.get("args") or {}).get("window_s")
                if win:
                    metrics[lf["metric"] + ".window_days"] = float(win) / DAY
    metrics.update(getattr(task, "public_metrics", {}) or {})
    return GoalSpec(goal_id=new_id("g_"), text=str(getattr(task, "description", "")), metrics=metrics,
                    deadline_days=horizon_s / DAY, family=str(getattr(task, "protocol", "") or "openworld"),
                    meta={"sustain_checks": int(getattr(task, "sustain_checks", 1) or 1)})


# ============================================================================================== agent side
class _ActorPort:
    """The adapter's only link to the operator: a fixed set of actor operations, held in closures (no attribute
    path from here to the server, the worlds or the goals)."""
    __slots__ = ("reset", "observe", "tools", "execute", "collect", "advance", "goals", "now", "done", "env_ids",
                 "close")


def _make_port(op: OpenWorldOperator) -> _ActorPort:
    p = _ActorPort()
    p.reset = lambda seeds: op.reset(seeds)
    p.observe = lambda b: op.observe(b)
    p.tools = lambda b: op.tools(b)
    p.execute = lambda b, a: op.execute(b, a)
    p.collect = lambda b, since: op.collect(b, since)
    p.advance = lambda d: op.advance(d)
    p.goals = lambda: [copy.deepcopy(e.goal) for e in op.envs]
    p.now = lambda b: op.now_days(b)
    p.done = lambda b: op.done(b)
    p.env_ids = lambda: [e.env_id for e in op.envs]
    p.close = lambda: op.close()
    return p


class OpenWorldAdapter:
    """Agent-side `EnvAdapter` over the real Crucible OpenWorld (see module docstring). Only records cross it."""

    def __init__(self, port: _ActorPort, n: int, max_wait_s: int):
        self._port = port
        self.n = int(n)
        self.max_wait_s = int(max_wait_s)

    def _batch(self, obs: list) -> ObservationBatch:
        return ObservationBatch(self._port.env_ids(), obs, [self._port.now(b) for b in range(self.n)],
                                [self._port.done(b) for b in range(self.n)])

    def reset(self, seeds: Sequence[int] | None = None) -> ObservationBatch:
        return self._batch(self._port.reset(seeds))

    def observe(self) -> ObservationBatch:
        return self._batch([self._port.observe(b) for b in range(self.n)])

    def available_tools(self) -> list:
        return [self._port.tools(b) for b in range(self.n)]

    def execute(self, actions: Sequence[Optional[Action]]) -> list:
        if len(actions) != self.n:
            raise ValueError(f"expected {self.n} actions, got {len(actions)}")
        return [None if a is None else self._port.execute(b, a) for b, a in enumerate(actions)]

    def collect_events(self, since: Sequence[int]) -> tuple:
        got = [self._port.collect(b, int(since[b])) for b in range(self.n)]
        return [g[0] for g in got], [g[1] for g in got]

    def advance_time(self, duration: Sequence[float]) -> ObservationBatch:
        if len(duration) != self.n:
            raise ValueError(f"expected {self.n} durations")
        return self._batch(self._port.advance(list(duration)))

    def goals(self) -> list:
        return self._port.goals()

    def close(self) -> None:
        self._port.close()


def open_world(tasks: Sequence[Any] | Callable[[int, int], Any], n_envs: int | None = None, *,
               shared_world: bool = False, store_url: str = "memory://", max_wait_s: int = 86400,
               handles: Sequence[str] | None = None, seed: int = 0, participants: Sequence[tuple] = (),
               participant_dt: int = 3600) -> tuple[OpenWorldAdapter, OpenWorldOperator]:
    """Create the agent-side adapter and its operator.

    `tasks`: GoalContract-like objects (``world = {family, seed, overrides[, warmup_s]}``, ``endowment`` cents,
    ``horizon_s``, ``description``; `goals.contracts.GoalContract` and the protocol tasks qualify), one per env or
    one for all, or a factory ``(env_index, seed | None) -> task``. `participants`: [(handle, crucible Policy,
    endowment_cents)] independently controlled actors added to every world and driven every `participant_dt`
    simulated seconds (heterogeneous populations; the worlds' own minds, firms and consumers are always there).
    Hand the adapter to the harness; keep the operator (and the evaluator built from it) away from the learner."""
    op = OpenWorldOperator(tasks, n_envs, shared_world=shared_world, store_url=store_url, max_wait_s=max_wait_s,
                           handles=handles, seed=seed, participants=participants, participant_dt=participant_dt)
    return OpenWorldAdapter(_make_port(op), op.n, max_wait_s), op
