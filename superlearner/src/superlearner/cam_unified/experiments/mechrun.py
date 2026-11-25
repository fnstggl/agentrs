"""CAM Unified on real Crucible hidden-mechanism worlds: task streams, the SMDP episode driver, condition wrappers.

Clock. One decision = one tool call; the harness then advances the world by the action's charged duration plus the
think time (architecture decision 5). With ``tick_s = set_cost_s + think_s`` (60 s + 600 s) one lever setting is
followed by exactly one facility tick, so "k observations" means k ticks of the mechanism's response (a ``wait``
lets several ticks pass, a ``noop`` none or one).

Rewards. The verified return increments come from G's `MechWorldEvaluator.take_rewards()` (tick net cents over the
task's public money scale; exactly MechVecEnv's reward). Per-tick nets for the private normalisation (`Task.score`:
0 = defaults held, 1 = the task's reference policy) are read operator-side through `MechWorldOperator.tick_hooks` and
never reach the agent.

Splits. Training batches come from `TaskStream`, which checks every family and seed with G's `worldgen.SplitGuard`
(validation / test families and validation-range seeds are refused); evaluation tasks are validated tasks from
Crucible's validation seed range (`VAL_SEED0 + offset`, register: 100,000+ development, 200,000+ confirmation).

Condition wrappers (operator side; what the agent receives is ordinary observations):

* `PredecessorNotes` (Experiment 1 condition C): at reset each facility comes with up to m natural-language notes of a
  previous operator, who ran single-lever experiments from the resting settings on its own earlier instance of the
  same facility (a separate simulator run, not a fork of the agent's world). Only public quantities appear (lever
  names, values, net per tick).
* `DelayedPartial` (condition D): a fixed random half of the gauges is never reported (partial observability) and the
  money report of each tick (revenue / cost / net / funds, and the verified reward) arrives `delay` ticks late.

`fork_rewards` (evaluator side only) gives the true next-tick reward of hypothetical settings from a deep copy of the
world (same hidden state and noise stream): the targets of the prediction queries and of decision regret. Nothing it
computes is ever an agent input.
"""
from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from ..interfaces import Action, ActionSource, Observation, ObservationBatch, ObsKind

TICK_S, SET_COST_S, THINK_S = 660, 60, 600
DAY = 86400
WINDOWS = (0, 1, 2, 4, 8, 16, 32, 64, 128)


# ============================================================================ families and task streams

def ensure_registered() -> None:
    import crucible
    crucible.load_all_modules()
    from ..envs.worldgen import register_families
    register_families()


def domain_families(domain: str) -> list:
    """Training families of a domain (training split only): ``scm`` = Crucible's scm training families + the
    catalogue's scm training families; ``abm`` = the agent-based-market training families (a different simulator
    implementation); ``scm_core`` = Crucible's three scm training families only."""
    ensure_registered()
    from crucible.mechanisms.base import MECH_FAMILIES
    out = []
    for n, f in sorted(MECH_FAMILIES.items()):
        if f.split != "train" or getattr(f, "seats", 1) != 1:
            continue
        if domain == "scm" and f.engine == "scm":
            out.append(n)
        elif domain == "abm" and f.engine == "abm":
            out.append(n)
        elif domain == "scm_core" and f.engine == "scm" and not n.startswith("cw1_"):
            out.append(n)
    if not out:
        raise ValueError(f"no training families for domain {domain!r}")
    return out


def family_horizon(family: str) -> int:
    from crucible.research.tasks import make_task
    return int(make_task(family, 0, validate=False).horizon)


class TaskStream:
    """Endless guarded stream of unvalidated training tasks over `families`, batched so that every batch holds tasks of
    one horizon (no idle environments). Seeds run from `start` (disjoint per training seed); every task passes
    `SplitGuard.check_task` before it is yielded. State (`state_dict`) makes a run resumable."""

    def __init__(self, families: Sequence[str], start: int, seed: int = 0, guard=None):
        from ..envs.worldgen import SplitGuard
        ensure_registered()
        self.guard = guard or SplitGuard()
        self.families = list(families)
        for f in self.families:
            self.guard.check(f)
        self.horizon = {f: family_horizon(f) for f in self.families}
        self.next_seed = int(start)
        self.rng = random.Random(seed)
        self.drawn = 0

    def batch(self, n: int) -> list:
        from crucible.research.tasks import make_task
        hs = sorted(set(self.horizon.values()))
        h = self.rng.choice(hs)
        fams = [f for f in self.families if self.horizon[f] == h]
        out = []
        for _ in range(n):
            f = self.rng.choice(fams)
            t = make_task(f, self.next_seed, validate=False)
            self.guard.check_task(t)
            self.next_seed += 1
            out.append(t)
        self.drawn += n
        return out

    def state_dict(self) -> dict:
        return {"next_seed": self.next_seed, "rng": self.rng.getstate(), "drawn": self.drawn,
                "families": self.families}

    def load_state_dict(self, st: dict) -> None:
        self.next_seed = int(st["next_seed"])
        s = st["rng"]
        self.rng.setstate((s[0], tuple(s[1]), s[2]) if isinstance(s, list) else s)
        self.drawn = int(st.get("drawn", 0))


def eval_task_set(families: Sequence[str], n: int, offset: int, workers: int = 1) -> list:
    """Validated evaluation tasks (Crucible validation seed range, `VAL_SEED0 + offset + i`); families whose draws fail
    validity are left out and listed in `eval_task_set.failures`."""
    ensure_registered()
    from crucible.research.tasks import build_tasks, VAL_SEED0
    pairs = [(f, VAL_SEED0 + int(offset) + i) for f in families for i in range(n)]
    tasks = build_tasks(pairs, workers=workers, skip_failures=True)
    eval_task_set.failures = list(getattr(build_tasks, "failures", []))       # type: ignore[attr-defined]
    return tasks


# ============================================================================ worlds

@dataclass
class World:
    """One batch of MechWorld environments: the agent-side adapter, the operator (private) and the evaluator."""
    adapter: object
    operator: object
    evaluator: object
    tasks: list
    nets: list = field(default_factory=list)          # [B] per-tick net cents (operator side, metrics only)
    rewards: list = field(default_factory=list)       # [B] per-tick verified reward

    @property
    def B(self) -> int:
        return len(self.tasks)


def make_world(tasks: Sequence, *, tick_s: int = TICK_S, set_cost_s: int = SET_COST_S, max_wait_ticks: int = 4,
               surface: Optional[Callable] = None, seed: int = 0) -> World:
    from ..envs.mechworld import MechWorldEvaluator, mech_world
    ensure_registered()
    ad, op = mech_world(list(tasks), tick_s=tick_s, set_cost_s=set_cost_s, max_wait_s=max_wait_ticks * tick_s,
                        surface=surface, seed=seed)
    ev = MechWorldEvaluator(op)
    w = World(ad, op, ev, list(tasks))
    w.nets = [[] for _ in tasks]
    w.rewards = [[] for _ in tasks]

    def hook(b, net, reward, last):
        w.nets[b].append(float(net))
        w.rewards[b].append(float(reward))
    op.tick_hooks.append(hook)

    def begin(b):
        w.nets[b] = []
        w.rewards[b] = []
    op.episode_hooks.append(begin)
    return w


def fork_rewards(world: World, b: int, settings: Sequence[Sequence[float]]) -> list:
    """Evaluator side: the true reward of the next tick under each raw-settings vector (engine lever order) from a deep
    copy of env b's current state (hidden state and noise stream included). Never an agent input."""
    op = world.operator
    e = op.envs[b]
    out = []
    for raw in settings:
        env = copy.deepcopy(e.env)
        a = np.zeros((1, op.spec.max_levers))
        for j, lv in enumerate(e.lever_spec):
            span = lv.hi - lv.lo
            a[0, j] = 0.0 if span <= 0 else 2.0 * (float(lv.clip(raw[j])) - lv.lo) / span - 1.0
        _, rew, _, _ = env.step(a)
        out.append(float(rew[0]))
    return out


def numeric_view(world: World, b: int) -> np.ndarray:
    """Crucible's architecture-neutral numeric observation of env b after its latest tick (MechVecEnv's own vector):
    the input of the numeric baselines, the same information the adapter presents as entities and numbers."""
    return world.operator.envs[b].env._obs[0].copy()


def window_scores(task, nets: Sequence[float], windows: Sequence[int] = WINDOWS) -> dict:
    """Normalised score (Task.score of the mean net) of the ticks in [w_i, w_{i+1}) for consecutive window bounds,
    keyed by the window start (the number of observations already made when the window's decisions were taken)."""
    out = {}
    n = len(nets)
    bounds = list(windows) + [10 ** 9]
    for i, w in enumerate(windows):
        lo, hi = w, min(bounds[i + 1], n)
        if lo >= n or hi <= lo:
            break
        out[str(w)] = task.score(float(np.mean(nets[lo:hi])))
    return out


# ============================================================================ condition wrappers

NOTE_WORDING = (
    "Notes from the previous operator of this facility: with every other control at rest, setting {name} to {v} "
    "(it rests at {d}) changed the net income per tick from {a} to {b} cents.",
    "Earlier operator's log: moving {name} from its resting value {d} to {v} made net income go from {a} to {b} "
    "cents per tick (other controls left alone).",
    "Handover note: I once tried {name} = {v} instead of the default {d}; net per tick was {b} cents afterwards versus "
    "{a} cents at the defaults.",
)


class PredecessorNotes:
    """Condition C. `notes(world, b, m, rng)` -> m TEXT observations from single-lever experiments a previous operator
    ran on its own fresh instance of the same facility (two ticks at rest, then two ticks with one lever moved; the
    note reports the last tick of each)."""

    def __init__(self, ticks_before: int = 2, ticks_after: int = 2):
        self.tb, self.ta = int(ticks_before), int(ticks_after)

    def _experiment(self, task, spec, lever: int, value: float) -> tuple:
        from crucible.research.envs import MechVecEnv
        env = MechVecEnv([task], spec)
        env.reset()
        rest = np.asarray(env._norm_settings(0), dtype=np.float64)[None]
        base = None
        for _ in range(self.tb):
            _, _, _, info = env.step(rest)
            base = float(info["net_cents"][0])
        a = rest.copy()
        lo, hi = env._lo[0, lever], env._hi[0, lever]
        a[0, lever] = 0.0 if hi <= lo else 2.0 * (value - lo) / (hi - lo) - 1.0
        after = None
        for _ in range(self.ta):
            _, _, _, info = env.step(a)
            after = float(info["net_cents"][0])
        return base, after

    def notes(self, world: World, b: int, m: int, rng: random.Random) -> list:
        op = world.operator
        e = op.envs[b]
        out = []
        for i in range(int(m)):
            j = rng.randrange(len(e.lever_spec))
            lv = e.lever_spec[j]
            v = rng.uniform(lv.lo, lv.hi)
            v = float(int(round(v))) if lv.integer else round(v, 3)
            base, after = self._experiment(e.task, op.spec, j, v)
            txt = NOTE_WORDING[rng.randrange(len(NOTE_WORDING))].format(
                name=e.lever_shown[j], v=v, d=lv.default, a=int(round(base)), b=int(round(after)))
            out.append(Observation(ObsKind.TEXT, content=txt, entity=e.facility, entity_type="facility", t=0.0,
                                   source="predecessor", meta={"note": i, "declared_condition": "C:predecessor_notes"}))
        return out


MONEY_FIELDS = ("revenue_cents", "cost_cents", "net_cents")


class DelayedPartial:
    """Condition D over a World: per env a fixed random half of the gauges is hidden (their readings, reset relations
    and report entries are removed), and the money of each tick (revenue / cost / net / funds observations, the money
    fields of the report and the verified reward) is released `delay` ticks later as a separate 'facility.money_report'
    event plus the money numbers. Use `reset`, `advance` and `take_rewards` in place of the adapter / evaluator calls."""

    def __init__(self, world: World, delay: int = 3, hide_frac: float = 0.5, seed: int = 0):
        self.w = world
        self.delay = int(delay)
        self.hide_frac = float(hide_frac)
        self.rng = random.Random(seed)
        self.hidden: list = []
        self.queue: list = []          # [B] list of (release_tick, observations)
        self.rqueue: list = []         # [B] list of (release_tick, reward)
        self.ticks_seen: list = []
        world.operator.tick_hooks.append(self._on_tick)

    def _on_tick(self, b, net, reward, last):
        if b < len(self.rqueue):
            t = self.w.operator.envs[b].ticks
            self.rqueue[b].append((t + self.delay, float(reward)))

    def _filter(self, b: int, obs: list) -> list:
        hid = self.hidden[b]
        keep, money = [], []
        for o in obs:
            ent = o.entity or ""
            if ent.startswith("gauge:") and ent[6:] in hid:
                continue
            meta = o.meta if isinstance(o.meta, dict) else {}
            if o.kind == ObsKind.NUMERIC and (meta.get("field") in MONEY_FIELDS or meta.get("field") == "funds") \
                    and meta.get("tick") is not None:
                money.append(o)
                continue
            if o.kind == ObsKind.EVENT and o.content.startswith("{"):
                try:
                    d = json.loads(o.content)
                except ValueError:
                    d = None
                if isinstance(d, dict) and d.get("type") == "facility.report":
                    p = dict(d.get("payload") or {})
                    p["readings"] = {k: v for k, v in (p.get("readings") or {}).items() if k not in hid}
                    mon = {k: p.pop(k) for k in MONEY_FIELDS if k in p}
                    mon["tick"] = p.get("tick")
                    d["payload"] = p
                    o = copy.copy(o)
                    o.content = json.dumps(d)
                    money.append(Observation(ObsKind.EVENT, json.dumps({"type": "facility.money_report",
                                                                        "text": f"money of tick {mon['tick']}",
                                                                        "payload": mon}),
                                             entity=o.entity, entity_type=o.entity_type, t=o.t, seq=None,
                                             source=o.source, meta={"tick": mon.get("tick"), "delayed": True}))
            keep.append(o)
        if money:
            t = self.w.operator.envs[b].ticks
            self.queue[b].append((t + self.delay, money))
        return keep

    def _release(self, b: int, final: bool) -> list:
        t = self.w.operator.envs[b].ticks
        out, rest = [], []
        for rt, obs in self.queue[b]:
            if rt <= t or final:
                for o in obs:
                    o2 = copy.copy(o)
                    o2.seq = None
                    o2.t = t * self.w.operator.tick_s / DAY
                    o2.meta = {**(o.meta or {}), "delayed": True}
                    out.append(o2)
            else:
                rest.append((rt, obs))
        self.queue[b] = rest
        return out

    def reset(self) -> ObservationBatch:
        ob = self.w.adapter.reset()
        B = len(ob.env_ids)
        self.hidden, self.queue, self.rqueue = [], [[] for _ in range(B)], [[] for _ in range(B)]
        for b in range(B):
            names = list(self.w.operator.envs[b].gauge_shown)
            k = int(math.floor(self.hide_frac * len(names)))
            self.hidden.append(set(self.rng.sample(names, k)))
        obs = [self._filter(b, list(ob.observations[b])) for b in range(B)]
        return ObservationBatch(ob.env_ids, obs, ob.t, ob.done)

    def advance(self, durations) -> ObservationBatch:
        ob = self.w.adapter.advance_time(durations)
        obs = []
        for b in range(len(ob.env_ids)):
            o = self._filter(b, list(ob.observations[b]))
            obs.append(o + self._release(b, bool(ob.done[b])))
        return ObservationBatch(ob.env_ids, obs, ob.t, ob.done)

    def take_rewards(self, final: Optional[Sequence[bool]] = None) -> list:
        self.w.evaluator.take_rewards()        # keep the evaluator's pending counter in step (value unused)
        out = []
        for b in range(len(self.rqueue)):
            t = self.w.operator.envs[b].ticks
            fin = bool(final[b]) if final is not None else False
            r = sum(v for rt, v in self.rqueue[b] if rt <= t or fin)
            self.rqueue[b] = [(rt, v) for rt, v in self.rqueue[b] if not (rt <= t or fin)]
            out.append(float(r))
        return out


# ============================================================================ the episode driver

@dataclass
class EpisodeResult:
    task_id: str
    family: str
    seed: int
    nets: list
    rewards: list
    decisions: int
    tools: dict
    score: Optional[float]
    success: Optional[bool]
    windows: dict
    sim_days: float
    returns: float

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def run_batch(agent, world: World, *, learn: bool = False, update_every: int = 8, think_s: int = THINK_S,
              notes: Optional[Callable] = None, delayed: Optional[DelayedPartial] = None,
              max_rounds: int = 100_000, meter=None, on_update: Optional[Callable] = None,
              on_round: Optional[Callable] = None) -> list:
    """Run one episode of every env of `world` with `agent` (CamUnifiedAgent) to the horizon. ``learn``: call
    ``agent.update()`` every `update_every` rounds (continual learning on the fly); ``notes``: callable(world) -> [B]
    lists of extra observations delivered at reset, called after the reset built the facilities (condition C);
    ``delayed``: a DelayedPartial wrapper over this world (condition D); ``on_round(r, actions, results, rewards,
    next_obs)`` is called after every round r >= 1 and once with r = 0 (only the reset observations) after the reset
    (e.g. `predict.TrajectoryRecorder`)."""
    ad, ev = world.adapter, world.evaluator
    think = think_s / DAY
    ob = delayed.reset() if delayed is not None else ad.reset()
    B = len(ob.env_ids)
    if notes is not None:
        extra = notes(world)
        ob = ObservationBatch(ob.env_ids, [list(ob.observations[b]) + list(extra[b]) for b in range(B)], ob.t, ob.done)
    if agent.state is not None and set(ob.env_ids) != set(agent.env_ids):
        # CamUnifiedAgent.reset refuses an all-new batch once it holds state (agent.py:582-584 raises whenever the
        # id sets differ); archive the old batch's pending observations and start the new batch from scratch
        agent.flush("reset")
        agent.state = None
    agent.reset(ob.env_ids)
    agent.set_goals(ad.goals())
    if on_round is not None:
        on_round(0, None, None, None, ob)
    tools_used = [dict() for _ in range(B)]
    n_dec = [0] * B
    returns = [0.0] * B
    sim_days = 0.0
    rounds = 0
    updates = []
    while not all(ob.done) and rounds < max_rounds:
        tools = ad.available_tools()
        acts = agent.act(ob, tools)
        res = ad.execute(acts)
        dur = [((r.duration if r is not None else 0.0) + think) if a is not None else 0.0
               for a, r in zip(acts, res)]
        adv = delayed.advance(dur) if delayed is not None else ad.advance_time(dur)
        rew = delayed.take_rewards(adv.done) if delayed is not None else ev.take_rewards()
        nxt = [list(res[b].observations if res[b] is not None else []) + list(adv.observations[b]) for b in range(B)]
        nob = ObservationBatch(list(adv.env_ids), nxt, list(adv.t), list(adv.done))
        agent.observe(res, rew, nob)
        for b, a in enumerate(acts):
            if a is not None:
                n_dec[b] += 1
                tools_used[b][a.tool.split("_", 1)[0]] = tools_used[b].get(a.tool.split("_", 1)[0], 0) + 1
            returns[b] += float(rew[b])
        sim_days += float(sum(dur))
        rounds += 1
        if meter is not None:
            meter.add(env_steps=B, decisions=sum(a is not None for a in acts), sim_days=float(sum(dur)))
        if learn and update_every and rounds % update_every == 0:
            u = agent.update()
            if meter is not None and u.get("updated"):
                meter.add(updates=1)
            updates.append(u)
            if on_update is not None:
                on_update(u)
        if on_round is not None:
            on_round(rounds, acts, res, rew, nob)
        ob = nob
    agent.flush("episode_end")
    out = []
    for b, t in enumerate(world.tasks):
        nets = list(world.nets[b])
        h = max(1, len(nets))
        sc = t.score(float(np.sum(nets)) / h) if t.validated else None
        out.append(EpisodeResult(t.id, t.family, int(t.seed), nets, list(world.rewards[b]), n_dec[b], tools_used[b],
                                 sc, t.success(float(np.sum(nets))) if t.validated else None,
                                 window_scores(t, nets) if t.validated else {}, sim_days / B, returns[b]))
    run_batch.last_updates = updates                   # type: ignore[attr-defined]
    return out
