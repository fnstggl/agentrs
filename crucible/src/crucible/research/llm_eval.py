"""The frozen language-model baseline: an `agents.llm.LLMPolicy` (any backend) run on hidden-mechanism tasks and
on an open-economy objective, with the outcome measured by the independent evaluator and the failure diagnosed.

Why this module exists. A frozen pretrained model is the reference for "performance from pretrained knowledge
alone, with no learning from experience". The 3.0 measurement of it was confounded by the interface (a truncated
tool list, cut prompts, first-line descriptions), and afterwards interface failure could not be told from
strategic failure. Here every run returns, together:

* ``outcome``      what the operator-side evaluator measured, never anything the agent said;
* ``transcript``   every turn: what the model wrote, the action, the complete result, its diagnosed class;
* ``accounting``   tokens and seconds;
* ``interface``    `discovery.interface_stamp` of the world, and an **audit** of what the prompt cut
                   (`interface_audit`): truncated prompts, the detail level the tool catalogue reached, results
                   longer than their budget, generations that hit the token limit;
* ``diagnosis``    `research.diagnostics.diagnose` with the controls on the same task as references.

`conduct(record)` then reads a stored record by fixed rules and names the kind of failure - lost to the
interface, refused by the world, passive, repetitive, or strategic - including the case the diagnosis leaves
unnamed (every call valid, none changing anything).

(a) Mechanism tasks (`run_mechanism`). A `research.tasks.Task` is hosted in its kernel world
(`Task.world_config()`), the agent is admitted with the task's starting balance and given **only** the public
objective text of `Task.public()`. It acts through the ordinary tool interface with `agents.base.Runner`. The
facility ticks every 3600 simulated seconds and the runner step is `RUNNER_DT` = 1800 seconds, so the agent gets
two actions per tick: enough to read the report of the last tick and then set levers for the next one (a
`rig.set` made at either turn applies to the coming tick; the latest report is also in every observation). Any
step that divides the tick can be passed (`dt=1200`: three actions). The outcome is the episode record of
`research.envs.MechVecEnv` - same keys, same arithmetic - computed from the world: totals from the operator-side
metric `mechanisms.module.rig_window`, the per-tick series from the facility's retained reports
(`mechanism_outcome`). `sim_check` replays the settings the world recorded through the kernel-free simulator
and requires the identical record, so the table of a tool-using agent and the table of a numeric learner are
the same measurement. Hypotheses filed with `rig.hypothesize` are scored by `mechanisms.evaluator.score_claims`.

Shortened horizons (`shorten`). A task evaluated on its first n ticks is a different problem: the reference
policy and the default policy are re-estimated on n ticks with the validity module's procedure (same search
budget, same evaluation seeds), and the public objective is restated for n ticks at the same fraction of the
gap. Nothing is rescaled by a factor.

(b) Open economy (`run_open_economy`). One outcome objective (`open_economy_goal`) in an `ow_*` world, verified
by `goals.contracts.GoalVerifier`; every service the agent or its organisations operate gets a
`usefulness.module.usefulness_report`.

Controls through the same harness: `IdlePolicy`, `RandomPolicy` (uniformly random tool, schema-valid arguments),
`RandomRigPolicy` (uniformly random valid lever settings every tick: the tool-interface twin of the numeric
`RandomLearner`) and `RigExperimenter`, a scripted operator that runs a one-lever-at-a-time experiment through
`rig.*` tool calls, files what it found and then exploits. The scripted operator is the *competent control*: its
result enters the diagnosis as a reference, which is what lets a diagnosis tell task difficulty from agent
failure. `SchedulePolicy(reference_schedule(task))` plays the task's private reference settings through the
same tools: it shows what the normalisation calls 1.0 in the kernel world and is context, never a competent
reference (it knows the hidden mechanism). None of these is an agent under test.

`EvalLLMPolicy` is `LLMPolicy` with two corrections to the v4 prompt that the base class does not offer: a budget
for the characters kept of each past result in "Your recent actions" (240 in the base class, less than one
`rig.read` report), and the removal of the observation's ``last action: none`` line, which the base class prints
on every turn, also right after an action. It also never takes a tool call out of an unfinished ``<think>``
block (a thinking model that ran out of generation budget has not answered).

`RunStore` makes a batch of runs resumable: one JSON file of finished records, one JSONL transcript per run;
several processes may write to it. Nothing in this module is reachable from inside a world.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from ..agents.base import Policy, Runner
from ..agents.llm import LLMPolicy, render_actor_observation, render_tool_list
from ..agents.scripted import IdlePolicy, RandomPolicy, ScriptPolicy, call, ok
from ..api import render as R
from ..core import discovery as D
from ..core.canonical import digest
from . import diagnostics as X
from .tasks import GOAL_FRACTION, TASK_VERSION, Task, cache_dir

FORMAT = "crucible-llm-eval/1"
RUNNER_DT = 1800                 # seconds between an agent's turns: two turns per 3600 s facility tick
OPEN_DT = 3600                   # open economy: one turn per simulated hour
DAY = 86400
RIG = "r1"                       # the facility of a task world (`Task.world_config`)
DISCOVERY_TOOLS = ("world.find_tools", "world.hosts", "world.describe", "world.validate", "world.help", "world.api")
SUMMARY_CHARS = 400              # `api.render.render_observation` cuts every module summary to this length
SHORT_VERSION = "short/1"

__all__ = ["FORMAT", "RUNNER_DT", "OPEN_DT", "EvalLLMPolicy", "RigExperimenter", "RandomRigPolicy", "IdlePolicy",
           "RandomPolicy", "SchedulePolicy", "reference_schedule", "shorten", "mechanism_outcome", "facility_trace",
           "sim_check", "score_hypotheses", "facility_behaviour", "interface_audit", "run_mechanism",
           "open_economy_goal", "run_open_economy", "reference_of", "RunStore", "conduct", "summary_rows",
           "render_table", "policy_config"]


# ====================================================================== the LLM policy under evaluation
class _CloseThought:
    """Backend proxy for models that think before they answer. `api.render.parse_action` removes complete
    ``<think>...</think>`` blocks only: when the generation budget ends inside a thought, a tool call the model
    merely *considered* there would be parsed and executed as its action. The proxy closes an unfinished block,
    so the reply contains no tool call (it is then counted as an unparseable reply, which is what it is)."""

    def __init__(self, backend: Any):
        self._backend = backend

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)

    def generate(self, messages, *args, **kw):
        g = self._backend.generate(messages, *args, **kw)
        if "<think>" in g.text and "</think>" not in g.text.rsplit("<think>", 1)[-1]:
            g.text += "\n</think>"
        return g


class EvalLLMPolicy(LLMPolicy):
    """`LLMPolicy` with two corrections to its v4 prompt, both off by default (with default arguments the prompt
    is the base class's, byte for byte - a test compares them):

    * ``history_chars`` / ``history_arg_chars``: the base class shows each of the last ``history`` actions with
      its arguments cut to 200 characters and its result cut to 240. A `rig.read` report is 500-900 characters,
      so the model sees every report it has read, except the latest, without most of its readings.
    * ``last_action_line=False``: the base class's v4 prompt renders the observation with no last result (the
      result has its own section) and the observation therefore always says ``last action: none`` - also right
      after an action, two lines above "Result of your last action". With False that line is left out whenever a
      result section follows (it stays when the agent really did nothing in its previous turn).

    Independently of the prompt, a reply that ends inside an unfinished ``<think>`` block is never mined for a
    tool call (`_CloseThought`); this only concerns models run with thinking enabled.

    Everything else is inherited; the v3 prompt is never touched."""

    def __init__(self, *args, history_chars: int = 240, history_arg_chars: int = 200, last_action_line: bool = True,
                 **kw):
        super().__init__(*args, **kw)
        self.history_chars, self.history_arg_chars = int(history_chars), int(history_arg_chars)
        self.last_action_line = bool(last_action_line)
        if self._backend is not None:
            self._backend = _CloseThought(self._backend)

    def build_messages(self, obs: Any, tools: list[dict]) -> tuple[list[dict], list[str]]:
        if self.interface != "v4":
            return super().build_messages(obs, tools)
        tools = [t for t in tools if not t["name"].startswith("admin.")]
        names = {t["name"] for t in tools}
        new = sorted(names - self.known_tools) if self.known_tools is not None else []
        self.known_tools = names
        recent = [h["action"]["tool"] for h in reversed(self.hist) if h.get("action")]
        parts = []
        if self.goal:
            parts.append(f"Goal: {self.goal}")
        parts.append(f"You are {getattr(self, 'actor', '?')}.")
        if self.hist:
            lines = []
            for h in self.hist[-self.k_hist:]:
                if h.get("action"):
                    a = h["action"]
                    shown = D.render_result(h.get("result"), self.history_chars) if h.get("result") else "no result"
                    args = R.compact_json(a.get("args", {}), self.history_arg_chars)
                    lines.append(f"- {a['tool']} {args} -> {shown}")
                else:
                    lines.append(f"- (no action: {h.get('note', 'wait')})")
            parts.append("Your recent actions:\n" + "\n".join(lines))
        seen = render_actor_observation(obs, None, self.max_events, self.payload_chars)
        if self.last_result and not self.last_action_line:
            seen = seen.replace("\nlast action: none\n", "\n", 1)
        parts.append("Observation:\n" + seen)
        if self.last_result:
            parts.append("Result of your last action:\n" + D.render_result(self.last_result, self.result_chars))
        if new:
            parts.append("Newly available tools: " + ", ".join(new))
        prio = tuple(new) + tuple(dict.fromkeys(recent))[:8]
        parts.append(f"Tools ({len(tools)}):\n" + D.render_catalogue(tools, self.tool_chars, hosts=self.host_summaries,
                                                                     priority=prio))
        parts.append("Your next action (one JSON tool call):")
        return [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": "\n\n".join(parts)}], new

    def usage(self) -> dict:
        u = super().usage()
        u["interface"].update(history_chars=self.history_chars, history_arg_chars=self.history_arg_chars,
                              last_action_line=self.last_action_line, payload_chars=self.payload_chars,
                              max_events=self.max_events)
        return u


def policy_config(policy: Policy) -> dict:
    """What identifies a policy's configuration in a record (JSON)."""
    out: dict[str, Any] = {"name": policy.name, "kind": policy.kind, "class": type(policy).__name__}
    if isinstance(policy, LLMPolicy):
        b = policy._backend
        out.update(model=policy.model_name, interface=policy.interface, tool_chars=policy.tool_chars,
                   tool_order=policy.tool_order, result_chars=policy.result_chars, history=policy.k_hist,
                   history_chars=getattr(policy, "history_chars", 240),
                   history_arg_chars=getattr(policy, "history_arg_chars", 200),
                   last_action_line=getattr(policy, "last_action_line", True),
                   payload_chars=policy.payload_chars, max_events=policy.max_events,
                   max_new_tokens=policy.max_new_tokens, temperature=policy.temperature,
                   host_summaries=bool(policy.host_summaries),
                   max_context=getattr(b, "max_context", None), thinking=getattr(b, "enable_thinking", None),
                   test_double=bool(getattr(b, "is_test_double", False)),
                   system_prompt_hash=digest([policy.system_prompt], 6))
    for k in ("horizon", "explore_frac", "min_hold", "max_hold", "settle", "z"):
        if hasattr(policy, k) and not isinstance(policy, LLMPolicy):
            out[k] = getattr(policy, k)
    return out


# ====================================================================== the private reference, through the tools
def reference_schedule(task: Task, budget: int = 96) -> list[dict]:
    """OPERATOR SIDE. The settings of the task's private reference policy (`[{"from": tick, "settings": {...}}]`),
    found again with the validity module's search on the task's horizon (deterministic: for a full-horizon task
    these are the settings whose net is `task.reference_net`)."""
    from ..mechanisms import evaluator as E
    from ..mechanisms.base import build_engine
    mech = task.mechanism()
    eng = build_engine(mech)
    orc = E.oracle(mech, 0, budget=budget, seeds=1, seed=0, ticks=int(task.horizon), engine=eng)
    if orc["net_constant"] >= orc["expected_net"] - 1e-9:
        return [{"from": 0, "settings": dict(orc["settings"])}]
    return [{"from": int(e["from"]), "settings": dict(e["settings"])} for e in orc["schedule"]]


class SchedulePolicy(Policy):
    """PRIVILEGED REFERENCE - not an agent and not a control. Applies a fixed schedule of settings through
    `rig.set` (with `reference_schedule(task)`: the settings found by search on the hidden mechanism). It shows
    what the task's normalisation calls 1.0 when it is played through the same tools in the same world; it says
    nothing about what an agent without the hidden mechanism can reach."""
    name, kind = "privileged_reference", "reference"

    def __init__(self, schedule: Sequence[Mapping], tick_s: int = 3600):
        self.schedule = sorted(({"from": int(e["from"]), "settings": dict(e["settings"])} for e in schedule),
                               key=lambda e: e["from"])
        self.tick_s = int(tick_s)

    def reset(self, actor, rng):
        super().reset(actor, rng)
        self.k = 0

    def act(self, obs, tools):
        if self.k < len(self.schedule) and int(obs["time"]) // self.tick_s >= self.schedule[self.k]["from"]:
            self.k += 1
            return call("rig.set", settings=dict(self.schedule[self.k - 1]["settings"]))
        return None


# ====================================================================== controls for facility tasks
class RandomRigPolicy(Policy):
    """Random valid actions on a facility: reads `rig.spec` once, then at its first turn of every tick sets every
    lever to a uniformly random value inside its public range. The tool-interface twin of
    `research.baselines.RandomLearner`; a floor, not an agent."""
    name, kind = "random_settings", "random"

    def reset(self, actor, rng):
        super().reset(actor, rng)
        self.levers: list[dict] | None = None
        self.tick_s = 3600
        self.last_tick = -1

    def act(self, obs, tools):
        if self.levers is None:
            return call("rig.spec")
        k = int(obs["time"]) // self.tick_s
        if k == self.last_tick:
            return None
        self.last_tick = k
        settings = {}
        for lv in self.levers:
            v = float(self.rng.uniform(lv["min"], lv["max"]))
            settings[lv["name"]] = float(min(lv["max"], max(lv["min"], round(v)))) if lv.get("integer") else v
        return call("rig.set", settings=settings)

    def observe_result(self, action, result):
        if action["tool"] == "rig.spec" and result.get("ok"):
            self.levers = result["result"]["levers"]
            self.tick_s = int(result["result"]["tick_s"])


class RigExperimenter(ScriptPolicy):
    """The competent control for a hidden-mechanism facility. Tool calls only (`rig.spec`, `rig.read`, `rig.set`,
    `rig.history`, `rig.hypothesize`) and no knowledge of any mechanism: a one-lever-at-a-time screening
    experiment, a confirmation run, then exploitation.

    1. *Baseline.* Leave the default settings for `hold` ticks.
    2. *Screen.* One lever at a time, everything else at the defaults: run the lever at the first and at the third
       quarter of its range for `hold` ticks each (a level it already sits at is skipped). Then measure the
       defaults again; the reference level is the mean of the two baseline arms (it averages out a drift).
    3. *Combine.* Every lever whose better level beat the reference by more than `z` standard errors is moved to
       that level, all at once, for `hold` ticks. If the combination beats the second baseline it is kept;
       otherwise (the levers interact) only the single best lever is moved.
    4. *Refine*, while the exploration budget (`explore_frac` of the horizon) allows: each moved lever, largest
       gain first, is pushed one more quarter in the direction that paid and stays there if that beats the arm
       before it.
    5. *Exploit.* The best settings stay for the rest of the horizon, and the signs measured in the screen are
       filed with `rig.hypothesize` (third-quarter arm minus first-quarter arm of every lever, on every gauge and
       on revenue, cost, net and score; a difference within two standard errors is filed as "no effect").

    The noise of one tick is pooled over all arms; the first `settle` ticks of an arm are left out of its mean (a
    setting applies from the next tick and some mechanisms respond with a lag). `hold` is the budget divided by
    the arms of steps 1-3, between `min_hold` and `max_hold`; when even `min_hold` does not fit, the last levers
    are not screened. `horizon` is the public number of ticks of the objective (default: the episode length
    `rig.spec` suggests). What it measured is kept in `arms`, `picks`, `best`, `claims` for inspection."""
    name = "scripted_experimenter"

    def __init__(self, horizon: int | None = None, explore_frac: float = 0.5, min_hold: int = 3, max_hold: int = 6,
                 settle: int | None = None, z: float = 1.5, file_claims: bool = True):
        self.horizon, self.explore_frac = horizon, float(explore_frac)
        self.min_hold, self.max_hold, self.z = int(min_hold), int(max_hold), float(z)
        self.settle = settle
        self.file_claims = bool(file_claims)
        self.arms: list[dict] = []
        self.picks: dict[str, float] = {}
        self.best: dict[str, float] = {}
        self.claims: list[dict] = []
        self.explored_ticks = 0
        self.hold = 0

    # ---- measuring one arm
    def _measure(self, settings: dict, hold: int, label: tuple):
        """Apply `settings` (a partial mapping; empty: keep), let `hold` ticks run, and return the arm: per
        target (mean, sum of squared deviations, n) over the ticks after `settle`. None if a call was refused."""
        if settings:
            r = yield call("rig.set", settings=settings)
            if not ok(r):
                return None
            first, at = int(r["result"]["applies_at_tick"]), int(r["result"]["applies_at"])
            want = r["result"]["settings"]
        else:
            r = yield call("rig.read")
            if not ok(r):
                return None
            first, at = int(r["result"]["ticks_done"]), int(r["result"]["next_tick_at"])
            want = r["result"]["next_settings"]
        ready = at + (hold - 1) * self.tick_s
        while int(self.obs["time"]) < ready:
            yield None
        r = yield call("rig.history", n=hold)
        if not ok(r):
            return None
        self.explored_ticks += hold
        reps = [x for x in r["result"]["reports"] if first <= x["t"] < first + hold]
        valid = len(reps) == hold and all(x["status"] == "ok" and x["settings"] == want for x in reps)
        rows = reps[self._settle:] if len(reps) > self._settle + 1 else reps
        stats = {}
        for tg in self._targets:
            key = {"revenue": "revenue_cents", "cost": "cost_cents", "net": "net_cents", "score": "score"}.get(tg)
            xs = [float(x[key]) if key else float(x["readings"][tg]) for x in rows]
            m = sum(xs) / len(xs) if xs else 0.0
            stats[tg] = (m, sum((x - m) ** 2 for x in xs), len(xs))
            if valid:
                ss, df = self._pool.get(tg, (0.0, 0))
                self._pool[tg] = (ss + stats[tg][1], df + max(0, len(xs) - 1))
        arm = {"phase": label[0], "lever": label[1], "value": label[2], "valid": valid, "stats": stats,
               "net": stats["net"][0] if valid else -math.inf, "n": stats["net"][2], "first_tick": first}
        self.arms.append({k: (v if k != "net" or valid else None) for k, v in arm.items() if k != "stats"})
        return arm

    def _se(self, target: str, n_a: int, n_b: int) -> float:
        """Standard error of the difference of two arm means of `target`, from the pooled one-tick variance."""
        ss, df = self._pool.get(target, (0.0, 0))
        if df <= 0:
            return math.inf
        return math.sqrt(ss / df * (1.0 / max(1, n_a) + 1.0 / max(1, n_b)))

    @staticmethod
    def _level(lv: dict, frac: float) -> float:
        v = float(lv["min"]) + frac * (float(lv["max"]) - float(lv["min"]))
        return float(round(v)) if lv.get("integer") else v

    def script(self):
        r = yield call("rig.spec")
        if not ok(r):
            return
        sp = r["result"]
        self.tick_s = int(sp["tick_s"])
        levers, self._targets = sp["levers"], list(sp["targets"])
        self._pool: dict[str, tuple[float, int]] = {}
        horizon = int(self.horizon or sp["horizon_ticks"])
        budget = max(1, int(self.explore_frac * horizon))
        dflt = {lv["name"]: float(lv["default"]) for lv in levers}
        by_name = {lv["name"]: lv for lv in levers}

        def levels(lv):
            span = float(lv["max"]) - float(lv["min"])
            out = []
            for f in (0.25, 0.75):
                v = self._level(lv, f)
                if abs(v - dflt[lv["name"]]) > 1e-9 * max(1.0, span) and v not in out:
                    out.append(v)
            return out
        plan = [(lv["name"], levels(lv)) for lv in levers]
        n_arms = 3 + sum(len(v) for _, v in plan)                    # two baselines, the screen, the combination
        hold = self.hold = min(self.max_hold, max(self.min_hold, budget // n_arms))
        self._settle = int(self.settle) if self.settle is not None else (1 if hold < 5 else 2)
        # ---- 1. baseline
        base0 = yield from self._measure({}, hold, ("baseline", None, None))
        if base0 is None:
            return
        # ---- 2. screen, one lever at a time
        screen: dict[str, list[dict]] = {}
        pending: dict[str, float] = {}
        for name, vals in plan:
            if not vals or self.explored_ticks + (len(vals) + 2) * hold > budget:
                continue
            arms = []
            for v in vals:
                arm = yield from self._measure({**pending, name: v}, hold, ("screen", name, v))
                pending = {}
                if arm is None:
                    return
                arms.append(arm)
            pending = {name: dflt[name]}
            screen[name] = arms
        base1 = yield from self._measure(dict(pending), hold, ("baseline", None, None))
        if base1 is None:
            return
        bases = [b for b in (base0, base1) if b["valid"]]
        if not bases:
            return
        ref = sum(b["net"] * b["n"] for b in bases) / sum(b["n"] for b in bases)
        n_ref = sum(b["n"] for b in bases)
        # ---- 3. combine the levels that paid
        gains = {}
        for name, arms in screen.items():
            top = max((a for a in arms if a["valid"]), key=lambda a: a["net"], default=None)
            if top is not None and top["net"] > ref + self.z * self._se("net", top["n"], n_ref):
                gains[name] = (top["net"] - ref, top["value"])
        cur = dict(dflt)
        last = base1                                 # the arm that measured the settings now in force
        order = sorted(gains, key=lambda n: -gains[n][0])
        if order:
            self.picks = {n: gains[n][1] for n in order}
            comb = None
            if len(order) > 1 and self.explored_ticks + hold <= budget:
                comb = yield from self._measure(dict(self.picks), hold, ("combine", None, None))
                if comb is None:
                    return
            if comb is not None and comb["valid"] and \
                    comb["net"] > base1["net"] + self.z * self._se("net", comb["n"], base1["n"]):
                cur.update(self.picks)
                last = comb
            else:                                    # one lever, or levers that do not add up: the best one alone
                back = {n: dflt[n] for n in order[1:]} if comb is not None else {}
                cur[order[0]] = gains[order[0]][1]
                r = yield call("rig.set", settings={**back, order[0]: cur[order[0]]})
                last = next(a for a in screen[order[0]] if a["value"] == cur[order[0]])
                order = order[:1]
        # ---- 4. refine: one more quarter in the direction that paid
        for name in order:
            lv = by_name[name]
            nxt = self._level(lv, 1.0 if cur[name] > self._level(lv, 0.5) else 0.0)
            if nxt == cur[name] or self.explored_ticks + hold > budget:
                continue
            arm = yield from self._measure({name: nxt}, hold, ("refine", name, nxt))
            if arm is None:
                return
            if arm["valid"] and arm["net"] > last["net"] + self.z * self._se("net", arm["n"], last["n"]):
                cur[name], last = nxt, arm
            else:
                r = yield call("rig.set", settings={name: cur[name]})
        # ---- 5. exploit, and file what the screen measured
        self.best = dict(cur)
        effects = [e for name in sorted(screen) for e in self._signs(name, screen[name], base0, base1, dflt[name])]
        if self.file_claims and effects:
            self.claims = effects
            for i in range(0, len(effects), 64):
                yield call("rig.hypothesize", effects=effects[i:i + 64],
                           note="signs measured one lever at a time: highest level minus lowest level")

    def _signs(self, lever: str, arms: list[dict], base0: dict, base1: dict, default: float) -> list[dict]:
        pts = [a for a in arms if a["valid"]]
        if len(pts) < 2:                              # one level only: compare it with the baseline arms
            pts = pts + [dict(b, value=default) for b in (base0, base1) if b["valid"]][:1]
        if len(pts) < 2:
            return []
        pts = sorted(pts, key=lambda a: a["value"])
        low, high = pts[0], pts[-1]
        out = []
        for tg in self._targets:
            d = high["stats"][tg][0] - low["stats"][tg][0]
            se = self._se(tg, high["stats"][tg][2], low["stats"][tg][2])
            if d == 0.0 or abs(d) <= 2.0 * se:
                out.append({"lever": lever, "target": tg, "sign": 0, "confidence": 0.6})
            else:
                out.append({"lever": lever, "target": tg, "sign": 1 if d > 0 else -1,
                            "confidence": 0.9 if abs(d) > 4.0 * se else 0.7})
        return out


# ====================================================================== shortened horizons
def _short_path(task: Task, ticks: int, key: str) -> str:
    return os.path.join(cache_dir(), "tasks_short", digest([TASK_VERSION, SHORT_VERSION, key], 6), task.family,
                        f"{task.seed}_{int(ticks)}.json")


def shorten(task: Task, ticks: int, *, n_seeds: int = 6, budget: int = 96, n_random: int = 160,
            use_cache: bool = True) -> Task:
    """The task restricted to its first `ticks` ticks, **with its private reference re-estimated on that horizon**.

    A shorter episode is a different problem (less time to recover the cost of exploring; lagged mechanisms have
    not settled), so the full-horizon numbers are not rescaled. Instead the validity module's procedure is run
    again on `ticks` ticks: the reference policy is searched with the same budget (`evaluator.oracle(...,
    ticks=ticks)`), and the reference, default and random nets are means over the same evaluation seeds. The
    public objective is restated at the same fraction (`tasks.GOAL_FRACTION`) of the new gap:
    ``goal = round(ticks * (default + GOAL_FRACTION * (reference - default)))``. The mechanism, the world seed
    (hence the noise of every tick), the income and the starting balance are unchanged, so the first `ticks`
    ticks of the kernel world are the same ticks.

    Raises ValueError when the reference does not beat the defaults on the short horizon by the validity
    module's margin (the shortened task would have no objective worth the name). Deterministic; cached under
    `CRUCIBLE_CACHE`."""
    ticks = int(ticks)
    if not task.validated:
        raise ValueError("only validated tasks carry a reference to re-estimate")
    if ticks >= task.horizon:
        return task
    if ticks < 8:
        raise ValueError("a horizon below 8 ticks leaves nothing to learn from")
    key = digest([n_seeds, budget, n_random, task.mech_seed, task.world_seed, task.horizon, task.income], 6)
    path = _short_path(task, ticks, key)
    if use_cache:
        try:
            with open(path) as fh:
                return Task.from_dict(json.load(fh))
        except (OSError, ValueError, TypeError):
            pass
    from ..mechanisms import evaluator as E
    from ..mechanisms import validity as V
    from ..mechanisms.base import build_engine
    mech = task.mechanism()
    eng = build_engine(mech)
    seat = 0
    seeds = [41_000_009 + k for k in range(n_seeds)]               # `validity.check_mechanism(seed=0)`
    orc = E.oracle(mech, seat, budget=budget, seeds=1, seed=0, ticks=ticks, engine=eng)
    ref = [{"from": e["from"], "settings": e["settings"]} for e in orc["schedule"]]
    ref_nets = np.asarray([float(E.run_schedule(mech, ref, s, ticks, seat, eng)["net"][:, seat].mean()) for s in seeds])
    if orc["net_constant"] >= orc["expected_net"] - 1e-9:
        ref_nets = V._nets(mech, eng, orc["settings"], seeds, ticks, seat)
    d_nets = V._nets(mech, eng, eng.defaults(seat), seeds, ticks, seat)
    rr = np.random.default_rng([0, 19, seat])
    levers = eng.levers(seat)
    r_nets = np.asarray([V._nets(mech, eng, {lv.name: lv.clip(rr.uniform(lv.lo, lv.hi)) for lv in levers},
                                 [seeds[k % n_seeds]], ticks, seat)[0] for k in range(n_random)])
    gap = ref_nets - d_nets
    gap_m = float(gap.mean())
    gap_se = float(np.std(gap, ddof=1) / math.sqrt(len(gap))) if len(gap) > 1 else 0.0
    margin = max(V.M_DEFAULT * float(task.income) / max(1, task.n_seats), 3 * gap_se)
    if gap_m < margin:
        raise ValueError(f"{task.id} shortened to {ticks} ticks: the reference beats the defaults by {gap_m:.1f} "
                         f"cents per tick, below the validity margin {margin:.1f}")
    d, r = float(d_nets.mean()), float(ref_nets.mean())
    out = dataclasses.replace(task, horizon=ticks, goal_net_cents=int(round(ticks * (d + GOAL_FRACTION * (r - d)))),
                              default_net=d, reference_net=r, random_net=float(r_nets.mean()))
    if use_cache:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as fh:
            json.dump(out.to_dict(), fh)
        os.replace(tmp, path)
    return out


# ====================================================================== the outcome of a facility episode
def _rig(world):
    from ..mechanisms.module import rig_module
    mod = rig_module(world)
    if mod is None or RIG not in mod.rigs:
        raise ValueError("this world hosts no facility 'r1'")
    return mod


def _own_reports(world, actor: str, rig: str = RIG) -> dict[int, dict]:
    """tick -> the report credited to `actor` (operator-side read of the retained reports of every seat)."""
    mod = _rig(world)
    out: dict[int, dict] = {}
    for seat in range(mod.rigs[rig].n_seats):
        for r in mod.reports(world, rig, seat):
            if r["holder"] == actor:
                out[int(r["t"])] = dict(r, seat=seat)
    return out


def mechanism_outcome(world, task: Task, actor: str, rig: str = RIG) -> dict:
    """The episode record of `research.envs.MechVecEnv` (same keys, same arithmetic), read from the world.

    Totals are the operator-side metric `rig_window` (the facility's exact running totals for the principal); the
    per-tick series - for the two half-episode scores and the fallback / idle counts - are the facility's
    retained reports. The world must have run exactly `task.horizon` ticks."""
    from ..mechanisms.module import rig_window
    mod = _rig(world)
    st = world.state
    h = int(task.horizon)
    done = int(st.get("rigs", rig)["tick"])
    if done != h:
        raise ValueError(f"the facility ran {done} ticks; the task's horizon is {h}")
    tot = rig_window(world, actor, rig, 0)
    reps = _own_reports(world, actor, rig)
    nets = np.zeros(h, dtype=float)
    fallback = idle = 0
    for t, r in reps.items():
        nets[t] = float(r["revenue_cents"]) - float(r["cost_cents"])
        fallback += r["status"] == "fallback"
        idle += r["status"] == "idle"
    net = float(tot["revenue"]) - float(tot["cost"])
    if len(reps) != tot["ticks"] or abs(float(nets.sum()) - net) > 1e-6:
        if h <= mod.rigs[rig].history:             # with a complete report history the two reads must agree
            raise RuntimeError("facility totals and retained reports disagree")
    bank = world.modules["bank"]
    acct = bank.primary_account(st, actor)
    half = h // 2
    return {"task": task.id, "family": task.family, "seed": task.seed, "engine": task.engine,
            "transfer": task.transfer, "split": task.split, "ticks": h, "net_cents": net,
            "net_per_tick": float(net / h), "revenue_cents": float(tot["revenue"]), "cost_cents": float(tot["cost"]),
            "objective_quantity": float(tot["score"]), "ticks_fallback": int(fallback), "ticks_idle": int(idle),
            "funds_end_cents": int(bank.balance(st, acct)) if acct is not None else None,
            "score": task.score(float(net / h)), "success": task.success(net),
            "score_first_half": task.score(float(nets[:half].mean())) if half else None,
            "score_second_half": task.score(float(nets[half:].mean())) if h - half else None,
            "random_score": None if task.random_net is None else task.score(task.random_net)}


def facility_trace(world, actor: str, rig: str = RIG, since_seq: int = 0) -> dict:
    """What happened at the facility, tick by tick, from the world: the settings the actor's seat *asked for* at
    every tick (the last accepted `rig.set` before the tick; the defaults before the first), the settings that
    actually ran and the status (from the reports), and whether anything other than operating the seat touched
    the measurement (the seat changed hands, or money other than the facility's settlements moved on the
    actor's accounts)."""
    mod = _rig(world)
    d = mod.rigs[rig]
    st = world.state
    h = int(st.get("rigs", rig)["tick"])
    reps = _own_reports(world, actor, rig)
    seats = sorted({r["seat"] for r in reps.values()}) or [0]
    seat = seats[0]
    sets: list[tuple[int, dict]] = []
    seat_events = other_money = 0
    accts = set(st.find_ids("accounts", "owner", actor))
    for e in world.events_range(max(int(since_seq), world.log_base), None):
        p = e.payload
        if e.type == "rig.set" and p.get("rig") == rig and p.get("seat") == seat:
            sets.append((int(e.t), dict(p["settings"])))
        elif e.type == "rig.seat" and p.get("rig") == rig:
            seat_events += 1
        elif e.type == "bank.txn" and p.get("kind") != "rig_settle" and \
                any(a in accts for a, _ in p.get("postings", [])):
            other_money += 1
    desired, k = [], 0
    cur = dict(d.defaults[seat])
    for t in range(h):
        while k < len(sets) and sets[k][0] < (t + 1) * d.tick_s:
            cur = dict(sets[k][1])
            k += 1
        desired.append(dict(cur))
    return {"rig": rig, "seat": seat, "ticks": h, "tick_s": d.tick_s, "defaults": dict(d.defaults[seat]),
            "desired": desired, "ran": [dict(reps[t]["settings"]) if t in reps else None for t in range(h)],
            "status": [reps[t]["status"] if t in reps else "empty" for t in range(h)],
            "net": [reps[t]["revenue_cents"] - reps[t]["cost_cents"] if t in reps else 0 for t in range(h)],
            "sets": len(sets), "seat_events": seat_events, "other_money_events": other_money,
            "clean": len(seats) == 1 and seat_events == 0 and other_money == 0 and len(reps) == h}


def sim_check(task: Task, trace: Mapping, outcome: Mapping) -> dict:
    """Replay the settings the world recorded through the kernel-free simulator (`Task.sim()`, the thing
    `MechVecEnv` steps) and compare the episode record with the one read from the world. `equal` is True when
    every number of the record is identical. Not applicable when something other than operating the seat touched
    the measurement (`facility_trace(...)["clean"]`)."""
    if not trace["clean"]:
        return {"applicable": False, "reason": "the seat changed hands, or other money moved on the actor's accounts"}
    sim = task.sim()
    seat = int(trace["seat"])
    h = int(task.horizon)
    nets = np.zeros(h)
    rev = cost = 0.0
    score = 0.0
    fallback = idle = 0
    prev = None
    for t in range(h):
        x = trace["desired"][t]
        arg = None if x == prev else (x if sim.n_seats == 1 else [x if s == seat else None for s in range(sim.n_seats)])
        res = sim.step(arg)
        prev = x
        rev += float(res.revenue[seat])
        cost += float(res.cost[seat])
        score += float(res.score[seat])
        nets[t] = float(res.revenue[seat]) - float(res.cost[seat])
        status = sim.last[seat]["status"]
        fallback += status == "fallback"
        idle += status == "idle"
    net = float(nets.sum())
    half = h // 2
    rec = {"net_cents": net, "net_per_tick": float(net / h), "revenue_cents": rev, "cost_cents": cost,
           "objective_quantity": score, "ticks_fallback": int(fallback), "ticks_idle": int(idle),
           "funds_end_cents": sim.funds[seat], "score": task.score(float(net / h)), "success": task.success(net),
           "score_first_half": task.score(float(nets[:half].mean())) if half else None,
           "score_second_half": task.score(float(nets[half:].mean())) if h - half else None}
    diff = {k: [outcome.get(k), v] for k, v in rec.items() if outcome.get(k) != v}
    return {"applicable": True, "equal": not diff, "differences": diff}


_EFFECTS: dict[tuple, dict] = {}


def score_hypotheses(world, task: Task, actor: str, rig: str = RIG) -> dict:
    """Score everything `actor` filed with `rig.hypothesize` against the true effects
    (`mechanisms.evaluator.score_claims`; a later claim about the same lever and target replaces an earlier one)."""
    mod = _rig(world)
    filed = mod.claims(world, rig, actor)
    effects = [e for c in filed for e in c["effects"]]
    preds = [p for c in filed for p in c["predictions"]]
    out: dict[str, Any] = {"filings": len(filed), "effects_filed": len(effects), "predictions_filed": len(preds),
                           "first_filed_at_tick": min((c["t"] for c in filed), default=None)}
    if not effects and not preds:
        return out
    from ..mechanisms import evaluator as E
    mech = task.mechanism()
    key = (task.family, task.mech_seed)
    if key not in _EFFECTS:
        if len(_EFFECTS) > 64:
            _EFFECTS.clear()
        _EFFECTS[key] = E.effect_matrix(mech, 0)
    s = E.score_claims(mech, {"effects": effects, "predictions": preds}, 0, effects=_EFFECTS[key])
    if s.get("prediction"):
        s["prediction"] = {k: v for k, v in s["prediction"].items() if k != "items"}
    out.update(s)
    return out


def facility_behaviour(trace: Mapping, steps: Sequence, n_levers: int | None = None) -> dict:
    """What the agent did with the facility, in numbers (no judgement): which `rig.*` tools it called and when
    first, how often it changed settings, how many levers it ever moved, how many of its changes moved exactly
    one lever, how many distinct settings ran, and how long the final settings were left alone."""
    tools: dict[str, int] = {}
    accepted: dict[str, int] = {}
    first: dict[str, int] = {}
    for i, s in enumerate(steps):
        if s.action:
            t = str(s.action.get("tool"))
            tools[t] = tools.get(t, 0) + 1
            if s.result and s.result.get("ok"):
                accepted[t] = accepted.get(t, 0) + 1
                first.setdefault(t, i)
    dflt = trace["defaults"]
    des = trace["desired"]
    h = len(des)
    names = sorted(dflt)
    changes = single = 0
    moved: set[str] = set()
    values: dict[str, set] = {n: {dflt[n]} for n in names}
    last_change = None
    prev = dflt
    for t, x in enumerate(des):
        diff = [n for n in names if x[n] != prev[n]]
        if diff:
            changes += 1
            single += len(diff) == 1
            last_change = t
        for n in names:
            values[n].add(x[n])
            if x[n] != dflt[n]:
                moved.add(n)
        prev = x
    distinct = len({json.dumps(x, sort_keys=True) for x in des})
    at_default = sum(1 for x in des if x == dflt)
    rig_calls = {k.split(".", 1)[1]: v for k, v in sorted(accepted.items()) if k.startswith("rig.")}
    return {"calls": dict(sorted(tools.items(), key=lambda kv: (-kv[1], kv[0]))),
            "rig_calls_accepted": rig_calls,
            "first_turn": {k: first[k] for k in sorted(first) if k.startswith("rig.")},
            "discovery_calls": sum(v for k, v in tools.items() if k in DISCOVERY_TOOLS),
            "read_spec": "rig.spec" in accepted, "set_levers": trace["sets"] > 0,
            "filed_hypotheses": "rig.hypothesize" in accepted,
            "setting_changes": changes, "single_lever_changes": single,
            "levers_moved": sorted(moved), "levers_total": n_levers if n_levers is not None else len(names),
            "values_tried_per_lever": {n: len(values[n]) for n in names},
            "distinct_settings": distinct, "ticks_at_default": at_default,
            "final_differs_from_default": bool(des) and des[-1] != dflt,
            "last_change_tick": last_change,
            "final_settings_held_ticks": h - last_change if last_change is not None else h}


# ====================================================================== what the prompt cut
def _catalogue_level(text: str) -> str:
    last = text.splitlines()[-1] if text else ""
    for level, mark in (("full", "complete:"), ("descriptions", "descriptions longer than"),
                        ("signatures", "Omitted: all descriptions"), ("arguments", "Omitted: argument types"),
                        ("names", "every tool is named"), ("partial", "(+k more per host)"),
                        ("hosts", "the names of all")):
        if mark in last:
            return level
    return "below_hosts"


def interface_audit(policy: Policy, steps: Sequence, tools: Sequence[Mapping]) -> dict:
    """Everything the prompt of this run shortened, counted from the run itself. For a policy that is not an
    `LLMPolicy` only ``{"applies": False}`` (a scripted control reads complete results).

    * ``prompts``        truncated at the context limit (per-turn flags of the backend), largest and mean prompt;
    * ``generation``     replies that used the whole ``max_new_tokens`` (a tool call may have been cut), and
                         replies of a thinking model that ended without anything after the thought;
    * ``catalogue``      the detail level the static tool list reached under ``tool_chars`` (v4: one level for
                         every tool; v3: how many tools kept a signature) and whether descriptions were first-line
                         only;
    * ``results``        results that were longer than ``result_chars`` when shown as "Result of your last
                         action" (v3: than the fixed 600), and longer than the width of a history line;
    * ``observation``    module summaries over the renderer's fixed 400 characters and event payloads over
                         ``payload_chars``, by turn.

    ``clean`` is True when no prompt was truncated, the catalogue was complete, and no result of the last action
    was cut - the conditions under which the interface did not withhold anything the world returned."""
    if not isinstance(policy, LLMPolicy):
        return {"applies": False}
    tools = [t for t in tools if not str(t.get("name", "")).startswith("admin.")]
    tr = policy.transcript
    toks = [r["prompt_tokens"] for r in tr if r.get("prompt_tokens") is not None]
    truncated = [r.get("turn", i) for i, r in enumerate(tr) if r.get("truncated")]
    at_limit = [r.get("turn", i) for i, r in enumerate(tr) if r.get("gen_tokens", 0) >= policy.max_new_tokens]
    thoughts = [r for r in tr if "<think>" in str(r.get("text", ""))]
    no_answer = [r.get("turn", i) for i, r in enumerate(tr) if "<think>" in str(r.get("text", ""))
                 and not str(r["text"]).rsplit("</think>", 1)[-1].strip()]
    v4 = policy.interface == "v4"
    hist_chars = getattr(policy, "history_chars", 240)
    if v4:
        text = D.render_catalogue(tools, policy.tool_chars, hosts=policy.host_summaries)
        level = _catalogue_level(text)
        cat = {"renderer": "catalogue", "budget_chars": policy.tool_chars, "chars": len(text), "level": level,
               "complete": level == "full", "tools": len(tools),
               "tools_with_signature": len(tools) if level in ("full", "descriptions", "signatures", "arguments")
               else 0,
               "first_line_descriptions": False}
        show_last = lambda res: D.render_result(res, policy.result_chars)          # noqa: E731
        show_full = lambda res: D.render_result(res, 10 ** 9)                      # noqa: E731
        show_hist = lambda res: D.render_result(res, hist_chars)                   # noqa: E731
        last_budget = policy.result_chars
    else:
        text = render_tool_list(tools, policy.tool_chars, (), 70, policy.tool_order)
        shown = {ln.split("(", 1)[0] for ln in text.splitlines() if "(" in ln and not ln.startswith("(+")}
        shown &= {t["name"] for t in tools}
        hosts_all = {t["name"].partition(".")[0] for t in tools}
        cat = {"renderer": "tool_list", "budget_chars": policy.tool_chars, "chars": len(text),
               "level": "full_lines" if len(shown) == len(tools) else "partial", "complete": False, "tools": len(tools),
               "tools_with_signature": len(shown),
               "hosts_without_any_signature": sorted(hosts_all - {n.partition(".")[0] for n in shown}),
               "first_line_descriptions": True, "description_chars": 70}
        show_last = lambda res: R.render_result(res)                               # noqa: E731
        show_full = lambda res: R.render_result(res, 10 ** 9)                      # noqa: E731
        show_hist = lambda res: R.render_result(res, 240)                          # noqa: E731
        last_budget = 600
    n_res = cut_last = cut_hist = 0
    cut_tools: dict[str, int] = {}
    longest = 0
    summaries_cut: dict[str, int] = {}
    payloads_cut = turns_obs = 0
    for s in steps:
        if s.result:
            n_res += 1
            full = show_full(s.result)
            longest = max(longest, len(full))
            if show_last(s.result) != full:
                cut_last += 1
                t = str(s.result.get("tool"))
                cut_tools[t] = cut_tools.get(t, 0) + 1
            cut_hist += show_hist(s.result) != full
        if isinstance(s.obs, Mapping):
            turns_obs += 1
            for k, v in s.obs.items():
                if k not in ("time", "principal", "events", "events_dropped") and \
                        len(R.compact_json(v, None)) > SUMMARY_CHARS:
                    summaries_cut[k] = summaries_cut.get(k, 0) + 1
            payloads_cut += sum(1 for e in s.obs.get("events", [])[-policy.max_events:]
                                if len(R.compact_json(e.get("payload", {}), None)) > policy.payload_chars)
    ctx = getattr(policy._backend, "max_context", None)
    out = {"applies": True, "version": policy.interface,
           "prompts": {"n": len(tr), "truncated": len(truncated), "truncated_turns": truncated[:20],
                       "context_limit": ctx, "max_prompt_tokens": max(toks) if toks else None,
                       "mean_prompt_tokens": round(sum(toks) / len(toks), 1) if toks else None},
           "generation": {"max_new_tokens": policy.max_new_tokens, "at_limit": len(at_limit),
                          "at_limit_turns": at_limit[:20], "replies_with_thought": len(thoughts),
                          "thought_without_answer": len(no_answer), "thought_without_answer_turns": no_answer[:20]},
           "catalogue": cat,
           "results": {"n": n_res, "last_result_budget_chars": last_budget, "cut_as_last_result": cut_last,
                       "cut_as_last_result_by_tool": dict(sorted(cut_tools.items(), key=lambda kv: (-kv[1], kv[0]))),
                       "history_line_chars": hist_chars, "cut_in_history": int(cut_hist),
                       "longest_result_chars": longest, "error_detail_shown": v4},
           "observation": {"turns": turns_obs, "summary_chars": SUMMARY_CHARS,
                           "summaries_cut_by_module": dict(sorted(summaries_cut.items())),
                           "payload_chars": policy.payload_chars, "event_payloads_cut": int(payloads_cut)}}
    out["clean"] = bool(len(truncated) == 0 and cat["complete"] and cut_last == 0)
    return out


# ====================================================================== running
def _diag_interface(policy: Policy) -> dict | None:
    if not isinstance(policy, LLMPolicy):
        return None
    if policy.interface == "v4":
        return {"renderer": "catalogue", "budget_chars": policy.tool_chars}
    return {"renderer": "tool_list", "tool_chars": policy.tool_chars, "tool_order": policy.tool_order}


def _diagnose(world, actor: str, policy: Policy, runner: Runner, since: int, goal: Mapping,
              references: Iterable[Mapping] | None) -> dict:
    llm = isinstance(policy, LLMPolicy)
    return X.diagnose(world, actor, since_seq=since,
                      transcript=policy.transcript if llm else None, usage=policy.usage() if llm else None,
                      context_limit=getattr(policy._backend, "max_context", None) if llm else None,
                      max_new_tokens=policy.max_new_tokens if llm else None, interface=_diag_interface(policy),
                      goal=goal, references=list(references) if references else None,
                      trajectory=runner.trajs[actor].to_records(), with_args=False)


def _transcript(steps: Sequence, policy: Policy, diagnosis: Mapping, tick_s: int | None = None) -> list[dict]:
    """One record per turn: time, what the model wrote (LLM policies), the action, the complete result, how many
    events it emitted, and the class the diagnosis gave it."""
    llm = policy.transcript if isinstance(policy, LLMPolicy) else None
    acts = diagnosis.get("actions", [])
    by_turn = {a["turn"]: a for a in acts if a.get("turn") is not None} if llm is not None else {}
    queue = [a for a in acts if a.get("tool")] if llm is None else []
    out = []
    k = 0
    for i, s in enumerate(steps):
        rec: dict[str, Any] = {"turn": i, "t": s.obs.get("time") if isinstance(s.obs, Mapping) else None}
        if tick_s and rec["t"] is not None:
            rec["tick"] = int(rec["t"]) // tick_s
        if llm is not None and i < len(llm):
            lr = llm[i]
            rec.update(text=lr.get("text"), prompt_tokens=lr.get("prompt_tokens"), gen_tokens=lr.get("gen_tokens"),
                       prompt_chars=lr.get("prompt_chars"))
            for key in ("truncated", "wait", "parse_error", "new_tools"):
                if lr.get(key):
                    rec[key] = lr[key]
        rec["action"] = s.action
        rec["result"] = s.result
        rec["events"] = len(s.events or [])
        a = by_turn.get(i) if llm is not None else None
        if llm is None and s.action is not None and k < len(queue):
            a = queue[k]
            k += 1
        if a is not None:
            rec["class"] = a["class"]
            if a.get("sub"):
                rec["sub"] = a["sub"]
            if a.get("flags"):
                rec["flags"] = a["flags"]
        if isinstance(s.obs, Mapping):
            rec["obs"] = {key: v for key, v in s.obs.items() if key not in ("events", "principal")}
            rec["obs"]["events"] = len(s.obs.get("events", []))
        rec["wall_ms"] = round(float(s.wall_ms), 1)
        out.append(rec)
    return out


def _accounting(policy: Policy, wall_s: float, sim_s: int, turns: int) -> dict:
    u = policy.usage() if isinstance(policy, LLMPolicy) else {}
    gen_s = float(u.get("gen_s", 0.0))
    return {"turns": turns, "wall_s": round(wall_s, 2), "model_s": round(gen_s, 2),
            "world_s": round(max(0.0, wall_s - gen_s), 2), "simulated_s": int(sim_s),
            "calls": int(u.get("calls", 0)), "prompt_tokens": int(u.get("prompt_tokens", 0)),
            "gen_tokens": int(u.get("gen_tokens", 0)), "parse_failures": int(u.get("parse_failures", 0)),
            "waits": int(u.get("waits", 0)), "truncated_prompts": int(u.get("truncated_prompts", 0)),
            "policy_usage": u}


def reference_of(record: Mapping, competent: bool) -> dict:
    """A run record as a reference for `diagnostics.diagnose(references=...)`."""
    o = record["outcome"]
    return {"name": record["policy"]["name"], "competent": bool(competent), "success": bool(o.get("success")),
            "progress": float(o.get("progress") or 0.0), "score": o.get("score"), "run": record.get("run")}


def _progress(score: float | None) -> float:
    """Progress toward the public objective of a mechanism task in [0, 1]: the objective sits at normalised score
    `GOAL_FRACTION`, the defaults at 0."""
    if score is None:
        return 0.0
    return float(min(1.0, max(0.0, score / GOAL_FRACTION)))


def _give_host_summaries(policy: Policy, world, enabled: bool) -> None:
    """A v4 `LLMPolicy` without host summaries gets the world's own one-line host descriptions
    (`discovery.host_summaries`, what `world.hosts` returns) for the headers of its tool catalogue."""
    if enabled and isinstance(policy, LLMPolicy) and policy.interface == "v4" and policy.host_summaries is None:
        policy.host_summaries = D.host_summaries(world)


def run_mechanism(task: Task, policy: Policy, *, run: str | None = None, dt: int = RUNNER_DT, handle: str = "agent",
                  seed: int = 0, references: Iterable[Mapping] | None = None, goal_text: str | None = None,
                  on_tick: Callable[[int, Any], None] | None = None, strict: bool = True,
                  host_summaries: bool = True) -> dict:
    """Run `policy` on a hidden-mechanism task in its kernel world and return the complete record.

    The agent is admitted as `handle` with the task's starting balance; if the policy has ``set_goal`` it is
    given the public objective text (`Task.public()["objective"]`) and nothing else. `dt` must divide the
    facility's tick (3600 s): the agent gets ``3600 // dt`` turns per tick, and the run lasts exactly
    `task.horizon` ticks. `references`: `reference_of(...)` records of controls on the same task, for the
    diagnosis. `on_tick(tick, world)` is called after every tick (progress printing). `host_summaries`: a v4
    `LLMPolicy` is given the world's one-line host descriptions for its catalogue headers.

    Returns ``{"format", "kind": "mechanism", "run", "task" (public view and labels), "policy", "runner",
    "outcome" (the `MechVecEnv` episode record + "progress"), "sim_check", "hypotheses", "behaviour",
    "accounting", "interface" ({"stamp", "audit"}), "diagnosis" (without the per-action list), "invariants",
    "transcript"}``."""
    import crucible
    crucible.load_all_modules()
    from ..core.kernel import World
    from ..worldgen import generate
    if not task.validated or task.goal_net_cents is None:
        raise ValueError("a mechanism evaluation needs a validated task (it carries the objective and the reference)")
    cfg, wseed = task.world_config(holders=(handle,))
    world = World.create(generate(cfg, wseed), world_id=f"llmeval-{task.family}-{task.seed}-{run or policy.name}",
                         strict=strict)
    try:
        actor = world.system_call("admin.join", {"handle": handle, "endowment": int(task.endowment)})["principal"]
        mod = _rig(world)
        tick_s = int(mod.rigs[RIG].tick_s)
        if tick_s % int(dt):
            raise ValueError(f"dt must divide the facility tick ({tick_s} s)")
        per_tick = tick_s // int(dt)
        objective = goal_text if goal_text is not None else task.public()["objective"]
        if hasattr(policy, "set_goal"):
            policy.set_goal(objective)
        _give_host_summaries(policy, world, host_summaries)
        stamp = D.interface_stamp(world, actor)
        tools = world.available_tools(actor)
        since = world.seq
        runner = Runner(world, [(actor, policy)], dt=int(dt), seed=seed)
        t0 = time.time()
        for tick in range(int(task.horizon)):
            for _ in range(per_tick):
                runner.round()
            if on_tick is not None:
                on_tick(tick, world)
        wall = time.time() - t0
        steps = runner.trajs[actor].steps
        outcome = mechanism_outcome(world, task, actor)
        outcome["progress"] = _progress(outcome["score"])
        trace = facility_trace(world, actor, since_seq=since)
        check = sim_check(task, trace, outcome)
        diag = _diagnose(world, actor, policy, runner, since,
                         {"success": bool(outcome["success"]), "progress": outcome["progress"]}, references)
        transcript = _transcript(steps, policy, diag, tick_s)
        diag = {k: v for k, v in diag.items() if k != "actions"}
        pub = task.public()
        full_h = int(mod.rigs[RIG].engine.horizon)
        return {"format": FORMAT, "kind": "mechanism", "run": run or f"{task.id}/{policy.name}",
                "task": {**pub, "family": task.family, "seed": task.seed, "engine": task.engine,
                         "transfer": task.transfer, "split": task.split, "full_horizon_ticks": full_h,
                         "shortened": int(task.horizon) < full_h, "n_levers": len(trace["defaults"]),
                         "world_family": cfg["family"], "world_seed": wseed},
                "goal_text": objective, "policy": policy_config(policy),
                "runner": {"dt_s": int(dt), "tick_s": tick_s, "turns_per_tick": per_tick,
                           "turns": len(steps), "seed": seed},
                "outcome": outcome, "sim_check": check, "hypotheses": score_hypotheses(world, task, actor),
                "behaviour": facility_behaviour(trace, steps),
                "settings_by_tick": [trace["desired"][t] for t in range(trace["ticks"])],
                "accounting": _accounting(policy, wall, world.now, len(steps)),
                "interface": {"stamp": stamp, "audit": interface_audit(policy, steps, tools)},
                "diagnosis": diag, "invariants": world.check_invariants(), "transcript": transcript}
    finally:
        world.close()


# ====================================================================== (b) the open economy
def open_economy_goal(world_family: str = "ow_tiny", seed: int = 1, overrides: Mapping | None = None, days: int = 8,
                      window_days: int = 3, revenue_cents: int = 10_000, requests: int = 100, sustain_checks: int = 2,
                      endowment: int = 3_000_000):
    """One broad objective for an OpenWorld economy, stated as an outcome only: sustained revenue from independent
    customers through a service they actually use.

    Success (verified from authoritative state by `GoalVerifier`): on `sustain_checks` consecutive daily checks
    within `days` days, for an organisation the agent owns, `independent_revenue` over the preceding
    `window_days` days is at least `revenue_cents`, and its service has served independent customers at least
    `requests` successful requests in the same window (`use_independent_requests`). Revenue and requests of
    parties related to the agent never count (`goals.metrics.related_parties`). Progress is the revenue as a
    fraction of the threshold."""
    from ..goals.contracts import GoalContract
    w = int(window_days) * DAY
    text = (f"Earn sustainable revenue from independent customers with a service they actually use. Success is "
            f"measured on an organisation you own: on {int(sustain_checks)} consecutive daily checks within "
            f"{int(days)} days, its revenue from customers unrelated to you over the preceding {int(window_days)} "
            f"days is at least {int(revenue_cents)} cents, and a service it operates has served such customers at "
            f"least {int(requests)} successful requests in the same {int(window_days)} days. You start with "
            f"{int(endowment)} cents.")
    success = {"all": [{"metric": "independent_revenue", "args": {"principal": "$org", "window_s": w}, "op": ">=",
                        "value": int(revenue_cents)},
                       {"metric": "use_independent_requests", "args": {"host": "$host", "window_s": w}, "op": ">=",
                        "value": int(requests)}]}
    progress = {"metric": "independent_revenue", "args": {"principal": "$org", "window_s": w},
                "target": int(revenue_cents)}
    return GoalContract("ow_service_revenue", text, {"family": world_family, "seed": int(seed),
                                                     "overrides": dict(overrides or {})}, "$self", success, progress,
                        int(days) * DAY, int(sustain_checks), endowment=int(endowment), split="dev",
                        tags=("open_economy", "outcome_only"))


def _economy_state(world, actor: str, goal, days: int, battery: int) -> dict:
    """What exists and what it earned, from authoritative state: the organisations the actor belongs to, their
    repositories, instances, deployments and listings, the usefulness report of every service they operate, and
    independent revenue per principal over the whole run."""
    from ..goals.metrics import m_cash, m_independent_revenue
    from ..usefulness.module import usefulness_report
    st = world.state
    orgs = sorted(m["org"] for m in st.find("members", "person", actor))
    mine = [actor] + orgs
    span = int(days) * DAY

    def owned(table: str) -> list[tuple[str, dict]]:
        return [(k, v) for k, v in st.scan(table) if v.get("owner") in mine] if table in st.tables else []
    hosts = sorted({d["host"] for _, d in owned("deployments")})
    reports = {}
    for h in hosts:
        try:
            reports[h] = usefulness_report(world, h, span, battery=battery)
        except Exception as e:                                   # noqa: BLE001 - reported, never hidden
            reports[h] = {"error": f"{type(e).__name__}: {e}"}
    listings = [(k, v) for k, v in st.scan("listings") if v.get("seller") in mine] if "listings" in st.tables else []
    measures = [r["measures"] for r in reports.values() if "measures" in r]
    best = {k: max(m[k] for m in measures) for k in measures[0]} if measures else \
        {"artifact_creation": 0.0, "technical_correctness": 0.0, "real_usage": 0.0, "customer_value": 0.0,
         "financial_viability": 0.0, "long_term_adoption": 0.0}
    return {"organisations": orgs, "repositories": len(owned("repos")),
            "commits": int(sum(v.get("commits", 0) for _, v in owned("repos"))),
            "instances": len(owned("instances")), "deployments": len(owned("deployments")),
            "live_deployments": sum(1 for _, d in owned("deployments") if d.get("status") == "live"),
            "hosts": hosts, "listings": len(listings),
            "active_service_listings": sum(1 for _, l in listings if l.get("status") == "active" and l.get("host")),
            "independent_revenue_cents": {p: float(m_independent_revenue(world, p, span)) for p in mine},
            "cash_start_cents": int(goal.endowment), "cash_end_cents": float(sum(m_cash(world, p) for p in mine)),
            "usefulness": reports, "usefulness_best": best}


def run_open_economy(goal, policy: Policy, *, run: str | None = None, dt: int = OPEN_DT, handle: str = "nova",
                     seed: int = 0, references: Iterable[Mapping] | None = None, battery: int = 6,
                     on_day: Callable[[int, Any], None] | None = None, host_summaries: bool = True) -> dict:
    """Run `policy` on an open-economy objective (`open_economy_goal`) and return the complete record.

    The agent is admitted with the goal's endowment and (if it has ``set_goal``) given the goal's description.
    One turn every `dt` simulated seconds until the goal's horizon; `GoalVerifier.daily_check` runs at the end of
    every simulated day. The outcome is the verifier's status, the best progress seen at a daily check, and what
    exists in the world for the agent (`usefulness_report` of each service, with a hidden battery of `battery`
    requests). Returns the same keys as `run_mechanism` with ``"kind": "open_economy"`` and ``"economy"``."""
    from ..goals.contracts import GoalVerifier, instantiate
    world, actor = instantiate(goal, handle, strict=False)
    try:
        if hasattr(policy, "set_goal"):
            policy.set_goal(goal.description)
        _give_host_summaries(policy, world, host_summaries)
        stamp = D.interface_stamp(world, actor)
        tools = world.available_tools(actor)
        ver = GoalVerifier(goal, actor)
        ver.begin(world)
        since = world.seq
        days = int(goal.horizon_s // DAY)
        per_day = DAY // int(dt)
        if DAY % int(dt):
            raise ValueError("dt must divide a day")
        runner = Runner(world, [(actor, policy)], dt=int(dt), seed=seed)
        best = 0.0
        checks = []
        t0 = time.time()
        for day in range(days):
            for _ in range(per_day):
                runner.round()
            ver.daily_check(world)
            p = ver.progress(world)
            best = max(best, p)
            checks.append({"day": day + 1, "predicate": bool(ver.checks[-1][1]) if ver.checks else None,
                           "progress": round(p, 4)})
            if on_day is not None:
                on_day(day + 1, world)
        wall = time.time() - t0
        steps = runner.trajs[actor].steps
        status = ver.status(world)
        economy = _economy_state(world, actor, goal, days, battery)
        outcome = {"success": bool(status["success"]), "progress": float(status["progress"]),
                   "best_progress": round(best, 4), "success_day": None if status["success_t"] is None
                   else int(status["success_t"] // DAY), "streak": status["streak"], "daily_checks": checks,
                   "initially_true": status["initially_true"], "measures": economy["usefulness_best"],
                   "independent_revenue_cents": max(economy["independent_revenue_cents"].values(), default=0.0),
                   "score": None}
        diag = _diagnose(world, actor, policy, runner, since,
                         {"success": outcome["success"], "progress": outcome["best_progress"]}, references)
        transcript = _transcript(steps, policy, diag)
        diag = {k: v for k, v in diag.items() if k != "actions"}
        calls: dict[str, int] = {}
        for s in steps:
            if s.action:
                calls[str(s.action.get("tool"))] = calls.get(str(s.action.get("tool")), 0) + 1
        by_host: dict[str, int] = {}
        for k, v in calls.items():
            by_host[k.partition(".")[0]] = by_host.get(k.partition(".")[0], 0) + v
        return {"format": FORMAT, "kind": "open_economy", "run": run or f"{goal.id}/{policy.name}",
                "task": {"task": goal.id, "objective": goal.description, "world": goal.world, "days": days,
                         "success": goal.success, "sustain_checks": goal.sustain_checks,
                         "starting_balance_cents": goal.endowment},
                "goal_text": goal.description, "policy": policy_config(policy),
                "runner": {"dt_s": int(dt), "turns_per_day": per_day, "turns": len(steps), "seed": seed},
                "outcome": outcome, "economy": economy,
                "behaviour": {"calls": dict(sorted(calls.items(), key=lambda kv: (-kv[1], kv[0]))),
                              "calls_by_host": dict(sorted(by_host.items(), key=lambda kv: (-kv[1], kv[0]))),
                              "discovery_calls": sum(v for k, v in calls.items() if k in DISCOVERY_TOOLS)},
                "accounting": _accounting(policy, wall, world.now, len(steps)),
                "interface": {"stamp": stamp, "audit": interface_audit(policy, steps, tools)},
                "diagnosis": diag, "invariants": world.check_invariants(), "transcript": transcript}
    finally:
        world.close()


# ====================================================================== resumable batches
class RunStore:
    """Finished runs on disk, so a batch can be stopped and resumed.

    ``path`` is one JSON file ``{"format", "runs": {run id: record}}`` rewritten atomically after every run; the
    transcript of a run goes to ``<transcripts>/<prefix><run id>.jsonl`` (one turn per line) and is not kept in
    the JSON. `run(run_id, fn, config=...)` calls `fn` only if no record with that id **and the same
    configuration** is stored; a stored record with another configuration is replaced.

    Several processes may write to one store (stages run side by side): every write happens under a lock on the
    store's directory and starts from what is on disk, so a writer never drops another writer's runs."""

    def __init__(self, path: str, transcripts: str | None = None, prefix: str = ""):
        self.path = path
        self.transcripts = transcripts if transcripts is not None else os.path.dirname(os.path.abspath(path))
        self.prefix = prefix
        self.skipped: list[str] = []
        self.executed: list[str] = []
        self.runs: dict[str, dict] = {}
        self._dirty: set[str] = set()
        self.reload()

    def _read(self) -> dict[str, dict]:
        if not os.path.exists(self.path):
            return {}
        with open(self.path) as fh:
            data = json.load(fh)
        if data.get("format") != FORMAT:
            raise ValueError(f"{self.path}: not a {FORMAT} file")
        return data["runs"]

    def reload(self) -> None:
        """Take in what other writers stored since this store last looked (its own unwritten runs win)."""
        self.runs = {**self._read(), **{k: self.runs[k] for k in self._dirty if k in self.runs}}

    @staticmethod
    def _key(config: Any) -> str:
        return digest(json.loads(json.dumps(config, sort_keys=True, default=str)), 8)

    def has(self, run_id: str, config: Any = None) -> bool:
        r = self.runs.get(run_id)
        return r is not None and (config is None or r.get("config_key") == self._key(config))

    def get(self, run_id: str) -> dict | None:
        return self.runs.get(run_id)

    def transcript_path(self, run_id: str) -> str:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in run_id)
        return os.path.join(self.transcripts, f"{self.prefix}{safe}.jsonl")

    def put(self, run_id: str, record: Mapping, config: Any = None) -> dict:
        rec = {k: v for k, v in record.items() if k != "transcript"}
        rec["run"] = run_id
        if config is not None:
            rec["config_key"] = self._key(config)
        if record.get("transcript") is not None:
            os.makedirs(self.transcripts, exist_ok=True)
            tp = self.transcript_path(run_id)
            tmp = f"{tp}.{os.getpid()}.tmp"
            with open(tmp, "w") as fh:
                for row in record["transcript"]:
                    fh.write(json.dumps({"format": "crucible-llm-eval-turn/1", "run": run_id, **row},
                                        default=str) + "\n")
            os.replace(tmp, tp)
            rec["transcript_file"] = os.path.basename(tp)
            rec["transcript_turns"] = len(record["transcript"])
        self.runs[run_id] = json.loads(json.dumps(rec, default=str))
        self._dirty.add(run_id)
        self.flush()
        return self.runs[run_id]

    def flush(self) -> None:
        """Write this store's new or changed runs on top of what is on disk, under a lock on the directory."""
        import fcntl
        folder = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(folder, exist_ok=True)
        fd = os.open(folder, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            self.reload()
            tmp = f"{self.path}.{os.getpid()}.tmp"
            with open(tmp, "w") as fh:
                json.dump({"format": FORMAT, "runs": self.runs}, fh, indent=1, default=str)
            os.replace(tmp, self.path)
            self._dirty.clear()
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def touch(self, run_id: str) -> None:
        """Mark a stored record as changed in memory (it is written at the next `flush`)."""
        self._dirty.add(run_id)

    def run(self, run_id: str, fn: Callable[[], Mapping], config: Any = None) -> dict:
        self.reload()
        if self.has(run_id, config):
            self.skipped.append(run_id)
            return self.runs[run_id]
        rec = self.put(run_id, fn(), config)
        self.executed.append(run_id)
        return rec


# ====================================================================== interface failure or strategic failure?
def conduct(record: Mapping) -> dict:
    """Read one run's diagnosis and interface audit by fixed rules, to answer one question: was the run lost to
    the interface, or to what the agent chose to do? This adds no measurement; it names what the diagnosis
    leaves unnamed (`diagnose` has no label for an agent whose calls are all valid but never change anything).

    Shares (of turns): ``interface_share`` - replies that never became a well-formed call of an existing tool
    (unparsed, schema errors, unknown tools with a near name: the diagnosis families ``form`` and ``discovery``);
    ``refused_share`` - well-formed calls the world refused (wrong beliefs, rules, permissions, resources);
    ``accepted_share``; ``effect_share`` - accepted calls that changed the world; ``idle_share`` - waits.

    ``kind`` is the first of these that holds:

      goal_met          the objective was reached
      no_activity       no action was submitted
      interface         at least half of the prompts were truncated, or the tool list was uneven and the calls
                        were confined to its visible part (`diagnostics.tool_list_exposure`), or at least half
                        of the attempts were lost to form or discovery
      invalid_actions   fewer than half of the attempts were accepted (``largest`` names the biggest family)
      passive           attempts were accepted, and not one of them changed the world
      repetition        one identical call made up more than half of the accepted ones, or at least half of the
                        attempts repeated a call without new information (`diagnostics.find_loops`)
      strategic         accepted, effective, varied activity that did not reach the objective

    ``interface_confound`` is True when the prompt withheld something (`interface_audit(...)["clean"]` is False)
    or the tool-list exposure detector fired - whatever ``kind`` is.

    ``agent_progress`` against ``reference_progress`` / ``reference_score`` (the best competent reference on the
    same task, from the diagnosis' references) is the graded form of the difficulty question: a reference that
    missed the objective but got most of the way says something `success: false` does not."""
    d, audit = record["diagnosis"], record["interface"]["audit"]
    c, fam, st = d["counts"], d["families"], d["strategy"]
    turns, n = max(1, c["turns"]), c["attempts"]
    lost = fam["form"]["n"] + fam["discovery"]["n"]
    refused = {k: fam[k]["n"] for k in ("belief", "permission", "rule", "resource", "service", "code")}
    effects = int(st.get("effects", 0))
    truncated = audit["prompts"]["truncated"] if audit.get("applies") else 0
    exposure = (d["context"].get("tool_list") or {}).get("status") == "established"
    top = float(st.get("top_call_share", 0.0))
    out = {"interface_share": round(lost / turns, 4), "refused_share": round(sum(refused.values()) / turns, 4),
           "accepted_share": round(c["accepted"] / turns, 4), "effect_share": round(effects / turns, 4),
           "idle_share": round((c["waits"] + c["no_action"]) / turns, 4), "effects": effects,
           "top_call_share": top, "loop_rate": d["loops"]["rate"], "truncated_prompts": truncated,
           "interface_confound": bool(exposure or (audit.get("applies") and not audit["clean"]))}
    comp = [r for r in d["difficulty"].get("references", []) if r.get("competent")]
    o = record["outcome"]
    out["agent_progress"] = float(o.get("best_progress", o.get("progress")) or 0.0)
    out["reference_progress"] = max((float(r.get("progress") or 0.0) for r in comp), default=None)
    scores = [r["score"] for r in comp if r.get("score") is not None]
    out["reference_score"] = max(scores) if scores else None
    out["reference_succeeded"] = any(r.get("success") for r in comp) if comp else None
    if record["outcome"].get("success"):
        kind = "goal_met"
    elif n == 0:
        kind = "no_activity"
    elif truncated >= 0.5 * turns or exposure or lost >= 0.5 * n:
        kind = "interface"
    elif c["accepted"] < 0.5 * n:
        kind = "invalid_actions"
        out["largest"] = max(sorted(refused), key=lambda k: refused[k]) if sum(refused.values()) >= lost else "form"
    elif effects == 0:
        kind = "passive"
    elif top > 0.5 or d["loops"]["rate"] >= 0.5:
        kind = "repetition"
    else:
        kind = "strategic"
    out["kind"] = kind
    return out


# ====================================================================== tables
def summary_rows(records: Iterable[Mapping]) -> list[dict]:
    """One row per run with the numbers that make runs comparable: the outcome, the diagnosis breakdown (rate of
    every failure family, loop rate, primary failure, how many things were not determinable), what the interface
    cut, tokens and seconds."""
    rows = []
    for r in records:
        o, d, a, acc = r["outcome"], r["diagnosis"], r["interface"]["audit"], r["accounting"]
        c = d["counts"]
        row = {"run": r.get("run"), "kind": r["kind"], "task": r["task"]["task"],
               "engine": r["task"].get("engine"), "ticks": o.get("ticks"), "policy": r["policy"]["name"],
               "interface": r["policy"].get("interface"), "net_cents": o.get("net_cents"), "score": o.get("score"),
               "success": o.get("success"), "progress": o.get("progress"),
               "score_first_half": o.get("score_first_half"), "score_second_half": o.get("score_second_half"),
               "ticks_fallback": o.get("ticks_fallback"), "ticks_idle": o.get("ticks_idle"),
               "turns": c["turns"], "attempts": c["attempts"],
               "accepted_rate": round(c["accepted"] / c["attempts"], 4) if c["attempts"] else 0.0,
               "distinct_tools": c["distinct_tools"], "primary": d["primary"], "loop_rate": d["loops"]["rate"],
               "not_determinable": len(d["not_determinable"])}
        cd = conduct(r)
        row.update(conduct=cd["kind"], interface_share=cd["interface_share"], refused_share=cd["refused_share"],
                   effect_share=cd["effect_share"], top_call_share=cd["top_call_share"],
                   interface_confound=cd["interface_confound"], reference_score=cd["reference_score"],
                   reference_progress=cd["reference_progress"], reference_succeeded=cd["reference_succeeded"])
        for f in X.FAMILIES:
            row[f"rate_{f}"] = d["families"][f]["rate"]
        for k in ("code", "resources", "strategy", "difficulty"):
            row[k] = d[k]["status"]
        if a.get("applies"):
            row.update(prompts_truncated=a["prompts"]["truncated"], max_prompt_tokens=a["prompts"]["max_prompt_tokens"],
                       catalogue=a["catalogue"]["level"], results_cut=a["results"]["cut_as_last_result"],
                       gen_at_limit=a["generation"]["at_limit"], interface_clean=a["clean"])
        b = r.get("behaviour") or {}
        if r["kind"] == "mechanism":
            row.update(setting_changes=b.get("setting_changes"), levers_moved=len(b.get("levers_moved", [])),
                       levers_total=b.get("levers_total"), read_spec=b.get("read_spec"),
                       hypotheses=(r.get("hypotheses") or {}).get("effects_filed", 0),
                       sign_accuracy=(r.get("hypotheses") or {}).get("sign_accuracy"),
                       identification=(r.get("hypotheses") or {}).get("identification"),
                       sim_equal=(r.get("sim_check") or {}).get("equal"))
        else:
            e = r.get("economy") or {}
            row.update(organisations=len(e.get("organisations", [])), commits=e.get("commits"),
                       live_deployments=e.get("live_deployments"), listings=e.get("active_service_listings"),
                       independent_revenue_cents=o.get("independent_revenue_cents"),
                       real_usage=(o.get("measures") or {}).get("real_usage"))
        row.update(prompt_tokens=acc["prompt_tokens"], gen_tokens=acc["gen_tokens"], wall_s=acc["wall_s"],
                   model_s=acc["model_s"])
        rows.append(row)
    return rows


def render_table(rows: Sequence[Mapping], columns: Sequence[str] | None = None) -> str:
    """Rows as a GitHub-style text table (numbers rounded for reading; the JSON keeps them exact)."""
    if not rows:
        return ""
    cols = list(columns) if columns else list(rows[0])

    def cell(v: Any) -> str:
        if v is None:
            return "-"
        if isinstance(v, bool):
            return "yes" if v else "no"
        if isinstance(v, float):
            return f"{v:.3f}" if abs(v) < 100 else f"{v:.0f}"
        return str(v)
    body = [[cell(r.get(c)) for c in cols] for r in rows]
    width = [max(len(c), *(len(b[i]) for b in body)) for i, c in enumerate(cols)]
    lines = ["| " + " | ".join(c.ljust(width[i]) for i, c in enumerate(cols)) + " |",
             "|" + "|".join("-" * (width[i] + 2) for i in range(len(cols))) + "|"]
    lines += ["| " + " | ".join(b[i].ljust(width[i]) for i in range(len(cols))) + " |" for b in body]
    return "\n".join(lines)
