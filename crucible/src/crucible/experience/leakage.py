"""Explicit reward-leakage tests (MASTER_SPEC 11.3, Gate R) and the legacy-compatibility check.

Every place where Crucible computes a reward for a learner is exercised in observation-only mode:

    MechVecEnv (reward + observation slot scalars[0])     KernelMechEnv (the kernel bridge)
    SelfPlayEnv / KernelSelfPlayEnv (per-seat slots)        CrucibleEnv (Gymnasium: reward, info, termination)
    CrucibleParallelEnv (PettingZoo)                        scale.rollout.WorldEnv (balance-change default)
    MechExperienceEnv / KernelExperienceEnv (and the numeric view of the former)

and checked three ways: (1) the reward and the reward slot are zero at every step; (2) no evaluator key (goal
progress, streak, success time, objective status, private task numbers) appears anywhere in what the learner gets;
(3) **invariance**: what the learner gets is identical when only evaluator-side data differ (the task's private
reference numbers, the goal's evaluator parameters, the non-legacy reward mode), and none of the canaries planted in
evaluator-side or other participants' private data shows up in it. Legitimate consequences (revenue, cost, balance,
readings, events) must still be there and equal to what legacy mode shows; legacy mode itself may show more (the 4.0
contract's evaluator tools and fields, e.g. `goals.check_now`, the mechanism's `score`), so it is compared on the
legitimate fields only.

Key names and invariance cannot see a tool that hands out evaluator data identically in every mode; the
information-flow probes in `crucible.experience.probes` (exact-value, regression, predicate-oracle, tool-sweep, ...)
cover that, and Gate R runs both.

`legacy_check` runs the 4.0 implementation of each environment (loaded from the 4.0 commit `c07098b` with `git
show`) next to the current one in an explicitly selected legacy mode and requires byte-identical observations and
rewards (OO.8: historical reward-based experiments stay reproducible).
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import types
import uuid
from dataclasses import replace
from typing import Any, Callable

import numpy as np

from .modes import LEGACY, OBSERVATION_ONLY, VERIFIED

V4_COMMIT = "c07098b"
PRIVILEGED_KEYS = frozenset({"progress", "streak", "success_t", "initially_true", "objective", "reward",
                             "shaped_reward", "reference_net", "default_net", "random_net", "solvable",
                             "random_score", "score_first_half", "score_second_half", "objective_quantity"})


# ====================================================================== scanning helpers
def find_keys(x: Any, keys=PRIVILEGED_KEYS, path: str = "", out: list | None = None, depth: int = 0) -> list[str]:
    """Paths of dict keys in `keys` anywhere inside `x` (dict keys, and the last segment of flat dotted keys).
    An ``objective`` whose value is text (or None) is the public objective an observation carries by contract, not an
    evaluator status, and is not counted."""
    out = [] if out is None else out
    if depth > 32:
        return out
    if isinstance(x, dict):
        for k, v in x.items():
            last = str(k).rsplit(".", 1)[-1]
            if (k in keys or last in keys) and not (last == "objective" and (v is None or isinstance(v, str))):
                out.append(f"{path}.{k}")
            find_keys(v, keys, f"{path}.{k}", out, depth + 1)
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            find_keys(v, keys, f"{path}[{i}]", out, depth + 1)
    return out


def registered_hidden(x: Any, depth: int = 0) -> Any:
    """`x` (legacy-mode observations, events, results) without the fields the evaluator-only registry hides from
    participants (`core.evaluator_only.EVENT_FIELDS` / `RESULT_FIELDS`): what observation-only mode removes on
    purpose is not a legitimate field missing (e.g. the controller label of an admitted participant)."""
    from ..core import evaluator_only as EO
    if depth > 24:
        return x
    if isinstance(x, dict):
        keys = ()
        if isinstance(x.get("type"), str) and isinstance(x.get("payload"), dict):
            keys = EO.EVENT_FIELDS.get(x["type"], ())
            if keys:
                x = dict(x, payload=EO._strip(x["payload"], frozenset(keys)))
        if isinstance(x.get("tool"), str) and "result" in x:
            keys = EO.RESULT_FIELDS.get(x["tool"], ())
            if keys:
                x = dict(x, result=EO._strip(x["result"], frozenset(keys)))
        return {k: registered_hidden(v, depth + 1) for k, v in x.items()}
    if isinstance(x, list):
        return [registered_hidden(v, depth + 1) for v in x]
    return x


def dump(x: Any) -> str:
    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return repr(o)
    return json.dumps(x, sort_keys=True, default=default)


def canary_hits(outputs: list, canaries: list[str]) -> list[str]:
    text = "\n".join(dump(o) for o in outputs)
    return [c for c in canaries if c in text]


def new_canary(tag: str) -> str:
    return f"CANARY-{tag}-{uuid.uuid4().hex[:12]}"


# ====================================================================== the 4.0 implementations
def repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=here, text=True).strip()


def v4_module(relpath: str, package: str, name: str) -> types.ModuleType:
    """The 4.0 source of `crucible/src/crucible/<relpath>` at `V4_COMMIT`, imported as `<package>.<name>` (its
    relative imports resolve to the current, unchanged sibling modules)."""
    full = f"{package}.{name}"
    if full in sys.modules:
        return sys.modules[full]
    try:
        src = subprocess.check_output(["git", "show", f"{V4_COMMIT}:crucible/src/crucible/{relpath}"], cwd=repo_root(),
                                      text=True, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"the 4.0 commit {V4_COMMIT} is not readable in {repo_root()} ({(e.stderr or '').strip()[:200]});"
                           f" a shallow clone lacks it: git fetch --depth 1 origin <full sha of {V4_COMMIT}>") from e
    spec = importlib.util.spec_from_loader(full, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = package
    mod.__file__ = f"<{V4_COMMIT}:{relpath}>"
    sys.modules[full] = mod
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


# ====================================================================== fixtures
def tasks(validated: bool = True):
    from ..research.tasks import make_task
    one = make_task("scm_additive", 3, validate=validated)
    two = make_task("abm_duopoly", 2, validate=False)
    return one, two


def with_canary_private(task, value: float):
    """The same task with its private (operator-side) numbers replaced by a canary value."""
    return replace(task, reference_net=value, default_net=value / 3.0, random_net=-value)


def actions(rng: np.random.Generator, n: int, dim: int, steps: int) -> list[np.ndarray]:
    return [rng.uniform(-1, 1, size=(n, dim)) for _ in range(steps)]


def cash_goal(world_family: str = "tiny", seed: int = 0, target: int = 2_000_000, canary: str = ""):
    """A goal whose evaluator progress moves with the agent's cash, so legacy rewards are nonzero."""
    from ..goals.contracts import GoalContract
    return GoalContract("cash", f"Hold at least ${target // 100:,} in your own account.",
                        {"family": world_family, "seed": seed}, "$self",
                        {"metric": "cash", "args": {"principal": "$self"}, "op": ">=", "value": target},
                        {"metric": "cash", "args": {"principal": "$self"}, "target": target}, 30 * 86400,
                        params={"canary": canary} if canary else {})


# ====================================================================== individual leakage tests
def _numeric_slot(make_env: Callable[[str], Any], n_steps: int, dim: int, seats: int = 1, seed: int = 0) -> dict:
    """Run the same actions in legacy and observation-only mode: zero reward and reward slot in observation-only;
    the legitimate scalars (revenue, cost, funds, time) and every other observation entry equal to legacy's."""
    from ..research.learner import EnvSpec
    sp = EnvSpec()
    runs = {}
    for mode in (LEGACY, OBSERVATION_ONLY):
        env = make_env(mode)
        rng = np.random.default_rng(seed)
        obs = env.reset()
        rec = {"rewards": [], "obs": []}
        for _ in range(n_steps):
            n = getattr(env, "n", 1)
            act = rng.uniform(-1, 1, size=(n, dim))
            out = env.step([act] * seats if seats > 1 else act)
            o, r = out[0], out[1]
            rec["rewards"].append(np.asarray(r, dtype=np.float64).ravel())
            rec["obs"].append(np.concatenate([np.asarray(x).ravel() for x in (o if seats > 1 else [o])]))
            if np.asarray(out[2]).all():
                break
        runs[mode] = rec
    leg, obs_only = runs[LEGACY], runs[OBSERVATION_ONLY]
    nonzero_reward = sum(int(np.any(r != 0)) for r in obs_only["rewards"])
    obs_dim = sp.obs_dim
    slot = sp.scalars.start
    nonzero_slot, legit_missing = 0, 0
    for ol, oo in zip(leg["obs"], obs_only["obs"]):
        ol, oo = ol.reshape(-1, obs_dim), oo.reshape(-1, obs_dim)
        nonzero_slot += int(np.any(oo[:, slot] != 0))
        mask = np.ones(obs_dim, bool)
        mask[slot] = False
        legit_missing += int(not np.array_equal(ol[:, mask], oo[:, mask]))
    legacy_nonzero = sum(int(np.any(r != 0)) for r in leg["rewards"])
    return {"steps": len(obs_only["rewards"]), "nonzero_native_reward_steps": nonzero_reward + nonzero_slot,
            "legacy_nonzero_reward_steps": legacy_nonzero, "legitimate_fields_missing": legit_missing}


def test_mechvec(n_steps: int = 40) -> dict:
    from ..research.envs import MechVecEnv
    one, _ = tasks(False)
    m = _numeric_slot(lambda mode: MechVecEnv([one, one], reward_mode=mode), n_steps, 12)
    return _verdict("MechVecEnv: reward and obs slot zero, other entries equal to legacy", m)


def test_bridge(n_steps: int = 24) -> dict:
    from ..research.bridge import KernelMechEnv
    one, _ = tasks(False)
    m = _numeric_slot(lambda mode: KernelMechEnv(one, reward_mode=mode), n_steps, 12)
    return _verdict("KernelMechEnv (kernel bridge): reward and obs slot zero", m)


def test_selfplay(n_steps: int = 30) -> dict:
    from ..selfplay.arena import SelfPlayEnv
    _, two = tasks(False)
    m = _numeric_slot(lambda mode: SelfPlayEnv([two], reward_mode=mode), n_steps, 12, seats=2)
    return _verdict("SelfPlayEnv: every seat's reward and obs slot zero", m)


def test_kernel_selfplay(n_steps: int = 16) -> dict:
    from ..selfplay.arena import KernelSelfPlayEnv
    _, two = tasks(False)
    m = _numeric_slot(lambda mode: KernelSelfPlayEnv(two, reward_mode=mode), n_steps, 12, seats=2)
    return _verdict("KernelSelfPlayEnv: every seat's reward and obs slot zero", m)


def _gym_runs(make_env, act_fn, n_steps: int, multi: bool = False) -> dict:
    out = {}
    for mode in (LEGACY, OBSERVATION_ONLY):
        env = make_env(mode)
        obs, info = env.reset()
        rec = {"obs": [obs], "infos": [info], "rewards": [], "term": []}
        for k in range(n_steps):
            if multi:
                obs, rew, term, trunc, infos = env.step({a: act_fn(a, k) for a in env.agents})
                rec["rewards"].append(sorted(rew.values()))
                rec["term"].append(any(term.values()))
                rec["infos"].append(infos)
            else:
                obs, rew, term, trunc, info = env.step(act_fn(None, k))
                rec["rewards"].append(rew)
                rec["term"].append(term)
                rec["infos"].append(info)
            rec["obs"].append(obs)
            if (not multi and (term or trunc)) or (multi and not env.agents):
                break
        rec["verify"] = env.verify()
        env.close()
        out[mode] = rec
    return out


def _gym_metrics(runs: dict, multi: bool) -> dict:
    leg, oo = runs[LEGACY], runs[OBSERVATION_ONLY]
    nonzero = sum(1 for r in oo["rewards"] if (any(x != 0 for x in r) if multi else r != 0))
    priv = find_keys(oo["infos"])
    term = sum(1 for t in oo["term"] if t)
    # legitimate consequences: the events each step delivered (the rendered text also lists the tools, and legacy
    # mode's tool list carries the 4.0 evaluator tools)
    def evs(infos):
        return [(i.get("events") if not multi else {a: x.get("events") for a, x in i.items()}) for i in infos]
    legit = sum(1 for a, b in zip(evs(leg["infos"]), evs(oo["infos"])) if dump(registered_hidden(a)) != dump(b))
    verified = 1 if (oo["verify"] and (("progress" in oo["verify"]) if not multi else
                                       all("progress" in v for v in oo["verify"].values()))) else 0
    legacy_nonzero = sum(1 for r in leg["rewards"] if (any(x != 0 for x in r) if multi else r != 0))
    return {"steps": len(oo["rewards"]), "nonzero_native_reward_steps": nonzero, "privileged_progress_fields": len(priv),
            "outcome_terminations": term, "legitimate_fields_missing": legit, "verifier_outcomes_recorded": verified,
            "legacy_nonzero_reward_steps": legacy_nonzero, "privileged_paths": priv[:5]}


def test_gym(n_steps: int = 12) -> dict:
    from ..api.gym_env import CrucibleEnv
    from ..goals.contracts import RewardContract

    def make_env(mode):
        env = CrucibleEnv(spec={"family": "tiny", "overrides": {"n_consumers": 100}}, seed=4, endowment=1_500_000,
                          objective=RewardContract(cash_goal()), dt=3600, max_steps=n_steps, reward_mode=mode)
        env.server.join_world(env.world_id, "payee", endowment=0)
        return env

    acct: dict = {}

    def act(agent, k):
        return {"tool": "bank.transfer", "args": {"from_account": acct["a"], "to": "p:payee",
                                                  "amount": 1000 + 100 * k, "memo": "test"}}
    runs = {}
    for mode in (LEGACY, OBSERVATION_ONLY):
        env = make_env(mode)
        w = env.server.world(env.world_id)
        acct["a"] = w.modules["bank"].primary_account(w.state, env.principal)
        env.close()
    runs = _gym_runs(make_env, act, n_steps)
    return _verdict("Gymnasium CrucibleEnv: reward 0, no objective in info, no outcome termination, verifier on",
                    _gym_metrics(runs, False))


def test_pettingzoo(n_steps: int = 8) -> dict:
    from ..api.pettingzoo_env import CrucibleParallelEnv
    from ..goals.contracts import RewardContract

    def make_env(mode):
        return CrucibleParallelEnv(spec={"family": "tiny", "overrides": {"n_consumers": 100}}, seed=5,
                                   agents=["ann", "ben"], endowment=1_500_000,
                                   objectives={"ann": RewardContract(cash_goal()), "ben": RewardContract(cash_goal())},
                                   dt=3600, max_steps=n_steps, reward_mode=mode)

    accts: dict = {}

    def act(agent, k):
        other = "p:ben" if agent == "ann" else "p:ann"
        return {"tool": "bank.transfer", "args": {"from_account": accts[agent], "to": other,
                                                  "amount": 1000 + 500 * k + (300 if agent == "ann" else 0),
                                                  "memo": "test"}}
    env = make_env(LEGACY)
    w = env.server.world(env.world_id)
    for a, ident in env.identities.items():
        accts[a] = w.modules["bank"].primary_account(w.state, ident.principal)
    env.close()
    runs = _gym_runs(make_env, act, n_steps, multi=True)
    return _verdict("PettingZoo CrucibleParallelEnv: rewards 0, no objective in infos, verifier on",
                    _gym_metrics(runs, True))


def test_worldenv(n_steps: int = 10) -> dict:
    from ..scale.rollout import WorldEnv
    out = {}
    for mode in (LEGACY, OBSERVATION_ONLY):
        env = WorldEnv({"family": "tiny", "seed": 2, "overrides": {"n_consumers": 100}}, think_s=3600,
                       reward_mode=mode)
        w = env.world
        w.system_call("admin.join", {"handle": "payee", "endowment": 0})
        acct = w.modules["bank"].primary_account(w.state, env.actor)
        env.reset()
        rews, obs = [], []
        for k in range(n_steps):
            o, r, d, info = env.step({"tool": "bank.transfer", "args": {"from_account": acct, "to": "p:payee",
                                                                        "amount": 500 + k, "memo": "x"}})
            rews.append(r)
            obs.append(o)
        out[mode] = (rews, obs)
        env.close()
    m = {"steps": n_steps, "nonzero_native_reward_steps": sum(1 for r in out[OBSERVATION_ONLY][0] if r != 0),
         "legacy_nonzero_reward_steps": sum(1 for r in out[LEGACY][0] if r != 0),
         "legitimate_fields_missing": sum(1 for a, b in zip(out[LEGACY][1], out[OBSERVATION_ONLY][1])
                                          if dump(a) != dump(b))}
    return _verdict("scale.rollout.WorldEnv: no balance-change reward in observation-only mode", m)


def _drive_tier1(env, steps: int, seed: int = 0) -> list[dict]:
    """A fixed exploratory schedule of tool calls; returns everything the agent received."""
    rng = np.random.default_rng(seed)
    a = env.agents()[0]
    got = [env.observe(a), env.tools(a)]
    spec = env.execute(a, {"tool": "rig.spec", "args": {}})
    got.append(spec)
    levers = spec["result"]["levers"]
    rewards = []
    for k in range(steps):
        st = {lv["name"]: float(np.round(rng.uniform(lv["min"], lv["max"]), 4)) for lv in levers}
        got.append(env.execute(a, {"tool": "rig.set", "args": {"settings": st}}))
        env.advance(env.tick_s)
        rewards.append(env.reward(a))
        got.append(env.execute(a, {"tool": "rig.read", "args": {}}))
        got.append(env.observe(a))
    got.append(env.observe_since(a, 0))
    env._rewards_seen = rewards
    return got


def test_tier1_invariance(steps: int = 30) -> dict:
    """MechExperienceEnv: identical observations in all three modes and when the task's private numbers are replaced
    by a canary; zero reward in observation-only mode; the numeric view's reward slot is zero too."""
    from .tier1 import MechExperienceEnv
    one, _ = tasks(True)
    canary_val = 7777.123456
    outs = {}
    for label, task, mode in (("legacy", one, LEGACY), ("verified", one, VERIFIED), ("obs", one, OBSERVATION_ONLY),
                              ("obs_canary", with_canary_private(one, canary_val), OBSERVATION_ONLY)):
        env = MechExperienceEnv(task, reward_mode=mode)
        got = _drive_tier1(env, steps)
        outs[label] = (dump(got), env._rewards_seen, env)
    nonzero = sum(1 for r in outs["obs"][1] + outs["obs_canary"][1] if r != 0)
    differ = sum(int(outs[k][0] != outs["obs"][0]) for k in ("verified", "obs_canary"))
    legit_missing = _legit_missing_tier1(json.loads(outs["legacy"][0]), json.loads(outs["obs"][0]))
    hits = canary_hits([outs["obs_canary"][0]], [repr(canary_val), "7777.12"])
    priv = find_keys(json.loads(outs["obs"][0]))
    vec = outs["obs"][2].numeric()
    slot_nonzero = int(np.any(vec._obs[:, vec.spec.scalars.start] != 0))
    m = {"steps": steps, "nonzero_native_reward_steps": nonzero + slot_nonzero, "observation_differences": differ,
         "canary_hits": len(hits), "privileged_progress_fields": len(priv), "privileged_paths": priv[:5],
         "legacy_nonzero_reward_steps": sum(1 for r in outs["legacy"][1] if r != 0),
         "legitimate_fields_missing": legit_missing}
    return _verdict("Tier 1 experience env: observations invariant to the non-legacy mode and private task data; "
                    "legitimate fields equal to legacy's", m,
                    ok=lambda x: x["nonzero_native_reward_steps"] == 0 and x["observation_differences"] == 0
                    and x["canary_hits"] == 0 and x["privileged_progress_fields"] == 0
                    and x["legitimate_fields_missing"] == 0)


def _drive_kernel(env, steps: int, payee: str) -> list:
    a = env.agents()[0]
    w = env.world
    acct = w.modules["bank"].primary_account(w.state, a)
    got = [env.observe(a), env.tools(a)]
    for k in range(steps):
        got.append(env.execute(a, {"tool": "bank.transfer", "args": {"from_account": acct, "to": payee,
                                                                     "amount": 2000 + 10 * k, "memo": "m"}}))
        got.append(env.execute(a, {"tool": "forum.post", "args": {"board": "general", "title": f"note {k}",
                                                                  "body": "progress report from a participant"}}))
        env.advance(3600)
        got.append(env.observe(a))
    got.append(env.observe_since(a, 0))
    return got


CANARY_TAGS = ("RUNTIME", "INTERNAL", "PRIVATEMAIL", "RECSYS", "SPEC")


def plant_kernel_canaries(env, other: str, third: str, values: dict[str, str]) -> list[str]:
    """Plant values where no learner may look: evaluator runtime state (also the v5 recommender-private slot), an
    internal-visibility event, another participant's private mail, the world spec's evaluator parameters. Runs that
    are compared with a canary run plant neutral values of the same length, so the worlds are otherwise identical."""
    w = env.world
    w.runtime["evaluator_canary"] = values["RUNTIME"]
    w.runtime["recsys_private"] = {"params": values["RECSYS"]}
    w._run_system("canary", lambda ctx: ctx.emit("note", {"canary": values["INTERNAL"]}, vis="internal"))
    r = w.execute(other, {"tool": "mail.send", "args": {"to": [third], "subject": "private",
                                                       "body": f"between us: {values['PRIVATEMAIL']}"}})
    if not r.ok:
        raise RuntimeError(f"could not plant the private-mail canary: {r.error}")
    w.spec.params["evaluator_note"] = values["SPEC"]
    return [values[k] for k in CANARY_TAGS]


def test_kernel_invariance(steps: int = 8) -> dict:
    """KernelExperienceEnv with a cash objective: zero reward in observation-only mode, observations identical
    across reward modes and when evaluator-side and other participants' private data are replaced by canaries, and
    no planted canary in anything the agent receives."""
    from .kernel import KernelExperienceEnv
    canary = {k: new_canary(k) for k in CANARY_TAGS + ("GOAL",)}
    neutral = {k: "N" * len(v) for k, v in canary.items()}
    outs = {}
    for label, mode, vals in (("legacy", LEGACY, neutral), ("verified", VERIFIED, neutral),
                              ("obs", OBSERVATION_ONLY, neutral), ("obs_canary", OBSERVATION_ONLY, canary)):
        env = KernelExperienceEnv({"family": "tiny", "overrides": {"n_consumers": 100}}, 6, agents=["learner"],
                                  endowment=1_500_000, reward_mode=mode, objectives=cash_goal(canary=vals["GOAL"]))
        other = env.world.system_call("admin.join", {"handle": "other", "endowment": 10_000})["principal"]
        third = env.world.system_call("admin.join", {"handle": "third", "endowment": 10_000})["principal"]
        plant_kernel_canaries(env, other, third, vals)
        got = _drive_kernel(env, steps, other)
        rewards = [env.reward(env.agents()[0])]
        status = env.verify()
        outs[label] = (got, rewards, status)
    differ = sum(int(dump(outs[k][0]) != dump(outs["obs"][0])) for k in ("verified", "obs_canary"))
    legit_missing = _legit_missing_kernel(outs["legacy"][0], outs["obs"][0])
    canaries = list(canary.values())
    hits = canary_hits(outs["obs_canary"][0], canaries)
    priv = find_keys([g for g in outs["obs"][0]])
    verified = int(all("progress" in s for s in outs["obs"][2]["agents"].values()))
    m = {"steps": steps, "nonzero_native_reward_steps": sum(1 for r in outs["obs"][1] if r != 0),
         "legacy_nonzero_reward_steps": sum(1 for r in outs["legacy"][1] if r != 0),
         "observation_differences": differ, "canaries_planted": len(canaries), "canary_hits": len(hits),
         "privileged_progress_fields": len(priv), "privileged_paths": priv[:5], "verifier_outcomes_recorded": verified,
         "legitimate_fields_missing": legit_missing}
    return _verdict("Kernel experience env: invariance to the non-legacy mode and evaluator data; canaries absent; "
                    "legitimate fields equal to legacy's", m,
                    ok=lambda x: x["nonzero_native_reward_steps"] == 0 and x["observation_differences"] == 0
                    and x["canary_hits"] == 0 and x["privileged_progress_fields"] == 0
                    and x["verifier_outcomes_recorded"] == 1 and x["legitimate_fields_missing"] == 0)


LEGIT_T1 = ("revenue_cents", "cost_cents", "net_cents", "shortfall_cents", "ticks_done", "readings", "settings",
            "balance")


def _legit_missing_tier1(leg: list, obs: list) -> int:
    """Legitimate observation numbers (revenue, cost, net, readings, balances, settings) of legacy mode that the
    observation-only run lacks or shows differently (the 4.0 `score` is evaluator data and not compared)."""
    def nums(got):
        out = {}
        for i, o in enumerate(got):
            if isinstance(o, dict) and "numbers" in o:
                for k, v in o["numbers"].items():
                    if any(t in k for t in LEGIT_T1) and not k.endswith(".score"):
                        out[(i, k)] = v
        return out
    a, b = nums(leg), nums(obs)
    return sum(1 for k, v in a.items() if b.get(k) != v)


def _legit_missing_kernel(leg: list, obs: list) -> int:
    """Balances, resources and the events' consequence payloads of legacy mode that the observation-only run lacks:
    observations are compared field by field except `tools` (legacy lists the 4.0 evaluator tools) and `missing`
    (observation-only states that evaluator data are not observable)."""
    n = 0
    for x, y in zip(leg, obs):
        if isinstance(x, dict) and "numbers" in x and isinstance(y, dict):
            x = registered_hidden(x)
            for f in ("numbers", "resources", "events", "entities", "analytics"):
                if dump(x.get(f)) != dump(y.get(f)):
                    n += 1
    return n


def test_rig_hidden_canary(steps: int = 6) -> dict:
    """A canary inside the hidden mechanism's configuration of a kernel facility and inside a participant's
    internal filing (`rig.hypothesize` is recorded with internal visibility) never reaches the operator."""
    from ..worldgen import generate
    from .kernel import KernelExperienceEnv
    one, _ = tasks(False)
    cfg, seed = one.world_config(tools=False)
    spec = generate(cfg, seed)
    cm, cf = new_canary("MECH"), new_canary("FILING")
    for m in spec.modules:
        if m["name"] == "rig":
            m["config"]["rigs"][0]["mech"]["evaluator_canary"] = cm
    env = KernelExperienceEnv(spec, agents=["agent"], endowment=one.endowment, reward_mode=OBSERVATION_ONLY,
                              experiments=False)
    a = env.agents()[0]
    got = [env.observe(a), env.tools(a)]
    sp = env.execute(a, {"tool": "rig.spec", "args": {}})
    got.append(sp)
    lv = sp["result"]["levers"][0]
    got.append(env.execute(a, {"tool": "rig.hypothesize", "args": {
        "effects": [{"lever": lv["name"], "target": "net", "sign": 1}], "note": cf}}))
    for _ in range(steps):
        got.append(env.execute(a, {"tool": "rig.set", "args": {"settings": {lv["name"]: lv["max"]}}}))
        env.advance(3600)
        got.append(env.execute(a, {"tool": "rig.read", "args": {}}))
        got.append(env.observe(a))
    got.append(env.observe_since(a, 0))
    hits = canary_hits(got, [cm, cf])
    m = {"canaries_planted": 2, "canary_hits": len(hits), "nonzero_native_reward_steps": int(env.reward(a) != 0)}
    return _verdict("hidden mechanism configuration and internal filings stay hidden", m,
                    ok=lambda x: x["canary_hits"] == 0 and x["nonzero_native_reward_steps"] == 0)


def test_verified_terminal(n_steps: int = 200) -> dict:
    """Verified-terminal mode: the numeric env pays nothing before the episode's last tick, at most 1.0 there, and
    never writes a reward into the observation."""
    from ..research.envs import MechVecEnv
    one, _ = tasks(True)
    env = MechVecEnv([one], reward_mode=VERIFIED)
    env.reset()
    rng = np.random.default_rng(1)
    early, slot, final = 0, 0, None
    for _ in range(n_steps):
        o, r, d, _ = env.step(rng.uniform(-1, 1, size=(1, 12)))
        slot += int(o[0, env.spec.scalars.start] != 0)
        if d[0]:
            final = float(r[0])
            break
        early += int(r[0] != 0)
    m = {"nonterminal_reward_steps": early, "reward_slot_nonzero": slot, "terminal_reward": final if final is not None
         else -1.0, "terminal_matches_success": int(final == (1.0 if env.finished[-1]["success"] else 0.0))
         if final is not None else 0}
    return _verdict("verified_terminal_outcome: only the terminal verified outcome", m,
                    ok=lambda x: x["nonterminal_reward_steps"] == 0 and x["reward_slot_nonzero"] == 0
                    and x["terminal_matches_success"] == 1)


def _verdict(name: str, metrics: dict, ok: Callable[[dict], bool] | None = None) -> dict:
    if ok is None:
        def ok(x):
            return (x.get("nonzero_native_reward_steps", 0) == 0 and x.get("privileged_progress_fields", 0) == 0
                    and x.get("legitimate_fields_missing", 0) == 0 and x.get("outcome_terminations", 0) == 0
                    and x.get("verifier_outcomes_recorded", 1) >= 1)
    return {"name": name, "passed": bool(ok(metrics)), "metrics": metrics}


BATTERY = (test_mechvec, test_bridge, test_selfplay, test_kernel_selfplay, test_gym, test_pettingzoo, test_worldenv,
           test_tier1_invariance, test_kernel_invariance, test_rig_hidden_canary, test_verified_terminal)


def run_battery(tests=BATTERY) -> list[dict]:
    out = []
    for fn in tests:
        r = fn()
        r["test"] = fn.__name__
        out.append(r)
    return out


# ====================================================================== legacy (4.0) equivalence
def legacy_check(n_steps: int = 30) -> list[dict]:
    """Run the 4.0 implementation (from `V4_COMMIT`) and the current one in legacy mode with the same inputs.
    Each entry: {name, identical, steps, legacy_nonzero_reward_steps}."""
    out = []
    one, two = tasks(False)
    rng = np.random.default_rng(3)
    seq = [rng.uniform(-1, 1, size=(2, 12)) for _ in range(n_steps)]

    def numeric(name, make_old, make_new, seats=1, n=2):
        res = []
        for mk in (make_old, make_new):
            env = mk()
            rec = [dump(env.reset())]
            for a in seq:
                act = a[:n]
                o, r, d, info = env.step([act] * seats if seats > 1 else act)
                rec.append(dump([o, r, d, info]))
                if np.asarray(d).all():
                    break
            rec.append(dump(getattr(env, "finished", None)))
            res.append(rec)
        out.append({"name": name, "identical": res[0] == res[1], "steps": len(res[0]) - 2})

    old_envs = v4_module("research/envs.py", "crucible.research", "_v4_envs")
    from ..research.envs import MechVecEnv
    numeric("MechVecEnv(reward_mode=legacy)", lambda: old_envs.MechVecEnv([one, one]),
            lambda: MechVecEnv([one, one], reward_mode=LEGACY))
    old_bridge = v4_module("research/bridge.py", "crucible.research", "_v4_bridge")
    from ..research.bridge import KernelMechEnv
    numeric("KernelMechEnv(reward_mode=legacy)", lambda: old_bridge.KernelMechEnv(one),
            lambda: KernelMechEnv(one, reward_mode=LEGACY), n=1)
    old_arena = v4_module("selfplay/arena.py", "crucible.selfplay", "_v4_arena")
    from ..selfplay.arena import KernelSelfPlayEnv, SelfPlayEnv
    numeric("SelfPlayEnv(reward_mode=legacy)", lambda: old_arena.SelfPlayEnv([two]),
            lambda: SelfPlayEnv([two], reward_mode=LEGACY), seats=2, n=1)
    numeric("KernelSelfPlayEnv(reward_mode=legacy)", lambda: old_arena.KernelSelfPlayEnv(two),
            lambda: KernelSelfPlayEnv(two, reward_mode=LEGACY), seats=2, n=1)
    # learners from the 4.0 registry, run on the 4.0 and the current environment
    from ..research import learners
    from ..research.envs import run_episodes
    for name in ("hill_climb", "probe_exploit", "ppo_gru"):
        res = []
        for env_cls in (old_envs.MechVecEnv, lambda ts: MechVecEnv(ts, reward_mode=LEGACY)):
            L = learners.make(name, seed=0)
            recs = run_episodes(env_cls([one, one]), L, explore=True, learn=True)
            L.update()
            res.append(dump([{k: v for k, v in r.items() if k != "wall_s_batch"} for r in recs]))
        out.append({"name": f"learner {name} on MechVecEnv", "identical": res[0] == res[1], "steps": int(one.horizon)})
    # Gymnasium, PettingZoo, WorldEnv
    old_gym = v4_module("api/gym_env.py", "crucible.api", "_v4_gym_env")
    from ..api.gym_env import CrucibleEnv
    from ..goals.contracts import RewardContract
    res = []
    for cls in (old_gym.CrucibleEnv, CrucibleEnv):
        kw = {} if cls is old_gym.CrucibleEnv else {"reward_mode": LEGACY}
        env = cls(spec={"family": "tiny", "overrides": {"n_consumers": 100}}, seed=4, endowment=1_500_000,
                  objective=RewardContract(cash_goal()), dt=3600, max_steps=8, **kw)
        env.server.join_world(env.world_id, "payee", endowment=0)
        w = env.server.world(env.world_id)
        acct = w.modules["bank"].primary_account(w.state, env.principal)
        obs, info = env.reset()
        rec = [dump([obs, info])]
        for k in range(8):
            rec.append(dump(env.step({"tool": "bank.transfer", "args": {"from_account": acct, "to": "p:payee",
                                                                       "amount": 1000 + k, "memo": "x"}})))
        env.close()
        res.append(rec)
    out.append({"name": "Gymnasium CrucibleEnv(reward_mode=legacy)", "identical": res[0] == res[1], "steps": 8})
    old_pz = v4_module("api/pettingzoo_env.py", "crucible.api", "_v4_pettingzoo_env")
    from ..api.pettingzoo_env import CrucibleParallelEnv
    res = []
    for cls in (old_pz.CrucibleParallelEnv, CrucibleParallelEnv):
        kw = {} if cls is old_pz.CrucibleParallelEnv else {"reward_mode": LEGACY}
        env = cls(spec={"family": "tiny", "overrides": {"n_consumers": 100}}, seed=5, agents=["ann", "ben"],
                  endowment=1_500_000, objectives={"ann": RewardContract(cash_goal()),
                                                   "ben": RewardContract(cash_goal())}, dt=3600, max_steps=6, **kw)
        w = env.server.world(env.world_id)
        accts = {a: w.modules["bank"].primary_account(w.state, i.principal) for a, i in env.identities.items()}
        obs, infos = env.reset()
        rec = [dump([obs, infos])]
        for k in range(6):
            acts = {a: {"tool": "bank.transfer", "args": {"from_account": accts[a], "to": "p:ben" if a == "ann"
                                                          else "p:ann", "amount": 700 + k, "memo": "x"}}
                    for a in env.agents}
            rec.append(dump(env.step(acts)))
        env.close()
        res.append(rec)
    out.append({"name": "PettingZoo CrucibleParallelEnv(reward_mode=legacy)", "identical": res[0] == res[1], "steps": 6})
    old_roll = v4_module("scale/rollout.py", "crucible.scale", "_v4_rollout")
    from ..scale.rollout import WorldEnv
    res = []
    for cls in (old_roll.WorldEnv, WorldEnv):
        kw = {} if cls is old_roll.WorldEnv else {"reward_mode": LEGACY}
        env = cls({"family": "tiny", "seed": 2, "overrides": {"n_consumers": 100}}, think_s=3600, **kw)
        env.world.system_call("admin.join", {"handle": "payee", "endowment": 0})
        acct = env.world.modules["bank"].primary_account(env.world.state, env.actor)
        rec = [dump(env.reset())]
        for k in range(6):
            rec.append(dump(env.step({"tool": "bank.transfer", "args": {"from_account": acct, "to": "p:payee",
                                                                       "amount": 500 + k, "memo": "x"}})))
        env.close()
        res.append(rec)
    out.append({"name": "scale.rollout.WorldEnv(reward_mode=legacy)", "identical": res[0] == res[1], "steps": 6})
    return out
