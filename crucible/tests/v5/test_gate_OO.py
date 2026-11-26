"""Gate OO (Clarification 2: observation-only learning as the universal default), workstream L: OO.1-OO.12, OO.14,
OO.16 (OO.13 and OO.15 are workstream P's).

Every test measures the real system and records evidence. A tier counts only where its worlds were really built and
measured in this run: Tier 1 (`make(1, ...)`: 4.0 mechanism families, v5 rapid families), Tier 2 (`make(2, ...)`: a
`crucible.domains.base.Domain`, D-17), Tier 3 (`make(3, profile)`: a `crucible.civ` civilisation). Tier 0 - the 4.0
kernel worlds exposed through `crucible.experience` - is reported separately and never counted as a tier. When a
tier cannot be built (no domain registered, the generator raising), the test records the honest numbers and the
error, then FAILS naming it. Nothing is skipped and no other world type stands in."""
import ast
import importlib.util
import inspect
import json
import os
import resource
import sys
import time

import numpy as np
import pytest

from crucible.acceptance.v5.evidence import record
from crucible.core import evaluator_only as EO
from crucible.experience import make, validate_observation
from crucible.experience.modes import LEGACY, OBSERVATION_ONLY, VERIFIED
from crucible.experience.registry import civ_profiles, domains

pytestmark = pytest.mark.timeout(2400)
CMD = "PYTHONPATH=src python -m pytest tests/v5/test_gate_OO.py"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DAY = 86400
HOUR = 3600
CIV = "balanced"                       # the Tier 3 profile used where one civilisation is enough


def _attempt(fn):
    """One tier's measurement: (result, None), or (None, "Error: message") when its worlds cannot be built or run -
    the tier is then not covered and the test fails after recording."""
    try:
        return fn(), None
    except Exception as e:  # noqa: BLE001 - recorded and asserted on, never swallowed
        return None, f"{type(e).__name__}: {str(e)[:400]}"


def _tier2_names() -> list[str]:
    from crucible.experience.probes_tiers import tier2_names
    return tier2_names()


def _fail_tiers(errors: dict) -> str:
    return "; ".join(f"Tier {t}: {e}" for t, e in sorted(errors.items()))


# ====================================================================== OO.1
def test_OO1_observation_only_is_the_default_everywhere():
    from crucible import rapid
    from crucible.mechanisms.base import MECH_FAMILIES
    from crucible.worldgen.generator import FAMILIES
    import crucible
    crucible.load_all_modules()
    checked, bad, per_tier = [], [], {0: 0, 1: 0, 2: 0, 3: 0}

    def check(label, tier, env, agent, steps=2):
        ok = env.reward_mode == OBSERVATION_ONLY
        w = getattr(env, "world", None)
        if w is not None:
            ok = ok and EO.hidden(w) and not ({t["name"] for t in env.tools(agent)} & set(EO.TOOLS))
        for _ in range(steps):
            env.advance(int(getattr(env, "tick_s", 3600)))
            ok = ok and env.reward(agent) == 0.0
        checked.append(label)
        per_tier[tier] += 1
        if not ok:
            bad.append(label)
    for fam in sorted(rapid.REGISTRY):
        env = make(1, f"rapid:{fam}", 0)
        check(f"tier1 rapid:{fam}", 1, env, env.agents()[0])
    for fam in sorted(MECH_FAMILIES):
        env = make(1, f"mech:{fam}", 0, validate=False)
        check(f"tier1 mech:{fam}", 1, env, env.agents()[0])
    errors = {}

    def tier2():
        for name in _tier2_names():
            for cfg in domains()[name.split(":")[0]].configs:
                env = make(2, f"{name.split(':')[0]}:{cfg.name}", 0)
                check(f"tier2 {name.split(':')[0]}:{cfg.name}", 2, env, env.agents()[0])

    def tier3():
        profs = civ_profiles()
        if not profs:
            raise LookupError("crucible.civ.PROFILES is empty")
        for prof in profs:
            env = make(3, prof, 0)
            check(f"tier3 {prof}", 3, env, env.agents()[0])
    for t, fn in ((2, tier2), (3, tier3)):
        _, err = _attempt(fn)
        if err:
            errors[t] = err
    for fam in sorted(FAMILIES):
        if fam.startswith("d5_"):
            continue                                       # Tier 2 domain families are checked above
        env = make(0, fam, 0, agents=["learner"])
        check(f"4.0 kernel world {fam}", 0, env, env.agents()[0], steps=1)
    # every constructor and wrapper, default-constructed
    from crucible.api.gym_env import CrucibleEnv
    from crucible.api.pettingzoo_env import CrucibleParallelEnv
    from crucible.experience.gym import ExperienceGymEnv, ExperienceParallelEnv
    from crucible.experience.kernel import KernelExperienceEnv
    from crucible.experience.reference import ReferenceLearner
    from crucible.experience.tier1 import MechExperienceEnv, RapidExperienceEnv
    from crucible.research.bridge import KernelMechEnv
    from crucible.research.envs import MechVecEnv
    from crucible.research.tasks import make_task
    from crucible.scale.rollout import WorldEnv
    from crucible.selfplay.arena import KernelSelfPlayEnv, SelfPlayEnv
    from crucible.selfplay.league import SelfPlayLeague
    import inspect
    from crucible.research import protocols as P
    from crucible.research.learners import training
    from crucible.selfplay import modes as SM
    t1, t2 = make_task("scm_additive", 3, validate=False), make_task("abm_duopoly", 2, validate=False)
    spec = {"family": "tiny", "overrides": {"n_consumers": 40}}
    ctors = {
        "MechVecEnv": lambda: MechVecEnv([t1]), "KernelMechEnv": lambda: KernelMechEnv(t1),
        "SelfPlayEnv": lambda: SelfPlayEnv([t2]), "KernelSelfPlayEnv": lambda: KernelSelfPlayEnv(t2),
        "MechExperienceEnv": lambda: MechExperienceEnv(t1), "RapidExperienceEnv": lambda: RapidExperienceEnv(
            "net_routing", 0), "KernelExperienceEnv": lambda: KernelExperienceEnv(spec, 0, agents=["learner"]),
        "CrucibleEnv": lambda: CrucibleEnv(spec=spec, seed=0), "CrucibleParallelEnv": lambda: CrucibleParallelEnv(
            spec=spec, seed=0), "WorldEnv": lambda: WorldEnv({"family": "tiny", "seed": 0, "overrides":
                                                              {"n_consumers": 40}}),
        "ExperienceGymEnv": lambda: ExperienceGymEnv(1, "rapid:net_routing", 0),
        "ExperienceParallelEnv": lambda: ExperienceParallelEnv(1, "rapid:team_commons", 0),
        "ExperienceGymEnv tier3": lambda: ExperienceGymEnv(3, CIV, 0),
        "civ.experience.make": lambda: __import__("crucible.civ.experience", fromlist=["make"]).make(CIV, 0),
        "ReferenceLearner": lambda: ReferenceLearner(),
        "SelfPlayLeague": lambda: SelfPlayLeague(os.path.join(os.environ.get("TMPDIR", "/tmp"), "oo1-league"), t2),
    }
    ctor_bad = []
    for name, mk in ctors.items():
        obj, err = _attempt(mk)
        if err:
            ctor_bad.append(f"{name}: {err}")
            continue
        mode = getattr(obj, "reward_mode", None)
        w = getattr(obj, "world", None)
        w = w if hasattr(w, "spec") and hasattr(w, "runtime") else None        # a kernel World, not a family name
        if mode != OBSERVATION_ONLY or (w is not None and not EO.hidden(w)):
            ctor_bad.append(name)
        if hasattr(obj, "close") and name not in ("ReferenceLearner", "SelfPlayLeague"):
            obj.close()
    # entry points whose default is a parameter (research protocols, self-play harness, training loop)
    for fn in (P.evaluate, P.lifetime, P.timescales, P.internalisation, P.transfer, P.compare, P.evaluate_many,
               P.kernel_diagnosis, training.evaluate, training.train, SM.evaluate, SM.run_pair, SM.run_population,
               SM.run_historical, SM.run_cross_play, SM.best_response, SM.cross_play):
        d = inspect.signature(fn).parameters.get("reward_mode")
        if d is None or d.default is not None:
            ctor_bad.append(f"{fn.__module__}.{fn.__name__}")
    from crucible.api import http_server, mcp_server
    for mod in (http_server, mcp_server):
        if "default=\"observation_only\"" not in inspect.getsource(mod.main):
            ctor_bad.append(f"{mod.__name__} CLI")
    tiers = sum(1 for t in (1, 2, 3) if per_tier[t] > 0)
    metrics = {"tiers_covered": tiers, "env_families_checked": len(checked),
               "families_not_observation_only_by_default": len(bad), "tier1_families": per_tier[1],
               "tier2_families": per_tier[2], "tier3_profiles": per_tier[3], "kernel40_families": per_tier[0],
               "constructors_checked": len(ctors) + 17 + 2, "constructors_not_observation_only_by_default":
               len(ctor_bad)}
    record("OO.1", metrics, command=CMD, notes="default-constructed (no reward_mode named) through make() for every "
           "rapid family, every 4.0 mechanism family, every v5 domain configuration and civilisation profile present, "
           "and every 4.0 world family exposed through crucible.experience (tier 0, not counted as a tier): "
           "reward_mode == observation_only, the world hides evaluator tools/fields (core.evaluator_only), reward 0.0. "
           f"Also every constructor/wrapper and harness entry point. Bad: {bad[:10]} {ctor_bad[:10]}. "
           f"Tiers not measured: {_fail_tiers(errors)}")
    assert not bad and not ctor_bad, (bad, ctor_bad)
    assert metrics["env_families_checked"] >= 100
    assert tiers == 3, f"observation-only measured on {tiers} tier(s); " + _fail_tiers(errors)


# ====================================================================== OO.2
def test_OO2_no_native_or_privileged_reward_reaches_the_learner():
    from crucible.experience import probes
    res = probes.run_all(errors="record")
    pc = probes.positive_control()
    m = dict(res["metrics"])
    m["positive_control_leak_kinds"] = len(pc["kinds_with_leaks"])
    m["positive_control_leaks"] = pc["n_leaks"]
    path = os.path.join(os.environ.get("TMPDIR", "/tmp"), "OO2_probes.json")
    with open(path, "w") as fh:
        json.dump({"run": res, "positive_control": pc}, fh, indent=1, default=str)
    record("OO.2", m, command=CMD, notes=f"information-flow probes ({', '.join(res['kinds_checked'])}) over "
           f"{m['surfaces_probed']} surfaces: {[s['surface'] for s in res['surfaces']]}; per kind: {res['by_kind']}; "
           f"errors: {res['errors']}; positive control (the same probes on the 4.0 contract, legacy mode) found "
           f"{pc['n_leaks']} leaks of kinds {pc['kinds_with_leaks']}. Leaks: "
           f"{[l for s in res['surfaces'] for l in s['leaks']][:5]}")
    assert m["reward_leak_probe_kinds"] >= 8 and m["reward_leaks_found"] == 0, (res["errors"], [
        l for s in res["surfaces"] for l in s["leaks"]][:10])
    assert m["nonzero_scalar_rewards_default_mode"] == 0
    assert m["positive_control_leak_kinds"] >= 3            # the probes are sensitive


# ====================================================================== OO.3
def _consequences(env) -> dict:
    """Seven kinds of consequence, each read from the participant's OWN observations in a kernel world (any tier):
    a builder deploys a service a client pays for, a neighbour replies to the builder's post, the builder's own
    experiment resolves days later, a failed transfer comes back as a typed error."""
    from crucible.experience.scripted import call, deploy_service, discover, need_input
    b, c, n = env.agents()[:3]
    deploy_service(env, b, handle="obsco", host="obssvc", price=4)
    discover(env, c, "obssvc")
    post = call(env, b, "forum.post", board="general", title="obssvc is live", body="sorting, 4 cents a call")
    reg = env.execute(b, {"tool": "exp.register", "args": {
        "hypothesis": "my balance rises within 3 days", "measure": {"tool": "bank.accounts", "where": {"owner": "$self"},
                                                                    "field": "balance", "aggregate": "sum"},
        "prediction": {"op": "increase", "min_change": 1}, "deadline_in_s": 3 * DAY}})
    assert reg["ok"], reg
    env.observe(b)
    for d in range(4):
        r = env.execute(c, {"tool": "obssvc.solve", "args": need_input("sort.records", d)})
        assert r["ok"], r
        env.execute(n, {"tool": "forum.reply", "args": {"post": post["post"], "body": f"tried it, day {d}"}})
        env.advance(DAY)
    fail = env.execute(b, {"tool": "bank.transfer", "args": {"from_account": "acct_nope", "to": c, "amount": 1,
                                                             "memo": "x"}})
    obs = env.observe(b)
    evs = env.observe_since(b, 0)[0]
    status = call(env, b, "cloud.status", host="obssvc", days=7)
    types = {e["type"] for e in evs}
    return {"transactions": any(t.startswith("bank.") for t in types),
            "customer_behaviour": sum(d.get("requests", 0) for d in status["days"]) >= 4,
            "technical_measurements": any(k in json.dumps(status) for k in ("cpu", "latency", "errors")),
            "social_responses": any(e["type"].startswith("forum.") and e["actor"] == n for e in evs),
            "resource_changes": obs["resources"].get("balance_cents") is not None and
            any(k.startswith("bank.balances.") for k in obs["numbers"]),
            "failures": not fail["ok"] and isinstance(fail["error"], dict),
            "delayed_outcomes": any(e["type"] == "exp.resolved" and (e["cause"] or {}).get("action_seq") ==
                                    reg["action_seq"] for e in evs)}


def test_OO3_real_observable_consequences_remain():
    per, errors = {}, {}
    # tier 0 (4.0 kernel world, reported separately)
    per["tier0 tiny"] = _consequences(make(0, "tiny", 3, agents=["builder", "client", "neighbour"],
                                           endowment=3_000_000, overrides={"n_consumers": 200}))

    def tier1():
        t1 = make(1, "rapid:net_routing", 3)
        a = t1.agents()[0]
        t1.advance(t1.tick_s)
        o1 = t1.observe(a)
        m1 = make(1, "mech:scm_additive", 3, validate=False)
        am = m1.agents()[0]
        r = m1.execute(am, {"tool": "rig.set", "args": {"settings": {}}})
        m1.advance(m1.tick_s)
        om = m1.observe(am)
        bad = m1.execute(am, {"tool": "rig.set", "args": {"settings": {"no_such_lever": 1.0}}})
        return {"readings": any(k.startswith("rig.seats.r1.readings.") for k in om["numbers"]) and bool(o1["numbers"]),
                "revenue_and_cost": "rig.seats.r1.revenue_cents" in om["numbers"] and
                "rig.seats.r1.cost_cents" in om["numbers"],
                "resource_changes": "cash_cents" in o1["resources"] and "balance_cents" in om["resources"],
                "failures": not bad["ok"] and isinstance(bad["error"], dict),
                "action_results": r["ok"] and om["last_result"] is not None}

    def tier2():
        out = {}
        for name in _tier2_names():
            out[name] = _consequences(make(2, name, 3, agents=["builder", "client", "neighbour"],
                                           endowment=3_000_000))
        return out

    def tier3():
        return _consequences(make(3, CIV, 3, agents=["builder", "client", "neighbour"], endowment=3_000_000))
    for t, fn in ((1, tier1), (2, tier2), (3, tier3)):
        r, err = _attempt(fn)
        if err:
            errors[t] = err
        elif t == 2:
            per.update({f"tier2 {k}": v for k, v in r.items()})
        else:
            per[f"tier{t}"] = r
    kernel = {k: v for k, v in per.items() if not k.startswith("tier1")}
    covered = [t for t in (1, 2, 3) if any(k.startswith(f"tier{t}") and all(v.values()) for k, v in per.items())]
    kinds = min((sum(1 for x in v.values() if x) for v in kernel.values()), default=0)
    metrics = {"consequence_kinds_observable": kinds, "tiers_covered": len(covered),
               "kernel40_consequence_kinds": sum(1 for x in per["tier0 tiny"].values() if x),
               "tier1_consequence_kinds": sum(1 for x in per.get("tier1", {}).values() if x),
               "tier3_consequence_kinds": sum(1 for x in per.get("tier3", {}).values() if x)}
    record("OO.3", metrics, command=CMD, notes=f"consequence kinds read from the participant's own observations, per "
           f"world (all must be present for a tier to count): {per}; consequence_kinds_observable = the fewest kinds "
           f"of any kernel world measured. Tiers not measured: {_fail_tiers(errors)}")
    assert metrics["consequence_kinds_observable"] >= 7, per
    assert len(covered) == 3, f"consequences observable in tiers {covered}; " + _fail_tiers(errors)


# ====================================================================== OO.4
def _verify_goal(env) -> bool:
    """The GoalVerifier's verdict against a direct evaluation of the predicate's metric (evaluator side)."""
    from crucible.goals.contracts import GoalContract
    from crucible.goals.metrics import evaluate_metric
    a = env.agents()[0]
    goal = GoalContract("cash", "Hold at least $20,000.", {"family": "probe", "seed": 5}, "$self",
                        {"metric": "cash", "args": {"principal": "$self"}, "op": ">=", "value": 2_000_000},
                        {"metric": "cash", "args": {"principal": "$self"}, "target": 2_000_000}, 30 * DAY)
    env.set_objective(a, goal)
    env.advance(HOUR)
    st = env.verify()["agents"][a]
    cash = evaluate_metric(env.world, "cash", {"principal": a})
    tool_reach = {t["name"] for t in env.tools(a)} & set(EO.TOOLS)
    return abs(st["progress"] - min(1.0, cash / 2_000_000)) < 1e-9 and not tool_reach


def _hosted(ref: dict) -> tuple[bool, bool, int]:
    """A world in its own process: (verifier matches the world's own record, verifier reachable from the participant
    handle, participant requests refused)."""
    from crucible.experience import host
    part, ev = host(ref, timeout=900)
    try:
        a = part.agents()[0]
        for _ in range(3):
            part.advance(int(part.tick_s) if ref["tier"] == 1 else HOUR)
        refused = [part._raw(op)[0] == "error" for op in ("verify", "digest", "describe", "env", "world", "_obs")]
        reach = (not all(refused)) or any(hasattr(part, x) for x in ("verify", "env", "world"))
        v = ev.verify()
        if ref["tier"] == 1:
            o = v["outcome"]
            match = abs(o["value"] - (o["revenue"] - o["cost"])) < 1e-6 if "revenue" in o else o["value"] is not None
        else:
            seen = part.observe(a)["resources"]["balance_cents"]
            match = v["measures"][a]["cash_cents"] == float(seen) and v["invariant_violations"] == []
    finally:
        ev.close()
    return bool(match), bool(reach), sum(refused)


def test_OO4_independent_verification_through_a_separate_interface():
    per, errors, refused, reach_any = {}, {}, 0, False

    def tier1():
        m, r, n = _hosted({"tier": 1, "name": "rapid:net_routing", "seed": 3})
        me = make(1, "mech:scm_additive", 3, validate=True)
        while not me.done:
            me.advance(me.tick_s)
        rec = me.verify()["episodes"][-1]
        m2 = abs(rec["score"] - me.task.score(rec["net_per_tick"])) < 1e-9 and \
            rec["success"] == me.task.success(rec["net_cents"])
        return m and m2, r, n

    def tier2():
        out = []
        for name in _tier2_names():
            m, r, n = _hosted({"tier": 2, "name": name, "seed": 3})
            out.append((m and _verify_goal(make(2, name, 5)), r, n))
        return all(x[0] for x in out), any(x[1] for x in out), sum(x[2] for x in out)

    def tier3():
        m, r, n = _hosted({"tier": 3, "name": CIV, "seed": 3, "kw": {"agents": ["learner"]}})
        return m and _verify_goal(make(3, CIV, 5, agents=["learner"], endowment=2_500_000)), r, n
    tier0 = _verify_goal(make(0, "tiny", 5, agents=["learner"], endowment=2_500_000))
    for t, fn in ((1, tier1), (2, tier2), (3, tier3)):
        r, err = _attempt(fn)
        if err:
            errors[t] = err
            continue
        per[t] = r[0]
        reach_any |= r[1]
        refused += r[2]
    covered = [t for t, ok in per.items() if ok]
    metrics = {"tiers_covered": len(covered), "verifier_matches_ground_truth": bool(per) and all(per.values()) and
               tier0, "verifier_reachable_from_learner": bool(reach_any), "participant_requests_refused": refused,
               "verifications_compared": len(per) + 1}
    record("OO.4", metrics, command=CMD, notes="per tier, the world hosted in its own process "
           "(crucible.experience.remote): the participant pipe refuses verify/digest/describe/attribute access; the "
           "evaluator's verdict is recomputed independently (rapid outcome from its ledger; mechanism score from the "
           "task's private reference; kernel tiers: the evaluator's cash measure equals the balance the participant "
           f"observes, and the GoalVerifier's progress equals a direct evaluation of its metric). Per tier: {per}; "
           f"tier 0: {tier0}. Tiers not measured: {_fail_tiers(errors)}")
    assert metrics["verifier_matches_ground_truth"] and not metrics["verifier_reachable_from_learner"], metrics
    assert len(covered) == 3, f"verification measured in tiers {covered}; " + _fail_tiers(errors)


# ====================================================================== OO.5
def _rss_mb() -> float:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _objective_free(env, days: int, limits: dict, label: str, violations: list, *, act=None) -> dict:
    """Run a kernel-world environment without any objective for `days` simulated days; check limits every day."""
    a = env.agents()[0]
    t0, r0 = time.perf_counter(), _rss_mb()
    peak = r0
    for d in range(days):
        if act is not None:
            act(env, a, d)
        env.advance(DAY)
        if d % 5 == 4 or d == days - 1:
            obs = env.observe(a)
            if validate_observation(obs):
                violations.append(f"{label} day {d}: invalid observation")
        if len(env.world.log) > limits["max_event_log_in_memory"]:
            violations.append(f"{label} day {d}: {len(env.world.log)} events in memory")
        if env.reward(a) != 0.0:
            violations.append(f"{label} day {d}: nonzero reward")
        peak = max(peak, _rss_mb())
    wall = time.perf_counter() - t0
    if wall > limits["max_wall_s"]:
        violations.append(f"{label}: wall {wall:.0f}s > {limits['max_wall_s']}")
    if peak - r0 > limits["max_rss_growth_mb"]:
        violations.append(f"{label}: rss growth {peak - r0:.0f} MB > {limits['max_rss_growth_mb']}")
    v = env.verify()
    if v["invariant_violations"]:
        violations.append(f"{label}: invariant violations {v['invariant_violations'][:3]}")
    return {"days": int(env.now // DAY), "wall_s": round(wall, 1), "rss_growth_mb": round(peak - r0, 1),
            "events": int(env.world.seq), "log_in_memory": len(env.world.log)}


@pytest.mark.slow
@pytest.mark.timeout(7200)
def test_OO5_operation_without_an_objective_within_resource_limits():
    # configured limits per run (stated before running): wall time, resident-memory growth, in-memory event log
    limits = {"tier1": {"max_rss_growth_mb": 600.0, "max_wall_s": 900.0},
              "kernel": {"max_rss_growth_mb": 600.0, "max_wall_s": 900.0, "max_event_log_in_memory": 400_000},
              "civ": {"max_rss_growth_mb": 6000.0, "max_wall_s": 5400.0, "max_event_log_in_memory": 400_000}}
    violations, errors, runs = [], {}, {}
    # Tier 1: objective-free operation across many episodes (rapid and mechanism worlds reset at their horizon)

    def tier1():
        t0, r0 = time.perf_counter(), _rss_mb()
        eps = {}
        for name in ("rapid:net_routing", "rapid:team_commons", "mech:scm_additive"):
            env = make(1, name, 1, **({"validate": False} if name.startswith("mech") else {}))
            a = env.agents()[0]
            n = 0
            for _ in range(600):
                env.advance(env.tick_s)
                if env.reward(a) != 0.0:
                    violations.append(f"{name}: nonzero reward")
                if env.done:
                    env.reset()
                    n += 1
            eps[name] = n
        wall, growth = time.perf_counter() - t0, _rss_mb() - r0
        if wall > limits["tier1"]["max_wall_s"] or growth > limits["tier1"]["max_rss_growth_mb"]:
            violations.append(f"tier1: wall {wall:.0f}s, rss growth {growth:.0f} MB")
        return {"episodes": eps, "wall_s": round(wall, 1)}

    def post(env, a, d):
        env.execute(a, {"tool": "forum.post", "args": {"board": "general", "title": f"day {d}", "body": "notes"}})
    runs["tier0 tiny"] = _objective_free(make(0, "tiny", 2, agents=["learner"], overrides={
        "n_consumers": 100, "params": {"log_retain": 200_000}}), 90, limits["kernel"], "tier0", violations, act=post)

    def tier2():
        return {n: _objective_free(make(2, n, 2), 90, limits["kernel"], f"tier2 {n}", violations, act=post)
                for n in _tier2_names()}

    def tier3():
        return _objective_free(make(3, CIV, 2, agents=["learner"]), 90, limits["civ"], "tier3", violations,
                               act=post)
    for t, fn in ((1, tier1), (2, tier2), (3, tier3)):
        r, err = _attempt(fn)
        if err:
            errors[t] = err
        else:
            runs[f"tier{t}"] = r
    civ_days = int((runs.get("tier3") or {}).get("days", 0))
    t2_ok = "tier2" in runs and all(x["days"] >= 90 for x in runs["tier2"].values())
    tiers = sum([("tier1" in runs), t2_ok, civ_days >= 90])
    metrics = {"objectiveless_tiers": tiers, "objectiveless_civ_sim_days": civ_days,
               "resource_limit_violations": len(violations),
               "objectiveless_kernel40_sim_days": runs["tier0 tiny"]["days"],
               "tier1_episodes_run": sum((runs.get("tier1") or {}).get("episodes", {}).values()),
               "civ_wall_s": float((runs.get("tier3") or {}).get("wall_s", 0.0)),
               "civ_rss_growth_mb": float((runs.get("tier3") or {}).get("rss_growth_mb", 0.0))}
    record("OO.5", metrics, command=CMD, notes=f"limits {limits}; runs {runs}; violations {violations}. Tiers not "
           f"measured: {_fail_tiers(errors)}")
    assert metrics["resource_limit_violations"] == 0, violations
    assert tiers == 3 and civ_days >= 90, f"objective-free operation in {tiers} tier(s); " + _fail_tiers(errors)


# ====================================================================== OO.6
LEARNER = os.path.join(ROOT, "examples", "external_learner", "curiosity_learner.py")
ALLOWED_TOP = set(getattr(sys, "stdlib_module_names", ())) | {"__future__", "numpy"}


def _imports(path: str) -> list[str]:
    import crucible.experience as X
    bad = []
    for node in ast.walk(ast.parse(open(path).read())):
        if isinstance(node, ast.Import):
            bad += [a.name for a in node.names if a.name.split(".")[0] not in ALLOWED_TOP]
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod.split(".")[0] == "crucible":
                if mod != "crucible.experience" or any(a.name not in X.__all__ for a in node.names):
                    bad.append(mod)
            elif mod.split(".")[0] not in ALLOWED_TOP:
                bad.append(mod)
    return bad


def test_OO6_an_external_learner_supplies_its_own_rewards():
    import hashlib
    src = os.path.join(ROOT, "src", "crucible")

    def tree():
        h = hashlib.sha256()
        for d, _, fs in sorted(os.walk(src)):
            for f in sorted(fs):
                if f.endswith(".py"):
                    h.update(open(os.path.join(d, f), "rb").read())
        return h.hexdigest()
    before = tree()
    spec = importlib.util.spec_from_file_location("curiosity_learner", LEARNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    runs, errors = {}, {}

    def go(tier, name, **kw):
        env = make(tier, name, 3, **kw)
        L = mod.CuriosityLearner(seed=1)
        return mod.run(env, L, 100, update_every=20, think_s=None if tier == 1 else HOUR)
    runs["tier0 tiny"] = go(0, "tiny", agents=["learner"])
    for t, fn in ((1, lambda: {"rapid:net_routing": go(1, "rapid:net_routing"),
                               "mech:scm_additive": go(1, "mech:scm_additive", validate=False)}),
                  (2, lambda: {n: go(2, n) for n in _tier2_names()}),
                  (3, lambda: {f"civ:{CIV}": go(3, CIV, agents=["learner"])})):
        r, err = _attempt(fn)
        if err:
            errors[t] = err
        else:
            runs.update({f"tier{t} {k}": v for k, v in r.items()})
    covered = sorted({int(k[4]) for k, v in runs.items() if k[4] in "123" and v["param_updates_changed"] >= 1})
    updates = sum(r["param_updates_changed"] for r in runs.values())
    env_paid = sum(r["env_reward_sum"] for r in runs.values())
    own = {n: round(r["own_reward_sum"], 4) for n, r in runs.items()}
    changes = int(tree() != before) + len(_imports(LEARNER))
    metrics = {"tiers_covered": len(covered), "external_reward_param_updates": updates, "env_changes_required": changes,
               "env_reward_paid": env_paid, "distinct_weight_hashes": sum(len(set(r["hashes"])) for r in runs.values())}
    record("OO.6", metrics, command=CMD, notes="examples/external_learner/curiosity_learner.py (numpy only, public "
           f"crucible.experience names only: violations {_imports(LEARNER)}) computes its own prediction-error "
           f"curiosity reward {own}; the environment paid {env_paid}; updates that changed its weight hash: "
           f"{ {n: r['param_updates_changed'] for n, r in runs.items()} }; src tree unchanged: {changes == 0}. "
           f"Tiers not measured: {_fail_tiers(errors)}")
    assert updates >= 3 and changes == 0 and env_paid == 0.0, metrics
    assert len(covered) == 3, f"external learner updated in tiers {covered}; " + _fail_tiers(errors)


# ====================================================================== OO.7
def test_OO7_delayed_outcomes_linked_to_their_actions():
    env = make(0, "tiny", 6, agents=["learner"], extra_modules=["wf"])
    a = env.agents()[0]
    reg = env.execute(a, {"tool": "exp.register", "args": {
        "hypothesis": "in 31 days my balance is lower than now", "plan": "pay a supplier weekly",
        "measure": {"tool": "bank.accounts", "where": {"owner": "$self"}, "field": "balance", "aggregate": "sum"},
        "prediction": {"op": "decrease", "min_change": 0}, "deadline_in_s": 31 * DAY}})
    flow = env.execute(a, {"tool": "wf.define", "args": {"name": "books", "steps": [
        {"id": "b", "tool": "bank.accounts", "args": {}}]}})
    later = env.execute(a, {"tool": "wf.run_later", "args": {"flow": flow["result"]["flow"], "in_s": 40 * DAY}})
    assert reg["ok"] and later["ok"], (reg, later)
    env.advance(45 * DAY)                                     # the learner does not look for 45 days
    evs = env.observe_since(a, 0)[0]
    mine = {r["action_seq"]: r["tool"] for r in (reg, later)}
    delayed = [e for e in evs if e["cause"] and e["cause"]["action_seq"] in mine and e["cause"]["delay_s"] > 0]
    linked = all(mine[e["cause"]["action_seq"]] == e["cause"]["tool"] for e in delayed)
    rapid = make(1, "rapid:net_routing", 3)
    ra = rapid.agents()[0]
    r1 = rapid.execute(ra, {"tool": "rapid.levers", "args": {"values": {}}})
    for _ in range(5):
        rapid.advance(rapid.tick_s)
    tier1 = [e for e in rapid.observe_since(ra, 0)[0] if e["cause"] and e["cause"]["action_seq"] == r1["action_seq"]]
    metrics = {"delayed_outcomes_retrieved": len(delayed), "max_retrieved_delay_sim_days": round(
        max((e["cause"]["delay_s"] for e in delayed), default=0) / DAY, 3), "provenance_links_to_action": bool(
        delayed) and linked, "tier1_consequences_linked": len(tier1)}
    record("OO.7", metrics, command=CMD, notes=f"4.0 kernel world: exp.register (31-day deadline) and wf.run_later "
           f"(40 days) retrieved with observe_since after 45 unobserved days; types "
           f"{sorted({e['type'] for e in delayed})}, via {sorted({e['cause']['via'] for e in delayed})}; Tier 1 rapid: "
           f"{len(tier1)} later events name the levers action (in_force / direct)")
    assert metrics["delayed_outcomes_retrieved"] >= 1 and metrics["max_retrieved_delay_sim_days"] >= 30
    assert metrics["provenance_links_to_action"] and tier1


# ====================================================================== OO.8
def test_OO8_historical_reward_experiments_reproducible_by_explicit_selection():
    from crucible.experience import leakage
    out = leakage.legacy_check()
    reproduced = [o for o in out if o["identical"]]
    from crucible.research.envs import MechVecEnv
    from crucible.research.tasks import make_task
    t = make_task("scm_additive", 3, validate=False)
    default, legacy = MechVecEnv([t]), MechVecEnv([t], reward_mode=LEGACY)
    default.reset(), legacy.reset()
    a = np.full((1, 12), 0.3)
    _, rd, _, _ = default.step(a)
    _, rl, _, _ = legacy.step(a)
    explicit = bool(rd[0] == 0.0 and rl[0] != 0.0 and default.reward_mode == OBSERVATION_ONLY)
    # every current-side environment of the comparison names the legacy mode (the learner runs use MechVecEnv with
    # reward_mode=LEGACY too; leakage.legacy_check builds them that way)
    explicit = explicit and all("legacy" in o["name"] for o in out if not o["name"].startswith("learner"))
    metrics = {"legacy_results_reproduced": len(reproduced), "legacy_runs": len(out),
               "legacy_mode_requires_explicit_selection": explicit}
    record("OO.8", metrics, command=CMD, notes="the 4.0 implementations (git c07098b) next to the current ones with "
           "reward_mode='legacy_shaped_reward' named: byte-identical observations/rewards/infos for "
           + "; ".join(f"{o['name']}={o['identical']}" for o in out) +
           ". The default-constructed environment pays 0.0 where the explicitly legacy one pays the shaped reward.")
    assert len(reproduced) == len(out) and len(reproduced) >= 2 and explicit, out


# ====================================================================== OO.9
def test_OO9_every_public_entry_point_defaults_to_observation_only(tmp_path):
    """Each public entry point constructed or called with NOTHING selected; then each explicit selection is checked
    to be recorded (the environment reports it, experience records carry it)."""
    from crucible.api import http_server, mcp_server
    from crucible.api.gym_env import CrucibleEnv
    from crucible.api.pettingzoo_env import CrucibleParallelEnv
    from crucible.experience import ExperienceStore
    from crucible.experience.__main__ import main as cli_main
    from crucible.experience.gym import ExperienceGymEnv, ExperienceParallelEnv
    from crucible.experience.loop import run_learner
    from crucible.experience.reference import ReferenceLearner
    from crucible.research import protocols as P
    from crucible.research.bridge import KernelMechEnv
    from crucible.research.envs import MechVecEnv
    from crucible.research.learners import training
    from crucible.research.tasks import make_task
    from crucible.scale.rollout import WorldEnv
    from crucible.selfplay import modes as SM
    from crucible.selfplay.arena import KernelSelfPlayEnv, SelfPlayEnv
    t1, t2 = make_task("scm_additive", 3, validate=False), make_task("abm_duopoly", 2, validate=False)
    spec = {"family": "tiny", "overrides": {"n_consumers": 40}}
    checked, bad, errors = [], [], {}

    def obj_ok(name, obj):
        mode = getattr(obj, "reward_mode", None)
        w = getattr(obj, "world", None)
        w = w if hasattr(w, "spec") and hasattr(w, "runtime") else None
        checked.append(name)
        if mode != OBSERVATION_ONLY or (w is not None and not EO.hidden(w)):
            bad.append(f"{name}: {mode}")
    entries = {
        "experience.registry.make tier1 rapid": lambda: make(1, "rapid:net_routing", 0),
        "experience.registry.make tier1 mech": lambda: make(1, "mech:scm_additive", 0, validate=False),
        "experience.registry.make tier0": lambda: make(0, "tiny", 0, agents=["learner"]),
        "api.gym_env.CrucibleEnv": lambda: CrucibleEnv(spec=spec, seed=0),
        "api.pettingzoo_env.CrucibleParallelEnv": lambda: CrucibleParallelEnv(spec=spec, seed=0),
        "research.envs.MechVecEnv": lambda: MechVecEnv([t1]), "research.bridge.KernelMechEnv": lambda: KernelMechEnv(t1),
        "selfplay.arena.SelfPlayEnv": lambda: SelfPlayEnv([t2]),
        "selfplay.arena.KernelSelfPlayEnv": lambda: KernelSelfPlayEnv(t2),
        "scale.rollout.WorldEnv": lambda: WorldEnv({"family": "tiny", "seed": 0, "overrides": {"n_consumers": 40}}),
        "experience.reference.ReferenceLearner": lambda: ReferenceLearner(),
        "experience.gym.ExperienceGymEnv": lambda: ExperienceGymEnv(1, "rapid:net_routing", 0),
        "experience.gym.ExperienceParallelEnv": lambda: ExperienceParallelEnv(1, "rapid:team_commons", 0),
        "api.mcp_server.MCPHandler": lambda: mcp_server.MCPHandler(),
    }
    for name, mk in entries.items():
        o, err = _attempt(mk)
        if err:
            bad.append(f"{name}: {err}")
            checked.append(name)
        else:
            obj_ok(name, o)
    for name, mk in (("experience.registry.make tier2", lambda: make(2, _tier2_names()[0], 0)),
                     ("experience.registry.make tier3", lambda: make(3, CIV, 0)),
                     ("civ.experience.make", lambda: __import__("crucible.civ.experience", fromlist=["make"]).make(
                         CIV, 0))):
        o, err = _attempt(mk)
        if err:
            errors[name] = err
        else:
            obj_ok(name, o)
    # parameter defaults of the harness entry points (research protocols and their positive controls, self-play,
    # training): the default is None = observation_only (check_mode(None))
    fns = [P.evaluate, P.lifetime, P.timescales, P.internalisation, P.transfer, P.compare, P.evaluate_many,
           P.kernel_diagnosis, training.evaluate, training.train, SM.evaluate, SM.run_pair, SM.run_population,
           SM.run_historical, SM.run_cross_play, SM.best_response, SM.cross_play]
    fns += [getattr(P, n) for n in dir(P) if "control" in n.lower() and callable(getattr(P, n))]
    for fn in fns:
        d = inspect.signature(fn).parameters.get("reward_mode")
        checked.append(f"{fn.__module__}.{fn.__name__}")
        if d is None or d.default is not None:
            bad.append(f"{fn.__module__}.{fn.__name__}: default {None if d is None else d.default!r}")
    # the CLIs: argparse defaults
    for name, mod in (("crucible.api.http_server CLI", http_server), ("crucible.api.mcp_server CLI", mcp_server)):
        checked.append(name)
        if 'default="observation_only"' not in inspect.getsource(mod.main):
            bad.append(name)
    out = tmp_path / "cli"
    rc = cli_main(["run", "--tier", "1", "--name", "rapid:net_routing", "--steps", "6", "--out", str(out)])
    checked.append("python -m crucible.experience run")
    cli_records = [json.loads(x) for f in sorted(out.rglob("*.jsonl")) for x in open(f) if x.strip()]
    if rc not in (0, None) or not cli_records or any(r.get("reward_mode") != OBSERVATION_ONLY for r in cli_records):
        bad.append(f"experience CLI: rc={rc}, records={len(cli_records)}")
    # explicit selections are recorded: the environment reports the mode and every stored record carries it
    unrecorded = 0
    for mode in (LEGACY, VERIFIED, OBSERVATION_ONLY):
        env = make(1, "rapid:net_routing", 1, mode)
        store = ExperienceStore(str(tmp_path / f"store-{mode}"))
        run_learner(env, ReferenceLearner(seed=0, reward_mode=env.reward_mode), 12, store=store, update_every=50)
        store.close()
        recs = list(ExperienceStore(str(tmp_path / f"store-{mode}")).iterate())
        unrecorded += int(env.describe().get("reward_mode") != mode) + sum(1 for r in recs
                                                                          if r.get("reward_mode") != mode)
        unrecorded += int(not recs)
    metrics = {"entry_points_checked": len(checked), "entry_points_not_observation_only_by_default": len(bad),
               "unrecorded_mode_selections": unrecorded, "tier_entry_points_not_built": len(errors)}
    record("OO.9", metrics, command=CMD, notes=f"entry points default-constructed/called with no mode named: "
           f"{checked}; not observation-only: {bad}; tier entry points that could not be built: {errors}. Explicit "
           "legacy/verified/observation-only selections recorded by the environment and in every stored record.")
    assert len(checked) >= 12 and not bad and unrecorded == 0, (bad, unrecorded)
    assert not errors, f"tier entry points not built: {errors}"


# ====================================================================== OO.10
def test_OO10_assigned_objectives_delivered_progress_never():
    """In every tier an objective is assigned; the learner receives its public text and nothing of its progress;
    the evaluator verifies it."""
    from crucible.experience import probes_tiers as PT
    from crucible.experience.probes import _plain, exact_hits, text_numbers
    per, errors = {}, {}

    def kernel_tier(mk):
        env = mk()
        a = env.agents()[0]
        ids = PT.revenue_scenario(env, {t: "N" * 31 for t in ("RUNTIME", "INTERNAL", "PRIVATEMAIL", "SPEC")},
                                  unit=HOUR, window_units=720)
        reach = [env.observe(a), env.observe_since(a, 0), env.tools(a)]
        for _ in range(3):
            env.advance(HOUR)
            reach.append(env.observe(a))
        v = env.verify()["agents"][a]
        prog = float(v["progress"])
        data = _plain(reach)
        hits = len(exact_hits({"progress": prog}, data)) + len(exact_hits({"progress": prog}, data,
                                                                          text_numbers(data)))
        delivered = isinstance(reach[-1].get("objective"), str) and bool(reach[-1]["objective"])
        return {"delivered": delivered, "progress": prog, "exposed": hits, "verified": "success" in v,
                "rewards": [env.reward(a)]}

    def tier1():
        out = {}
        for name, kw in (("rapid:net_routing", {}), ("mech:scm_additive", {"validate": True})):
            env = make(1, name, 3, **kw)
            a = env.agents()[0]
            reach = []
            for _ in range(6):
                env.advance(env.tick_s)
                reach.append(env.observe(a))
            ver = env.verify()
            vals = {}
            for p, x in __import__("crucible.experience.probes", fromlist=["numbers"]).numbers(
                    {k: v for k, v in (ver.get("agents") or {}).get(a, {}).items() if k in ("progress",)}):
                vals[p] = x
            data = _plain(reach)
            hits = sum(len(exact_hits({"p": x}, data)) for x in vals.values() if abs(x) > 1e-9 and x != 1.0)
            out[name] = {"delivered": isinstance(reach[-1].get("objective"), str) and bool(reach[-1]["objective"]),
                         "exposed": hits, "verified": bool(ver), "rewards": [env.reward(a)]}
        return out
    for t, fn in ((1, tier1), (2, lambda: {n: kernel_tier(lambda n=n: make(2, n, 3)) for n in _tier2_names()}),
                  (3, lambda: {CIV: kernel_tier(lambda: make(3, CIV, 3, agents=["learner"]))})):
        r, err = _attempt(fn)
        if err:
            errors[t] = err
        else:
            per[t] = r
    tier0 = kernel_tier(lambda: make(0, "tiny", 3, agents=["learner"], overrides={"n_consumers": 60}))
    delivered = [t for t, r in per.items() if all(x["delivered"] and x["verified"] for x in r.values())]
    exposed = sum(x["exposed"] for r in per.values() for x in r.values()) + tier0["exposed"]
    paid = sum(1 for r in per.values() for x in r.values() for y in x["rewards"] if y != 0.0)
    metrics = {"tiers_covered": len(per), "objective_delivered_tiers": len(delivered),
               "objective_progress_values_exposed": exposed, "nonzero_rewards_with_objective": paid,
               "tier0_objective_delivered": tier0["delivered"]}
    record("OO.10", metrics, command=CMD, notes=f"per tier: {per}; tier 0: {tier0}. Progress values (the "
           "verifier's, distinctive by construction in kernel tiers: a circular flow the evaluator nets out) searched "
           "as numbers and in text across observations, events and tools. Tiers not measured: "
           f"{_fail_tiers(errors)}")
    assert exposed == 0 and paid == 0, metrics
    assert len(per) == 3 and len(delivered) == 3, f"objectives in tiers {sorted(per)}; " + _fail_tiers(errors)


# ====================================================================== OO.11
def _self_experiment(env, number_or_measure, deadline_s: int, waits: int, dt: int) -> dict:
    a = env.agents()[0]
    measure = {"number": number_or_measure} if isinstance(number_or_measure, str) else number_or_measure
    r = env.execute(a, {"tool": "exp.register", "args": {
        "hypothesis": "time and my own actions change this quantity", "measure": measure,
        "prediction": {"op": "increase", "min_change": 0}, "deadline_in_s": int(deadline_s)}})
    if not r["ok"]:
        raise RuntimeError(f"exp.register refused: {r.get('error')}")
    rewards = []
    for _ in range(waits):
        env.advance(dt)
        rewards.append(env.reward(a))
    evs = [e for e in env.observe_since(a, 0)[0] if e["type"] == "exp.resolved"
           and (e["cause"] or {}).get("action_seq") == r["action_seq"]]
    return {"resolved": len(evs), "rewards_paid": sum(1 for x in rewards if x != 0.0)}


def test_OO11_self_generated_experiments_in_every_tier_unrewarded():
    per, errors = {}, {}
    bank = {"tool": "bank.accounts", "where": {"owner": "$self"}, "field": "balance", "aggregate": "sum"}

    def tier1():
        out = {}
        for name, kw in (("rapid:net_routing", {}), ("mech:scm_additive", {"validate": False}),
                         ("rapid:team_commons", {})):
            env = make(1, name, 4, **kw)
            out[name] = _self_experiment(env, "world.time_s", 2 * env.tick_s, 3, env.tick_s)
        return out
    for t, fn in ((1, tier1),
                  (2, lambda: {n: _self_experiment(make(2, n, 4), bank, 2 * HOUR, 3, HOUR) for n in _tier2_names()}),
                  (3, lambda: {CIV: _self_experiment(make(3, CIV, 4, agents=["learner"]), bank, 2 * HOUR, 3, HOUR)})):
        r, err = _attempt(fn)
        if err:
            errors[t] = err
        else:
            per[t] = r
    # the same experiments with every reward mode: rewards identical with and without them (none paid in default)
    with_tiers = [t for t, r in per.items() if all(x["resolved"] >= 1 for x in r.values())]
    paid = sum(x["rewards_paid"] for r in per.values() for x in r.values())
    metrics = {"tiers_with_self_experiments": len(with_tiers), "self_experiment_rewards_paid": paid,
               "experiments_resolved": sum(x["resolved"] for r in per.values() for x in r.values())}
    record("OO.11", metrics, command=CMD, notes=f"exp.register -> exp.resolved linked to the registering action, per "
           f"tier: {per}. Tier 1 adapters: crucible.experience.selfexp (measures = the agent's own observation "
           f"numbers); kernel tiers: the exp module. Tiers not measured: {_fail_tiers(errors)}")
    assert paid == 0, metrics
    assert len(with_tiers) == 3, f"self-experiments in tiers {with_tiers}; " + _fail_tiers(errors)


# ====================================================================== OO.12
def test_OO12_verified_labels_through_a_separate_authorised_channel(tmp_path):
    from crucible.experience.labels import SCHEMA, LabelChannel, LabelChannelError
    from crucible.experience.probes import _plain, dump

    def run(with_labels: bool, name: str, kw: dict):
        env = make(1, name, 6, **kw) if not name.startswith("tiny") else make(0, "tiny", 6, agents=["learner"],
                                                                                overrides={"n_consumers": 40})
        a = env.agents()[0]
        ch = LabelChannel(env, authorised_by="operator:gate-OO12", purpose="meta_learning",
                          path=str(tmp_path / f"{name.replace(':', '_')}.labels.jsonl")) if with_labels else None
        views, rewards, labels = [], [], []
        dt = getattr(env, "tick_s", HOUR) if not name.startswith("tiny") else HOUR
        for k in range(8):
            env.advance(dt)
            rewards.append(env.reward(a))
            views.append(env.observe(a))
            if ch is not None and k % 2 == 1:
                labels += ch.issue()
        return views, rewards, labels, ch
    results = {}
    for name, kw in (("mech:scm_additive", {"validate": True}), ("rapid:net_routing", {}), ("tiny", {})):
        v0, r0, _, _ = run(False, name, kw)
        v1, r1, labels, ch = run(True, name, kw)
        same = dump(_plain(v0)) == dump(_plain(v1))
        in_obs = any(lb["hash"] in dump(_plain(v1)) for lb in labels)
        stored = LabelChannel.read(ch.path)
        results[name] = {"views_identical": same, "labels": len(labels), "label_hash_in_observations": in_obs,
                         "nonzero_rewards": sum(1 for x in r0 + r1 if x != 0.0), "rewards_identical": r0 == r1,
                         "chain_ok": LabelChannel.verify_chain(stored) and len(stored) == len(labels),
                         "schema_ok": all(lb["schema"] == SCHEMA for lb in labels)}
    refused = 0
    for bad in ({"authorised_by": "", "purpose": "meta_learning"}, {"authorised_by": "op", "purpose": "reward"}):
        try:
            LabelChannel(make(1, "rapid:net_routing", 1), **bad)
        except LabelChannelError:
            refused += 1
    separate = all(r["views_identical"] and not r["label_hash_in_observations"] and r["chain_ok"] and r["schema_ok"]
                   and r["labels"] > 0 for r in results.values()) and refused == 2
    converted = sum(r["nonzero_rewards"] for r in results.values()) + sum(0 if r["rewards_identical"] else 1
                                                                         for r in results.values())
    metrics = {"label_channel_separate_from_observations": bool(separate), "labels_converted_to_step_rewards": converted,
               "labels_issued": sum(r["labels"] for r in results.values()), "unauthorised_channels_refused": refused}
    record("OO.12", metrics, command=CMD, notes=f"crucible.experience.labels.LabelChannel (evaluator side, a named "
           f"authority and a declared purpose; crucible-label/1 records, hash-chained). Twin runs with and without "
           f"the channel: {results}")
    assert separate and converted == 0, metrics


# ====================================================================== OO.14 / OO.16
@pytest.fixture(scope="module")
def tier_probes():
    from crucible.experience import probes_tiers as PT
    return PT.run_tiers()


def test_OO14_leak_channels_probed_per_tier(tier_probes):
    res = tier_probes
    per = {t: {"channels": r["channels_probed"], "leaks": r["n_leaks"], "kinds": r["kinds"]}
           for t, r in res["tiers"].items()}
    covered = [t for t, r in per.items() if len(r["channels"]) >= 4]
    min_channels = min((len(r["channels"]) for r in per.values()), default=0)
    recovered = sum(r["leaks"] for r in per.values())
    guide = open(os.path.join(ROOT, "docs", "v5", "LEARNER_GUIDE.md")).read()
    gym_doc = inspect.getsource(__import__("crucible.experience.gym", fromlist=["x"]))
    documented = "not a learning signal" in guide and "not a learning signal" in gym_doc
    metrics = {"tiers_covered": len(covered), "leak_channels_probed_per_tier": min_channels if per else 0,
               "verifier_values_recovered": recovered, "scalar_reward_documented": documented}
    path = os.path.join(os.environ.get("TMPDIR", "/tmp"), "OO14_tier_probes.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    record("OO.14", metrics, command=CMD, notes=f"crucible.experience.probes_tiers per tier: {per}; leaks: "
           f"{[l for r in res['tiers'].values() for l in r['leaks']][:6]}; tiers that could not run: "
           f"{res['errors']}")
    assert recovered == 0 and documented, metrics
    assert len(covered) == 3 and min_channels >= 4, f"channels per tier {per}; not run: {res['errors']}"


def test_OO16_no_split_or_identity_reaches_participants(tier_probes):
    res = tier_probes
    per = {t: r["kinds"].get("identity", {"checked": 0, "leaks": 0}) for t, r in res["tiers"].items()}
    covered = [t for t, k in per.items() if k["checked"] > 0]
    metrics = {"tiers_covered": len(covered), "identity_probes": sum(k["checked"] for k in per.values()),
               "split_identity_leaks": sum(k["leaks"] for k in per.values())}
    record("OO.16", metrics, command=CMD, notes=f"identity tokens (family/profile/configuration/split names, spec "
           f"hash, world id, a distinctive seed) searched in everything the participant reached, per tier: {per}; "
           f"leaks {[l for r in res['tiers'].values() for l in r['leaks'] if l['kind'] == 'identity'][:6]}; tiers "
           f"that could not run: {res['errors']}")
    assert metrics["split_identity_leaks"] == 0 and metrics["identity_probes"] >= 3, metrics
    assert len(covered) == 3, f"identity probed in tiers {covered}; not run: {res['errors']}"
