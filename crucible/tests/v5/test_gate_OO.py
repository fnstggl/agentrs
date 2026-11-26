"""Gate OO (Clarification 2: observation-only learning as the universal default), OO.1-OO.8, workstream L.

Every test measures the real system and records evidence. Where Tier 2 (v5 domains) or Tier 3 (civilisations) do not
exist in this checkout, the test records the honest number (``tiers_covered`` counts only v5 tiers that were really
measured; Tier 0 - the 4.0 kernel worlds exposed through `crucible.experience` - is reported separately and never
counted as a tier) and FAILS naming what is missing. Nothing is skipped and no other world type stands in."""
import ast
import importlib.util
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
from crucible.experience.modes import LEGACY, OBSERVATION_ONLY
from crucible.experience.registry import civ_available, civ_profiles, domains

pytestmark = pytest.mark.timeout(2400)
CMD = "PYTHONPATH=src python -m pytest tests/v5/test_gate_OO.py"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DAY = 86400


def _missing_tiers() -> list[str]:
    out = []
    if not domains():
        out.append("Tier 2: no v5 domain is registered (crucible.domains.base.REGISTRY is empty; D-17: domains become "
                   "configurations of crucible.civ)")
    if not civ_available():
        out.append("Tier 3: crucible.civ.generate does not exist (workstream J)")
    return out


def _tier_count(*present: bool) -> int:
    return sum(1 for p in present if p)


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
    for dn, dom in sorted(domains().items()):
        for cfg in dom.configs:
            env = make(2, f"{dn}:{cfg.name}", 0)
            check(f"tier2 {dn}:{cfg.name}", 2, env, env.agents()[0])
    for prof in civ_profiles():
        env = make(3, prof, 0)
        check(f"tier3 {prof}", 3, env, env.agents()[0])
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
        "ReferenceLearner": lambda: ReferenceLearner(),
        "SelfPlayLeague": lambda: SelfPlayLeague(os.path.join(os.environ.get("TMPDIR", "/tmp"), "oo1-league"), t2),
    }
    ctor_bad = []
    for name, mk in ctors.items():
        obj = mk()
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
    tiers = _tier_count(per_tier[1] > 0, per_tier[2] > 0, per_tier[3] > 0)
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
           f"Missing tiers: {_missing_tiers()}")
    assert not bad and not ctor_bad, (bad, ctor_bad)
    assert metrics["env_families_checked"] >= 100
    assert tiers == 3, "observation-only measured on Tier 1 only; " + "; ".join(_missing_tiers())


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
def test_OO3_real_observable_consequences_remain():
    from crucible.experience.scripted import call, deploy_service, discover, need_input
    kinds = {}
    # 4.0 kernel world (tier 0): transactions, customer behaviour, technical measurements, social responses,
    # resource changes, failures, delayed outcomes - all from the participant's own observations
    env = make(0, "tiny", 3, agents=["builder", "client", "neighbour"], endowment=3_000_000,
               overrides={"n_consumers": 200})
    b, c, n = env.agents()
    deploy_service(env, b, handle="obsco", host="obssvc", price=4)
    discover(env, c, "obssvc")
    acct_c = [a for a in call(env, c, "bank.accounts") if a["owner"] == c][0]["account"]
    post = call(env, b, "forum.post", board="general", title="obssvc is live", body="sorting, 4 cents a call")
    reg = env.execute(b, {"tool": "exp.register", "args": {
        "hypothesis": "my balance rises within 3 days", "measure": {"tool": "bank.accounts", "where": {"owner": "$self"},
                                                                    "field": "balance", "aggregate": "sum"},
        "prediction": {"op": "increase", "min_change": 1}, "deadline_in_s": 3 * DAY}})
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
    kinds["transactions"] = any(t.startswith("bank.") for t in types)
    kinds["customer_behaviour"] = sum(d.get("requests", 0) for d in status["days"]) >= 4
    kinds["technical_measurements"] = any(k in json.dumps(status) for k in ("cpu", "latency", "errors"))
    kinds["social_responses"] = any(e["type"].startswith("forum.") and e["actor"] == n for e in evs)
    kinds["resource_changes"] = obs["resources"].get("balance_cents") is not None and \
        any(k.startswith("bank.balances.") for k in obs["numbers"])
    kinds["failures"] = not fail["ok"] and isinstance(fail["error"], dict)
    kinds["delayed_outcomes"] = any(e["type"] == "exp.resolved" and (e["cause"] or {}).get("action_seq") ==
                                    reg["action_seq"] for e in evs)
    # Tier 1: readings, revenue/cost, cash in rapid and mechanism worlds
    t1 = make(1, "rapid:net_routing", 3)
    a = t1.agents()[0]
    t1.advance(t1.tick_s)
    o1 = t1.observe(a)
    m1 = make(1, "mech:scm_additive", 3, validate=False)
    am = m1.agents()[0]
    m1.advance(m1.tick_s)
    om = m1.observe(am)
    tier1_ok = bool(o1["numbers"]) and "cash_cents" in o1["resources"] and \
        "rig.seats.r1.revenue_cents" in om["numbers"] and any(k.startswith("rig.seats.r1.readings.") for k in om["numbers"])
    tiers = _tier_count(tier1_ok, bool(domains()), civ_available())
    metrics = {"consequence_kinds_observable": sum(1 for v in kinds.values() if v), "tiers_covered": tiers,
               "kernel40_consequence_kinds": sum(1 for v in kinds.values() if v), "tier1_consequences": tier1_ok}
    record("OO.3", metrics, command=CMD, notes=f"kinds (participant's own observations in an observation-only "
           f"4.0 kernel world): {kinds}; Tier 1 rapid/mechanism readings, revenue, cost, cash present: {tier1_ok}. "
           f"Missing tiers: {_missing_tiers()}")
    assert metrics["consequence_kinds_observable"] >= 7, kinds
    assert tiers == 3, "consequences measured in Tier 1 and 4.0 kernel worlds only; " + "; ".join(_missing_tiers())


# ====================================================================== OO.4
def test_OO4_independent_verification_through_a_separate_interface():
    from crucible.experience import host
    from crucible.goals.contracts import GoalContract
    from crucible.goals.metrics import evaluate_metric
    matches = []
    # Tier 1 rapid, hosted in its own process: the participant handle cannot reach the verifier
    part, ev = host({"tier": 1, "name": "rapid:net_routing", "seed": 3})
    try:
        a = part.agents()[0]
        for _ in range(5):
            part.advance(3600)
        refused = [part._raw(op)[0] == "error" for op in ("verify", "digest", "describe", "env", "world", "_obs")]
        reach = (not all(refused)) or any(hasattr(part, x) for x in ("verify", "env", "world"))
        v = ev.verify()["outcome"]
        matches.append(abs(v["value"] - (v["revenue"] - v["cost"])) < 1e-6 if "revenue" in v else v["value"] is not None)
    finally:
        ev.close()
    # Tier 1 mechanism: the operator record's normalised score recomputed from the episode's net
    m = make(1, "mech:scm_additive", 3, validate=True)
    am = m.agents()[0]
    while not m.done:
        m.advance(m.tick_s)
    rec = m.verify()["episodes"][-1]
    matches.append(abs(rec["score"] - m.task.score(rec["net_per_tick"])) < 1e-9 and
                   rec["success"] == m.task.success(rec["net_cents"]))
    # 4.0 kernel world: the GoalVerifier's verdict against a direct evaluation of the predicate's metric
    env = make(0, "tiny", 5, agents=["learner"], endowment=2_500_000)
    goal = GoalContract("cash", "Hold at least $20,000.", {"family": "tiny", "seed": 5}, "$self",
                        {"metric": "cash", "args": {"principal": "$self"}, "op": ">=", "value": 2_000_000},
                        {"metric": "cash", "args": {"principal": "$self"}, "target": 2_000_000}, 30 * DAY)
    env.set_objective(env.agents()[0], goal)
    env.advance(DAY)
    st = env.verify()["agents"][env.agents()[0]]
    cash = evaluate_metric(env.world, "cash", {"principal": env.agents()[0]})
    matches.append(abs(st["progress"] - min(1.0, cash / 2_000_000)) < 1e-9)
    tool_reach = {t["name"] for t in env.tools(env.agents()[0])} & set(EO.TOOLS)
    tiers = _tier_count(True, bool(domains()), civ_available())
    metrics = {"tiers_covered": tiers, "verifier_matches_ground_truth": all(matches),
               "verifier_reachable_from_learner": bool(reach or tool_reach), "verifications_compared": len(matches),
               "participant_requests_refused": sum(refused)}
    record("OO.4", metrics, command=CMD, notes="the evaluator's verdicts recomputed independently (rapid outcome "
           "from its ledger, mechanism score from the task's private reference, GoalVerifier progress from the "
           "metric); the hosted world (crucible.experience.remote) refuses verify/digest/describe/attribute access on "
           "the participant pipe; evaluator tools are absent from the participant's tools. Missing tiers: "
           f"{_missing_tiers()}")
    assert metrics["verifier_matches_ground_truth"] and not metrics["verifier_reachable_from_learner"]
    assert tiers == 3, "verification measured in Tier 1 and 4.0 kernel worlds only; " + "; ".join(_missing_tiers())


# ====================================================================== OO.5
def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


@pytest.mark.slow
def test_OO5_operation_without_an_objective_within_resource_limits():
    limits = {"max_rss_growth_mb": 600.0, "max_wall_s": 900.0, "max_event_log_in_memory": 400_000}
    violations = []
    t0, r0 = time.perf_counter(), _rss_mb()
    days = {}
    # Tier 1: objective-free operation across many episodes (rapid and mechanism worlds reset at their horizon)
    for name in ("rapid:net_routing", "rapid:team_commons", "mech:scm_additive"):
        env = make(1, name, 1, **({"validate": False} if name.startswith("mech") else {}))
        a = env.agents()[0]
        start, eps = env.now, 0
        for _ in range(600):
            env.advance(env.tick_s)
            assert env.reward(a) == 0.0
            if env.done:
                env.reset()
                eps += 1
        days[name] = {"episodes": eps}
    # 4.0 kernel world, no objective, 90 simulated days, with the world's log retention as the memory limit
    env = make(0, "tiny", 2, agents=["learner"], overrides={"n_consumers": 100, "params": {"log_retain": 200_000}})
    a = env.agents()[0]
    acct = [x for x in env.execute(a, {"tool": "bank.accounts", "args": {}})["result"]][0]["account"]
    for d in range(90):
        env.execute(a, {"tool": "forum.post", "args": {"board": "general", "title": f"day {d}", "body": "notes"}})
        env.advance(DAY)
        obs = env.observe(a)
        if validate_observation(obs):
            violations.append(f"day {d}: invalid observation")
        if len(env.world.log) > limits["max_event_log_in_memory"]:
            violations.append(f"day {d}: {len(env.world.log)} events in memory")
        if env.reward(a) != 0.0:
            violations.append(f"day {d}: nonzero reward")
    kernel_days = env.now // DAY
    v = env.verify()
    wall, growth = time.perf_counter() - t0, _rss_mb() - r0
    if wall > limits["max_wall_s"]:
        violations.append(f"wall {wall:.0f}s")
    if growth > limits["max_rss_growth_mb"]:
        violations.append(f"rss growth {growth:.0f} MB")
    civ_days = 0
    if civ_available():
        prof = civ_profiles()[0]
        cenv = make(3, prof, 0)
        ca = cenv.agents()[0]
        for _ in range(90):
            cenv.advance(DAY)
            if cenv.reward(ca) != 0.0:
                violations.append("civ nonzero reward")
        civ_days = cenv.now // DAY
    tiers = _tier_count(True, bool(domains()), civ_available() and civ_days >= 90)
    metrics = {"objectiveless_tiers": tiers, "objectiveless_civ_sim_days": int(civ_days),
               "resource_limit_violations": len(violations), "objectiveless_kernel40_sim_days": int(kernel_days),
               "tier1_episodes_run": sum(x["episodes"] for x in days.values()), "rss_growth_mb": round(growth, 1),
               "wall_s": round(wall, 1), "kernel_invariant_violations": len(v["invariant_violations"])}
    record("OO.5", metrics, command=CMD, notes=f"limits {limits}; Tier 1 episodes {days}; a 4.0 kernel world ran "
           f"{kernel_days} simulated days with no objective (evaluator measures: {v['measures']}); violations "
           f"{violations}. Missing tiers: {_missing_tiers()}")
    assert metrics["resource_limit_violations"] == 0, violations
    assert tiers == 3 and civ_days >= 90, "objective-free operation measured in Tier 1 and a 4.0 kernel world only; " + \
        "; ".join(_missing_tiers())


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
    runs = {}
    for tier, name in ((1, "rapid:net_routing"), (1, "mech:scm_additive"), (0, "tiny")):
        kw = {"agents": ["learner"]} if tier == 0 else ({"validate": False} if name.startswith("mech") else {})
        env = make(tier, name, 3, **kw)
        L = mod.CuriosityLearner(seed=1)
        runs[name] = mod.run(env, L, 100, update_every=20)
    updates = sum(r["param_updates_changed"] for r in runs.values())
    env_paid = sum(r["env_reward_sum"] for r in runs.values())
    own = {n: round(r["own_reward_sum"], 4) for n, r in runs.items()}
    changes = int(tree() != before) + len(_imports(LEARNER))
    tiers = _tier_count(True, bool(domains()), civ_available())
    metrics = {"tiers_covered": tiers, "external_reward_param_updates": updates, "env_changes_required": changes,
               "env_reward_paid": env_paid, "distinct_weight_hashes": sum(len(set(r["hashes"])) for r in runs.values())}
    record("OO.6", metrics, command=CMD, notes="examples/external_learner/curiosity_learner.py (numpy only, public "
           f"crucible.experience names only: violations {_imports(LEARNER)}) computes its own prediction-error "
           f"curiosity reward {own}; the environment paid {env_paid}; updates that changed its weight hash: "
           f"{ {n: r['param_updates_changed'] for n, r in runs.items()} }; src tree unchanged: {changes == 0}. "
           f"Missing tiers: {_missing_tiers()}")
    assert updates >= 3 and changes == 0 and env_paid == 0.0
    assert tiers == 3, "external learner run in Tier 1 and a 4.0 kernel world only; " + "; ".join(_missing_tiers())


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
