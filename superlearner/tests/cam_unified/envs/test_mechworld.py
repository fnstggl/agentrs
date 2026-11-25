"""MechWorldAdapter: parity with MechVecEnv, entity/tool presentation, firewall, invariance pairs, evaluator; the
ow_works rig task inside the real OpenWorld."""
from __future__ import annotations

import json
import random

import numpy as np
import pytest

import crucible
crucible.load_all_modules()
from crucible.research.envs import MechVecEnv, money_scale, symlog      # noqa: E402
from crucible.research.learner import EnvSpec                          # noqa: E402
from crucible.research.tasks import Task, make_task                    # noqa: E402

from superlearner.cam_unified.interfaces import Action, EnvAdapter, GoalEvaluator, ObsKind   # noqa: E402
from superlearner.cam_unified.envs.mechworld import (MechWorldEvaluator, RigNetObjective, RigOperatorControl,  # noqa: E402
                                                     RigTask, mech_world)
from superlearner.cam_unified.envs.openworld import WAIT_TOOL         # noqa: E402
from superlearner.cam_unified.envs.worldgen import Surface, invariance_pair, register_families   # noqa: E402


class RandomSetter:
    """Test agent: random lever changes (values from the public ranges it was shown) and random waits."""

    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.ranges = {}

    def act(self, batch, tools):
        out = []
        for b, olist in enumerate(batch.observations):
            for o in olist:
                if o.kind is ObsKind.CONSTRAINT:
                    self.ranges.setdefault(b, {})[o.entity.split(":", 1)[1]] = o.meta
            tl = [t for t in tools[b] if t.name.startswith("set_")]
            if self.rng.random() < 0.5 and tl:
                t = self.rng.choice(tl)
                m = self.ranges[b][t.name[4:]]
                v = self.rng.uniform(m["min"], m["max"])
                out.append(Action(t.name, {"value": int(round(v)) if m["integer"] else v}))
            else:
                out.append(Action(WAIT_TOOL, {"seconds": self.rng.choice([600, 3600, 7200])}))
        return out


def drive(ad, agent, steps, think_s=600, log=None):
    """Run `agent`; `log` (optional list) receives per step (actions, results, report count per env after the
    advance), so a test can reconstruct which settings the agent had made before each tick."""
    batch = ad.reset()
    seen = [list(batch.observations[b]) for b in range(ad.n)]
    for _ in range(steps):
        acts = agent.act(batch, ad.available_tools())
        res = ad.execute(acts)
        batch = ad.advance_time([(r.duration if r else 0) + think_s / 86400 for r in res])
        for b in range(ad.n):
            seen[b] += batch.observations[b]
        if log is not None:
            log.append((acts, res, [sum(1 for o in seen[b] if o.kind is ObsKind.EVENT) for b in range(ad.n)]))
        if all(batch.done):
            break
    return seen


def test_parity_with_mechvecenv():
    register_families()
    tasks = [make_task("scm_additive", 3, validate=False), make_task("cw1_scm_delay_stock", 5, validate=False),
             make_task("abm_utility", 2, validate=False)]
    ad, op = mech_world(tasks, seed=1)
    ev = MechWorldEvaluator(op)
    assert isinstance(ad, EnvAdapter) and isinstance(ev, GoalEvaluator)
    log = []
    seen = drive(ad, RandomSetter(0), 400, log=log)
    rewards = ev.take_rewards()
    for b, task in enumerate(tasks):
        e = op.envs[b]
        assert e.ticks > 20
        # 1. an independent MechVecEnv fed the per-tick action vectors reproduces every tick bit for bit
        spec = EnvSpec(max_levers=32, max_gauges=32)
        ref = MechVecEnv([task], spec)
        ref.reset()
        reports = [json.loads(o.content)["payload"] for o in seen[b] if o.kind is ObsKind.EVENT]
        assert len(reports) == e.ticks
        tot = 0.0
        unit, scale = e.gauge_unit
        for k, a in enumerate(e.actions):
            act = np.zeros((1, 32))
            act[0, :len(a)] = a
            obs, rew, done, info = ref.step(act)
            rep = reports[k]
            assert rep["net_cents"] == float(info["net_cents"][0])
            g = obs[0, spec.gauges][:len(e.gauge_engine)]
            shown = np.array([rep["readings"][e.gauge_shown[i]] / scale for i in range(len(e.gauge_engine))])
            np.testing.assert_allclose(symlog(shown).astype(np.float32), g, rtol=1e-6, atol=1e-6)
            tot += float(rew[0])
        assert tot == pytest.approx(rewards[b], rel=1e-6, abs=1e-6)
        assert e.net_total == pytest.approx(sum(r["net_cents"] for r in reports))
        # 2. the vector that drove the sim at every tick is exactly the agent's own settings (the last value it set
        #    for each lever before that tick, else the initial setting it was shown), normalised with the public
        #    ranges it was shown: 2 (v - lo) / (hi - lo) - 1
        cons = {o.entity: o.meta for o in seen[b] if o.kind is ObsKind.CONSTRAINT}
        current = {o.entity.split(":", 1)[1]: o.value for o in seen[b][:40]
                   if o.kind is ObsKind.NUMERIC and o.entity_type == "lever" and o.t == 0.0}
        assert set(current) == set(e.lever_shown)
        expected, done_ticks = [], 0
        for acts, res, n_reports in log:
            a, r = acts[b], res[b]
            if a is not None and a.tool.startswith("set_") and r.ok:
                m = cons[f"lever:{a.tool[4:]}"]
                current[a.tool[4:]] = float(min(max(a.args["value"], m["min"]), m["max"]))
            for _ in range(n_reports[b] - done_ticks):     # ticks that ran during this step's advance
                vec = []
                for name in e.lever_shown:                 # engine order (the order of e.actions)
                    m = cons[f"lever:{name}"]
                    span = m["max"] - m["min"]
                    vec.append(0.0 if span <= 0 else 2.0 * (current[name] - m["min"]) / span - 1.0)
                expected.append(vec)
            done_ticks = n_reports[b]
        assert len(expected) == e.ticks == len(e.actions)
        np.testing.assert_allclose(np.array(e.actions), np.array(expected), rtol=0, atol=1e-12)
        sets = sum(1 for acts, res, _ in log if acts[b].tool.startswith("set_") and res[b].ok)
        assert sets >= 10                                  # the comparison covers many agent decisions
        # 3. units: money in cents, gauges exactly as the simulator states them (the default surface never rescales)
        assert (unit, scale) == ("reading", 1.0)
        units = {o.unit for o in seen[b] if o.kind is ObsKind.NUMERIC and o.entity_type == "gauge"}
        assert units == {unit}
        assert {o.unit for o in seen[b] if o.meta.get("field") in ("net_cents", "funds")} == {"cents"}
    ad.close()


def test_entities_tools_and_rejections():
    task = make_task("scm_gate", 4, validate=False)
    ad, op = mech_world([task], seed=2)
    b0 = ad.reset()
    e = op.envs[0]
    obs = b0.observations[0]
    levers = [o for o in obs if o.kind is ObsKind.CONSTRAINT]
    assert len(levers) == len(e.lever_engine) and all(o.entity_type == "lever" for o in levers)
    gauges = [o for o in obs if o.kind is ObsKind.RELATION and o.entity_type == "gauge"]
    assert len(gauges) == len(e.gauge_engine)
    tools = ad.available_tools()[0]
    sets = [t for t in tools if t.name.startswith("set_")]
    assert len(sets) == len(e.lever_engine) and tools[-1].name == WAIT_TOOL
    # shown names are per-episode neutral names, not the engine's
    assert not set(e.lever_shown) & set(e.lever_engine)
    for t in sets:
        lv = e.lever_spec[e.lever_shown.index(t.name[4:])]
        p = t.parameters["properties"]["value"]
        assert p["type"] == ("integer" if lv.integer else "number") and p["minimum"] == pytest.approx(lv.lo)
    t = sets[0]
    hi = t.parameters["properties"]["value"]["maximum"]
    bad = ad.execute([Action(t.name, {"value": hi + 1000})])[0]
    assert not bad.ok and bad.error == "invalid_action"
    unk = ad.execute([Action("set_nothing", {"value": 1})])[0]
    assert not unk.ok and unk.error == "unknown_tool"
    good = ad.execute([Action(t.name, {"value": hi})])[0]
    assert good.ok and good.duration > 0
    nb = ad.advance_time([1 / 24 + 1e-9])
    rep = [json.loads(o.content)["payload"] for o in nb.observations[0] if o.kind is ObsKind.EVENT]
    assert len(rep) == 1 and rep[0]["settings"][t.name[4:]] == pytest.approx(hi)
    # a new episode renames everything
    names0 = list(e.lever_shown)
    ad.reset()
    assert op.envs[0].lever_shown != names0


def test_special_tools_and_end_of_episode_success():
    """Contract special tools; success is decided on the episode's final net (Task.success), not at the first
    crossing of the goal, which is reported separately."""
    from dataclasses import replace
    task = make_task("scm_additive", 3, validate=False)
    ad, op = mech_world([task], seed=4, tick_s=3600)
    ev = MechWorldEvaluator(op)
    ad.reset()
    n = ad.execute([Action("noop", {})])[0]
    assert n.ok and n.duration == 0.0
    a = ad.execute([Action("advance_time", {"days": 2 / 24})])[0]
    assert a.ok and a.duration == pytest.approx(2 / 24)
    nb = ad.advance_time([a.duration])
    assert op.envs[0].ticks == 2 and nb.t[0] == pytest.approx(2 / 24)
    assert not ad.execute([Action("advance_time", {"days": 0})])[0].ok
    assert ad.goals()[0].meta["reward_scale_cents"] == pytest.approx(money_scale(task))
    ad.close()
    # success is Task.success on the final net, decided at the episode's end; the first crossing is reported apart.
    # The dynamics are deterministic given the task and the actions (parity test), so a first run finds the best
    # cumulative net M of a losing random schedule; a goal of M is then crossed during the replay and lost by the end.
    def run(goal):
        ad, op = mech_world([replace(task, goal_net_cents=goal)], seed=4)
        ev = MechWorldEvaluator(op)
        nets, mid = [], []
        batch, agent = ad.reset(), RandomSetter(5)
        while not all(batch.done):
            res = ad.execute(agent.act(batch, ad.available_tools()))
            batch = ad.advance_time([r.duration + 600 / 86400 for r in res])
            nets.append(op.envs[0].net_total)
            if not batch.done[0]:
                mid.append((ev.success_t[0], ev.evaluate_goal(0).achieved))
        out = (ev.evaluate_goal(0), list(ev.success_t), list(ev.reached_t), op.envs[0].t_s, nets, mid)
        ad.close()
        return out
    _, _, _, _, nets, _ = run(None)
    best, final_net = max(nets[:-1]), nets[-1]
    assert best > final_net + 1                              # random settings lose money after a while
    res, succ, reached, t_end, nets2, mid = run(best)
    assert nets2 == nets                                     # the replay is identical
    assert reached[0] is not None and succ[0] is None and not res.achieved
    assert res.details["reached_day"] is not None and res.details["episode_done"]
    assert all(s is None and not a for s, a in mid)          # never "achieved" before the end
    res, succ, reached, t_end, _, mid = run(final_net - 1)
    assert res.achieved and succ[0] == t_end and all(s is None and not a for s, a in mid)


def test_mechworld_firewall():
    task = make_task("scm_additive", 3, validate=True)          # carries private reference/default nets
    ad, op = mech_world([task], seed=3)
    ev = MechWorldEvaluator(op)
    seen = drive(ad, RandomSetter(1), 30)
    from test_openworld import _walk
    from crucible.mechanisms.sim import MechSim
    reach = _walk(ad)
    assert not [x for x in reach if isinstance(x, (Task, MechSim, MechVecEnv))]
    assert [x for x in _walk(ev) if isinstance(x, Task)]          # negative control
    blob = json.dumps([o.to_dict() for o in seen[0]]) + json.dumps([t.to_dict() for t in ad.available_tools()[0]])
    blob += json.dumps([g.to_dict() for g in ad.goals()])
    eng = op.envs[0].env.sims[0].engine
    for secret in ["scm_additive", str(task.reference_net), str(task.default_net)] + \
            [lv.name for lv in eng.levers(0)] + [g.name for g in eng.gauges(0)]:
        assert f'"{secret}"' not in blob and f":{secret}\"" not in blob, secret
    res = ev.evaluate_goal(0)
    assert res.details["reference_net"] == task.reference_net and res.score == pytest.approx(
        task.score(op.envs[0].net_total / op.envs[0].ticks))


def test_invariance_pair_same_dynamics_different_surface():
    task = make_task("cw1_scm_confounder_stock", 2, validate=False)
    sa, sb = invariance_pair(4)
    surfaces = {0: sa, 1: sb}
    ad, op = mech_world([task, task], seed=0, surface=lambda b, ep: surfaces[b])
    batch = ad.reset()
    ea, eb = op.envs
    assert ea.lever_shown != eb.lever_shown
    assert ea.surface.variant != eb.surface.variant
    ta = [t.name for t in ad.available_tools()[0]]
    tb = [t.name for t in ad.available_tools()[1]]
    assert ta != tb
    rng = random.Random(0)
    for step in range(120):
        j = rng.randrange(len(ea.lever_engine))
        lv = ea.lever_spec[j]
        v = rng.uniform(lv.lo, lv.hi)
        acts = [Action(f"set_{ea.lever_shown[j]}", {"value": round(v) if lv.integer else v}),
                Action(f"set_{eb.lever_shown[j]}", {"value": round(v) if lv.integer else v})]
        if step % 3:
            acts = [Action(WAIT_TOOL, {"seconds": 3600})] * 2
        res = ad.execute(acts)
        ad.advance_time([r.duration + 600 / 86400 for r in res])
    assert ea.ticks == eb.ticks > 50
    assert ea.net_total == eb.net_total                              # identical dynamics, tick for tick
    ra = [json.loads(o.content)["payload"] for o in ea.buffer if o.kind is ObsKind.EVENT]
    rb = [json.loads(o.content)["payload"] for o in eb.buffer if o.kind is ObsKind.EVENT]
    for x, y in zip(ra, rb):
        assert x["net_cents"] == y["net_cents"]
        for k in range(len(ea.gauge_engine)):
            assert x["readings"][ea.gauge_shown[k]] / ea.gauge_unit[1] == pytest.approx(
                y["readings"][eb.gauge_shown[k]] / eb.gauge_unit[1], rel=1e-12)


def test_rig_task_inside_openworld_dense_verified_reward():
    from superlearner.cam_unified.envs.agent_protocol import PolicyAgent
    from superlearner.cam_unified.envs.harness import run_lifetime
    from superlearner.cam_unified.envs.openworld import open_world
    from superlearner.cam_unified.envs.openworld_eval import OpenWorldEvaluator
    task = RigTask(world={"family": "ow_works", "seed": 1, "overrides": {}}, endowment=2_000_000,
                   horizon_s=2 * 86400, goal_net_cents=1)
    ad, op = open_world([task], seed=0)
    ev = OpenWorldEvaluator(op)
    agent = PolicyAgent(lambda b: RigOperatorControl(horizon=40))
    res = run_lifetime(agent, ad, ev, think_s=600)
    assert ad.goals()[0].meta["reward_scale_cents"] == task.reward_scale_cents
    ad.close()
    tools = res.accounting["by_tool"]
    assert tools.get("rig.take") == 1 and tools.get("rig.set", 0) >= 3
    m = res.final[0]["metrics"]
    assert m["rig_ticks"] >= 40
    # dense: the verified return is the facility's realised net in the declared money unit (Crucible's public
    # scale: starting balance / ENDOWMENT_TICKS cents per reward unit), non-zero; the unit is in the GoalSpec
    from crucible.research.tasks import ENDOWMENT_TICKS
    scale = 2_000_000 / ENDOWMENT_TICKS
    assert task.reward_scale_cents == scale and res.final[0]["details"]["reward_scale_cents"] == scale
    assert res.returns[0] * scale == pytest.approx(m["rig_net"]) and m["rig_net"] != 0
    assert abs(res.returns[0]) < 5                                 # O(1) rewards, comparable with RewardContract's
    assert res.final[0]["evaluator"] == "crucible.goals.GoalVerifier"
    # success is the net at the horizon, not the first crossing
    st = res.final[0]["details"]
    assert res.final[0]["achieved"] == (m["rig_net"] >= 1 and st["success_t"] is not None)
