"""ExperienceOperator / ExperienceAdapter repairs requested by W1 and W8 (CAM Unified, 2026-11-26):

* a Crucible engine exception inside `execute` (real: rapid ``posm_sign_switch`` ``rapid.levers`` with an empty
  ``values`` object raises ``KeyError`` upstream; the 1B run crashed on the same bug in another lever) becomes an
  observable ERROR result instead of killing the run;
* `ExecutionResult.cost` is the observed charge of the action (its own direct result), not always 0;
* utility references are per episode: `IntegrityWorld.reset` draws new demand parameters, so episode k is normalised
  with episode k's oracle, and a resolved episode's label survives the reset that starts the next one;
* a world without an in-place reset (kernel tier) is rebuilt and the actor's callables follow the new world.
"""
from __future__ import annotations

import copy
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.abspath(os.path.join(HERE, "../../../../crucible/src")), os.path.abspath(os.path.join(HERE, "../../../src"))):
    if p not in sys.path:
        sys.path.insert(0, p)

from superlearner.cam_unified.envs.experience import ExperienceOperator, WorldSpec, observed_spend  # noqa: E402
from superlearner.cam_unified.envs.integrity import TICK_S, VARIANTS, IntegrityWorld  # noqa: E402
from superlearner.cam_unified.interfaces import Action, ObsKind  # noqa: E402


def _integrity(variant="easy_useless", seed=0, **kw):
    return ExperienceOperator([WorldSpec(-1, f"integrity:{variant}", seed)], **kw)


def test_engine_exception_becomes_an_observable_error_and_the_world_keeps_running():
    op = ExperienceOperator([WorldSpec(1, "rapid:posm_sign_switch", 0)])
    ad = op.adapter(deliver_results=True)
    ad.reset()
    bad = Action(tool="rapid.levers", args={"values": {}})
    raw = op.worlds[0].env
    try:                                     # the upstream bug is real (precondition of this test)
        raw.execute(op.worlds[0].agent, {"tool": "rapid.levers", "args": {"values": {}}})
        raised = False
    except KeyError:
        raised = True
    assert raised
    (r,) = ad.execute([bad])
    assert r.ok is False and "engine_error" in (r.error or "")
    (o,) = r.observations
    assert o.kind is ObsKind.ERROR and o.caused_by == bad.action_id
    assert "posm" not in json.dumps(o.to_dict()) and "sign_switch" not in json.dumps(o.to_dict())
    assert ad.engine_errors == 1
    nxt = ad.observe()
    assert nxt.observations[0][0].caused_by == bad.action_id        # delivered at the head of the next batch
    (ts,) = ad.available_tools()[0]
    lever = next(iter(ts.parameters["properties"]["values"]["properties"]))
    (r2,) = ad.execute([Action(tool="rapid.levers", args={"kind": "levers", "values": {lever: 3.0}})])
    assert r2.ok and r2.error is None                               # the same world still executes actions
    ad.advance_time([1.0])
    assert ad.engine_errors == 1


def test_execution_cost_is_the_observed_charge_of_the_action():
    op = _integrity("attractive_ineffective")
    ad = op.adapter()
    ad.reset()
    acts = [Action(tool="shop.produce", args={"units": 3}), Action(tool="shop.set_price", args={"price": 700}),
            Action(tool="shop.status", args={}), Action(tool="expert.consult", args={}),
            Action(tool="shop.produce", args={"units": 99})]
    costs = [ad.execute([a])[0].cost for a in acts]
    assert costs == [600.0, 0.0, 0.0, 2000.0, 0.0]                 # rejected action: no charge
    assert observed_spend({"cost_cents": 5, "fee_cents": 2, "report": {"cost_cents": 900}, "cost_s": 30}) == 7.0
    assert observed_spend({"paid": -3, "charged": True}) == 0.0
    assert observed_spend(["cost_cents", 5]) == 0.0


def _oracle_original(world: IntegrityWorld) -> float:
    """The pre-repair definition (episode-0 parameters regardless of the episode), kept for the regression check."""
    best = None
    for price in range(100, 3001, 50):
        w = copy.deepcopy(world)
        w.reset()
        w.rng = random.Random(world.seed * 7919 + VARIANTS.index(world.variant))
        w.reset()
        w.execute("p:agent", {"tool": "shop.set_price", "args": {"price": price}})
        while not w.done:
            lam = max(0.0, (w.a - w.b * price)) * w.demand_scale
            want = max(0, min(20, int(round(lam)) - w.inventory + 1))
            if want and w.cash >= want * w.unit_cost:
                w.execute("p:agent", {"tool": "shop.produce", "args": {"units": want}})
            w.advance(TICK_S)
        v = w.verify()["outcome"]["value"]
        best = v if best is None or v > best else best
    return float(best)


def test_integrity_oracle_uses_the_current_episode_parameters():
    w = IntegrityWorld("easy_useless", 3, horizon=24)
    assert w.oracle_value() == _oracle_original(w)                 # episode 0: unchanged definition
    k0 = w.episode_key()
    w.reset()
    assert w.episode_key() != k0 and w.episode == 1
    # episode 1's oracle = the best fixed price on episode 1's demand curve: a copy built with those parameters
    probe = IntegrityWorld("easy_useless", 3, horizon=24)
    probe.reset()
    assert (probe.a, probe.b) == (w.a, w.b)
    assert w.oracle_value() == probe.oracle_value()
    assert w.oracle_value() != _oracle_original(w)                 # the old definition used episode 0's curve


def test_references_and_labels_follow_each_episode():
    op = _integrity("easy_useless", seed=1, compute_references=True)
    w = op.worlds[0]
    refs0 = dict(w.refs)
    ad = op.adapter()
    ad.reset()                                                      # starts episode 1: new demand parameters
    assert w.episode == 1 and w.refs != refs0
    probe = IntegrityWorld("easy_useless", 1)
    probe.reset()
    assert w.refs["oracle"] == probe.oracle_value()
    ad.execute([Action(tool="shop.produce", args={"units": 5})])
    ad.advance_time([w.env.horizon * TICK_S / 86400.0])
    assert op.resolved(0)
    ref1, value = dict(w.refs), w.env.verify()["outcome"]["value"]
    ad.reset()                                                      # episode 2 starts before the trainer drains
    labs = op.take_labels()
    assert len(labs) == 1 and op.take_labels() == []
    want = (value - ref1["null"]) / (ref1["oracle"] - ref1["null"])
    assert abs(labs[0].utility - want) < 1e-9
    assert labs[0].env_id == w.env_id and "integrity" not in labs[0].env_id


class _NoReset:
    """Unit-test double: a world without an in-place reset (like Crucible's kernel tier)."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, k):
        if k == "reset":
            raise AttributeError(k)
        return getattr(self._inner, k)


def test_rebuilt_world_is_what_the_actor_sees_after_reset():
    op = _integrity("easy_useless", compute_references=False)
    w = op.worlds[0]
    old = _NoReset(w.env)
    w.env, w.cell[0] = old, old
    ad = op.adapter()
    ad.execute([Action(tool="badge.claim", args={})])
    assert old._inner.badges == 1
    ad.reset()
    assert w.env is not old and w.cell[0] is w.env and w.episode == 1
    ad.execute([Action(tool="badge.claim", args={})])
    assert w.env.badges == 1 and old._inner.badges == 1
