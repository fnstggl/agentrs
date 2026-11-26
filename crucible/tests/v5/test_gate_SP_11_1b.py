"""SP-11.1b: several learners act in one world through the interface in every tier; exact structured observations on
the Gymnasium, PettingZoo, HTTP and MCP surfaces; one unit system (seconds, cost_s, action provenance) in every
adapter.

* concurrent learners: `crucible.experience.loop.run_learners` drives one `ReferenceLearner` per agent of one world
  (Tier 1: the rapid team family `team_commons`, joint ticks; Tier 2: a `crucible.domains` configuration; Tier 3: a
  `crucible.civ` civilisation), each learner on its own observation, every observation validated;
* exact observations: on each surface a number the world holds exactly (an integer balance in cents, the clock in
  seconds, a facility reading) is read back from the surface's own structured payload and compared with ``==``;
* units: every adapter reports ``resources.time_unit == "s"``, an integer ``t`` equal to the environment clock,
  integer ``cost_s`` and ``action_seq`` in every action result, and integer ``delay_s`` in every event cause.

A tier that cannot be built is recorded with its error and the test fails; nothing is skipped."""
import json

import numpy as np
import pytest

from crucible.acceptance.v5.evidence import record
from crucible.experience import validate_observation
from crucible.experience.loop import run_learners
from crucible.experience.reference import ReferenceLearner
from crucible.experience.registry import make

pytestmark = pytest.mark.timeout(5400)                # concurrent learners in every Part 9 domain
CMD = "PYTHONPATH=src python -m pytest tests/v5/test_gate_SP_11_1b.py"
HOUR = 3600
CIV = "balanced"


def _attempt(fn):
    try:
        return fn(), None
    except Exception as e:  # noqa: BLE001 - recorded and asserted on
        return None, f"{type(e).__name__}: {str(e)[:400]}"


def _tier2_names():
    from crucible.experience.probes_tiers import tier2_names
    return tier2_names()


def _concurrent(env, steps: int, think_s: int | None) -> dict:
    learners = {a: ReferenceLearner(seed=10 + k, reward_mode=env.reward_mode) for k, a in enumerate(env.agents())}
    out = run_learners(env, learners, steps, think_s=think_s, update_every=max(4, steps // 2))
    per = out["agents"]
    acting = [a for a, r in per.items() if r["actions"] >= 1]
    fails = sum(r["schema_failures"] for r in per.values())
    return {"learners": len(per), "acting": len(acting), "schema_failures": fails,
            "actions": sum(r["actions"] for r in per.values()), "world_changes": int(out.get("world_changes", 0))}


def _units(env) -> list[str]:
    """Unit violations of one adapter after a few actions and some time."""
    bad = []
    a = env.agents()[0]
    tools = env.tools(a)
    ro = [t for t in tools if (t.get("annotations") or {}).get("readOnlyHint")]
    acts = [{"tool": ro[0]["name"], "args": {}}] if ro else []
    acts.append({"tool": "exp.register", "args": {"hypothesis": "time passes", "measure": (
        {"number": "world.time_s"} if getattr(env, "tier", 0) == 1 else
        {"tool": "bank.accounts", "where": {"owner": "$self"}, "field": "balance", "aggregate": "sum"}),
        "prediction": {"op": "change", "min_change": -1}, "deadline_in_s": 2 * int(getattr(env, "tick_s", HOUR))}})
    n_ok = 0
    for act in acts:
        r = env.execute(a, act)
        if not isinstance(r.get("cost_s"), int) or isinstance(r.get("cost_s"), bool):
            bad.append(f"{act['tool']}: cost_s {r.get('cost_s')!r}")
        if r.get("ok"):                       # a carried-out action is numbered (what its consequences name)
            n_ok += 1
            if not isinstance(r.get("action_seq"), int):
                bad.append(f"{act['tool']}: action_seq {r.get('action_seq')!r}")
    if not n_ok:
        bad.append("no action was carried out (the unit check proves nothing)")
    for _ in range(3):
        env.advance(int(getattr(env, "tick_s", HOUR)))
    o = env.observe(a)
    if o["resources"].get("time_unit") != "s":
        bad.append(f"time_unit {o['resources'].get('time_unit')!r}")
    if not isinstance(o["t"], int) or o["t"] != int(env.now):
        bad.append(f"t {o['t']!r} vs now {env.now!r}")
    evs = env.observe_since(a, 0)[0]
    for e in evs:
        c = e.get("cause")
        if not isinstance(e.get("t"), int) or e["t"] > env.now:
            bad.append(f"event {e.get('type')} t {e.get('t')!r}")
            break
        if c is not None and (not isinstance(c.get("delay_s"), int) or not isinstance(c.get("action_seq"), int)):
            bad.append(f"event {e.get('type')} cause {c}")
            break
    if not any(e.get("cause") for e in evs):
        bad.append("no event carries action provenance")
    return bad


def _gym_exact() -> bool:
    from crucible.experience.gym import ExperienceGymEnv
    g = ExperienceGymEnv(1, "mech:scm_additive", 3, validate=False)
    obs, info = g.reset()
    keys = list(info["numbers_keys"])
    ok = True
    for _ in range(3):
        obs, r, term, trunc, info = g.step(None)
        keys = list(info.get("numbers_keys") or keys)
        js = json.loads(obs["json"])
        for i, k in enumerate(keys):
            if obs["numbers_mask"][i]:
                v = js["numbers"][k]
                ok &= float(obs["numbers"][i]) == float(v) and obs["numbers"].dtype == np.float64
        ok &= float(obs["t"][0]) == float(js["t"]) == float(g.env.now)
    bal = g.env.observe(g.agent)["resources"]["balance_cents"]
    ok &= isinstance(bal, int) and bal == g.env._balance(g.agent)
    return bool(ok)


def _pz_exact() -> bool:
    from crucible.experience.gym import ExperienceParallelEnv
    pz = ExperienceParallelEnv(1, "rapid:team_commons", 3)
    obs, infos = pz.reset()
    ok = bool(obs)
    acts = {a: None for a in pz.agents}
    for _ in range(3):
        obs, rew, term, trunc, infos = pz.step(acts)
        for a, o in obs.items():
            js = json.loads(o["json"])
            keys = pz.slots[a].keys
            for i, k in enumerate(keys):
                if o["numbers_mask"][i]:
                    ok &= float(o["numbers"][i]) == float(js["numbers"][k])
            ok &= float(o["t"][0]) == float(js["t"])
            ok &= all(v == 0.0 for v in rew.values())
    return bool(ok)


def _http_exact() -> bool:
    from fastapi.testclient import TestClient
    from crucible.api.http_server import make_app
    from crucible.api.local import WorldServer
    srv = WorldServer("memory://")
    op = {"Authorization": "Bearer op-sp111b"}
    with TestClient(make_app(srv, operator_token="op-sp111b")) as c:
        info = c.post("/v1/worlds", json={"spec": {"family": "tiny", "overrides": {"n_consumers": 30}}, "seed": 3},
                      headers=op).json()
        wid = info["world_id"]
        ann = c.post(f"/v1/worlds/{wid}/join", json={"handle": "ann", "endowment": 123_457}, headers=op).json()
        bob = c.post(f"/v1/worlds/{wid}/join", json={"handle": "bob", "endowment": 1000}, headers=op).json()
        me = {"Authorization": f"Bearer {ann['token']}"}
        r = c.post(f"/v1/worlds/{wid}/act", json={"tool": "bank.transfer", "args": {
            "from_account": ann["account"], "to": bob["principal"], "amount": 1234, "memo": "exact"}}, headers=me)
        ok = r.status_code == 200 and r.json().get("ok")
        o = c.get(f"/v1/worlds/{wid}/observe", headers=me).json()
        w = srv.world(wid)
        true_bal = w.state.get("accounts", ann["account"])["balance"]
        text = json.dumps(o)
        ok &= true_bal == 123_457 - 1234 and str(true_bal) in text
        accts = c.post(f"/v1/worlds/{wid}/act", json={"tool": "bank.accounts", "args": {}}, headers=me).json()
        got = [x["balance"] for x in accts["result"] if x["account"] == ann["account"]]
        ok &= got == [true_bal] and isinstance(got[0], int)
        ok &= o.get("time") == w.now if "time" in o else True
    srv.close()
    return bool(ok)


def _mcp_exact() -> bool:
    from crucible.api.local import WorldServer
    from crucible.api.mcp_server import MCPHandler
    srv = WorldServer("memory://")
    wid = srv.create_world({"family": "tiny", "overrides": {"n_consumers": 30}}, 3)
    ann = srv.join_world(wid, "ann", endowment=98_765)
    bob = srv.join_world(wid, "bob", endowment=10)
    h = MCPHandler(srv, wid, ann)
    resp, _ = h.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
        "name": "bank.transfer", "arguments": {"from_account": ann.account, "to": bob.principal, "amount": 765,
                                               "memo": "exact"}}})
    ok = not resp["result"]["isError"]
    resp, _ = h.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                        "params": {"name": "bank.accounts", "arguments": {}}})
    sc = resp["result"]["structuredContent"]
    got = [x["balance"] for x in sc["result"] if x["account"] == ann.account]
    true_bal = srv.world(wid).state.get("accounts", ann.account)["balance"]
    ok &= got == [true_bal] == [98_765 - 765] and isinstance(got[0], int)
    resp, _ = h.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                        "params": {"name": "crucible.observe", "arguments": {}}})
    ok &= not resp["result"]["isError"] and resp["result"]["structuredContent"]["time"] == srv.world(wid).now
    srv.close()
    return bool(ok)


def test_SP_11_1b_concurrent_learners_exact_surfaces_one_unit_system():
    conc, errors = {}, {}
    for t, fn in ((1, lambda: {"rapid:team_commons": _concurrent(make(1, "rapid:team_commons", 4), 30, None)}),
                  (2, lambda: {n: _concurrent(make(2, n, 4, agents=["ann", "bob"]), 20, HOUR)
                               for n in _tier2_names()}),
                  (3, lambda: {CIV: _concurrent(make(3, CIV, 4, agents=["ann", "bob", "cat"]), 20, HOUR)})):
        r, err = _attempt(fn)
        if err:
            errors[t] = err
        else:
            conc[t] = r
    tiers_ok = [t for t, r in conc.items() if all(x["acting"] >= 2 and x["schema_failures"] == 0
                                                 for x in r.values())]
    surfaces = {}
    for name, fn in (("gymnasium", _gym_exact), ("pettingzoo", _pz_exact), ("http", _http_exact),
                     ("mcp", _mcp_exact)):
        r, err = _attempt(fn)
        surfaces[name] = bool(r) if err is None else err
    units, unit_errors = {}, {}
    for name, mk in (("tier1 mech", lambda: make(1, "mech:scm_additive", 5, validate=False)),
                     ("tier1 rapid", lambda: make(1, "rapid:net_routing", 5)),
                     ("tier1 rapid team", lambda: make(1, "rapid:team_commons", 5)),
                     ("tier0 kernel", lambda: make(0, "tiny", 5, agents=["learner"])),
                     ("tier2 domain", lambda: make(2, _tier2_names()[0], 5)),
                     ("tier3 civ", lambda: make(3, CIV, 5, agents=["learner"]))):
        env, err = _attempt(mk)
        if err:
            unit_errors[name] = err
            continue
        units[name] = _units(env)
    inconsistent = sum(1 for v in units.values() if v)
    metrics = {"tiers_with_concurrent_learners": len(tiers_ok),
               "surfaces_with_exact_observations": sum(1 for v in surfaces.values() if v is True),
               "adapters_with_inconsistent_units": inconsistent, "adapters_unit_checked": len(units)}
    record("SP-11.1b", metrics, command=CMD, notes=f"concurrent learners per tier: {conc}; surfaces (True = exact): "
           f"{surfaces}; unit violations per adapter: {units}; tiers / adapters that could not be built: "
           f"{errors} {unit_errors}")
    assert metrics["surfaces_with_exact_observations"] >= 4, surfaces
    assert inconsistent == 0, units
    assert len(tiers_ok) == 3 and not unit_errors, f"concurrent learners in tiers {tiers_ok}; {errors} {unit_errors}"
