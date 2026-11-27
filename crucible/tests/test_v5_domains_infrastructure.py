"""The datacenter fleet repairs (domains.infra_world; SP-9.7, DEF-I-4)."""
import numpy as np

import crucible
from crucible.core.kernel import World, WorldSpec
from crucible.domains import infra_world as F

crucible.load_all_modules()


def _world(seed=3, pop=True, **fleet):
    mods = [{"name": m, "config": {}} for m in ["identity", "bank", "admin", "artifacts", "cols", "space", "code",
                                                 "docs"]]
    if pop:
        mods.append({"name": "pop", "config": {"profile": "development", "n": 300,
                                               "classes": {"adaptive": 0, "strategic": 0, "semantic": 0}}})
        mods.append({"name": "elsewhere", "config": {}})
    mods.append({"name": "fleet", "config": dict(fleet)})
    return World.create(WorldSpec("infra-test", seed, mods, {"cost_model": "deterministic"}))


def test_cost_structure_has_no_one_call_staffing_shortcut():
    w = _world()
    w.advance_time(14 * 86400)
    c = F.operator_costs(w, 0, w.now)
    assert c["wage"] < 0.12 * c["total"] and c["hardware"] > 0.5 * c["total"]
    assert F.reliability(w, 0, w.now // 3600)["attainment"] > 0.995


def test_experiment_replays_recorded_traffic_only():
    """fleet.experiment replays the operator's own recorded hours (offered requests are exactly what was recorded),
    never the hidden demand process or the hidden CPU cost (audit I S.3: the hidden cpu_ms was exfiltrated)."""
    w = _world()
    w.advance_time(3 * 86400)
    st = w.state
    conf = st.get("fleet", "conf")
    dep = st.get("fleet", "deployed")
    res = w.execute(conf["lead"], {"tool": "fleet.experiment",
                                   "args": {"repo": dep["repo"], "target": dep["target"], "hours": 24}})
    assert res.ok, res
    h1 = w.now // 3600
    recorded = sum(float(v) * 3600.0 for h in range(h1 - 24, h1) for d in st.get("fleet_hours", f"h{h:06d}")["rps"].values()
                   for v in d.values())
    out = res.result["deployed"]
    assert out["hours_replayed"] == 24
    assert abs(out["offered"] - recorded) <= 1e-6 * recorded
    src = open(F.__file__).read()
    body = src[src.index("    def simulate(self"):src.index("    def _sim_obs(self")]
    assert "self.regime(" not in body and "demand_rps(" not in body


def test_population_users_carry_the_tenants_traffic():
    w = _world()
    w.advance_time(5 * 86400)
    uf = w.modules["fleet"].user_factor(w.state)
    assert any(abs(v - 1.0) > 1e-6 for d in uf.values() for v in d.values())
    assert any(e.type == "cols.batch" and e.payload["kind"] == "fleet.use" for e in w.log)
    w0 = _world(pop=False)
    assert all(v == 1.0 for d in w0.modules["fleet"].user_factor(w0.state).values() for v in d.values())


def test_batch_jobs_are_simulated_in_production_and_in_the_replay():
    """Batch worlds simulate every hour (before, every fleet.hour with an open job failed with KeyError
    'core_hours_left' in production and in fleet.experiment: two of the five 9.7 configurations never ran)."""
    w = _world(batch=True, n_sites=2, n_regions=3, topology="mesh")
    w.advance_time(2 * 86400 + 3 * 3600)               # the day's jobs (submitted at midnight) are open
    assert not [e for e in w.log if e.type == "proc.failed"]
    hours = [e.payload for e in w.log if e.type == "fleet.hour"]
    assert len(hours) == 51 and sum(sum(h["exact"]["batch"].values()) for h in hours) > 0
    assert F.open_jobs(w.state)
    conf = w.state.get("fleet", "conf")
    dep = w.state.get("fleet", "deployed")
    res = w.execute(conf["lead"], {"tool": "fleet.experiment",
                                   "args": {"repo": dep["repo"], "target": dep["target"], "hours": 24}})
    assert res.ok, res
    assert res.result["deployed"]["hours_replayed"] == 24 and res.result["deployed"]["batch_core_h"] > 0


def test_a_failed_machine_goes_back_only_at_the_lessors_refurbishment_charge():
    """The lease returns machines in working order (DEF-I-4 reopened: fire the crew and hand failed machines back
    passed 9.7 on 7 of 7 instances): without include_failed a failed machine cannot be returned; with it the lessor
    bills its refurbishment (parts + contractor labour, more than an in-house repair) and the 9.7 evaluator still
    counts the machine as failure debt."""
    from crucible.domains import engineering_infrastructure as E
    w = _world()
    st = w.state
    conf = st.get("fleet", "conf")
    for _ in range(20):
        w.advance_time(w.now + 86400)
        if (st.arrays["fl_status"] == F.FAILED).any():
            break
    status = st.arrays["fl_status"]
    assert (status == F.FAILED).any() and not (status == F.OFF).any()          # always-on: nothing is off
    site = conf["sites"][int(st.arrays["fl_site"][int(np.flatnonzero(status == F.FAILED)[0])])]["id"]
    res = w.execute(conf["lead"], {"tool": "fleet.retire", "args": {"site": site, "n": 1}})
    assert not res.ok and "include_failed" in str(res.error)
    bank = w.modules["bank"]
    cash0 = bank.balance(st, conf["account"])
    debt0 = E._backlog(w)
    res = w.execute(conf["lead"], {"tool": "fleet.retire", "args": {"site": site, "n": 1, "include_failed": True}})
    assert res.ok and res.result["returned_broken"] == 1
    fee = conf["refurbish_cents"]
    assert res.result["refurbish_cents"] == fee and bank.balance(st, conf["account"]) == cash0 - fee
    assert fee > conf["repair_parts_cents"] + conf["tech_wage_cents_day"] * 6 // 8      # more than in-house
    assert E._backlog(w) == debt0                       # the machine left the fleet but stays failure debt


def test_the_replay_starts_with_the_recorded_day_before_it():
    """The controller's forecast reads the same hours of the day before: the replay hands it the recorded day
    before the replayed hours, so the incumbent replayed on its own hours reproduces their attainment."""
    w = _world()
    w.advance_time(3 * 86400)
    st = w.state
    conf, dep = st.get("fleet", "conf"), st.get("fleet", "deployed")
    res = w.execute(conf["lead"], {"tool": "fleet.experiment",
                                   "args": {"repo": dep["repo"], "target": dep["target"], "hours": 48}})
    assert res.ok, res
    h1 = w.now // 3600
    rows = [st.get("fleet_hours", f"h{h:06d}") for h in range(h1 - 48, h1)]
    good = sum(v["good"] for r in rows for v in r["exact"]["services"].values())
    off = sum(v["offered"] for r in rows for v in r["exact"]["services"].values())
    assert abs(res.result["deployed"]["attainment"] - good / off) < 0.002
