"""The datacenter fleet repairs (domains.infra_world; SP-9.7, DEF-I-4)."""
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
