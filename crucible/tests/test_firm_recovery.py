"""Firms whose service lost its instance restart it if they can pay for a week of compute, or wind up (found in a
120-day run where dead services sat on open listings for months)."""
import crucible
from crucible.agents.base import Runner
from crucible.core.kernel import World
from crucible.worldgen import generate

crucible.load_all_modules()
DAY = 86400


def _kill(w, host):
    st = w.state
    dep = st.tables["deployments"][st.tables["hosts"][host]["deployment"]]
    cloud = w.modules["cloud"]
    w._run_system("test-terminate", lambda ctx: cloud._terminate(ctx, dep["instance"], "test"))
    assert st.tables["deployments"][st.tables["hosts"][host]["deployment"]]["status"] == "stopped"


def test_a_solvent_firm_restarts_its_terminated_service():
    w = World.create(generate({"family": "open_digital", "overrides": {"n_consumers": 400}}, 3))
    st = w.state
    firm = sorted(m["org"] for _, m in st.scan("minds") if m["kind"] == "firm" and m["role"] == "service")[0]
    host = sorted(st.find_ids("hosts", "owner", firm))[0]
    _kill(w, host)
    Runner(w, [], dt=3600, record=False).run(26)
    dep = st.tables["deployments"][st.tables["hosts"][host]["deployment"]]
    assert dep["status"] == "live" and st.tables["instances"][dep["instance"]]["status"] == "running"
    assert World.replay(w.spec, [e.to_dict() for e in w.log]).digest() == w.digest()


def test_an_insolvent_firm_winds_up_instead_of_serving_nothing():
    w = World.create(generate({"family": "open_digital", "overrides": {"n_consumers": 400}}, 3))
    st = w.state
    firm = sorted(m["org"] for _, m in st.scan("minds") if m["kind"] == "firm" and m["role"] == "service")[0]
    host = sorted(st.find_ids("hosts", "owner", firm))[0]
    bank = w.modules["bank"]
    acct = bank.primary_account(st, firm)
    w._run_system("test-drain", lambda ctx: bank.move(ctx, acct, "acct_genesis", bank.balance(st, acct), "test", "drain"))
    _kill(w, host)
    Runner(w, [], dt=3600, record=False).run(26)
    assert st.tables["companies"][firm]["status"] != "registered"


def test_recovery_can_be_ablated():
    w = World.create(generate({"family": "open_digital", "overrides": {"n_consumers": 400,
                                                                       "params": {"ablate_recovery": True}}}, 3))
    st = w.state
    firm = sorted(m["org"] for _, m in st.scan("minds") if m["kind"] == "firm" and m["role"] == "service")[0]
    host = sorted(st.find_ids("hosts", "owner", firm))[0]
    _kill(w, host)
    Runner(w, [], dt=3600, record=False).run(26)
    assert st.tables["deployments"][st.tables["hosts"][host]["deployment"]]["status"] == "stopped"


def test_demand_survives_the_highest_slot_offerings_closing():
    """Regression: closing the offerings with the highest consumer slots crashed the hourly demand process
    (IndexError) for the rest of the run, so no firm was served at all."""
    w = World.create(generate({"family": "open_tight_compute"}, 1))
    st = w.state
    Runner(w, [], dt=3600, record=False).run(30)
    slots = sorted((o["slot"], lid) for lid, o in st.scan("cons_offers"))
    top = [lid for _, lid in slots[-2:]]
    for lid in top:
        seller = st.tables["listings"][lid]["seller"]
        owner = next(m["person"] for m in st.find("members", "org", seller) if m["role"] == "owner")
        assert w.execute(owner, {"tool": "biz.dissolve", "args": {"org": seller}}).ok
    Runner(w, [], dt=3600, record=False).run(30)
    assert not [e for e in w.log if e.type == "proc.failed" and e.payload["proc"] == "consumers.hour"]
    day = w.now // DAY - 1
    assert sum(r["ok"] for k, r in st.scan("svc_days") if k.endswith(f"|{day}")) > 0
