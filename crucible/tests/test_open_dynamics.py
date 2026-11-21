"""Endogenous dynamics and structurally different world families."""
import pytest

import crucible

crucible.load_all_modules()
from crucible.agents.base import Runner  # noqa: E402
from crucible.agents.inventors import CapacityInventor  # noqa: E402
from crucible.core.kernel import World  # noqa: E402
from crucible.worldgen import FAMILIES, families, generate  # noqa: E402

DAY = 86400
OPEN = ["open_digital", "open_industry", "open_mixed", "open_scarce_energy", "open_fragile_suppliers",
        "open_weak_institutions", "open_perishable"]


def test_open_families_have_partitions_with_held_out_mechanisms():
    assert set(OPEN) <= set(FAMILIES)
    test = set(families("test"))
    assert {"open_weak_institutions", "open_perishable"} <= test
    # the held-out mechanisms appear only in test families
    for f in families("train") + families("val"):
        spec = generate(f, 0)
        assert spec.params.get("public_reputation", True) is not False, f
        ind = next((m for m in spec.modules if m["name"] == "industry"), None)
        assert not (ind and ind["config"].get("decay")), f


@pytest.mark.parametrize("fam", OPEN)
def test_every_open_family_runs_and_keeps_invariants(fam):
    w = World.create(generate({"family": fam, "overrides": {"n_consumers": 600, "n_shoppers": 3000}}, 7))
    w.advance_time(4 * DAY)
    assert w.check_invariants() == [] and w.modules["bank"].check_ledger_consistency(w) == []


def test_families_differ_causally_not_cosmetically():
    out = {}
    for fam in ("open_industry", "open_scarce_energy", "open_perishable"):
        w = World.create(generate({"family": fam, "overrides": {"n_shoppers": 6000}}, 3))
        w.advance_time(12 * DAY)
        st = w.state
        losses = {}
        for _, r in st.scan("ops_days"):
            for k, v in r["loss_n"].items():
                losses[k] = losses.get(k, 0) + v
        out[fam] = losses
    # perishability destroys stock; scarce energy refuses runs (curtailment) - different mechanisms, not parameters
    assert out["open_perishable"].get("spoilage", 0) > 0 and out["open_industry"].get("spoilage", 0) == 0
    assert out["open_scarce_energy"].get("energy_curtailment", 0) >= out["open_industry"].get("energy_curtailment", 0)


def test_providers_reprice_with_utilisation():
    w = World.create(generate({"family": "open_digital", "overrides": {"n_consumers": 300}}, 2))
    st = w.state
    prov = sorted(st.ids("providers"))[0]
    p0 = st.tables["providers"][prov]["price"]
    # an agent rents most of the provider's capacity: utilisation > 85% for several days
    w.system_call("admin.join", {"handle": "hog", "endowment": 50_000_000})
    me = w.modules["bank"].primary_account(st, "p:hog")
    free = st.tables["providers"][prov]["cores"] - sum(i["cores"] for i in st.find("instances", "provider", prov)
                                                        if i["status"] == "running")
    for _ in range(max(0, free - 2) // 8):
        assert w.execute("p:hog", {"tool": "cloud.rent", "args": {"provider": prov, "cores": 8, "account": me}}).ok
    w.advance_time(w.now + 4 * DAY)
    assert st.tables["providers"][prov]["price"] > p0
    assert any(o["provider"] == prov for _, o in st.scan("hw_orders"))       # and it invested in capacity


@pytest.mark.slow
def test_entrepreneurs_enter_an_emergent_market_by_forking_an_open_source_invention():
    w = World.create(generate({"family": "open_digital", "overrides": {"firms_per_need": 3, "workers": 20}}, 1))
    w.system_call("admin.join", {"handle": "ada", "endowment": 1_000_000})
    inv = CapacityInventor(open_source=True)
    Runner(w, [("p:ada", inv)], dt=3600, record=False).run(24 * 40)
    st = w.state
    copies = [c for _, c in st.scan("capabilities") if c["improves"] == inv.capability]
    cats = w.modules["caps"].categories(w, 1)
    assert any(len(c["independent_buyers"]) >= 2 for c in cats), cats
    founded = [m for _, m in st.scan("minds") if m["kind"] == "person" and m["routines"].get("entrepreneur", {}).get("started")]
    assert founded, "no entrepreneur looked"
    assert copies, [m["routines"]["entrepreneur"] for m in founded]
