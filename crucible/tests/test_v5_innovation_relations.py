"""Relation providers of the one evaluator-side relation graph (`crucible.innovation.relations`) over the compute
providers' capacity and the freight network (wave 3b, SP-6.4b): edges and observables are read off world state, and
the generic scarcity rule labels needs from them without naming any entity."""
import crucible

crucible.load_all_modules()
from crucible.core.kernel import World, WorldSpec  # noqa: E402
from crucible.innovation import reference as REF  # noqa: E402
from crucible.innovation import relations as R  # noqa: E402
from crucible.innovation.worlds import innovation_spec  # noqa: E402
from crucible.worldgen import generate  # noqa: E402

DAY = 86400


def test_compute_market_utilisation_and_refusals_are_read_off_the_cloud():
    sp = innovation_spec(3, with_adopters=False)
    # with the operations ledger, a refused rental is recorded as the renter's loss (capacity unavailable)
    w = World.create(WorldSpec(sp.family, sp.seed, list(sp.modules) + [{"name": "ops", "config": {}}], sp.params))
    try:
        REF.join(w, "kim", 50_000_000)
        acct = w.modules["bank"].primary_account(w.state, "p:kim")
        prov = w.query("p:kim", "cloud.providers", {})[0]
        E = [e for e in R.graph(w) if e["provider"] == "compute_market"]
        node = f"compute-market/{prov['provider']}"
        assert {"src": f"capacity/{prov['provider']}", "rel": "constrains", "dst": node} in \
            [{k: e[k] for k in ("src", "rel", "dst")} for e in E]
        assert R.observe(w, E)[node]["utilisation"] == (prov["cores_total"] - prov["cores_free"]) / prov["cores_total"]
        r = w.execute("p:kim", {"tool": "cloud.rent", "args": {"provider": prov["provider"], "cores": prov["cores_free"],
                                                               "account": acct}})
        assert r.ok, r.error
        refused = w.execute("p:kim", {"tool": "cloud.rent", "args": {"provider": prov["provider"], "cores": 1,
                                                                     "account": acct}})
        assert not refused.ok
        o = R.observe(w, [e for e in R.graph(w) if e["provider"] == "compute_market"])[node]
        assert o["utilisation"] == 1.0 and 0 < o["refused_share"] < 1
    finally:
        w.close()


def test_freight_links_are_read_off_the_network_and_congestion_labels_a_transport_need():
    w = World.create(generate({"family": "open_industry", "overrides": {}}, 1))
    w.advance_time(8 * DAY)
    st = w.state
    E = [e for e in R.graph(w) if e["provider"] == "freight"]
    links = sorted(st.tables["links"])
    assert sorted(e["dst"].split("/", 1)[1] for e in E if e["rel"] == "loads") == links
    obs = R.observe(w, E)
    today = w.now // DAY
    for lid in links:
        peak = max((st.get("link_days", f"{lid}|{d}") or {}).get("used", 0) for d in range(today - 2, today + 1))
        assert abs(obs[f"transport/{lid}"]["utilisation"] - peak / st.tables["links"][lid]["capacity"]) < 1e-9
    payers = {s["payer"] for _, s in st.scan("shipments")}
    loaded = [lid for lid in links if obs[f"carrier/{lid}"]["holders"]]
    assert loaded and all(set(obs[f"carrier/{lid}"]["holders"]) & payers for lid in loaded)
    # the generic scarcity rule, at a lower scarcity threshold, labels access between the loaded regions
    busy = max(links, key=lambda lid: obs[f"transport/{lid}"]["utilisation"])
    th = obs[f"transport/{busy}"]["utilisation"] * 0.99
    needs = [n for n in R.label(w, {"scarce": th}) if n["path"][0]["provider"] == "freight"]
    a, b = st.tables["links"][busy]["a"], st.tables["links"][busy]["b"]
    hit = [n for n in needs if n["target"] == f"access/{a}-{b}"]
    assert hit and hit[0]["rule"] == "scarcity" and len(hit[0]["holders"]) >= 2
    assert not [n for n in R.label(w) if n["path"][0]["provider"] == "freight" and
                obs[n["path"][0]["dst"]]["utilisation"] < R.THRESHOLDS["scarce"] and
                obs[n["path"][0]["dst"]]["refused_share"] < R.THRESHOLDS["refused"]]


def test_coordination_overhead_of_organisations_is_read_off_the_coupling():
    """Provider `orgs`: an organisation's coordination requirement carries the share of operator hours the
    civilisation's org_coordination coupling takes (1 - coordination_factor), for its members."""
    from crucible.civ import couplings as CP
    from crucible.innovation import inventworld as IW
    w = IW.generate("balanced", 3)
    try:
        st = w.state
        E = [e for e in R.graph(w) if e["provider"] == "orgs"]
        assert E
        for e in E:
            org = e["src"].split(":", 1)[1]
            o = e["attrs"]["obs"]
            assert e["rel"] == "requires" and e["dst"] == "capability/coordination"
            assert abs(o["time_share"] - (1 - CP.coordination_factor(w, st, org))) < 1e-4
            assert o["operators"] == CP.headcount(w, st, org) and org in o["holders"]
        # below the generic bottleneck threshold nothing is labelled; at a lower one the same rule labels it
        assert not [n for n in R.label(w) if n["path"][0]["provider"] == "orgs"]
        low = min(e["attrs"]["obs"]["time_share"] for e in E)
        assert [n for n in R.label(w, {"bottleneck": low}) if n["path"][0]["provider"] == "orgs"]
    finally:
        w.close()
