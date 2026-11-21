"""Researcher views of the open world (dashboard pages, JSON mirrors and CLI subcommands): the synthetic internet,
society of minds, markets, infrastructure and active agents. All of them only read state."""
import json

import pytest

import crucible

crucible.load_all_modules()
from crucible.cli import main  # noqa: E402
from crucible.core.kernel import World  # noqa: E402
from crucible.worldgen import generate  # noqa: E402

DAY = 86400
WEB_CONFIG = {"family": "open_digital", "overrides": {"n_consumers": 200, "module_config": {"webgen": {"sites": 2}}}}
VIEWS = ("internet", "society", "markets", "infrastructure", "agents")


def _sites(w) -> dict[str, str]:
    """host -> title of every live web site (read through the net module)."""
    net = w.modules["net"]
    out = {}
    for h in w.state.ids("hosts"):
        _, dep = net.web_dep(w.state, h)
        if dep is not None:
            out[h] = dep["manifest"]["web"].get("title", "")
    return out


@pytest.fixture(scope="module")
def web_world():
    """open_digital world with two generated sites; an agent (alice) browses one of them; one day passes."""
    w = World.create(generate(WEB_CONFIG, 5))
    alice = w.system_call("admin.join", {"handle": "alice", "endowment": 1_000_000})["principal"]
    sites = _sites(w)
    assert len(sites) == 2
    first = sorted(sites)[0]
    assert w.execute(alice, {"tool": "net.fetch", "args": {"url": f"http://{first}.crucible/"}}).ok
    w.advance_time(w.now + DAY + 3600)
    yield w, sites, alice
    w.close()


@pytest.fixture(scope="module")
def client(web_world):
    from fastapi.testclient import TestClient

    from crucible.dashboard.app import make_app
    w, _, _ = web_world
    return TestClient(make_app(world=w))


def test_internet_page_lists_sites_pages_and_traffic(web_world, client):
    w, sites, _ = web_world
    r = client.get(f"/w/{w.world_id}/internet")
    assert r.status_code == 200 and "Web sites" in r.text and "Link graph" in r.text and "<svg" in r.text
    for host, title in sites.items():
        assert host in r.text and title in r.text
    j = client.get(f"/api/w/{w.world_id}/internet").json()
    assert j["available"] and {s["host"] for s in j["sites"]} == set(sites)
    assert j["totals"]["live_sites"] == 2 and j["totals"]["pages"] == w.state.count("web_pages") > 0
    for s in j["sites"]:
        assert s["title"] == sites[s["host"]] and s["live"] and s["provider"] in w.state.ids("providers")
        assert s["pages"] >= 1 and s["records"] == len(w.state.find_ids("app_docs", "ns", s["host"])) > 0
        assert s["requests_last_day"].get("ok", 0) > 0                     # crawler (and alice) requests
    assert {p["host"] for p in j["pages"]} <= set(sites)


def test_society_markets_and_agents_pages(web_world, client):
    w, _, alice = web_world
    st = w.state
    r = client.get(f"/w/{w.world_id}/society")
    assert r.status_code == 200 and "Firms" in r.text and "Trust graph" in r.text
    j = client.get(f"/api/w/{w.world_id}/society").json()
    assert j["available"] and j["counts"]["firm"] == len(st.find_ids("minds", "kind", "firm")) > 0
    firm = j["firms"][0]
    assert firm["id"] in r.text and all("kind" in g for g in firm["goals"])

    r = client.get(f"/w/{w.world_id}/markets")
    assert r.status_code == 200 and "Problem requests" in r.text and "Capabilities" in r.text
    j = client.get(f"/api/w/{w.world_id}/markets").json()
    assert j["totals"]["requests"] == st.count("ops_requests") and len(j["requests"]) == st.count("ops_requests")
    assert j["retail"] == [] and j["offers"] == []                         # no industry in this family

    r = client.get(f"/w/{w.world_id}/agents")
    assert r.status_code == 200 and alice in r.text and "net.fetch" in r.text
    j = client.get(f"/api/w/{w.world_id}/agents").json()
    assert [a["id"] for a in j["agents"]] == [alice]                      # NPC persons are excluded
    assert j["agents"][0]["actions"] == 1 and j["agents"][0]["last_tool"] == "net.fetch"
    assert j["recent"][-1]["actor"] == alice and j["recent"][-1]["ok"]


def test_overview_and_open_pages_link_to_the_views(web_world, client):
    w, _, _ = web_world
    for path in ("", "/open"):
        r = client.get(f"/w/{w.world_id}{path}")
        assert r.status_code == 200
        for v in VIEWS:
            assert f'href="/w/{w.world_id}/{v}"' in r.text


def test_views_never_change_the_world(web_world, client):
    w, _, _ = web_world
    seq, digest = w.seq, w.digest()
    for v in VIEWS:
        assert client.get(f"/w/{w.world_id}/{v}").status_code == 200
        assert client.get(f"/api/w/{w.world_id}/{v}").status_code == 200
    assert client.post(f"/w/{w.world_id}/internet").status_code == 405
    assert w.seq == seq and w.digest() == digest and not w.check_invariants()


def test_infrastructure_page_on_an_industrial_world():
    from fastapi.testclient import TestClient

    from crucible.dashboard.app import make_app
    w = World.create(generate({"family": "open_industry", "overrides": {"n_shoppers": 2000}}, 2))
    w.advance_time(w.now + DAY + 3600)
    st = w.state
    c = TestClient(make_app(world=w))
    r = c.get(f"/w/{w.world_id}/infrastructure")
    assert r.status_code == 200 and "Compute providers" in r.text and "Energy grids" in r.text
    assert "Freight links" in r.text
    for k in st.ids("regions") + st.ids("links") + st.ids("providers"):
        assert k in r.text
    j = c.get(f"/api/w/{w.world_id}/infrastructure").json()
    assert {g["region"] for g in j["grids"]} == set(st.ids("regions"))
    assert {f["link"] for f in j["freight"]} == set(st.ids("links"))
    for p in j["providers"]:
        used = sum(i["cores"] for _, i in st.scan("instances") if i["provider"] == p["id"] and i["status"] == "running")
        assert p["cores_used"] == used and p["utilisation"] == round(used / p["cores"], 3)
    assert any(g["used_yesterday"] > 0 for g in j["grids"])               # production ran on day 0
    m = c.get(f"/api/w/{w.world_id}/markets").json()
    assert len(m["retail"]) == st.count("retail") > 0 and len(m["offers"]) == st.count("offers") > 0
    r = c.get(f"/w/{w.world_id}/markets")
    assert r.status_code == 200 and "Retail listings" in r.text and "B2B offers" in r.text
    w.close()


def test_views_tolerate_worlds_without_the_modules():
    from fastapi.testclient import TestClient

    from crucible.dashboard.app import make_app
    w = World.create(generate({"family": "tiny"}, 1))
    c = TestClient(make_app(world=w))
    for v in VIEWS:
        assert c.get(f"/w/{w.world_id}/{v}").status_code == 200, v
        assert c.get(f"/api/w/{w.world_id}/{v}").status_code == 200, v
    assert "no synthetic internet" in c.get(f"/w/{w.world_id}/internet").text
    assert "no simulated minds" in c.get(f"/w/{w.world_id}/society").text
    assert c.get(f"/api/w/{w.world_id}/internet").json()["available"] is False
    assert c.get(f"/api/w/{w.world_id}/society").json()["available"] is False
    w.close()


def test_svg_graph_draws_nodes_and_edges():
    from crucible.dashboard.app import svg_graph
    g = str(svg_graph([("a", 3), ("b", 1), ("c", 1)], [("a", "b", 2), ("b", "c", 1), ("c", "c", 1), ("x", "a", 1)]))
    assert g.count("<circle") == 3 and g.count("<line") == 2               # self-loops and unknown nodes dropped
    assert "<title>a -> b: 2</title>" in g and "marker-end" in g
    assert "no data" in str(svg_graph([], []))


# ====================================================================== CLI
def _cli(capsys, *args):
    assert main([*args, "--json"]) == 0, args
    return json.loads(capsys.readouterr().out)


def test_cli_open_world_views(tmp_path, capsys):
    from crucible.api.local import WorldServer
    store = f"sqlite:///{tmp_path}/views.db"
    with WorldServer(store, lease_ttl_s=None) as srv:
        wid = srv.create_world(WEB_CONFIG, 5, world_id="web1")
        srv.advance_time(wid, until=srv.world(wid).now + DAY + 3600)
        sites = _sites(srv.world(wid))
        n_req = srv.world(wid).state.count("ops_requests")
        srv.snapshot_world(wid)
        srv.create_world({"family": "tiny"}, 1, world_id="tiny1")
    S = ("--store", store)
    d = _cli(capsys, "internet", "web1", *S)
    assert {s["host"] for s in d["sites"]} == set(sites) and d["totals"]["pages"] > 0
    assert main(["internet", "web1", *S]) == 0                             # text tables
    out = capsys.readouterr().out
    assert "sites:" in out and all(h in out and t in out for h, t in sites.items())
    soc = _cli(capsys, "society", "web1", *S)
    assert soc["available"] and soc["counts"]["firm"] > 0
    assert main(["society", "web1", *S]) == 0 and "firms:" in capsys.readouterr().out
    mk = _cli(capsys, "markets", "web1", *S)
    assert mk["totals"]["requests"] == n_req
    assert main(["markets", "web1", *S]) == 0 and "problem requests" in capsys.readouterr().out
    inf = _cli(capsys, "infrastructure", "web1", *S)
    assert len(inf["providers"]) >= 1
    cen = _cli(capsys, "census", "web1", *S)
    assert cen["events"] > 0 and "capabilities" in cen and "problems" in cen
    assert main(["census", "web1", *S]) == 0 and json.loads(capsys.readouterr().out)["day"] == 1   # JSON either way
    assert _cli(capsys, "internet", "tiny1", *S)["available"] is False
    assert _cli(capsys, "society", "tiny1", *S)["available"] is False
    assert main(["internet", "tiny1", *S]) == 0 and "no synthetic internet" in capsys.readouterr().out
