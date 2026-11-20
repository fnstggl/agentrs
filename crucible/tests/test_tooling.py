"""Researcher tooling: CLI, dashboard, trajectory export, league / self-play, counterfactual forks, emergence
metrics and causal tracing - end to end on small real worlds."""
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np
import pytest

import crucible

crucible.load_all_modules()
from crucible.agents.base import Runner  # noqa: E402
from crucible.agents.scripted import Builder, Composer  # noqa: E402
from crucible.analytics import metrics_export, trace  # noqa: E402
from crucible.cli import main  # noqa: E402
from crucible.core.kernel import World  # noqa: E402
from crucible.evaluation import counterfactual as CF  # noqa: E402
from crucible.evaluation import emergence as EM  # noqa: E402
from crucible.storage.postgres import open_store  # noqa: E402
from crucible.training import trajectories as TJ  # noqa: E402
from crucible.training.league import League, MatchRecord, parse_spec, policy_from_spec, trueskill_update  # noqa: E402
from crucible.worldgen import generate  # noqa: E402

DAY = 86400
SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


def cli(*args, capsys=None) -> dict | str:
    """Run the CLI in-process with --json and return the parsed output."""
    rc = main([*args, "--json"])
    assert rc == 0, args
    out = capsys.readouterr().out
    return json.loads(out)


# ====================================================================== shared worlds
def _world(name, seed, agents, hours):
    spec = generate({"family": "tiny", "overrides": {"params": {"cost_model": "deterministic"}}}, seed)
    w = World.create(spec, name)
    pairs = [(w.system_call("admin.join", {"handle": h, "endowment": 1_000_000})["principal"], pol)
             for h, pol in agents]
    run = Runner(w, pairs, dt=3600, seed=1)
    run.run(hours)
    return w, run


@pytest.fixture(scope="module")
def builder_world():
    """Tiny deterministic world with one scripted Builder (alice) on route.shortest, 1.25 days."""
    w, run = _world("tooling-b", 3, [("alice", Builder(need="route.shortest"))], 30)
    yield w, run
    w.close()


@pytest.fixture(scope="module")
def market_world():
    """Tiny deterministic world: a Builder (alice) and a Composer (bob, calls the incumbent from its own code)."""
    w, run = _world("tooling-m", 3, [("alice", Builder(need="route.shortest")),
                                     ("bob", Composer(need="route.shortest"))], 30)
    yield w, run
    w.close()


# ====================================================================== CLI
def test_cli_end_to_end(tmp_path, capsys):
    store = f"sqlite:///{tmp_path}/c.db"
    S = ["--store", store]
    info = cli("world", "create", "--family", "tiny", "--seed", "2", "--id", "w1", "--deterministic", *S,
               capsys=capsys)
    assert info["world_id"] == "w1" and info["seq"] > 0
    j = cli("actor", "join", "w1", "alice", "--endowment", "1000000", *S, capsys=capsys)
    assert j["principal"] == "p:alice" and j["token"].startswith("cru_")
    traj = str(tmp_path / "alice.jsonl")
    r = cli("actor", "run", "w1", "alice", "--policy", "builder:need=route.shortest", "--days", "0.75",
            "--trajectory", traj, *S, capsys=capsys)
    assert r["actions"] > 10 and r["rejected"] == 0 and r["trajectory"]["records"] == 18
    r = cli("world", "run", "w1", "--days", "2", "--snapshot-every-days", "1", "--quiet", *S, capsys=capsys)
    assert r["days"] == 2 and not r["interrupted"] and len(r["snapshots"]) == 2
    snap = cli("world", "snapshot", "w1", *S, capsys=capsys)["snapshot"]
    f = cli("world", "fork", snap, "--id", "w1f", *S, capsys=capsys)
    assert f["world_id"] == "w1f" and f["parent"]["world"] == "w1"
    cli("world", "run", "w1f", "--days", "0.25", "--quiet", *S, capsys=capsys)
    for wid in ("w1", "w1f"):
        rp = cli("world", "replay", wid, "--verify", *S, capsys=capsys)
        assert rp["ok"] and rp["digest_match"] and rp["hash_match"], rp
        v = cli("world", "verify", wid, *S, capsys=capsys)
        assert v["ok"] and v["regen_hash_match"] and not v["invariant_violations"], v
    ls = cli("world", "list", *S, capsys=capsys)
    assert {x["world_id"] for x in ls} == {"w1", "w1f"}
    ins = cli("world", "inspect", "w1", *S, capsys=capsys)
    assert any(o["id"] == "o:alice-labs" and o["hosts"] for o in ins["orgs"])
    host = next(o["hosts"][0] for o in ins["orgs"] if o["id"] == "o:alice-labs")
    assert any(s["host"] == host and s["customers"] > 0 for s in ins["services"])
    assert ins["recent_events"] and not ins["invariant_violations"]
    t = cli("trace", "w1", "o:alice-labs", *S, capsys=capsys)
    assert t["kind"] == "org" and "o:alice-labs" in t["headline"] and t["timeline"]
    t = cli("trace", "w1", host, *S, capsys=capsys)
    assert t["kind"] == "host" and t["facts"]["shares"][host] > 0
    sale = next(x["seq"] for x in reversed(t["timeline"]) if "sale" in x["text"])
    ch = cli("trace", "w1", f"#{sale}", *S, capsys=capsys)
    assert {l["relation"] for l in ch["links"]} >= {"transaction", "references"}
    # experiment + report
    out = str(tmp_path / "exp.json")
    e = cli("experiment", "run", "--goal", "revenue", "--world", "tiny", "--level", "0", "--seeds", "5",
            "--policies", "idle,builder:need=route.shortest", "--max-days", "1", "--out", out, capsys=capsys)
    assert e["policies"]["idle"]["progress"][0] == 0.0
    assert e["policies"]["builder:need=route.shortest"]["progress"][0] > 0.0
    assert e["curves"]["builder:need=route.shortest"][0]["points"][-1][1] > 0
    assert main(["report", out]) == 0
    md = capsys.readouterr().out
    assert "| idle |" in md and "builder:need=route.shortest - idle" in md
    # dashboard on the same store (read-only)
    from fastapi.testclient import TestClient

    from crucible.dashboard.app import make_app
    st = open_store(store)
    before = {w: st.last_seq(w) for w in ("w1", "w1f")}
    app = make_app(store_url=store, experiments=[out], trajectories=[traj])
    with TestClient(app) as c:
        for p in ["/", "/w/w1", "/w/w1/actors", "/w/w1/orgs", "/w/w1/economy?kind=sale", "/w/w1/comms",
                  "/w/w1/services", "/w/w1/events", f"/w/w1/events?entity={host}", "/w/w1/trajectories",
                  "/w/w1f/lineage", "/w/w1/evaluation?emergence=1", "/w/w1/trace", "/w/w1/trace/o:alice-labs",
                  f"/w/w1/trace/{host}", f"/w/w1/chain/{sale}", "/experiments", "/experiments/0"]:
            r = c.get(p)
            assert r.status_code == 200, (p, r.text[:500])
        assert "alice" in c.get("/w/w1/trajectories").text and "<svg" in c.get("/experiments/0").text
        assert "crucible_customers{" in c.get("/w/w1/metrics").text
        assert c.get("/api/w/w1/overview").json()["metrics"]["seq"] == before["w1"]
        assert c.get("/api/w/w1f/lineage").json()["lineage"][1]["id"] == "w1"
        assert c.post("/w/w1").status_code == 405
        dig = app.state.worlds.get("w1").digest()
    assert {w: st.last_seq(w) for w in ("w1", "w1f")} == before          # nothing was written
    fresh = World.load(st, "w1")
    assert fresh.digest() == dig
    fresh.close()
    st.close()


def test_world_run_stops_cleanly_on_sigint_and_resumes(tmp_path):
    store = f"sqlite:///{tmp_path}/s.db"
    assert main(["world", "create", "--family", "tiny", "--seed", "4", "--id", "ws", "--store", store]) == 0
    env = dict(os.environ, PYTHONPATH=SRC)
    p = subprocess.Popen([sys.executable, "-m", "crucible.cli", "world", "run", "ws", "--days", "60",
                          "--store", store, "--json"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         env=env)
    try:
        deadline = time.time() + 60
        line = ""
        while time.time() < deadline:      # wait for the first daily progress line
            line = p.stdout.readline()
            if line.startswith("[crucible]") or not line:
                break
        assert line.startswith("[crucible]"), p.stderr.read()
        p.send_signal(signal.SIGINT)
        out, err = p.communicate(timeout=60)
    finally:
        if p.poll() is None:
            p.kill()
    assert p.returncode == 0, err
    res = json.loads(out[out.index("{"):])
    assert res["interrupted"] and 0 < res["days"] < 60 and res["snapshots"]
    st = open_store(store)
    w = World.load(st, "ws")                      # resume: latest snapshot + replay of the tail
    assert w.seq == res["seq"] and w.now == res["to"] and not w.check_invariants()
    w.close()
    st.close()
    assert main(["world", "run", "ws", "--days", "0.5", "--quiet", "--store", store]) == 0


# ====================================================================== trajectories
def test_trajectory_export_round_trip(tmp_path):
    spec = generate({"family": "tiny", "overrides": {"params": {"cost_model": "deterministic"}}}, 6)
    w = World.create(spec, "traj-w")
    a = w.system_call("admin.join", {"handle": "tess", "endowment": 1_000_000})["principal"]
    run = TJ.RecordingRunner(w, [(a, Builder(need="route.shortest"))], dt=3600, seed=0)
    run.run(14)
    recs = TJ.step_records(run)
    assert len(recs) == 14 and all(r["tools"] and r["seq_after"] >= r["seq_before"] for r in recs)
    acted = [r for r in recs if r["action"]]
    assert acted and all(r["tool"] in r["tools"] for r in acted)
    assert recs[0]["world_id"] == "traj-w" and recs[0]["spec_hash"] == spec.hash()
    j, z = str(tmp_path / "t.jsonl"), str(tmp_path / "t.npz")
    assert TJ.export_jsonl(run, j) == 14
    TJ.export_npz(run, z)
    assert TJ.read_jsonl(j) == recs == TJ.load_npz(z)
    cols = TJ.columns(z)
    assert cols["reward"].shape == (14,) and set(np.unique(cols["ok"])) <= {-1, 0, 1}
    ds = TJ.ImitationDataset(j)
    assert len(ds) == len(acted)
    prompt, target = ds[0]
    assert "you are p:tess" in prompt and "tools:" in prompt
    assert json.loads(target) == {"tool": acted[0]["tool"], "args": acted[0]["args"]}
    assert ds.to_jsonl(str(tmp_path / "sft.jsonl")) == len(acted)
    ev = TJ.event_records(w)
    TJ.save_npz(ev, str(tmp_path / "e.npz"), kind="events")
    back = TJ.load_npz(str(tmp_path / "e.npz"))
    assert len(back) == w.seq and back[-1]["hash"] == w.last_hash and back == ev
    mine = TJ.event_records(w, visible_to=a)
    ps = w.principals_of(a)          # the actor and the organisations it administers
    assert mine and len(mine) < len(ev) and all(e["vis"] == "public" or set(e["vis"]) & ps for e in mine)
    w.close()


# ====================================================================== league
def _rec(i, a, b, s, heldout=False):
    return MatchRecord(f"m{i}", "tiny", 1, "h", "competition", 1.0, [a, b], {a: "p:seat1", b: "p:seat2"},
                       {p: {"revenue": 0, "customers": 0, "share": 0.0} for p in (a, b)}, {a: s, b: 1 - s},
                       [[a, b, s]], heldout)


def test_league_sampling_ratings_and_leakage():
    lg = League(seed=3)
    for p in ("learner", "easy", "hard", "mid"):
        lg.add_policy(p, spec="idle")
    lg.add_policy("hold", spec="idle", heldout=True)
    lg.add_policy("child", spec="idle", parent="learner")
    assert lg.entries["child"].generation == 1 and lg.lineage("child") == ["child", "learner"]
    for i in range(6):
        lg.records.append(_rec(i, "learner", "easy", 1.0))
        lg.records.append(_rec(100 + i, "learner", "hard", 0.0))
    lg.records.append(_rec(999, "learner", "hold", 1.0, heldout=True))
    assert lg.win_rate("learner", "easy") > 0.8 > 0.2 > lg.win_rate("learner", "hard")
    assert lg.win_rate("learner", "hold") == 0.5               # held-out results never leak into training stats
    picks = [x for _ in range(300) for x in lg.sample_opponents("learner", 1, "pfsp")]
    assert "hold" not in picks and "learner" not in picks
    assert picks.count("hard") > 3 * picks.count("easy")
    rec = [x for _ in range(300) for x in lg.sample_opponents("learner", 1, "recent")]
    assert rec.count("child") > rec.count("easy")
    uni = [x for _ in range(400) for x in lg.sample_opponents("learner", 2, "uniform")]
    assert set(uni) == {"easy", "hard", "mid", "child"}
    with pytest.raises(ValueError):
        lg.play(["learner", "hold"], 1)                         # held-out policies never play training matches
    (m1, s1), (m2, s2) = trueskill_update((25.0, 8.3), (25.0, 8.3), 1)
    assert m1 > 25 > m2 and s1 < 8.3
    (d1, _), (d2, _) = trueskill_update((30.0, 2.0), (20.0, 2.0), 0.5)
    assert d1 < 30 and d2 > 20                                  # a draw pulls ratings together
    for r in lg.records:
        if not r.heldout_eval:
            lg._update_ratings(r)
    assert lg.entries["hard"].elo > lg.entries["learner"].elo > lg.entries["easy"].elo
    g = lg.generation_compare(0, 1)
    assert g["games"] == 0 and g["elo_b"] is not None
    d = League.load(lg.to_dict())
    assert d.win_rate("learner", "hard") == lg.win_rate("learner", "hard") and d.entries["hold"].heldout
    assert parse_spec("builder:need=route.shortest:undercut=0.8:advertise=false") == \
        ("builder", {"need": "route.shortest", "undercut": 0.8, "advertise": False})
    assert policy_from_spec("builder:undercut=0.7").undercut == 0.7


def test_selfplay_short_real_run(tmp_path, capsys):
    out = str(tmp_path / "league.json")
    a, b = "builder:need=route.shortest:undercut=0.9", "builder:need=route.shortest:undercut=0.6:advertise=0"
    assert main(["selfplay", "run", "--world", "tiny", "--seeds", "1", "--days", "1", "--policies", f"{a},{b}",
                 "--measured-cost", "--out", out]) == 0
    txt = capsys.readouterr().out
    assert "| A | B | games" in txt and "Elo" in txt
    d = json.load(open(out))
    assert len(d["records"]) == 2
    r1, r2 = d["records"]
    assert r1["seed"] == r2["seed"] and r1["spec_hash"] == r2["spec_hash"]          # paired worlds
    assert r1["seats"][a] == r2["seats"][b] == "p:seat1"                             # both seatings
    for r in d["records"]:
        for pid in (a, b):
            o = r["outcomes"][pid]
            assert o["orgs"] and o["hosts"] and o["customers"] > 0 and o["revenue"] > 0
            assert o["needs"] == ["route.shortest"]
    m = d["matchups"][0]
    assert m["games"] == 2 and m["a_wins"] + m["b_wins"] + m["draws"] == 2
    assert sum(e["games"] for e in d["entries"]) == 4
    assert main(["report", out]) == 0 and "mu-3sigma" in capsys.readouterr().out


# ====================================================================== counterfactual / emergence / trace
def test_counterfactual_removing_builder_host_shifts_customers_to_incumbents(builder_world, market_world):
    w, _ = builder_world
    seq0, dig0 = w.seq, w.digest()
    host = "alice-shorte"
    assert w.modules["consumers"].market_shares(w).get(host, 0) > 0
    r = CF.fork_compare(w, CF.undeploy_host(host), DAY // 2, check_determinism=True)
    assert r["exact"] and r["determinism_check"] is True
    assert w.seq == seq0 and w.digest() == dig0                                    # source world untouched
    rb, rt = r["baseline"]["revenue"], r["treatment"]["revenue"]
    assert rb["o:alice-labs"] > 0 and rt["o:alice-labs"] < rb["o:alice-labs"]
    cb, ct = r["baseline"]["customers"], r["treatment"]["customers"]
    assert ct[host] < cb[host]
    assert ct["shorte1"] > cb["shorte1"]                                           # the incumbent gains
    assert r["delta"]["customers"][host] < 0 and r["delta"]["revenue"]["o:alice-labs"] < 0
    assert r["applied"][0]["tool"] == "cloud.undeploy"
    assert "revenue o:alice-labs" in r["text"]
    # an upstream removal hurts the composer that depends on it (service calls service)
    r2 = CF.fork_compare(market_world[0], CF.undeploy_host("shorte1"), DAY // 4)
    dep = r2["dependents"]["shorte1"]
    assert "svc:bob-mesh" in dep and dep["svc:bob-mesh"]["delta_ok"] < 0


def test_emergence_metrics_on_short_run(market_world):
    w, _ = market_world
    rep = EM.emergence_report(w)
    deps = rep["dependencies"]
    assert any(e["src"] == "host:bob-mesh" and e["dst"] == "host:shorte1" and e["kind"] == "calls" and e["n"] > 0
               for e in deps["edges"])
    assert deps["max_service_chain"] >= 1
    assert any(e["kind"] == "deploys" and e["src"] == "host:alice-shorte" for e in deps["edges"])
    tr = rep["tool_reuse"]
    assert set(EM.agent_hosts(w)) == {"alice-shorte", "bob-mesh"} and tr["n_agent_hosts"] == 2
    curves = rep["diffusion"]["curves"]
    assert curves["alice-shorte"][-1][1] > 0 and rep["diffusion"]["stats"]["alice-shorte"]["peak"] > 0
    conc = rep["concentration"]
    assert 0 < conc["current"]["route.shortest"] < 1
    assert all(0 <= x["hhi"] <= 1 for x in conc["series"]["route.shortest"])
    sd = rep["strategy_diversity"]
    assert set(sd["agents"]) == {"p:alice", "p:bob"} and sd["n_clusters"] >= 1 and 0 <= sd["mean_pairwise_js"] <= 1
    assert "deployed_api" not in sd["agents"]["p:alice"]["histogram"]
    sp = rep["specialization"]
    assert sp["orgs"]["o:alice-labs"]["categories"] == {"api:route.shortest": 1}
    assert sp["summary"]["orgs_with_offerings"] >= 4
    nov = EM.useful_novelty(w, horizon_s=DAY // 4, hosts=["alice-shorte"])
    assert nov["n_candidates"] == 1 and nov["hosts"]["alice-shorte"]["reductions"]["households_served"] > 0
    assert nov["hosts"]["alice-shorte"]["useful"]


def test_trace_explains_service_failure_after_unpaid_bills(builder_world):
    w, _ = builder_world
    f = World.from_blob(w.snapshot_blob(), world_id="trace-fork")
    try:
        # step scheduled jobs one at a time; right after the hourly consumer market has paid the company, an
        # operator intervention empties its account, so the next compute bill fails
        last = None
        while last != "consumers.hour":
            f.advance_time(f.next_job_time(), max_jobs=1)
            last = next(e for e in reversed(f.log) if e.type == "proc").payload["proc"]
        bank = f.modules["bank"]
        acct = bank.primary_account(f.state, "o:alice-labs")
        f.system_call("bank.transfer", {"from_account": acct, "to": bank.primary_account(f.state, "p:alice"),
                                        "amount": bank.balance(f.state, acct), "memo": "drain"},
                      as_principal="o:alice-labs")
        f.advance_time(f.now + DAY)
        ex = trace.explain(f, "o:alice-labs", since=w.seq)
        kinds = {x["kind"] for x in ex["findings"]}
        assert "instance_suspended" in kinds and "failures" in kinds, ex["text"]
        assert "risks" in ex["headline"]
        h = trace.explain(f, "alice-shorte", since=w.seq)
        assert h["kind"] == "host" and any(x["kind"] == "reliability" and x["severity"] == 3 for x in h["findings"])
        tl = trace.timeline(f, "o:alice-labs", since=w.seq)
        assert any(r["type"] == "cloud.instance_status" for r in tl)
        sus = next(r["seq"] for r in tl if r["type"] == "cloud.instance_status")
        chain = trace.causal_chain(f, sus)
        assert any(l["relation"] == "transaction" and "cloud.hourly" in l["text"] for l in chain["links"])
        m = metrics_export.world_metrics(f)
        assert m["money_total"] == 0 and m["instances"].get("suspended", 0) >= 1
        prom = metrics_export.prometheus_text(f, m)
        assert 'crucible_instances{world="trace-fork",status="suspended"}' in prom
    finally:
        f.close()


def test_dashboard_hides_private_communication_unless_operator():
    from fastapi.testclient import TestClient

    from crucible.dashboard.app import make_app
    w = World.create(generate({"family": "tiny"}, 9), "priv-w")
    for h in ("ann", "ben"):
        w.system_call("admin.join", {"handle": h, "endowment": 10_000})
    r = w.execute("p:ann", {"tool": "mail.send", "args": {"to": ["ben"], "subject": "TOPSECRET-123",
                                                          "body": "the plan TOPSECRET-123"}})
    assert r.ok
    w.advance_time(w.now + 3600)
    seq, dig = w.seq, w.digest()
    with TestClient(make_app(world=w)) as c:
        for p in ("/w/priv-w/events", "/w/priv-w/events?type=mail", "/w/priv-w/events?type=cmd", "/w/priv-w/comms",
                  "/w/priv-w/trace/p:ann", "/api/w/priv-w/events?type=mail", "/api/w/priv-w/explain/p:ann"):
            t = c.get(p).text
            assert "TOPSECRET" not in t, p
        counts = c.get("/api/w/priv-w/comms").json()["counts"]
        assert any(x["type"] == "mail.sent" and x["visibility"] == "private" for x in counts)
    with TestClient(make_app(world=w, operator=True)) as c:
        assert "TOPSECRET-123" in c.get("/w/priv-w/events?type=mail").text
    assert w.seq == seq and w.digest() == dig
    w.close()
    with pytest.raises(SystemExit):           # no authentication: loopback only unless explicitly allowed
        main(["dashboard", "priv-w", "--host", "0.0.0.0", "--store", "memory://"])


def test_cli_help_runs_as_module():
    out = subprocess.run([sys.executable, "-m", "crucible.cli", "--help"], capture_output=True, text=True,
                         env=dict(os.environ, PYTHONPATH=SRC), timeout=60)
    assert out.returncode == 0
    for cmd in ("world", "actor", "experiment", "selfplay", "evaluate", "trace", "report", "dashboard", "serve"):
        assert cmd in out.stdout
