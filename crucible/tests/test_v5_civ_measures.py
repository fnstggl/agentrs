"""crucible.civ.experiments: evaluator-side stage measures of the chain experiments, on a small CPU world.

`m_agent_compute` (the computing-infrastructure stage of the agent-shop chains) must count the compute a deployment
served to requests from outside its operator's side and leave out the operator's own requests (`cloud.usage_rel`).
"""
import json
from types import SimpleNamespace

import crucible

crucible.load_all_modules()

from crucible.civ import experiments as X  # noqa: E402
from crucible.core.kernel import World  # noqa: E402
from crucible.worldgen import generate  # noqa: E402

APP = r'''
import crucible_sdk as sdk

def index(req, caller):
    rows = sdk.call("db.query", collection="hits", limit=5)
    return {"html": "<!doctype html><html><head><title>shop</title></head><body><p>%d</p></body></html>" % len(rows)}
'''
MANIFEST = json.dumps({"entry": "service.py", "web": {"title": "shop", "rate_per_min": 120,
                                                       "routes": {"GET /": "index"}}})


def _ex(w, who):
    return lambda t, **k: w.execute(who, {"tool": t, "args": k})  # noqa: E731


def test_agent_compute_counts_requests_from_others_and_not_the_operators_own():
    w = World.create(generate({"family": "open_digital", "overrides": {"n_consumers": 50}}, 1))
    w.system_call("admin.join", {"handle": "shopper", "endowment": 3_000_000, "password": "pw"})
    w.system_call("admin.join", {"handle": "visitor", "endowment": 500_000, "password": "pw"})
    op, vis = _ex(w, "p:shopper"), _ex(w, "p:visitor")
    acct = w.modules["bank"].primary_account(w.state, "p:shopper")
    repo = op("code.create_repo", name="refshop").result["repo"]
    assert op("code.commit", repo=repo, files={"crucible.json": MANIFEST, "service.py": APP}).ok
    provs = w.query("p:shopper", "cloud.providers")
    inst = op("cloud.rent", provider=sorted(p["provider"] for p in provs)[0], cores=1, account=acct).result
    assert op("cloud.deploy", repo=repo, host="refshop", instance=inst["instance"]).result["status"] == "live"
    ctx = {"agent": SimpleNamespace(handle="refshop"), "seq0": w.seq}
    assert X.m_agent_compute(w, 0, w.now, ctx) == 0.0
    for _ in range(3):
        assert vis("net.fetch", url="http://refshop.crucible/").result["status"] == 200
    after_visitor = X.m_agent_compute(w, 0, w.now, ctx)
    assert op("net.fetch", url="http://refshop.crucible/").result["status"] == 200      # the operator's own request
    assert X.m_agent_compute(w, 0, w.now, ctx) == after_visitor
    usage = [e for e in w.events_range(ctx["seq0"]) if e.type == "cloud.usage"]
    rel = [e for e in w.events_range(ctx["seq0"]) if e.type == "cloud.usage_rel"]
    assert len(usage) == 4 and len(rel) == 1
    assert after_visitor > 0 and abs(after_visitor - sum(e.payload["cu"] for e in usage[:3])) < 1e-9
    # another host's requests never count
    assert X.m_agent_compute(w, 0, w.now, {"agent": SimpleNamespace(handle="nosuch"), "seq0": ctx["seq0"]}) == 0.0


def test_analyze_reports_failed_process_runs_of_the_arms_and_keeps_the_every_seed_sign_rule():
    """Arms run non-strict: their failed process runs are summed into the result. A seed with no effect (zero
    difference) breaks the every-seed sign rule: the effect is not confirmed although the interval excludes zero."""
    from crucible.civ import counterfactual as CF
    e = X.EXPERIMENTS["agent_shop"]

    def per(effects):
        out = []
        for s, eff in enumerate(effects):
            arms = {a: {m: 100.0 + s for m in e.measures} for a in CF._arms(e)}
            for a in e.interventions:
                arms[a] = {m: 100.0 + s - eff for m in e.measures}
            out.append({"seed": s, "arms": arms, "footprint": {a: {"placebo_1": 1, "placebo_5": 5}.get(a, 0)
                                                               for a in arms},
                        "invariants": {a: [] for a in arms}, "proc_failed": {a: int(a == "factual" and s == 0)
                                                                             for a in arms},
                        "wall_s": {a: 1.0 for a in arms}, "warmup_wall_s": 1.0, "chains": []})
        return out
    r = CF.analyze(e, per([5.0, 6.0, 7.0, 5.0]))
    assert r["proc_failed"] == 1 and r["effects"]["no_agent"]["agent_compute"]["confirmed"]
    r0 = CF.analyze(e, per([5.0, 6.0, 0.0, 7.0, 5.0, 6.0]))
    v = r0["effects"]["no_agent"]["agent_compute"]
    assert v["ci95"][1] < 0 and not v["sign_consistent"] and not v["confirmed"]


def test_trace_keeps_the_root_transactions_own_events_in_the_root_stage():
    """A command that records events in two modules (a forum post mirrored into the content catalogue) is ONE
    action: its events form the root stage; only events of later transactions reached through provenance are later
    stages (a mirror was counted as a scripted stage after a scripted agent's root at a54f53d)."""
    from crucible.civ import chains as CH
    ev = lambda seq, typ, actor="p:refshop": SimpleNamespace(seq=seq, t=seq, type=typ, actor=actor, payload={})  # noqa: E731
    pv = CH.Provenance()
    rows = [(1, "cmd", None), (2, "forum.post", "search_information"), (3, "content.published", "social_media"),
            (4, "cmd", None), (5, "content.view", "social_media"), (6, "bank.txn", "finance")]
    for seq, typ, sub in rows:
        pv.events[seq] = ev(seq, typ, "p1" if seq >= 4 else "p:refshop")
        pv.subsystem[seq] = sub
    for seq in (2, 3):
        pv.roots[seq] = 1
    for seq in (5, 6):
        pv.roots[seq] = 4
    for a, b in ((1, 2), (1, 3), (2, 3), (3, 5), (4, 5), (5, 6)):
        pv.link(a, b)
    stages = CH.trace(None, 2, pv=pv)
    assert [s["subsystem"] for s in stages] == ["search_information", "social_media", "finance"]
    assert stages[0]["n_events"] == 2 and stages[1]["types"] == {"content.view": 1}


def test_failed_requests_counts_web_pages_refused_during_a_provider_outage():
    """O.5 category 4: `m_failed_requests` must see a web request the network refuses because the site's provider is
    down (`net.hit` unavailable) as well as a refused API call (`cloud.usage` unavailable). At a54f53d it counted the
    API path only and read 0 in every arm and seed of the grid-cut experiment while datacenter draws were refused."""
    import crucible.evaluation.longrun  # noqa: F401 - registers `disturb` (the operator's outage tool)
    spec = generate({"family": "open_digital", "overrides": {"n_consumers": 50}}, 1)
    spec.modules.append({"name": "disturb", "config": {}})
    w = World.create(spec)
    w.system_call("admin.join", {"handle": "shopper", "endowment": 3_000_000, "password": "pw"})
    w.system_call("admin.join", {"handle": "visitor", "endowment": 500_000, "password": "pw"})
    op, vis = _ex(w, "p:shopper"), _ex(w, "p:visitor")
    acct = w.modules["bank"].primary_account(w.state, "p:shopper")
    repo = op("code.create_repo", name="refshop").result["repo"]
    assert op("code.commit", repo=repo, files={"crucible.json": MANIFEST, "service.py": APP}).ok
    prov = sorted(p["provider"] for p in w.query("p:shopper", "cloud.providers"))[0]
    inst = op("cloud.rent", provider=prov, cores=1, account=acct).result
    assert op("cloud.deploy", repo=repo, host="refshop", instance=inst["instance"]).result["status"] == "live"
    ctx = {"seq0": w.seq}
    assert vis("net.fetch", url="http://refshop.crucible/").result["status"] == 200
    assert X.m_failed_requests(w, 0, w.now, ctx) == 0.0
    w.system_call("admin.shock_outage", {"provider": prov, "hours": 2})
    for _ in range(3):
        assert vis("net.fetch", url="http://refshop.crucible/").result["status"] == 503
    hits = [e for e in w.events_range(ctx["seq0"]) if e.type == "net.hit" and e.payload["outcome"] == "unavailable"]
    assert len(hits) == 3
    assert X.m_failed_requests(w, 0, w.now, ctx) == 3.0
    assert X.m_curtailed_hours(w, 0, w.now, ctx) == 0.0       # an operator outage is not a grid curtailment
