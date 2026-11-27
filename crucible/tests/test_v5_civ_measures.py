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
