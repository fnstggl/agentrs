"""Security of the open-world surfaces: isolation backends, delegation scope, institutional rules that delegates
cannot bypass, private minds, and sandboxed services cannot reach operator tools."""
import json

import pytest

import crucible

crucible.load_all_modules()
from crucible.core.kernel import World  # noqa: E402
from crucible.sandbox import backends  # noqa: E402
from crucible.sandbox.runner import SandboxConfig, Worker  # noqa: E402
from crucible.worldgen import generate  # noqa: E402

DAY = 86400


def test_backend_detection_is_honest_and_unavailable_backends_are_refused():
    d = backends.detect()
    assert d["landlock-seccomp"]["available"]
    for name in ("gvisor", "bubblewrap", "firecracker"):
        assert "available" in d[name] and d[name]["reason"]
        if not d[name]["available"]:
            with pytest.raises(backends.SandboxUnavailable):
                backends.require(name)
    assert backends.wrap("gvisor", ["python", "x"], "/w")[:2] == ["runsc", "--rootless"]
    assert "--unshare-all" in backends.wrap("bubblewrap", ["python"], "/w")


def test_worker_refuses_to_start_under_an_unavailable_backend(monkeypatch):
    if backends.detect()["gvisor"]["available"]:
        pytest.skip("gVisor present on this host")
    monkeypatch.setenv("CRUCIBLE_SANDBOX_BACKEND", "gvisor")
    with pytest.raises(Exception) as e:
        Worker({"m.py": "x = 1\n"}, SandboxConfig()).start()
    assert "unavailable" in str(e.value)


def _world():
    w = World.create(generate({"family": "open_digital", "overrides": {"n_consumers": 300}}, 4))
    w.system_call("admin.join", {"handle": "eve", "endowment": 3_000_000})
    ex = lambda t, **k: w.execute("p:eve", {"tool": t, "args": k})  # noqa: E731
    me = w.modules["bank"].primary_account(w.state, "p:eve")
    assert ex("id.create_org", handle="eveco", name="E", kind="company").ok
    assert ex("biz.register", org="o:eveco", account=me).ok
    acct = ex("bank.open_account", owner="o:eveco").result["account"]
    ex("bank.transfer", from_account=me, to=acct, amount=1_000_000)
    return w, ex, acct


def test_sandboxed_service_cannot_call_operator_tools_or_act_for_others():
    w, ex, acct = _world()
    src = ("import crucible_sdk as sdk\n"
           "def go(args, caller):\n"
           "    out = {}\n"
           "    for tool, a in [('admin.join', {'handle': 'x', 'endowment': 10**9}),\n"
           "                    ('bank.transfer', {'from_account': args['victim'], 'to': 'p:eve', 'amount': 100}),\n"
           "                    ('wf.trigger', {'tool': 'bank.transfer', 'every_s': 600, 'as_org': 'o:shorteco1'}),\n"
           "                    ('caps.install', {'capability': 'cap_000001', 'org': 'o:shorteco1'})]:\n"
           "        try:\n"
           "            sdk.call(tool, **a); out[tool] = 'ok'\n"
           "        except Exception as e:\n"
           "            out[tool] = getattr(e, 'code', 'err')\n"
           "    return out\n")
    repo = ex("code.create_repo", name="evil", owner="o:eveco").result["repo"]
    ex("code.commit", repo=repo, files={"crucible.json": json.dumps({"entry": "service.py", "endpoints": {"go": {"handler": "go"}}}),
                                        "service.py": src})
    inst = ex("cloud.rent", provider="o:cloud1", cores=1, account=acct, owner="o:eveco").result["instance"]
    assert ex("cloud.deploy", repo=repo, host="evilsvc", instance=inst).result["status"] == "live"
    victim = w.modules["bank"].primary_account(w.state, "o:shorteco1")
    r = ex("evilsvc.go", victim=victim)
    body = r.result["body"]
    assert all(v != "ok" for v in body.values()), body


def test_a_delegate_cannot_bypass_an_organisations_constitution():
    w, ex, acct = _world()
    w.system_call("admin.join", {"handle": "del", "endowment": 100_000})
    assert ex("id.grant", grantee="p:del", cap="bank.spend", owner="o:eveco", scope=acct).ok
    assert ex("gov.adopt", org="o:eveco", rules=[{"match": {"tool": "bank.transfer", "min_amount": 1000},
                                                  "require": {"type": "vote", "electorate": "members"}}]).ok
    r = w.execute("p:del", {"tool": "bank.transfer", "args": {"from_account": acct, "to": "p:del", "amount": 5000}})
    assert not r.ok and "constitution" in r.error["message"]


def test_minds_private_state_is_not_exposed_by_any_tool():
    w, ex, acct = _world()
    tools = w.available_tools("p:eve")
    assert not any(t["name"].startswith("society.") for t in tools)
    # reservation wages and beliefs are only in internal events
    for e in w.log:
        if e.type in ("society.add", "society.mind"):
            assert e.vis == "internal"


def test_dashboard_open_world_page_renders_census():
    from fastapi.testclient import TestClient

    from crucible.dashboard.app import make_app
    w = World.create(generate({"family": "open_industry", "overrides": {"n_shoppers": 2000}}, 2))
    w.advance_time(3 * DAY)
    c = TestClient(make_app(world=w))
    r = c.get(f"/w/{w.world_id}/open")
    assert r.status_code == 200 and "Census" in r.text
    j = c.get(f"/api/w/{w.world_id}/open").json()
    assert j["industry"]["runs"] >= 1 and "capabilities" in j
