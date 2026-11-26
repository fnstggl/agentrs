"""SP-16 (workstream O): security boundaries - default isolation from the public internet, the real-world adapter
disabled by default and latched shut in every world-hosting process, and consent, moderation and abuse prevention.

Measured on the real system: a fresh interpreter for the adapter defaults, the world-hosting entry points run as
subprocesses, the network probes of the 5.0 red team (process guard over v5 worlds, OS-level jails of the
subprocess tools, sandbox sockets), and a social world for blocking, rate limits, moderation and fake engagement."""
import ast
import json
import os
import subprocess
import sys

import pytest

from crucible.acceptance.v5.evidence import record

pytestmark = [pytest.mark.slow, pytest.mark.timeout(1800)]
CMD = "PYTHONPATH=src python -m pytest tests/v5/test_gate_SP_16.py"
SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src")
ENTRY_POINTS = ("crucible.cli", "crucible.api.http_server", "crucible.api.mcp_server", "crucible.experience.__main__",
                "crucible.web.app")


def _py(code: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "CRUCIBLE_HOSTS_WORLDS"}
    env["PYTHONPATH"] = SRC
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, env=env)


def _defaults() -> dict:
    code = ("import json\nfrom crucible.realworld import RealWorldBridge, BridgeRefused\nb = RealWorldBridge()\n"
            "out = {'enabled': b.enabled}\n"
            "for m, args in (('get', ('https://example.org/',)), ('request', ('POST', 'https://example.org/'))):\n"
            "    try:\n        getattr(b, m)(*args)\n        out[m] = 'performed'\n"
            "    except BridgeRefused as e:\n        out[m] = 'refused'\n"
            "print(json.dumps(out))\n")
    r = _py(code)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def _latch(module: str) -> str:
    code = (f"import importlib\nm = importlib.import_module({module!r})\n"
            "try:\n    m.main(['--help'])\nexcept SystemExit:\n    pass\n"
            "from crucible.realworld import RealWorldBridge, BridgeRefused\n"
            "try:\n    RealWorldBridge()\n    print('OPEN')\nexcept BridgeRefused:\n    print('LATCHED')\n")
    r = _py(code)
    lines = r.stdout.strip().splitlines()
    return lines[-1] if lines and r.returncode == 0 else f"failed: {r.stderr[-300:]}"


def _pool_worker_engages() -> bool:
    path = os.path.join(SRC, "crucible", "scale", "pool.py")
    tree = ast.parse(open(path, encoding="utf-8").read())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_worker_main")
    first = [n for n in fn.body if not isinstance(n, ast.Expr) or not isinstance(n.value, ast.Constant)][:3]
    return any(isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and getattr(n.value.func, "id", "") == "engage"
               for n in first)


@pytest.fixture(scope="module")
def network():
    from crucible.redteam.framework import sandbox_available
    ok, why = sandbox_available()
    assert ok, f"needs the real sandbox: {why}"
    from crucible.redteam.v5 import network as N, run, sandbox as SB
    rep = run(N.probes() + [SB.SandboxEscapeMatrix()])
    return {r["name"]: r for r in rep["probes"]}


@pytest.fixture(scope="module")
def social():
    import crucible
    crucible.load_all_modules()
    from crucible.core.kernel import World, WorldSpec
    from crucible.social.modules import social_internet_modules
    mods = social_internet_modules()
    for m in mods:
        if m["name"] == "msg":
            m["config"] = {"max_per_day": 3}
    w = World.create(WorldSpec("sp16-social", 16, mods, {"cost_model": "deterministic"}))
    P = {h: w.system_call("admin.join", {"handle": h, "endowment": 100_000})["principal"] for h in ("alice", "eve", "bob")}
    X = lambda p, tool, **a: w.execute(P[p], {"tool": tool, "args": a})  # noqa: E731
    out = {}
    assert X("alice", "msg.block", who=P["eve"]).ok
    out["block_prevents_messages"] = not X("eve", "msg.send", to=P["alice"], text="hi").ok
    sent = [X("bob", "msg.send", to=P["alice"], text=f"m{i}") for i in range(4)]
    out["message_rate_limit_enforced"] = all(r.ok for r in sent[:3]) and not sent[3].ok and \
        (sent[3].error or {}).get("code") == "rate_limited"
    tools = {t["name"] for t in w.available_tools(P["alice"])}
    out["moderation_tools"] = sorted(t for t in tools if t in ("community.report", "community.moderate",
                                                                "community.modlog", "msg.block"))
    w.close()
    from crucible.redteam.v5 import run, social as SO
    rep = run([SO.SybilRing(), SO.PaidEngagement(), SO.OwnOrganisationsFollow()])
    out["fabricated_engagement_probes"] = {r["name"]: r["result"] for r in rep["probes"]}
    return out


def test_SP16_security_boundaries(network, social):
    d = _defaults()
    latches = {m: _latch(m) for m in ENTRY_POINTS}
    pool = _pool_worker_engages()
    net_names = [n for n in network if n.startswith("v5.network.")]
    escapes = sum(int((network[n].get("evidence") or {}).get("network_escapes") or 0) for n in net_names)
    matrix = network["v5.sandbox.escape_matrix"]
    sock = {k: v for k, v in ((matrix.get("evidence") or {}).get("matrix") or {}).items()
            if k in ("socket_inet", "socket_unix", "socket_raw_syscall", "getaddrinfo", "urlopen")}
    public_reachable = (any(network[n]["result"] != "held" for n in net_names) or escapes > 0
                        or matrix["result"] != "held" or any(v[0] == "allowed" for v in sock.values()) or not sock)
    fab = social["fabricated_engagement_probes"]
    metrics = {
        "realworld_enabled_by_default": bool(d["enabled"]),
        "public_network_reachable_from_world": bool(public_reachable),
        "realworld_get_without_authorization": d["get"], "realworld_post_without_authorization": d["request"],
        "latch_engaged_entry_points": sum(v == "LATCHED" for v in latches.values()) + int(pool),
        "world_hosting_entry_points": len(ENTRY_POINTS) + 1,
        "network_probes": len(net_names), "network_probes_held": sum(network[n]["result"] == "held" for n in net_names),
        "sandbox_socket_checks_refused": sum(v[0] == "blocked" for v in sock.values()),
        "block_prevents_messages": social["block_prevents_messages"],
        "message_rate_limit_enforced": social["message_rate_limit_enforced"],
        "moderation_tools": ",".join(social["moderation_tools"]),
        "fabricated_engagement_excluded": all(v == "held" for v in fab.values()),
    }
    record("SP-16", metrics, command=CMD, notes="latches: " + json.dumps(latches) + "; engagement probes: "
           + json.dumps(fab))
    assert metrics["realworld_enabled_by_default"] is False
    assert metrics["public_network_reachable_from_world"] is False, {n: network[n]["result"] for n in network}
    assert d["get"] == d["request"] == "refused"
    assert metrics["latch_engaged_entry_points"] == metrics["world_hosting_entry_points"], latches
    assert metrics["block_prevents_messages"] and metrics["message_rate_limit_enforced"]
    assert {"community.report", "community.moderate", "msg.block"} <= set(social["moderation_tools"])
    assert metrics["fabricated_engagement_excluded"], fab
