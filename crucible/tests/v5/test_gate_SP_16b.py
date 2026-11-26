"""SP-16b (workstream O): participant consent, moderation, abuse prevention and integrity mechanisms exist in the
world; no real-world spam, fabricated engagement, covert influence, surveillance or impersonation; agent-supplied
media is processed outside the sandbox only under sandbox-equivalent restrictions; sandbox tenants are isolated;
learners can be evaluated out of process.

Measured on the real system:

* integrity mechanisms, each counted only when it works against an attempt in a real world: blocking stops a
  sender's messages; the messaging rate limit stops the fourth message; fabricated engagement (a sybil ring, paid
  engagement, follows by one's own organisations) is excluded from organic engagement; related-party revenue
  (transfers, commitment escrow, project rewards, goods purchases) is netted from evaluator revenue; ratings bought
  with the seller's money do not enter the public rating;
* real-world abuse paths: the real-world adapter off by default and refusing without authorisation, latched shut in
  every world-hosting entry point, unreachable from world code (static import closure), and the worlds' content and
  fetch tools reaching no network (process guard; jailed browser and media tools against a loopback canary);
* media: every FFmpeg/ffprobe/SoX/Chromium launch at the participant call sites (studio and media tools) observed
  through audit events, and Pillow decodes inside the world process; an unjailed launch or an in-process decode of
  participant bytes is an unrestricted media-tool run;
* tenants: per-world sandbox uids, per-world pools and meter memo, the artifact pool across worlds, and the jail
  uids (distinct per tool kind, outside the tenant range, not root or nobody);
* out-of-process evaluation: Tier 1 and a 4.0 kernel world hosted by `crucible.experience.remote.host`, driven only
  through the participant handle, verified through the evaluator handle; the participant pipe refuses evaluator
  operations and runs nothing it receives.
"""
import json
import os
import subprocess
import sys

import pytest

from crucible.acceptance.v5.evidence import record

pytestmark = [pytest.mark.slow, pytest.mark.timeout(2400)]
CMD = "PYTHONPATH=src python -m pytest tests/v5/test_gate_SP_16b.py"
SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src")
ENTRY_POINTS = ("crucible.cli", "crucible.api.http_server", "crucible.api.mcp_server", "crucible.experience.__main__",
                "crucible.web.app")
INTEGRITY = {"fabricated_engagement": ("v5.social.sybil_ring_engagement", "v5.social.engagement_paid_by_the_target",
                                       "v5.social.own_organisations_inflate_followers"),
             "related_party_revenue": ("v5.commerce.circular_revenue_by_transfer",
                                       "v5.commerce.circular_revenue_through_commitment_escrow",
                                       "v5.commerce.circular_revenue_through_project_reward",
                                       "v5.commerce.circular_revenue_through_goods_purchase"),
             "verified_ratings": ("v5.commerce.owner_funded_five_star_ratings",)}
ABUSE = ("v5.isolation.world_code_cannot_import_evaluator_side_code", "v5.network.population_world_is_offline",
         "v5.network.social_studio_with_hostile_urls_is_offline",
         "v5.network.jailed_media_tools_cannot_reach_network_or_host_files",
         "v5.network.agent_javascript_in_jailed_browser_cannot_exfiltrate")
MEDIA = ("v5.network.media_tools_at_participant_call_sites_are_jailed",
         "v5.network.image_decoders_run_outside_the_world_process")
TENANTS = ("v5.sandbox.tenants_do_not_share_a_uid", "v5.crossworld.sandbox_pools_and_meter_memo_are_per_world",
           "v5.crossworld.artifact_bytes_do_not_cross_worlds")
REMOTE = ("v5.isolation.remote_participant_cannot_run_code_in_the_world",)


def _py(code: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "CRUCIBLE_HOSTS_WORLDS"}
    env["PYTHONPATH"] = SRC
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, env=env)


def _realworld_defaults() -> dict:
    code = ("import json\nfrom crucible.realworld import RealWorldBridge, BridgeRefused\nb = RealWorldBridge()\n"
            "out = {'enabled': b.enabled}\n"
            "for m, args in (('get', ('https://example.org/',)), ('request', ('POST', 'https://example.org/'))):\n"
            "    try:\n        getattr(b, m)(*args)\n        out[m] = 'performed'\n"
            "    except BridgeRefused as e:\n        out[m] = 'refused'\n"
            "print(json.dumps(out))\n")
    r = _py(code)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def _latched(module: str) -> bool:
    code = (f"import importlib\nm = importlib.import_module({module!r})\n"
            "try:\n    m.main(['--help'])\nexcept SystemExit:\n    pass\n"
            "from crucible.realworld import RealWorldBridge, BridgeRefused\n"
            "try:\n    RealWorldBridge()\n    print('OPEN')\nexcept BridgeRefused:\n    print('LATCHED')\n")
    r = _py(code)
    lines = r.stdout.strip().splitlines()
    return r.returncode == 0 and bool(lines) and lines[-1] == "LATCHED"


def _messaging() -> dict:
    import crucible
    crucible.load_all_modules()
    from crucible.core.kernel import World, WorldSpec
    from crucible.social.modules import social_internet_modules
    mods = social_internet_modules()
    for m in mods:
        if m["name"] == "msg":
            m["config"] = {"max_per_day": 3}
    w = World.create(WorldSpec("sp16b-social", 17, mods, {"cost_model": "deterministic"}))
    try:
        P = {h: w.system_call("admin.join", {"handle": h, "endowment": 100_000})["principal"]
             for h in ("alice", "eve", "bob")}
        X = lambda p, tool, **a: w.execute(P[p], {"tool": tool, "args": a})  # noqa: E731
        assert X("alice", "msg.block", who=P["eve"]).ok
        blocked = not X("eve", "msg.send", to=P["alice"], text="hi").ok
        sent = [X("bob", "msg.send", to=P["alice"], text=f"m{i}") for i in range(4)]
        limited = all(r.ok for r in sent[:3]) and not sent[3].ok and (sent[3].error or {}).get("code") == "rate_limited"
        tools = {t["name"] for t in w.available_tools(P["alice"])}
    finally:
        w.close()
    return {"block": blocked, "rate_limit": limited,
            "moderation_tools": sorted(t for t in tools if t in ("community.report", "community.moderate",
                                                                  "community.modlog", "msg.block"))}


def _hosted(ref: dict, action: dict | None = None) -> dict:
    from crucible.experience.remote import host
    part, ev = host(ref)
    try:
        a = part.agents()[0]
        obs = part.observe(a)
        tools = part.tools(a)
        if action is None:                    # a read-only tool without required arguments
            t = next(t for t in tools if (t.get("annotations") or {}).get("readOnlyHint")
                     and not (t.get("inputSchema") or {}).get("required"))
            action = {"tool": t["name"], "args": {}}
        r = part.execute(a, action)
        part.advance(3600)
        refused = [part._raw(op)[0] == "error" for op in ("verify", "digest", "describe", "env", "world", "_obs")]
        v = ev.verify()
    finally:
        ev.close()
    return {"observed": bool(obs), "acted": r is not None, "verified": isinstance(v, dict) and bool(v),
            "evaluator_ops_refused": all(refused), "participant_handle_attrs": sorted(
                x for x in ("verify", "env", "world", "_world", "_env") if hasattr(part, x))}


@pytest.fixture(scope="module")
def red_team():
    from crucible.redteam.framework import sandbox_available
    ok, why = sandbox_available()
    assert ok, f"SP-16b needs the real sandbox: {why}"
    from crucible.redteam.v5 import probes, run
    names = set(sum(INTEGRITY.values(), ())) | set(ABUSE) | set(MEDIA) | set(TENANTS) | set(REMOTE)
    ps = [p for p in probes() if p.name in names]
    assert {p.name for p in ps} == names, sorted(names - {p.name for p in ps})
    rep = run(ps)
    return {r["name"]: r for r in rep["probes"]}


def test_SP16b_integrity_abuse_media_tenants_and_out_of_process_evaluation(red_team):
    held = lambda n: red_team[n]["result"] == "held"  # noqa: E731
    msg = _messaging()
    mechanisms = {"blocking": msg["block"], "message_rate_limit": msg["rate_limit"],
                  **{k: all(held(n) for n in v) for k, v in INTEGRITY.items()}}
    d = _realworld_defaults()
    latches = {m: _latched(m) for m in ENTRY_POINTS}
    abuse = {"adapter_enabled_by_default": bool(d["enabled"]), "get_without_authorisation": d["get"] != "refused",
             "post_without_authorisation": d["request"] != "refused",
             **{f"latch_open:{m}": not v for m, v in latches.items()},
             **{n: not held(n) for n in ABUSE}}
    ev_media = (red_team[MEDIA[0]].get("evidence") or {}), (red_team[MEDIA[1]].get("evidence") or {})
    unjailed = int(ev_media[0].get("unrestricted_media_tool_runs") or 0) if red_team[MEDIA[0]]["result"] != "held" else 0
    decodes = int(ev_media[1].get("in_process_image_decodes") or 0) if red_team[MEDIA[1]]["result"] != "held" else 0
    from crucible.sandbox import isolation
    from crucible.sandbox.runner import TENANT_UID_BASE
    jail_uids = {k: isolation.uid_for(k) for k in ("media-tools", "media-renderer", "web-browser", "image-decoder")}
    tenant_fail = {n: not held(n) for n in TENANTS}
    tenant_fail["jail_uids_distinct"] = len(set(jail_uids.values())) != len(jail_uids)
    tenant_fail["jail_uids_outside_tenant_range"] = any(TENANT_UID_BASE <= u < TENANT_UID_BASE + 65536 or u in (0, 65534)
                                                        for u in jail_uids.values())
    hosted = {"tier1": _hosted({"tier": 1, "name": "pe_rule_induction", "seed": 3, "reward_mode": "observation_only"},
                               {"tool": "rapid.query", "args": {"kind": "query", "target": "examples"}}),
              "kernel": _hosted({"tier": 0, "name": "tiny", "seed": 5, "kw": {"agents": ["learner"]}})}
    oop = all(h["observed"] and h["acted"] and h["verified"] and h["evaluator_ops_refused"]
              and not h["participant_handle_attrs"] for h in hosted.values()) and all(held(n) for n in REMOTE)
    metrics = {
        "integrity_mechanisms": sum(1 for v in mechanisms.values() if v),
        "realworld_abuse_paths": sum(1 for v in abuse.values() if v),
        "unrestricted_media_tool_runs": unjailed + decodes,
        "cross_tenant_isolation_failures": sum(1 for v in tenant_fail.values() if v),
        "out_of_process_evaluation_mode": bool(oop),
        "integrity_mechanisms_checked": len(mechanisms), "abuse_paths_checked": len(abuse),
        "media_launches_observed": int(ev_media[0].get("media_launches") or 0),
        "media_launches_jailed": int(ev_media[0].get("jailed") or 0),
        "unjailed_media_tool_launches": unjailed, "in_process_image_decodes": decodes,
        "tenant_checks": len(tenant_fail), "moderation_tools": ",".join(msg["moderation_tools"]),
    }
    record("SP-16b", metrics, command=CMD, notes="mechanisms: " + json.dumps(mechanisms) + "; abuse paths: "
           + json.dumps({k: v for k, v in abuse.items() if v}) + "; media: " + json.dumps(ev_media[0].get("unjailed"))
           + "; tenants: " + json.dumps(tenant_fail) + "; hosted: " + json.dumps(hosted))
    assert all(r["result"] not in ("error", "not_applicable") for r in red_team.values()), \
        {n: r["result"] for n, r in red_team.items()}
    assert metrics["media_launches_observed"] >= 5, ev_media[0]
    assert metrics["integrity_mechanisms"] >= 4, mechanisms
    assert metrics["realworld_abuse_paths"] == 0, abuse
    assert metrics["cross_tenant_isolation_failures"] == 0, tenant_fail
    assert metrics["out_of_process_evaluation_mode"] is True, hosted
    assert metrics["unrestricted_media_tool_runs"] == 0, (ev_media[0].get("unjailed"), decodes)
