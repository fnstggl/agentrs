"""SP-11.2: an open-ended tool interface with discovery and schemas; agent-created services add discoverable
affordances; everything governed by permissions, prices and ordinary world rules.

World: the 4.0 open digital economy plus the experience package's `exp` module, through `make(0, ...)`. A scripted
builder deploys a service; a second participant discovers it with `world.describe` and gains its endpoints as new
tools (category ``agent_services``) and pays for a call. One representative tool per category is executed."""
import pytest

from crucible.acceptance.v5.evidence import record
from crucible.experience import TOOL_CATEGORIES, validate_observation
from crucible.experience.observation import validate_tool
from crucible.experience.registry import make
from crucible.experience.scripted import call, deploy_service, discover, need_input, personal_account

pytestmark = pytest.mark.timeout(900)
CMD = "PYTHONPATH=src python -m pytest tests/v5/test_gate_SP_11_2.py"


def test_SP_11_2_open_tool_interface():
    env = make(0, "open_digital", 0, "observation_only", agents=["builder", "client"], endowment=3_000_000)
    b, c = env.agents()
    before = {t["name"] for t in env.tools(c)}
    svc = deploy_service(env, b, handle="buildco", host="buildsvc")
    assert not any(n.startswith("buildsvc.") for n in {t["name"] for t in env.tools(c)}), "undiscovered host listed"
    gained = discover(env, c, "buildsvc")
    tools = env.tools(c)
    schema_errors = sum(len(validate_tool(t)) for t in tools)
    new_agent_tools = [t for t in tools if t["name"] in gained and t["annotations"]["category"] == "agent_services"]
    cats = sorted({t["annotations"]["category"] for t in tools} & set(TOOL_CATEGORIES))
    acct = personal_account(env, c)
    bal0 = [a for a in call(env, c, "bank.accounts") if a["account"] == acct][0]["balance"]
    used = env.execute(c, {"tool": "buildsvc.solve", "args": need_input("sort.records")})
    bal1 = [a for a in call(env, c, "bank.accounts") if a["account"] == acct][0]["balance"]
    # one ordinary call per category, by the participant entitled to make it
    exercised = {}

    def ex(cat, who, tool, **args):
        r = env.execute(who, {"tool": tool, "args": args})
        exercised[cat] = exercised.get(cat) or bool(r.get("ok"))
        return r
    sites = ex("browser", c, "net.sites", limit=5)
    if sites.get("ok") and sites["result"]:
        first = sites["result"][0]
        url = first.get("url") or f"http://{first.get('name') or first.get('domain')}/"
        ex("browser", c, "net.fetch", url=url)
    ex("search", c, "search.query", q="sort records service")
    ex("code_execution", b, "code.run", repo=svc["repo"], target="service:solve", input=need_input("sort.records", 1))
    ex("file_creation", b, "code.commit", repo=svc["repo"], files={"NOTES.md": "release notes\n"}, message="notes")
    ex("program_deployment", b, "cloud.deploy", repo=svc["repo"], host="buildsvc", instance=svc["instance"])
    ex("communication", c, "mail.send", to=[b], subject="your service", body="It sorted my records correctly.")
    ex("financial", c, "bank.transfer", from_account=acct, to=b, amount=100, memo="tip")
    listed = env.execute(c, {"tool": "market.browse", "args": {"kind": "api", "limit": 20}})
    needs = [x["need"] for x in (listed.get("result") or []) if x.get("need")]
    ex("research", c, "research.survey", need=needs[0] if needs else "sort.records", account=acct)
    ex("experiments", c, "exp.register", hypothesis="the tip leaves my balance above 2.9M cents",
       measure={"tool": "bank.accounts", "where": {"owner": "$self"}, "field": "balance", "aggregate": "sum"},
       prediction={"op": ">=", "value": 2_900_000}, deadline_in_s=3600)
    ex("organization", c, "id.create_org", handle="clientco", name="Client Co", kind="company")
    ex("content_creation", c, "docs.create", title="Review", body="A short review of buildsvc.", visibility="public")
    flow = ex("tool_composition", c, "wf.define", name="check balance", steps=[{"id": "a", "tool": "bank.accounts",
                                                                              "args": {}}])
    if flow.get("ok"):
        ex("tool_composition", c, "wf.run", flow=flow["result"]["flow"])
    exercised["agent_services"] = bool(used.get("ok") and used["result"].get("status") == 200)
    # ordinary rules: operator tools, other people's money, other people's services
    refused = [env.execute(c, {"tool": "admin.join", "args": {"handle": "intruder"}}),
               env.execute(c, {"tool": "bank.transfer", "args": {"from_account": svc["account"], "to": c,
                                                                 "amount": 10}}),
               env.execute(c, {"tool": "cloud.undeploy", "args": {"host": "buildsvc"}})]
    enforced = sum(1 for r in refused if not r["ok"] and r["error"]["code"] in ("forbidden", "not_found",
                                                                                "unauthenticated"))
    obs = env.observe(c)
    metrics = {"tool_categories": len(cats), "tool_categories_exercised": sum(1 for v in exercised.values() if v),
               "affordances_from_agent_services": len(new_agent_tools), "tool_schema_errors": schema_errors,
               "service_call_paid_cents": int(bal0 - bal1), "permission_checks_enforced": enforced,
               "tools_listed": len(tools), "tools_before_discovery": len(before)}
    record("SP-11.2", metrics, command=CMD, notes=f"categories available: {cats}; exercised: "
           f"{sorted(k for k, v in exercised.items() if v)}; not exercised successfully: "
           f"{sorted(k for k, v in exercised.items() if not v)}; new tools from the agent's service: {gained}")
    assert not validate_observation(obs)
    assert schema_errors == 0 and enforced == len(refused)
    assert metrics["service_call_paid_cents"] == svc["price"]
    assert len(cats) >= 13 and len(new_agent_tools) >= 1, metrics
