"""SP-2.2: compositional freedom through the 13 affordances of MASTER_SPEC Part 2.2, with no predetermined menu of
strategies or task-specific moves.

World: a Tier 3 civilisation (`make(3, "balanced", ...)`: every subsystem, the one population) - the most open world
the interface offers. Two participants compose ordinary tools; each affordance counts only when its call succeeded
AND it consumed the output of another affordance (an identifier, a host, a tool, an account another step created):

    organisation creation -> executable code (a repository the organisation owns, run in the sandbox) -> dynamic
    service (that repository deployed) -> agent-created tool (the service's endpoint) -> tool discovery (a second
    participant finds it) -> data processing (its own records through that tool, output checked) -> market
    transaction (the call is paid to the organisation) -> product creation (a product delivering the service) ->
    market transaction (the product bought) -> communication (a message naming the product) -> agent-created website
    (a name pointed at the service, fetched by the other participant) -> long-running workflow (a deferred flow over
    the agent-created tool) -> experimentation (a prediction about the service's own requests) -> research (an
    apparatus run paid from the organisation's account)

"No predetermined menu": every tool of the world is scanned; a tool whose name, description or an enumerated
argument offers strategies / tactics / playbooks (rather than world operations) is a menu. The scan is a stated
heuristic (word list below); its findings are listed in the evidence."""
import re

import pytest

from crucible.acceptance.v5.evidence import record
from crucible.experience.registry import make
from crucible.experience.scripted import call, deploy_service, need_input, personal_account

pytestmark = pytest.mark.timeout(2400)
CMD = "PYTHONPATH=src python -m pytest tests/v5/test_gate_SP_2_2.py"
HOUR, DAY = 3600, 86400
MENU_WORDS = re.compile(r"\b(strateg(y|ies|ic_choice)|playbook|tactic(s)?|autopilot|auto_?play|recipe_choice|"
                        r"growth_plan|go_to_market_plan)\b", re.I)
STRATEGY_ENUM_VALUES = {"aggressive", "conservative", "undercut", "price_war", "growth_hack", "collude", "defend",
                        "expand_fast", "milk", "harvest", "blitz"}


def _menus(tools: list[dict]) -> list[str]:
    out = []
    for t in tools:
        text = f"{t['name']} {t.get('description') or ''}"
        if MENU_WORDS.search(text.replace(".", " ")):
            out.append(f"{t['name']}: wording")
            continue
        for k, p in ((t.get("inputSchema") or {}).get("properties") or {}).items():
            vals = {str(v).lower() for v in (p.get("enum") or [])}
            if MENU_WORDS.search(k) or len(vals & STRATEGY_ENUM_VALUES) >= 2:
                out.append(f"{t['name']}.{k}: enum {sorted(vals)[:6]}")
    return out


def test_SP_2_2_compositional_freedom():
    env = make(3, "balanced", 8, agents=["builder", "client"], endowment=3_000_000)
    b, c = env.agents()
    done: dict[str, str] = {}

    def ok(aff: str, r: dict, uses: str) -> dict:
        if r.get("ok"):
            done.setdefault(aff, uses)
        return r

    def ex(who, tool, **args):
        return env.execute(who, {"tool": tool, "args": args})
    svc = deploy_service(env, b, handle="compco", host="compsvc", price=4)
    done["organisation_creation"] = "id.create_org + biz.register + bank.open_account (capital from the founder)"
    inp = need_input("sort.records", 3)
    ok("executable_code", ex(b, "code.run", repo=svc["repo"], target="service:solve", input=inp),
       f"the organisation's repository {svc['repo']}")
    st = call(env, b, "cloud.status", host="compsvc", days=1)
    if st is not None:
        done["dynamic_services"] = f"{svc['repo']} deployed on {svc['instance']} as host compsvc"
    found = ex(c, "world.find_tools", query="compsvc sort records")
    desc = ok("tool_discovery", ex(c, "world.describe", host="compsvc"), "world.find_tools / world.describe compsvc")
    tools_c = {t["name"]: t for t in env.tools(c)}
    if "compsvc.solve" in tools_c and tools_c["compsvc.solve"]["annotations"].get("category") == "agent_services":
        done["agent_created_tools"] = "compsvc.solve (the builder's service endpoint) in the client's tool list"
    acct_c = personal_account(env, c)
    bal0 = [a for a in call(env, c, "bank.accounts") if a["account"] == acct_c][0]["balance"]
    data = need_input("sort.records", 11)
    r = ex(c, "compsvc.solve", **data)
    body = (r.get("result") or {}).get("body") if r.get("ok") else None
    if r.get("ok") and (r["result"].get("status") == 200) and body is not None:
        done["data_processing"] = "the client's own records through the agent-created tool (HTTP 200 with output)"
    bal1 = [a for a in call(env, c, "bank.accounts") if a["account"] == acct_c][0]["balance"]
    if bal0 - bal1 == svc["price"]:
        done["market_transactions"] = f"the call paid {svc['price']} cents to the organisation"
    prod = ex(b, "prod.create", name="CompSort", kind="saas", owner=svc["org"],
              description="hosted record sorting by compsvc.crucible")
    offer = None
    if prod.get("ok"):
        pid = prod["result"].get("product")
        ex(b, "prod.update", product=pid, status="launched",
           delivery={"type": "service", "host": "compsvc", "endpoint": "solve", "need": "sort.records"})
        off = ex(b, "prod.offer", product=pid, kind="one_time", price_cents=250)
        offer = (off.get("result") or {}).get("offer") if off.get("ok") else None
        if offer:
            done["product_creation"] = f"product {pid} of {svc['org']} offered as {offer}"
    if offer:
        buy = ex(c, "prod.buy", offer=offer, account=acct_c)
        if buy.get("ok"):
            done["market_transactions"] = done.get("market_transactions", "") + f"; product offer {offer} bought"
    m = ex(c, "mail.send", to=[b], subject="CompSort", body=f"Bought {offer}; compsvc sorted my records.")
    # an agent-created website: the builder ships a page in its repository (crucible.json "web" routes/static) and
    # points its own domain at the deployment; the client browses it
    import json as _json
    mf = _json.loads(call(env, b, "code.read", repo=svc["repo"], path="crucible.json")["content"])
    src = call(env, b, "code.read", repo=svc["repo"], path=mf["entry"])["content"]
    mf["web"] = {"routes": {"GET /": "index"}, "title": "CompSort"}
    src += ("\n\ndef index(request, caller):\n    return {'html': '<html><body><h1>CompSort</h1><p>Sorting records "
            "at 4 cents a call.</p></body></html>'}\n")
    call(env, b, "code.commit", repo=svc["repo"], files={mf["entry"]: src, "crucible.json": _json.dumps(mf)},
         message="website")
    ex(b, "cloud.deploy", repo=svc["repo"], host="compsvc", instance=svc["instance"])
    reg = ex(b, "net.register", name="compco-web", org=svc["org"], account=svc["account"])
    pt = ex(b, "net.point", name="compco-web", host="compsvc") if reg.get("ok") else reg
    fetched = ex(c, "net.fetch", url="http://compco-web.crucible/") if pt.get("ok") else pt
    if fetched.get("ok") and isinstance(fetched.get("result"), dict) and fetched["result"].get("status") == 200 \
            and "CompSort" in str(fetched["result"].get("body")):
        done["agent_created_websites_and_services"] = "compco-web.crucible (the builder's domain) -> its deployment " \
                                                      "compsvc, page fetched by the client (HTTP 200)"
    website = {"register": reg.get("ok") or reg.get("error"), "point": pt.get("ok") or pt.get("error"),
               "fetch": (fetched.get("result") or {}).get("status") if fetched.get("ok") else fetched.get("error")}
    flow = ex(c, "wf.define", name="nightly sort", steps=[{"id": "s", "tool": "compsvc.solve",
                                                          "args": need_input("sort.records", 12)}])
    later = ex(c, "wf.run_later", flow=flow["result"]["flow"], in_s=6 * HOUR) if flow.get("ok") else flow
    expr = ex(b, "exp.register", hypothesis="the nightly job raises my service's request count",
              measure={"tool": "cloud.status", "args": {"host": "compsvc", "days": 2}, "path": ["days"],
                       "field": "requests", "aggregate": "sum"}, prediction={"op": "increase", "min_change": 0},
              deadline_in_s=12 * HOUR)
    apps = sorted(call(env, b, "sci.apparatus"), key=lambda x: (x["price_cents"], x["apparatus"]))
    research = None
    if apps:
        ap = apps[0]
        point = {"at": {v: (lo + hi) / 2.0 for v, (lo, hi) in sorted(ap["range"].items())}, "n": 1}
        research = ex(b, "sci.run", apparatus=ap["apparatus"], design=[point], account=svc["account"],
                      note="what does the phenomenon read at mid-range?")
    env.advance(13 * HOUR)
    inbox = call(env, b, "mail.inbox")
    if m.get("ok") and any("CompSort" in str(x) for x in inbox):
        done["communication"] = "mail naming the bought offer, received by the builder"
    jobs = call(env, c, "wf.jobs")
    if later.get("ok") and any(j.get("job") == later["result"]["job"] and j.get("ok") for j in jobs):
        done["long_running_workflows"] = "a deferred flow (6 h) over the agent-created tool compsvc.solve ran"
    evs = env.observe_since(b, 0)[0]
    if expr.get("ok") and any(e["type"] == "exp.resolved" and (e["cause"] or {}).get("action_seq") ==
                              expr["action_seq"] for e in evs):
        done["experimentation"] = "a prediction about the builder's own service requests resolved"
    if research is not None and research.get("ok"):
        done["research"] = "an apparatus run paid from the organisation's account"
    menus = _menus(env.tools(b) + [t for t in env.tools(c) if t["name"] not in {x["name"] for x in env.tools(b)}])
    metrics = {"affordances_composable": len(done), "predetermined_strategy_menus": len(menus),
               "tools_scanned": len(env.tools(b))}
    missing = sorted({"executable_code", "dynamic_services", "tool_discovery", "data_processing", "communication",
                      "market_transactions", "organisation_creation", "product_creation", "research",
                      "experimentation", "long_running_workflows", "agent_created_tools",
                      "agent_created_websites_and_services"} - set(done))
    record("SP-2.2", metrics, command=CMD, labels={"scripted": True},
           notes=f"composed (scripted reference participants acting through the interface; each affordance uses "
                 f"another's output): {done}; not demonstrated: {missing}; website steps: {website}; research call: "
                 f"{research}; find_tools: "
                 f"{found.get('ok')}; describe: {desc.get('ok')}; menu scan ({MENU_WORDS.pattern}, strategy enum "
                 f"values {sorted(STRATEGY_ENUM_VALUES)}): {menus}")
    assert not menus, menus
    assert len(done) >= 13, missing
