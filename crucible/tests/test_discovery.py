"""Tool discovery (core/discovery.py): search, host map, dry-run validation, reference help, API versions,
structured error detail, and the catalogue renderer. Every test runs on a real kernel world."""
from __future__ import annotations

import copy
import json
import re

import numpy as np
import pytest

import crucible

crucible.load_all_modules()
from crucible.agents.scripted import _random_args  # noqa: E402
from crucible.api import render as R  # noqa: E402
from crucible.api.local import API_VERSION  # noqa: E402
from crucible.core import discovery as D  # noqa: E402
from crucible.core import errors as E  # noqa: E402
from crucible.core import kernel as K  # noqa: E402
from crucible.core import schema as S  # noqa: E402
from crucible.core.kernel import SYSTEM, Module, World, WorldSpec, endpoint, register_module  # noqa: E402
from crucible.worldgen import generate  # noqa: E402

NEW = ("world.api", "world.find_tools", "world.help", "world.hosts", "world.validate")


@register_module
class _DiscoProbe(Module):
    """Class docstring that must lose to the explicit summary."""
    name = "_discoprobe"
    host = "probe"
    version = "7"
    summary = "Probe service for discovery tests."

    @endpoint("ping", S.obj({"n": S.s_int("how many", minimum=0, default=3)}), read_only=True, cost_s=0)
    def ep_ping(self, ctx, n=3):
        """Echo a number."""
        return {"n": n}

    @endpoint("pong", S.obj(), version="2.1")
    def ep_pong(self, ctx):
        """Second endpoint with its own version."""
        return {}


def tiny(seed=3, **params):
    ov = {"n_consumers": 300}
    if params:
        ov["params"] = params
    w = World.create(generate({"family": "tiny", "overrides": ov}, seed))
    for h in ("alice", "bob"):
        w.system_call("admin.join", {"handle": h, "endowment": 500_000})
    return w


def small(**params):
    spec = WorldSpec("disco-test", 5, [{"name": "identity"}, {"name": "bank"}, {"name": "admin"},
                                       {"name": "_discoprobe"}], params=dict(params))
    w = World.create(spec, "w-disco")
    for h in ("alice", "bob"):
        w.system_call("admin.join", {"handle": h, "endowment": 10_000})
    return w


def ex(w, actor, tool_name, /, **args):
    return w.execute(actor, {"tool": tool_name, "args": args})


def log(w):
    return copy.deepcopy([e.to_dict() for e in w.events_range(0)])


A, B = "p:alice", "p:bob"


@pytest.fixture(scope="module")
def tw():
    """A read-mostly tiny world shared by the tests that only search / render (they do not depend on its seq)."""
    return tiny()


# ====================================================================== registration and versions
def test_new_endpoints_are_readonly_free_and_every_spec_is_versioned():
    w = small()
    tools = {t["name"]: t for t in w.available_tools(A)}
    for n in NEW:
        ann = tools[n]["annotations"]
        assert ann["readOnlyHint"] is True and ann["costSeconds"] == 0 and ann["host"] == "world"
    assert all(isinstance(t["annotations"]["version"], str) and t["annotations"]["version"] for t in tools.values())
    assert tools["probe.ping"]["annotations"]["version"] == "7"          # module version
    assert tools["probe.pong"]["annotations"]["version"] == "2.1"        # explicit endpoint version
    assert tools["bank.transfer"]["annotations"]["version"] == "1"
    # the 3.0 interface is exactly recoverable
    old = small(discovery=False)
    names = [t["name"] for t in old.available_tools(A)]
    assert not set(NEW) & set(names) and set(names) | set(NEW) == set(tools)
    r = ex(old, A, "no.such")
    assert r.error == {"code": "unknown_tool", "message": "unknown tool 'no.such'"}


def test_api_reports_versions_and_a_stable_catalogue_hash(tw):
    assert D.API_VERSION == API_VERSION and D._MALFORMED == K._MALFORMED
    w = small()
    seq0 = w.seq
    r = ex(w, A, "world.api")
    assert r.ok and r.cost_s == 0
    api = r.result
    assert api["kernel"] == K.KERNEL_VERSION and api["api"] == API_VERSION and api["discovery"] == D.DISCOVERY_VERSION
    tools = w.available_tools(A)
    assert api["tools"] == len(tools) and api["hosts"] == len({t["annotations"]["host"] for t in tools})
    assert api["catalogue"] == D.catalogue_hash(tools) == api["visible"]["hash"] and len(api["catalogue"]) == 32
    assert api["modules"]["_discoprobe"] == "7"
    # recorded: the log itself states the interface (internal note, invisible to actors)
    notes = [e for e in w.events_range(seq0) if e.type == "note"]
    assert len(notes) == 1 and notes[0].vis == "internal" and notes[0].payload["catalogue"] == api["catalogue"]
    # stable: same spec -> same hash in a fresh world; independent of order; sensitive to any change of a spec
    assert small().execute(A, {"tool": "world.api", "args": {}}).result["catalogue"] == api["catalogue"]
    assert D.catalogue_hash(list(reversed(tools))) == api["catalogue"]
    for field, change in (("description", lambda t: t.update(description=t["description"] + " ")),
                          ("schema", lambda t: t["inputSchema"]["properties"].update(x={"type": "string"})),
                          ("cost", lambda t: t["annotations"].update(costSeconds=99)),
                          ("version", lambda t: t["annotations"].update(version="2"))):
        mod = copy.deepcopy(tools)
        change(mod[0])
        assert D.catalogue_hash(mod) != api["catalogue"], field
    # the operator-side stamp agrees, and the visible hash follows what an actor has discovered
    assert D.interface_stamp(w)["catalogue"] == api["catalogue"]
    w2 = tiny()
    before = ex(w2, A, "world.api").result
    assert ex(w2, A, "search.query", q="route shortest").ok
    after = ex(w2, A, "world.api").result
    assert after["catalogue"] == before["catalogue"] and after["visible"]["hash"] != before["visible"]["hash"]
    assert after["visible"]["tools"] > before["visible"]["tools"]


# ====================================================================== search
QUERIES = {   # a sanity set written for this test (not a benchmark): plain-language need -> the tool that serves it
    "create an organisation": "id.create_org", "register a company": "biz.register",
    "deploy a service": "cloud.deploy", "send money": "bank.transfer", "rent compute": "cloud.rent",
    "publish a listing": "market.list", "what do customers want": "research.survey",
    "write code to a repository": "code.commit", "run tests": "code.test", "open a bank account": "bank.open_account",
    "search the web": "search.query", "post on the forum": "forum.post", "advertise": "ads.create_campaign",
    "check my balance": "bank.balance", "manifest crucible.json": "cloud.deploy", "sign a contract": "contracts.sign",
    "send a message": "mail.send", "price of compute providers": "cloud.providers", "organization": "id.create_org",
    "deployment": "cloud.deploy", "invoice": "bank.invoice", "grant permission": "id.grant",
    "repository": "code.create_repo", "which hosts exist": "world.hosts",
}


def test_find_tools_ranks_the_serving_tool_and_reads_whole_descriptions(tw):
    w = tw
    ranks = {}
    for q, want in QUERIES.items():
        res = w.query(A, "world.find_tools", {"query": q, "limit": 8})["results"]
        names = [x["name"] for x in res]
        ranks[q] = names.index(want) + 1 if want in names else None
        assert [x["score"] for x in res] == sorted((x["score"] for x in res), reverse=True)
    top3 = sum(1 for r in ranks.values() if r is not None and r <= 3)
    top8 = sum(1 for r in ranks.values() if r is not None)
    assert top8 >= 0.9 * len(QUERIES) and top3 >= 0.75 * len(QUERIES), ranks
    # the manifest format sits on the second line of the description: a first-line index could not find it
    spec = next(t for t in w.available_tools(A) if t["name"] == "cloud.deploy")
    assert '"entry": "service.py"' in spec["description"] and "entry" not in spec["description"].splitlines()[0]
    assert ranks["manifest crucible.json"] == 1
    hit = w.query(A, "world.find_tools", {"query": "handler price_cents entry", "limit": 3, "detail": True})["results"][0]
    assert hit["name"] == "cloud.deploy" and hit["spec"] == spec           # complete, machine-readable
    assert hit["signature"] == "cloud.deploy(host: str, instance: str, repo: str, ref?: str)"
    # parameter names and parameter descriptions are searched too
    assert w.query(A, "world.find_tools", {"query": "daily_budget_cents"})["results"][0]["name"] == "ads.create_campaign"
    r = w.query(A, "world.find_tools", {"query": "max_per_period payee"})
    assert r["results"][0]["name"] == "bank.authorize_debit"
    # an exact tool name ranks first; a host filter restricts; an empty query with a host browses it
    assert w.query(A, "world.find_tools", {"query": "bank.balance"})["results"][0]["name"] == "bank.balance"
    only = w.query(A, "world.find_tools", {"query": "list", "host": "market", "limit": 50})["results"]
    assert only and all(x["host"] == "market" for x in only)
    browse = w.query(A, "world.find_tools", {"query": "", "host": "bank", "limit": 50})
    assert [x["name"] for x in browse["results"]] == sorted(t["name"] for t in w.available_tools(A)
                                                             if t["name"].startswith("bank."))
    assert w.query(A, "world.find_tools", {"query": "the of"})["results"] == []
    nohost = w.query(A, "world.find_tools", {"query": "pay", "host": "banking"})
    assert nohost["results"] == [] and nohost["unknown_host"] == "banking" and nohost["near_hosts"] == ["bank"]
    assert len(w.query(A, "world.find_tools", {"query": "account", "limit": 2})["results"]) == 2
    # operator tools are never returned to an in-world actor
    assert not any(x["name"].startswith("admin.") for x in
                   w.query(A, "world.find_tools", {"query": "endow join admin operator", "limit": 50})["results"])


def test_find_tools_is_deterministic_across_processes_of_the_same_state():
    w1, w2 = tiny(), tiny()
    f = w1.fork("fork-a")
    q = {"query": "pay invoice from account", "limit": 10, "detail": True}
    a, b, c = (x.query(A, "world.find_tools", q) for x in (w1, w2, f))
    assert a == b == c and a["matched"] >= 3
    assert w1.query(A, "world.find_tools", q) == a                       # cached index: same answer again
    idx = D.ToolIndex(w1.available_tools(A))                             # and a cold index agrees with the cache
    assert [n for _, n in idx.search(q["query"], 10)[0]] == [x["name"] for x in a["results"]]


def test_find_tools_never_reveals_a_service_the_caller_has_not_discovered():
    w = tiny()
    hidden = sorted(h for h, _ in w.state.scan("hosts"))                 # incumbents' deployed services
    assert "shorte1" in hidden and not w.knows(A, "host", "shorte1")
    seq0 = w.seq
    for q in ("shorte1", "shorte1 solve", "solve", "route shortest path", "shorte1.solve"):
        res = ex(w, A, "world.find_tools", query=q, limit=50, detail=True).result
        assert not any(x["host"] in hidden for x in res["results"]), q
    assert ex(w, A, "world.hosts").result["discovered"] == {}
    assert set(ex(w, A, "world.hosts").result["hosts"]).isdisjoint(hidden)
    for bad in ("shorte1.solv", "shorte.solve", "shorte2.solve"):       # near misses of a hidden tool
        r = ex(w, A, "world.validate", tool=bad).result
        assert not r["valid"] and not any(s.startswith("shorte1.") for s in r["suggestions"])
        assert "shorte1" not in r.get("near_hosts", [])
        e = ex(w, A, bad).error
        assert not any(s.startswith("shorte1.") for s in e["details"]["suggestions"])
        assert "shorte1" not in e["details"].get("near_hosts", [])
    # none of the discovery calls taught the caller anything
    assert not [e for e in w.events_range(seq0) if e.type == "know"]
    assert not w.knows(A, "host", "shorte1")
    # the world's own search engine is what reveals it; afterwards discovery covers it - for that caller only
    assert any("shorte1" in x["hosts"] for x in ex(w, A, "search.query", q="route shortest").result["results"])
    assert w.knows(A, "host", "shorte1")
    res = ex(w, A, "world.find_tools", query="shorte1 solve", detail=True).result["results"]
    assert res[0]["name"] == "shorte1.solve" and res[0]["spec"]["annotations"]["builtin"] is False
    assert ex(w, A, "world.hosts").result["discovered"] == {"shorte1": 1}
    assert "shorte1.solve" in ex(w, A, "shorte1.solv").error["details"]["suggestions"]
    assert not any(x["host"] == "shorte1" for x in
                   ex(w, B, "world.find_tools", query="shorte1 solve", limit=50).result["results"])


# ====================================================================== hosts
def test_hosts_is_a_complete_small_map_derived_from_docstrings(tw):
    w = tw
    tools = w.available_tools(A)
    res = w.query(A, "world.hosts", {})
    counts = {}
    for t in tools:
        counts[t["annotations"]["host"]] = counts.get(t["annotations"]["host"], 0) + 1
    assert {h: v[0] for h, v in res["hosts"].items()} == counts and res["tools"] == len(tools)
    assert "admin" not in res["hosts"] and all(len(v[1]) <= 72 for v in res["hosts"].values())
    # summaries come from the code: the file docstring, or the class docstring, or an explicit `summary`
    import crucible.services.bank as bank_mod
    assert res["hosts"]["bank"][1] == "Payments and banking on a double-entry ledger (integer cents)."
    assert bank_mod.__doc__.strip().startswith(res["hosts"]["bank"][1])
    full = D.host_summaries(w)
    assert all(full[h] for h in counts), [h for h in counts if not full[h]]
    assert not re.search(r"\((host|module)", " ".join(full.values()))    # the redundant "(host `x`)" is stripped
    sm = small()
    assert sm.query(A, "world.hosts", {})["hosts"]["probe"] == [2, "Probe service for discovery tests."]
    assert sm.query(A, "world.hosts", {"summary_chars": 12})["hosts"]["probe"][1] == "Probe ser..."
    assert all(v[1] == "" for v in sm.query(A, "world.hosts", {"summary_chars": 0})["hosts"].values())
    # small: the whole map as text stays within a few hundred tokens' worth of characters
    text = D.render_host_map(res)
    assert len(text.splitlines()) == len(counts) and len(text) < 90 * len(counts)
    assert len(json.dumps(res)) < 110 * len(counts)
    # the operator sees the operator host as well
    assert "admin" in D.hosts(w, SYSTEM)["hosts"]


# ====================================================================== dry-run validation
def test_validate_checks_without_executing_and_leaves_state_and_log_unchanged():
    w = small()
    acct = w.modules["bank"].primary_account(w.state, A)
    args = {"from_account": acct, "to": "bob", "amount": 250}
    mark = (w.seq, w.last_hash, w.digest())
    n_txns = w.state.count("txns")
    # the rollback-only query path: no trace at all
    r = w.query(A, "world.validate", {"tool": "bank.transfer", "args": args})
    assert r["valid"] is True and r["cost_s"] == 10 and r["read_only"] is False
    assert r["args"] == {"amount": 250, "from_account": acct, "to": "bob"}
    assert "permissions" in r["not_checked"]
    assert (w.seq, w.last_hash, w.digest()) == mark
    # as a recorded action: one `cmd` event for world.validate itself, and no effect of the validated tool
    res = ex(w, A, "world.validate", tool="bank.transfer", args=args)
    evs = w.events_range(mark[0])
    assert res.ok and res.result == r and res.cost_s == 0
    assert [(e.type, e.payload.get("tool")) for e in evs] == [("cmd", "world.validate")]
    assert w.modules["bank"].balance(w.state, acct) == 10_000 and w.state.count("txns") == n_txns
    # defaults are applied and reported
    r = w.query(A, "world.validate", {"tool": "probe.ping"})
    assert r["valid"] and r["args"] == {"n": 3} and r["cost_s"] == 0 and r["read_only"] is True and r["version"] == "7"
    # structured errors: path, what was expected, what was given
    r = w.query(A, "world.validate", {"tool": "bank.transfer", "args": {"from_account": acct, "amount": "ten", "too": "bob"}})
    assert r["valid"] is False and r["signature"].startswith("bank.transfer(amount: int, from_account: str, to: str")
    by = {e["path"]: e for e in r["errors"]}
    assert by["args.to"]["kind"] == "required" and by["args.to"]["given"] == "missing"
    assert by["args.amount"]["kind"] == "type" and by["args.amount"]["expected"] == "integer" \
        and by["args.amount"]["given"] == "ten" and by["args.amount"]["given_type"] == "str"
    assert by["args.too"]["kind"] == "unexpected" and by["args.too"]["nearest"] == "to"
    assert [e["message"] for e in r["errors"]] == S.validate(w.endpoints["bank.transfer"].params,
                                                             {"from_account": acct, "amount": "ten", "too": "bob"})
    r = w.query(A, "world.validate", {"tool": "id.create_org", "args": {"handle": "p_nova", "name": "N", "kind": "firm"}})
    by = {e["path"]: e for e in r["errors"]}
    assert by["args.handle"]["kind"] == "pattern" and "[a-z][a-z0-9-]{1,30}" in by["args.handle"]["expected"]
    assert by["args.kind"]["kind"] == "enum" and "'company'" in by["args.kind"]["expected"]
    # unknown tool: closest existing names by edit distance and by search
    r = w.query(A, "world.validate", {"tool": "bank.transfr", "args": args})
    assert r["valid"] is False and r["errors"][0]["kind"] == "unknown_tool" and r["suggestions"][0] == "bank.transfer"
    r = w.query(A, "world.validate", {"tool": "identity.create_org"})
    assert r["host_exists"] is False and r["near_hosts"] == ["id"] and r["suggestions"][0] == "id.create_org"
    r = w.query(A, "world.validate", {"tool": "bank.send_money"})
    assert r["host_exists"] is True and "bank.transfer" in r["suggestions"] and "transfer" in r["host_tools"]
    r = w.query(A, "world.validate", {"tool": "zzz.qqq"})
    assert r["valid"] is False and r["suggestions"] == [] and r["near_hosts"] == []
    # operator tools: refused, and never suggested
    r = w.query(A, "world.validate", {"tool": "admin.endow", "args": {"principal": A, "amount": 5}})
    assert r["valid"] is False and r["errors"][0]["kind"] == "forbidden"
    assert not any(s.startswith("admin.") for s in w.query(A, "world.validate", {"tool": "admin.endw"})["suggestions"])
    assert w.modules["bank"].balance(w.state, acct) == 10_000 and w.check_invariants() == []


def test_structured_validation_agrees_with_schema_validate_on_the_whole_catalogue(tw):
    """Same checks, same order, same messages as core.schema.validate, for every tool schema under mutation."""
    rng = np.random.default_rng(7)
    tools = tw.available_tools(A)
    n_err = n_ok = 0
    kinds = set()
    junk = ["x", "", 3, -1, 2.5, 10**9, True, None, [], [1, "a"], {}, {"k": 1}, "A B", "x" * 5000, float("inf")]
    for t in tools:
        sch = t["inputSchema"]
        for _ in range(12):
            args = _random_args(sch, rng, A)
            for k in list(sch.get("properties", {})):
                u = rng.random()
                if u < 0.25:
                    args[k] = junk[int(rng.integers(0, len(junk)))]
                elif u < 0.35:
                    args.pop(k, None)
            if rng.random() < 0.3:
                args["zz_extra"] = 1
            got = D.validate_structured(sch, args)
            assert [e["message"] for e in got] == S.validate(sch, args), (t["name"], args)
            for e in got:
                assert e["path"].startswith("args") and e["message"].startswith(e["path"] + ":")
                assert e["expected"] and "given" in e and "kind" in e
                json.dumps(e, allow_nan=False)
                kinds.add(e["kind"])
            n_err += bool(got)
            n_ok += not got
    assert n_err > 300 and n_ok > 100
    assert {"type", "required", "unexpected", "enum", "pattern", "minimum", "max_length"} <= kinds, kinds
    nested = S.obj({"xs": S.s_arr(S.obj({"n": S.s_int(minimum=1, maximum=3)}, ["n"]), minItems=2)}, ["xs"])
    bad = {"xs": [{"n": 0}, {"m": 1}, 5]}
    got = D.validate_structured(nested, bad)
    assert [e["message"] for e in got] == S.validate(nested, bad)
    assert [e["path"] for e in got] == ["args.xs[0].n", "args.xs[1].n", "args.xs[1].m", "args.xs[2]"]


# ====================================================================== help
BANNED = ("you should", "we recommend", "recommend", "it is best", "best way", "best to", "a good idea", "strategy",
          "strategic", "advice", "advise", "tip:", "hint", "make sure", "be sure to", "try to", "consider ",
          "start by", "first step", "first,", "then you", "in order to", "so that you can", "you need to",
          "you must", "you can earn", "to earn", "earn ", "profit", "revenue", "income", "customer", "competitor",
          "rival", "market share", "undercut", "cheaper", "to succeed", "success", "to win", "winning", "goal",
          "objective", "reward", "useful", "valuable", "worth", "opportunit", "demand for", "grow",
          "invest", "business plan", "important to", "remember to", "don't forget", "avoid ", "always ", "never ",
          "prefer", "ideally", "efficient way", "quickest", "easiest")


def test_help_is_reference_material_without_guidance():
    w = tiny()
    index = ex(w, A, "world.help").result
    assert index["topic"] == "index" and index["topics"] == list(D.TOPICS)
    texts = {t: ex(w, A, "world.help", topic=t).result["text"] for t in D.TOPICS}
    assert all(len(v) > 60 for v in texts.values())
    unknown = ex(w, A, "world.help", topic="how to get rich").result
    assert unknown["topic"] == "index" and unknown["unknown_topic"] == "how to get rich"
    # no imperative business advice: in the help text, and in the descriptions of the discovery tools themselves
    own = [t["description"] for t in w.available_tools(A) if t["name"] in NEW]
    assert len(own) == 5
    for text in list(texts.values()) + own + [D.WORLD_SUMMARY]:
        low = text.lower()
        assert not [b for b in BANNED if b in low], ([b for b in BANNED if b in low], text[:200])
    # every error code the kernel can return is explained, and the list is generated from the error classes
    codes = {c.code for c in D._error_classes()}
    assert codes == set(D.ERROR_MEANINGS) and len(codes) >= 14
    for c in codes | {"parse_error"}:
        assert re.search(rf"^{c}: \S", texts["errors"], re.M), c
    assert all(issubclass(c, E.CrucibleError) for c in D._error_classes())
    # the identifier formats it states are the formats the services really produce
    ids = texts["identifiers"]
    org = ex(w, A, "id.create_org", handle="alice-co", name="Alice Co", kind="company").result["org"]
    me = w.modules["bank"].primary_account(w.state, A)
    assert ex(w, A, "biz.register", org=org, account=me).ok
    acct = ex(w, A, "bank.open_account", owner=org).result["account"]
    assert org == "o:alice-co" and "o:<handle>" in ids and "p:<handle>" in ids and A == "p:alice"
    assert re.fullmatch(r"acct_\d{6}", acct) and "account = acct_<6 digits>" in ids
    import crucible.services.cloud as cloud_mod
    import crucible.services.code as code_mod
    import crucible.services.identity as id_mod
    assert id_mod.HANDLE in ids and cloud_mod.HOST in ids and code_mod.NAME in ids
    repo = ex(w, A, "code.create_repo", name="svc", owner=org).result["repo"]
    commit = ex(w, A, "code.commit", repo=repo, files={"a.py": "x = 1\n"}).result["commit"]
    assert repo == "alice-co/svc" and "<owner handle>/<name>" in ids and re.fullmatch(r"c[0-9a-f]{20}", commit)
    assert ex(w, A, "bank.transfer", from_account=me, to=acct, amount=200_000).ok
    prov = sorted(p["provider"] for p in w.query(A, "cloud.providers"))[0]
    inst = ex(w, A, "cloud.rent", provider=prov, cores=1, account=acct, owner=org).result["instance"]
    assert re.fullmatch(r"inst_\d{6}", inst) and "instance = inst_<6 digits>" in ids
    lst = [l for l, _ in w.state.scan("listings")]
    assert lst and all(re.fullmatch(r"lst_\d{6}", l) for l in lst) and "listing = lst_<6 digits>" in ids
    assert all(re.fullmatch(r"dep_\d{6}", d) for d, _ in w.state.scan("deployments")) and "deployment = dep_<6 digits>" in ids
    # a world without a module does not document that module's identifiers; the action limit is this world's own
    sm = small()
    smids = ex(sm, A, "world.help", topic="identifiers").result["text"]
    assert "acct_<6 digits>" in smids and "lst_" not in smids and "inst_" not in smids
    assert "at most 240 actions" in texts["time"] and "no limit" in ex(sm, A, "world.help", topic="time").result["text"]
    assert '{"tool": "<host>.<endpoint>", "args": {...}}' in texts["actions"]
    assert '{"tool": "<tool name>", "args": {...}}' in R.action_format_help()       # the same shape the prompts state


# ====================================================================== error detail on rejected actions
def test_rejections_return_structured_detail_while_recorded_events_keep_the_3_0_error():
    w = small()
    acct = w.modules["bank"].primary_account(w.state, A)
    seq0 = w.seq
    r1 = ex(w, A, "identity.create_org", handle="acme", name="Acme")
    assert r1.error["code"] == "unknown_tool" and r1.error["message"] == "unknown tool 'identity.create_org'"
    d = r1.error["details"]
    assert d["suggestions"][0] == "id.create_org" and d["near_hosts"] == ["id"] and d["host_exists"] is False
    r2 = ex(w, A, "bank.transfer", from_account=acct, to="bob", amount=-5, memo=7)
    assert r2.error["code"] == "invalid_action"
    assert r2.error["message"] == "args.amount: below minimum 1; args.memo: expected string, got int"   # 3.0 text
    d2 = r2.error["details"]
    assert d2["tool"] == "bank.transfer" and [e["path"] for e in d2["errors"]] == ["args.amount", "args.memo"]
    assert d2["errors"][0]["expected"] == ">= 1" and d2["errors"][0]["given"] == -5 and d2["signature"]
    r3 = ex(w, A, "world.describe", host="banking")
    assert r3.error["message"] == "no service at host 'banking'" and r3.error["details"]["near_hosts"] == ["bank"]
    r4 = w.execute(A, {"tool": 5, "args": "x"})
    assert r4.error["message"] == K._MALFORMED and [e["path"] for e in r4.error["details"]["errors"]] == ["tool", "args"]
    r5 = ex(w, A, "bank.transfer", from_account=acct, to="nobody", amount=5)       # a handler's own refusal
    assert r5.error["code"] == "not_found" and "details" not in r5.error
    # what was recorded is exactly what 3.0 recorded
    rec = [e.payload["error"] for e in w.events_range(seq0) if e.type == "cmd.rejected"]
    assert rec == [{"code": "unknown_tool", "message": "unknown tool 'identity.create_org'"},
                   {"code": "invalid_action", "message": r2.error["message"], "details": {"tool": "bank.transfer"}},
                   {"code": "unknown_tool", "message": "no service at host 'banking'"},
                   {"code": "invalid_action", "message": K._MALFORMED},
                   r5.error]
    # rendering for prompts shows the detail an agent can act on
    line = D.render_result(r1.to_dict())
    assert "FAILED unknown_tool" in line and "id.create_org" in line
    line = D.render_result(r2.to_dict())
    assert "args.amount: expected >= 1, given -5" in line and "signature: bank.transfer(" in line
    assert D.render_result({"ok": True, "tool": "t", "result": {"a": 1}, "cost_s": 3}) == \
        R.render_result({"ok": True, "tool": "t", "result": {"a": 1}, "cost_s": 3})
    # switching the detail off gives the same world, event for event
    w0 = small(error_hints=False)
    acct0 = w0.modules["bank"].primary_account(w0.state, A)
    s0 = w0.seq
    q1 = ex(w0, A, "identity.create_org", handle="acme", name="Acme")
    q2 = ex(w0, A, "bank.transfer", from_account=acct0, to="bob", amount=-5, memo=7)
    ex(w0, A, "world.describe", host="banking")
    w0.execute(A, {"tool": 5, "args": "x"})
    ex(w0, A, "bank.transfer", from_account=acct0, to="nobody", amount=5)
    assert "details" not in q1.error and q2.error["details"] == {"tool": "bank.transfer"}
    strip = lambda evs: [(e.type, e.actor, e.payload, e.vis) for e in evs]  # noqa: E731
    assert strip(w0.events_range(s0)) == strip(w.events_range(seq0))


def test_discovery_actions_replay_regenerate_and_fork_exactly():
    w = tiny(cost_model="deterministic")       # executed-lines cost model: forks of a consumer economy are bit-identical
    acct = w.modules["bank"].primary_account(w.state, A)

    def drive(x, who):
        ex(x, who, "world.hosts")
        ex(x, who, "world.find_tools", query="organisation", detail=True)
        ex(x, who, "world.validate", tool="id.create_org", args={"handle": "x"})
        ex(x, who, "world.help", topic="errors")
        ex(x, who, "world.api")
        ex(x, who, "identity.create_org", handle="acme")
        ex(x, who, "id.create_org", handle="Bad Handle", name=3)
        ex(x, who, "world.describe", host="nowhere")
        x.execute(who, {"tool": None, "args": {}})
        ex(x, who, "search.query", q="route shortest")
        ex(x, who, "world.find_tools", query="shorte1")
        ex(x, who, "bank.transfer", from_account=acct if who == A else "acct_x", to="bob", amount=10)
        x.advance_time(x.now + 3600)
    drive(w, A)
    drive(w, B)
    assert w.check_invariants() == []
    events = log(w)
    assert World.replay(w.spec, events).digest() == w.digest()
    regen = World.regenerate(w.spec, events)
    assert [e.hash for e in regen.events_range(0)] == [e["hash"] for e in events] and regen.digest() == w.digest()
    f1, f2 = w.fork("f1"), w.fork("f2")
    for f in (f1, f2):
        drive(f, A)
    assert f1.digest() == f2.digest() and f1.last_hash == f2.last_hash and f1.digest() != w.digest()


# ====================================================================== catalogue renderer
def _hosts_in(text: str) -> set[str]:
    hs = set()
    for ln in text.splitlines():
        m = re.match(r"== ([\w.-]+) \(\d+\)", ln) or re.match(r"([\w.-]+) \(\d+\): ", ln)
        if m:
            hs.add(m.group(1))
        if ln.startswith("hosts: "):
            hs |= set(re.findall(r"([\w.-]+)\(\d+\)", ln))
    return hs


def _level(text: str) -> int:
    last = text.splitlines()[-1]
    marks = ["complete:", "descriptions longer than", "Omitted: all descriptions", "Omitted: argument types",
             "every tool is named", "(+k more per host)", "the names of all"]
    return next(i for i, m in enumerate(marks) if m in last)


def _synthetic(n_hosts: int, per_host: int, rng) -> list[dict]:
    words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma".split()
    tools = []
    for h in range(n_hosts):
        for e in range(per_host):
            props = {f"arg{k}": S.s_str(" ".join(rng.choice(words, 4))) for k in range(int(rng.integers(0, 5)))}
            desc = " ".join(rng.choice(words, int(rng.integers(3, 60))))
            tools.append({"name": f"h{h:02d}.ep_{e:02d}", "description": desc, "inputSchema": S.obj(props, list(props)[:1]),
                          "annotations": {"host": f"h{h:02d}", "costSeconds": int(rng.integers(0, 60)), "builtin": True}})
    return tools


def test_catalogue_renderer_never_drops_a_host_and_always_fits(tw):
    rng = np.random.default_rng(11)
    real = tw.available_tools(A)
    summ = D.host_summaries(tw)
    cats = {"real": (real, summ), "wide": (_synthetic(60, 30, rng), None), "few": (_synthetic(3, 4, rng), None),
            "deep": (_synthetic(8, 120, rng), None)}
    for label, (tools, hs) in cats.items():
        hosts = {t["annotations"]["host"] for t in tools}
        names = {t["name"] for t in tools}
        prev_level = None
        for budget in list(range(2000, 60001, 1000)):
            text = D.render_catalogue(tools, budget, hosts=hs)
            assert len(text) <= budget, (label, budget, len(text))
            assert _hosts_in(text) == hosts, (label, budget, sorted(hosts - _hosts_in(text)))
            last = text.splitlines()[-1]
            assert last.startswith(f"[catalogue: {len(tools)} tools on {len(hosts)} hosts") and last.endswith("]")
            lvl = _level(text)
            if lvl > 0:                                   # anything omitted is named, with where to get it
                assert "Omitted:" in last and "world.describe" in last and "world.find_tools" in last
            assert prev_level is None or lvl <= prev_level, (label, budget)    # more budget never means less detail
            prev_level = lvl
            if lvl <= 4:                                  # every tool is named
                body = text.rsplit("\n", 1)[0]
                if lvl <= 3:
                    shown = set(re.findall(r"^([\w.-]+\.\w+)\(", body, re.M))
                else:
                    shown = {f"{ln.split(' (')[0]}.{e}" for ln in body.splitlines() for e in ln.split(": ", 1)[1].split(", ")}
                assert shown == names, (label, budget, lvl)
            if lvl == 5:                                  # names that do not fit are counted, host by host
                hidden = sum(int(m) for m in re.findall(r"\+(\d+) more", text.rsplit("\n", 1)[0]))
                assert f"the names of {hidden} tools" in last and 0 < hidden <= len(tools)
            assert D.render_catalogue(tools, budget, hosts=hs) == text       # deterministic
        assert prev_level == 0 or label in ("wide", "deep")                   # 60,000 characters hold the real one whole


def test_catalogue_renderer_degrades_uniformly_and_keeps_whole_descriptions_when_they_fit(tw):
    tools = tw.available_tools(A)
    summ = D.host_summaries(tw)
    full = D.render_catalogue(tools, 10**6, hosts=summ)
    assert _level(full) == 0
    for t in tools:                               # nothing is cut to a first line: every description is whole
        assert D.flat(t["description"]) in full, t["name"]
        assert D.signature(t) in full
        for p in t["inputSchema"].get("properties", {}).values():
            if p.get("description"):
                assert D.flat(p["description"]) in full
    assert '{"entry": "service.py", "endpoints": {"name": {"handler": "fn", "params": <JSON schema>' in full
    # signatures agree with the 3.0 renderer (which the recorded prompts used)
    for t in tools:
        assert R.render_tool(t, 0) == D.signature(t) + (f" [{t['annotations']['costSeconds']}s]" if t["annotations"]["costSeconds"] else "")
    # one step down: every description is cut to the same length N, stated in the footer
    budget = len(D.render_catalogue(tools, 10**6, "descriptions", hosts=summ)) - 1500
    mid = D.render_catalogue(tools, budget, hosts=summ)
    assert _level(mid) == 1 and len(mid) <= budget
    n = int(re.search(r"longer than (\d+) characters", mid).group(1))
    cut = int(re.search(r"(\d+) of \d+ cut", mid).group(1))
    lines = {ln.split("(")[0]: ln for ln in mid.splitlines() if re.match(r"[\w.-]+\.\w+\(", ln)}
    n_cut = 0
    for t in tools:
        d = D.flat(t["description"])
        shown = lines[t["name"]].split(" - ", 1)[1] if " - " in lines[t["name"]] else ""
        assert len(shown) <= n
        if len(d) <= n:
            assert shown == d                     # short ones are untouched
        else:
            n_cut += 1
            assert shown.endswith("...") and d.startswith(shown[:-3].rstrip(" ,;:("))
    assert n_cut == cut > 0
    # N is the largest that fits: the next larger N first appears at a budget whose text would not have fitted
    for b2 in range(budget + 1, budget + 2000):
        nxt = D.render_catalogue(tools, b2, hosts=summ)
        if int(re.search(r"longer than (\d+) characters", nxt).group(1)) > n:
            assert len(nxt) > budget
            break
    else:
        raise AssertionError("N never grew with the budget")
    # explicit modes start the ladder lower; an unknown mode is an error
    assert _level(D.render_catalogue(tools, 10**6, "signatures")) == 2
    assert _level(D.render_catalogue(tools, 10**6, "names")) == 4
    assert _level(D.render_catalogue(tools, 900, "signatures")) >= 5
    with pytest.raises(ValueError):
        D.render_catalogue(tools, 1000, "everything")
    # priority tools keep signature and whole description below the full level, while that fits
    sig_only = D.render_catalogue(tools, 12_000, "signatures", priority=("cloud.deploy",))
    assert _level(sig_only) == 2 and '"entry": "service.py"' in sig_only
    assert '"entry": "service.py"' not in D.render_catalogue(tools, 12_000, "signatures")
    # budgets below the list of hosts: explicit about what is missing, never over budget
    for b in (0, 1, 20, 40, 80, 120, 200, 300, 400, 600):
        text = D.render_catalogue(tools, b, hosts=summ)
        assert len(text) <= b
        if b >= 40 and _hosts_in(text) != {t["annotations"]["host"] for t in tools}:
            assert "not shown" in text


def test_first_line_truncation_of_3_0_is_what_the_renderer_fixes(tw):
    """The 3.0 tool list showed only the first line of a description (api.render.render_tool); that default is
    unchanged, and it is exactly what hides the deployment manifest."""
    spec = next(t for t in tw.available_tools(A) if t["name"] == "cloud.deploy")
    old = R.render_tool(spec, 160)
    assert old == ("cloud.deploy(host: str, instance: str, repo: str, ref?: str) [120s] - Deploy a repository as a "
                   "service at <host>.crucible. Requires crucible.json:")
    assert '"entry"' not in old
    new = D.render_catalogue([spec], 2000)
    assert '"entry": "service.py"' in new and "fn(args, caller)" in new
    # api.render gained an opt-in for the whole description; its default output is what it was
    whole = R.render_tool(spec, None, whole_description=True)
    assert whole.startswith(old) and whole.endswith("Each handler is called as fn(args, caller) and must return JSON.")
    assert D.flat(spec["description"]) in whole and "\n" not in whole
    assert len(R.render_tool(spec, 100, whole_description=True).split(" - ", 1)[1]) == 100
    tools = tw.available_tools(A)
    assert R.render_tools(tools) == "\n".join(R.render_tool(t) for t in tools)
    obs = {"time": 0, "principal": A, "tools": tools, "tools_changed": True}
    assert R.render_observation(obs) == R.render_observation(obs, whole_descriptions=False)
    assert '"entry": "service.py"' not in R.render_observation(obs)
    assert '"entry": "service.py"' in R.render_observation(obs, description_chars=None, whole_descriptions=True)


# ====================================================================== text helpers
def test_edit_distance_and_near_names():
    assert D.edit_distance("transfer", "transfer") == 0 and D.edit_distance("transfer", "transfr") == 1
    assert D.edit_distance("tranfser", "transfer") == 1                  # adjacent transposition
    assert D.edit_distance("abc", "xyz") == 3 and D.edit_distance("", "abc") == 3
    assert D.edit_distance("a" * 30, "b" * 30, cap=3) == 4               # capped: "more than 3"
    rng = np.random.default_rng(3)
    for _ in range(300):                                                 # symmetric; triangle with a pivot; cap is exact
        a, b, c = ("".join(rng.choice(list("abcd"), int(rng.integers(0, 9)))) for _ in range(3))
        d = D.edit_distance(a, b)
        assert d == D.edit_distance(b, a) and (d == 0) == (a == b)
        assert d <= max(len(a), len(b)) and d >= abs(len(a) - len(b))
        assert D.edit_distance(a, c) <= d + D.edit_distance(b, c) + 1    # OSA is not a metric: slack of 1 suffices here
        for cap in (0, 1, 2, 5):
            assert D.edit_distance(a, b, cap) == (d if d <= cap else cap + 1)
    assert D.near_names("bank.transfr", ["bank.transfer", "bank.balance", "mail.send"]) == ["bank.transfer"]
    assert D.tokenize("Listing accounts, listed pricing; prices") == ["list", "account", "list", "pric", "pric"]
    assert D.first_sentence("Deploy it at <host>.crucible. Requires a manifest, e.g. x. More.") == "Deploy it at <host>.crucible."
