"""Failure diagnosis (research/diagnostics.py).

Small real worlds are driven by purpose-built scripted policies that each commit one kind of failure - calling
tools that do not exist, looping, replying with malformed JSON through the text action parser, overspending,
deploying broken code, acting sensibly without moving the goal, losing customers to a rival - and the diagnosis
of the resulting event logs is checked class by class, with its evidence, counts and rates. The same checks run
on the recorded Crucible 3.0 LLM trajectories.

Tests that execute agent code need the Crucible sandbox (run them outside the tool sandbox).
"""
from __future__ import annotations

import copy
import glob
import json
import os

import crucible

crucible.load_all_modules()
from crucible.agents.base import Runner  # noqa: E402
from crucible.agents.llm import GenResult, LLMPolicy  # noqa: E402
from crucible.agents.scripted import Builder, IdlePolicy, ScriptPolicy, call, service_files  # noqa: E402
from crucible.core import discovery as D  # noqa: E402
from crucible.core.kernel import World  # noqa: E402
from crucible.economy.needs import NEEDS, best_tier, tier_source  # noqa: E402
from crucible.goals.metrics import m_independent_revenue  # noqa: E402
from crucible.research import diagnostics as X  # noqa: E402
from crucible.training.trajectories import step_records  # noqa: E402
from crucible.worldgen import generate  # noqa: E402

DAY = 86400
EXPERIMENTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "experiments")


def tiny(n_consumers=300, seed=3, **params):
    ov = {"n_consumers": n_consumers}
    if params:
        ov["params"] = params
    return World.create(generate({"family": "tiny", "overrides": ov}, seed))


def join(w, *handles, cash=1_000_000):
    for h in handles:
        w.system_call("admin.join", {"handle": h, "endowment": cash})
    return [f"p:{h}" for h in handles]


def event_at(w, seq):
    return w.events_range(seq - 1, seq)[0]


def check_shape(w, actor, d):
    """Invariants of every diagnosis: one class per attempt, counts and rates add up, evidence points at the
    actor's own command records, the whole thing is JSON."""
    json.dumps(d, allow_nan=False)
    acts = d["actions"]
    assert d["counts"]["turns"] == len(acts) == sum(c["n"] for c in d["classes"].values())
    assert abs(sum(c["rate"] for c in d["classes"].values()) - 1.0) < 1e-3 or not acts
    assert sum(f["n"] for f in d["families"].values()) == len(acts)
    assert all(a["class"] in X.CLASSES for a in acts)
    for a in acts:
        if a["seq"] is not None and w is not None:
            e = event_at(w, a["seq"])
            assert e.type in ("cmd", "cmd.rejected") and (e.actor == actor or e.payload.get("actor") == actor)
            assert (e.type == "cmd") == bool(a["ok"])
    for c in d["classes"].values():
        assert len(c["evidence"]) <= X.EVIDENCE_MAX
    assert d["primary"] and isinstance(d["not_determinable"], list)


# ====================================================================== scripted failure policies
class Nonexistent(ScriptPolicy):
    """Calls tools and hosts that do not exist: some are near misses of real tools, some are inventions."""
    name = "nonexistent"
    CALLS = [("identity.create_org", {"handle": "nova", "name": "Nova"}),      # host alias     -> id.create_org
             ("id.create_organization", {"handle": "nova"}),                   # same words     -> id.create_org
             ("bank.transfr", {"to": "bob", "amount": 5}),                     # typo           -> bank.transfer
             ("cloud.deploy_service", {}),                                     # extra word     -> cloud.deploy
             ("marketplace.list", {}),                                         # host extension -> market.list
             ("world.describe", {"host": "identity"}),                         # host alias     -> id
             ("bank.send_money", {"to": "bob", "amount": 5}),                  # real host, invented endpoint
             ("shop.buy", {"item": "x"}), ("payments.pay", {}), ("hr.hire", {"who": "bob"}),   # invented hosts
             ("world.describe", {"host": "organizations"})]                    # a host that does not exist

    def script(self):
        for t, a in self.CALLS:
            yield call(t, **a)


class Looper(ScriptPolicy):
    """One valid read, then the same invalid call twelve times, then the same post six times."""
    name = "looper"

    def script(self):
        yield call("bank.accounts")
        for _ in range(12):
            yield call("id.create_org", handle="p_nova", name="Nova")
        for _ in range(6):
            yield call("forum.post", board="market", title="Hello", body="Hello everyone, I am new here.")


class Overspender(ScriptPolicy):
    """Spends nearly everything, then keeps trying to pay amounts it no longer has."""
    name = "overspender"

    def script(self):
        me = yield from self.personal_account()
        yield call("bank.transfer", from_account=me, to="bob", amount=40_000)
        yield call("bank.transfer", from_account=me, to="bob", amount=9_000)
        for amt in (50_000, 20_000, 5_000, 2_000, 1_500, 1_200):
            yield call("bank.transfer", from_account=me, to="bob", amount=amt)
        yield call("bank.balance", account=me)


class BrokenCoder(ScriptPolicy):
    """Founds a company and ships code that does not work at every stage."""
    name = "broken"
    MANIFEST = {"entry": "service.py", "endpoints": {"solve": {"handler": "solve", "params": {
        "type": "object", "additionalProperties": True}}}}

    def script(self):
        org, acct, _ = yield from self.found_company("labs", 300_000)
        repo = (yield call("code.create_repo", name="svc", owner=org))["result"]["repo"]
        yield call("code.commit", repo=repo, files={
            "service.py": "def solve(args, caller):\n    return undefined_name\n",
            "tests/test_a.py": "def test_x():\n    assert 1 == 2\n"})
        yield call("code.test", repo=repo)                                         # tests fail
        yield call("code.run", repo=repo, target="service:solve", input={})        # program raises
        prov = sorted(p["provider"] for p in (yield call("cloud.providers"))["result"])[0]
        inst = (yield call("cloud.rent", provider=prov, cores=1, account=acct, owner=org))["result"]["instance"]
        yield call("cloud.deploy", repo=repo, host="brokensvc", instance=inst)      # no manifest
        yield call("code.commit", repo=repo, files={"crucible.json": json.dumps(dict(self.MANIFEST, entry="gone.py"))})
        yield call("cloud.deploy", repo=repo, host="brokensvc", instance=inst)      # entry missing
        yield call("code.commit", repo=repo, files={"crucible.json": json.dumps(self.MANIFEST),
                                                    "service.py": "import nonexistent_module_xyz\n"})
        yield call("cloud.deploy", repo=repo, host="brokensvc", instance=inst)      # import probe fails
        yield call("code.commit", repo=repo, files={"service.py": "def solve(args, caller):\n    raise ValueError('boom')\n"})
        yield call("cloud.deploy", repo=repo, host="brokensvc", instance=inst)      # live, raises on every call
        for i in range(5):
            yield call("brokensvc.solve", n=i)


class Idler(ScriptPolicy):
    """Valid, varied, well-formed activity that has nothing to do with earning revenue."""
    name = "idler"

    def script(self):
        me = yield from self.personal_account()
        yield call("world.hosts")
        yield call("bank.balance", account=me)
        yield call("forum.boards")
        yield call("search.query", q="route shortest")
        yield call("market.browse", kind="api", limit=5)
        yield call("id.update_profile", bio="I am new here.")
        yield call("forum.post", board="market", title="Hello", body="Hello, I am looking around.")
        yield call("mail.send", to=["bob"], subject="hi", body="Hello Bob")
        yield call("world.clock")
        yield call("docs.create", title="notes", body="things I saw")
        yield call("cloud.providers")
        yield call("jobs.openings")
        yield call("forum.list", board="market")
        yield call("world.find_tools", query="company")
        yield call("id.me")


class Mixed(ScriptPolicy):
    """A known number of every kind of rejection, between valid actions."""
    name = "mixed"

    def script(self):
        me = yield from self.personal_account()                                              # ok_read
        yield call("world.clock")                                                            # ok_read
        yield call("identity.create_org", handle="mix-co", name="Mix")                       # unknown_tool / near_name
        yield call("hr.hire", who="bob")                                                     # unknown_tool / unknown_host
        yield call("id.create_org", handle="Mix Co", name="Mix")                             # schema (pattern)
        yield call("id.create_org", handle="mix-co")                                         # schema (required)
        yield call("id.create_org", handle="mix-co", name="Mix", kind="company")             # ok_effect
        yield call("id.create_org", handle="mix-co", name="Mix", kind="company")             # conflict
        yield call("bank.balance", account="acct_999999")                                    # nonexistent
        yield call("bank.transfer", from_account="acct_000001", to="bob", amount=1)          # permission
        yield call("bank.transfer", from_account=me, to="bob", amount=10**9)                 # resource (funds)
        yield call("bank.transfer", from_account=me, to="bob", amount=100)                   # ok_effect
        yield {"tool": "bank.accounts", "args": "not an object"}                             # syntax (malformed)
        yield call("forum.post", board="market", title="Hi", body="Hello there")             # ok_effect
        yield call("bank.transfer", from_account=me, to=me, amount=5)                        # rule
        yield call("bank.accounts")                                                          # ok_read


EXPECTED_MIXED = {"ok_read": 3, "ok_effect": 3, "unknown_tool": 2, "schema": 2, "conflict": 1, "nonexistent": 1,
                  "permission": 1, "resource": 1, "syntax": 1, "rule": 1}


def run(w, agents, rounds, dt=3600):
    since = w.seq
    r = Runner(w, agents, dt=dt, seed=1)
    r.run(rounds)
    return since, r


# ====================================================================== one failure kind per policy
def test_unknown_tools_split_into_discovery_failures_and_inventions():
    w = tiny()
    nova, _ = join(w, "nova", "bob")
    since, _ = run(w, [(nova, Nonexistent())], 12)
    d = X.diagnose(w, nova, since_seq=since)
    check_shape(w, nova, d)
    assert d["counts"] == {"turns": 11, "attempts": 11, "accepted": 0, "rejected": 11, "unparsed": 0, "waits": 0,
                           "no_action": 0, "distinct_tools": 10}
    assert d["classes"]["unknown_tool"]["n"] == 11 and d["classes"]["unknown_tool"]["rate"] == 1.0
    assert d["classes"]["unknown_tool"]["subs"] == {"near_name": 6, "unknown_host": 4, "no_near_name": 1}
    near = {a["tool"] if a["tool"] != "world.describe" else "describe:" + a["message"].split("'")[1]: a["near"]
            for a in d["actions"]}
    assert near["identity.create_org"] == ["id.create_org"] and near["id.create_organization"] == ["id.create_org"]
    assert near["bank.transfr"] == ["bank.transfer"] and near["cloud.deploy_service"][0] == "cloud.deploy"
    assert near["marketplace.list"] == ["market.list"] and near["describe:identity"] == ["id"]
    for invented in ("bank.send_money", "shop.buy", "payments.pay", "hr.hire", "describe:organizations"):
        assert near[invented] == [], invented
    by = {a["tool"]: a["sub"] for a in d["actions"] if a["tool"] != "world.describe"}
    assert by["bank.send_money"] == "no_near_name" and by["shop.buy"] == "unknown_host"
    # families: a near name is a discovery failure, an invention is a wrong belief about the world
    assert d["families"]["discovery"] == {"n": 6, "rate": round(6 / 11, 4)}
    assert d["families"]["belief"] == {"n": 5, "rate": round(5 / 11, 4)}
    assert d["primary"] == "unknown_tool/near_name" and d["loops"]["status"] == "absent"
    assert d["code"]["attempted"] is False and d["resources"]["status"] == "absent"
    # the near names are exactly what the world itself would have suggested first
    for a in d["actions"]:
        if a["tool"] != "world.describe" and a["near"]:
            assert D.suggest(w, nova, a["tool"])["suggestions"][: len(a["near"])] == a["near"]


def test_loop_detector_flags_repeats_without_new_information():
    w = tiny()
    loop, _ = join(w, "loop", "bob")
    since, _ = run(w, [(loop, Looper())], 20)
    d = X.diagnose(w, loop, since_seq=since)
    check_shape(w, loop, d)
    acts = d["actions"]
    assert [a["tool"] for a in acts] == ["bank.accounts"] + ["id.create_org"] * 12 + ["forum.post"] * 6
    # the third identical outcome of the same call is the first repeat without new information
    assert [("loop" in a["flags"]) for a in acts] == [False] + [False, False] + [True] * 10 + [False, False] + [True] * 4
    lp = d["loops"]
    assert lp["status"] == "established" and lp["actions"] == 14 and lp["rate"] == round(14 / 19, 4)
    assert lp["by_tool"] == {"id.create_org": 10, "forum.post": 4} and lp["longest_run"] == 10 and lp["runs"] == 2
    kinds = {r["tool"]: r["kind"] for r in lp["run_list"]}
    assert kinds == {"id.create_org": "repeated_rejection", "forum.post": "repeated_effect"}
    assert lp["evidence"] == [a["seq"] for a in acts if "loop" in a["flags"]][: X.EVIDENCE_MAX]
    assert acts[3]["loop_of"] == [acts[1]["seq"], acts[2]["seq"]]
    assert d["classes"]["schema"] == {"n": 12, "rate": round(12 / 19, 4), "evidence": [a["seq"] for a in acts[1:13]],
                                      "subs": {"pattern": 12}, "tools": {"id.create_org": 12}}
    assert d["primary"] == "schema" and d["strategy"]["valid_and_varied"] is False


def test_near_same_definition_and_repeated_reads_need_results():
    A = lambda tool, **args: {"tool": tool, "args": args}  # noqa: E731
    assert X.near_same(A("t.x", a=1, b="k"), A("t.x", b="k", a=1))
    assert not X.near_same(A("t.x", a=1), A("t.y", a=1))
    assert not X.near_same(A("ads.stats", campaign="adc_000001"), A("ads.stats", campaign="adc_000002"))   # another id
    assert not X.near_same(A("bank.transfer", to="bob", amount=5), A("bank.transfer", to="bob", amount=6))  # another number
    assert not X.near_same(A("code.commit", repo="r", files={"a": "1"}), A("code.commit", repo="r", files={"a": "2"}))
    long1 = "Get genuine and unbiased product reviews to make informed decisions."
    long2 = "Get genuine and unbiased product reviews to make informed purchasing decisions."
    assert X.near_same(A("market.list", title="R", description=long1), A("market.list", title="R", description=long2))
    assert not X.near_same(A("market.list", title="R", description=long1),
                           A("market.list", title="R", description="A fast route planner for delivery fleets and couriers."))
    assert not X.near_same(A("forum.post", body="hello 0"), A("forum.post", body="hello 1"))               # short texts
    assert not X.near_same(A("t.x", a=1), A("t.x", a=1, b=2))

    class Poller(ScriptPolicy):
        def script(self):
            me = yield from self.personal_account()
            for _ in range(5):
                yield call("bank.balance", account=me)          # nothing changes: the same answer five times
            for i in range(5):
                yield call("bank.transfer", from_account=me, to="bob", amount=10 + i)
                yield call("bank.balance", account=me)          # the balance changed: each answer is new
    w = tiny()
    poll, _ = join(w, "poll", "bob")
    since, r = run(w, [(poll, Poller())], 18)
    blind = X.diagnose(w, poll, since_seq=since)
    # the log does not record results: repeated reads are counted, not called loops, and the report says so
    assert blind["loops"]["actions"] == 0 and blind["loops"]["unverified_reads"] == 8
    assert any(x.startswith("loop: 8 repeated reads") for x in blind["not_determinable"])
    seen = X.diagnose(w, poll, since_seq=since, trajectory=step_records(r))
    flags = [("loop" in a["flags"]) for a in seen["actions"] if a["tool"] == "bank.balance"]
    assert flags == [False, False, True, True, True] + [False] * 5      # only the polls that returned nothing new
    assert seen["loops"]["unverified_reads"] == 0 and seen["loops"]["actions"] == 3
    assert seen["loops"]["run_list"][0]["kind"] == "repeated_read"


def test_malformed_replies_through_the_text_action_parser_are_syntax_failures():
    from crucible.agents.llm import EchoLLM
    replies = ["I think I should look around first.",                       # prose: no tool call
               '{"tool": "bank.accounts", "args": {}}',
               '{"tool": "bank.balance", "args": {"account": ',             # cut off
               "```json\n{tool: bank.accounts}\n```",                       # not JSON
               '{"tool": "wait", "args": {}}',                              # explicit no-op
               '{"tool": "world.clock"}',
               "<tool_call>bank accounts</tool_call>",                      # tag without JSON
               '{"args": {"account": "x"}}',                                # no tool
               '{"tool": "bank.accounts", "args": {}}']
    w = tiny()
    nova, _ = join(w, "nova", "bob")
    pol = LLMPolicy(backend=EchoLLM(replies), goal="test", max_new_tokens=64)      # the real prompt/parse/record path
    since, _ = run(w, [(nova, pol)], len(replies))
    assert pol.usage()["parse_failures"] == 5 and pol.usage()["waits"] == 1
    d = X.diagnose(w, nova, since_seq=since, transcript=pol.transcript, usage=pol.usage())
    check_shape(w, nova, d)
    assert [a["class"] for a in d["actions"]] == ["syntax", "ok_read", "syntax", "syntax", "wait", "ok_read", "syntax",
                                                  "syntax", "ok_read"]
    assert d["counts"] == {"turns": 9, "attempts": 8, "accepted": 3, "rejected": 0, "unparsed": 5, "waits": 1,
                           "no_action": 0, "distinct_tools": 2}
    assert d["classes"]["syntax"] == {"n": 5, "rate": round(5 / 9, 4), "evidence": [], "subs": {"unparseable": 5}}
    assert d["families"]["form"]["n"] == 5 and d["families"]["idle"]["n"] == 1
    # unparseable replies never reached the world: they have no sequence number, and the log has no trace of them
    assert all(a["seq"] is None for a in d["actions"] if a["class"] in ("syntax", "wait"))
    assert len([e for e in w.events_range(since) if e.type in ("cmd", "cmd.rejected") and nova in (e.actor, e.payload.get("actor"))]) == 3
    assert [a["turn"] for a in d["actions"]] == list(range(9))
    assert d["primary"] == "syntax"
    # without the transcript the log alone shows three valid reads and nothing else: the failure is invisible
    blind = X.diagnose(w, nova, since_seq=since)
    assert blind["counts"]["turns"] == 3 and "syntax" not in blind["classes"]
    # a malformed action that does reach the kernel is recorded there, and classified the same way
    r = w.execute(nova, {"tool": 7, "args": {}})
    assert not r.ok
    d2 = X.diagnose(w, nova, since_seq=since)
    assert d2["classes"]["syntax"]["subs"] == {"malformed_action": 1} and d2["classes"]["syntax"]["evidence"] == r.events[:1]


class TruncatingReader:
    """Test double for a model with a small context. It truncates exactly like `agents.llm.HFBackend` (first third
    and the tail of the token sequence; whitespace tokens here) and then *reads what is left*: it founds an
    organisation with the real tool only if that tool's signature is still in its prompt, and otherwise calls
    the tool it imagines."""
    name = "truncating-reader"

    def __init__(self, max_context: int):
        self.max_context = max_context
        self.done = False

    def generate(self, messages, max_new_tokens=256, temperature=0.0, seed=None) -> GenResult:
        toks = " \n ".join(m["content"] for m in messages).split()
        room = self.max_context - max_new_tokens
        truncated = len(toks) > room
        if truncated:
            toks = toks[: room // 3] + toks[-(room - room // 3):]
        visible = " ".join(toks)
        if self.done or "id.create_org ->" in visible and "ok" in visible.split("id.create_org ->")[-1][:40]:
            self.done = True
            text = '{"tool": "wait", "args": {}}'
        elif "id.create_org(handle: str" in visible:
            text = '{"tool": "id.create_org", "args": {"handle": "nova-co", "name": "Nova"}}'
        else:
            text = '{"tool": "org.create", "args": {"name": "Nova"}}'
        return GenResult(text, len(toks), len(text.split()), truncated)


def test_context_truncation_is_measured_per_turn_and_explains_the_unknown_tools():
    out = {}
    for label, ctx in (("small", 1200), ("large", 100_000)):
        w = tiny()
        nova, _ = join(w, "nova", "bob")
        be = TruncatingReader(ctx)
        pol = LLMPolicy(backend=be, goal="Found an organisation.", max_new_tokens=64, tool_chars=60_000)
        since, _ = run(w, [(nova, pol)], 6)
        d = X.diagnose(w, nova, since_seq=since, transcript=pol.transcript, usage=pol.usage(), context_limit=ctx,
                       max_new_tokens=64, goal={"success": bool(w.state.find("members", "person", nova))})
        check_shape(w, nova, d)
        out[label] = (d, pol, w, nova)
    d, pol, w, nova = out["small"]
    cx = d["context"]
    assert pol.usage()["truncated_prompts"] == 6
    assert cx["status"] == "established" and cx["per_turn"] is True and cx["truncated_turns"] == 6 and cx["rate"] == 1.0
    assert cx["max_prompt_tokens"] == 1200 - 64 and cx["evidence_turns"] == list(range(6))
    assert all("context_truncated" in a["flags"] for a in d["actions"])
    assert d["classes"]["unknown_tool"]["n"] == 6 and d["classes"]["unknown_tool"]["subs"] == {"unknown_host": 6}
    assert d["primary"] == "context_truncation" and not w.state.find("members", "person", nova)
    # same reader, same world, a context that holds the prompt: the tool is found and nothing is truncated
    d, pol, w, nova = out["large"]
    assert pol.usage()["truncated_prompts"] == 0 and d["context"]["status"] == "absent" and d["context"]["rate"] == 0.0
    assert "unknown_tool" not in d["classes"] and w.state.find("members", "person", nova)
    assert d["primary"] == "none" and d["actions"][0]["tool"] == "id.create_org" and d["actions"][0]["class"] == "ok_effect"
    # usage counters alone give the episode rate but not the turns; nothing at all is "not determinable"
    d, pol, w, nova = out["small"]
    agg = X.diagnose(w, nova, transcript=None, usage=pol.usage(), since_seq=0)
    assert agg["context"]["status"] == "established" and agg["context"]["per_turn"] is False and agg["context"]["rate"] == 1.0
    assert not any("context_truncated" in a["flags"] for a in agg["actions"])
    none = X.diagnose(w, nova)
    assert none["context"]["status"] == "not_determinable"
    assert any(x.startswith("context_truncation:") for x in none["not_determinable"])
    # a transcript that records truncation per turn says so without a context limit ...
    flagged = X.diagnose(w, nova, transcript=pol.transcript)
    assert flagged["context"]["status"] == "established" and flagged["context"]["truncated_turns"] == 6
    # ... one in the 3.0 format (no per-turn flag) and without a context limit cannot say which turns were truncated
    old_format = [{k: v for k, v in r.items() if k != "truncated"} for r in pol.transcript]
    half = X.diagnose(w, nova, transcript=old_format)
    assert half["context"]["status"] == "not_determinable"


def test_overspending_is_resource_exhaustion_with_the_cash_trace():
    w = tiny()
    spend, bob = join(w, "spend", "bob", cash=50_000)
    since, _ = run(w, [(spend, Overspender())], 12)
    d = X.diagnose(w, spend, since_seq=since)
    check_shape(w, spend, d)
    assert d["classes"]["resource"]["n"] == 6 and d["classes"]["resource"]["subs"] == {"funds": 6}
    res = d["resources"]
    assert res["status"] == "established" and res["funds"]["n"] == 6 and res["funds"]["rate"] == round(6 / 10, 4)
    bank = w.modules["bank"]
    left = bank.balance(w.state, bank.primary_account(w.state, spend))
    assert res["cash"]["start"] == 50_000 and res["cash"]["end"] == left == 1_000 and res["cash"]["min"] == 1_000
    assert res["cash"]["spent_by_kind"] == {"transfer": 49_000}
    assert event_at(w, res["cash"]["min_seq"]).type == "bank.txn"
    assert any(r.startswith("funds exhausted: 1000 cents left of at most 50000") for r in res["reasons"])
    assert all(event_at(w, s).payload["error"]["code"] == "insufficient_funds" for s in res["funds"]["evidence"])
    assert d["primary"] == "resource_exhaustion" and d["families"]["resource"]["n"] == 6
    # the payee received money and overspent nothing
    other = X.diagnose(w, bob, since_seq=since)
    assert other["resources"]["status"] == "absent" and other["resources"]["cash"]["end"] == 99_000
    # a refused payment from one account while the actor still holds funds elsewhere is not exhaustion
    w2 = tiny()
    rich, _ = join(w2, "rich", "bob", cash=50_000)
    acct = w2.modules["bank"].primary_account(w2.state, rich)
    for amt in (60_000, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110):
        w2.execute(rich, {"tool": "bank.transfer", "args": {"from_account": acct, "to": "bob", "amount": amt}})
    d2 = X.diagnose(w2, rich)
    assert d2["classes"]["resource"]["n"] == 1 and d2["resources"]["status"] == "absent" and "note" in d2["resources"]


def test_rate_limits_and_compute_capacity_are_resource_failures_too():
    w = tiny(max_actions_per_hour=5)
    fast, _ = join(w, "fast", "bob")
    since = w.seq
    for _ in range(12):                                     # twelve actions inside one simulated hour, limit five
        w.execute(fast, {"tool": "world.clock", "args": {}})
    d = X.diagnose(w, fast, since_seq=since)
    check_shape(w, fast, d)
    assert {c: v["n"] for c, v in d["classes"].items()} == {"ok_read": 5, "resource": 7}
    assert d["classes"]["resource"]["subs"] == {"rate_limit": 7} and d["resources"]["rate_limit"]["rate"] == round(7 / 12, 4)
    assert d["resources"]["status"] == "established" and d["resources"]["reasons"] == ["rate limit: 7 of 12 attempts refused"]
    assert d["primary"] == "resource_exhaustion" and d["loops"]["actions"] == 5       # and it kept hammering the limit
    assert [a["handler_ran"] for a in d["actions"]] == [True] * 5 + [False] * 7
    # compute: asking a provider for more cores than it has free
    w2 = tiny()
    big, _ = join(w2, "big", "bob")
    acct = w2.modules["bank"].primary_account(w2.state, big)
    prov = sorted(p["provider"] for p in w2.query(big, "cloud.providers"))[0]
    free = next(p["cores_free"] for p in w2.query(big, "cloud.providers") if p["provider"] == prov)
    assert free < 64
    since = w2.seq
    for _ in range(3):
        r = w2.execute(big, {"tool": "cloud.rent", "args": {"provider": prov, "cores": 64, "account": acct}})
        assert r.error["code"] == "capacity_exceeded"
    d2 = X.diagnose(w2, big, since_seq=since)
    assert d2["classes"]["resource"]["subs"] == {"capacity": 3} and d2["resources"]["capacity"]["n"] == 3
    assert d2["resources"]["status"] == "established" and d2["resources"]["cash"]["end"] == d2["resources"]["cash"]["start"]
    assert d2["loops"]["actions"] == 1 and d2["loops"]["run_list"][0]["kind"] == "repeated_rejection"


def test_broken_code_is_a_code_failure_at_every_stage():
    w = tiny()
    brk, _ = join(w, "brk", "bob")
    since, _ = run(w, [(brk, BrokenCoder())], 26)
    d = X.diagnose(w, brk, since_seq=since)
    check_shape(w, brk, d)
    subs = d["classes"]["code"]["subs"]
    assert subs == {"own_service_error": 5, "manifest_rejected": 2, "deploy_failed": 1, "program_error": 1, "tests_failed": 1}
    code = d["code"]
    assert code["status"] == "established" and code["attempted"] is True
    assert (code["tests"], code["tests_failed"], code["runs"], code["runs_failed"]) == (1, 1, 1, 1)
    assert (code["deploys"], code["deploys_failed"], code["manifest_rejections"], code["commits"]) == (2, 1, 2, 4)
    assert code["requests"] == {"ok": 0, "error": 5, "overload": 0, "unavailable": 0} and code["live_deployments"] == 1
    assert "its deployed code raised on 5 of 5 executed requests" in code["reasons"]
    assert code["current_deployments"]["requests"] == {"ok": 0, "error": 5}
    assert "the last test run of every tested repository failed" in code["reasons"]
    assert "every program it ran raised" in code["reasons"]
    by = {a["sub"]: a for a in d["actions"] if a["class"] == "code"}
    assert by["tests_failed"]["tool"] == "code.test" and by["program_error"]["tool"] == "code.run"
    assert by["deploy_failed"]["tool"] == "cloud.deploy" and by["manifest_rejected"]["tool"] == "cloud.deploy"
    # evidence: the usage records of the failed requests on its own deployment
    usage = [event_at(w, s) for a in d["actions"] if a["sub"] == "own_service_error" for s in a["evidence"][1:]]
    assert len(usage) == 5 and all(e.type == "cloud.usage" and e.payload["outcome"] == "error" for e in usage)
    assert d["primary"] == "code_failure" and d["families"]["code"]["n"] == 10
    # the same broken service called by somebody else is a failure of the callee, not of the caller's code
    bob = "p:bob"
    seq0 = w.seq
    w.execute(bob, {"tool": "world.describe", "args": {"host": "brokensvc"}})
    for i in range(3):
        w.execute(bob, {"tool": "brokensvc.solve", "args": {"n": i}})
    db = X.diagnose(w, bob, since_seq=seq0)
    assert db["classes"]["service"] == {"n": 3, "rate": 0.75, "evidence": db["classes"]["service"]["evidence"],
                                       "subs": {"callee_error": 3}, "tools": {"brokensvc.solve": 3}}
    assert db["code"]["attempted"] is False and db["code"]["status"] == "absent"


def test_sensible_idling_is_strategic_failure_and_difficulty_needs_a_competent_reference():
    base = tiny()
    idle, ref, _ = join(base, "idle", "ref", "bob")
    horizon = 4 * 24
    fa, fb, fc = base.fork("ref-run"), base.fork("idle-run"), base.fork("null-run")
    Runner(fa, [(ref, Builder(need="route.shortest"))], dt=3600, seed=1).run(horizon)     # the competent reference
    Runner(fb, [(idle, Idler())], dt=3600, seed=1).run(horizon)
    Runner(fc, [(ref, IdlePolicy())], dt=3600, seed=1).run(horizon)
    earned = m_independent_revenue(fa, "o:ref-labs", 30 * DAY)
    assert earned > 0 and m_independent_revenue(fb, idle, 30 * DAY) == 0

    def reference(threshold, competent=True, world=fa, name="builder"):
        got = m_independent_revenue(world, "o:ref-labs", 30 * DAY) if world is fa else 0.0
        return {"name": name, "competent": competent, "success": got >= threshold, "progress": min(1.0, got / threshold)}
    feasible, infeasible = 1.0, 1e12          # cents of independent revenue within the horizon
    goal = {"success": False, "progress": 0.0}
    d = X.diagnose(fb, idle, goal=goal)
    check_shape(fb, idle, d)
    st = d["strategy"]
    assert st["status"] == "established" and st["valid_and_varied"] and st["valid_share"] == 1.0
    assert st["distinct_tools"] == 16 and st["loop_rate"] == 0.0 and d["loops"]["status"] == "absent"
    assert set(d["classes"]) == {"ok_effect", "ok_read"} and d["code"]["attempted"] is False
    # no reference: strategic failure, with difficulty explicitly not established
    assert d["primary"] == "strategic_failure" and d["difficulty"]["status"] == "not_established"
    assert "whether the task is feasible is not established" in d["primary_reason"]
    assert any(x.startswith("task_difficulty: not established") for x in d["not_determinable"])
    # a competent reference met the goal in the same world: the task is feasible, the failure is the agent's
    easy = X.diagnose(fb, idle, goal=goal, references=[reference(feasible)])
    assert easy["difficulty"]["status"] == "absent" and easy["difficulty"]["succeeded"] == ["builder"]
    assert easy["primary"] == "strategic_failure" and "a competent reference succeeded" in easy["primary_reason"]
    # the competent reference also failed: task difficulty
    hard = X.diagnose(fb, idle, goal=goal, references=[reference(infeasible)])
    assert hard["difficulty"]["status"] == "established" and hard["primary"] == "task_difficulty"
    assert 0 < hard["difficulty"]["best_reference_progress"] < 1e-3
    # a reference that is not competent (it does nothing) establishes nothing, whatever it scored
    null = X.diagnose(fb, idle, goal=goal, references=[reference(infeasible, competent=False, world=fc, name="idle")])
    assert null["difficulty"]["status"] == "not_established" and null["primary"] == "strategic_failure"
    # without the goal's outcome the detector does not guess
    unk = X.diagnose(fb, idle)
    assert unk["strategy"]["status"] == "not_determinable" and unk["primary"] == "not_determinable"
    assert X.diagnose(fb, idle, goal={"success": True})["primary"] == "none"
    # the reference itself: working code, live service, no failure class
    dr = X.diagnose(fa, ref, goal={"success": True})
    assert dr["code"]["status"] == "absent" and dr["code"]["live_deployments"] == 1 and dr["code"]["requests"]["ok"] > 0
    assert dr["code"]["quality"]["mean_q"] > 0.9 and dr["primary"] == "none"
    # an actor that never acted
    never = X.diagnose(fc, ref, goal=goal)
    assert never["counts"]["turns"] == 0 and never["primary"] == "no_activity" and never["actions"] == []


def test_mixed_episode_gets_the_right_breakdown_and_agrees_across_sources():
    w = tiny()
    mix, _ = join(w, "mix", "bob", cash=50_000)
    since, r = run(w, [(mix, Mixed())], 18)
    d = X.diagnose(w, mix, since_seq=since, goal={"success": False})
    check_shape(w, mix, d)
    assert {c: v["n"] for c, v in d["classes"].items()} == EXPECTED_MIXED
    n = sum(EXPECTED_MIXED.values())
    assert {c: v["rate"] for c, v in d["classes"].items()} == {c: round(k / n, 4) for c, k in EXPECTED_MIXED.items()}
    assert d["classes"]["unknown_tool"]["subs"] == {"near_name": 1, "unknown_host": 1}
    assert d["classes"]["schema"]["subs"] == {"pattern": 1, "required": 1}
    assert d["classes"]["syntax"]["subs"] == {"malformed_action": 1} and d["classes"]["resource"]["subs"] == {"funds": 1}
    assert d["classes"]["permission"]["subs"] == {"forbidden": 1}
    assert d["families"] == {f: {"n": k, "rate": round(k / n, 4)} for f, k in {
        "valid": 6, "idle": 0, "form": 3, "discovery": 1, "belief": 3, "permission": 1, "rule": 1, "resource": 1,
        "service": 0, "code": 0, "simulator": 0}.items()}
    # 10 of 16 did not succeed, no class dominates the others: the largest is named, with its count
    assert d["counts"]["accepted"] == 6 and d["counts"]["rejected"] == 10
    assert d["primary"] in ("schema", "unknown_tool/near_name") and "10 of 16" in d["primary_reason"]
    assert d["loops"]["actions"] == 0 and d["resources"]["status"] == "absent" and d["code"]["status"] == "absent"
    kinds = [f["kind"] for f in d["findings"]]
    assert kinds == ["syntax", "unknown_tool", "permission", "schema", "nonexistent", "conflict", "rule", "resource"]
    assert all(f["evidence"] and 0 < f["share"] < 1 for f in d["findings"])
    # deterministic, and the same from a replayed world given the log
    assert X.diagnose(w, mix, since_seq=since, goal={"success": False}) == d
    events = [e.to_dict() for e in w.events_range(0)]
    rw = World.replay(w.spec, copy.deepcopy(events))
    assert rw.digest() == w.digest()
    dr = X.diagnose(rw, mix, since_seq=since, events=events, goal={"success": False})
    assert {k: v for k, v in dr.items() if k != "world_id"} == {k: v for k, v in d.items() if k != "world_id"}
    # the trajectory loader (what a recorded run leaves behind) reaches the same per-action classes
    recs = json.loads(json.dumps(step_records(r)))
    assert all(x["format"] == "crucible-traj/1" for x in recs)
    dt = X.diagnose_trajectory(recs, actor=mix, catalogue=w.available_tools(mix), aliases=D.host_aliases(w),
                               goal={"success": False})
    check_shape(None, mix, dt)
    acted = [a for a in dt["actions"] if a["class"] != "no_action"]
    assert [(a["tool"], a["class"], a["sub"]) for a in acted] == [(a["tool"], a["class"], a["sub"]) for a in d["actions"]]
    assert dt["counts"]["no_action"] == 2 and dt["source"] == "trajectory"          # the script ended: idle turns
    assert dt["competitor"]["status"] == "not_determinable"
    assert any(x.startswith("no_action: 2 turns") for x in dt["not_determinable"])
    # without a catalogue the trajectory path cannot look for near names, and says so
    blind = X.diagnose_trajectory(recs, actor=mix)
    assert blind["classes"]["unknown_tool"]["subs"] == {"undetermined": 2}
    assert any(x.startswith("unknown_tool: no tool catalogue") for x in blind["not_determinable"])
    # comparison rows carry the rates
    rows = X.compare({"log": d, "trajectory": dt})
    assert [x["run"] for x in rows] == ["log", "trajectory"] and rows[0]["form"] == round(3 / 16, 4)
    assert rows[0]["accepted_rate"] == round(6 / 16, 4) and rows[0]["discovery"] == round(1 / 16, 4)
    assert "primary = " in X.render(d) and "not determinable:" in X.render(dt)


# ====================================================================== competitor response
NEED = "sort.records"


class Scheduled(ScriptPolicy):
    """Acts at fixed rounds ({round: (tool, args)}); ``$listing`` stands for the listing this policy created."""

    def __init__(self, plan):
        self.plan = dict(plan)

    def script(self):
        listing = None
        for rnd in range(max(self.plan, default=-1) + 1):
            if rnd not in self.plan:
                yield None
                continue
            tool, args = self.plan[rnd]
            r = yield call(tool, **{k: (listing if v == "$listing" else v) for k, v in args.items()})
            if tool == "market.list" and r and r.get("ok"):
                listing = r["result"]["listing"]


class Shopper(ScriptPolicy):
    """Buys the cheapest active listing for the need, every other round."""

    def script(self):
        me = yield from self.personal_account()
        while True:
            r = yield call("market.browse", kind="service", need=NEED, sort="price", limit=5)
            rows = r["result"] if r and r.get("ok") else []
            if rows:
                yield call("market.buy", listing=rows[0]["listing"], account=me)


def offer(price):
    return ("market.list", {"kind": "service", "title": "Sorting as a service", "price_cents": price, "need": NEED,
                            "description": "We sort your records."})


def market(nova_plan, rival_plan, rounds=22):
    w = tiny()
    names = join(w, "nova", "rival", "s1", "s2", "s3", "s4")
    since = w.seq
    agents = [(names[0], Scheduled(nova_plan)), (names[1], Scheduled(rival_plan))] + [(s, Shopper()) for s in names[2:]]
    Runner(w, agents, dt=3600, seed=1).run(rounds)
    return w, since, names[0]


def test_customers_lost_to_a_rival_that_launched_or_cut_its_price_after_the_agents_move():
    # 1. the rival launches a cheaper listing after the agent's
    w, since, nova = market({0: offer(500)}, {8: offer(300)})
    d = X.diagnose(w, nova, since_seq=since, goal={"success": False})
    check_shape(w, nova, d)
    o = d["competitor"]["orders"]
    assert d["competitor"]["status"] == "established" and o["status"] == "established"
    assert o["buyers"] == 4 and o["lost_to_rivals"] == 4 and o["share_lost"] == 1.0 and o["rivals"] == ["p:rival"]
    assert {x["move"] for x in o["lost"]} == {"launched"}
    launch = event_at(w, o["lost"][0]["seq"])
    assert launch.type == "market.listing" and launch.payload["seller"] == "p:rival" and launch.payload["price"] == 300
    for x in o["lost"]:                                         # evidence: last order from the agent, first from the rival
        last, switch = event_at(w, x["last_order_seq"]), event_at(w, x["switch_seq"])
        assert last.type == switch.type == "market.order" and last.payload["buyer"] == switch.payload["buyer"] == x["buyer"]
        assert last.payload["seller"] == nova and switch.payload["seller"] == "p:rival"
        assert x["last_order_seq"] < x["seq"] < x["switch_seq"] or x["seq"] < x["switch_seq"]
    assert d["primary"] == "competitor_response" and "competitor_response" in [f["kind"] for f in d["findings"]]
    assert X.diagnose(w, "p:rival", since_seq=since)["competitor"]["status"] == "absent"      # the winner lost nobody

    # 2. an incumbent rival cuts its price below the agent's after the agent entered
    w, since, nova = market({2: offer(500)}, {0: offer(800), 10: ("market.update", {"listing": "$listing", "price_cents": 300})})
    o = X.diagnose(w, nova, since_seq=since)["competitor"]["orders"]
    assert o["status"] == "established" and o["lost_to_rivals"] == 4 and {x["move"] for x in o["lost"]} == {"price_cut"}
    cut = event_at(w, o["lost"][0]["seq"])
    assert cut.type == "market.listing_update" and cut.payload["changes"] == {"price": 300}

    # 3. control: the agent withdraws its own listing; buyers go to a rival that did nothing after the agent's entry
    w, since, nova = market({2: offer(500), 10: ("market.update", {"listing": "$listing", "status": "closed"})}, {0: offer(800)})
    d = X.diagnose(w, nova, since_seq=since)
    o = d["competitor"]["orders"]
    bought_elsewhere = {e.payload["buyer"] for e in w.events_range(since) if e.type == "market.order" and e.payload["seller"] == "p:rival"}
    assert len(bought_elsewhere) == 4 and o["buyers"] == 4          # every buyer did move to the rival ...
    assert o["status"] == "absent" and o["lost_to_rivals"] == 0     # ... but not because of anything the rival did
    assert d["competitor"]["status"] == "absent" and d["primary"] != "competitor_response"

    # 4. no sales at all: nothing to reverse
    w, since, nova = market({}, {0: offer(800)}, rounds=6)
    assert X.diagnose(w, nova, since_seq=since)["competitor"]["orders"]["reason"].startswith("the actor sold nothing")


def launch(w, handle, need, tier, price, cash=600_000):
    """Found a company and serve `need` to the consumer population with a solver of the given quality tier."""
    w.system_call("admin.join", {"handle": handle, "endowment": 2_000_000})
    me = f"p:{handle}"
    ex = lambda t, **k: w.execute(me, {"tool": t, "args": k})  # noqa: E731
    acct0 = w.modules["bank"].primary_account(w.state, me)
    org = ex("id.create_org", handle=f"{handle}-co", name=handle, kind="company").result["org"]
    assert ex("biz.register", org=org, account=acct0).ok
    acct = ex("bank.open_account", owner=org).result["account"]
    assert ex("bank.transfer", from_account=acct0, to=acct, amount=cash).ok
    repo = ex("code.create_repo", name="svc", owner=org).result["repo"]
    assert ex("code.commit", repo=repo, files=service_files(need, tier_source(need, tier), price)).ok
    prov = sorted(p["provider"] for p in w.query(me, "cloud.providers"))[0]
    inst = ex("cloud.rent", provider=prov, cores=1, account=acct, owner=org).result["instance"]
    r = ex("cloud.deploy", repo=repo, host=f"{handle}-svc", instance=inst)
    assert r.ok and r.result["status"] == "live", r.to_dict()
    assert ex("market.list", kind="api", title=f"{handle}: {NEEDS[need].title}", price_cents=price, seller=org,
              host=f"{handle}-svc", endpoint="solve", need=need, description=" ".join(NEEDS[need].keywords)).ok
    return me, org, repo, inst


def test_consumer_population_lost_to_a_new_entrant_versus_lost_to_ones_own_broken_release():
    """Counterfactual forks of one consumer economy (deterministic cost model, so forks differ only by what is
    done in them). The agent serves route.shortest with a mediocre solver at a high price and wins customers.
    Fork A: a rival then launches a better, cheaper service. Fork B: no entrant; the agent ships a release that
    raises on most requests."""
    need = "route.shortest"
    w = tiny(n_consumers=1500, cost_model="deterministic")
    since = w.seq
    nova, org, repo, inst = launch(w, "nova", need, 0, 4)
    w.advance_time(4 * DAY)
    base = list(w.log)
    fa, fb = w.fork("entrant"), w.fork("own-break")
    launch(fa, "rival", need, best_tier(need), 1)
    fa.advance_time(9 * DAY)
    ex = lambda t, **k: fb.execute(nova, {"tool": t, "args": k})  # noqa: E731
    assert ex("code.commit", repo=repo, files={"service.py": "def solve(args, caller):\n    raise RuntimeError('bad release')\n"}).ok
    assert ex("cloud.deploy", repo=repo, host="nova-svc", instance=inst).result["status"] == "live"
    fb.advance_time(7 * DAY)

    # ---- fork A: the entrant took the customers
    da = X.diagnose(fa, nova, since_seq=since, events=base + list(fa.log), goal={"success": False})
    pop = da["competitor"]["population"]
    off = pop["offerings"][0]
    assert da["competitor"]["status"] == "established" and off["status"] == "established" and off["need"] == need
    assert off["peak_customers"] >= 300 and off["final_customers"] <= 0.7 * off["peak_customers"]
    assert off["own_causes"] == [] and off["share_of_loss_matched"] >= X.RIVAL_SHARE
    rivals = {r["seller"]: r for r in off["rivals"]}
    assert rivals["o:rival-co"]["gain"] > 100 and [m["move"] for m in rivals["o:rival-co"]["moves"]] == ["launched"]
    ev = {e.seq: e for e in base + list(fa.log)}
    assert ev[rivals["o:rival-co"]["moves"][0]["seq"]].type == "consumers.offer"
    assert ev[off["peak_seq"]].type == ev[off["end_seq"]].type == "consumers.hour"
    assert ev[off["peak_seq"]].payload["slots"][str(off["slot"])]["customers"] == off["peak_customers"]
    assert da["code"]["status"] == "absent" and da["code"]["requests"]["error"] == 0       # its code kept working
    assert da["primary"] == "competitor_response"
    # the entrant itself lost nothing
    dr = X.diagnose(fa, "p:rival", since_seq=since, events=base + list(fa.log))
    assert dr["competitor"]["status"] == "absent" and dr["code"]["quality"]["mean_q"] > 0.9

    # ---- fork B: customers left too, but the agent's own release broke - the detector does not blame a rival
    db = X.diagnose(fb, nova, since_seq=since, events=base + list(fb.log), goal={"success": False})
    offb = db["competitor"]["population"]["offerings"][0]
    assert offb["final_customers"] <= 0.7 * offb["peak_customers"]                         # the loss is as real
    assert offb["status"] in ("not_determinable", "absent") and db["competitor"]["status"] != "established"
    assert any("failed requests rose" in c for c in offb["own_causes"])
    assert not any(r["seller"] == "o:rival-co" for r in offb.get("rivals", []))            # no entrant in this fork
    cur = db["code"]["current_deployments"]["requests"]                                    # the release that is live now
    assert db["code"]["status"] == "established" and cur["error"] > 5 * cur["ok"] and cur["error"] > 1000
    assert db["code"]["requests"]["ok"] > 1000                                             # the earlier release worked
    assert db["primary"] == "code_failure"
    # the two forks share their history up to the fork and differ after it
    assert fa.digest() != fb.digest() and off["peak_customers"] >= 300


# ====================================================================== what the prompt showed
def test_tool_list_exposure_separates_an_uneven_list_from_a_uniform_one():
    w = tiny()
    nova, _ = join(w, "nova", "bob")
    for t in ("ads.stats", "bank.accounts", "bank.accounts", "biz.registry"):
        w.execute(nova, {"tool": t, "args": {}})
    tools = w.available_tools(nova)
    d = X.diagnose(w, nova, interface={"tool_chars": 4000, "tool_order": "alpha"})
    tl = d["context"]["tool_list"]
    assert tl["renderer"] == "tool_list" and tl["tools"] == len(tools) and 0 < tl["with_signature"] < 0.5 * len(tools)
    assert "id" in tl["hosts_with_no_signature"] and "ads" not in tl["hosts_with_no_signature"]
    assert tl["calls_to_tools_with_signature"] == tl["calls_to_existing_tools"] == 4 and tl["status"] == "established"
    assert d["primary"] == "tool_list_truncation"
    assert [f for f in d["findings"] if f["kind"] == "tool_list_truncation"][0]["share"] == round(1 - tl["share_with_signature"], 4)
    # the same calls under the 4.0 catalogue renderer: every tool gets the same treatment, nothing to attribute
    for budget, level, sig in ((60_000, "full", len(tools)), (9_000, "signatures", len(tools)), (2_000, "names", 0)):
        tl = X.diagnose(w, nova, interface={"renderer": "catalogue", "budget_chars": budget})["context"]["tool_list"]
        assert (tl["level"], tl["with_signature"], tl["status"]) == (level, sig, "absent") and tl["hosts_listed"] == tl["hosts"]
    # an agent that used tools the uneven list did not show was not confined by it
    w.execute(nova, {"tool": "id.me", "args": {}})
    w.execute(nova, {"tool": "world.clock", "args": {}})
    d2 = X.diagnose(w, nova, interface={"tool_chars": 4000, "tool_order": "alpha"})
    assert d2["context"]["tool_list"]["status"] == "absent" and d2["primary"] != "tool_list_truncation"


# ====================================================================== the recorded 3.0 trajectories
def _final_challenge_spec():
    return generate({"family": "open_tight_compute", "overrides": {
        "cheap_cores": 40, "module_config": {"webgen": {"sites": 3}, "civ": {"innovators": 1},
                                             "adversary": {"injection_pages": 3, "fraud_capability": True,
                                                           "spam_posts": 3}}}}, 5)


def test_recorded_3_0_final_challenge_runs_are_diagnosed_from_their_trajectory_files():
    files = sorted(glob.glob(os.path.join(EXPERIMENTS, "final_challenge_*.jsonl")))
    assert len(files) == 8
    spec = _final_challenge_spec()
    tools, aliases = X.catalogue_from_spec(spec, discovery=False)     # the catalogue those runs saw, no genesis needed
    out = {}
    for p in files:
        name = os.path.basename(p)[len("final_challenge_"):-len(".jsonl")]
        summ = json.load(open(p[:-1]))
        recs = X.load_trajectory(p)
        assert recs[0]["spec_hash"] == spec.hash() and recs[0]["n_tools"] == len(tools) == 191, \
            "the 3.0 catalogue of this world is no longer reproduced with discovery=False"
        iface = dict(summ.get("interface") or {"tool_order": "alpha", "tool_chars": 9000})   # first runs: 3.0 defaults
        goal = {"success": summ["explicit_goal"]["success"], "progress": summ["explicit_goal"]["best_progress"]} \
            if summ["objective_kind"] == "explicit" else None
        d = X.diagnose_trajectory(recs, catalogue=tools, aliases=aliases, usage=summ["usage"], interface=iface,
                                  context_limit=iface.get("max_context", 8192), max_new_tokens=320, goal=goal)
        check_shape(None, "p:nova", d)
        assert d["counts"]["turns"] == 120 and d["source"] == "trajectory"
        # the file's own outcome counts are reproduced exactly
        acted = [a for a in d["actions"] if a["tool"]]
        assert sum(1 for a in acted if a["ok"]) == summ["outcomes"].get("ok", 0)
        assert d["counts"]["distinct_tools"] == summ["distinct_tools"]
        assert d["competitor"]["status"] == "not_determinable" and d["code"]["attempted"] is False
        out[name] = d
    # first three runs: alphabetical 9,000-character list - most hosts had no signature and no call left that subset
    for name in ("broad_Qwen3-14B_alpha9k", "broad_Qwen3-4B_alpha9k", "explicit_Qwen3-4B_alpha9k"):
        tl = out[name]["context"]["tool_list"]
        assert out[name]["primary"] == "tool_list_truncation" and tl["status"] == "established"
        assert 55 <= tl["with_signature"] <= 85 and tl["share_of_calls_to_tools_with_signature"] == 1.0
        assert {"id", "market", "net", "ops", "world"} <= set(tl["hosts_with_no_signature"])
        assert not {"ads", "bank", "caps"} & set(tl["hosts_with_no_signature"])
    assert out["broad_Qwen3-4B_alpha9k"]["classes"]["wait"]["n"] == 10          # settled by the usage counters
    # the 8k-context run: 107 of 120 prompts truncated (from the usage counters: which turns is not in the file)
    cx = out["broad_Qwen3-14B_interleave24k_ctx8k"]["context"]
    assert (cx["truncated_turns"], cx["turns"], cx["per_turn"]) == (107, 120, False)
    assert out["broad_Qwen3-14B_interleave24k_ctx8k"]["primary"] == "context_truncation"
    # the four runs on the corrected interface: everything shown, nothing truncated - and perseveration
    for name in ("broad_Qwen3-14B", "broad_Qwen3-4B", "explicit_Qwen3-14B", "explicit_Qwen3-4B"):
        d = out[name]
        assert d["context"]["status"] == "absent" and d["context"]["tool_list"]["with_signature"] == 191
        assert d["context"]["tool_list"]["status"] == "absent" and d["loops"]["status"] == "established"
        assert d["loops"]["rate"] >= 0.55, (name, d["loops"]["rate"])
    d = out["broad_Qwen3-4B"]                 # 120 calls to id.create_org with a handle that violates the pattern
    assert d["classes"] == {"schema": {"n": 120, "rate": 1.0, "evidence": d["classes"]["schema"]["evidence"],
                                       "subs": {"pattern": 120}, "tools": {"id.create_org": 120}}}
    assert d["primary"] == "schema" and d["loops"]["longest_run"] == 117 and d["loops"]["by_tool"] == {"id.create_org": 117}
    d = out["explicit_Qwen3-4B"]              # 87 conflicting account openings, 26 debits to a principal that does not exist
    assert d["classes"]["conflict"]["n"] == 87 and d["classes"]["conflict"]["subs"] == {"account limit reached": 87}
    assert d["classes"]["nonexistent"]["n"] == 26 and d["primary"] == "conflict" and d["families"]["belief"]["n"] == 113
    d = out["broad_Qwen3-14B"]                # the same listing over and over; 38 rejected (15 schema, 23 by the tool)
    assert d["loops"]["by_tool"]["market.list"] == 69 and d["primary"] == "loop"
    assert d["classes"]["schema"]["n"] + d["classes"]["rule"]["n"] == 38 and d["classes"]["rule"]["subs"] == {"unknown need": 23}
    d = out["explicit_Qwen3-14B"]
    assert d["primary"] == "loop" and d["loops"]["by_tool"]["market.browse"] == 41 and d["strategy"]["status"] == "absent"
    # no run called a tool that does not exist; none wrote, ran or deployed code
    assert all("unknown_tool" not in d["classes"] and d["families"]["discovery"]["n"] == 0 for d in out.values())
    # what these files cannot establish is stated, not guessed
    for d in out.values():
        nd = " | ".join(d["not_determinable"])
        assert "competitor_response" in nd and "task_difficulty" in nd and "effects:" in nd
        assert d["difficulty"]["status"] == "not_established" and d["resources"]["cash"] is None
    assert all(out[n]["strategy"]["status"] == "not_determinable" for n in out if n.startswith("broad"))
    rows = X.compare(out)
    assert len(rows) == 8 and {r["primary"] for r in rows} == {"tool_list_truncation", "context_truncation", "loop",
                                                              "schema", "conflict"}
