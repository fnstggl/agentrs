"""Class-4 semantic participants (`crucible.semantic.generalists`): binding to the population, persistent state,
provenance of every action, recorded inference, retrieval and argument planning.

Fast tests use the model-free brain (`brain="ablated"`) where the property does not depend on what a model decides;
recording and regeneration are checked with a real small model (SmolLM2-135M-Instruct on CPU).
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import crucible  # noqa: E402

crucible.load_all_modules()

from crucible import population as pop  # noqa: E402
from crucible.core.kernel import World  # noqa: E402
from crucible.semantic import generalists as G, routing  # noqa: E402
from crucible.semantic.generalists import catalog, mind as M, verify  # noqa: E402

SMALL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def _world(brain="ablated", seed=0, **cfg):
    return G.civ_world("balanced", seed, population="development", config={"brain": brain, "steps": 2, **cfg})


def _activate(w):
    mod = w.modules["pop_semantic"]
    idx = pop.class_members(w, 4)
    w._run_system("test.activate", lambda ctx: mod._activate(ctx, idx, ctx.now))


@pytest.fixture(scope="module")
def ablated():
    import torch
    torch.set_num_threads(1)
    w = _world()
    _activate(w)
    w.advance_time(w.now + 3600)
    _activate(w)
    yield w
    w.close()


def test_class4_rows_are_bound_to_the_runtime_with_persona_from_population_state(ablated):
    w = ablated
    rows = G.members(w)
    c4 = pop.class_members(w, 4)
    assert [r["i"] for r in rows] == [int(i) for i in c4] and len(rows) >= 2
    assert verify.bound(w)["bound"] == len(rows)
    for r in rows:
        assert r["principal"] == f"p{r['i']}" and w.state.has("principals", r["principal"])
        assert r["account"] and w.state.get("accounts", r["account"])["owner"] == r["principal"]
        assert "interested in" in r["persona"]["about"] and len(r["persona"]["topics"]) == 3
    assert rows[0]["persona"] != rows[1]["persona"]
    assert w.runtime["pop.class_runtimes"][4]["owner"] == "pop_semantic"


def test_activation_persists_goals_memory_history_in_world_state(ablated):
    rows = G.members(ablated)
    for r in rows:
        assert r["counts"]["activations"] == 2
        assert r["goals"] and r["memory"] and r["next_at"] > 0
        assert all(set(h) >= {"t", "tool", "ok", "decision"} for h in r["history"])
    assert sum(r["counts"].get("calls", 0) for r in rows) > 0


def test_every_participant_event_lies_inside_a_brain_chosen_call(ablated):
    prov = verify.provenance(ablated)
    assert prov["outside_model_chosen_calls"] == 0, prov


def test_state_survives_snapshot_and_resume_exactly(ablated):
    blob = ablated.snapshot_blob()
    r = World.from_blob(blob, world_id="resumed-semp")
    assert G.members(r) == G.members(ablated)
    assert r.digest() == ablated.digest()
    r.close()


def test_participants_only_act_through_participant_tools(ablated):
    for a in verify.actions(ablated):
        assert not a["tool"].startswith(("admin.", "pop_adaptive.", "pop_strategic.", "world."))
    for r in G.members(ablated):
        assert all(t not in json.dumps(r["persona"]) for t in ("evaluator", "verified_quality"))


def test_retrieval_finds_the_service_of_general_action_phrases():
    """The service directory search over a participant's full tool interface, for the general action phrases its
    intention step produces."""
    import crucible.semantic.generalists.bench as B
    it = B.build(0, 1, ("dev",))[0]
    a = B.agent_of(it)
    cases = [("send a private message", {"mail", "msg"}), ("read my mail", {"mail"}), ("follow a person", {"social"}),
             ("pay someone money", {"bank"}), ("write an article", {"blog", "studio", "docs", "posts"}),
             ("check my bank balance", {"bank"}), ("look for open jobs", {"jobs", "labor"}),
             ("rent a server", {"cloud"}), ("ask a question on the forum", {"forum", "kb", "dev"}),
             ("create a code repository", {"code"})]
    hs = catalog.rank_hosts([c for c, _ in cases], [a.tools] * len(cases), 10, [""] * len(cases))
    hits = sum(bool(want & set(h)) for (_, want), h in zip(cases, hs))
    assert hits == len(cases), [(c, h[:5]) for (c, _), h in zip(cases, hs)]


def test_argument_slots_offer_only_compatible_known_identifiers():
    a = M.Agent(i=1, principal="p1", account="acct_000001", persona={"about": ""})
    a.known = {"acct_000001": "your account", "p7": "wrote to you", "mail_000003": "unread mail",
               "sha256:" + "a" * 64: "your artifact"}
    a.new_goal(0, "I will pay p7 500 cents.")
    tool = {"name": "bank.transfer", "inputSchema": {"type": "object", "required": ["from_account", "to", "amount"],
                                                     "properties": {"from_account": {"type": "string", "maxLength": 40},
                                                                    "to": {"type": "string", "maxLength": 60,
                                                                           "description": "account id or principal"},
                                                                    "amount": {"type": "integer", "minimum": 1}}}}
    slots = {s.name: s for s in M.plan_slots(a, tool)}
    assert slots["from_account"].values[:-1] == ["acct_000001"]
    assert "p7" in slots["to"].values and "mail_000003" not in slots["to"].values
    assert slots["amount"].kind == "number" and 500 in slots["amount"].values
    pub = {"name": "blog.publish", "inputSchema": {"type": "object", "required": ["ref", "title"],
                                                   "properties": {"ref": {"type": "string", "maxLength": 71},
                                                                  "title": {"type": "string", "maxLength": 200}}}}
    s = {x.name: x for x in M.plan_slots(a, pub)}
    assert s["ref"].kind == "compose" and s["title"].kind == "text"     # it owns no artifact yet: it writes the text
    a.known["sha256:" + "b" * 64] = "from your blog.latest ref ('someone else's post')"
    M.own_ref(a, "sha256:" + "a" * 64)
    s = {x.name: x for x in M.plan_slots(a, pub)}
    assert s["ref"].values[:-1] == ["sha256:" + "a" * 64]               # only its own artifacts can be published
    # an optional addressee (either `to` or `conversation`) is offered among the people it knows
    send = {"name": "msg.send", "inputSchema": {"type": "object", "required": ["body"], "properties": {
        "to": {"type": "string", "maxLength": 60}, "conversation": {"type": "string", "maxLength": 40},
        "body": {"type": "string", "maxLength": 4000}}}}
    s = {x.name: x for x in M.plan_slots(a, send)}
    assert s["to"].values == ["p7", None] and "conversation" not in s
    # an object argument (a listing, an offer ...) never takes a person
    buy = {"name": "market.buy", "inputSchema": {"type": "object", "required": ["listing"], "properties": {
        "listing": {"type": "string", "maxLength": 40}}}}
    s = {x.name: x for x in M.plan_slots(a, buy)}
    assert "p7" not in s["listing"].values


def test_written_values_reusing_the_goal_are_kept_and_echoes_of_the_prompt_are_not():
    a = M.Agent(i=1, principal="p1", account=None, persona={"about": "You like travel."})
    a.new_goal(0, "I will write a travel article about a hidden gem in the Amazon rainforest and publish it.")
    tool = {"name": "blog.publish", "description": "Publish an artifact you own.",
            "inputSchema": {"type": "object", "required": ["title"],
                            "properties": {"title": {"type": "string", "maxLength": 200}}}}
    title = M.Slot("title", tool["inputSchema"]["properties"]["title"], True, "text", tokens=24)
    req = M.slot_request(a, 0, tool, title, {}, None)
    assert M.accept(a, title, req, "A hidden gem in the Amazon rainforest") == "A hidden gem in the Amazon rainforest"
    assert M.accept(a, title, req, "You are calling blog.publish: Publish an artifact you own.") is None
    body = M.Slot("ref", {"type": "string"}, True, "compose", tokens=200)
    a.remember(0, "act", "You called search.query {\"q\":\"amazon\"} -> ok: [] nothing was found for the query "
                         "you sent to the search engine today")
    req = M.slot_request(a, 0, tool, body, {}, None)
    text = ("The Amazon rainforest hides a small village on the Rio Negro where boats arrive once a week. "
            "Travellers sleep in hammocks, eat river fish and learn to read the forest with local guides.")
    assert M.accept(a, body, req, text) == {M.COMPOSED: text}
    echo = ("You called search.query q amazon ok nothing was found for the query you sent to the search engine "
            "today and nothing was found for the query you sent")
    assert M.accept(a, body, req, echo) is None and a.counts["copied_values"] == 2


def test_object_arguments_never_take_a_person():
    """mind.OBJECT_ARGS (an offer, thread, posting, invoice, contract ... is not a person): for every such argument a
    known person is not offered, a known identifier of a non-person kind is."""
    a = M.Agent(i=1, principal="p1", account="acct_000001", persona={"about": ""})
    a.known = {"p7": "wrote to you", "o:acme": "an organisation", "offer_000012": "an offer you saw",
               "mail_000003": "unread mail"}
    a.new_goal(0, "I will accept the offer offer_000012.")
    for arg in M.OBJECT_ARGS:
        tool = {"name": "deal.accept", "inputSchema": {"type": "object", "required": [arg], "properties": {
            arg: {"type": "string", "maxLength": 40}}}}
        s = {x.name: x for x in M.plan_slots(a, tool)}
        if arg in s and s[arg].kind == "choose":
            assert not {"p7", "o:acme"} & set(s[arg].values), arg
    tool = {"name": "deal.accept", "inputSchema": {"type": "object", "required": ["offer"], "properties": {
        "offer": {"type": "string", "maxLength": 40}}}}
    assert "offer_000012" in {x.name: x for x in M.plan_slots(a, tool)}["offer"].values


class _ScriptedBrain:
    """A scripted brain (no model) that tests the runtime's retry rule only: greedy requests get an echo of the
    prompt's scaffolding, sampled requests (temperature > 0, the rewrite) a usable title."""

    def __init__(self):
        self.generated = []

    def choose(self, ctx, capability, requests):
        return [0 for _ in requests]

    def generate(self, ctx, capability, requests):
        self.generated += [r.get("temperature", 0) for r in requests]
        return ["A week on the Rio Negro" if r.get("temperature", 0) > 0 else
                "You are calling blog.publish: Publish an artifact you own." for r in requests]


def test_a_required_value_written_unusably_is_rewritten_once_by_sampling():
    """mind.fill_arguments: a required text value that is discarded (an echo) is written once more by sampling
    (temperature 0.7, a seed of its own); the rewrite is counted (`rewritten_values`) and used when usable; an
    optional value is not rewritten."""
    a = M.Agent(i=3, principal="p3", account=None, persona={"about": "You like travel."})
    a.new_goal(0, "I will write about the Amazon rainforest.")
    tool = {"name": "blog.publish", "description": "Publish an artifact you own.",
            "inputSchema": {"type": "object", "required": ["title"],
                            "properties": {"title": {"type": "string", "maxLength": 200}}}}
    b = _ScriptedBrain()
    vals = M.fill_arguments(None, b, [(a, tool, None)], 3600, M.Config())
    assert vals == [{"title": "A week on the Rio Negro"}]
    assert a.counts["copied_values"] == 1 and a.counts["rewritten_values"] == 1
    assert b.generated[0] == 0.0 and b.generated[1] == 0.7 and len(b.generated) == 2


def _copy_case():
    a = M.Agent(i=1, principal="p1", account=None,
                persona={"about": "You are interested in travel (trips, hotels, flights) and cooking (recipes, "
                                  "bread, ovens). You are good at writing. You have a job."})
    a.situation = ["money: $12.00 in your account acct_000123", "unread mail: mail_000004 from p7: 'Lunch on Friday'"]
    a.tools = [{"name": n} for n in ("blog.publish", "blog.latest", "mail.send", "cal.create", "search.query")]
    a.new_goal(0, "I will write a travel article about a hidden gem in the Amazon rainforest and publish it.")
    a.remember(0, "act", "You called search.query {\"q\":\"amazon\"} -> ok: [] nothing was found for the query you "
                         "sent")
    a.remember(0, "act", "You called blog.latest {} -> ok: [{\"title\": \"Ten cheap ways to heat a small flat\"}]")
    tool = {"name": "blog.publish", "description": "Publish an artifact you own as a blog post with a title.",
            "inputSchema": {"type": "object", "required": ["title"],
                            "properties": {"title": {"type": "string", "maxLength": 200}}}}
    return a, tool


def test_copy_filter_rejects_foreign_prompt_text_tool_names_and_goal_restatements():
    """Verifier probe P1 (2a6fd4d accepted verbatim copies of non-goal prompt text in short values, and tool names):
    short values copied from someone else's title, the tool description, a memory line, the persona or a situation
    line, a tool or service name, and the goal sentence itself are rejected; titles reusing the goal's words, fresh
    titles and a rewrite of the participant's own earlier value are kept."""
    a, tool = _copy_case()
    title = M.Slot("title", tool["inputSchema"]["properties"]["title"], True, "text", tokens=24)
    req = M.slot_request(a, 0, tool, title, {}, None)
    junk = ["Ten cheap ways to heat", "Ten cheap ways to heat a small flat", "Publish an artifact you own",
            "Publish an artifact you own as a blog post", "nothing was found for the query",
            "You are interested in travel (trips,", "money: $12.00 in your account acct_000123", "blog.publish",
            "blog", "mail.send",
            "I will write a travel article about a hidden gem in the Amazon rainforest and publish it."]
    for v in junk:
        assert M.accept(a, title, req, v, "blog.publish") is None, v
    for v in ("A hidden gem in the Amazon rainforest", "Seven days on the Rio Negro", "Lunch on Friday"):
        assert M.accept(a, title, req, v, "blog.publish") == v, v
    # a value it wrote itself in an earlier call (now in its last actions) may be written again
    a.history.append({"t": 0, "tool": "blog.publish", "args": '{"title":"Seven boats a week on the Rio Negro"}',
                      "ok": False, "code": "InvalidAction", "error": "args.ref: required"})
    req = M.slot_request(a, 0, tool, title, {}, None)
    assert M.accept(a, title, req, "Seven boats a week on the Rio Negro", "blog.publish") is not None
    # a calendar entry titled after its service, a mail body that is a tool name (the audit's junk values)
    cal = M.Slot("title", {"type": "string", "maxLength": 120}, True, "text", tokens=24)
    req = M.slot_request(a, 0, {"name": "cal.create", "description": "Create a calendar event."}, cal, {}, None)
    assert M.accept(a, cal, req, "cal", "cal.create") is None
    body = M.Slot("body", {"type": "string", "maxLength": 4000}, True, "text", tokens=200)
    reqb = M.slot_request(a, 0, {"name": "mail.send", "description": "Send a mail."}, body, {}, None)
    for v in ("mail.send", "cal", "Ten cheap ways to heat a small flat"):
        assert M.accept(a, body, reqb, v, "mail.send") is None, v
    assert M.accept(a, body, reqb, "Yes, Friday works for me. Shall we meet at noon?", "mail.send") is not None


def test_composed_publication_restating_the_goal_is_rejected():
    """Verifier probe P1b: a composed article that is only the goal sentence (once, twice, or with filler) is not an
    article; a real article about the goal is kept, as is a message whose content is what the goal asks for."""
    goal = ("I will write a blog post about the challenges of winter travel and the importance of preparing for cold "
            "weather, and publish it on the blog.")
    a = M.Agent(i=1, principal="p1", account=None, persona={"about": "You like travel."})
    a.new_goal(0, goal)
    tool = {"name": "blog.publish", "description": "Publish an artifact you own as a blog post.",
            "inputSchema": {"type": "object", "required": ["ref"], "properties": {"ref": {"type": "string"}}}}
    body = M.Slot("ref", {"type": "string"}, True, "compose", tokens=200)
    req = M.slot_request(a, 0, tool, body, {}, None)
    for v in (goal, goal + " " + goal, goal + " I hope you like it."):
        assert M.accept(a, body, req, v, "blog.publish") is None, v
    real = ("Winter travel needs planning. Trains run late when snow falls, so leave an hour early. Pack layers, a "
            "thermos and a phone charger. Check road reports before driving and keep a blanket in the car.")
    assert M.accept(a, body, req, real, "blog.publish") == {M.COMPOSED: real}
    b = M.Agent(i=2, principal="p2", account=None, persona={"about": "You bake bread."})
    b.new_goal(0, "I will ask p7 whether he wants to join my bread-baking class on Saturday.")
    msg = M.Slot("body", {"type": "string", "maxLength": 4000}, True, "text", tokens=200)
    reqm = M.slot_request(b, 0, {"name": "mail.send", "description": "Send a mail."}, msg, {}, None)
    text = "Hi p7, do you want to join my bread-baking class on Saturday? We start at ten."
    assert M.accept(b, msg, reqm, text, "mail.send") == text


def test_only_artifacts_the_participant_made_or_owns_become_its_own(ablated):
    """Verifier probe P4: a read-only call (artifact.info / artifact.get) returning another participant's published
    artifact does not make it the caller's own: it is not offered for the caller's publication; an artifact the
    participant stores itself is."""
    import base64
    w = ablated
    rows = G.members(w)
    a = M.Agent.from_row(rows[0])
    other = rows[1]["principal"]
    tools = {t["name"]: t for t in w.available_tools(a.principal)}
    box = {}

    def other_publishes(ctx):
        r = ctx.call("artifact.put", {"data_b64": base64.b64encode(b"Rye bread at home: a recipe by someone else.")
                                      .decode(), "mime": "text/plain", "kind": "text"}, as_principal=other)
        ctx.call("artifact.publish", {"ref": r["ref"]}, as_principal=other)
        box["ref"] = r["ref"]
    w._run_system("test.other", other_publishes)
    ref = box["ref"]
    for name in ("artifact.info", "artifact.get"):
        if name in tools:
            w._run_system("test.read", lambda ctx: M.execute(ctx, a, tools[name], {"ref": ref}, ctx.now, M.Config()))
            assert a.history[-1]["ok"] and ref not in a.seen.get("own_refs", [])
    slots = {s.name: s for s in M.plan_slots(a, tools["blog.publish"])}
    assert ref not in (slots["ref"].values or [])
    mine = {}

    def put(ctx):
        M.execute(ctx, a, tools["artifact.put"], {"data_b64": base64.b64encode(b"My own notes on rye.").decode(),
                                                  "mime": "text/plain", "kind": "text"}, ctx.now, M.Config())
        mine["ref"] = json.loads(a.history[-1]["result"])["ref"] if a.history[-1]["ok"] else None
    w._run_system("test.put", put)
    assert mine["ref"] and mine["ref"] in a.seen["own_refs"]
    w._run_system("test.info", lambda ctx: M.execute(ctx, a, tools["artifact.info"], {"ref": mine["ref"]}, ctx.now,
                                                    M.Config()))
    assert a.seen["own_refs"].count(mine["ref"]) == 1
    slots = {s.name: s for s in M.plan_slots(a, tools["blog.publish"])}
    assert mine["ref"] in slots["ref"].values and ref not in slots["ref"].values


def test_recorded_inference_regenerates_without_running_a_model():
    import torch
    from crucible.semantic import service as SV
    torch.set_num_threads(1)
    from crucible.semantic.generalists.world import ensure_module
    from crucible.social.worlds import social_world
    sp = ensure_module(social_world("development", 5, n=200, classes={"adaptive": 0, "strategic": 0, "semantic": 2}),
                       {"brain": "lm", "routes": {c: SMALL for c in routing.TABLE}, "steps": 1,
                        "gap_s": [3 * 86400, 4 * 86400]})
    w = World.create(sp)
    for _ in range(16):            # until the population's tick has activated a participant (as in any world)
        w.advance_time(w.now + 3600)
        if any(r["counts"].get("activations") for r in G.members(w)):
            break
    rows = G.members(w)
    assert sum(r["counts"].get("goals", 0) for r in rows) >= 1
    events = [e.to_dict() for e in w.events_range(0)]
    before = sum(s.stats["requests"] for s in list(SV._SERVICES_ALL))
    g = World.regenerate(w.spec, events, world_id="regen-semp")
    after = sum(s.stats["requests"] for s in list(SV._SERVICES_ALL))
    assert g.digest() == w.digest() and G.members(g) == rows
    assert after == before                          # no model request was served during regeneration
    g.close()
    w.close()


def test_a_refused_publication_undoes_the_text_it_stored(ablated):
    """A publication whose text the participant wrote in the call (mind.COMPOSE) is undone as a whole when the
    publishing call is refused: no stored artifact remains outside a recorded decision (SP-3.2d provenance)."""
    w = ablated
    row = G.members(w)[0]
    a = M.Agent.from_row(row)
    tool = next(t for t in w.available_tools(a.principal) if t["name"] == "blog.publish")
    out = {}

    def run(ctx):
        n0 = len(ctx.events)
        M.execute(ctx, a, tool, {"ref": {M.COMPOSED: "A short article about baking bread at home with rye flour."},
                                 "title": "Bread", "kind": "not-a-kind"}, ctx.now, M.Config())
        out["types"] = [e.type for e in ctx.events[n0:]]
    w._run_system("test.refused", run)
    assert a.history[-1]["ok"] is False and "artifact.stored" not in out["types"]
    # the participant does not keep the identifier of the text the rollback removed
    assert not [k for k in a.known if a.known[k].startswith("the text you wrote")] and not a.seen.get("own_refs")
    ok = {}

    def run_ok(ctx):
        n0 = len(ctx.events)
        M.execute(ctx, a, tool, {"ref": {M.COMPOSED: "A short article about baking bread at home with rye flour."},
                                 "title": "Bread"}, ctx.now, M.Config())
        ok["types"] = [e.type for e in ctx.events[n0:]]
    w._run_system("test.published", run_ok)
    assert a.history[-1]["ok"] is True and "artifact.stored" in ok["types"]
    assert len(a.seen["own_refs"]) == 1 and a.seen["own_refs"][0] in a.known
    assert a.history[-1]["seq"][0] <= a.history[-1]["seq"][1]


def test_a_reasoning_first_choice_runs_the_thinking_mode_then_picks_an_option():
    """LMBrain with a reasoning budget (D-38): the Qwen3 model first generates in its thinking mode (bounded), then an
    option is named or scored after the closed reasoning; without a budget the options are scored directly."""
    import torch
    torch.set_num_threads(1)
    from crucible.semantic.generalists import brain as B
    small = "Qwen/Qwen3-0.6B"
    req = {"messages": [{"role": "system", "content": "You choose services."},
                        {"role": "user", "content": "You want to send $20 to a friend. Which service do you use?\n"
                                                    "- bank: accounts and payments\n- blog: articles\n- tube: videos\n"
                                                    "Answer with the service name only."}],
           "choices": ["bank", "blog", "tube"], "normalize": "mean", "prefix_messages": 1}
    b = B.LMBrain({c: small for c in routing.TABLE}, think={"tool_choice": 24})
    svc = b.service("tool_choice")
    bound = svc.max_new_tokens
    write = {"messages": req["messages"], "max_new_tokens": 200, "temperature": 0.0, "seed": 0}
    before = svc._norm(write)
    n0, g0 = svc.stats["requests"], svc.stats["generate_calls"]
    picks = b.choose(None, "tool_choice", [req])
    assert len(picks) == 1 and 0 <= picks[0] < 3 and b.calls["think"] == 1
    assert svc.stats["requests"] - n0 >= 1          # the thinking generation (plus a rescoring when it named none)
    # verifier P2: the reasoning's generation is attributed to the class-4 brain (SP-4.2b) ...
    assert b.generation_calls() == svc.stats["generate_calls"] - g0 == 1 and b.calls["generate"] == 0
    # ... and the shared service's bound is not raised by it: a later request normalises exactly as before (its
    # recorded inputs digest does not depend on what ran earlier in the process)
    assert svc.max_new_tokens == bound and svc._norm(write) == before
    assert before["max_new_tokens"] == min(200, bound)
    # thinking-mode requests have their own fixed bound (H.5 asks for 384 reasoning tokens: not clamped to 128)
    assert svc._norm(dict(write, max_new_tokens=384, thinking=True))["max_new_tokens"] == min(384, svc.think_bound)
    assert svc.think_bound >= 384
    with pytest.raises(ValueError):
        B.LMBrain({c: small for c in routing.TABLE}, think={"tool_choice": svc.think_bound + 1}).choose(
            None, "tool_choice", [req])
    plain = B.LMBrain({c: small for c in routing.TABLE}, think={})
    assert plain.think_tokens("tool_choice") == 0 and plain.choose(None, "tool_choice", [req])[0] in (0, 1, 2)
    assert B.LMBrain({c: SMALL for c in routing.TABLE}, think={"tool_choice": 24}).think_tokens("tool_choice") == 0


def test_evaluator_excludes_junk_content_and_counts_repeated_content_once(ablated):
    """H.6 / SP-3.2d (audit D skeptic, verifier H.6 overclaim): a consequential call whose written content is junk (a
    tool name as a mail body) does not count, and the same text sent again later counts once; read by the evaluator
    from the event log, not from the participant's record. A mail counts through its delivery to the recipient, an
    event of the job the call scheduled."""
    from crucible.substrate import bus
    w = ablated
    rows = G.members(w)
    a = M.Agent.from_row(rows[-1])
    to = rows[0]["principal"]
    tool = next(t for t in w.available_tools(a.principal) if t["name"] == "mail.send")
    g = a.goal() or a.new_goal(w.now, "I will tell my friend about the rye bread I baked this morning.")
    a.active = g["id"]
    before = verify.summary(w)
    good = "I baked rye bread this morning with a sourdough starter; it took eleven hours to rise."

    def send(body, subject):
        def run(ctx):
            M.execute(ctx, a, tool, {"to": [to], "subject": subject, "body": body}, ctx.now, M.Config())
            bus.emit(ctx, "semp.update", M.delta(a, int(ctx.now) + 86400), vis="internal")
            a.add_memory, a.add_history, a.add_known = [], [], {}
        w._run_system("test.mail", run)
        w.advance_time(w.now + 600)          # the mail is delivered by a job the call scheduled (a consequence)
        assert a.history[-1]["ok"], a.history[-1]
    send("mail.send", "mail.send")
    send(good, "Rye bread")
    send(good, "Rye bread")
    after = verify.summary(w)
    acts = [x for x in verify.actions(w) if x["principal"] == a.principal][-3:]
    assert [x["consequential"] for x in acts] == [True, True, True]
    assert [x["junk"] for x in acts] == [True, False, False]
    assert acts[1]["signature"] == acts[2]["signature"]
    assert after["consequential_raw"] - before["consequential_raw"] == 3
    assert after["consequential"] - before["consequential"] == 1
    assert after["consequential_junk"] - before["consequential_junk"] == 1
    goal = "I will tell my friend about the rye bread I baked this morning."
    assert verify.junk_text(goal, set(), goal) and verify.junk_text("mail", {"mail", "mail.send"}, goal)
    assert not verify.junk_text(good, {"mail", "mail.send"}, goal)
