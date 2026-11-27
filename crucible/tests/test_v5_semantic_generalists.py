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
    n0 = svc.stats["requests"]
    picks = b.choose(None, "tool_choice", [req])
    assert len(picks) == 1 and 0 <= picks[0] < 3 and b.calls["think"] == 1
    assert svc.stats["requests"] - n0 >= 1          # the thinking generation (plus a rescoring when it named none)
    plain = B.LMBrain({c: small for c in routing.TABLE}, think={})
    assert plain.think_tokens("tool_choice") == 0 and plain.choose(None, "tool_choice", [req])[0] in (0, 1, 2)
    assert B.LMBrain({c: SMALL for c in routing.TABLE}, think={"tool_choice": 24}).think_tokens("tool_choice") == 0
