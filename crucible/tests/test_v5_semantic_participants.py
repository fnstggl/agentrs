"""Crucible 5.0 workstream D2: semantic participants (prompting, candidates, batching, recording) and the Gate H
society's evaluator-side verifiers. Model-backed tests use SmolLM2-135M-Instruct on the CPU."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import crucible  # noqa: E402
from crucible.core.kernel import World  # noqa: E402
from crucible.semantic import participants as SP  # noqa: E402
from crucible.semantic import scenarios as SC  # noqa: E402
from crucible.worldgen import generator as G  # noqa: E402

SMALL = "HuggingFaceTB/SmolLM2-135M-Instruct"
crucible.load_all_modules()


def _world(family="tiny"):
    return World.create(G.generate({"family": family, "extra_modules": ["semantic_inference"],
                                    "overrides": {"params": {"cost_model": "deterministic"}}}, 0))


def test_parse_json_object():
    assert SP.parse_json_object('Sure! ```json\n{"a": 1, "b": "x}"}\n``` done') == {"a": 1, "b": "x}"}
    assert SP.parse_json_object("<think>{\"no\": 1}</think> {'to': ['p:bob']}") == {"to": ["p:bob"]}
    assert SP.parse_json_object("no json here") is None


def test_candidates_follow_goal_and_discovery_without_a_model():
    w = _world()
    me = w.system_call("admin.join", {"handle": "ana", "endowment": 1000})["principal"]
    p = SP.SemanticParticipant(None, "You owe ben 500 cents for the logo. Pay him from your bank account.")
    p.reset(me, None)
    p.ingest(w.observe(me))
    names = [t["name"] for t in p.candidates(w.available_tools(me))]
    assert "bank.transfer" in names and names[-2:] == ["world.find_tools", "wait"]
    assert len(names) <= p.n_candidates + 1
    act = {"tool": "world.find_tools", "args": {"query": "calendar meeting"}}
    p.observe_result(act, w.execute(me, act).to_dict())
    names = [t["name"] for t in p.candidates(w.available_tools(me))]
    assert any(n.startswith("cal.") for n in names)      # tools it discovered itself become candidates


def test_repeated_read_only_call_is_not_offered_again_until_something_new_happens():
    w = _world()
    me = w.system_call("admin.join", {"handle": "ana", "endowment": 1000})["principal"]
    p = SP.SemanticParticipant(None, "Check how much money is in your bank accounts and list them.")
    p.reset(me, None)
    tools = w.available_tools(me)
    p.ingest(w.observe(me))
    assert "bank.accounts" in [t["name"] for t in p.candidates(tools)]
    a = {"tool": "bank.accounts", "args": {}}
    p.observe_result(a, w.execute(me, a).to_dict())
    assert "bank.accounts" not in [t["name"] for t in p.candidates(tools)]
    obs = w.observe(me)
    obs["events"] = [{"seq": 1, "t": 5, "type": "bank.txn", "actor": "p:ben", "payload": {"amount": 3}}]
    p.ingest(obs)
    assert "bank.accounts" in [t["name"] for t in p.candidates(tools)]


def test_prompts_are_short_and_carry_own_goal_and_history_only():
    w = _world()
    a = w.system_call("admin.join", {"handle": "ana", "endowment": 1000})["principal"]
    b = w.system_call("admin.join", {"handle": "ben", "endowment": 1000})["principal"]
    pa = SP.SemanticParticipant(None, "Publish a public document about bread.")
    pb = SP.SemanticParticipant(None, "Ask cara about the meeting.")
    pa.reset(a, None)
    pb.reset(b, None)
    for p, me in ((pa, a), (pb, b)):
        p.ingest(w.observe(me))
    act = {"tool": "docs.create", "args": {"title": "Bread", "body": "Rye.", "visibility": "public"}}
    pa.observe_result(act, w.execute(a, act).to_dict())
    ra = pa.choice_request(pa.candidates(w.available_tools(a)))
    rb = pb.choice_request(pb.candidates(w.available_tools(b)))
    ua, ub = ra["messages"][1]["content"], rb["messages"][1]["content"]
    assert "bread" in ua and "bread" not in ub and "docs.create" in ua.split("Your recent actions:")[1]
    assert ra["messages"][0] == rb["messages"][0]           # the shared prefix (system prompt) is identical
    assert len(ua) < 6000 and len(ub) < 6000


def test_consequential_actions_are_read_from_the_log():
    w = _world()
    a = w.system_call("admin.join", {"handle": "ana", "endowment": 1000})["principal"]
    b = w.system_call("admin.join", {"handle": "ben", "endowment": 1000})["principal"]
    w.execute(a, {"tool": "mail.inbox", "args": {}})
    w.execute(a, {"tool": "mail.send", "args": {"to": [b], "subject": "hi", "body": "hello"}})
    w.execute(a, {"tool": "bank.transfer", "args": {"from_account": "acct_nope", "to": b, "amount": 5}})
    ca = SP.consequential_actions(w, [a])
    assert [c["tool"] for c in ca] == ["mail.send"]


def test_society_verifiers_accept_scripted_successes_and_reject_nothing_done():
    from crucible.semantic.service import SemanticService
    svc = SemanticService(SMALL, device="cpu", max_new_tokens=640)   # weights load lazily: never used here
    soc = SC.build_society(svc, record=False)
    w, P = soc.world, soc.principals
    assert not any(v["verified"] for v in SC.verify(w, soc).values())
    acct = {m.handle: m.account for m in soc.members}
    ex = lambda h, tool, args: w.execute(P[h], {"tool": tool, "args": args})
    assert ex("ana", "docs.create", {"title": "Opening hours", "body": "Monday to Friday 8 to 18",
                                     "visibility": "public"}).ok
    assert ex("ben", "bank.transfer", {"from_account": acct["ben"], "to": P["cara"], "amount": 2500}).ok
    assert ex("cara", "mail.send", {"to": [P["dev"]], "subject": "meeting?", "body": "When is the team meeting?"}).ok
    assert ex("dev", "mail.send", {"to": [P["cara"]], "subject": "re", "body": "It is on Thursday at 14:00."}).ok
    assert ex("eli", "code.create_repo", {"name": "textutils"}).ok
    assert ex("eli", "code.commit", {"repo": "eli/textutils", "files": {
        "textutils.py": "def word_count(text):\n    return len(text.split())\n"}}).ok
    assert ex("fay", "forum.post", {"board": "general", "title": "Cheapest cloud?", "body": "Which provider?"}).ok
    provs = w.execute(P["gus"], {"tool": "cloud.providers", "args": {}}).result
    assert ex("gus", "mail.send", {"to": [P["hana"]], "subject": "providers",
                                   "body": ", ".join(p["name"] for p in provs)}).ok
    assert ex("hana", "cloud.rent", {"provider": provs[0]["provider"], "cores": 1, "account": acct["hana"]}).ok
    w.advance_time(w.now + 600)
    res = SC.verify(w, soc)
    assert all(v["verified"] for v in res.values()), json.dumps(res)
    assert res["eli"]["tests_passed"] == res["eli"]["tests"] == 5


def test_program_verifier_rejects_wrong_code():
    from crucible.semantic.service import SemanticService
    svc = SemanticService(SMALL, device="cpu", max_new_tokens=640)
    soc = SC.build_society(svc, record=False)
    w, P = soc.world, soc.principals
    w.execute(P["eli"], {"tool": "code.create_repo", "args": {"name": "textutils"}})
    w.execute(P["eli"], {"tool": "code.commit", "args": {"repo": "eli/textutils", "files": {
        "textutils.py": "def word_count(text):\n    return len(text.split(' '))\n"}}})
    r = SC.check_program(w, P)
    assert not r["verified"] and 0 < r["tests_passed"] < r["tests"]


@pytest.fixture(scope="module")
def small_service():
    from crucible.semantic.service import SemanticService, local_revision
    try:
        local_revision(SMALL)
    except FileNotFoundError:
        pytest.fail(f"{SMALL} is not in the local Hugging Face cache (download it once, about 270 MB)")
    s = SemanticService(SMALL, device="cpu", max_batch=8, max_new_tokens=160)
    yield s
    s.close()


def test_batched_group_turns_are_recorded_and_regenerate_without_the_model(small_service):
    w = _world()
    group = SP.SemanticGroup(small_service, w)
    goals = ["Send ben a mail saying hello.", "Pay ana 100 cents.", "Publish a public document about cats."]
    agents = []
    for h, g in zip(["ana", "ben", "cara"], goals):
        pid = w.system_call("admin.join", {"handle": h, "endowment": 5000})["principal"]
        agents.append((pid, group.participant(g)))
    r = SP.BatchedRunner(w, agents, dt=600, seed=3)
    r.run(2)
    infer = [e for e in w.events_range(0) if e.type == "semantic.inference"]
    # per round: one tool-choice batch for all three, then at most one value-choice and one value-writing batch
    assert 2 <= len(infer) <= 6 and 2 <= group.calls["choose"] <= 4
    assert set(infer[0].payload["principals"]) == {a for a, _ in agents}
    assert all(p.stats["turns"] == 2 for _, p in agents)
    assert len({id(p.history) for _, p in agents}) == 3
    calls = []
    orig = small_service.run
    small_service.run = lambda *a, **k: calls.append(1) or orig(*a, **k)
    try:
        rg = World.regenerate(w.spec, [e.to_dict() for e in w.events_range(0)])
    finally:
        small_service.run = orig
    assert calls == [] and rg.last_hash == w.last_hash
