"""Tier 2 domain 9.11 (information ecosystem, kernel module `infoeco`): the reader, the world's mechanics, the
participant's tools and the independent evaluator, on real generated worlds."""
import numpy as np
import pytest

from crucible.domains import _dom3 as K
from crucible.domains import information as I


def _env(config="local_hazard", seed=0, **kw):
    return K.make_env("information", config, seed, **kw)


# ------------------------------------------------------------------------------------------------ the reader
def test_reader_takes_the_value_nearest_after_the_attribute_keyword():
    body = "Copper Lantern Bakery opens at 07:30 and closes at 18:00. A standard visit costs $4.50."
    assert I.read_claim("x", body, "Copper Lantern Bakery", "open", 3) == ("07:30", -1)
    assert I.read_claim("x", body, "Copper Lantern Bakery", "close", 3) == ("18:00", -1)
    # the price sentence does not name the entity: it counts only on a page whose title names it
    assert I.read_claim("x", body, "Copper Lantern Bakery", "price", 3) is None
    assert I.read_claim("Copper Lantern Bakery", body, "Copper Lantern Bakery", "price", 3) == ("$4.50", -1)


def test_reader_skips_announcements_not_yet_in_effect_and_prefers_dated_claims():
    body = ("Harbor Line first departure: 05:15. From day 9, the first departure of the Harbor Line is at 05:45. "
            "As of day 2, the Harbor Line runs its first service at 05:30.")
    assert I.read_claim("t", body, "Harbor Line", "first", 5) == ("05:30", 2)
    assert I.read_claim("t", body, "Harbor Line", "first", 9) == ("05:45", 9)


def test_reader_needs_the_entity_named_in_the_sentence_or_the_title():
    other = (I.name_tokens("Maple Cafe"),)
    assert I.read_claim("Some page", "It opens at 07:00.", "Copper Gym", "open", 1) is None
    assert I.read_claim("Copper Gym hours", "It opens at 07:00.", "Copper Gym", "open", 1, other) == ("07:00", -1)
    # a sentence about another entity on a page about this one is not about this one
    assert I.read_claim("Copper Gym hours", "Maple Cafe opens at 08:00.", "Copper Gym", "open", 1,
                        other + (I.name_tokens("Copper Gym"),)) is None
    # gibberish with the right words but no value says nothing
    assert I.read_claim("Copper Gym", "Copper Gym opens opening open best guide", "Copper Gym", "open", 1) is None


def test_generated_instances_differ_by_seed_and_keep_entity_names_unambiguous():
    a = I.generate_instance(np.random.default_rng(1), {"kind": "local"})
    b = I.generate_instance(np.random.default_rng(2), {"kind": "local"})
    assert [e["name"] for e in a["entities"]] != [e["name"] for e in b["entities"]]
    toks = [I.name_tokens(e["name"]) for e in a["entities"]]
    assert all(not (x < y) for x in toks for y in toks)          # no name is a strict subset of another
    assert abs(sum(e["popularity"] for e in a["entities"]) - 1.0) < 1e-3


# ------------------------------------------------------------------------------------------------ the world
def test_world_is_observation_only_and_tools_never_return_hidden_values():
    env = _env()
    a = env.agents()[0]
    assert env.reward_mode == "observation_only"
    st = env.world.state
    truth = {r["text"] for r in st.tables["info_facts"].values()}
    ents = env.execute(a, {"tool": "info.entities", "args": {}})
    assert ents["ok"] and not any(v in str(ents["result"]) for v in truth)
    obs = env.observe(a)
    assert "infoeco" not in str(obs.get("numbers", {}))
    env.advance(86400)
    assert env.reward(a) == 0.0


def test_same_seed_same_log_and_facts_change_and_official_pages_follow():
    e1, e2 = _env(seed=3), _env(seed=3)
    for e in (e1, e2):
        e.advance(12 * 86400)
    assert e1.world.digest() == e2.world.digest()
    assert [ev.hash for ev in e1.world.events_range(0)][-1] == [ev.hash for ev in e2.world.events_range(0)][-1]
    w = e1.world
    led = I._ledger(w)
    changed = [fid for fid, h in led["facts"].items() if len(h) > 1]
    assert changed, "no fact changed in 12 days"
    edits = K.event_counts(w)["docs.edit"]
    assert edits > 0
    assert not w.check_invariants()


def test_service_tools_ownership_verification_cost_and_capacity():
    env = _env()
    a = env.agents()[0]
    w = env.world
    r = env.execute(a, {"tool": "info.service_create", "args": {"name": "Probe", "description": "opening hours"}})
    sid = r["result"]["service"]
    docs = [x["id"] for x in env.execute(a, {"tool": "search.query", "args": {"q": I.KINDS["local"]["b"][0],
                                                                             "kind": "doc"}})["result"]["results"]]
    assert env.execute(a, {"tool": "info.service_add", "args": {"service": sid, "docs": docs}})["ok"]
    other = env.add_agent("other")
    assert not env.execute(other, {"tool": "info.service_stats", "args": {"service": sid}})["ok"]
    bank = w.modules["bank"]
    acct = bank.primary_account(w.state, a)
    b0 = bank.balance(w.state, acct)
    ent = w.modules["infoeco"].ents[0]
    res = env.execute(a, {"tool": "info.verify", "args": {"entity": ent["id"], "attribute": "open"}})
    assert res["ok"] and res["result"]["value"] == w.state.tables["info_facts"][f"{ent['id']}|open"]["text"]
    assert bank.balance(w.state, acct) == b0 - 400
    n_ok = sum(env.execute(a, {"tool": "info.verify", "args": {"entity": ent["id"], "attribute": "close"}})["ok"]
               for _ in range(15))
    assert n_ok == 11                                           # 12 calls a day in total


def test_population_lookups_have_provenance_and_outcomes_follow_truth():
    env = _env()
    env.advance(6 * 86400)
    w = env.world
    led = I._ledger(w)
    looks = led["lookups"]
    assert len(looks) > 100
    assert {r["outcome"] for r in looks} >= {"ok", "wrong"}
    for r in looks[:200]:
        if r["claim"] is not None:
            assert (r["outcome"] == "ok") == (r["claim"] == I.truth_at(led, f"{r['entity']}|{r['attr']}", r["t"]))
    prov = K.session_provenance(w, I.TABLE, ["info.lookup"])
    assert prov["batches"] > 10 and prov["forced_rows"] == 0 and prov["organic_rows"] == prov["rows"], prov


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_reference_succeeds_where_idle_and_index_everything_fail():
    out = {}
    for name, pol in (("ref", I.Reference), ("idle", lambda w, a: K.IdlePolicy()), ("all", I.IndexAllProbe)):
        env = _env(seed=1)
        a = env.agents()[0]
        K.run_env(env, a, pol(env.world, a), days=30)
        out[name] = env.verify()["evaluation"][a]
    assert out["ref"]["success"], out["ref"]
    assert not out["idle"]["success"] and out["idle"]["details"]["independent_users"] == 0
    assert not out["all"]["success"]
    assert out["ref"]["details"]["success_rate"] > out["all"]["details"]["success_rate"] + 0.2
