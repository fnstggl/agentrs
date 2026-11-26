"""Crucible 5.0 D-21: the duplicate-fact check is mandatory - it scans every table, column table and dense array of
every registered module (not opt-in declarations), and known duplicates pass only through an explicit, reviewed
allow-list with expiry dates; D-11 is resolved (product owned by `products`, 4.0 market listings are `listing`)."""
import datetime

import pytest

import crucible
from crucible.core.errors import InvariantViolation
from crucible.core.kernel import MODULES, Module, World, WorldSpec, register_module
from crucible.substrate import facts
from crucible.substrate import primitives as P

crucible.load_all_modules()


@pytest.fixture(scope="module")
def registry_audit():
    return facts.audit()


def test_every_module_of_the_registry_is_scanned_and_classified(registry_audit):
    a = registry_audit
    assert a["uninstallable"] == {} and a["modules"] == len(facts.shipped_modules()) >= 85
    assert set(facts.shipped_modules()) <= set(MODULES)
    assert a["problems"] == [], a["problems"]
    assert a["unreviewed"] == [] and a["items"] >= 250
    # every shared kind with a module owner has its owner among the scanned items
    for kind, o in P.OWNERS.items():
        if o["module"] != P.KERNEL and not o["table"].startswith("@"):
            assert f"{o['module']}.{o['table']}" in a["owners"], kind


def test_the_known_duplicates_are_exactly_the_reviewed_allow_list(registry_audit):
    a = registry_audit
    assert a["duplicates"] == a["known_duplicates"] == sorted(facts.KNOWN_DUPLICATES)
    for key, e in facts.KNOWN_DUPLICATES.items():
        assert set(e) >= {"target_owner", "decision", "workstream", "migration", "expires"}
        datetime.date.fromisoformat(e["expires"])
        assert facts.REVIEWED[key]["role"] == "duplicate" and facts.REVIEWED[key]["kind"] in P.OWNERS
    kinds = {facts.REVIEWED[k]["kind"] for k in facts.KNOWN_DUPLICATES}
    # the audit's duplicates (A6) are all on record: people, employment, relationships, communities, institutions,
    # contracts, subscriptions, listings, content metadata
    assert kinds >= {"population", "employment", "relationship", "community", "institution", "contract",
                     "subscription", "listing", "content_artifact"}
    assert sum(1 for k in facts.KNOWN_DUPLICATES if facts.REVIEWED[k]["kind"] == "population") >= 8


def test_expired_allow_list_entries_fail_the_check():
    later = datetime.date(2099, 1, 1)
    a = facts.audit(names=["crowd", "forum", "social", "pop"], today=later)
    assert any("expired" in p and "crowd.cols:pop" in p for p in a["problems"])
    assert any("expired" in p and "social.cols:follow" in p for p in a["problems"])


def test_d11_product_is_owned_by_products_and_market_listings_are_listings():
    assert P.owner("product") == {**P.owner("product"), "module": "products", "table": "products"}
    assert P.owner("listing")["module"] == "market" and P.PRIMITIVE_EVENTS["market.listing"][0] == "listing"
    assert facts.classify("market.listings")["kinds"] == ["listing"]
    assert facts.classify("products.products")["kinds"] == ["product"]
    assert facts.classify("xchg.xchg_listings")["role"] == "duplicate"


@register_module
class FactsDupFollows(Module):
    """Keeps its own follow list although `relationship` has an owner - and declares nothing."""
    name = "factstest_follows"

    def install(self, world):
        world.state.declare_table("factstest_followers", index=("target",))


@register_module
class FactsPrivate(Module):
    """A table without any shared-kind signature: module-private by default."""
    name = "factstest_private"

    def install(self, world):
        world.state.declare_table("factstest_scratch")


def _world(*names):
    return World(WorldSpec("facts", 1, [{"name": m, "config": {}} for m in ["identity", "bank", "forum", *names]]),
                 "facts")


def test_an_undeclared_copy_of_a_shared_fact_is_caught_without_any_declaration():
    w = _world("factstest_follows")
    probs = P.problems(w)
    assert any("unreviewed table factstest_follows.factstest_followers" in p and "relationship" in p for p in probs)
    with pytest.raises(InvariantViolation):
        P.check_no_duplicates(w)
    assert P.problems(_world("factstest_private")) == []


def test_a_duplicate_off_the_allow_list_fails(monkeypatch):
    monkeypatch.setitem(facts.REVIEWED, "factstest_follows.factstest_followers",
                        {"kind": "relationship", "role": "duplicate", "why": "test"})
    probs = P.problems(_world("factstest_follows"))
    assert any("not on the reviewed allow-list" in p for p in probs)
    monkeypatch.setitem(facts.KNOWN_DUPLICATES, "factstest_follows.factstest_followers",
                        {"target_owner": "forum.forum_follows", "decision": "test", "workstream": "A",
                         "migration": "test", "expires": "2099-01-01"})
    w = _world("factstest_follows")
    assert P.problems(w) == []
    rep = P.check_no_duplicates(w)
    known = [d for d in rep["facts"]["known_duplicates"] if d["table"] == "factstest_follows.factstest_followers"]
    assert known and known[0]["rows"] == 0 and known[0]["owner"] == "forum.forum_follows"
    assert any("known duplicate relationship" in x for x in rep["warnings"])


@pytest.mark.timeout(600)
def test_the_data_level_report_counts_the_facts_each_copy_holds():
    """In a running population world with the v5 social graph loaded, people follow on the forum (the 4.0 owner)
    while the social graph that the recommenders read stays empty: the report shows it (audit P6)."""
    from crucible.population.worlds import population_world
    sp = population_world("development", 2, n=240, sites=6, adaptive=False, strategic=False)
    have = {m["name"] for m in sp.modules}
    for m in facts._with_deps(["social"]):
        if m not in have:
            sp.modules.append({"name": m, "config": {}})
            have.add(m)
    w = World.create(sp, world_id="facts-data")
    w.advance_time(2 * 86400)
    rep = P.check_no_duplicates(w)
    fr = rep["facts"]
    assert fr["problems"] == []
    rel = [d for d in fr["known_duplicates"] if d["kind"] == "relationship"]
    assert rel and rel[0]["table"] == "social.cols:follow"
    assert rel[0]["owner"] == "forum.forum_follows" and rel[0]["owner_rows"] == len(w.state.tables["forum_follows"])
    assert rel[0]["owner_rows"] > 0 and rel[0]["rows"] == 0       # two unsynchronised graphs, measured
    from crucible.substrate import columns
    assert fr["owners"]["population"]["rows"] == columns.rows(w, "pop") > 0
