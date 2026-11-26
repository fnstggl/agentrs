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
    rel = [d for d in fr["known_duplicates"] if d["table"] == "social.cols:follow"]
    assert rel and rel[0]["kind"] == "relationship"
    assert rel[0]["owner"] == "forum.forum_follows" and rel[0]["owner_rows"] == len(w.state.tables["forum_follows"])
    assert rel[0]["owner_rows"] > 0 and rel[0]["rows"] == 0       # two unsynchronised graphs, measured
    from crucible.substrate import columns
    assert fr["owners"]["population"]["rows"] == columns.rows(w, "pop") > 0


@register_module
class FactsLegacyOld(Module):
    """A '4.0' module keeping relationship facts of its own (like forum.forum_follows) and, where the kind's owner is
    installed, writing through it instead (the migration a legacy entry certifies)."""
    name = "factstest_oldties"

    def install(self, world):
        world.state.declare_table("factstest_old_ties")

    def add(self, ctx, a, b):
        if "factstest_follows" in ctx.world.modules:       # the owner is here: write through it
            ctx.state.put("factstest_followers", f"{a}|{b}", {"target": b})
        else:
            ctx.state.put("factstest_old_ties", f"{a}|{b}", {"b": b})


def _bare(*names):
    """Without the forum: in these worlds the test module is the relationship owner (monkeypatched)."""
    return World(WorldSpec("facts", 1, [{"name": m, "config": {}} for m in ["identity", "bank", *names]]), "facts")


def _tie(w, a, b, split=False):
    def go(ctx):
        if split:
            ctx.state.put("factstest_old_ties", f"{a}|{b}", {"b": b})
        else:
            w.modules["factstest_oldties"].add(ctx, a, b)
    w._run_system("test", go)


def test_a_legacy_table_never_holds_facts_where_the_owner_is_installed(monkeypatch):
    """The migration target for 4.0 tables (4.0 worlds stay byte-identical, so they cannot be retired): a legacy
    table holds a kind's facts only in worlds without its owner; with the owner installed, holding facts is a split
    measured on data in that world. A legacy entry needs a test showing the delegation (`verified_by`), and does not
    make its kind a known duplicate."""
    monkeypatch.setitem(P.OWNERS, "relationship", dict(P.OWNERS["relationship"], module="factstest_follows",
                                                       table="factstest_followers"))
    entry = facts._r("relationship", "legacy", "test 4.0 ties", verified_by=__name__)
    monkeypatch.setitem(facts.REVIEWED, "factstest_oldties.factstest_old_ties", entry)
    alone = _bare("factstest_oldties")                  # a '4.0 world': the legacy table is the only copy
    _tie(alone, "p:a", "p:b")
    assert len(alone.state.tables["factstest_old_ties"]) == 1 and facts.world_problems(alone) == []
    both = _bare("factstest_follows", "factstest_oldties")
    _tie(both, "p:a", "p:b")                             # delegated: the owner holds the tie, the legacy table none
    assert facts.world_problems(both) == [] and len(both.state.tables["factstest_followers"]) == 1
    assert facts.world_report(both)["legacy"] == [{"table": "factstest_oldties.factstest_old_ties",
                                                   "kind": "relationship", "rows": 0}]
    _tie(both, "p:c", "p:d", split=True)                 # a module that keeps writing its own copy: split facts
    probs = facts.world_problems(both)
    assert len(probs) == 1 and "split relationship facts" in probs[0] and "holds 1" in probs[0]
    with pytest.raises(InvariantViolation):
        P.check_no_duplicates(both)
    a = facts.audit(["factstest_follows", "factstest_oldties"])
    assert a["legacy"] == ["factstest_oldties.factstest_old_ties"] and a["known_duplicates"] == []
    # without a verifying test, or naming the owner's own module, the entry is refused
    monkeypatch.setitem(facts.REVIEWED, "factstest_oldties.factstest_old_ties",
                        facts._r("relationship", "legacy", "unverified"))
    assert any("names no test" in p for p in facts.audit(["factstest_oldties"])["problems"])


@register_module
class FactsColumns(Module):
    """Columns on the population-like column table of a test module: one carries a tie (contacts), one does not."""
    name = "factstest_cols"
    depends = ("cols",)

    def install(self, world):
        from crucible.substrate import columns
        columns.declare(world, "factstest_people", 8, {"contacts": ("int32", (4,), -1), "mood": ("float32", (), 0.0)})


def test_every_column_is_scanned_on_its_own_name():
    """A per-individual tie list declared as a column (as `pop.contacts` was) is a relationship fact whoever
    declares it: the column-level scan finds it without any declaration; a neutral column stays private."""
    w = World(WorldSpec("facts", 1, [{"name": m, "config": {}} for m in ["identity", "bank", "artifacts", "cols",
                                                                          "factstest_cols"]]), "facts")
    items = facts.world_items(w)
    assert "factstest_cols.cols:factstest_people.contacts" in items
    assert "factstest_cols.cols:factstest_people.mood" not in items
    probs = facts.world_problems(w)
    assert any("factstest_cols.cols:factstest_people.contacts" in p and "relationship" in p for p in probs)
    a = facts.audit(["factstest_cols"])
    assert "factstest_cols.cols:factstest_people.contacts" in a["unreviewed"]
    # the population's own columns of other kinds are reviewed, and the duplicates are on the allow-list
    for k in ("pop.cols:pop.contacts", "pop.cols:pop.communities", "aud.cols:aud_creators.followers"):
        assert facts.REVIEWED[k]["role"] == "duplicate" and k in facts.KNOWN_DUPLICATES
