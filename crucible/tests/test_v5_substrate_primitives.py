"""Crucible 5.0 substrate: shared primitives with one owner each (contract 1.5) and the generic asset primitive."""
import pytest

import crucible
from crucible.core.errors import InvariantViolation, ProtocolError
from crucible.core.kernel import MODULES, Module, World, WorldSpec, register_module
from crucible.substrate import primitives as P
from crucible.worldgen.admission import admit
from crucible.worldgen.generator import FAMILIES, generate

crucible.load_all_modules()

CONTRACT_KINDS = ["identity", "organization", "account", "money", "asset", "resource_inventory", "content_artifact",
                  "product", "listing", "service", "software_repository", "relationship", "community", "institution",
                  "contract", "knowledge_artifact", "tool_definition", "subscription", "employment", "population",
                  "need", "time", "provenance", "permission", "visibility"]


def with_deps(names):
    order = []

    def add(m):
        if m in order:
            return
        for d in MODULES[m].depends:
            add(d)
        order.append(m)
    for m in names:
        add(m)
    return order


def owners_world(extra=()):
    mods = with_deps(["identity", "bank", "admin", *P.owner_modules(), *extra])
    return World.create(WorldSpec("prim", 1, [{"name": m, "config": {}} for m in mods]), strict=True)


def test_every_contract_kind_has_exactly_one_real_owner():
    assert P.kinds() == CONTRACT_KINDS
    assert P.implemented() == CONTRACT_KINDS
    w = owners_world()
    rep = P.check_no_duplicates(w)
    assert rep["present"] == len(CONTRACT_KINDS)
    from crucible.substrate import columns
    for k in CONTRACT_KINDS:
        o = P.owner(k)
        assert set(o) >= {"module", "table", "doc", "api", "ids", "since"}
        if o["module"] == P.KERNEL:
            assert o["table"].startswith("@") or o["table"] in w.state.tables
        elif o["table"].startswith("cols:"):        # a column table: the owner module declares its columns
            rec = columns._registry(w)[o["table"][5:]]
            assert o["module"] in w.modules and o["module"] in set(rec["by"].values())
        else:
            assert o["module"] in w.modules and o["table"] in w.state.tables
            assert o["module"] in w.state.declared_by[o["table"]]       # the owner module declares its table
    # D-11: the product is owned by v5 products; a 4.0 market listing is a listing
    assert (P.owner("product")["module"], P.owner("listing")["module"]) == ("products", "market")
    with pytest.raises(ProtocolError):
        P.owner("vibes")


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_no_4_0_family_keeps_a_shared_fact_twice(family):
    spec = generate({"family": family}, 0)
    w = World(spec, f"prim-{family}")          # modules installed (tables declared); genesis not needed
    assert P.problems(w) == []


@register_module
class DupCommunity(Module):
    """A module that keeps its own community table although `community` has an owner: a duplicate shared fact."""
    name = "primtest_dupcomm"
    shared_facts = {"community": "my_communities"}

    def install(self, world):
        world.state.declare_table("my_communities")


@register_module
class SharesTable(Module):
    """A module that writes another module's table."""
    name = "primtest_shares"

    def install(self, world):
        world.state.declare_table("accounts", index=("owner",))


@register_module
class NewKind(Module):
    name = "primtest_newkind"
    shared_facts = {"vibes": "vibes"}


def test_duplicates_are_detected():
    for bad, needle in (("primtest_dupcomm", "duplicate community"), ("primtest_shares", "table 'accounts'"),
                        ("primtest_newkind", "unknown shared kind")):
        w = World(WorldSpec("dup", 1, [{"name": m, "config": {}} for m in with_deps(["identity", "bank", "forum", bad])]),
                  "dup")
        probs = P.problems(w)
        assert any(needle in p for p in probs), (bad, probs)
        with pytest.raises(InvariantViolation):
            P.check_no_duplicates(w)


# ---------------------------------------------------------------------------------------------- assets
def asset_world(strict=True):
    mods = with_deps(["identity", "bank", "admin", "artifacts", "assets"])
    w = World.create(WorldSpec("assets", 2, [{"name": m, "config": {}} for m in mods]), strict=strict)
    for h, e in (("ann", 1000), ("ben", 300)):
        w.system_call("admin.join", {"handle": h, "endowment": e})
    return w


def bal(w, p):
    b = w.modules["bank"]
    return b.balance(w.state, b.primary_account(w.state, p))


def test_asset_sale_is_atomic_with_payment():
    w = asset_world()
    r = w.execute("p:ann", {"tool": "asset.register", "args": {"class": "licence", "title": "engine licence"}})
    assert r.ok, r.error
    aid = r.result["asset"]
    assert not w.execute("p:ann", {"tool": "asset.register", "args": {"class": "equity", "title": "x"}}).ok
    off = w.execute("p:ann", {"tool": "asset.offer", "args": {"asset": aid, "to": "p:ben", "price_cents": 500}})
    assert off.ok
    td = w.state.table_digests()
    r = w.execute("p:ben", {"tool": "asset.accept", "args": {"offer": off.result["offer"]}})   # ben has 300
    assert not r.ok and r.error["code"] == "insufficient_funds"
    assert w.state.table_digests() == td and w.state.get("assets", aid)["owner"] == "p:ann"
    w.system_call("admin.endow", {"principal": "p:ben", "amount": 300})
    r = w.execute("p:ben", {"tool": "asset.accept", "args": {"offer": off.result["offer"]}})
    assert r.ok, r.error
    assert w.state.get("assets", aid)["owner"] == "p:ben" and bal(w, "p:ann") == 1500 and bal(w, "p:ben") == 100
    tr = [e for e in w.log if e.type == "asset.transferred"][-1]
    assert tr.payload["txn"] == r.result["txn"] and tr.vis == ("p:ann", "p:ben")
    assert not w.execute("p:ann", {"tool": "asset.transfer", "args": {"asset": aid, "to": "p:ann"}}).ok
    assert w.check_invariants() == []
    evs = [e.to_dict() for e in w.log]
    assert World.replay(w.spec, evs).digest() == w.digest()
    assert World.regenerate(w.spec, evs).last_hash == w.last_hash


def test_asset_visibility_and_backing_artifact():
    import base64
    w = asset_world()
    ref = w.execute("p:ann", {"tool": "artifact.put", "args": {"data_b64": base64.b64encode(b"font").decode(),
                                                               "mime": "font/ttf"}}).result["ref"]
    aid = w.execute("p:ann", {"tool": "asset.register", "args": {"class": "digital-good", "title": "font",
                                                                 "ref": ref}}).result["asset"]
    assert not w.execute("p:ben", {"tool": "asset.info", "args": {"asset": aid}}).ok
    assert not w.execute("p:ben", {"tool": "asset.register", "args": {"class": "digital-good", "title": "stolen",
                                                                      "ref": ref}}).ok
    assert w.execute("p:ann", {"tool": "asset.transfer", "args": {"asset": aid, "to": "p:ben"}}).ok
    assert w.execute("p:ben", {"tool": "asset.info", "args": {"asset": aid}}).result["owner"] == "p:ben"


def test_assets_module_passes_admission():
    rep = admit(["assets"], steps=300)
    assert rep.passed, (rep.checks, rep.errors[:5])
