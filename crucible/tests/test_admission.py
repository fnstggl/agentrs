"""Module admission: existing services pass; a nondeterministic module and a money-minting module are rejected."""
import time

import crucible

crucible.load_all_modules()
from crucible.core import schema as S  # noqa: E402
from crucible.core.kernel import Module, applier, endpoint, register_module  # noqa: E402
from crucible.worldgen.admission import admit  # noqa: E402


@register_module
class _Flaky(Module):
    name = "flaky_test_module"

    def install(self, world):
        world.state.declare_table("flaky")

    @endpoint("note", S.obj({"text": S.s_str(maxLength=50)}, ["text"]), doc="store a note")
    def ep_note(self, ctx, text):
        # BUG on purpose: wall-clock time in the event payload -> regeneration diverges
        ctx.emit("flaky.note", {"id": ctx.state.new_id("fl"), "text": text, "stamp": time.time_ns()})
        return {"ok": True}

    @applier("flaky.note")
    def ap(self, ev):
        self.world.state.insert("flaky", ev.payload["id"], {"text": ev.payload["text"], "stamp": ev.payload["stamp"]})


@register_module
class _Minter(Module):
    name = "minter_test_module"
    depends = ("bank",)

    @endpoint("gift", S.obj({}), doc="free money")
    def ep_gift(self, ctx):
        acct = ctx.world.modules["bank"].primary_account(ctx.state, ctx.actor)
        ctx.emit("minter.gift", {"acct": acct})
        return {"ok": True}

    @applier("minter.gift")
    def ap(self, ev):   # BUG on purpose: bypasses the ledger
        a = self.world.state.tables["accounts"][ev.payload["acct"]]
        self.world.state.patch("accounts", ev.payload["acct"], balance=a["balance"] + 100)


def test_existing_services_are_admissible():
    for m in (["cal"], ["docs"], ["forum", "search"], ["contracts"]):
        r = admit(m, steps=250)
        assert r.passed, (m, r.checks, r.errors[:5])


def test_nondeterministic_module_rejected():
    r = admit(["flaky_test_module"], steps=150)
    assert not r.passed and not r.checks["regeneration"]


def test_money_minting_module_rejected():
    r = admit(["minter_test_module"], steps=150)
    assert not r.passed and not r.checks["invariants"]
