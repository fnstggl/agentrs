"""Crucible 5.0 substrate: typed events and same-transaction subscribers (contract 1.4), kernel extension point."""
import numpy as np
import pytest

import crucible
from crucible.core import schema as S
from crucible.core.errors import InvalidAction, ProtocolError
from crucible.core.kernel import Module, World, WorldSpec, applier, endpoint, register_module
from crucible.substrate import bus
from crucible.substrate import columns as C

crucible.load_all_modules()

bus.event_type("bustest.ping", S.obj({"n": {"type": "integer", "minimum": 0}, "note": {"type": "string"}}, ["n"]),
               owner="bustest_src", doc="test ping")
bus.event_type("bustest.pong", S.obj({"n": {"type": "integer"}}, ["n"]), owner="bustest_dst", doc="test pong")


@register_module
class BusSrc(Module):
    name = "bustest_src"
    host = "bsrc"
    depends = ("identity", "bank")

    @endpoint("ping", S.obj({"n": S.s_int(minimum=-5, maximum=1000), "pay": S.s_int(minimum=0)}, ["n"]),
              doc="emit a typed ping (optionally paying the system first)")
    def ep_ping(self, ctx, n, pay=0):
        if pay:
            bank = ctx.world.modules["bank"]
            bank.move(ctx, bank.primary_account(ctx.state, ctx.actor), "acct_fees", pay, "fee", "ping")
        bus.emit(ctx, "bustest.ping", {"n": n})
        return {"n": n, "pongs": ctx.state.count("bt_pongs")}

    @endpoint("raw", S.obj({"bad": S.s_bool()}), doc="emit a ping with plain ctx.emit")
    def ep_raw(self, ctx, bad=False):
        ctx.emit("bustest.ping", {"n": "x"} if bad else {"n": 1}, vis="internal")
        return {}

    @applier("bustest.ping")
    def ap_ping(self, ev):
        st = self.world.state
        st.put("bt_pings", str(ev.seq), {"n": ev.payload["n"], "pongs_seen": st.count("bt_pongs")})

    def install(self, world):
        world.state.declare_table("bt_pings")


@register_module
class BusDst(Module):
    """Couples to bustest_src only through the typed event: every ping makes a pong, charges a fee, and a ping
    with n > 100 is refused by this module (rolling back the source's command)."""
    name = "bustest_dst"
    host = "bdst"
    depends = ("bustest_src",)

    def install(self, world):
        world.state.declare_table("bt_pongs")
        if self.config.get("subscribe", True):
            bus.subscribe(world, "bustest.ping", self.on_ping, owner=self.name)
        if self.config.get("recurse"):
            bus.subscribe(world, "bustest.pong", self.on_pong, owner=self.name)

    def on_ping(self, ctx, ev):
        assert ctx.actor == "sys" and ctx.state.in_tx
        if ev.payload["n"] > 100:
            raise InvalidAction("pings above 100 are refused downstream")
        if ev.payload["n"] == 99:
            raise ZeroDivisionError("a downstream bug")
        bank = ctx.world.modules["bank"]
        acct = bank.primary_account(ctx.state, ev.actor)
        bank.move(ctx, acct, "acct_fees", 1, "fee", "pong")
        ctx.emit("bustest.pong", {"n": ev.payload["n"]}, vis=(ev.actor,))

    def on_pong(self, ctx, ev):
        ctx.emit("bustest.pong", {"n": ev.payload["n"] + 1}, vis="internal")    # unbounded recursion

    @applier("bustest.pong")
    def ap_pong(self, ev):
        self.world.state.put("bt_pongs", str(ev.seq), {"n": ev.payload["n"]})


def make(dst_cfg=None, strict=True, params=None):
    sp = WorldSpec("bus-test", 5, [{"name": "identity", "config": {}}, {"name": "bank", "config": {}},
                                   {"name": "admin", "config": {}}, {"name": "bustest_src", "config": {}},
                                   {"name": "bustest_dst", "config": dst_cfg or {}}], params or {})
    w = World.create(sp, strict=strict)
    w.system_call("admin.join", {"handle": "ann", "endowment": 100})
    return w


def test_registry_types_and_conflicts():
    t = bus.types()
    for name in ("artifact.stored", "artifact.published", "cols.batch", "cols.grow", "bustest.ping"):
        assert name in t and t[name]["owner"] and t[name]["schema"]["type"] == "object"
    bus.event_type("bustest.ping", t["bustest.ping"]["schema"], owner="bustest_src")       # idempotent
    with pytest.raises(ProtocolError):
        bus.event_type("bustest.ping", S.obj({"m": {"type": "integer"}}), owner="bustest_src")
    with pytest.raises(ProtocolError):
        bus.event_type("noDot", S.obj(), owner="x")
    with pytest.raises(ProtocolError):
        bus.event_type("bustest.bad", {"type": "object", "properties": {"a": {"type": "wat"}}}, owner="x")


def test_subscriber_runs_in_the_same_transaction_after_the_applier():
    w = make()
    r = w.execute("p:ann", {"tool": "bsrc.ping", "args": {"n": 3}})
    assert r.ok, r.error
    assert r.result["pongs"] == 1                     # the handler saw the subscriber's effect when emit returned
    types = [e.type for e in w.log[-6:]]
    i = types.index("bustest.ping")
    assert types[i + 1:i + 3] == ["bank.txn", "bustest.pong"]
    ping = [p for p in w.state.tables["bt_pings"].values()][0]
    assert ping["pongs_seen"] == 0                    # the applier ran before the subscriber
    bank = w.modules["bank"]
    assert bank.balance(w.state, bank.primary_account(w.state, "p:ann")) == 99
    assert bus.subscribers(w, "bustest.ping") == {"bustest.ping": ["bustest_dst"]}


def test_a_failing_subscriber_rolls_the_whole_command_back_across_modules():
    w = make()
    w.execute("p:ann", {"tool": "bsrc.ping", "args": {"n": 1}})
    td, arrays, seq = w.state.table_digests(), dict(w.state.meta), w.seq
    r = w.execute("p:ann", {"tool": "bsrc.ping", "args": {"n": 500, "pay": 10}})
    assert not r.ok and r.error["code"] == "invalid_action" and "downstream" in r.error["message"]
    assert w.state.table_digests() == td                 # no ping, no pong, no payment
    assert [e.type for e in w.log[seq - w.log_base:]] == ["cmd.rejected"]
    # a bug (non-Crucible exception) in a subscriber: rolled back; a strict world re-raises it
    with pytest.raises(ZeroDivisionError):
        w.execute("p:ann", {"tool": "bsrc.ping", "args": {"n": 99}})
    assert w.state.table_digests() == td and not w.state.in_tx
    lax = make(strict=False)
    td2 = lax.state.table_digests()
    r = lax.execute("p:ann", {"tool": "bsrc.ping", "args": {"n": 99}})
    assert not r.ok and "internal error" in r.error["message"] and lax.state.table_digests() == td2


def test_replay_does_not_call_subscribers_and_regeneration_reproduces_them():
    w = make()
    for n in (1, 2, 500, 3):
        w.execute("p:ann", {"tool": "bsrc.ping", "args": {"n": n}})
    evs = [e.to_dict() for e in w.log]
    calls = []
    dst = w.modules["bustest_dst"]
    orig = type(dst).on_ping
    type(dst).on_ping = lambda self, ctx, ev: (calls.append(ev.seq), orig(self, ctx, ev))[1]
    try:
        r = World.replay(w.spec, evs)
        assert r.digest() == w.digest() and calls == []
        g = World.regenerate(w.spec, evs)
        assert g.last_hash == w.last_hash and len(calls) == 4
    finally:
        type(dst).on_ping = orig


def test_subscription_depth_is_bounded():
    w = make({"recurse": True})
    r = w.execute("p:ann", {"tool": "bsrc.ping", "args": {"n": 1}})
    assert not r.ok and "depth limit" in r.error["message"]
    assert w.state.count("bt_pongs") == 0 and w.state.count("bt_pings") == 0


def test_strict_worlds_validate_payloads_plain_worlds_do_not():
    lax = make({"subscribe": False}, strict=False)
    assert bus.enforce(lax) == 0                                         # nothing to enforce in a plain world
    assert lax.execute("p:ann", {"tool": "bsrc.raw", "args": {"bad": True}}).ok   # not validated
    w = make({"subscribe": False})
    assert bus.enforce(w, owner="bustest_src") == 1
    with pytest.raises(ProtocolError):
        w.execute("p:ann", {"tool": "bsrc.raw", "args": {"bad": True}})
    assert w.execute("p:ann", {"tool": "bsrc.raw", "args": {"bad": False}}).ok
    with pytest.raises(ProtocolError):
        w.execute("p:ann", {"tool": "bsrc.ping", "args": {"n": -3}})   # schema minimum 0, checked by bus.emit
    with pytest.raises(ProtocolError):
        bus.subscribe(w, "ctr", lambda c, e: None, owner="bustest_dst")
    with pytest.raises(ProtocolError):
        bus.subscribe(w, "bustest.ping", lambda c, e: None, owner="not_a_module")


def test_no_subscriber_means_identical_logs():
    """A world whose modules register no subscriber produces exactly the same events as one where an unrelated
    subscriber is registered for a type that never occurs."""
    a, b = make({"subscribe": False}), make({"subscribe": False})
    bus.subscribe(b, "bustest.never", lambda c, e: None, owner="bustest_dst")
    for w in (a, b):
        for n in (1, 2, 3):
            w.execute("p:ann", {"tool": "bsrc.ping", "args": {"n": n}})
        w.advance_time(3600)
    assert a.last_hash == b.last_hash and a.digest() == b.digest()
