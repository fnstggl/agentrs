"""Crucible 5.0 kernel repairs D-21 / audit K1-K3, measured on a real population world (sandboxed web sites, the
columnar population, the 4.0 internet):

* K2 - observers never perturb a trajectory. A same-transaction subscriber runs in a *side* context: its events carry
  `via`, its random draws come from its own stream, and main-line random draws are keyed by main-line positions. A
  passive observer (draws randomness, emits nothing), an emitting observer (its own events and id counters) and typed
  validation (`params.typed_events`) all leave the rest of the world's events exactly as they are.
* K3 - the same spec and seed give the same event hashes run to run (sandbox paths are not recorded).
* K1 - cross-process snapshot / resume / replay with the adaptive population (its weights are rebuilt in
  `on_load` from checkpoint blobs that live only in the store): tests/test_v5_substrate_resume.py.
"""
import numpy as np
import pytest

import crucible
from crucible.core.kernel import SIDE_KEY, Module, World, applier, register_module
from crucible.population.worlds import population_world
from crucible.substrate import bus, history

crucible.load_all_modules()
DAY = 86400


@register_module
class StreamObserver(Module):
    """Watches every cohort decision (`cols.batch`). Draws randomness for each one; with `emit`, it also records an
    event of its own with an id from its own counter, and (with `call`) makes a nested tool call."""
    name = "stream_observer"

    def install(self, world):
        world.state.declare_table("stream_seen")
        bus.subscribe(world, "cols.batch", self.on_batch, owner=self.name)

    def on_batch(self, ctx, ev):
        r = float(ctx.rng("observe", ev.payload["kind"]).random())
        if self.config.get("call"):
            ctx.call("world.clock", {})
        if self.config.get("emit"):
            n = ctx.state.next_counter("stream_observer")
            ctx.emit("stream_observer.seen", {"id": f"o{n}", "kind": ev.payload["kind"], "r": r}, vis="internal")

    @applier("stream_observer.seen")
    def ap_seen(self, ev):
        self.world.state.put("stream_seen", ev.payload["id"], {"kind": ev.payload["kind"], "r": ev.payload["r"]})


def _world(extra=None, params=None, days=1.0):
    sp = population_world("development", 5, n=240, sites=6, adaptive=False, strategic=False, params=params)
    if extra is not None:
        sp.modules.append(extra)
    w = World.create(sp, world_id="streams")
    w.advance_time(int(days * DAY))
    return w


@pytest.fixture(scope="module")
def base():
    return _world()


def _rest(w, skip_tables=()):
    """Everything of the world's state outside the observer: table digests and column arrays."""
    t = {k: v for k, v in w.state.table_digests().items() if k not in skip_tables}
    return t, {k: w.state.arrays[k].tobytes() for k in sorted(w.state.arrays)}


@pytest.mark.timeout(600)
def test_same_spec_and_seed_give_the_same_hashes_run_to_run(base):
    """K3: sandboxed site programs that fail record tracebacks; their host paths differ per process and per
    worker, and are no longer part of the recorded outcome."""
    again = _world()
    assert again.last_hash == base.last_hash and again.digest() == base.digest()
    assert any(e.type == "cloud.usage" and e.payload.get("outcome") == "error" for e in base.log), \
        "the world must exercise failing sandboxed requests for this test to mean anything"


@pytest.mark.timeout(600)
def test_a_passive_observer_leaves_the_event_hash_chain_of_the_world_unchanged(base):
    w = _world({"name": "stream_observer", "config": {"call": True}})
    assert w.seq == base.seq
    # the spec differs (one more module), so the genesis note differs; every event after it is byte-identical
    assert history.mainline_digest(w.log, after_genesis=True) == history.mainline_digest(base.log, after_genesis=True)
    assert [e.body() for e in w.log[1:]] == [e.body() for e in base.log[1:]]
    assert _rest(w, {"stream_seen"}) == _rest(base, {"stream_seen"})


@pytest.mark.timeout(600)
def test_an_emitting_observer_leaves_every_main_line_event_unchanged(base):
    w = _world({"name": "stream_observer", "config": {"emit": True, "call": True}})
    side = [e for e in w.log if e.via is not None]
    assert side and {e.via for e in side} <= {"stream_observer", "kernel"}
    assert sum(1 for e in side if e.type == "stream_observer.seen") == sum(1 for e in base.log
                                                                          if e.type == "cols.batch")
    assert w.state.meta[SIDE_KEY] == len(side) and w.main_seq == base.seq
    # the main line (events not emitted by the observer, at their main-line positions) is identical
    assert history.mainline_digest(w.log, after_genesis=True) == history.mainline_digest(base.log, after_genesis=True)
    # so is all state of the rest of the world, except where a 4.0 applier stores an event's absolute seq number
    # (search_docs.seq): a position, which side events shift, not a fact
    t1, a1 = _rest(w, {"stream_seen", "search_docs"})
    t0, a0 = _rest(base, {"stream_seen", "search_docs"})
    assert t1 == t0 and a1 == a0
    docs = lambda x: {k: {f: v for f, v in r.items() if f != "seq"} for k, r in x.state.tables["search_docs"].items()}
    assert docs(w) == docs(base)
    # the observer's own draws are its own stream: deterministic, and replay / regeneration are exact
    assert World.replay(w.spec, [e.to_dict() for e in w.log]).digest() == w.digest()
    reg = World.regenerate(w.spec, [e.to_dict() for e in w.log])
    assert reg.last_hash == w.last_hash and reg.digest() == w.digest()


@pytest.mark.timeout(600)
def test_typed_validation_does_not_alter_the_simulation(base):
    w = _world(params={"typed_events": True})
    n_enforced = sum(1 for subs in w.subscribers.values() for o, _ in subs if o == "kernel")
    assert n_enforced >= 10           # the validators really ran on this world's event types
    assert history.mainline_digest(w.log, after_genesis=True) == history.mainline_digest(base.log, after_genesis=True)
    assert _rest(w) == _rest(base)
    assert w.state.meta.get(SIDE_KEY, 0) == 0


def test_side_contexts_draw_from_their_own_stream_and_never_shift_the_emitter():
    """Unit level: the emitter's draws after a notification are the same whatever the subscriber drew."""
    from crucible.core import schema as S
    from crucible.core.kernel import WorldSpec, endpoint

    @register_module
    class StreamEmitter(Module):
        name = "stream_emitter"
        host = "se"

        def install(self, world):
            world.state.declare_table("se_log")
            if self.config.get("subscriber_draws"):
                bus.subscribe(world, "se.tick", self.sub, owner=self.name)

        def sub(self, ctx, ev):
            for _ in range(int(self.config["subscriber_draws"])):
                ctx.rng("x").random()

        @endpoint("go", S.obj())
        def go(self, ctx):
            a = float(ctx.rng("a").random())
            ctx.emit("se.tick", {"a": a}, vis="public")
            b = float(ctx.rng("b").random())
            ctx.emit("se.tock", {"b": b}, vis="public")
            return {"a": a, "b": b}

        @applier("se.tick")
        def ap_tick(self, ev):
            self.world.state.put("se_log", f"t{ev.seq}", dict(ev.payload))

        @applier("se.tock")
        def ap_tock(self, ev):
            self.world.state.put("se_log", f"k{ev.seq}", dict(ev.payload))

    outs = []
    for draws in (0, 1, 7):
        sp = WorldSpec("se", 3, [{"name": "identity", "config": {}}, {"name": "bank", "config": {}},
                                 {"name": "admin", "config": {}},
                                 {"name": "stream_emitter", "config": {"subscriber_draws": draws}}])
        w = World.create(sp)
        p = w.system_call("admin.join", {"handle": "se-user", "endowment": 0})["principal"]
        outs.append([w.execute(p, {"tool": "se.go", "args": {}}).result for _ in range(3)])
    assert outs[0] == outs[1] == outs[2]
    assert len({o["b"] for o in outs[0]}) == 3 and all(np.isfinite(o["a"]) for o in outs[0])
