"""crucible.substrate.integrity (SP-2.3b): attribution of every state write to the module that performs it, the
approved interfaces (column batches under the ownership rule, kernel counters), and world-kind hashes. Unit level,
on small kernel worlds; the gate (tests/v5/test_gate_SP_2_3b.py) measures the real world kinds."""
import numpy as np
import pytest

import crucible
from crucible.core import schema as S
from crucible.core.kernel import Module, World, WorldSpec, applier, endpoint, register_module
from crucible.substrate import columns as C
from crucible.substrate import integrity

crucible.load_all_modules()


@register_module
class ItOwner(Module):
    name = "it_owner"
    host = "itown"
    depends = ("cols",)

    def install(self, world):
        world.state.declare_table("it_owned")
        world.state.set_array("it_dense", np.zeros(8))
        C.declare(world, "itpop", 16, {"score": ("float32", (), 0.0)})

    @endpoint("add", S.obj())
    def ep_add(self, ctx):
        n = ctx.state.next_counter("it")                              # a counter: approved (ctr event at commit)
        ctx.emit("it.added", {"id": f"o{n}"}, vis="internal")
        C.batch(ctx, "itpop", "it.score", np.arange(4), {"score": np.ones(4, np.float32)}, mode="add")
        return {}

    @applier("it.added")
    def ap_added(self, ev):
        self.world.state.put("it_owned", ev.payload["id"], {"n": 1})
        self.world.state.array_rows_mut("it_dense", np.array([0]))[0] += 1


@register_module
class ItOther(Module):
    name = "it_other"
    host = "itoth"
    depends = ("it_owner",)

    def install(self, world):
        world.state.declare_table("it_mine")

    @endpoint("poke", S.obj({"how": S.s_enum(["table", "array", "meta", "handler", "column", "ok"])}, ["how"]))
    def ep_poke(self, ctx, how):
        if how == "handler":
            ctx.state.put("it_mine", "h", {"n": 1})                     # outside any applier
            return {}
        if how == "column":
            # a batch the column rule refuses is rejected live; the trace never sees it applied
            C.batch(ctx, "itpop", "it.steal", np.arange(2), {"score": np.ones(2, np.float32)})
            return {}
        ctx.emit("it.poked", {"how": how}, vis="internal")
        return {}

    @applier("it.poked")
    def ap_poked(self, ev):
        st = self.world.state
        how = ev.payload["how"]
        st.put("it_mine", f"p{ev.seq}", {"how": how})                  # its own table: fine
        if how == "table":
            st.put("it_owned", f"x{ev.seq}", {"n": 2})
        elif how == "array":
            st.array_mut("it_dense")[1] = 5.0
        elif how == "meta":
            st.meta_set("it_other_flag", 1)


def _world():
    w = World.create(WorldSpec("integrity", 1, [{"name": m, "config": {}} for m in
                                                 ("identity", "bank", "admin", "artifacts", "cols", "it_owner",
                                                  "it_other")]))
    p = w.system_call("admin.join", {"handle": "tracer", "endowment": 0})["principal"]
    return w, p


def test_owners_of_tables_arrays_columns_and_kernel_state():
    w, _ = _world()
    assert integrity.table_owners(w, "it_owned") == ("it_owner",)
    assert integrity.table_owners(w, "sched") == ("kernel",)
    assert integrity.array_owner(w, "it_dense") == ("it_owner", None, None)
    assert integrity.array_owner(w, "itpop.score") == ("it_owner", "itpop", "score")


def test_own_writes_counters_and_admitted_batches_are_not_cross_module():
    w, p = _world()
    with integrity.trace_writes(w) as tr:
        assert w.execute(p, {"tool": "itown.add", "args": {}}).ok
        assert w.execute(p, {"tool": "itoth.poke", "args": {"how": "ok"}}).ok
    r = tr.report()
    assert r["cross_module_state_writes"] == 0, r
    assert r["writes_traced"] >= 5 and r["events_applied"] >= 4
    # the column batch was attributed to its recorded writer through the columns module
    assert any(x["writer"].startswith("cols(by=it_owner)") for x in
               [{"writer": k[0]} for k in tr.writes]), sorted(tr.writes)


@pytest.mark.parametrize("how,kind", [("table", "cross"), ("array", "cross"), ("meta", "cross"),
                                      ("handler", "handler")])
def test_every_kind_of_foreign_write_is_found(how, kind):
    w, p = _world()
    with integrity.trace_writes(w) as tr:
        assert w.execute(p, {"tool": "itoth.poke", "args": {"how": how}}).ok
    r = tr.report()
    assert r["cross_module_state_writes"] == 1, r
    row = r[kind][0]
    assert row["writer"] == ("handler" if how == "handler" else "it_other")
    assert row["target"] == {"table": "it_owned", "array": "it_dense", "meta": "it_other_flag",
                             "handler": "it_mine"}[how]


def test_a_column_batch_is_attributed_to_its_recorded_writer_under_the_ownership_rule():
    """Both test modules live in one Python module, i.e. one package for the column rule (columns.may_write admits
    the owner's package siblings), so it_other's batch on it_owner's column is an approved interface write; the
    trace attributes it to the writer the batch recorded and checks the rule."""
    w, p = _world()
    with integrity.trace_writes(w) as tr:
        assert w.execute(p, {"tool": "itoth.poke", "args": {"how": "column"}}).ok
    r = tr.report()
    assert r["cross_module_state_writes"] == 0 and r["column_writes_refused_by_rule"] == 0
    assert ("cols(by=it_other)", "column", "itpop.score", "it_owner") in tr.writes
    # the rule itself refuses a module of another package (crucible.substrate.columns tests cover the live refusal)
    assert C.may_write(w, "itpop", "score", "bank") is not None


def test_tracing_changes_nothing_and_is_removed_afterwards():
    a, pa = _world()
    b, pb = _world()
    with integrity.trace_writes(a):
        a.execute(pa, {"tool": "itown.add", "args": {}})
        a.execute(pa, {"tool": "itoth.poke", "args": {"how": "table"}})
    b.execute(pb, {"tool": "itown.add", "args": {}})
    b.execute(pb, {"tool": "itoth.poke", "args": {"how": "table"}})
    assert a.last_hash == b.last_hash and a.digest() == b.digest()
    assert "put" not in a.state.__dict__ and all(not hasattr(f, "__wrapped__") for f in a.appliers.values())
    assert World.replay(a.spec, [e.to_dict() for e in a.log]).digest() == a.digest()


def test_hashes_of_a_rapid_environment_are_reproducible():
    x, y = integrity.hashes("rapid", 0, 24), integrity.hashes("rapid", 0, 24)
    assert x["last_hash"] == y["last_hash"] and x["events"] == 24
    assert integrity.hashes("rapid", 1, 24)["last_hash"] != x["last_hash"]
