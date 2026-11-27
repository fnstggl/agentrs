"""crucible.substrate.integrity (SP-2.3b): attribution of every state write to the module that performs it, the
approved interfaces (column batches under the ownership rule, kernel counters), and world-kind hashes. Unit level,
on small kernel worlds; the gate (tests/v5/test_gate_SP_2_3b.py) measures the real world kinds."""
import numpy as np
import pytest

import crucible
from crucible.core import schema as S
from crucible.core.kernel import Module, World, WorldSpec, applier, derives, endpoint, register_module
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

    @endpoint("poke", S.obj({"how": S.s_enum(["table", "array", "meta", "handler", "column", "ok", "nested"])},
                            ["how"]))
    def ep_poke(self, ctx, how):
        if how == "nested":
            # a write that bypasses the state store's methods: a nested value of a row a read returned
            rows = [r for _, r in ctx.state.scan("it_mine") if isinstance(r.get("tags"), list)]
            if rows:
                rows[0]["tags"].append("silent")
            return {}
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
        st.put("it_mine", f"p{ev.seq}", {"how": how, "tags": []})      # its own table: fine
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


def test_a_write_that_bypasses_the_state_store_is_caught_by_replay():
    """Wave 3b: mutating a nested value of a row in a handler leaves no trace in the store's write methods (and no
    journal entry: a failing command would not even roll it back), and replay from the log cannot reproduce it.
    Since the wave-3b repair the content fingerprints of the trace find it too (an untracked handler write)."""
    w, p = _world()
    with integrity.trace_writes(w) as tr:
        assert w.execute(p, {"tool": "itoth.poke", "args": {"how": "ok"}}).ok
    assert tr.report()["cross_module_state_writes"] == 0 and integrity.replay_check(w)["replay_exact"] is True
    with integrity.trace_writes(w) as tr:
        assert w.execute(p, {"tool": "itoth.poke", "args": {"how": "nested"}}).ok
    r = tr.report()
    assert r["cross_module_state_writes"] == 1 and r["untracked_writes"] == 1, r
    assert [(x["writer"], x["target"]) for x in r["untracked"]] == [("handler", "it_mine")]
    assert integrity.replay_check(w)["replay_exact"] is False       # replay cannot reproduce it either


# ---------------------------------------------------------------------------------------------- wave-3b repair
@register_module
class ItSneak(Module):
    """Writes other modules' state without the store's methods inside its own appliers (replay re-applies these
    writes, so only the trace's content fingerprints can find them), and writes at genesis."""
    name = "it_sneak"
    host = "itsn"
    depends = ("it_owner",)

    def install(self, world):
        world.state.declare_table("it_sneak_own")
        world.state.declare_meta("sneak_counter")

    def genesis(self, ctx):
        ctx.emit("itsn.gen", {}, vis="internal")

    @endpoint("go", S.obj({"how": S.s_enum(["nested", "cached", "own_meta", "ns_meta", "foreign_meta", "fail",
                                            "ok"])}, ["how"]))
    def ep_go(self, ctx, how):
        ctx.emit("itsn.went", {"how": how}, vis="internal")
        if how == "fail":
            from crucible.core.errors import InvalidAction
            raise InvalidAction("rolled back after the applier wrote")
        return {}

    @applier("itsn.gen")
    def ap_gen(self, ev):
        self.world.state.put("it_owned", "from_genesis", {"n": 9, "tags": []})     # another module's table

    @applier("itsn.went")
    def ap_went(self, ev):
        st = self.world.state
        how = ev.payload["how"]
        st.put("it_sneak_own", f"s{ev.seq}", {"how": how})
        if how == "nested":
            row = st.get("it_owned", "from_genesis")
            row["tags"].append(f"x{ev.seq}")                    # in place: no put
        elif how == "cached":
            ref = self.world.runtime.get("it_sneak.dense")
            if ref is None:
                ref = self.world.runtime["it_sneak.dense"] = st.arrays["it_dense"]
            ref[3] += 1.0                                       # a cached reference: no array_mut
        elif how == "own_meta":
            st.meta_set("sneak_counter", 1)                     # declared by this module
        elif how == "ns_meta":
            st.meta_set("it_sneak.flag", 1)                     # namespaced by this module's name
        elif how == "foreign_meta":
            st.meta_set("itown.flag", 1)                        # namespaced by it_owner's host
        elif how == "fail":
            st.put("it_owned", "fail_row", {"n": 1})            # cross-module, then rolled back
            st.array_mut("it_dense")[5] = 2.0


@register_module
class ItDeriver(Module):
    """Keeps a derived count of it_owner's `it.added` events in its own table (kernel.derives)."""
    name = "it_deriver"
    depends = ("it_owner",)

    def install(self, world):
        world.state.declare_table("it_derived")

    @derives("it.added")
    def dv_added(self, ev):
        st = self.world.state
        r = st.get("it_derived", "n")
        st.put("it_derived", "n", {"n": (r["n"] if r else 0) + 1})


def _sneak_world(*extra):
    return World.create(WorldSpec("integrity-sneak", 1, [{"name": m, "config": {}} for m in
                                                         ("identity", "bank", "admin", "artifacts", "cols",
                                                          "it_owner", "it_sneak", *extra)]), world_id="sneak")


def _go(w, how):
    p = w.system_call("admin.join", {"handle": f"sn-{how.replace('_', '-')}", "endowment": 0})["principal"]
    with integrity.trace_writes(w) as tr:
        r = w.execute(p, {"tool": "itsn.go", "args": {"how": how}})
    return r, tr.report()


@pytest.mark.parametrize("how,kind,target", [("nested", "table", "it_owned"), ("cached", "array", "it_dense")])
def test_an_applier_writing_another_modules_state_without_the_store_is_found(how, kind, target):
    """Verifier probe p02 (a)/(b): an applier mutating a nested value of another module's row in place, or writing
    another module's array through a cached reference, was invisible to the trace and replays exactly (replay
    re-applies it). The content fingerprints find it and attribute it to the applier's module."""
    w = _sneak_world()
    r, rep = _go(w, how)
    assert r.ok and rep["cross_module_state_writes"] == 1 and rep["untracked_writes"] == 1, rep
    assert [(x["writer"], x["kind"], x["target"], x["owner"]) for x in rep["untracked"]] == \
        [("it_sneak", kind, target, "it_owner")]
    assert integrity.replay_check(w)["replay_exact"] is True        # replay alone cannot see it
    assert _go(w, "ok")[1]["cross_module_state_writes"] == 0        # and the trace is clean otherwise


def test_a_rolled_back_transaction_leaves_no_untracked_write():
    """The journal restores rows and arrays without the store's methods: the fingerprints follow it."""
    w = _sneak_world()
    r, rep = _go(w, "fail")
    assert not r.ok and rep["untracked_writes"] == 0 and rep["cross_module_write_count"] == 2, rep   # put + array_mut
    assert not w.state.has("it_owned", "fail_row") and w.state.array("it_dense")[5] == 0.0


def test_genesis_writes_are_traced_inside_trace_genesis():
    """Verifier probe p02 (c)/p03: the gate traced only after build, so cross-module writes of genesis (on every
    civilisation: bank -> ops_days, module metadata keys) were never measured."""
    with integrity.trace_genesis() as gen:
        w = _sneak_world()
    [(wid, tr)] = gen
    rep = tr.report()
    assert wid == "sneak" and rep["cross_module_state_writes"] == 1, rep
    assert [(x["writer"], x["target"]) for x in rep["cross"]] == [("it_sneak", "it_owned")]
    assert World._run_system.__name__ == "_run_system"               # the patch is gone
    assert integrity.merged_report(w, tr)["cross_module_state_writes"] == 1


@pytest.mark.parametrize("how,cross", [("own_meta", 0), ("ns_meta", 0), ("foreign_meta", 1)])
def test_metadata_keys_belong_to_their_declaring_or_namespacing_module(how, cross):
    w = _sneak_world()
    r, rep = _go(w, how)
    assert r.ok and rep["cross_module_state_writes"] == cross, rep
    assert integrity.meta_owner(w, "sneak_counter") == "it_sneak"
    assert integrity.meta_owner(w, "it_sneak.flag") == "it_sneak"
    assert integrity.meta_owner(w, "itown.flag") == "it_owner"      # by host
    assert integrity.meta_owner(w, "clock") == "kernel" and integrity.meta_owner(w, "unknown_key") == "kernel"
    if cross:
        assert [(x["writer"], x["target"], x["owner"]) for x in rep["cross"]] == [("it_sneak", "itown.flag",
                                                                                   "it_owner")]


def test_legacy_metadata_owners_name_shipped_modules():
    from crucible.core.kernel import MODULES
    assert all(m in MODULES for m in integrity.LEGACY_META_OWNERS.values()), integrity.LEGACY_META_OWNERS


def test_a_derived_state_applier_writes_its_own_table_in_the_event_transaction_and_in_replay():
    """kernel.derives (wave-3b repair): ops keeps its journal of bank.txn this way instead of a hook bank's applier
    called (on every civilisation the trace counted bank -> ops_days)."""
    w = _sneak_world("it_deriver")
    p = w.system_call("admin.join", {"handle": "dv", "endowment": 0})["principal"]
    with integrity.trace_writes(w) as tr:
        for _ in range(3):
            assert w.execute(p, {"tool": "itown.add", "args": {}}).ok
    rep = tr.report()
    assert rep["cross_module_state_writes"] == 0, rep
    assert ("it_deriver", "table", "it_derived", "it_deriver") in tr.writes
    assert w.state.get("it_derived", "n")["n"] == 3
    rep_w = World.replay(w.spec, [e.to_dict() for e in w.log])
    assert rep_w.digest() == w.digest() and rep_w.state.get("it_derived", "n")["n"] == 3


def test_the_ops_journal_is_a_derived_state_applier_of_bank_transactions():
    from crucible.economy.operations import Ops
    w = World.create(WorldSpec("ops-derived", 1, [{"name": m, "config": {}} for m in
                                                   ("identity", "bank", "admin", "ops")]))
    assert [getattr(f, "__func__", None) for f in w.derivers["bank.txn"]] == [Ops._on_txn]
    assert "txn_observers" not in w.hooks
