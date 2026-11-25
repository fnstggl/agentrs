"""Crucible 5.0 kernel extension points used by the substrate: byte-identical 4.0 behaviour, row journaling,
table-declaration tracking, replay with a blob store."""
import numpy as np
import pytest

from crucible.core.kernel import World
from crucible.core.state import StateStore
from v5_kernel_scenarios import KERNEL40_DIGESTS, SCENARIOS, scenario


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_kernel_is_byte_identical_to_crucible_40(name):
    """Same events, same hash chain, same state digest as the untouched 4.0 kernel (recorded at c07098b)."""
    assert scenario(name) == KERNEL40_DIGESTS[name]


def test_row_journal_rolls_back_exactly_through_savepoints_repeats_and_growth():
    st = StateStore()
    st.arrays["c"] = np.arange(10, dtype=np.float32)
    orig = st.arrays["c"].copy()
    st.begin()
    a = st.array_rows_mut("c", np.array([1, 1, 4]))
    a[[1, 4]] += 100
    mark = st.savepoint()
    b = st.array_rows_mut("c", np.array([4, 7]))
    b[[4, 7]] = -1
    st.set_array("c", np.concatenate([st.arrays["c"], np.zeros(3, np.float32)]))     # growth after row writes
    st.array_rows_mut("c", np.array([11]))[11] = 9
    st.rollback_to(mark)
    exp = orig.copy()
    exp[[1, 4]] += 100
    assert np.array_equal(st.arrays["c"], exp)
    full = st.array_mut("c")                          # a full copy journaled after row writes
    full[:] = 0
    st.rollback()
    assert np.array_equal(st.arrays["c"], orig)
    st.begin()
    st.array_rows_mut("c", np.array([2]))[2] = 55
    st.commit()
    assert st.arrays["c"][2] == 55


def test_table_declarations_are_attributed_to_modules_and_survive_snapshots():
    import crucible
    from crucible.core.kernel import WorldSpec
    crucible.load_all_modules()
    sp = WorldSpec("decl", 1, [{"name": "identity", "config": {}}, {"name": "bank", "config": {}},
                               {"name": "artifacts", "config": {}}])
    w = World.create(sp)
    d = w.state.declared_by
    assert d["principals"] == ["kernel"] and d["accounts"] == ["bank"] and d["persons"] == ["identity"]
    assert d["artifacts"] == ["artifacts"]
    f = World.from_blob(w.snapshot_blob())
    assert f.state.declared_by == d
    # runtime only: a bare state snapshot carries no attribution, so digests and snapshots are unchanged from 4.0
    assert StateStore.loads(w.state.dumps()).declared_by == {}
