"""Exact hybrid identity sets (crucible.social.idset): membership, conversion, journaling."""
import numpy as np
import pytest

from crucible.core.state import StateStore
from crucible.social import idset


def _add(st, name, m, n):
    st.begin()
    new = idset.add(st, name, np.unique(np.asarray(m, np.int64)), n)
    st.commit()
    return new


@pytest.mark.parametrize("n", [100, 10_000, 1_000_003])
def test_matches_python_sets_through_conversion(n):
    rng = np.random.default_rng(n)
    st = StateStore()
    ref = set()
    for step in range(40):
        m = rng.integers(0, n, size=int(rng.integers(1, max(2, n // 20))))
        u = np.unique(m)
        new = _add(st, "audset.vw.1", m, n)
        assert new.tolist() == [int(x) not in ref for x in u]
        ref |= set(int(x) for x in u)
        assert idset.size(st, "audset.vw.1") == len(ref)
        q = rng.integers(0, n, size=500)
        assert idset.contains(st, "audset.vw.1", q).tolist() == [int(x) in ref for x in q]
    assert idset.members(st, "audset.vw.1").tolist() == sorted(ref)
    # large sets end dense (a bitset smaller than the sorted array), small ones sparse
    dense = idset.is_dense(st.arrays["audset.vw.1"])
    assert dense == (4 * len(ref) > (n + 7) // 8)


def test_rollback_restores_exactly():
    st = StateStore()
    _add(st, "audset.ex.0", np.arange(0, 50, 2), 64)          # dense already (64 bits = 8 bytes)
    _add(st, "audset.ex.1", [5, 9], 1_000_000)                  # sparse
    before = {k: v.copy() for k, v in st.arrays.items()}
    d0 = st.digest()
    st.begin()
    idset.add(st, "audset.ex.0", np.array([1, 3, 63]), 64)
    idset.add(st, "audset.ex.1", np.arange(10, 40_000, dtype=np.int64), 1_000_000)   # converts to dense
    idset.add(st, "audset.ex.2", np.array([7]), 1_000_000)                            # new set
    st.rollback()
    assert st.digest() == d0
    assert set(st.arrays) == set(before)
    for k, v in before.items():
        assert np.array_equal(st.arrays[k], v) and st.arrays[k].dtype == v.dtype


def test_universe_growth_extends_bitset():
    st = StateStore()
    _add(st, "audset.vw.3", np.arange(0, 64), 64)
    assert idset.is_dense(st.arrays["audset.vw.3"]) and st.arrays["audset.vw.3"].size == 8
    new = _add(st, "audset.vw.3", [63, 64, 200], 256)
    assert new.tolist() == [False, True, True]
    assert idset.contains(st, "audset.vw.3", np.array([64, 200, 201, 10**6])).tolist() == [True, True, False, False]
    assert idset.size(st, "audset.vw.3") == 66
