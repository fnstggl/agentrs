"""The population package API (ONE_WORLD_CONTRACTS section 1): exports, interests in the shared space, decision
provenance, payments and learning from experience."""
from __future__ import annotations

import numpy as np
import pytest

import crucible
from crucible.core.kernel import Module, World, register_module
from crucible import population as P
from crucible.population import semantics
from crucible.population.worlds import population_world

DAY = 86400


@register_module
class _ProbeActivity(Module):
    """Test-only activity: every activated individual takes one recorded decision through the API."""
    name = "pop_api_probe"
    host = "pop_api_probe"
    version = "1"
    depends = ("pop",)

    def install(self, world):
        P.register_activity(world, "probe", self.act, owner="pop_api_probe", prior=3.0)

    def act(self, ctx, idx, now):
        P.record_decisions(ctx, "test.probe", idx, {"last_utility": np.full(idx.size, 0.25, np.float32)},
                           model_id="probe-model", model_version="1", fidelity_id="test-probe")
        return np.full(idx.size, 0.1)


@pytest.fixture(scope="module")
def world():
    crucible.load_all_modules()
    spec = population_world("development", 3, n=60, sites=3, adaptive=False, strategic=False, tick_s=1800)
    spec.modules.append({"name": "pop_api_probe", "config": {}})
    w = World.create(spec)
    w.advance_time(w.now + DAY)
    return w


def test_exports_match_the_contract():
    for name in ("register_activity", "n_individuals", "features", "interests", "spendable", "pay", "wake",
                 "class_members", "record_decisions", "experience"):
        assert callable(getattr(P, name)), name
    for name in ("DIM", "SPACE_ID", "SPACE_VERSION", "item_vector", "item_vectors", "text_vector", "item_features",
                 "interest_prior"):
        assert hasattr(semantics, name), name


def test_interests_live_in_the_shared_space(world):
    n = P.n_individuals(world)
    u = P.interests(world, np.arange(n))
    assert u.shape == (n, semantics.DIM) and u.dtype == np.float32
    assert np.allclose(np.linalg.norm(u, axis=1), 1.0, atol=1e-3)
    # an affinity on projected item coordinates equals the full-space cosine
    e = semantics.text_vectors(world, ["The quarterback threw three touchdowns in the fourth quarter.",
                                       "Bake the bread for forty minutes until golden brown."])
    full = u @ e.T
    proj = P.interest_coords(world, np.arange(n)) @ P.to_coords(world, e).T
    assert np.allclose(full, proj, atol=5e-3)
    # heterogeneous: individuals do not share one interest
    assert float(np.std(u @ e[0])) > 0.02
    # and the prior comes from a real corpus: interests are nearer the corpus topics than random directions
    rnd = np.random.default_rng(0).standard_normal((n, semantics.DIM)).astype(np.float32)
    rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
    C = semantics.asset()["centroids"]
    assert float((u @ C.T).max(axis=1).mean()) > float((rnd @ C.T).max(axis=1).mean()) + 0.2


def test_decision_provenance_separates_population_decisions_from_injection(world):
    rep = P.check_provenance(world, prefixes=("test.",))
    assert rep["batches"] > 0 and rep["forced_rows"] == 0
    assert rep["organic_population_rows"] == rep["rows"]
    assert "probe-model@1" in world.state.tables["pop_decision_models"]
    w = world.fork("api-inject")
    # module code (a system transaction, not the population's activation) records a decision with valid provenance
    w._run_system("inject", lambda ctx: P.record_decisions(ctx, "test.probe", np.arange(10),
                                                           {"last_utility": np.ones(10, np.float32)},
                                                           model_id="probe-model", model_version="1",
                                                           fidelity_id="test-probe"))
    hist = lambda: list(world.log) + list(w.log)  # noqa: E731  (a fork's log starts at the fork)
    rep2 = P.check_provenance(w, hist(), prefixes=("test.",))
    assert rep2["forced_rows"] == 10 and rep2["organic_population_rows"] == rep["organic_population_rows"]
    # a batch without provenance (direct columns.batch) is forced too
    from crucible.substrate import columns
    w._run_system("inject2", lambda ctx: columns.batch(ctx, "pop", "test.raw", np.arange(5),
                                                       {"last_utility": np.ones(5, np.float32)}))
    rep3 = P.check_provenance(w, hist(), prefixes=("test.",))
    assert rep3["missing_provenance_batches"] == 1 and rep3["forced_rows"] == 15


def test_pay_moves_money_from_the_individuals_own_account(world):
    w = world.fork("api-pay")
    bank = w.modules["bank"]
    out = {}

    def run(ctx):
        before = int(P.spendable(w, np.array([7]))[0])
        out["txn"] = P.pay(ctx, 7, "acct_external", 1234, memo="test purchase")
        out["before"] = before
    w._run_system("pay", run)
    acct = bank.primary_account(w.state, "p7")
    assert acct is not None and out["txn"]
    assert int(P.spendable(w, np.array([7]))[0]) == out["before"] - 1234
    with pytest.raises(Exception):
        w._run_system("pay2", lambda ctx: P.pay(ctx, 7, "acct_external", 10 ** 12, memo="too much"))


def test_experience_moves_interests_towards_enjoyed_content(world):
    w = world.fork("api-exp")
    idx = np.arange(20)
    e = semantics.text_vectors(w, ["The quarterback threw three touchdowns in the fourth quarter."] * 20)
    before = P.interests(w, idx) @ e[0]
    for _ in range(5):
        w._run_system("exp", lambda ctx: P.experience(ctx, idx, "probe", np.full(20, 0.8), item_vecs=e))
    after = P.interests(w, idx) @ e[0]
    assert np.all(after > before) and float((after - before).mean()) > 0.02
