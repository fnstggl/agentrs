"""The population contract API as the economy uses it (ONE_WORLD_CONTRACTS section 1, D-15).

Workstream CORE-POP owns `crucible.population` and is extending its package exports to the contract surface
(`register_activity`, `n_individuals`, `features`, `interests`, `spendable`, `pay`, `wake`, `class_members`,
`record_decisions`, `experience`). The economy imports those names only through this module: each function uses the
population package's export when it exists and otherwise the equivalent call on the current population core
(`crucible.population.core`), so the economy runs on the one population today and switches to the final exports
without a code change. Nothing here keeps population state of its own.

Fallback semantics (used only while the export is missing):
* `record_decisions` writes one `columns.batch` on the caller's table with the decision provenance (`model_id`,
  `model_version`, `fidelity_id`) in the batch's `extra` - the same recorded form the contract specifies;
* `experience` is a no-op: the population core already updates satisfaction and habits from the utility an
  activity handler returns;
* `pay` materialises the individual's principal and account, deposits its accrued income and moves the amount from
  its own account (ledger kind `purchase`);
* `fund` (economy helper, not a contract name) materialises the principal with an account and deposits accrued
  income, so that tools the individual calls itself (subscribe, buy) pay from its own money.
"""
from __future__ import annotations

import numpy as np

from ..substrate import columns, ids


def _pop():
    import crucible.population as P
    return P


def _core():
    from ..population import core
    return core


def _export(name):
    return getattr(_pop(), name, None)


def register_activity(world, name, handler, *, owner, prior=0.0, goals=()):
    fn = _export("register_activity") or _core().register_activity
    return fn(world, name, handler, owner=owner, prior=prior, goals=goals)


def n_individuals(world) -> int:
    fn = _export("n_individuals") or _core().n_individuals
    return int(fn(world))


def features(world, idx) -> dict:
    fn = _export("features") or _core().features
    return fn(world, np.asarray(idx, np.int64))


def interests(world, idx) -> np.ndarray:
    fn = _export("interests")
    if fn is not None:
        return fn(world, np.asarray(idx, np.int64))
    v = _core().features(world, np.asarray(idx, np.int64))["interests"].astype(np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-6)


def spendable(world, idx) -> np.ndarray:
    fn = _export("spendable") or _core().spendable
    return fn(world, np.asarray(idx, np.int64))


def wake(ctx, idx, at, activity) -> None:
    fn = _export("wake") or _core().wake
    fn(ctx, np.asarray(idx, np.int64), int(at), activity)


def class_members(world, cls) -> np.ndarray:
    fn = _export("class_members") or _core().class_members
    return fn(world, cls)


def record_decisions(ctx, kind, idx, values, *, model_id, model_version, fidelity_id, mode="set", table=None,
                     extra=None) -> str:
    """`extra`: further batch extras (e.g. `experiment` labels of a randomised design, calibration/RUNTIME.md)."""
    fn = _export("record_decisions")
    if fn is not None:
        kw = {"extra": extra} if extra else {}
        return fn(ctx, kind, np.asarray(idx, np.int64), values, model_id=model_id, model_version=model_version,
                  fidelity_id=fidelity_id, mode=mode, table=table, **kw)
    return columns.batch(ctx, table, kind, np.asarray(idx, np.int64), values, mode=mode,
                         extra={**(extra or {}), "decision": {"model_id": model_id, "model_version": str(model_version),
                                                              "fidelity_id": fidelity_id}})


def experience(ctx, idx, activity, utility, *, item_vecs=None, satisfaction=None) -> None:
    fn = _export("experience")
    if fn is not None:
        fn(ctx, np.asarray(idx, np.int64), activity, utility, item_vecs=item_vecs, satisfaction=satisfaction)


def fund(ctx, i: int) -> str:
    """Principal "p<i>" with a bank account holding the individual's deposited income; returns the principal."""
    core = _core()
    p = ids.ensure_principal(ctx, int(i), account=True)
    if int(core.accrued(ctx.world, np.array([int(i)]))[0]) > 0 or \
            int(columns.col(ctx.world, "pop", "wallet_cents")[int(i)]) > 0:
        core.money_deposit(ctx, int(i))
    return p


def pay(ctx, i: int, to_account: str, cents: int, *, memo: str) -> str:
    fn = _export("pay")
    if fn is not None:
        return fn(ctx, int(i), to_account, int(cents), memo=memo)
    p = fund(ctx, i)
    bank = ctx.world.modules["bank"]
    return bank.move(ctx, bank.primary_account(ctx.state, p), to_account, int(cents), "purchase", memo)
