"""The population API the civilisation modules use (ONE_WORLD_CONTRACTS section 1), in one place.

The contract names (`crucible.population.register_activity`, `pay`, `record_decisions`, `n_individuals`,
`features`, `spendable`, `wake`) are exported by the package once workstream CORE-POP lands. Until then this
checkout has the same mechanisms under `crucible.population.core`; each function below calls the contract export
when the package has it and the core function otherwise, with the same arguments and meaning. Nothing here keeps
population state: every per-person fact stays in the population's own table, every payment is a bank move from the
individual's own account, every decision is a recorded batch.
"""
from __future__ import annotations

import numpy as np

from ..substrate import columns


def _pkg():
    import crucible.population as P
    return P


def _core():
    from ..population import core
    return core


def has(name: str) -> bool:
    return callable(getattr(_pkg(), name, None))


def register_activity(world, name: str, handler, *, owner: str, prior: float = 0.0, goals: tuple = ()) -> None:
    fn = getattr(_pkg(), "register_activity", None) or _core().register_activity
    fn(world, name, handler, owner=owner, prior=prior, goals=goals)


def n_individuals(world) -> int:
    fn = getattr(_pkg(), "n_individuals", None) or _core().n_individuals
    return int(fn(world))


def features(world, idx) -> dict:
    fn = getattr(_pkg(), "features", None) or _core().features
    return fn(world, np.asarray(idx, np.int64))


def spendable(world, idx) -> np.ndarray:
    fn = getattr(_pkg(), "spendable", None) or _core().spendable
    return np.asarray(fn(world, np.asarray(idx, np.int64)), np.int64)


def wake(ctx, idx, at: int, activity: str) -> None:
    fn = getattr(_pkg(), "wake", None) or _core().wake
    fn(ctx, idx, at, activity)


def account_of(ctx, i: int) -> str:
    """The individual's own bank account (materialised on first need, accrued income deposited)."""
    acct = _core().money_deposit(ctx, int(i))
    if acct is None:
        raise RuntimeError("the population pays only in worlds with a bank")
    return acct


def pay(ctx, i: int, to_account: str, cents: int, *, memo: str, kind: str = "purchase", ref: str | None = None) -> str:
    """Pay `cents` from individual `i`'s own account to `to_account` (contract `pay`)."""
    fn = getattr(_pkg(), "pay", None)
    if fn is not None:
        return fn(ctx, int(i), to_account, int(cents), memo=memo)
    acct = account_of(ctx, i)
    return ctx.world.modules["bank"].move(ctx, acct, to_account, int(cents), kind, memo, ref=ref)


def record_decisions(ctx, kind: str, idx, values: dict, *, table: str, model_id: str, model_version: str,
                     fidelity_id: str, mode: str = "set", extra: dict | None = None) -> str:
    """One recorded cohort batch carrying decision provenance (who decided, with which model)."""
    fn = getattr(_pkg(), "record_decisions", None)
    if fn is not None:
        return fn(ctx, kind, np.asarray(idx, np.int64), values, model_id=model_id, model_version=model_version,
                  fidelity_id=fidelity_id, mode=mode, table=table)
    ex = {"decision": {"model_id": model_id, "model_version": model_version, "fidelity_id": fidelity_id}}
    ex.update(extra or {})
    return columns.batch(ctx, table, kind, np.asarray(idx, np.int64), values, mode=mode, extra=ex)
