"""Reward modes (MASTER_SPEC 11.3, Clarification 2, decision D-19).

    observation_only            THE DEFAULT, everywhere (`check_mode(None)`). No reward at all (0.0 always,
                                documented as not a learning signal), no goal progress, no objective satisfaction, no
                                action-quality judgement, no evaluator data in anything the learner sees: the world is
                                built with ``evaluator_access="evaluator"`` (`crucible.core.evaluator_only`), so
                                evaluator tools are refused and evaluator fields, event metadata and metrics are kept
                                from participants. The verifier keeps running on the evaluator side (`verify()`).
    verified_terminal_outcome   explicitly selected: no intermediate reward; 1.0 once when the independent verifier
                                confirms the public objective at the terminal outcome, else 0.0; outcome labels are
                                only success/failed. Evaluator data are hidden as in observation_only.
    legacy_shaped_reward        explicitly selected: the 4.0 reward contracts and the 4.0 evaluator-access contract,
                                unchanged and byte-identical (historical baselines, OO.8)

Observable world consequences (balances, revenue, costs, readings, analytics a participant could see) stay in the
observations in every mode: they are what the world shows, not what the evaluator thinks.

`RewardGate` is the one place an adapter turns the legacy reward stream and the verifier's terminal outcome into
what the learner receives in a given mode. `prepare_spec` / `prepare_world` give a world the evaluator-access policy
of a mode; every learner-facing constructor calls one of them.
"""
from __future__ import annotations

import os

LEGACY = "legacy_shaped_reward"
VERIFIED = "verified_terminal_outcome"
OBSERVATION_ONLY = "observation_only"
REWARD_MODES = (LEGACY, VERIFIED, OBSERVATION_ONLY)
DEFAULT_MODE = OBSERVATION_ONLY

# Keys of the evaluator's goal status (goals.contracts.GoalVerifier.status) and of operator-side task and episode
# records. None of them may reach a learner outside legacy mode; in verified mode only `success` and `failed` of a
# terminal outcome are authorised labels.
VERIFIER_STATUS_KEYS = ("success", "failed", "progress", "streak", "elapsed_s", "success_t", "initially_true")
TERMINAL_LABEL_KEYS = ("success", "failed")
PRIVATE_TASK_KEYS = ("default_net", "reference_net", "random_net", "solvable", "rejected", "mech_seed")
OPERATOR_RECORD_KEYS = ("score", "success", "random_score", "score_first_half", "score_second_half", "private",
                        "objective_quantity")


# --- historical baseline selection (workstream P, DECISION_LOG D-82; minimal edit in workstream L's module) -----------
# A process started to reproduce Crucible 4.0 - the unmodified 4.0 regression suite of Gate A.1, the 4.0 records of
# Gate A.5 - selects the 4.0 reward contract for every constructor that names no mode by exporting
# CRUCIBLE_HISTORICAL_BASELINE=4.0 (inherited by its worker processes). That is an explicit selection of the historical
# baseline (Clarification 2, OO.8), made by the experimenter for a whole process tree; nothing else ever sets it, the
# official acceptance runner removes it from every measured process, and evidence recorded under it is rejected.
HISTORICAL_BASELINE_ENV = "CRUCIBLE_HISTORICAL_BASELINE"
HISTORICAL_BASELINES = {"4.0": LEGACY}


def historical_baseline() -> str | None:
    """The historical baseline this process was explicitly started to reproduce ("4.0"), or None. Any other value of
    the variable is an error, never silently ignored."""
    v = os.environ.get(HISTORICAL_BASELINE_ENV)
    if not v:
        return None
    if v not in HISTORICAL_BASELINES:
        raise ValueError(f"{HISTORICAL_BASELINE_ENV} must be one of {sorted(HISTORICAL_BASELINES)}, got {v!r}")
    return v


def check_mode(mode: str | None) -> str:
    """The reward mode, validated. ``None`` (nothing selected) is observation-only: legacy and verified modes exist
    only when a caller names them, or when the process was explicitly started as a historical baseline
    (`historical_baseline`)."""
    hb = historical_baseline() if mode is None else None
    m = (HISTORICAL_BASELINES[hb] if hb else DEFAULT_MODE) if mode is None else str(mode)
    if m not in REWARD_MODES:
        raise ValueError(f"reward_mode must be one of {REWARD_MODES}, got {mode!r}")
    return m


def is_explicit_legacy(mode: str | None) -> bool:
    return (mode is not None and str(mode) == LEGACY) or (mode is None and historical_baseline() == "4.0")


def evaluator_access(mode: str | None) -> str:
    """The world evaluator-access policy a reward mode implies: the 4.0 contract only for legacy."""
    from ..core import evaluator_only as EO
    return EO.PARTICIPANTS if check_mode(mode) == LEGACY else EO.EVALUATOR


def prepare_spec(spec, mode: str | None):
    """Give a `WorldSpec` (before `World.create`) the evaluator-access policy of `mode`. Legacy leaves a spec
    without the parameter untouched (byte-identical 4.0 worlds); an explicit parameter is respected. Returns the
    spec."""
    from ..core import evaluator_only as EO
    from ..core.kernel import WorldSpec
    if EO.PARAM in spec.params or check_mode(mode) == LEGACY:
        return spec
    return WorldSpec(spec.family, spec.seed, spec.modules, dict(spec.params, **{EO.PARAM: EO.EVALUATOR}), spec.kernel)


def prepare_world(world, mode: str | None):
    """Give an existing `World` the policy of `mode` (a runtime override when its spec keeps the 4.0 contract and
    the mode is not legacy; see `core.evaluator_only.set_policy`). Returns the world."""
    from ..core import evaluator_only as EO
    if check_mode(mode) != LEGACY and not EO.hidden(world):
        EO.set_policy(world, EO.EVALUATOR)
    return world


def terminal_labels(status: dict | None) -> dict:
    """The authorised outcome labels of a verifier status in verified mode: success and failed only."""
    s = status or {}
    return {k: bool(s.get(k, False)) for k in TERMINAL_LABEL_KEYS}


def info_objective(mode: str, status: dict | None) -> dict | None:
    """What an adapter may put under ``info["objective"]`` in a mode: the full status (legacy), the terminal labels
    (verified), or nothing (observation-only: the key is left out)."""
    if mode == LEGACY:
        return dict(status or {})
    if mode == VERIFIED:
        return terminal_labels(status)
    return None


class RewardGate:
    """Per-agent reward bookkeeping for one adapter.

    `add_legacy(agent, r)` records the legacy (shaped) reward the 4.0 contract produced; `verified(agent, ok)`
    records the verifier's terminal outcome. `take(agent)` returns what the learner receives since the last call:
    the legacy sum (legacy), 1.0 once on the first verified success (verified) or 0.0 (observation-only)."""

    def __init__(self, mode: str | None = None):
        self.mode = check_mode(mode)
        self._legacy: dict[str, float] = {}
        self._terminal: dict[str, float] = {}
        self._paid: set[str] = set()
        self.paid_total: dict[str, float] = {}

    def add_legacy(self, agent: str, r: float) -> None:
        self._legacy[agent] = self._legacy.get(agent, 0.0) + float(r)

    def verified(self, agent: str, success: bool | None) -> None:
        if success and agent not in self._paid:
            self._paid.add(agent)
            self._terminal[agent] = self._terminal.get(agent, 0.0) + 1.0

    def take(self, agent: str) -> float:
        if self.mode == OBSERVATION_ONLY:
            self._legacy.pop(agent, None)
            self._terminal.pop(agent, None)
            return 0.0
        if self.mode == VERIFIED:
            self._legacy.pop(agent, None)
            r = self._terminal.pop(agent, 0.0)
        else:
            self._terminal.pop(agent, None)
            r = self._legacy.pop(agent, 0.0)
        self.paid_total[agent] = self.paid_total.get(agent, 0.0) + r
        return float(r)
