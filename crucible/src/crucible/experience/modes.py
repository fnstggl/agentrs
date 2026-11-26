"""Reward modes (MASTER_SPEC 11.3).

    legacy_shaped_reward        the 4.0 reward contracts, unchanged (default everywhere, byte-identical)
    verified_terminal_outcome   no intermediate reward; 1.0 once when the independent verifier confirms the
                                public objective, else 0.0; outcome labels are only success/failed
    observation_only            no reward at all (0.0 always, documented as not a learning signal), no goal
                                progress, no objective satisfaction, no evaluator data in anything the learner sees;
                                the verifier keeps running on the evaluator side (`verify()`)

Observable world consequences (balances, revenue, costs, readings, analytics a participant could see) stay in the
observations in every mode: they are what the world shows, not what the evaluator thinks.

`RewardGate` is the one place an adapter turns the legacy reward stream and the verifier's terminal outcome into
what the learner receives in a given mode.
"""
from __future__ import annotations

LEGACY = "legacy_shaped_reward"
VERIFIED = "verified_terminal_outcome"
OBSERVATION_ONLY = "observation_only"
REWARD_MODES = (LEGACY, VERIFIED, OBSERVATION_ONLY)

# Keys of the evaluator's goal status (goals.contracts.GoalVerifier.status) and of operator-side task and episode
# records. None of them may reach a learner outside legacy mode; in verified mode only `success` and `failed` of a
# terminal outcome are authorised labels.
VERIFIER_STATUS_KEYS = ("success", "failed", "progress", "streak", "elapsed_s", "success_t", "initially_true")
TERMINAL_LABEL_KEYS = ("success", "failed")
PRIVATE_TASK_KEYS = ("default_net", "reference_net", "random_net", "solvable", "rejected", "mech_seed")
OPERATOR_RECORD_KEYS = ("score", "success", "random_score", "score_first_half", "score_second_half", "private",
                        "objective_quantity")


def check_mode(mode: str | None) -> str:
    m = LEGACY if mode is None else str(mode)
    if m not in REWARD_MODES:
        raise ValueError(f"reward_mode must be one of {REWARD_MODES}, got {mode!r}")
    return m


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

    def __init__(self, mode: str):
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
