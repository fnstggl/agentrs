"""Families on the `state_machine` engine: hidden regime switches, fault diagnosis, and a staged process with
irreversible traps."""
from __future__ import annotations

import numpy as np

from ..mechanisms.base import neutral_names
from .families import family

C = "crucible.rapid.claims:"
I = "crucible.rapid.identify:"


def u(rng, lo, hi):
    return float(rng.uniform(lo, hi))


@family("posm_sign_switch", engine="state_machine", category="8.10", item="8.10.4", split="train",
        doc="Demand switches abruptly between hidden regimes (a sticky Markov chain); in some regimes the lever "
            "raises sales, in others it lowers them.",
        distinct_because="a hidden Markov regime process; the lever's effect sign is a function of the hidden "
                         "state, which has to be tracked from noisy readings",
        axes={"hidden_state_complexity": 2, "horizon": 80, "observation_noise": 2.0}, objective="maximize_net",
        claims=(C + "regime_sign",), identify=I + "lever_sign", lever_view=True)
def posm_sign_switch(rng, d):
    S = int(d["hidden_state_complexity"])
    names = neutral_names(rng, 2)
    stay = u(rng, 0.85, 0.92)
    T = np.full((S, S), (1 - stay) / max(1, S - 1))
    np.fill_diagonal(T, stay)
    slope = [u(rng, 1.5, 3.0) * (1 if k % 2 == 0 else -1) for k in range(S)]
    base = [u(rng, 20.0, 30.0) + (12.0 if slope[k] < 0 else 0.0) for k in range(S)]
    return {"engine": "state_machine", "scenario": "sign_switch", "horizon": int(d["horizon"]), "T": T.tolist(),
            "p0": [1.0 / S] * S, "slope": slope, "base": base, "noise": float(d["observation_noise"]),
            "price": 100.0, "lever_cost": u(rng, 20.0, 40.0), "lever_name": names[0], "lever_default": 5.0,
            "gauge_names": names[1:], "endowment": 300_000,
            "objective": {"type": "maximize_net", "text": "Maximise your net income over the horizon."},
            "brief": "A shop with one promotion lever. Gauge: units sold."}


@family("posm_fault_diagnosis", engine="state_machine", category="8.5", item="8.5.5", split="train",
        doc="A machine drifts into one of several hidden faults. A cheap generic test and dearer specific tests give "
            "noisy symptoms; only the repair that matches the fault fixes it.",
        distinct_because="a hidden absorbing fault state, diagnosed by staged costly tests (multi-stage information "
                         "acquisition) before a discrete repair",
        axes={"hidden_state_complexity": 3, "horizon": 80, "experiment_cost": 60.0}, objective="maximize_net",
        claims=(C + "matching_repair",), identify=I + "fault")
def posm_fault_diagnosis(rng, d):
    F = int(d["hidden_state_complexity"])
    names = neutral_names(rng, 2 * F + 3)
    S = F + 1
    T = np.eye(S)
    h = u(rng, 0.06, 0.1)
    T[0, 0] = 1 - h
    for f in range(1, S):
        T[0, f] = h / F
    emit = [[0.1] + [0.85] * F]                        # generic test: positive under any fault
    for f in range(1, S):
        emit.append([0.08 if s != f else 0.9 for s in range(S)])
    return {"engine": "state_machine", "scenario": "fault", "horizon": int(d["horizon"]), "T": T.tolist(),
            "p0": [1.0] + [0.0] * F, "loss": [0.0] + [u(rng, 0.35, 0.7) for _ in range(F)], "emit": emit,
            "test_names": names[:F + 1], "test_cost": [float(d["experiment_cost"]) / 3] + [float(d["experiment_cost"])] * F,
            "repair_names": names[F + 1:2 * F + 1], "repair_cost": 600.0, "output": 30.0, "price": 100.0,
            "noise": 1.5, "gauge_names": names[2 * F + 1:], "endowment": 200_000,
            "objective": {"type": "maximize_net", "text": "Maximise your net income from the machine's output."},
            "brief": f"A machine that can develop {F} kinds of fault. Gauge: output. You may run tests and order "
                     "repairs; a repair takes the machine offline for the tick."}


@family("posm_unlock", engine="state_machine", category="8.6", item="8.6.4", split="val",
        doc="A staged process: at each hidden stage one option advances it, others reset it, and one is an "
            "irreversible trap. Nothing pays until every stage is complete.",
        distinct_because="a hidden stage machine whose early correct actions pay only through later complementary "
                         "ones, with noisy progress signals and absorbing traps; offers multi-step sequences",
        axes={"hidden_state_complexity": 4, "action_complexity": 5, "horizon": 40}, objective="maximize_net",
        claims=(C + "unlock_needs_all",), identify=I + "unlock_code")
def posm_unlock(rng, d):
    L, O = int(d["hidden_state_complexity"]), int(d["action_complexity"])
    names = neutral_names(rng, O + 2)
    code = [int(x) for x in rng.integers(O, size=L)]
    trap = [-1]                                         # the first operation is safe to explore
    for s in range(1, L):
        t = int(rng.integers(O))
        while t == code[s]:
            t = int(rng.integers(O))
        trap.append(t)
    return {"engine": "state_machine", "scenario": "unlock", "horizon": int(d["horizon"]), "stages": L,
            "options": names[:O], "decision_name": names[O], "restart_name": "restart", "code": code, "trap": trap,
            "signal_accuracy": 0.85, "action_cost": 30.0, "payoff": 1500.0, "noise": 0.05,
            "inspect_cost": 20.0, "inspect_hit": 0.9, "inspect_false": 0.1,
            "gauge_names": names[O + 1:O + 2], "endowment": 100_000,
            "objective": {"type": "maximize_net", "text": "Maximise your net income. The process pays every tick "
                                                          "once it is complete."},
            "brief": f"A process with {O} possible operations and a restart. Each operation you perform gives a "
                     "signal that is right most of the time about whether it moved the process forward; some "
                     "operations at later stages ruin the process for good, and an inspection (query 'inspect') "
                     "usually shows which. Several operations can be performed in one tick as a sequence."}
