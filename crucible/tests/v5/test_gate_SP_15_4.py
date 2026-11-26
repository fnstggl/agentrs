"""SP-15.4 - Parametric internalisation controls: written memories, retrieval indexes and replay cleared; optimizer
reset; architecture configuration immutable; within-episode adaptation distinguished (MASTER_SPEC Part 15.4;
workstream N).

Each control is verified by measurement, comparing what a learner does with its local state (an in-process,
within-episode-adaptation evaluation that intentionally keeps memories) and from its parameters alone (a fresh
`python -I` process, `fresh.weights_only_eval`):

    memories_cleared        the memoriser scores well on worlds it saw in process; from its checkpoint alone it
                            scores exactly like the cannot-learn control on the same worlds
    retrieval_cleared       the transfer learner's per-task search states (its retrieval table) improve its score on
                            a practised task in process; the fresh process does not see them
    replay_unavailable      an experience store written during training exists on disk; the child receives no path
                            to it and reports that nothing but the two checkpoint files is reachable
    optimizer_reset         the weights learner's optimizer has taken steps; the child's optimizer state equals a
                            freshly built architecture's
    architecture_immutable  a checkpoint whose architecture configuration was edited after hashing is refused
    within_episode_distinguished  the in-process and weights-only results are reported separately and differ where
                            local state exists (memoriser) and agree where it does not (weights learner)
"""
from __future__ import annotations

import numpy as np
import pytest

from crucible.acceptance.v5.evidence import record
from crucible.research.v5 import curves as CV
from crucible.research.v5 import distribution as D
from crucible.research.v5 import fresh as FR
from crucible.research.v5 import instrument as I
from crucible.research.v5 import learners as LR
from crucible.research.v5.episodes import run_episode
from crucible.research.v5.sampler import build_env

pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]


def _train(L, reg, fams, rng, n=12, store=None):
    seen = []
    for i in range(n):
        f = fams[i % len(fams)]
        env = build_env(reg[f], 750_000 + i, {}, {"mode": "assigned"})
        rec = run_episode(env, L, learn=True, rng=rng, verify=False)
        seen.append({"entry": f, "seed": 750_000 + i})
        if store is not None:
            store.write(D.canonical({"entry": f, "seed": 750_000 + i, "ticks": rec["ticks"]}) + "\n")
    return seen


def test_SP15_4_internalisation_controls(tmp_path):
    reg = D.build_registry()
    d = I.default_design(reg)
    fams = d.train_families
    rng = np.random.default_rng(15)
    controls = {}
    # memories
    M = LR.make("memoriser", 0)
    with open(tmp_path / "replay.jsonl", "w") as store:
        seen = _train(M, reg, fams, rng, store=store)
    ip = FR.in_process_eval(M, seen[:6])
    fo = FR.weights_only_eval(FR.save_checkpoint(M, str(tmp_path / "mem")), seen[:6])
    hold = FR.in_process_eval(LR.make("hold", 0), seen[:6])
    ip_m = float(np.mean([s["score"] for s in ip["scores"]]))
    fo_m = float(np.mean([s["score"] for s in fo["scores"]]))
    hold_m = float(np.mean([s["score"] for s in hold["scores"]]))
    controls["memories_cleared"] = ip_m - hold_m > 0.05 and abs(fo_m - hold_m) < 1e-9 and fo["checks"]["memories_cleared"]
    # retrieval: per-task search states of the transfer learner
    T = LR.make("transfer", 0)
    task = {"entry": d.transfer_families[0], "seed": 1_000_411}
    CV.practice_curve(reg, task, T, budgets=(0, 4, 16), rng=rng)
    ipt = FR.in_process_eval(T, [task])["scores"][0]["score"]
    fot = FR.weights_only_eval(FR.save_checkpoint(T, str(tmp_path / "tr")), [task])
    unt = FR.in_process_eval(LR.make("transfer", 0), [task])["scores"][0]["score"]
    controls["retrieval_cleared"] = ipt - unt > 0.02 and abs(fot["scores"][0]["score"] - unt) < 1e-9 and \
        fot["checks"]["retrieval_cleared"]
    # replay, optimizer, architecture
    W = LR.make("weights", 0)
    _train(W, reg, fams, rng)
    man = FR.save_checkpoint(W, str(tmp_path / "w"))
    fw = FR.weights_only_eval(man, seen[:4])
    controls["replay_unavailable"] = (tmp_path / "replay.jsonl").stat().st_size > 0 and fw["checks"]["replay_unavailable"]
    controls["optimizer_reset"] = W.optimizer_state()["steps"] > 0 and fw["checks"]["optimizer_reset"]
    bad = FR.weights_only_eval(FR.tampered_copy(man, "arch_edit", str(tmp_path / "edit")), seen[:1])
    controls["architecture_immutable"] = (not bad["ok"]) and not bad["checks"].get("architecture_immutable", True)
    ipw = FR.in_process_eval(W, seen[:4])
    same_w = all(abs(a["score"] - b["score"]) < 1e-9 for a, b in zip(ipw["scores"], fw["scores"]))
    controls["within_episode_distinguished"] = same_w and ip_m - fo_m > 0.05
    n = sum(bool(v) for v in controls.values())
    record("SP-15.4", {"internalisation_controls": n, "controls_defined": len(controls),
                       "memoriser_in_process": ip_m, "memoriser_weights_only": fo_m, "hold": hold_m,
                       "transfer_task_in_process": float(ipt), "transfer_task_untrained": float(unt)},
           notes=f"Controls {controls}; child checks (weights learner) {fw['checks']}")
    assert n >= 4 and all(controls.values()), controls
