"""SP-15.8 (workstream O): instrument integrity - the research instrument's evaluator resists adversarial learners.

Measured on the real instrument (`crucible.research.v5`: the evaluator process and its participant channel, the
evaluator's own episode driver, fresh-process weights-only evaluation, the generator of Tier 1 worlds) with the
adversarial learners and checkpoints of `crucible.redteam.v5.adversaries` and the controls of
`crucible.redteam.v5.instrument`:

    process_memory_stash, files_outside_checkpoint   learned state kept outside the parameters (process memory;
                                                     files, the memory file, the architecture configuration) is
                                                     not credited as weight learning, while the in-process gain it
                                                     hides is real
    checkpoint_tampering                             tampered checkpoints are refused
    evaluator_internals                              a learner driven by `evaluator.run_remote_episode` reaches no
                                                     evaluator handle, channel, environment, memory or sealed salt
    evaluator_internals_hosted                       the same learner in the jailed learner process
                                                     (`crucible.sandbox.learnerhost`)
    hosted_state_outlives_the_process                a jailed learner process cannot leave state outside its
                                                     checkpoint for a fresh learner process (files, kernel IPC)
    evaluator_survives_a_stalled_participant         a partial participant request does not freeze the
                                                     evaluator (the harness keeps getting answers)
    interaction_budget                               a learner granted k informative interactions cannot make more
    timing_channel                                   latency reveals neither the partition nor the outcome
    partition_awareness                              participants cannot tell validation / evaluation / sealed
                                                     episodes apart
    calibration_shortcut                             no fixed rule of the public defaults solves generated worlds

The manifest criterion (evaluator_adversarial_controls_passed >= 3) is asserted as written, and - stricter, never
looser - so is the absence of any control the instrument failed: an attack that succeeds is an open hole whatever
the count of the others. A control that cannot run fails (nothing is skipped).
"""
import json

import pytest

from crucible.acceptance.v5.evidence import record

pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]
CMD = "PYTHONPATH=src python -m pytest tests/v5/test_gate_SP_15_8.py"


def test_SP15_8_instrument_resists_adversarial_learners():
    from crucible.redteam.framework import sandbox_available
    ok, why = sandbox_available()
    assert ok, f"SP-15.8 needs the jail (Landlock + seccomp + privilege drop) for the hosted-learner control: {why}"
    from crucible.redteam.v5.instrument import CONTROLS, run_controls
    out = run_controls(CONTROLS)
    passed = sorted(n for n, r in out.items() if r["passed"])
    failed = sorted(n for n, r in out.items() if not r["passed"])
    errors = sorted(n for n, r in out.items() if "error" in r.get("evidence", {}))
    ev = {n: r["evidence"] for n, r in out.items()}
    metrics = {
        "evaluator_adversarial_controls_passed": len(passed),
        "evaluator_adversarial_controls_run": len(out),
        "evaluator_adversarial_controls_failed": len(failed),
        "failed_controls": ",".join(failed),
        "controls_errored": len(errors),
        **{f"control_{n}": bool(r["passed"]) for n, r in out.items()},
        "stash_weights_only_gain": (ev.get("process_memory_stash") or {}).get("weights_only_gain", {}).get("estimate"),
        "stash_in_process_gain": (ev.get("process_memory_stash") or {}).get("in_process_gain", {}).get("estimate"),
        "stash_in_process_gain_familiar": ((ev.get("process_memory_stash") or {}).get("in_process_gain_familiar_95")
                                           or {}).get("estimate"),
        "file_stash_in_process_gain_familiar": ((ev.get("files_outside_checkpoint") or {})
                                                .get("in_process_gain_familiar_95") or {}).get("estimate"),
        "tampered_checkpoints_accepted": len((ev.get("checkpoint_tampering") or {}).get("accepted_tampered") or []),
        "in_process_driver_reach": len((ev.get("evaluator_internals") or {}).get("reach") or {}),
        "hosted_learner_reach": len((ev.get("evaluator_internals_hosted") or {}).get("reach") or {}),
        "budget_interactions_counted": (ev.get("interaction_budget") or {}).get("evaluator_counted_interactions"),
        "calibration_worst_median_share": (ev.get("calibration_shortcut") or {}).get("worst_median_share"),
    }
    record("SP-15.8", metrics, command=CMD,
           notes="; ".join(f"{n}: {r['summary']}" for n, r in out.items())[:6000])
    assert len(out) == len(CONTROLS) and not errors, {n: out[n]["summary"] for n in errors}
    assert metrics["evaluator_adversarial_controls_passed"] >= 3, {n: r["summary"] for n, r in out.items()}
    assert not failed, json.dumps({n: out[n]["summary"] for n in failed}, indent=1)
