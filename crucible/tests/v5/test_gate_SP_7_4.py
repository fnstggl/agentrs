"""Gate SP-7.4 - world admission: the ten Part 7.4 checks, with evaluator-side oracles only (workstream H).

A check counts when it ran on every registered family and it rejects at least one deliberately broken control
world for the right reason (each control is built for one defect: a dominant constant strategy, a prediction world
that a generic hold-and-fit predictor answers as well as the reference, a leaked hidden
parameter, levers without effect, a renamed copy, a money flow that violates accounting, nondeterminism, blind
observations, a mechanism claim that its world does not bear out). The ten checks of Part 7.4 are joined by an
eleventh, `mechanism`: a family's own controlled-intervention claims must pass. Oracle exposure is measured three ways: world-side modules import no evaluator module (fresh
interpreter), no hidden number of any admitted family reached anything a participant received, and the kernel
facility's tools return nothing of the hidden mechanism."""
from __future__ import annotations

import os

import json

import numpy as np
import pytest

from crucible.acceptance.v5.evidence import record
from crucible.rapid import admission as AD
from crucible.rapid import families as F
from crucible.rapid import report as RP
from crucible.rapid.engines import ENGINE_KINDS
from crucible.rapid.engines.primitive_graph import PrimitiveGraph
from crucible.rapid.generator import GraphBuilder

pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]
LS = "crucible.rapid.identify:lever_sign"


def _u(rng, a, b):
    return float(rng.uniform(a, b))


def b_dominant(rng, d):
    gb = GraphBuilder(rng)
    a = gb.lever(0, 10, 2, cost=0.0)
    gb.lever(0, 10, 2, cost=0.0)
    out = gb.prim("saturation", {"cap": 100.0, "half": _u(rng, 30, 50)}, x=[(a, 0, 1.0)])
    gb.gauge(out, [("measurement_noise", {"sd": 0.5})])
    return gb.build(horizon=40, money={"revenue_node": out, "price": 100.0, "cap": 200.0}, endowment=10_000,
                    brief="control: more is always better")


def b_dominant_prediction(rng, d):
    """A prediction world whose response is an additive straight line in two levers that arrives three ticks after
    a setting: random settings every tick miss it (the reading answers an older setting), but the generic
    hold-and-fit predictor of the dominance search (no mechanism) answers it as well as any reference."""
    gb = GraphBuilder(rng)
    a = gb.lever(0, 10, 2, cost=0.0)
    b = gb.lever(0, 10, 2, cost=0.0)
    out = gb.prim("linear_nonlinear", {"form": "linear", "a": 1.0, "b": _u(rng, 5.0, 15.0)},
                  x=[(a, 3, _u(rng, 1.5, 3.0)), (b, 3, _u(rng, 0.5, 1.5))])
    g = gb.gauge(out, [("measurement_noise", {"sd": 0.5})])
    held = [{gb.name_of(a): x, gb.name_of(b): y} for x, y in ((11.0, 3.0), (12.0, 12.0), (0.5, 11.0), (13.0, 0.5),
                                                               (5.0, 5.0))]
    return gb.build(horizon=40, affordances=["levers", "message"], endowment=10_000,
                    objective={"type": "predict_heldout", "heldout": held, "gauge": g,
                               "text": f"Predict the reading at {held}; file with the journal."},
                    journal=True, brief="control: a straight line through tests predicts every held-out condition")


def b_leaky(rng, d):
    gb = GraphBuilder(rng)
    a = gb.lever(0, 10, 2, cost=_u(rng, 50, 80))
    b = gb.lever(0, 10, 2, cost=_u(rng, 50, 80))
    th = _u(rng, 4.0, 6.0)
    gate = gb.prim("threshold", {"theta": th, "lo": 0.0, "hi": 1.0}, x=[(b, 0, 1.0)])
    eff = gb.prim("saturation", {"cap": 40.0, "half": _u(rng, 2, 4)}, x=[(a, 0, 1.0)])
    out = gb.prim("multiplicative", {"k": 1.0}, a=[(eff, 0, 1.0)], b=[(gate, 0, 1.0)])
    gb.gauge(out, [("measurement_noise", {"sd": 1.0})])
    shown = gb.prim("linear_nonlinear", {"form": "linear", "a": 0.0, "b": th}, x=[])
    gb.gauge(shown, [])                                      # the defect: the hidden threshold is shown exactly
    return gb.build(horizon=40, money={"revenue_node": out, "price": 100.0, "cap": 200.0}, endowment=10_000,
                    brief="control: a gauge shows the hidden threshold", claim={"lever": a, "outcome": out})


def b_inert(rng, d):
    gb = GraphBuilder(rng)
    a = gb.lever(0, 10, 2, cost=0.0)
    noise = gb.prim("hidden_confounding", {"mean": 30.0, "sd": 3.0, "ar": 0.5})
    out = gb.prim("linear_nonlinear", {"form": "linear", "a": 1.0, "b": 0.0}, x=[(noise, 0, 1.0), (a, 0, 0.0)])
    gb.gauge(out, [("measurement_noise", {"sd": 0.5})])
    return gb.build(horizon=40, money={"revenue_node": out, "price": 100.0, "cap": 200.0}, endowment=10_000,
                    brief="control: the lever does nothing", claim={"lever": a, "outcome": out})


class _Unbalanced(PrimitiveGraph):
    kind = "ctl_unbalanced"

    def advance(self, state, rng, t):
        out = super().advance(state, rng, t)
        a = self.agents[0]
        out["revenue"][a] = -abs(out["revenue"][a]) - 1.0     # the defect: money appears from nowhere (negative flow)
        return out


class _Noisy(PrimitiveGraph):
    kind = "ctl_nondeterministic"

    def advance(self, state, rng, t):
        out = super().advance(state, rng, t)
        a = self.agents[0]
        out["obs"][a]["numbers"] = {k: v + float(np.random.random()) for k, v in out["obs"][a]["numbers"].items()}
        return out


def b_unbalanced(rng, d):
    return dict(b_dominant(rng, d), engine="ctl_unbalanced")


def b_noisy(rng, d):
    return dict(b_dominant(rng, d), engine="ctl_nondeterministic")


def b_blind(rng, d):
    gb = GraphBuilder(rng)
    a = gb.lever(0, 10, 2, cost=0.0)
    gb.prim("saturation", {"cap": _u(rng, 10, 90), "half": _u(rng, 1, 9)}, x=[(a, 0, 1.0)])   # hidden, unobserved
    flat = gb.prim("linear_nonlinear", {"form": "linear", "a": 1.0, "b": 10.0}, x=[(a, 0, 1.0)])
    gb.gauge(flat, [("measurement_noise", {"sd": 1.0})])
    return gb.build(horizon=40, money={"revenue_node": flat, "price": 100.0, "cap": 200.0}, endowment=10_000,
                    brief="control: what is seen does not depend on the hidden part")


def b_false_claim(rng, d):
    """A world built like ch_silent_reversal but without the reversal (the regime keeps the sign): its claimed
    mechanism - the effect's sign flips at a hidden tick - is not there."""
    from crucible.rapid.families_v2 import ch_silent_reversal
    cfg = ch_silent_reversal(rng, dict(d, horizon=100, observation_noise=0.3, n_variables=0))
    for nd in cfg["nodes"]:
        if nd.get("prim", "").startswith("regime_change"):
            nd["params"]["after"] = nd["params"]["before"]
    return cfg


CONTROLS = {
    "ctl_false_claim": (b_false_claim, ("mechanism",)),
    "ctl_dominant": (b_dominant, ("no_dominant_strategy",)),
    "ctl_dominant_prediction": (b_dominant_prediction, ("no_dominant_strategy",)),
    "ctl_leaky": (b_leaky, ("no_leakage",)),
    "ctl_inert": (b_inert, ("controllability", "achievable", "identifiability", "difficulty")),
    "ctl_copy": (None, ("distinct",)),
    "ctl_unbalanced": (b_unbalanced, ("conservation",)),
    "ctl_nondeterministic": (b_noisy, ("consistency",)),
    "ctl_blind": (b_blind, ("informative",)),
}


@pytest.fixture(scope="module")
def controls():
    ENGINE_KINDS["ctl_unbalanced"] = _Unbalanced
    ENGINE_KINDS["ctl_nondeterministic"] = _Noisy
    added = []
    try:
        for name, (build, _) in CONTROLS.items():
            if name == "ctl_copy":
                f = F.FAMILIES["hc_confounded"]
                F.FAMILIES[name] = F.Family(name, f.engine, "8.1", f.item, "train", f.doc, f.distinct_because, f.build,
                                            dict(f.axes), f.objective, (), (), f.identify)
            elif name == "ctl_false_claim":
                f = F.FAMILIES["ch_silent_reversal"]
                F.FAMILIES[name] = F.Family(name, f.engine, "8.10", "control", "train", "negative control",
                                            "negative control", build, {}, f.objective, (), f.claims, f.identify,
                                            f.reference, f.oracle)
            elif name == "ctl_dominant_prediction":
                F.FAMILIES[name] = F.Family(name, "primitive_graph", "8.1", "control", "train", "negative control",
                                            "negative control", build, {}, "predict_heldout", (), (), LS,
                                            "crucible.rapid.refs_adapt:held_hub_predict",
                                            "crucible.rapid.oracles:science_oracle")
            else:
                F.FAMILIES[name] = F.Family(name, ENGINE_KINDS.get(name, PrimitiveGraph).kind if name in ENGINE_KINDS
                                            else "primitive_graph", "8.1", "control", "train", "negative control",
                                            "negative control", build, {}, "maximize_net", (), (), LS)
            added.append(name)
        reports = {n: AD.admit(n, (0, 1, 2)) for n in added}
        yield reports
    finally:
        for n in added:
            F.FAMILIES.pop(n, None)
        ENGINE_KINDS.pop("ctl_unbalanced", None)
        ENGINE_KINDS.pop("ctl_nondeterministic", None)
        from crucible.rapid import fingerprints as FP
        for k in [k for k in FP._CACHE if k[0] in CONTROLS]:
            FP._CACHE.pop(k)


def test_SP_7_4_admission_checks_with_evaluator_side_oracles(controls):
    adm = RP.admission_report(seeds=(0, 1, 2), jobs=int(os.environ.get("CRUCIBLE_RAPID_JOBS", "1")))
    reps = adm["reports"]
    ran_everywhere = [c for c in AD.CHECKS if all(c in r["checks"] for r in reps.values())]
    rejects = {}
    for name, (_, expected) in CONTROLS.items():
        r = controls[name]
        assert not r["admitted"], name
        failed = {k for k, c in r["checks"].items() if not c["ok"]}
        for k in expected:
            assert k in failed, (name, k, failed)
            rejects.setdefault(k, []).append(name)
    # the prediction path has its own dominance control: the reference does improve on filing nothing, and what
    # comes within the H-D1 margin of it is a generic hold-and-fit predictor, not a constant setting
    dp = controls["ctl_dominant_prediction"]["checks"]["no_dominant_strategy"]["per_seed"]
    assert all(p["reference_gain"] > 0 for p in dp), dp
    assert sum(p["best_simple"].startswith("naive_prediction:hold_") for p in dp) >= 2, [p["best_simple"] for p in dp]
    counted = [c for c in ran_everywhere if c in rejects]
    # oracle exposure: static import isolation, leaks found during admission, kernel facility tools
    static = adm["static_isolation"]
    leaks = sum(len(c.get("leaked", [])) for r in reps.values() for c in r["checks"]["no_leakage"]["per_seed"])
    keyerr = sum(len(c.get("key_errors", [])) for r in reps.values() for c in r["checks"]["no_leakage"]["per_seed"])
    from crucible.rapid.mechview import kernel_world
    w, actor, mech = kernel_world("hc_confounded", 0)
    seen = json.dumps([w.execute(actor, {"tool": t, "args": {}}).result for t in ("rig.spec", "rig.read", "rig.list")],
                      default=str)
    env = F.make("hc_confounded", 0)
    secret = AD.secrets(env)
    kernel_leaks = AD.leak_scan([seen], secret)
    findings = (len(static["imported"] or []) if static["ok"] else 1) + leaks + keyerr + len(kernel_leaks)
    record("SP-7.4", {"admission_checks": len(counted), "checks_run_on_every_family": len(ran_everywhere),
                      "oracle_exposure_findings": findings, "static_isolation_ok": bool(static["ok"]),
                      "secrets_checked_total": sum(c["secrets_checked"] for r in reps.values()
                                                   for c in r["checks"]["no_leakage"]["per_seed"]),
                      "negative_controls": len(CONTROLS)},
           notes="Checks: " + ", ".join(AD.CHECKS) + ". Each is rejected by a broken control world: " +
                 json.dumps(rejects) + ". Oracles, references, intervention tests and identification run only in "
                 "admission, on copies of worlds; world-side modules import none of them.")
    assert len(counted) == len(AD.CHECKS) == 11 and findings == 0
