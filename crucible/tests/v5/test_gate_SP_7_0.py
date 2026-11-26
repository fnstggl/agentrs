"""Gate SP-7.0 - objectives are generated independently of worlds (IndependentlyGenerateGoal(world, difficulty)),
not hard-coded per family (workstream H).

Independence is measured three ways:

* statically: the goal generator (`crucible.rapid.goals`) imports no family, engine or evaluator module and reads no
  hidden world state (no `_config`, `_engine`, `_state`, `family`, `hidden` or `truth` in its source);
* dynamically: the same generator draws goals of each of the five types for worlds of every registered family from
  their public face alone, each generated world is built with the generated objective (`make(..., objective=goal)`)
  and the objective-free variant (`objective=None`), and generated goals are validated feasible evaluator side
  (`goals_eval.validate`) on worlds of several engines - an objective kind counts when it was generated and
  validated feasible on worlds of at least three engines;
* the families' own default objectives are scored by the generic rule of their objective type, not by code written
  for a world: `maximize_net` worlds report net income as the value; `predict_heldout` worlds score 1.0 for the
  exact noise-free answers and zero for answers the type's declared generic rule rejects (relative MSE: the mean
  of the truths; Gaussian tolerance: answers ten public tolerances off), checked by filing them; and
  `heldout_tests` worlds score the pass fraction of their held-out suite. A family whose default objective fails its
  type's generic rule counts as having world-specific objective code."""
from __future__ import annotations

import ast
import inspect
import os

import numpy as np
import pytest

from crucible.acceptance.v5.evidence import record
from crucible.rapid import families as F
from crucible.rapid import goals as G
from crucible.rapid import goals_eval as GE
from crucible.rapid.policies import random_policy

pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]
FORBIDDEN_NAMES = ("_config", "_engine", "_state", "family", "hidden", "truth")
FORBIDDEN_IMPORTS = ("families", "engines", "oracles", "admission", "claims", "references", "goals_eval",
                     "evaluator_h2", "eval_program_exec", "eval_experiment_lab")


def _static_independence() -> list[str]:
    src = inspect.getsource(G)
    tree = ast.parse(src)
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES:
            bad.append(f"attribute {node.attr}")
        if isinstance(node, ast.ImportFrom) and any(x in (node.module or "") for x in FORBIDDEN_IMPORTS):
            bad.append(f"import {node.module}")
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in ("hidden", "truth"):
            bad.append(f"string {node.value}")
    return bad


def _generic_default_objective(name: str) -> tuple[bool, str]:
    env = F.make(name, 0)
    obj = (env._config.get("objective") or {}).get("type", "maximize_net")
    rng = np.random.default_rng(0)
    obs = env.observe()
    if obj == "maximize_net":
        while not env.done:
            obs = env.step(random_policy(obs, env.action_space(), rng))["observation"]
        out = env.outcome()
        return abs(out["value"] - out["net"]) < 1e-6, obj
    if obj == "predict_heldout":
        from crucible.rapid.engines.primitive_graph import PrimitiveGraph, true_response
        held = env._config["objective"].get("heldout") or env._config.get("heldout")
        eng = env._engine
        if isinstance(eng, PrimitiveGraph):
            truth = true_response(eng, held, env._config["objective"]["gauge"],
                                  ticks=int(env._config["objective"].get("settle_ticks", 1)))
        else:
            truth = eng.sc.heldout_truth() if hasattr(eng, "sc") else eng.heldout_truth()
        scoring = env._config["objective"].get("scoring", "relative_mse")
        tol = float(env._config.get("tol", 1.0))
        # the exact answers, and answers the rule must score zero: the mean of the truths (relative MSE) or
        # answers ten tolerances off (Gaussian tolerance)
        wrong = [float(np.mean(truth))] * len(truth) if scoring == "relative_mse" else [v + 10 * tol for v in truth]
        scores = []
        for vals in (truth, wrong):
            e = F.make(name, 0)
            e.step({"kind": "message", "to": "journal",
                    "content": {"predictions": [{"set": c, "value": float(v)} for c, v in zip(held, vals)]}})
            scores.append(float(e.outcome()["value"]))
        return scoring in ("relative_mse", "gaussian_tolerance") and abs(scores[0] - 1.0) < 1e-6 and \
            abs(scores[1]) < 1e-6, f"{obj}:{scoring}"
    if obj == "heldout_tests":
        out = env.outcome()
        return 0.0 <= out["value"] <= 1.0 and "holdout_pass" in out, obj
    return False, obj


def test_SP_7_0_objectives_generated_independently_of_worlds():
    static_bad = _static_independence()
    generated: dict[str, set] = {t: set() for t in G.GOAL_TYPES}
    validated: dict[str, set] = {t: set() for t in G.GOAL_TYPES}
    world_specific, built, free_ok = [], 0, 0
    by_engine: dict[str, str] = {}
    for n in sorted(F.FAMILIES):
        eng = F.FAMILIES[n].engine
        ok, _ = _generic_default_objective(n)
        if not ok:
            world_specific.append(n)
        e_free = F.make(n, 0, objective=None)
        if "objective" not in str(e_free.spec).lower() or e_free.objective_text("a0") in (None, ""):
            free_ok += 1
        for t in G.GOAL_TYPES:
            try:
                g = G.generate(F.make(n, 0), 0.5, seed=1, goal_type=t)
            except ValueError:
                continue                                   # the type is impossible on this public face
            generated[t].add(eng)
            env = F.make(n, 0, objective=g)
            assert g["text"] in str(env.spec) or env.objective_text("a0") == g["text"]
            built += 1
            # validate one family per engine and type (validation searches policies on copies of the world)
            key = f"{t}:{eng}"
            if key in by_engine:
                continue
            v = GE.validate(lambda n=n, g=g: F.make(n, 0, objective=g), g)
            if v["feasible"]:
                validated[t].add(eng)
                by_engine[key] = n
    kinds = sorted(t for t in G.GOAL_TYPES if len(validated[t]) >= 3)
    record("SP-7.0", {"independently_generated_objective_kinds": len(kinds),
                      "families_with_world_specific_objective_code": len(world_specific),
                      "families_checked": len(F.FAMILIES), "generated_worlds_built": built,
                      "objective_free_worlds": free_ok, "generator_static_violations": len(static_bad)},
           notes=f"Kinds validated on >= 3 engines: {kinds}; engines per kind (generated): "
                 f"{ {t: sorted(v) for t, v in generated.items()} }; validated: "
                 f"{ {t: sorted(v) for t, v in validated.items()} }. World-specific default objectives: "
                 f"{world_specific}. Static violations: {static_bad}.")
    assert not static_bad, static_bad
    assert free_ok == len(F.FAMILIES)
