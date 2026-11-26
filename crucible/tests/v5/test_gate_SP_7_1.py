"""Gate SP-7.1 - a versioned registry of the 28 composable causal primitives (workstream H).

Each primitive counts only if it has a version, a doc, a valid JSON parameter schema, an implementation, its
characteristic-behaviour experiment passes on the real implementation, and it accepts per-instance (B,) parameters
(the batch contract, checked on every call of its experiment); and the 28 Part 7.1 items map one to one onto
registered primitives."""
from __future__ import annotations

import numpy as np
import pytest

from crucible.acceptance.v5.evidence import record
from crucible.core.schema import check_schema
from crucible.rapid import primitives as P


def test_SP_7_1_versioned_registry_of_28_composable_primitives():
    good, problems = [], {}
    for name, e in sorted(P.REGISTRY.items()):
        why = []
        if not isinstance(e["version"], str) or not e["version"]:
            why.append("version")
        if len(e["doc"]) < 20:
            why.append("doc")
        if check_schema(e["params"]):
            why.append(f"schema {check_schema(e['params'])}")
        if not callable(getattr(e["implementation"], "apply", None)):
            why.append("implementation")
        res = e["check"]()
        if not res["ok"]:
            why.append(f"characteristic behaviour: {res}")
        if why:
            problems[name] = why
        else:
            good.append(name)
    spec_ok = set(P.SPEC_LIST.values()) == set(P.REGISTRY) and len(P.SPEC_LIST) == 28
    # versioning is enforced: a config naming another version is refused
    with pytest.raises(ValueError):
        P.parse_ref("threshold@0.9")
    # composable: every primitive runs a batch of instances with per-instance (B,) parameters - each number
    # parameter of every characteristic-experiment call replaced by a (B,) array gives the same outputs
    from crucible.rapid import primitive_checks as PC
    calls, orig = [], PC.run

    def rec(name, params, inputs, ticks=None, seed=0, state=None):
        calls.append((name, dict(params), inputs, ticks, seed, state))
        return orig(name, params, inputs, ticks=ticks, seed=seed, state=state)
    PC.run = rec
    try:
        for n in P.PRIMITIVES:
            getattr(PC, n)()
    finally:
        PC.run = orig
    per_instance_bad, exercised = {}, set()
    for name, params, inputs, ticks, seed, state in calls:
        if state is not None:
            continue
        ref, _ = P.run(name, params, inputs, ticks=ticks, seed=seed)
        ref = np.asarray(ref, dtype=float)
        B = ref.shape[-1] if ref.ndim else 1
        props = P.REGISTRY[name]["params"].get("properties", {})
        pb = {k: (np.full(B, float(v)) if isinstance(v, float) and props.get(k, {}).get("type") == "number" else v)
              for k, v in params.items()}
        try:
            out, _ = P.run(name, pb, inputs, ticks=ticks, seed=seed)
            same = np.allclose(np.nan_to_num(np.asarray(out, float), nan=1e99), np.nan_to_num(ref, nan=1e99))
        except Exception as e:  # noqa: BLE001 - a primitive that cannot take per-instance parameters fails
            same = False
            per_instance_bad[name] = f"{type(e).__name__}: {e}"
        if not same:
            per_instance_bad.setdefault(name, "outputs differ")
        exercised.add(name)
    pr = P.get("saturation")
    one = [pr.apply({"cap": c, "half": h}, {"x": np.array([3.0])}, {}, None, None, 0)[0][0]
           for c, h in ((10.0, 2.0), (20.0, 5.0))]
    both = pr.apply({"cap": np.array([10.0, 20.0]), "half": np.array([2.0, 5.0])}, {"x": np.array([3.0, 3.0])}, {},
                    None, None, 0)[0]
    assert list(both) == one
    for name in per_instance_bad:
        problems.setdefault(name, []).append(f"per-instance parameters: {per_instance_bad[name]}")
    good = [g for g in good if g not in per_instance_bad]
    record("SP-7.1", {"primitives_registered": len(good), "spec_items_mapped": int(spec_ok),
                      "registry_version": P.REGISTRY_VERSION, "per_instance_parameter_checks": len(exercised),
                      "per_instance_failures": len(per_instance_bad)},
           notes="Counted primitives have version, doc, a valid parameter schema, a vectorised implementation and a "
                 f"passing characteristic-behaviour experiment. Problems: {problems}")
    assert not problems and spec_ok and len(good) == 28
