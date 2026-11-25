"""Gate SP-7.1 - a versioned registry of the 28 composable causal primitives (workstream H).

Each primitive counts only if it has a version, a doc, a valid JSON parameter schema, an implementation, and its
characteristic-behaviour experiment passes on the real implementation; and the 28 Part 7.1 items map one to one
onto registered primitives."""
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
    # composable: the same implementation runs a batch of instances with per-instance parameters
    pr = P.get("saturation")
    one = [pr.apply({"cap": c, "half": h}, {"x": np.array([3.0])}, {}, None, None, 0)[0][0]
           for c, h in ((10.0, 2.0), (20.0, 5.0))]
    both = pr.apply({"cap": np.array([10.0, 20.0]), "half": np.array([2.0, 5.0])}, {"x": np.array([3.0, 3.0])}, {},
                    None, None, 0)[0]
    assert list(both) == one
    record("SP-7.1", {"primitives_registered": len(good), "spec_items_mapped": int(spec_ok),
                      "registry_version": P.REGISTRY_VERSION},
           notes="Counted primitives have version, doc, a valid parameter schema, a vectorised implementation and a "
                 f"passing characteristic-behaviour experiment. Problems: {problems}")
    assert not problems and spec_ok and len(good) == 28
