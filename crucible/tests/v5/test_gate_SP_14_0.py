"""SP-14.0 - Training diversity is measured along all six dimensions (parameters, causal structures, compositions,
objectives, participant policies, action spaces); seeds alone do not count (MASTER_SPEC Part 14; workstream N).

Two training streams are drawn from the registry: the default Tier 1 mixture (every training family, generated
objectives, three difficulty levels) and a seeds-only stream (one family, one level, its assigned objective). Every
drawn world is built and its six dimensions are *measured* on the built world (`rapid.fingerprints.instance_fingerprint`
and the evaluator-side configuration), not read from labels:

    parameters          distinct digests of the world's numeric configuration
    causal_structure    distinct (topology, temporal) fingerprint components
    composition         distinct (motifs, motif_edges) fingerprint components
    objective           distinct objective types of the built world (assigned or independently generated goal)
    participant_policies  distinct participant-policy fingerprint components
    action_space        distinct action-interface fingerprint components

A dimension counts as measured when it is computed for every world of both streams; the diverse stream must vary on
all six, the seeds-only stream only on parameters.
"""
from __future__ import annotations

import json

import pytest

from crucible.acceptance.v5.evidence import record
from crucible.research.v5 import distribution as D
from crucible.research.v5 import sampler as S

pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]


def _numbers(x, out):
    if isinstance(x, bool):
        return
    if isinstance(x, (int, float)):
        out.append(round(float(x), 6))
    elif isinstance(x, dict):
        for k in sorted(x):
            if k not in ("family", "name"):
                _numbers(x[k], out)
    elif isinstance(x, (list, tuple)):
        for v in x:
            _numbers(v, out)


def _dims(env) -> dict:
    from crucible.rapid.fingerprints import instance_fingerprint
    raw = env.env
    fp = instance_fingerprint(raw)
    nums = []
    _numbers(raw._config, nums)
    obj = raw.goal["type"] if getattr(raw, "goal", None) else (raw._config.get("objective") or {}).get("type")
    j = lambda *ks: json.dumps([fp.get(k) for k in ks], sort_keys=True, default=str)   # noqa: E731
    return {"parameters": D.sha(nums)[:16], "causal_structure": j("topology", "temporal"),
            "composition": j("motifs", "motif_edges"), "objective": str(obj),
            "participant_policies": j("participants"), "action_space": j("action_interface")}


def _stream(reg, mix, n, seed):
    s = S.Sampler(reg, mix, seed=seed)
    rows = []
    while len(rows) < n:
        sp = s.next()
        if not sp["entry"].startswith("t1:rapid:"):
            continue
        rows.append(_dims(S.make_env(sp, reg)))
    return rows


def test_SP14_0_six_dimensions_measured():
    reg = D.build_registry()
    diverse = _stream(reg, S.Mixture(tiers={1: 1.0}, sources=("crucible.rapid",), name="diverse"), 160, 3)
    fam = sorted(w["id"] for w in reg.worlds(tier=1, family_partition="train", source="crucible.rapid"))[0]
    seeds_only = _stream(reg, S.Mixture(tiers={1: 1.0}, entries=(fam,), difficulty={"0": 1.0},
                                        objectives={"assigned": 1.0}, name="seeds-only"), 40, 4)
    distinct = {k: len({r[k] for r in diverse}) for k in D.DIVERSITY_DIMENSIONS}
    distinct_seeds = {k: len({r[k] for r in seeds_only}) for k in D.DIVERSITY_DIMENSIONS}
    measured = [k for k in D.DIVERSITY_DIMENSIONS if all(r.get(k) is not None for r in diverse + seeds_only)]
    varying = [k for k in D.DIVERSITY_DIMENSIONS if distinct[k] > 1]
    seeds_vary = [k for k in D.DIVERSITY_DIMENSIONS if distinct_seeds[k] > 1]
    record("SP-14.0", {"diversity_dimensions_measured": len(measured), "dimensions_varying_in_training_stream":
                       len(varying), "dimensions_varying_with_seeds_only": len(seeds_vary),
                       **{f"distinct_{k}": distinct[k] for k in D.DIVERSITY_DIMENSIONS},
                       "worlds_measured": len(diverse) + len(seeds_only)},
           notes=f"Distinct values per dimension, diverse stream {distinct}; seeds-only stream ({fam}) "
                 f"{distinct_seeds}.")
    assert len(measured) == 6
    assert len(varying) == 6, distinct
    assert seeds_vary == ["parameters"], distinct_seeds
