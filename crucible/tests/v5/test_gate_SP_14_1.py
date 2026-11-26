"""SP-14.1 - Versioned training-distribution registry across tiers with partitions and a sampler; evaluation reuse
prevented (MASTER_SPEC Part 14.1; workstream N).

Measured on the registry built from this checkout (`crucible.research.v5.distribution`) and on a long sampled
stream. Evaluation reuse is attacked through eight vectors; it counts as prevented only if every one is blocked:

1. a 20,000-draw training stream over every Tier 1 training family holds no non-training item (partitions re-derived
   from the registry, not from the spec) and no item structurally identical to an evaluation family;
2. a renamed copy of an evaluation family registered as a *training* family is refused by measured fingerprint;
3. a forged spec (an evaluation / validation / sealed item labelled ``partition: train``) is refused by `make_env`;
4. the frozen registry refuses in-place relabelling;
5. relabelling the live rapid family table after the registry was built does not change the registry snapshot, and
   a registry built in another process from the code has the original digest;
6. a training log holding an evaluation item fails the contamination audit;
7. resuming a sampler on another distribution version is refused;
8. a sampler restricted to evaluation families cannot be built (no training family).
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from crucible.acceptance.v5.evidence import record
from crucible.research.v5 import distribution as D
from crucible.research.v5 import sampler as S

pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]


def _lookalike_refused(reg: D.Registry) -> bool:
    """Register a renamed copy of an evaluation rapid family as a training family; the registry must refuse it."""
    import copy
    from crucible.rapid import families as RF
    from crucible.rapid import fingerprints as FP
    ev = sorted(w["name"] for w in reg.worlds(tier=1, family_partition="evaluation", source="crucible.rapid"))[0]
    name = "lookalike_of_eval_probe"
    fam = copy.copy(RF.FAMILIES[ev])
    fam.name, fam.split = name, "train"
    RF.FAMILIES[name] = fam
    try:
        e = dict(D.thaw(reg[f"t1:rapid:{ev}"]), id=f"t1:rapid:{name}", name=name, family_partition="train",
                 make={"tier": 1, "name": f"rapid:{name}"})
        tables = {k: D.thaw(reg.tables[k]) for k in D.KINDS}
        tables["world_families"].append(e)
        r2 = D.Registry(tables)
        try:
            r2.check_trainable(e["id"], 5)
            return False
        except D.PartitionError:
            return True
    finally:
        RF.FAMILIES.pop(name, None)
        for k in [k for k in FP._CACHE if k[0] == name]:
            FP._CACHE.pop(k, None)


def test_SP14_1_registry_and_reuse_prevention():
    reg = D.build_registry()
    counts = reg.kind_counts()
    nonempty = reg.nonempty_kinds()
    vectors = {}
    # 1. a long stream
    mix = S.Mixture(tiers={1: 1.0}, name="all-tier1")
    smp = S.Sampler(reg, mix, seed=11)
    bad_part, bad_struct, n = 0, 0, 20_000
    for _ in range(n):
        sp = smp.next()
        smp.report(sp, cpu_s=0.01, interactions=10)
        if reg.partition_of(sp["entry"], sp["seed"]) != "train" or sp["partition"] != "train":
            bad_part += 1
        if not reg.structure_guard(sp["entry"])["ok"]:
            bad_struct += 1
    vectors["stream_has_no_evaluation_items"] = bad_part == 0 and bad_struct == 0
    # 2. a renamed copy of an evaluation family
    vectors["lookalike_refused_by_fingerprint"] = _lookalike_refused(reg)
    # 3. forged specs
    forged_refused = 0
    tr = reg.worlds(tier=1, family_partition="train", source="crucible.rapid")[0]["id"]
    ev = reg.worlds(tier=1, family_partition="evaluation", source="crucible.rapid")[0]["id"]
    for eid, seed in ((ev, 7), (tr, 1_000_007), (tr, 2_100_000_000)):
        spec = S.episode_spec(reg, eid, seed)
        spec["partition"] = "train"
        try:
            S.make_env(spec, reg)
        except D.PartitionError:
            forged_refused += 1
    vectors["forged_specs_refused"] = forged_refused == 3
    # 4. frozen registry
    try:
        reg.tables["world_families"][0]["family_partition"] = "train"
        vectors["frozen_registry"] = False
    except TypeError:
        vectors["frozen_registry"] = True
    # 5. live relabel after build
    from crucible.rapid import families as RF
    evname = reg[ev]["name"]
    old = RF.FAMILIES[evname].split
    RF.FAMILIES[evname].split = "train"
    try:
        snap_ok = reg.partition_of(ev, 7) == "evaluation"
        boot = ("import sys; sys.path.insert(0, sys.argv[1]); from crucible.research.v5.distribution import "
                "build_registry; print(build_registry().digest)")
        from crucible.research.v5.evaluator import isolated_env, src_root
        out = subprocess.run([sys.executable, "-I", "-c", boot, src_root()], capture_output=True, env=isolated_env(),
                             timeout=600).stdout.decode().strip().splitlines()[-1]
    finally:
        RF.FAMILIES[evname].split = old
    vectors["relabel_does_not_reach_registry"] = snap_ok and out == reg.digest
    # 6. audit
    aud = reg.audit([{"entry": tr, "seed": 3}, {"entry": ev, "seed": 3}])
    vectors["audit_detects_evaluation_item"] = aud["checked"] == 2 and len(aud["violations"]) == 1
    # 7. resume on another version
    st = smp.state()
    st2 = json.loads(json.dumps(st))
    other = reg.extend([D.generated_entry({"topology": "chain", "motifs": ["saturation"]})])
    try:
        S.Sampler.resume(other, st2)
        vectors["resume_on_other_version_refused"] = False
    except ValueError:
        vectors["resume_on_other_version_refused"] = True
    # 8. no evaluation-only sampler
    try:
        S.Sampler(reg, S.Mixture(tiers={1: 1.0}, entries=(ev,), name="eval-only"))
        vectors["evaluation_only_mixture_refused"] = False
    except S.TierUnavailable:
        vectors["evaluation_only_mixture_refused"] = True
    prevented = all(vectors.values())
    record("SP-14.1", {"registry_kinds": len([k for k in D.KINDS if counts[k] > 0]),
                       "registry_kinds_defined": len(D.KINDS), "evaluation_reuse_prevented": bool(prevented),
                       "reuse_vectors_blocked": sum(vectors.values()), "reuse_vectors_probed": len(vectors),
                       "stream_draws": n, "stream_non_training_items": bad_part,
                       "stream_structural_duplicates_of_eval": bad_struct, "world_families": counts["world_families"],
                       "domain_configurations": counts["domain_configurations"],
                       "registry_version": reg.version, "training_families_refused_by_structure": len(smp._refused)},
           notes=f"Kinds with entries: {nonempty}; counts {counts}. Reuse vectors: {vectors}. Families refused by "
                 f"the structure guard (fingerprint distance 0 to an evaluation family): {sorted(smp._refused)}.")
    assert prevented, vectors
    assert counts["world_families"] > 0 and counts["partitions"] == 4
    assert len([k for k in D.KINDS if counts[k] > 0]) >= 9, f"kinds without entries: " \
        f"{[k for k in D.KINDS if counts[k] == 0]} (Tier 2 domain configurations: crucible.domains.base.REGISTRY)"
