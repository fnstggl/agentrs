"""Gate SP-0.9 - Crucible 4.0 validation and sealed evaluation families are not moved into any v5 training
distribution, checked structurally (workstream H).

Every 4.0 validation and test mechanism family (`crucible.mechanisms`, engines scm/abm/techsys) is fingerprinted in
the v5 vocabulary from its instantiated engine (`fingerprints.legacy_fingerprint`: its hidden motifs mapped onto the
primitive vocabulary, its causal graph class, lags and seats). Every registered v5 training family is fingerprinted
from its instantiated worlds and compared with each of them two ways: the structural distance (0 = the same
structure) and mechanism recreation (`fingerprints.legacy_recreation`: the same mechanism motif set offered through a
lever-like interface, so a learner trained on it meets the held-out mechanism again). A v5 training family that
matches a 4.0 held-out family either way counts as contamination. Names play no part, except that no v5 family may
reuse a 4.0 family name."""
from __future__ import annotations

import pytest

from crucible.acceptance.v5.evidence import record
from crucible.rapid import families as F
from crucible.rapid import fingerprints as FP

pytestmark = [pytest.mark.timeout(1800)]


def test_SP_0_9_no_4_0_heldout_mechanism_in_v5_training():
    from crucible.mechanisms.base import mech_families
    from crucible.mechanisms.sim import ensure_engines
    ensure_engines()
    heldout = sorted(mech_families(split="val") + mech_families(split="test"))
    sealed = FP.legacy_sealed()
    assert sorted(sealed) == heldout                       # every held-out family was fingerprinted
    for n, rec in sealed.items():
        fp = rec["fingerprint"]
        assert fp["engine"] and fp["motifs"] and fp["topology"]       # measured from the 4.0 engine, not empty
    train = F.families(split="train")
    matches, nearest = {}, {}
    for n in train:
        fp = F.REGISTRY[n]["fingerprint"]
        d = {k: FP.distance(fp, rec["fingerprint"]) for k, rec in sealed.items()}
        k = min(d, key=d.get)
        nearest[n] = (k, d[k])
        rc = FP.legacy_recreation(fp)
        hit = sorted(set(rc["recreates"]) | {x for x, v in d.items() if v == 0.0})
        if hit:
            matches[n] = hit
    assert not set(F.FAMILIES) & set(heldout)
    # the check is not vacuous: a v5 world built to recreate a 4.0 held-out mechanism is caught
    probe = dict(F.REGISTRY[train[0]]["fingerprint"])
    target = sealed["scm_regime"]
    probe["motifs"] = list(target["motifs"])
    probe["action_interface"] = ["levers"]
    assert "scm_regime" in FP.legacy_recreation(probe)["recreates"]
    dmin = min(v for _, v in nearest.values())
    record("SP-0.9", {"v4_heldout_families_fingerprinted": len(sealed),
                      "v4_val_families": sum(1 for r in sealed.values() if r["split"] == "val"),
                      "v4_test_families": sum(1 for r in sealed.values() if r["split"] == "test"),
                      "v5_train_families_checked": len(train),
                      "v5_train_families_matching_v4_heldout": len(matches),
                      "min_distance_train_to_v4_heldout": float(dmin)},
           notes="4.0 val/test families fingerprinted from their engines in the v5 vocabulary; every registered v5 "
                 "training family compared by structural distance and mechanism recreation (same mechanism motif "
                 f"set through a lever-like interface). Matches: {matches}. Closest: "
                 f"{sorted(nearest.items(), key=lambda kv: kv[1][1])[:4]}.")
    assert not matches, matches
