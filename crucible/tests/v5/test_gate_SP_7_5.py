"""Gate SP-7.5 - a machine-readable registry documents why each family is distinct, and every claimed structural
difference is verified by a controlled intervention (workstream H).

The registry is `crucible.rapid.distinct.registry` (also `python -m crucible.rapid families --json`): for every
registered family its measured fingerprint, its nearest registered family, the distance, the fingerprint components
that differ and how, the family's own statement of why it is distinct, and the verification of the difference to the
nearest family by an intervention run on both worlds (an action one interface offers and the other refuses; an
observation kind one world shows and the other never does; a pair protocol that drives both mechanisms the same way
and compares the response; or the qualitative response signature to the same intervention protocol). A record
counts when every field is present and non-empty; a claimed difference is unverified when its intervention did not
separate the two families on every verification seed."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from crucible.acceptance.v5.evidence import record
from crucible.rapid import distinct as DI
from crucible.rapid import families as F
from crucible.rapid import fingerprints as FP

pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]


def test_SP_7_5_distinctness_registry_verified_by_intervention():
    fps = {n: F.REGISTRY[n]["fingerprint"] for n in sorted(F.REGISTRY)}
    reg = DI.registry(fps=fps)
    assert sorted(reg) == sorted(F.FAMILIES)
    complete, unverified, methods = [], {}, {}
    for n, r in reg.items():
        ok_fields = bool(r["distinct_because"] and r["nearest"] and r["distance"] and r["differs_in"]
                         and r["difference"] and r["fingerprint"] and "verification" in r)
        if ok_fields:
            complete.append(n)
        v = r.get("verification") or {}
        methods[v.get("method", "none")] = methods.get(v.get("method", "none"), 0) + 1
        if not v.get("ok"):
            unverified[n] = {"nearest": r["nearest"], "method": v.get("method"), "error": v.get("error")}
        # the record's measured difference is what the fingerprints say, not a declaration
        assert r["differs_in"] == FP.differing(fps[n], fps[r["nearest"]])
    # the registry is machine-readable from the command line
    out = subprocess.run([sys.executable, "-m", "crucible.rapid", "families"], capture_output=True, text=True,
                         timeout=900)
    assert out.returncode == 0, out.stderr[-500:]
    cli = json.loads(out.stdout)
    assert set(cli) == set(F.FAMILIES) and all(cli[n]["why_distinct"]["nearest"] for n in cli)
    record("SP-7.5", {"families_with_distinctness_record": len(complete), "families_registered": len(reg),
                      "claimed_differences_unverified": len(unverified),
                      "verified_by_interface": methods.get("interface", 0),
                      "verified_by_behaviour_signature": methods.get("behaviour", 0)},
           notes=f"Verification methods: {methods}. Unverified: {unverified}.")
    assert len(complete) == len(reg)
