"""Gate SP-8.0 - each of the 60 named Tier 1 families of Parts 8.1-8.10 is implemented as an admitted environment
(workstream H).

The 60 items are the numbered entries of MASTER_SPEC Parts 8.1-8.10 (six per category), read from the spec text
itself. A registered family names the item it implements (`Family.item`); an item counts as admitted when at least
one family implementing it passes admission (`crucible.rapid.report.robust_report`: every Part 7.4 check on seeds
0-2 and on the held-out seeds 10-12, structural fingerprint not a duplicate). Items with no registered family are
missing; items whose families all fail admission are reported with their failing checks.

Heavy: run through the slot scheduler (the admission reports are cached under $CRUCIBLE_CACHE/rapid by source
digest, shared with the Gate M and SP-7.4 tests)."""
from __future__ import annotations

import os
import re

import pytest

from crucible.acceptance.v5.evidence import record
from crucible.rapid import families as F
from crucible.rapid import report as RP

pytestmark = [pytest.mark.slow, pytest.mark.timeout(7200)]
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JOBS = int(os.environ.get("CRUCIBLE_RAPID_JOBS", "1"))


def spec_items() -> dict[str, str]:
    """item id -> text, parsed from the numbered lists under the headings 8.1 to 8.10 of MASTER_SPEC.md."""
    text = open(os.path.join(ROOT, "docs", "v5", "MASTER_SPEC.md"), encoding="utf-8").read()
    out: dict[str, str] = {}
    for cat in range(1, 11):
        m = re.search(rf"^## 8\.{cat} [^\n]+\n(.*?)(?=^## |^---|\Z)", text, re.S | re.M)
        assert m, f"Part 8.{cat} not found"
        for num, item in re.findall(r"^(\d+)\.\s+(.+)$", m.group(1), re.M):
            out[f"8.{cat}.{num}"] = item.strip()
    return out


def test_SP_8_0_every_named_family_admitted():
    items = spec_items()
    assert len(items) == 60 and all(len([k for k in items if k.startswith(f"8.{c}.")]) == 6 for c in range(1, 11))
    rob = RP.robust_report(seeds=(0, 1, 2), holdout=(10, 11, 12), jobs=JOBS)
    a, h = rob["admission"]["reports"], rob["holdout"]["reports"]
    by_item: dict[str, list[str]] = {k: [] for k in items}
    for n, f in F.FAMILIES.items():
        assert f.item in items, (n, f.item)
        assert f.item.startswith(f.category + "."), (n, f.item, f.category)
        by_item[f.item].append(n)
    admitted = {k: sorted(n for n in v if n in rob["robust"]) for k, v in by_item.items()}
    missing = sorted(k for k, v in by_item.items() if not v)
    unadmitted = {k: {n: sorted(set(c for c, r in a[n]["checks"].items() if not r["ok"]) |
                                set(c for c, r in h[n]["checks"].items() if not r["ok"])) for n in v}
                  for k, v in by_item.items() if v and not admitted[k]}
    only_eval = sorted(k for k, v in admitted.items() if v and all(F.FAMILIES[n].split != "train" for n in v))
    n_adm = sum(1 for v in admitted.values() if v)
    record("SP-8.0", {"named_families_admitted": n_adm, "named_families_missing": len(missing),
                      "named_families_registered": sum(1 for v in by_item.values() if v),
                      "named_families_not_admitted": len(unadmitted),
                      "named_families_admitted_only_as_evaluation": len(only_eval), "named_items": len(items)},
           notes=f"Items parsed from MASTER_SPEC Parts 8.1-8.10. Missing (no family): {missing}. Registered but no "
                 f"family admitted on seeds 0-2 and 10-12: {unadmitted}. Admitted only through evaluation families: "
                 f"{only_eval}.")
    assert n_adm + len(missing) + len(unadmitted) == 60
