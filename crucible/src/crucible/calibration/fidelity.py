"""The v5 fidelity register (MASTER_SPEC 5.4): which behavioural mechanisms exist, and whether each has a record that
states its category, the population, platform and period of its evidence, and its limits.

Records come from two places:

* the table in `docs/v5/FIDELITY_REGISTER.md` (v5 mechanisms; one row per mechanism, columns
  ``mechanism | module | category | evidence (dataset, period) | held-out result | limits``);
* the 55-entry Crucible 4.0 register `crucible.evaluation.fidelity.REGISTER`, whose four statuses map onto the v5
  categories (`V40_CATEGORY`); a 4.0 entry claims no empirical evidence, so its evidence population, platform and
  period are "none", and its limits are its caveat.

The *important behavioural mechanisms* are computed, not listed by hand:

1. every mechanism a registered fitted model grounds (`registry.card_view(card)["mechanisms"]`: the card field
   ``mechanisms``, or ``fidelity_mechanisms`` in loader-schema cards);
2. every 4.0 register mechanism whose kinds include ``behavioural``;
3. every kernel module that a v5 behaviour package registers (population, social, commerce, innovation, multiagent,
   civ, domains, semantic): it needs at least one register row naming it in the ``module`` column.

So a workstream that adds a behavioural module or a fitted model without a fidelity record fails SP-5.4.
"""
from __future__ import annotations

import os
import re

from . import registry as R
from .datasets import datasets

REGISTER_MD = os.path.join(R.CRUCIBLE_ROOT, "docs", "v5", "FIDELITY_REGISTER.md")
CATEGORIES = ("empirically fitted", "empirically validated", "inspired by research", "plausible, unvalidated",
              "deliberate simplification", "known unrealistic")
EMPIRICAL = ("empirically fitted", "empirically validated")
V40_CATEGORY = {"calibrated": "empirically fitted", "stylised-fact-checked": "empirically validated",
                "plausible-hypothesis": "plausible, unvalidated", "known-unrealistic": "known unrealistic"}
BEHAVIOUR_PACKAGES = ("population", "social", "commerce", "innovation", "multiagent", "civ", "domains", "semantic")
COLUMNS = ("mechanism", "module", "category", "evidence", "heldout", "limits")


def v5_rows(path: str = REGISTER_MD) -> list[dict]:
    """Rows of the register table (the first Markdown table whose header starts with 'mechanism')."""
    rows, in_table = [], False
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s.startswith("|"):
                if in_table:
                    break
                continue
            cells = [c.strip() for c in s.strip("|").split("|")]
            if not in_table:
                if cells and cells[0].lower() == "mechanism":
                    in_table = True
                continue
            if all(set(c) <= set("-: ") for c in cells):
                continue
            if len(cells) < len(COLUMNS):
                cells += [""] * (len(COLUMNS) - len(cells))
            rows.append(dict(zip(COLUMNS, cells[:len(COLUMNS)]), source="FIDELITY_REGISTER.md"))
    return rows


def v40_records() -> list[dict]:
    from ..evaluation.fidelity import REGISTER
    out = []
    for m in REGISTER:
        out.append({"mechanism": m.id, "module": ", ".join(m.modules), "category": V40_CATEGORY.get(m.status, ""),
                    "evidence": "none: no empirical evidence in the repository (Crucible 4.0 register); "
                                "population, platform and period: none" if m.status not in ("calibrated",
                                                                                           "stylised-fact-checked")
                    else str(m.evidence), "heldout": "", "limits": m.caveat, "kinds": list(m.kinds),
                    "source": "crucible.evaluation.fidelity.REGISTER"})
    return out


def _dataset_keys() -> dict[str, dict]:
    ds = datasets()
    keys = {}
    for name, e in ds.items():
        keys[name.lower()] = e
        keys[str(e.get("title", "")).lower()] = e
    return {k: v for k, v in keys.items() if k}


def record_problems(rec: dict, ds_keys: dict | None = None) -> list[str]:
    """Why a record is not a complete fidelity record ([] when it is)."""
    ds_keys = _dataset_keys() if ds_keys is None else ds_keys
    p = []
    cat = rec.get("category", "").strip().lower()
    if cat not in CATEGORIES:
        p.append(f"category {rec.get('category')!r} is not one of {CATEGORIES}")
    if not rec.get("limits", "").strip():
        p.append("no limits stated")
    ev = rec.get("evidence", "").strip()
    if not ev:
        p.append("no evidence statement (write 'none' when there is none)")
    if cat in EMPIRICAL:
        evl = ev.lower()
        hit = [e for k, e in ds_keys.items() if k in evl]
        if not hit:
            p.append("empirical category but the evidence names no registered dataset")
        else:
            prov = hit[0].get("provenance", {})
            if not (prov.get("population") and prov.get("platform") and prov.get("period")):
                p.append("the cited dataset lacks population, platform or period")
        if not re.search(r"(19|20)\d\d", ev):
            p.append("evidence states no period")
        if not rec.get("heldout", "").strip():
            p.append("empirical category but no held-out result")
    return p


def important_mechanisms() -> list[dict]:
    imp = []
    for name in (m.split("@")[0] for m in R.list_models()):
        card = R.read_card(name)
        for mech in R.card_view(card)["mechanisms"]:
            imp.append({"id": mech, "why": f"grounded by fitted model {name}", "match": "mechanism"})
    for r in v40_records():
        if "behavioural" in r["kinds"]:
            imp.append({"id": r["mechanism"], "why": "Crucible 4.0 behavioural mechanism", "match": "mechanism"})
    from .. import load_all_modules
    from ..core.kernel import MODULES
    load_all_modules()
    for mname, cls in sorted(MODULES.items()):
        pkg = cls.__module__.split(".")
        if len(pkg) > 1 and pkg[0] == "crucible" and pkg[1] in BEHAVIOUR_PACKAGES:
            imp.append({"id": f"module:{mname}", "why": f"v5 kernel module {cls.__module__}.{cls.__name__}",
                        "match": "module", "module": mname})
    seen, out = set(), []
    for m in imp:
        if m["id"] not in seen:
            seen.add(m["id"])
            out.append(m)
    return out


def sp54_metrics() -> dict:
    rows = v5_rows()
    v40 = v40_records()
    ds_keys = _dataset_keys()
    by_mech: dict[str, list[dict]] = {}
    for r in rows + v40:
        by_mech.setdefault(r["mechanism"], []).append(r)
    imp = important_mechanisms()
    missing, incomplete, ok = [], {}, []
    for m in imp:
        if m["match"] == "mechanism":
            recs = by_mech.get(m["id"], [])
        else:
            recs = [r for r in rows if re.search(rf"\b{re.escape(m['module'])}\b", r["module"])]
        if not recs:
            missing.append(m["id"])
            continue
        probs = [record_problems(r, ds_keys) for r in recs]
        if any(not pr for pr in probs):
            ok.append(m["id"])
        else:
            incomplete[m["id"]] = probs[0]
    row_problems = {r["mechanism"]: record_problems(r, ds_keys) for r in rows if record_problems(r, ds_keys)}
    return {"important_mechanisms": len(imp), "with_complete_record": len(ok),
            "important_mechanisms_without_record": len(missing) + len(incomplete),
            "v5_rows": len(rows), "v5_rows_with_problems": len(row_problems),
            "empirically_fitted_rows": sum(1 for r in rows if r["category"].lower() == "empirically fitted"),
            "_missing": missing, "_incomplete": incomplete, "_row_problems": row_problems}
