"""The v5 fidelity register (MASTER_SPEC 5.4): which behavioural mechanisms exist, and whether each has its own
record stating its category, the population, platform and period of its evidence, its limits and whether a fitted
model is actually used in the world.

Records come from two places:

* the tables in `docs/v5/FIDELITY_REGISTER.md`. Every Markdown table whose header starts with ``id`` is read (the
  register is split into sections; a blank line or a heading between tables does not end the register). Columns:
  ``id | mechanism | module | category | evidence (dataset, period) | held-out result | limits | in-world use``.
  ``id`` is the mechanism's unique identifier (backticks optional); a record covers exactly the one mechanism whose
  id it carries, so one row can never vouch for several mechanisms of a module;
* the 55-entry Crucible 4.0 register `crucible.evaluation.fidelity.REGISTER`, whose four statuses map onto the v5
  categories (`V40_CATEGORY`); a 4.0 entry claims no empirical evidence, so its evidence population, platform and
  period are "none", and its limits are its caveat.

The *important behavioural mechanisms* are computed, never listed by hand:

1. ``model:<name>@<version>`` for every registered fitted model (the offline model itself);
2. every in-world mechanism a fitted model's card says it grounds (card field ``mechanisms`` or
   ``fidelity_mechanisms``); its record states what actually runs in the world, which is not the fitted model unless
   the in-world-use column says so and the claim verifies;
3. every 4.0 register mechanism whose kinds include ``behavioural``;
4. ``module:<name>`` for every kernel module that a v5 behaviour package registers (population, social, commerce,
   innovation, multiagent, civ, domains, semantic);
5. every decision model of the population (``pop.decision.<name>``, from `crucible.population.models`) and every
   parameter group of the social consumption engine (``feed.behaviour.<group>``, from
   `crucible.social.consumption.BUILTIN`); a parameter group that `FEED_GROUPS` does not know becomes its own
   mechanism, so a new behaviour parameter without a record fails;
6. every runtime decision model of `crucible.calibration.runtime` (its ``fidelity_id``);
7. every kernel module outside those packages whose source records population decisions (calls
   ``record_decisions``), e.g. `economy.work`;
8. every fidelity id that recorded decisions carry (decision provenance, D-15): the string literals given as
   ``fidelity_id`` in the source of the behaviour packages and of the modules of item 7, so a decision batch can
   never name a fidelity record that does not exist.

The in-world-use column is checked (`inworld_claims`): ``none ...`` makes no claim; ``opt-in: `<module>` ...`` and
``default: `<module>` ...`` claim that kernel module ``<module>`` consumes the fitted model the row names
(``name@version``). A claim verifies only when the module is registered, a source file of its package references the
model id, and - for ``default`` claims - a probe in `INWORLD_PROBES` builds a default world with that module and
observes the model being evaluated. An unverified claim is a register problem.
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
COLUMNS = ("id", "mechanism", "module", "category", "evidence", "heldout", "limits", "inworld")
# parameter-name prefixes of social.consumption.BUILTIN -> the behaviour group they belong to
FEED_GROUPS = {"open": "open", "outside": "open", "window": "open", "bounce": "watch", "hazard": "watch",
               "scroll": "watch", "like": "engagement", "share": "engagement", "follow": "engagement",
               "continue": "continuation", "gap": "return", "notice": "return", "sat": "state", "habit": "state",
               "taste": "state", "need": "state"}
MODEL_ID = re.compile(r"\b([a-z][a-z0-9_]*@\d+)\b")
# (module, model id) -> callable() -> set of model ids a default world with that module evaluated. No default
# in-world use of a fitted model exists yet (wave 2, D-15), so the table is empty and a "default:" claim fails.
INWORLD_PROBES: dict = {}


def _clean_id(s: str) -> str:
    return s.strip().strip("`").strip()


def v5_rows(path: str = REGISTER_MD) -> list[dict]:
    """Rows of every register table (a header whose first cell is ``id``, followed by a separator line), in file
    order. Blank lines and headings never end the register: rows continue until the next table header. Malformed rows
    (wrong number of cells) are returned with ``_problem`` set so that `register_problems` reports them rather than
    dropping them."""
    rows, in_table, header = [], False, []
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()

    def is_sep(cells):
        return bool(cells) and all(c and set(c) <= set("-: ") for c in cells)

    def cells_of(s):
        return [c.strip() for c in s.strip().strip("|").split("|")]

    for lineno, line in enumerate(lines, 1):
        s = line.strip()
        if not s.startswith("|"):
            continue                                  # blank lines and headings never end the register
        cells = cells_of(s)
        nxt = lines[lineno].strip() if lineno < len(lines) else ""
        if nxt.startswith("|") and is_sep(cells_of(nxt)) and not is_sep(cells):
            in_table, header = cells[0].lower() == "id", cells      # a table header: register table or another
            continue
        if is_sep(cells) or not in_table:
            continue
        row = dict(zip(COLUMNS, cells[:len(COLUMNS)]), source="FIDELITY_REGISTER.md", line=lineno)
        for c in COLUMNS:
            row.setdefault(c, "")
        row["id"] = _clean_id(row["id"])
        if len(cells) != len(COLUMNS) or len(header) != len(COLUMNS):
            row["_problem"] = f"line {lineno}: {len(cells)} cells, the register has {len(COLUMNS)} columns"
        rows.append(row)
    return rows


def v40_records() -> list[dict]:
    from ..evaluation.fidelity import REGISTER
    out = []
    for m in REGISTER:
        out.append({"id": m.id, "mechanism": m.id, "module": ", ".join(f"`{x}`" for x in m.modules),
                    "category": V40_CATEGORY.get(m.status, ""),
                    "evidence": "none: no empirical evidence in the repository (Crucible 4.0 register); "
                                "population, platform and period: none" if m.status not in ("calibrated",
                                                                                           "stylised-fact-checked")
                    else str(m.evidence), "heldout": "", "limits": m.caveat, "inworld": "none (Crucible 4.0 rule)",
                    "kinds": list(m.kinds), "source": "crucible.evaluation.fidelity.REGISTER"})
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
    if rec.get("_problem"):
        p.append(rec["_problem"])
    cat = rec.get("category", "").strip().lower()
    if cat not in CATEGORIES:
        p.append(f"category {rec.get('category')!r} is not one of {CATEGORIES}")
    if not rec.get("limits", "").strip():
        p.append("no limits stated")
    ev = rec.get("evidence", "").strip()
    if not ev:
        p.append("no evidence statement (write 'none' when there is none)")
    if not str(rec.get("inworld", "")).strip():
        p.append("no in-world-use statement (write 'none' when no fitted model is used in the world)")
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


def _modules():
    from .. import load_all_modules
    from ..core.kernel import MODULES
    load_all_modules()
    return MODULES


def decision_mechanisms() -> list[dict]:
    """Per-decision mechanisms of the behaviour engines (one record each, never one per module)."""
    out = []
    from ..population import models as PM
    for name in sorted(PM._decisions()):
        out.append({"id": f"pop.decision.{name}", "why": "population decision model (crucible.population.models)",
                    "match": "id"})
    from ..social.consumption import BUILTIN
    groups = set()
    for k in BUILTIN:
        if k in ("id", "version"):
            continue
        pre = k.split("_")[0]
        groups.add(FEED_GROUPS.get(pre, pre))
    for g in sorted(groups):
        out.append({"id": f"feed.behaviour.{g}", "why": "social consumption engine parameter group "
                    "(crucible.social.consumption.BUILTIN)", "match": "id"})
    from . import runtime
    for spec in runtime.specs().values():
        out.append({"id": spec["fidelity_id"], "why": f"runtime decision model {spec['id']}", "match": "id"})
    for fid, where in sorted(provenance_fidelity_ids().items()):
        out.append({"id": fid, "why": f"fidelity id carried by recorded decisions ({where})", "match": "id"})
    return out


# fidelity ids given to record_decisions / ensure_model: keyword, dict entry, or a (model, version, fidelity) tuple
_FID_PATTERNS = (re.compile(r"""fidelity_id\s*=\s*["']([^"'\s]+)["']"""),
                 re.compile(r"""["']fidelity_id["']\s*:\s*["']([^"'\s]+)["']"""),
                 re.compile(r"""_MODEL\s*=\s*\(\s*["'][^"']+["']\s*,\s*["'][^"']+["']\s*,\s*["']([^"'\s]+)["']\s*\)"""))


def _source_files(paths):
    for d in paths:
        if os.path.isfile(d):
            yield d
            continue
        for root, _, files in os.walk(d):
            for fn in sorted(files):
                if fn.endswith(".py"):
                    yield os.path.join(root, fn)


def provenance_fidelity_ids() -> dict[str, str]:
    """{fidelity id: first source file} for every literal fidelity id in the behaviour packages and the modules
    that record population decisions."""
    import importlib
    roots = []
    for pkg in BEHAVIOUR_PACKAGES:
        try:
            roots += list(importlib.import_module(f"crucible.{pkg}").__path__)
        except ImportError:
            continue
    for cls in _decision_recording_modules().values():
        roots.append(importlib.import_module(cls.__module__).__file__)
    out: dict[str, str] = {}
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for f in _source_files(roots):
        with open(f, encoding="utf-8") as fh:
            text = fh.read()
        for pat in _FID_PATTERNS:
            for m in pat.finditer(text):
                out.setdefault(m.group(1), os.path.relpath(f, base))
    return out


def _decision_recording_modules() -> dict:
    """Kernel modules outside BEHAVIOUR_PACKAGES whose source file records population decisions."""
    import importlib
    out = {}
    for mname, cls in sorted(_modules().items()):
        pkg = cls.__module__.split(".")
        if len(pkg) > 1 and pkg[0] == "crucible" and pkg[1] in BEHAVIOUR_PACKAGES:
            continue
        try:
            with open(importlib.import_module(cls.__module__).__file__, encoding="utf-8") as fh:
                if "record_decisions(" in fh.read():
                    out[mname] = cls
        except (OSError, TypeError):
            continue
    return out


def important_mechanisms() -> list[dict]:
    imp = []
    for mid in R.list_models():
        name = mid.split("@")[0]
        imp.append({"id": f"model:{mid}", "why": f"registered fitted model {mid}", "match": "id"})
        card = R.read_card(name)
        for mech in R.card_view(card)["mechanisms"]:
            imp.append({"id": mech, "why": f"in-world mechanism the card of {mid} says it grounds", "match": "id"})
    for r in v40_records():
        if "behavioural" in r["kinds"]:
            imp.append({"id": r["mechanism"], "why": "Crucible 4.0 behavioural mechanism", "match": "id"})
    modules = _modules()
    for mname, cls in sorted(modules.items()):
        pkg = cls.__module__.split(".")
        if len(pkg) > 1 and pkg[0] == "crucible" and pkg[1] in BEHAVIOUR_PACKAGES:
            imp.append({"id": f"module:{mname}", "why": f"v5 kernel module {cls.__module__}.{cls.__name__}",
                        "match": "module", "module": mname})
    for mname, cls in _decision_recording_modules().items():
        imp.append({"id": f"module:{mname}", "why": f"kernel module {cls.__module__}.{cls.__name__} records "
                    "population decisions", "match": "module", "module": mname})
    imp += decision_mechanisms()
    seen, out = set(), []
    for m in imp:
        if m["id"] not in seen:
            seen.add(m["id"])
            out.append(m)
    return out


def _module_names(cell: str) -> list[str]:
    return re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", cell)


def _package_references(cls, model_id: str) -> bool:
    """Does a source file of the module's behaviour package (crucible.<package>) reference the fitted model, by its
    id or by the id of a runtime decision model that serves it?"""
    import importlib
    from . import runtime
    needles = {model_id} | {sid for sid, sp in runtime.specs().items() if sp["model"] == model_id}
    pkg = importlib.import_module(".".join(cls.__module__.split(".")[:2]))
    for d in getattr(pkg, "__path__", []):
        for root, _, files in os.walk(d):
            for fn in files:
                if fn.endswith(".py"):
                    with open(os.path.join(root, fn), encoding="utf-8") as fh:
                        text = fh.read()
                    if any(n in text for n in needles):
                        return True
    return False


def inworld_claims(rows: list[dict] | None = None) -> dict:
    """Verify every in-world-use claim of the register. Returns {row id: {claim, kind, model, module, ok,
    problems}} for the rows that make a claim."""
    rows = v5_rows() if rows is None else rows
    modules = None
    registered = set(R.list_models())
    out = {}
    for r in rows:
        cell = str(r.get("inworld", "")).strip()
        low = cell.lower()
        if not cell or low.startswith("none"):
            continue
        kind = "default" if low.startswith("default:") else "opt-in" if low.startswith("opt-in:") else None
        rec = {"claim": cell, "kind": kind, "model": None, "module": None, "problems": []}
        out[r["id"]] = rec
        if kind is None:
            rec["problems"].append("an in-world-use statement must start with 'none', 'opt-in:' or 'default:'")
            rec["ok"] = False
            continue
        models = MODEL_ID.findall(cell)
        mods = _module_names(cell)
        rec["model"], rec["module"] = (models[0] if models else None), (mods[0] if mods else None)
        if not models:
            rec["problems"].append("the claim names no fitted model id (name@version)")
        if not mods:
            rec["problems"].append("the claim names no kernel module in backticks")
        if models and mods:
            if modules is None:
                modules = _modules()
            cls = modules.get(mods[0])
            if models[0] not in registered:
                rec["problems"].append(f"{models[0]} is not a registered fitted model")
            if cls is None:
                rec["problems"].append(f"kernel module {mods[0]!r} is not registered")
            elif not _package_references(cls, models[0]):
                rec["problems"].append(f"no source file of {cls.__module__}'s package references {models[0]}")
            if kind == "default":
                probe = INWORLD_PROBES.get((mods[0], models[0]))
                if probe is None:
                    rec["problems"].append("a default in-world use needs a probe in fidelity.INWORLD_PROBES that "
                                           "observes the model being evaluated in a default world")
                elif models[0] not in probe():
                    rec["problems"].append(f"the probe did not observe {models[0]} being evaluated")
        rec["ok"] = not rec["problems"]
    return out


def register_problems(rows: list[dict] | None = None, ds_keys: dict | None = None) -> dict:
    """Problems of the register itself: malformed or incomplete rows, duplicate ids, unverified in-world claims, and
    an in-world mechanism labelled empirically fitted while its in-world use is 'none'."""
    rows = v5_rows() if rows is None else rows
    ds_keys = _dataset_keys() if ds_keys is None else ds_keys
    probs: dict[str, list[str]] = {}
    seen: dict[str, int] = {}
    for r in rows:
        key = r["id"] or f"line {r.get('line')}"
        p = record_problems(r, ds_keys)
        if not r["id"]:
            p.append("row has no id")
        if r["id"] in seen:
            p.append(f"duplicate id (first at line {seen[r['id']]})")
        seen.setdefault(r["id"], r.get("line"))
        cat = r.get("category", "").strip().lower()
        inw = str(r.get("inworld", "")).strip().lower()
        if cat == "empirically fitted" and not r["id"].startswith(("model:", "runtime:")) and inw.startswith("none"):
            p.append("an in-world mechanism is labelled empirically fitted although no fitted model is used in the "
                     "world (in-world use: none)")
        if p:
            probs.setdefault(key, []).extend(p)
    for rid, c in inworld_claims(rows).items():
        if not c["ok"]:
            probs.setdefault(rid, []).extend(f"in-world claim: {x}" for x in c["problems"])
    return probs


def sp54_metrics() -> dict:
    rows = v5_rows()
    v40 = v40_records()
    ds_keys = _dataset_keys()
    by_id: dict[str, list[dict]] = {}
    for r in rows + v40:
        by_id.setdefault(r["id"], []).append(r)
    imp = important_mechanisms()
    missing, incomplete, ok = [], {}, []
    for m in imp:
        recs = by_id.get(m["id"], [])
        if not recs:
            missing.append(m["id"])
            continue
        if m["match"] == "module" and not any(m["module"] in _module_names(r["module"]) for r in recs):
            incomplete[m["id"]] = [f"the record's module column does not name `{m['module']}`"]
            continue
        probs = [record_problems(r, ds_keys) for r in recs]
        if any(not pr for pr in probs):
            ok.append(m["id"])
        else:
            incomplete[m["id"]] = probs[0]
    reg = register_problems(rows, ds_keys)
    claims = inworld_claims(rows)
    imp_ids = {m["id"] for m in imp}
    return {"important_mechanisms": len(imp), "with_complete_record": len(ok),
            "important_mechanisms_without_record": len(missing) + len(incomplete),
            "v5_rows": len(rows), "v5_rows_with_problems": len(reg),
            "empirically_fitted_rows": sum(1 for r in rows if r["category"].lower() == "empirically fitted"),
            "fitted_models_registered": sum(1 for m in imp if m["id"].startswith("model:")),
            "inworld_claims": len(claims), "inworld_claims_verified": sum(1 for c in claims.values() if c["ok"]),
            "inworld_claims_unverified": sum(1 for c in claims.values() if not c["ok"]),
            "rows_without_important_mechanism": sum(1 for r in rows if r["id"] not in imp_ids),
            "_missing": missing, "_incomplete": incomplete, "_row_problems": reg, "_claims": claims}
