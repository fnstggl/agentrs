"""Measurements of the Part 5 requirements SP-5.1 (data governance and identification) and SP-5.3 (comparison with
the simulator's rules, selection on stability and cost, behaviour coverage), read from what is registered: the
dataset registry, the model cards and held-out reports, the E3 reviews and the derived calibration data on disk.
Owner: workstream E. Nothing here fits or changes anything.

    from crucible.calibration import audit_sp5
    audit_sp5.sp51_metrics(); audit_sp5.sp53_metrics()
"""
from __future__ import annotations

import glob
import json
import math
import os
import re

import numpy as np

from . import audit
from . import registry as R
from .datasets import data_root, datasets

# raw GitHub account / repository ids are below 2^33 (about 2.0e8 actors and 9e8 repositories in 2024); the keyed
# 63-bit pseudonyms of extractor 2 fall below 2^33 with probability 2^-30 per value
RAW_ID_BOUND = 2 ** 33
# the Part 5.3 behaviours that must each be fitted or their absence justified (canonical registry domains)
PART_5_3_BEHAVIOURS = {"attention allocation": "attention_allocation", "repeat consumption": "repeat_consumption",
                       "purchase": "purchase_behavior", "retention": "product_retention",
                       "community participation": "community_participation", "social diffusion": "social_diffusion"}
# a behaviour without a qualifying model counts only with a written justification here (none needed today)
JUSTIFIED_ABSENCE: dict[str, str] = {}


# ====================================================================== SP-5.1
def _finite_interval(v) -> bool:
    return (isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, (int, float)) and math.isfinite(x)
                                                                  for x in v) and v[1] > v[0])


USER_HOLDOUT = re.compile(r"\b(test|held[- ]out|new)\b[^.;]{0,40}\b(users?|customers?|developers?|repositor(y|ies)|"
                          r"repos?|products?)\b|\busers? \(held out\)|\b\d[\d,]* users\b", re.I)
TIME_HOLDOUT = re.compile(r"\b(19|20)\d\d-\d\d|\b\d\d-\d\d\b|\bhours? \d|\bweeks?\b|\blater\b|time-based|"
                          r"\bD\d -> D\d\b|\bday \d", re.I)


def holdout_kinds(rec: dict) -> list[str]:
    """Kinds of holdout ('user', 'time') a model's held-out report documents for its held-out splits."""
    raw = rec.get("report_raw") or {}
    sp = raw.get("splits") or raw.get("split") or {}
    kinds = set()
    texts = []
    for k in rec.get("heldout") or []:
        v = sp.get(k) if isinstance(sp, dict) else None
        if isinstance(v, dict):
            texts.append(str(v.get("definition") or ""))
        elif isinstance(v, (str, list)):
            texts.append(str(v))
    if isinstance(sp, dict) and "permutation" in str(sp.get("method", "")) and "users" in json.dumps(raw)[:200000]:
        kinds.add("user")             # rows are users (Criteo): a seeded permutation of users is a user-level holdout
    if isinstance(sp, dict) and isinstance(sp.get("test"), list) and len(sp["test"]) == 2:
        kinds.add("time")             # an [hour, hour] window after the training hours (GH Archive software models)
    for t in texts:
        if USER_HOLDOUT.search(t):
            kinds.add("user")
        if TIME_HOLDOUT.search(t):
            kinds.add("time")
    return sorted(kinds)


def gharchive_raw_id_values(dest: str | None = None) -> dict:
    """Identifier values in the derived GH Archive data that could be raw GitHub ids.

    * event tables (`extracted/<hour>.npz`, columns actor / repo): extractor 2 stores keyed 63-bit pseudonyms, so a
      value with |v| < 2^33 (the range of raw GitHub ids) counts as raw-like;
    * feature tables (`tables*.npz`, columns ending in actor / repo): the software fit stores dense codes (the rank
      of a pseudonym among the window's distinct pseudonyms), so a value counts as raw-like when it is not a valid
      code (< 0 or >= the number of distinct pseudonyms in the event tables);
    * an hour whose metadata records an extractor version other than 2 (raw ids) counts once."""
    root = os.path.join(dest or data_root(), "gharchive_2024-10-07_96h")
    out = {"files": 0, "values": 0, "raw_like_values": 0, "raw_extractor_files": 0, "present": os.path.isdir(root)}
    if not out["present"]:
        return out
    for meta in sorted(glob.glob(os.path.join(root, "extracted", "*.meta.json"))):
        with open(meta, encoding="utf-8") as fh:
            if str(json.load(fh).get("extractor_version")) != "2":
                out["raw_extractor_files"] += 1
    distinct = {"actor": [], "repo": []}
    for p in sorted(glob.glob(os.path.join(root, "extracted", "*.npz"))):
        with np.load(p, allow_pickle=False) as z:
            out["files"] += 1
            for k in ("actor", "repo"):
                v = z[k].astype(np.int64)
                out["values"] += int(v.size)
                out["raw_like_values"] += int(np.count_nonzero(np.abs(v) < RAW_ID_BOUND))
                distinct[k].append(np.unique(v))
    n_distinct = {k: (int(np.unique(np.concatenate(v)).size) if v else 0) for k, v in distinct.items()}
    out["distinct_pseudonyms"] = n_distinct
    for p in sorted(glob.glob(os.path.join(root, "tables*.npz"))):
        with np.load(p, allow_pickle=False) as z:
            out["files"] += 1
            for k in z.files:
                kind = k.split("/")[-1]
                if kind in ("actor", "repo"):
                    v = z[k].astype(np.int64)
                    out["values"] += int(v.size)
                    out["raw_like_values"] += int(np.count_nonzero((v < 0) | (v >= n_distinct[kind])))
    return out


def co_located_keys(dest: str | None = None) -> dict:
    """Pseudonym keys stored with the data they re-identify: a key file inside the data root (any dataset directory)
    makes every value pseudonymised under it re-identifiable by whoever holds that directory and the public source.
    Returns {"files": [...], "values": re-identifiable pseudonymised values}."""
    from . import gharchive as G, retail
    root = dest or data_root()
    files = sorted(set(glob.glob(os.path.join(root, "*", G.SALT_FILE)) +
                       glob.glob(os.path.join(root, "**", "*pseudonym_key*"), recursive=True)))
    out = {"files": [os.path.relpath(f, root) for f in files], "values": 0}
    for f in files:
        ds = os.path.relpath(f, root).split(os.sep)[0]
        if ds == "gharchive_2024-10-07_96h":
            out["values"] += int(gharchive_raw_id_values(root)["values"])
        elif ds == "online_retail_ii" and os.path.exists(retail.table_path(os.path.join(root, ds))):
            with np.load(retail.table_path(os.path.join(root, ds)), allow_pickle=False) as z:
                out["values"] += int(np.count_nonzero(z["customer"] != -1))
        else:
            out["values"] += 1
    return out


def sp51_metrics(records: list[dict] | None = None) -> dict:
    recs = audit.all_records() if records is None else records
    ds = datasets()
    lic = {}
    for name, e in ds.items():
        L = e.get("license") or {}
        lic[name] = bool(L.get("name")) and bool(L.get("url") or L.get("text_location") or L.get("terms"))
    gh = gharchive_raw_id_values()
    keys = co_located_keys()
    cited = {n for r in recs for n in (r.get("datasets") or [])}
    gh_missing = "gharchive_2024-10-07_96h" in cited and not gh["present"]
    q = [r for r in recs if r.get("qualifies")]
    unc = {r["name"]: any(_finite_interval(((r.get("report") or {}).get("baseline_comparison") or {}).get(k))
                          for k in ("ci95", "kept_ci95")) for r in q}
    hold = {r["name"]: holdout_kinds(r) for r in q}
    ident = {r["name"]: r.get("identification") in R.IDENTIFICATION and bool(
        (r.get("card") or {}).get("identification") or (r.get("report_raw") or {}).get("identification"))
        for r in recs if r.get("card")}
    return {"datasets_registered": len(ds), "datasets_with_licence_record": sum(lic.values()),
            "reidentifiable_ids_in_calibration_data": gh["raw_like_values"] + gh["raw_extractor_files"]
            + (1 if gh_missing else 0) + keys["values"],
            "pseudonym_keys_with_the_data": len(keys["files"]),
            "gharchive_id_values_checked": gh["values"], "qualifying_models": len(q),
            "models_with_uncertainty": sum(unc.values()),
            "models_with_time_or_user_holdout": sum(1 for v in hold.values() if v),
            "models_with_time_and_user_holdout": sum(1 for v in hold.values() if set(v) == {"time", "user"}),
            "effects_without_identification_label": sum(1 for v in ident.values() if not v),
            "_licence": lic, "_gharchive": gh, "_keys": keys, "_uncertainty": unc, "_holdout": hold, "_identification": ident}


# ====================================================================== SP-5.3
def _finite_numbers(o) -> list[float]:
    out = []
    if isinstance(o, bool):
        return out
    if isinstance(o, (int, float)):
        return [float(o)] if math.isfinite(o) else []
    if isinstance(o, dict):
        for v in o.values():
            out += _finite_numbers(v)
    elif isinstance(o, list):
        for v in o:
            out += _finite_numbers(v)
    return out


def rule_comparison(rec: dict) -> dict | None:
    """The comparison of a model with the simulator's rule-based behaviour on the same held-out rows: from its E3
    review (contender 'S') or its report's ``rule_comparison`` (E4, E5); None when there is no measured one."""
    name = rec["name"]
    p = os.path.join(audit.REVIEWS_DIR, f"{name}.json")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as fh:
            rv = json.load(fh)
        S = (rv.get("contenders") or {}).get("S")
        if S and _finite_numbers(S.get("primary")):
            return {"source": os.path.relpath(p, R.CRUCIBLE_ROOT), "rule": S.get("primary"),
                    "fitted": ((rv.get("contenders") or {}).get("K") or {}).get("primary")}
    rc = (rec.get("report_raw") or {}).get("rule_comparison") or {}
    if isinstance(rc, dict) and _finite_numbers(rc.get("test")):
        return {"source": rec["card"].get("report"), "rule": rc.get("test")}
    return None


def stability_and_cost(rec: dict) -> dict | None:
    """Was the kept model selected with a measured long-horizon (multi-step) criterion AND a compute criterion?"""
    name = rec["name"]
    p = os.path.join(audit.REVIEWS_DIR, f"{name}.json")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as fh:
            rv = json.load(fh)
        cs = rv.get("contenders") or {}
        lh = [c.get("long_horizon") for c in cs.values()]
        cu = [c.get("compute_us") for c in cs.values()]
        if _finite_numbers(lh) and _finite_numbers(cu):
            return {"source": os.path.relpath(p, R.CRUCIBLE_ROOT), "rule": rv.get("ordered_by_primary")}
    sel = (rec.get("card") or {}).get("selection") or {}
    if _finite_numbers(sel.get("long_horizon_validation")) and _finite_numbers(sel.get("compute_us_per_row")):
        return {"source": "card selection (" + str(sel.get("rule_version")) + ")"}
    return None


def sp53_metrics(records: list[dict] | None = None) -> dict:
    recs = audit.all_records() if records is None else records
    q = [r for r in recs if r.get("qualifies")]
    rules = {r["name"]: rule_comparison(r) for r in q}
    stab = {r["name"]: stability_and_cost(r) for r in q}
    domains = {}
    for label, dom in PART_5_3_BEHAVIOURS.items():
        fitted = sorted(r["name"] for r in q if r.get("domain") == dom)
        domains[label] = {"domain": dom, "fitted": fitted, "justified_absence": JUSTIFIED_ABSENCE.get(label)}
    return {"qualifying_models": len(q),
            "rule_based_comparisons": sum(1 for v in rules.values() if v),
            "selection_reports_with_stability_and_cost": sum(1 for v in stab.values() if v),
            "behaviour_domains_fitted_or_justified": sum(1 for v in domains.values()
                                                         if v["fitted"] or v["justified_absence"]),
            "_rules": rules, "_stability": stab, "_domains": domains}
