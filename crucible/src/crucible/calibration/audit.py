"""Gate measurements for empirical calibration (Gate F, SP-5.5), computed from what is registered at call time.

Nothing here is a fixed list of E1's models: every directory under `docs/v5/calibration/models/` with a
`model_card.json` is examined, so models registered by other workstreams (E2's commercial-response models) count as
soon as they are merged. A model counts as *empirically fitted* only if

* its checkpoint files exist and match the SHA-256 recorded in its card, and `registry.load` reconstructs it;
* its provenance names at least one dataset that `datasets()` knows, every file of which has a SHA-256, and which
  is not synthetic;
* its held-out report exists and evaluates the model on at least one held-out split with data.

Field aliases (`domain` / `provenance.domain`, `identification` / `provenance.identification`) are accepted so a card
written by another workstream's tooling is read the same way.
"""
from __future__ import annotations

import json
import os
import re
import subprocess

from . import registry as R
from .datasets import datasets

HEX64 = re.compile(r"^[0-9a-f]{64}$")
RANDOMIZED = ("randomized_controlled_trial", "randomized_exposure", "randomized", "rct", "randomised")
COMMERCIAL_DOMAINS = ("advertising_response", "purchase_behavior", "commercial_response", "commercial_intervention",
                      "uplift", "marketing_response")
CALIBRATION_KEYS = ("ece", "brier", "pit_ks", "coverage_80", "coverage_90", "count_calibration_error",
                    "calibration_error", "calibration_in_the_large")
MIN_LIMITS_CHARS = 200


def _domain(card: dict) -> str:
    return card.get("domain") or card.get("provenance", {}).get("domain") or ""


def _identification(card: dict) -> str:
    return (card.get("provenance", {}).get("identification") or card.get("identification") or "").lower()


def _report(card: dict) -> dict | None:
    rp = card.get("report")
    if not rp:
        return None
    p = rp if os.path.isabs(rp) else os.path.join(R.CRUCIBLE_ROOT, rp)
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def dataset_ok(name: str, ds: dict) -> bool:
    e = ds.get(name)
    if not e or e.get("synthetic", False):
        return False
    files = e.get("files") or []
    return bool(files) and all(HEX64.match(str(f.get("sha256", ""))) for f in files)


def heldout_splits(rep: dict) -> list[str]:
    """Held-out splits of a report that contain data."""
    out = []
    for s in rep.get("heldout_splits", []):
        info = rep.get("splits", {}).get(s, {})
        if (info.get("rows") or 0) > 0 or (info.get("users") or 0) > 0:
            out.append(s)
    return out


def examine(name: str, ds: dict | None = None) -> dict:
    """Everything the gates need to know about one registered model."""
    ds = datasets() if ds is None else ds
    d = R.model_dir(name)
    rec = {"name": name, "problems": []}
    try:
        card = R.read_card(name)
    except Exception as e:                       # unreadable card
        rec["problems"].append(f"card unreadable: {e}")
        rec.update(card=None, qualifies=False, hash_mismatches=1, checkpoint=False)
        return rec
    rec["card"] = card
    bad = R.verify_files(card, d)
    rec["hash_mismatches"] = len(bad)
    files = card.get("files", {})
    rec["checkpoint"] = bool(files) and all(os.path.isfile(os.path.join(d, f)) and
                                            os.path.getsize(os.path.join(d, f)) <= R.MAX_CHECKPOINT_BYTES
                                            for f in files)
    if bad:
        rec["problems"].append(f"checkpoint hash mismatch: {bad}")
    loaded = False
    try:
        R.load(f"{name}@{card['version']}")
        loaded = True
    except Exception as e:
        rec["problems"].append(f"load failed: {e}")
    names = card.get("provenance", {}).get("datasets", [])
    real = [n for n in names if dataset_ok(n, ds)]
    if not real:
        rec["problems"].append(f"no checksummed, non-synthetic dataset among {names}")
    rep = _report(card)
    ho = heldout_splits(rep) if rep else []
    if not ho:
        rec["problems"].append("no held-out report with data")
    rec.update(domain=_domain(card), identification=_identification(card), datasets=names, real_datasets=real,
               report=rep, heldout=ho, loaded=loaded)
    rec["qualifies"] = loaded and not bad and bool(real) and bool(ho) and rec["domain"] in R.DOMAINS
    if rec["domain"] not in R.DOMAINS:
        rec["problems"].append(f"domain {rec['domain']!r} is not one of registry.DOMAINS")
    bc = (rep or {}).get("baseline_comparison") or {}
    sel = card.get("selection") or (rep or {}).get("selection") or {}
    rec["baseline_compared"] = bool(rep) and all(k in bc for k in ("kept_test", "baseline_test", "baseline")) and \
        bc.get("baseline") in (sel.get("baselines") or [bc.get("baseline")])
    cal = (rep or {}).get("calibration") or {}
    rec["calibration_reported"] = any(isinstance(cal.get(k), (int, float)) for k in CALIBRATION_KEYS)
    limits_card = card.get("limits") or ""
    limits_rep = (rep or {}).get("extrapolation_limits") or ""
    rec["limits_documented"] = len(limits_card) >= MIN_LIMITS_CHARS and len(limits_rep) >= MIN_LIMITS_CHARS
    return rec


def all_records() -> list[dict]:
    ds = datasets()
    return [examine(m.split("@")[0], ds) for m in R.list_models()]


def provenance_complete(e: dict) -> bool:
    p = e.get("provenance") or {}
    lic = e.get("license") or {}
    need = ("publisher", "source", "citation", "population", "platform", "period", "retrieved")
    return all(str(p.get(k) or "").strip() for k in need) and bool(lic.get("name")) and bool(lic.get("text_location"))


def gate_f_metrics(records: list[dict] | None = None) -> dict:
    recs = all_records() if records is None else records
    ds = datasets()
    q = [r for r in recs if r.get("qualifies")]
    domains_all = {r.get("domain") for r in recs if r.get("card")}
    domains_q = {r["domain"] for r in q}
    return {
        "registered_models": len(recs),
        "qualifying_models": len(q),
        "empirical_domains_fitted": len(domains_q),
        "content_consumption_fitted": any(r["domain"] == "content_selection" for r in q),
        "randomized_commercial_response_fitted": any(r["domain"] in COMMERCIAL_DOMAINS and
                                                     any(x in r["identification"] for x in RANDOMIZED) for r in q),
        "heldout_reports": sum(1 for r in q if r["heldout"]),
        "baseline_comparisons": sum(1 for r in q if r["baseline_compared"]),
        "calibration_metric_reports": sum(1 for r in q if r["calibration_reported"]),
        "checkpoints_saved": sum(1 for r in recs if r.get("checkpoint")),
        "checkpoint_hash_mismatches": sum(int(r.get("hash_mismatches", 0)) for r in recs),
        "datasets_with_checksums": sum(1 for n in ds if dataset_ok(n, ds)),
        "provenance_records": sum(1 for n, e in ds.items() if dataset_ok(n, ds) and provenance_complete(e)),
        "extrapolation_limit_sections": sum(1 for r in q if r["limits_documented"]),
        "synthetic_only_domains": len({d for d in domains_all if d and d not in domains_q}),
    }


# ---------------------------------------------------------------- SP-5.5: prespecified metrics, better model kept
def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=R.CRUCIBLE_ROOT, capture_output=True, text=True)


def _is_ancestor(a: str, b: str) -> bool:
    return _git("merge-base", "--is-ancestor", a, b).returncode == 0


def plan_check(card: dict) -> dict:
    """Is the card's evaluation plan committed, unchanged at the recorded hash, earlier than the checkpoint, and does
    it prespecify the model's primary and secondary metrics?"""
    import hashlib
    plan = card.get("provenance", {}).get("evaluation_plan") or {}
    out = {"plan": plan.get("path"), "ok": False, "problems": []}
    if not plan.get("path") or not plan.get("commit") or not plan.get("sha256"):
        out["problems"].append("card has no evaluation_plan {path, sha256, commit}")
        return out
    show = subprocess.run(["git", "show", f"{plan['commit']}:./{plan['path']}"], cwd=R.CRUCIBLE_ROOT,
                          capture_output=True)
    if show.returncode != 0:
        out["problems"].append(f"plan {plan['path']} not found at commit {plan['commit'][:9]}")
        return out
    if hashlib.sha256(show.stdout).hexdigest() != plan["sha256"]:
        out["problems"].append("plan content at the recorded commit does not match the recorded sha256")
    if not _is_ancestor(plan["commit"], "HEAD"):
        out["problems"].append("plan commit is not in HEAD's history")
    card_rel = os.path.relpath(os.path.join(R.model_dir(card["name"]), "model_card.json"), R.CRUCIBLE_ROOT)
    added = _git("log", "--diff-filter=A", "--format=%H", "--", card_rel).stdout.split()
    first_add = added[-1] if added else None
    out["card_first_commit"] = first_add
    if first_add:
        if first_add == plan["commit"] or not _is_ancestor(plan["commit"], first_add):
            out["problems"].append("the checkpoint was committed in or before the plan's commit")
    fit_commit = card.get("provenance", {}).get("fit", {}).get("commit")
    if fit_commit and not (_is_ancestor(plan["commit"], fit_commit)):
        out["problems"].append("the fit ran at a commit that does not contain the plan")
    try:
        spec = json.loads(show.stdout.decode("utf-8"))["models"][card["name"]]
    except Exception:
        out["problems"].append(f"plan does not specify model {card['name']}")
        return out
    sel = card.get("selection", {})
    if spec.get("primary_metric") != sel.get("primary_metric"):
        out["problems"].append(f"primary metric {sel.get('primary_metric')} differs from the plan's "
                               f"{spec.get('primary_metric')}")
    reported = set(card.get("metrics", {}))
    missing = [m for m in [spec.get("primary_metric")] + list(spec.get("secondary_metrics", [])) if m not in reported]
    if missing:
        out["problems"].append(f"prespecified metrics not reported: {missing}")
    planned = set(spec.get("candidates", {}))
    if planned and set(sel.get("candidates", [])) != planned:
        out["problems"].append(f"candidates {sorted(sel.get('candidates', []))} differ from the plan's {sorted(planned)}")
    out["ok"] = not out["problems"]
    return out


def selection_check(card: dict) -> dict:
    """Was a worse model kept? The kept model must follow the rule (validation argmin unless a baseline beat it on
    test) and must be no worse than any simple baseline on the primary test metric."""
    sel = card.get("selection") or {}
    out = {"ok": False, "problems": []}
    val, test, bases = sel.get("validation") or {}, sel.get("test") or {}, sel.get("baselines") or []
    kept = sel.get("kept")
    if not (val and test and bases and kept in test):
        out["problems"].append("selection record incomplete (validation, test, baselines, kept)")
        return out
    sign = 1 if sel.get("lower_is_better", True) else -1
    if any(sign * test[b] < sign * test[kept] - 1e-12 for b in bases if b in test):
        out["problems"].append(f"kept {kept} is worse on test than a simple baseline")
    best_val = min(val, key=lambda k: sign * val[k])
    if kept != best_val and sign * val[kept] > sign * val[best_val] + 1e-4 and kept not in bases:
        out["problems"].append(f"kept {kept} is not the validation-best candidate {best_val}")
    if card.get("metrics", {}).get(sel.get("primary_metric")) is not None and \
            abs(card["metrics"][sel["primary_metric"]] - test[kept]) > 1e-9:
        out["problems"].append("card metrics are not the kept candidate's test metrics")
    out["ok"] = not out["problems"]
    return out


def sp55_metrics(records: list[dict] | None = None) -> dict:
    recs = all_records() if records is None else records
    cards = [r["card"] for r in recs if r.get("card")]
    plans = {c["name"]: plan_check(c) for c in cards}
    sels = {c["name"]: selection_check(c) for c in cards}
    return {"models_checked": len(cards),
            "metrics_prespecified_before_fit": bool(cards) and all(p["ok"] for p in plans.values()),
            "models_without_prespecified_plan": sum(1 for p in plans.values() if not p["ok"]),
            "worse_model_kept": sum(1 for s in sels.values() if not s["ok"]),
            "learned_model_kept": sum(1 for c in cards if not c.get("selection", {}).get("kept_is_baseline", True)),
            "_plans": plans, "_selection": sels}
