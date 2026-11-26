"""Gate measurements for empirical calibration (Gate F, SP-5.5), computed from what is registered at call time.

Nothing here is a fixed list of models: every directory under `docs/v5/calibration/models/` with a `model_card.json`
is examined, whichever of the registry's two card schemas it uses (`registry.SCHEMA`, `registry.LOADER_SCHEMA`).
Cards are read through `registry.card_view` and held-out reports through `report_view`, so the models of both fitting
toolchains are measured by the same rules. A model counts as *empirically fitted* only if

* its checkpoint files exist and match the SHA-256 recorded in its card, and `registry.load` reconstructs it;
* its card names at least one dataset that `datasets()` knows, every file of which has a SHA-256, and which is not
  synthetic;
* its held-out report exists and evaluates the model on at least one held-out split with data;
* its (canonical) domain is one of `registry.DOMAINS`.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
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
LOADER_REPORT_SCHEMA = "crucible.v5.calibration.report/1"     # reports written by fitmeta_e2


def _domain(card: dict) -> str:
    return R.card_view(card)["domain"]


def _identification(card: dict) -> str:
    return R.card_view(card)["identification"]


def _report(card: dict) -> dict | None:
    rp = card.get("report")
    if not rp:
        return None
    p = rp if os.path.isabs(rp) else os.path.join(R.CRUCIBLE_ROOT, rp)
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _kept_rows(raw: dict, split: str, kept_metrics: dict) -> int:
    rows = kept_metrics.get("n")
    if rows is None:                                  # per-arm models report n per pooled outcome
        pooled = kept_metrics.get("pooled") or {}
        rows = next((x.get("n") for x in pooled.values() if isinstance(x, dict) and x.get("n")), None)
    if rows is None:
        rows = (raw.get("counts") or {}).get(split) or ((raw.get("split") or {}).get("counts") or {}).get(split)
    return int(rows or 0)


def report_view(card: dict, raw: dict | None) -> dict | None:
    """The report in the audit's layout: ``heldout_splits``, ``splits`` {split: {rows, users}},
    ``baseline_comparison`` {metric, kept, baseline, kept_test, baseline_test, difference, ci95}, ``calibration``
    (the kept model's held-out calibration measures) and ``extrapolation_limits`` (text).

    Reports written by fitlib already have this layout. A `LOADER_REPORT_SCHEMA` report is re-indexed: every number
    is the report's or the card's own, except ``difference`` = kept_test - baseline_test; ``ci95`` is the report's
    paired-bootstrap interval of the same difference."""
    if raw is None:
        return None
    if raw.get("schema") != LOADER_REPORT_SCHEMA:
        return raw
    sel = R.card_view(card)["selection"]
    kept = sel["kept"]
    held = raw.get("heldout") or {}
    splits = {s: {"rows": _kept_rows(raw, s, (by or {}).get(kept) or {})} for s, by in held.items()}
    base = sel["baselines"][0] if sel["baselines"] else None
    bc = {}
    if kept in sel["test"] and base in sel["test"]:
        comp = next((c for c in sel["comparisons"] if c.get("other") == base), {})
        km = comp.get("kept_minus_other") or {}
        bc = {"metric": sel["primary_metric"], "kept": kept, "baseline": base, "kept_test": sel["test"][kept],
              "baseline_test": sel["test"][base], "difference": sel["test"][kept] - sel["test"][base],
              "ci95": km.get("ci95"), "bootstrap": {"resamples": km.get("n_boot"), "paired": True}}
    lim = raw.get("extrapolation_limits") or card.get("extrapolation_limits") or ""
    return {"schema": raw.get("schema"), "model": raw.get("model"),
            "heldout_splits": [s for s in held if s != "validation"], "splits": splits, "baseline_comparison": bc,
            "calibration": dict((((raw.get("calibration_metrics") or {}).get(kept) or {}).get("test")) or {}),
            "extrapolation_limits": " ".join(lim) if isinstance(lim, list) else str(lim)}


def _is_calibration_key(k: str) -> bool:
    return k in CALIBRATION_KEYS or any(k.endswith("_" + c) for c in CALIBRATION_KEYS)


def _is_calibration_table(v) -> bool:
    return isinstance(v, list) and len(v) >= 2 and all(
        isinstance(b, dict) and ("mean_pred" in b or "mean_predicted" in b) and ("mean_obs" in b or "mean_observed" in b)
        for b in v)


def calibration_values(rep: dict | None) -> dict[str, float]:
    """Scalar held-out calibration measures of a report view (ECE, Brier, PIT KS, coverage, ... per outcome)."""
    cal = (rep or {}).get("calibration") or {}
    return {k: float(v) for k, v in cal.items()
            if _is_calibration_key(k) and isinstance(v, (int, float)) and not isinstance(v, bool)
            and v == v and abs(v) != float("inf")}            # NaN or infinite values are not calibration reports


def calibration_tables(rep: dict | None) -> dict[str, int]:
    """Reliability tables (bins of mean predicted vs mean observed) of a report view: {key: number of bins}."""
    cal = (rep or {}).get("calibration") or {}
    return {k: len(v) for k, v in cal.items() if _is_calibration_table(v)}


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


def examine(name: str, ds: dict | None = None, models_dir: str | None = None) -> dict:
    """Everything the gates need to know about one registered model."""
    ds = datasets() if ds is None else ds
    d = R.model_dir(name, models_dir)
    rec = {"name": name, "problems": [], "models_dir": models_dir}
    try:
        card = R.read_card(name, models_dir)
        view = R.card_view(card)
    except Exception as e:                       # unreadable card or unknown schema
        rec["problems"].append(f"card unreadable: {e}")
        rec.update(card=None, view=None, qualifies=False, hash_mismatches=1, checkpoint=False)
        return rec
    rec["card"] = card
    rec["view"] = view
    bad = R.verify_files(card, d)
    ip = R.index_problems(name, models_dir)
    if ip:
        rec["problems"].append(f"registry index: {ip}")
    rec["hash_mismatches"] = len(bad) + (1 if ip else 0)
    files = view["files"]
    rec["checkpoint"] = bool(files) and all(os.path.isfile(os.path.join(d, f)) and
                                            os.path.getsize(os.path.join(d, f)) <= R.MAX_CHECKPOINT_BYTES
                                            for f in files)
    if bad:
        rec["problems"].append(f"checkpoint hash mismatch: {bad}")
    loaded = False
    try:
        R.load(f"{name}@{card['version']}", models_dir)
        loaded = True
    except Exception as e:
        rec["problems"].append(f"load failed: {e}")
    names = view["datasets"]
    real = [n for n in names if dataset_ok(n, ds)]
    if not real:
        rec["problems"].append(f"no checksummed, non-synthetic dataset among {names}")
    raw = _report(card)
    rep = report_view(card, raw)
    ho = heldout_splits(rep) if rep else []
    if not ho:
        rec["problems"].append("no held-out report with data")
    sel = view["selection"]
    rec.update(schema=view["schema"], domain=view["domain"], domain_label=view["domain_label"],
               identification=view["identification"], datasets=names, real_datasets=real, report=rep,
               report_raw=raw, heldout=ho, loaded=loaded, kept_is_baseline=sel["kept_is_baseline"])
    consistent = report_consistency(card, rep)
    if consistent:
        rec["problems"] += [f"report: {x}" for x in consistent]
    rec["qualifies"] = loaded and not bad and not ip and bool(real) and bool(ho) and rec["domain"] in R.DOMAINS \
        and not consistent
    if rec["domain"] not in R.DOMAINS:
        rec["problems"].append(f"domain {rec['domain']!r} is not one of registry.DOMAINS")
    bc = (rep or {}).get("baseline_comparison") or {}
    rec["baseline_compared"] = bool(rep) and all(k in bc for k in ("kept_test", "baseline_test", "baseline")) and \
        bc.get("baseline") in (sel["baselines"] or [bc.get("baseline")]) and \
        all(isinstance(bc.get(k), (int, float)) and bc[k] == bc[k] for k in ("kept_test", "baseline_test"))
    sign = 1 if sel["lower_is_better"] else -1
    rec["kept_beats_baseline"] = bool(rec["baseline_compared"]) and (
        sel["kept_is_baseline"] or sign * (bc["kept_test"] - bc["baseline_test"]) < 0)
    rec["calibration_reported"] = bool(calibration_values(rep)) or bool(calibration_tables(rep))
    limits_rep = (rep or {}).get("extrapolation_limits") or ""
    rec["limits_documented"] = len(view["limits"]) >= MIN_LIMITS_CHARS and len(limits_rep) >= MIN_LIMITS_CHARS
    return rec


def report_consistency(card: dict, rep: dict | None) -> list[str]:
    """Does the held-out report agree with the card it belongs to (kept candidate, kept and baseline test metrics)?
    A report rewritten to say something the card does not (audit E skeptic s2) disqualifies the model."""
    if not rep:
        return []
    sel = R.card_view(card)["selection"]
    probs = []
    bc = rep.get("baseline_comparison") or {}
    rsel = rep.get("selection") or {}
    if rsel.get("kept") is not None and rsel.get("kept") != sel["kept"]:
        probs.append(f"report keeps {rsel.get('kept')}, the card {sel['kept']}")
    if bc.get("kept") is not None and bc.get("kept") != sel["kept"]:
        probs.append("baseline comparison names another kept candidate")
    for key, cand in (("kept_test", sel["kept"]), ("baseline_test", bc.get("baseline"))):
        if key in bc and cand in sel["test"]:
            v = bc[key]
            if not isinstance(v, (int, float)) or v != v or abs(v - sel["test"][cand]) > 1e-9 * max(1.0, abs(v)):
                probs.append(f"{key} {v} differs from the card's test metric of {cand} {sel['test'][cand]}")
    return probs


def all_records(models_dir: str | None = None) -> list[dict]:
    ds = datasets()
    return [examine(m.split("@")[0], ds, models_dir) for m in R.list_models(models_dir)]


def provenance_complete(e: dict) -> bool:
    p = e.get("provenance") or {}
    lic = e.get("license") or {}
    need = ("publisher", "source", "citation", "population", "platform", "period", "retrieved")
    return all(str(p.get(k) or "").strip() for k in need) and bool(lic.get("name")) and bool(lic.get("text_location"))


def gate_f_metrics(records: list[dict] | None = None, datasets_scope: str = "registered") -> dict:
    """Gate F quantities. Datasets are counted over every registered dataset (``datasets_scope="registered"``) or
    only over those the given records cite (``"cited"``)."""
    recs = all_records() if records is None else records
    ds = datasets()
    if datasets_scope == "cited":
        cited = {n for r in recs for n in r.get("datasets", [])}
        ds = {n: e for n, e in ds.items() if n in cited}
    elif datasets_scope != "registered":
        raise ValueError("datasets_scope must be 'registered' or 'cited'")
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
        "kept_models_losing_to_their_baseline": sum(1 for r in q if r["baseline_compared"] and
                                                    not r["kept_beats_baseline"]),
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


def _commit_time(commit: str) -> _dt.datetime | None:
    r = _git("show", "-s", "--format=%cI", commit)
    try:
        return _dt.datetime.fromisoformat(r.stdout.strip()) if r.returncode == 0 else None
    except ValueError:
        return None


def _plan_file_check(p: dict, card_first: str | None, fit_commit: str | None, fit_started: str | None) -> dict:
    """One plan or amendment recorded in a card: present at the recorded commit with the recorded SHA-256, unchanged
    since, in HEAD's history, committed before the fit started (when the card records the start), contained in the
    fit commit and committed before the card was first committed."""
    out = {"path": p.get("path"), "role": p.get("role", "plan"), "commit": p.get("commit"),
           "sha256_card": p.get("sha256"), "problems": [], "content": None}
    if not p.get("path") or not p.get("commit") or not p.get("sha256"):
        out["problems"].append("no {path, sha256, commit} recorded")
        return out
    show = subprocess.run(["git", "show", f"{p['commit']}:./{p['path']}"], cwd=R.CRUCIBLE_ROOT, capture_output=True)
    if show.returncode != 0:
        out["problems"].append(f"not found at commit {p['commit'][:9]}")
        return out
    out["sha256_at_commit"] = hashlib.sha256(show.stdout).hexdigest()
    if out["sha256_at_commit"] != p["sha256"]:
        out["problems"].append("content at the recorded commit does not match the recorded sha256")
    now = os.path.join(R.CRUCIBLE_ROOT, p["path"])
    out["sha256_now"] = R.sha256_file(now) if os.path.isfile(now) else None
    if out["sha256_now"] != p["sha256"]:
        out["problems"].append("the file changed after it was recorded (or is missing)")
    if not _is_ancestor(p["commit"], "HEAD"):
        out["problems"].append("its commit is not in HEAD's history")
    committed = _commit_time(p["commit"])
    out["committed_at"] = committed.isoformat() if committed else None
    if fit_started:
        started = _dt.datetime.fromisoformat(fit_started)
        if committed is None or not committed < started:
            out["problems"].append(f"committed at {out['committed_at']}, not before the fit started at {fit_started}")
    if card_first and (card_first == p["commit"] or not _is_ancestor(p["commit"], card_first)):
        out["problems"].append("the checkpoint was committed in or before its commit")
    if fit_commit and not _is_ancestor(p["commit"], fit_commit):
        out["problems"].append("the fit ran at a commit that does not contain it")
    try:
        out["content"] = json.loads(show.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        out["problems"].append("not a JSON plan")
    return out


def _spec_problems_keyed(card: dict, sel: dict, spec: dict | None) -> list[str]:
    """fitlib plans: ``models`` {name: {primary_metric, secondary_metrics, candidates {id: description}}}."""
    if spec is None:
        return [f"plan does not specify model {card['name']}"]
    probs = []
    if spec.get("primary_metric") != sel["primary_metric"]:
        probs.append(f"primary metric {sel['primary_metric']} differs from the plan's {spec.get('primary_metric')}")
    reported = set(card.get("metrics", {}))
    missing = [m for m in [spec.get("primary_metric")] + list(spec.get("secondary_metrics", [])) if m not in reported]
    if missing:
        probs.append(f"prespecified metrics not reported: {missing}")
    planned = set(spec.get("candidates", {}))
    if planned and set(sel["candidates"]) != planned:
        probs.append(f"candidates {sorted(sel['candidates'])} differ from the plan's {sorted(planned)}")
    return probs


def _spec_problems_listed(view: dict, plan: dict, plan_path: str, amendments: list[dict]) -> list[str]:
    """fitmeta_e2 plans: ``models`` [id, ...]; per-model sections {id, candidates [{id, role}], metrics [...,
    "<key> (primary)"]}, or, for a one-model plan, top-level ``candidates`` and a ``primary_metric`` defined in words
    (stored by the fitting script under a key starting with ``primary``). Candidates absent from the plan must be
    introduced by an amendment of this plan (an amendment that names them and lists the plan in ``amends``)."""
    mid, sel = view["id"], view["selection"]
    if mid not in plan["models"]:
        return [f"plan does not name {mid}"]
    section = next((v for v in plan.values() if isinstance(v, dict) and v.get("id") == mid), None)
    spec = section or plan
    cands = spec.get("candidates") or []
    planned = [c["id"] for c in cands]
    planned_baselines = [c["id"] for c in cands if c.get("role") == "baseline"]
    probs = []
    primary = sel["primary_metric"]
    metrics = spec.get("metrics") if isinstance(spec.get("metrics"), list) else []
    tagged = [str(m)[:-len("(primary)")].strip() for m in metrics if str(m).endswith("(primary)")]
    if tagged:
        if primary not in tagged:
            probs.append(f"primary metric {primary} differs from the plan's {tagged}")
    elif spec.get("primary_metric"):
        if not str(primary or "").startswith("primary"):
            probs.append(f"primary metric {primary} is not the plan's composite primary metric")
    else:
        probs.append("plan names no primary metric")
    not_run = [c for c in planned if c not in sel["candidates"]]
    if not_run:
        probs.append(f"planned candidates not evaluated: {not_run}")
    amending = [json.dumps(a) for a in amendments if plan_path in (a.get("amends") or [])]
    for c in sel["candidates"]:
        if c not in planned and not any(re.search(rf"\b{re.escape(c)}\b", t) for t in amending):
            probs.append(f"candidate {c} is in neither the plan nor an amendment of it")
    if planned_baselines and not set(sel["baselines"]) <= set(planned_baselines):
        probs.append(f"baselines {sel['baselines']} are not the plan's {planned_baselines}")
    return probs


def plan_check(card: dict, models_dir: str | None = None) -> dict:
    """Is the card's evaluation plan (and every amendment) committed, unchanged at the recorded hash, earlier than
    the checkpoint and the fit, and does it prespecify the model's primary metric and candidate set?"""
    view = R.card_view(card)
    plans = view["plans"]
    out = {"model": view["id"], "plan": plans[0]["path"] if plans else None, "plans": [], "source": "git",
           "ok": False, "problems": []}
    if not plans:
        out["problems"].append("card has no evaluation plan {path, sha256, commit}")
        return out
    card_rel = os.path.relpath(os.path.join(R.model_dir(view["name"], models_dir), "model_card.json"),
                               R.CRUCIBLE_ROOT)
    added = _git("log", "--diff-filter=A", "--format=%H", "--", card_rel).stdout.split()
    out["card_first_commit"] = added[-1] if added else None
    fit = view["fit"]
    checks = [_plan_file_check(p, out["card_first_commit"], fit.get("commit"), fit.get("started_at")) for p in plans]
    for c in checks:
        out["problems"] += [f"{c['role']} {c['path']}: {x}" for x in c["problems"]]
        out["plans"].append({k: v for k, v in c.items() if k != "content"})
    plan = checks[0]["content"]
    if plan is not None:
        sel = view["selection"]
        models = plan.get("models")
        if isinstance(models, dict):
            out["problems"] += _spec_problems_keyed(card, sel, models.get(view["name"]))
        elif isinstance(models, list):
            out["problems"] += _spec_problems_listed(view, plan, plans[0]["path"],
                                                     [c["content"] for c in checks[1:] if c["content"] is not None])
        else:
            out["problems"].append(f"plan does not specify model {view['name']}")
    out["after_test_exposure"] = [c["path"] for c in checks[1:]
                                  if isinstance(c.get("content"), dict) and c["content"].get("written_after")]
    out["ok"] = not out["problems"]
    return out


def _e4_rule_problems(card: dict) -> list[str]:
    """The E4 rule (plan E4 global_rules.selection) recomputed from the numbers the card records: ranking on the
    validation primary metric (ties within 1e-4 to the simpler, in the plan's candidate order), eligibility by compute,
    drift rejections only with a bootstrap interval entirely above 0, test confirmation against the baselines."""
    raw = card.get("selection") or {}
    probs = []
    val, test, order = raw.get("validation") or {}, raw.get("test") or {}, list(raw.get("candidates") or [])
    lower = bool(raw.get("lower_is_better", True))
    sign = 1 if lower else -1
    remaining, ranked = list(order), []
    while remaining:
        best = min(sign * val[k] for k in remaining)
        pick = next(k for k in remaining if sign * val[k] <= best + 1e-4)
        ranked.append(pick)
        remaining.remove(pick)
    if ranked != list(raw.get("ranked_on_validation") or []):
        probs.append("recorded validation ranking differs from the recomputed one")
    budget = float(raw.get("compute_budget_us", 100.0))
    comp = raw.get("compute_us_per_row") or {}
    eligible = [k for k in ranked if float(comp.get(k, float("inf"))) <= budget]
    if eligible != list(raw.get("eligible") or []):
        probs.append("recorded eligible set differs from the compute budget")
    rej = raw.get("drift_rejected") or {}
    for k, r in rej.items():
        if not (r.get("difference_ci95") and r["difference_ci95"][0] > 0):
            probs.append(f"{k} rejected for drift without an interval entirely above 0")
    selected = next((k for k in eligible if k not in rej), None)
    if selected != raw.get("selected_on_validation"):
        probs.append(f"selected {raw.get('selected_on_validation')} but the rule gives {selected}")
    bases = list(raw.get("baselines") or [])
    if selected is not None and bases:
        best_base = min(bases, key=lambda b: sign * test[b])
        want = best_base if sign * test[best_base] < sign * test[selected] else selected
        if raw.get("kept") != want:
            probs.append(f"kept {raw.get('kept')} but the rule gives {want}")
    return probs


def selection_check(card: dict) -> dict:
    """Was a worse model kept? The kept model must follow the rule (validation argmin, ties to the simpler, unless a
    baseline beat it on test; for E4 cards the E4 rule with compute and long-horizon criteria) and must be no worse
    than any simple baseline on the primary test metric; when the card records paired-bootstrap comparisons, every
    other candidate is compared and none is significantly better."""
    sel = R.card_view(card)["selection"]
    out = {"ok": False, "problems": []}
    if (card.get("selection") or {}).get("rule_version") == "E4":
        out["problems"] += _e4_rule_problems(card)
        test, bases, kept = sel["test"], sel["baselines"], sel["kept"]
        sign = 1 if sel["lower_is_better"] else -1
        if kept in test and any(sign * test[b] < sign * test[kept] - 1e-12 for b in bases if b in test):
            out["problems"].append(f"kept {kept} is worse on test than a simple baseline")
        reported = card.get("metrics", {}).get(sel["primary_metric"])
        if reported is not None and kept in test and abs(reported - test[kept]) > 1e-9:
            out["problems"].append("card metrics are not the kept candidate's test metrics")
        out["ok"] = not out["problems"]
        return out
    val, test, bases = sel["validation"], sel["test"], sel["baselines"]
    kept = sel["kept"]
    if not (val and test and bases and kept in test and kept in val):
        out["problems"].append("selection record incomplete (validation, test, baselines, kept)")
        return out
    sign = 1 if sel["lower_is_better"] else -1
    if any(sign * test[b] < sign * test[kept] - 1e-12 for b in bases if b in test):
        out["problems"].append(f"kept {kept} is worse on test than a simple baseline")
    cx = sel["complexity"]
    best_val = min(val, key=lambda k: (sign * val[k], cx.get(k) if isinstance(cx.get(k), (int, float)) else 0))
    if kept != best_val and sign * val[kept] > sign * val[best_val] + 1e-4 and kept not in bases:
        out["problems"].append(f"kept {kept} is not the validation-best candidate {best_val}")
    reported = card.get("metrics", {}).get(sel["primary_metric"])
    if reported is not None and abs(reported - test[kept]) > 1e-9:
        out["problems"].append("card metrics are not the kept candidate's test metrics")
    if sel["worse_model_kept"] is not None:
        comps = sel["comparisons"]
        worse = sum(1 for c in comps if ((c.get("kept_minus_other") or {}).get("ci95") or [0])[0] > 0)
        if int(sel["worse_model_kept"]) != 0 or worse:
            out["problems"].append(f"a significantly better candidate was not kept (worse_model_kept = "
                                   f"{sel['worse_model_kept']}, comparisons with the interval above 0: {worse})")
        if {c.get("other") for c in comps} != set(val) - {kept}:
            out["problems"].append("not every other candidate is compared with the kept model on test")
    out["ok"] = not out["problems"]
    return out


REVIEW_PLAN = "docs/v5/calibration/plans/E3_review_plan.json"
REVIEWS_DIR = os.path.join(R.CAL_DOCS, "reviews")


def review_check(model_id: str) -> dict:
    """The E3 review of a kept model: present, its plan committed (unchanged) before the review was first committed
    and contained in the review's commit, and its kept contender equal to the plan's rule recomputed from the
    recorded numbers (`review_e3.decide` logic without the bootstrap, which is checked through the recorded
    intervals)."""
    name = model_id.split("@")[0]
    p = os.path.join(REVIEWS_DIR, f"{name}.json")
    out = {"model": model_id, "ok": False, "problems": [], "kept_after_review": None}
    if not os.path.isfile(p):
        out["problems"].append("no E3 review")
        return out
    with open(p, encoding="utf-8") as fh:
        rv = json.load(fh)
    out["kept_after_review"] = rv.get("kept_after_review")
    rel = os.path.relpath(p, R.CRUCIBLE_ROOT)
    added = _git("log", "--diff-filter=A", "--format=%H", "--", rel).stdout.split()
    first = added[-1] if added else None
    pl = dict(rv.get("plan") or {}, role="review plan")
    chk = _plan_file_check(pl, first, rv.get("review_commit"), None)
    out["problems"] += [f"plan: {x}" for x in chk["problems"]]
    if first is None:
        out["problems"].append("the review is not committed")
    if pl.get("path") != REVIEW_PLAN:
        out["problems"].append(f"review plan {pl.get('path')} is not {REVIEW_PLAN}")
    cont = rv.get("contenders") or {}
    budget = float(rv.get("budget_us_per_row", 100.0))
    eligible = [k for k, c in cont.items() if c.get("eligible_role", True) and float(c.get("compute_us", 1e9)) <= budget]
    order = sorted(eligible, key=lambda k: cont[k]["primary"])
    rej = rv.get("drift_rejected") or {}
    for k, r in rej.items():
        if not (r.get("difference_ci95") and r["difference_ci95"][0] > 0):
            out["problems"].append(f"{k} rejected for drift without an interval entirely above 0")
    want = next((k for k in order if k not in rej), None)
    if want != rv.get("kept_after_review"):
        out["problems"].append(f"kept_after_review {rv.get('kept_after_review')} but the rule gives {want}")
    if set(eligible) != set(rv.get("eligible") or []):
        out["problems"].append("recorded eligible contenders differ from the compute budget")
    if rv.get("model") != model_id:
        out["problems"].append(f"review is of {rv.get('model')}, not {model_id}")
    out["ok"] = not out["problems"]
    return out


def reviewed_models() -> list[str]:
    try:
        with open(os.path.join(R.CRUCIBLE_ROOT, REVIEW_PLAN), encoding="utf-8") as fh:
            return list(json.load(fh)["applies_to"])
    except (OSError, ValueError, KeyError):
        return []


def sp55_metrics(records: list[dict] | None = None) -> dict:
    recs = all_records() if records is None else records
    cards = [(r["card"], r.get("models_dir")) for r in recs if r.get("card")]
    plans = {c["name"]: plan_check(c, md) for c, md in cards}
    sels = {c["name"]: selection_check(c) for c, _ in cards}
    registered = {f"{c['name']}@{c['version']}" for c, _ in cards}
    reviews = {m: review_check(m) for m in reviewed_models() if m in registered} if not any(
        r.get("models_dir") for r in recs) else {}
    return {"models_checked": len(cards),
            "metrics_prespecified_before_fit": bool(cards) and all(p["ok"] for p in plans.values()),
            "models_without_prespecified_plan": sum(1 for p in plans.values() if not p["ok"]),
            "worse_model_kept": sum(1 for s in sels.values() if not s["ok"]) +
                                sum(1 for r in reviews.values() if not r["ok"]),
            "learned_model_kept": sum(1 for c, _ in cards if not R.card_view(c)["selection"]["kept_is_baseline"]),
            "reviews_checked": len(reviews), "reviews_ok": sum(1 for r in reviews.values() if r["ok"]),
            # protocol deviation (audit E S2.5): an amendment that added candidates after an earlier run had computed
            # test metrics; prespecified before the refit, but the test split had been seen. Flagged for the reviewer.
            "amendments_after_test_exposure": ",".join(sorted(n for n, p in plans.items()
                                                              if p.get("after_test_exposure"))),
            "reviews_keeping_a_non_fitted_contender": sum(1 for r in reviews.values()
                                                          if r["kept_after_review"] not in (None, "K")),
            "_plans": plans, "_selection": sels, "_reviews": reviews}
