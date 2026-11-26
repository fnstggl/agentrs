"""Compatibility layer: workstream E2's original registry entry points, served by the unified registry.

E2 was built before E1's `calibration/registry.py` existed and shipped its own section-5 registry here. The two are
now one: `crucible.calibration.load` / `registry.load` loads E2's cards (schema `registry.LOADER_SCHEMA`) and
`crucible.calibration.datasets()` serves E2's dataset manifests. World code uses `crucible.calibration.load`; this
module only keeps E2's names working for the fitting scripts (`fit_commercial`, `fit_software`, `fitmeta_e2`) and for
existing callers:

* every function delegates to `registry`, `datasets` or `audit`; nothing here reads a card or a dataset on its own;
* ``CRUCIBLE_E2_CAL_DIR`` still redirects cards, checkpoints and reports of the fitting scripts' smoke runs;
* model names must be ``name@version`` here (the unified registry also accepts a bare name);
* `datasets()` and `gate_metrics()` are restricted to E2's datasets and models.
"""
from __future__ import annotations

import os

from . import registry as _R
from .datasets import acquire as _acquire
from .datasets import datasets as _datasets

CRUCIBLE_ROOT = _R.CRUCIBLE_ROOT
# CRUCIBLE_E2_CAL_DIR redirects cards, checkpoints and reports (used only for smoke runs of the fitting scripts).
CAL_DIR = os.environ.get("CRUCIBLE_E2_CAL_DIR") or _R.CAL_DOCS
MODELS_DIR = os.path.join(CAL_DIR, "models")
REPORTS_DIR = os.path.join(CAL_DIR, "reports")
DATASETS_DIR = os.path.join(CRUCIBLE_ROOT, "docs", "v5", "calibration", "datasets")
PLANS_DIR = os.path.join(CRUCIBLE_ROOT, "docs", "v5", "calibration", "plans")
CARD_SCHEMA = _R.LOADER_SCHEMA
E2_DATASETS = ("criteo_uplift_v2.1", "gharchive_2024-10-07_96h")
E2_MODELS = ("commercial_response@1", "software_deep_engagement@1", "software_repo_growth@1")
REQUIRED_CARD_FIELDS = _R.REQUIRED_CARD_FIELDS[_R.LOADER_SCHEMA]

CalibrationError = _R.CalibrationError
FittedModel = _R.FittedModel
sha256_file = _R.sha256_file
card_problems = _R.card_problems


def rel(path: str) -> str:
    return os.path.relpath(path, CRUCIBLE_ROOT)


def split_name(model_name: str) -> tuple[str, int]:
    if "@" not in model_name:
        raise CalibrationError(f"model name must be '<name>@<version>', got {model_name!r}")
    name, ver = model_name.rsplit("@", 1)
    return name, int(ver)


def datasets() -> dict[str, dict]:
    """E2's datasets, as served by `crucible.calibration.datasets()`."""
    return {n: e for n, e in _datasets().items() if n in E2_DATASETS}


def acquire(name: str, dest: str | None = None) -> str:
    if name not in E2_DATASETS:
        raise CalibrationError(f"unknown E2 dataset {name!r}; use crucible.calibration.acquire")
    return _acquire(name, dest)


def model_dir(model_name: str) -> str:
    return os.path.join(MODELS_DIR, split_name(model_name)[0])


def model_card(model_name: str) -> dict:
    name, ver = split_name(model_name)
    card = _R.read_card(name, MODELS_DIR)
    if int(card.get("version", -1)) != ver:
        raise _R.ModelNotFoundError(f"{name} holds version {card.get('version')}, not {ver}")
    return card


def verify_checkpoint(model_name: str) -> dict:
    """E2's single-checkpoint view of `registry.verify_checkpoint`."""
    split_name(model_name)
    v = _R.verify_checkpoint(model_name, MODELS_DIR)
    fn, f = next(iter(v["files"].items()))
    return {"model": model_name, "file": rel(os.path.join(model_dir(model_name), fn)), "expected": f["expected"],
            "actual": f["actual"], "bytes": f["bytes"], "ok": v["ok"]}


def load(model_name: str) -> FittedModel:
    """`registry.load` on `MODELS_DIR` (the redirected directory in smoke runs)."""
    split_name(model_name)
    return _R.load(model_name, MODELS_DIR)


def list_models() -> list[str]:
    return _R.list_models(MODELS_DIR)


def report(model_name: str) -> dict:
    return _R.report(model_name, MODELS_DIR)


def prespecified_before_fit(model_name: str) -> dict:
    """`audit.plan_check` of the card: plan and amendments committed (git) before the fit started and before the card
    was first committed, unchanged since, and naming the model, its primary metric and its candidates."""
    from .audit import plan_check
    return plan_check(model_card(model_name), models_dir=MODELS_DIR)


def gate_metrics(models=E2_MODELS) -> dict:
    """The Gate F / SP-5.5 quantities of `audit` over the given models (E2's by default), counting only the datasets
    those models cite."""
    from . import audit
    ds = _datasets()
    recs = [audit.examine(split_name(m)[0], ds, models_dir=MODELS_DIR) for m in models]
    f = audit.gate_f_metrics(recs, datasets_scope="cited")
    s = audit.sp55_metrics(recs)
    cards = [r["card"] for r in recs if r.get("card")]
    return {
        "models": list(models),
        "empirical_domains": sorted({r["domain"] for r in recs if r.get("qualifies")}),
        **{k: f[k] for k in ("heldout_reports", "baseline_comparisons", "calibration_metric_reports",
                             "checkpoints_saved", "checkpoint_hash_mismatches", "datasets_with_checksums",
                             "provenance_records", "extrapolation_limit_sections", "synthetic_only_domains")},
        "metrics_prespecified_before_fit": s["metrics_prespecified_before_fit"],
        "worse_model_kept": s["worse_model_kept"],
        "card_problems": {f"{c['name']}@{c['version']}": card_problems(c) for c in cards if card_problems(c)},
    }
