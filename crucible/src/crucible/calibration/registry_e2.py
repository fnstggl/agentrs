"""Fitted-model registry of workstream E2 (INTERFACE_CONTRACTS section 5).

Workstream E1 owns `calibration/registry.py`; it did not exist in this worktree when E2 started, so this module
implements the same section-5 interface for E2's datasets and models (`datasets`, `acquire`, `load`, `FittedModel`)
and is data-driven, so the orchestrator can unify the two registries by scanning the same model cards:

* every model lives in `docs/v5/calibration/models/<name>/` with `model_card.json` and one `.npz` checkpoint
  (no pickle); the card names its loader class (`"loader": "crucible.calibration.<module>:<Class>"`);
* every dataset has a committed manifest `docs/v5/calibration/datasets/<name>.json` (URL, license, SHA-256 per file,
  size, schema, provenance); the data itself lives under `$CRUCIBLE_DATA` and is never committed.

World code depends only on `load(...)`: loading reads the card and the checkpoint, verifies the checkpoint's SHA-256
and never touches a dataset.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CRUCIBLE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
# CRUCIBLE_E2_CAL_DIR redirects cards, checkpoints and reports (used only for smoke runs of the fitting scripts).
CAL_DIR = os.environ.get("CRUCIBLE_E2_CAL_DIR") or os.path.join(CRUCIBLE_ROOT, "docs", "v5", "calibration")
MODELS_DIR = os.path.join(CAL_DIR, "models")
REPORTS_DIR = os.path.join(CAL_DIR, "reports")
DATASETS_DIR = os.path.join(CRUCIBLE_ROOT, "docs", "v5", "calibration", "datasets")
PLANS_DIR = os.path.join(CRUCIBLE_ROOT, "docs", "v5", "calibration", "plans")
CARD_SCHEMA = "crucible.v5.calibration.model_card/1"
E2_DATASETS = ("criteo_uplift_v2.1", "gharchive_2024-10-07_96h")
E2_MODELS = ("commercial_response@1", "software_deep_engagement@1", "software_repo_growth@1")
REQUIRED_CARD_FIELDS = ("schema", "name", "version", "workstream", "domain", "loader", "features", "target",
                        "outputs", "identification", "dataset", "checkpoint", "report", "plan", "fit", "metrics",
                        "baseline_metrics", "candidates", "selection", "limits", "extrapolation_limits",
                        "simulator_mapping", "provenance")


class CalibrationError(RuntimeError):
    pass


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: str) -> str:
    return os.path.relpath(path, CRUCIBLE_ROOT)


def split_name(model_name: str) -> tuple[str, int]:
    if "@" not in model_name:
        raise CalibrationError(f"model name must be '<name>@<version>', got {model_name!r}")
    name, ver = model_name.rsplit("@", 1)
    return name, int(ver)


# ----------------------------------------------------------------------------------------------- datasets

def datasets() -> dict[str, dict]:
    """name -> {url, license, sha256 per file, size, schema, provenance} for E2's datasets (committed manifests)."""
    out = {}
    for name in E2_DATASETS:
        p = os.path.join(DATASETS_DIR, f"{name}.json")
        if os.path.exists(p):
            with open(p) as fh:
                out[name] = json.load(fh)
    return out


def acquire(name: str, dest: str | None = None) -> str:
    """Download, verify checksums, validate and convert one dataset; returns its directory."""
    if name == "criteo_uplift_v2.1":
        from . import criteo
        return criteo.acquire(dest)
    if name == "gharchive_2024-10-07_96h":
        from . import gharchive
        man = datasets().get(name)
        expected = {f["file"]: f["sha256"] for f in man["files"]} if man else None
        return gharchive.acquire(dest, workers=int(os.environ.get("CRUCIBLE_ACQUIRE_WORKERS", "1")), expected=expected)
    raise CalibrationError(f"unknown dataset {name!r}")


# ----------------------------------------------------------------------------------------------- models

class FittedModel:
    """A fitted behavioural model loaded from its card and checkpoint.

    Attributes follow INTERFACE_CONTRACTS section 5: name, version, features, target, metrics (held-out test metrics
    of the kept model), baseline_metrics (same metrics of the baseline), provenance, limits (extrapolation limits
    in words). `predict(X)` takes X[n, len(features)] in the order of `features` and returns predictions of
    `outputs` ([n] for one output, [n, k] for several).

    The base class evaluates a single portable predictor stored under the prefix 'model'; subclasses with several
    outputs or extra simulator-facing methods override `predict`.
    """

    def __init__(self, card: dict, arrays: dict[str, np.ndarray]):
        self.card = card
        self.name = card["name"]
        self.version = int(card["version"])
        self.id = f"{self.name}@{self.version}"
        self.features = list(card["features"])
        self.target = card["target"]
        self.outputs = list(card["outputs"])
        self.metrics = card["metrics"]
        self.baseline_metrics = card["baseline_metrics"]
        self.provenance = card["provenance"]
        self.limits = card["limits"]
        self.identification = card["identification"]["kind"]
        self.simulator_mapping = card["simulator_mapping"]
        self.arrays = arrays

    def _check(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        if X.ndim != 2 or X.shape[1] != len(self.features):
            raise ValueError(f"{self.id} expects X[n, {len(self.features)}] in the order {self.features}")
        return X

    def submodel(self, prefix: str) -> dict[str, np.ndarray]:
        p = prefix + "/"
        out = {k[len(p):]: v for k, v in self.arrays.items() if k.startswith(p)}
        if not out:
            raise KeyError(f"{self.id} has no sub-model {prefix!r}")
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        from . import portable
        return portable.predict(self.submodel("model"), self._check(X))

    def matrix(self, columns: dict[str, np.ndarray]) -> np.ndarray:
        """Build X from named columns (all of `features` must be present)."""
        missing = [f for f in self.features if f not in columns]
        if missing:
            raise KeyError(f"{self.id}: missing feature columns {missing}")
        return np.column_stack([np.asarray(columns[f], dtype=np.float64) for f in self.features])

    def __repr__(self) -> str:
        return f"<FittedModel {self.id} target={self.target} identification={self.identification}>"


def model_dir(model_name: str) -> str:
    name, _ = split_name(model_name)
    return os.path.join(MODELS_DIR, name)


def model_card(model_name: str) -> dict:
    name, ver = split_name(model_name)
    p = os.path.join(MODELS_DIR, name, "model_card.json")
    if not os.path.exists(p):
        raise CalibrationError(f"no model card for {model_name} at {rel(p)}")
    with open(p) as fh:
        card = json.load(fh)
    if int(card.get("version", -1)) != ver:
        raise CalibrationError(f"{rel(p)} holds version {card.get('version')}, not {ver}")
    return card


def verify_checkpoint(model_name: str) -> dict:
    card = model_card(model_name)
    p = os.path.join(model_dir(model_name), card["checkpoint"]["file"])
    actual = sha256_file(p) if os.path.exists(p) else None
    return {"model": model_name, "file": rel(p), "expected": card["checkpoint"]["sha256"], "actual": actual,
            "bytes": os.path.getsize(p) if actual else 0, "ok": actual == card["checkpoint"]["sha256"]}


def load(model_name: str) -> FittedModel:
    """Load a fitted model by '<name>@<version>'. Verifies the checkpoint hash; never reads a dataset."""
    card = model_card(model_name)
    v = verify_checkpoint(model_name)
    if not v["ok"]:
        raise CalibrationError(f"checkpoint hash mismatch for {model_name}: {v}")
    mod_name, cls_name = card["loader"].split(":")
    if not mod_name.startswith("crucible.calibration."):
        raise CalibrationError(f"loader outside crucible.calibration refused: {card['loader']}")
    cls = getattr(importlib.import_module(mod_name), cls_name)
    with np.load(os.path.join(model_dir(model_name), card["checkpoint"]["file"]), allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    return cls(card, arrays)


def list_models() -> list[str]:
    """Every model card under docs/v5/calibration/models (any workstream), as '<name>@<version>'."""
    out = []
    if os.path.isdir(MODELS_DIR):
        for d in sorted(os.listdir(MODELS_DIR)):
            p = os.path.join(MODELS_DIR, d, "model_card.json")
            if os.path.exists(p):
                with open(p) as fh:
                    c = json.load(fh)
                out.append(f"{c['name']}@{c['version']}")
    return out


def report(model_name: str) -> dict:
    card = model_card(model_name)
    with open(os.path.join(CRUCIBLE_ROOT, card["report"])) as fh:
        return json.load(fh)


# ----------------------------------------------------------------------------------------------- audits

def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=CRUCIBLE_ROOT, stderr=subprocess.DEVNULL).decode().strip()


def prespecified_before_fit(model_name: str) -> dict:
    """Was the evaluation plan committed before the fit started, and is it unchanged?

    Uses git when available: the commit that first added the plan file must be an ancestor of HEAD, its commit
    time must precede the fit's start time recorded in the card, and the plan's SHA-256 at that commit, in the card
    and in the working tree must agree.
    """
    card = model_card(model_name)
    plan = card["plan"]
    path = os.path.join(CRUCIBLE_ROOT, plan["path"])
    out = {"model": model_name, "plan": plan["path"], "fit_started_at": card["fit"]["started_at"]}
    out["plan_sha256_now"] = sha256_file(path) if os.path.exists(path) else None
    out["plan_sha256_card"] = plan["sha256"]
    try:
        first = _git("log", "--diff-filter=A", "--format=%H %cI", "--", plan["path"]).splitlines()[-1].split()
        out["plan_commit"], out["plan_committed_at"] = first[0], first[1]
        blob = subprocess.check_output(["git", "show", f"{first[0]}:{os.path.relpath(path, _git('rev-parse', '--show-toplevel'))}"],
                                       cwd=CRUCIBLE_ROOT, stderr=subprocess.DEVNULL)
        out["plan_sha256_at_commit"] = hashlib.sha256(blob).hexdigest()
        out["source"] = "git"
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        out["plan_commit"], out["plan_committed_at"] = plan.get("commit"), plan.get("committed_at")
        out["plan_sha256_at_commit"] = plan["sha256"]
        out["source"] = "card"
    import datetime as _dt
    committed = _dt.datetime.fromisoformat(out["plan_committed_at"]) if out["plan_committed_at"] else None
    started = _dt.datetime.fromisoformat(card["fit"]["started_at"])
    out["ok"] = bool(committed is not None and committed < started
                     and out["plan_sha256_at_commit"] == out["plan_sha256_card"] == out["plan_sha256_now"])
    return out


def card_problems(card: dict) -> list[str]:
    probs = [f"missing field {f}" for f in REQUIRED_CARD_FIELDS if f not in card]
    if card.get("schema") != CARD_SCHEMA:
        probs.append(f"schema {card.get('schema')!r} != {CARD_SCHEMA}")
    if card.get("dataset", {}).get("synthetic", True):
        probs.append("dataset is synthetic or not declared")
    if not card.get("extrapolation_limits"):
        probs.append("no extrapolation limits")
    return probs


def gate_metrics(models=E2_MODELS) -> dict:
    """Registry-driven Gate F / SP-5.5 quantities over the given models (E2's by default)."""
    cards = {m: model_card(m) for m in models}
    reports = {}
    for m, c in cards.items():
        p = os.path.join(CRUCIBLE_ROOT, c["report"])
        if os.path.exists(p):
            with open(p) as fh:
                reports[m] = json.load(fh)
    ds = datasets()
    ver = [verify_checkpoint(m) for m in models]
    pre = [prespecified_before_fit(m) for m in models]
    domains_real = {c["domain"] for c in cards.values() if not c["dataset"].get("synthetic", True)}
    return {
        "models": list(models),
        "empirical_domains": sorted(domains_real),
        "heldout_reports": sum(1 for r in reports.values() if r.get("heldout", {}).get("test")),
        "baseline_comparisons": sum(1 for c in cards.values() if c["selection"].get("comparisons")),
        "calibration_metric_reports": sum(1 for r in reports.values() if r.get("calibration_metrics")),
        "checkpoints_saved": sum(1 for v in ver if v["actual"]),
        "checkpoint_hash_mismatches": sum(1 for v in ver if not v["ok"]),
        "datasets_with_checksums": sum(1 for d in ds.values() if d.get("files") and all(f.get("sha256") for f in d["files"])),
        "provenance_records": sum(1 for d in ds.values() if d.get("provenance")),
        "extrapolation_limit_sections": sum(1 for c in cards.values() if c.get("extrapolation_limits")),
        "synthetic_only_domains": sum(1 for c in cards.values() if c["dataset"].get("synthetic", True)),
        "metrics_prespecified_before_fit": all(p["ok"] for p in pre),
        "worse_model_kept": sum(int(c["selection"].get("worse_model_kept", 1)) for c in cards.values()),
        "card_problems": {m: card_problems(c) for m, c in cards.items() if card_problems(c)},
    }
