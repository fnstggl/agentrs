"""The fitted-model registry (INTERFACE_CONTRACTS section 5): the one entry point for every calibrated model.

A fitted model is a directory `docs/v5/calibration/models/<name>/` holding `model_card.json` and the checkpoint
files it lists with their SHA-256 (each < 5 MB, NumPy `.npz` loaded without pickle). The registry is the set of those
directories: a calibration workstream registers a model by writing it there, never by editing this file.
`load("content_consumption@1")` verifies every checkpoint hash and returns a `FittedModel`; world code depends only on
`load` (re-exported as `crucible.calibration.load`), never on datasets.

Two card schemas are registered. They were written by two fitting toolchains and both are read here, by `load`,
`card_view`, the gate audit (`audit.py`) and the fidelity register (`fidelity.py`); cards are never rewritten:

* ``crucible.v5.model_card/1`` (`SCHEMA`; written by `save_model` / `fitlib`, workstream E1). The card names a
  predictor ``kind`` from `predictors.KINDS` (``module`` names a `crucible.calibration.*` module to import first when
  the kind is defined elsewhere), the checkpoint file and ``files`` {file name: sha256}. ``domain``; ``features``,
  ``feature_map``, ``target``; ``metrics`` / ``baseline_metrics`` (held-out test metrics of the kept model and of the
  best simple baseline); ``candidates``; ``selection`` {rule, primary_metric, lower_is_better, validation, test,
  baselines, candidates, kept, kept_is_baseline}; ``provenance`` {datasets, identification, population, platform,
  period, evaluation_plan {path, sha256, commit}, fit {commit, command, seconds}}; ``limits``; ``mechanisms``;
  ``report``.
* ``crucible.v5.calibration.model_card/1`` (`LOADER_SCHEMA`; written by `fitmeta_e2`, workstream E2). The card names
  a loader class (``"loader": "crucible.calibration.<module>:<Class>"``, a `FittedModel` subclass) and one
  ``checkpoint`` {file, sha256, bytes}. ``domain``; ``identification`` {kind, statement}; ``dataset`` {name,
  synthetic, ...}; ``features``, ``feature_docs``, ``simulator_mapping``, ``target``, ``outputs``; ``metrics`` /
  ``baseline_metrics``; ``candidates`` {id: {complexity, validation, test}}; ``selection`` {rule, kept, baseline,
  comparisons (paired bootstrap), worse_model_kept}; ``plan`` and ``plan_amendments`` [{path, sha256, commit,
  committed_at}]; ``fit`` {started_at, commit, clean, dirty_paths, command, ...}; ``limits``,
  ``extrapolation_limits``; ``fidelity_mechanisms``; ``report``.

`card_view(card)` maps either schema onto one description (canonical domain and identification, datasets, checkpoint
files, report, limits, mechanisms, evaluation plans with their amendments, fit, the selection record) so that the
gates measure every model the same way. The two schemas label two vocabulary terms differently; `DOMAIN_ALIASES` and
`IDENTIFICATION_ALIASES` map them onto `DOMAINS` and `IDENTIFICATION`, and the card's own term is kept as
``domain_label`` / ``identification_label``.
"""
from __future__ import annotations

import fnmatch
import hashlib
import importlib
import json
import os

import numpy as np

from . import predictors as _pred

SCHEMA = "crucible.v5.model_card/1"                       # predictor-kind cards (save_model, fitlib)
LOADER_SCHEMA = "crucible.v5.calibration.model_card/1"    # loader-class cards (fitmeta_e2)
CARD_SCHEMAS = (SCHEMA, LOADER_SCHEMA)
CRUCIBLE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CAL_DOCS = os.path.join(CRUCIBLE_ROOT, "docs", "v5", "calibration")
MODELS_DIR = os.path.join(CAL_DOCS, "models")
REPORTS_DIR = os.path.join(CAL_DOCS, "reports")
MAX_CHECKPOINT_BYTES = 5 * 1024 * 1024
# Pinned hashes of every registered card and checkpoint, kept OUTSIDE the model directories (audit E skeptic s1: the
# card's behavioural fields - feature order, feature_quantiles, prior_rate, selection, response tables - were
# unprotected because the expected checkpoint hash sat in the card itself). `load` refuses a model of the canonical
# models directory whose card or checkpoint differs from this index, and a canonical model missing from it.
INDEX_PATH = os.path.join(CAL_DOCS, "registry_index.json")
CANONICAL_MODELS_DIR = MODELS_DIR

# Behavioural domains (MASTER_SPEC 5.3). Gate F counts distinct domains with a fitted, held-out-evaluated model.
DOMAINS = {
    "content_selection": "which exposed content an individual consumes (feed, news, search results)",
    "watch_duration": "how long an individual watches or reads consumed content",
    "browsing_frequency": "when individuals return to a service and how often they start sessions",
    "repeat_consumption": "re-consumption of previously consumed items",
    "attention_allocation": "how individuals split limited time across activities",
    "engagement_actions": "likes, shares, follows, comments after consumption",
    "purchase_behavior": "purchases and conversion",
    "advertising_response": "responses to advertising or commercial interventions (randomised where possible)",
    "product_retention": "continued use, churn and cancellation of products and subscriptions",
    "software_adoption": "software collaboration and adoption",
    "community_participation": "joining and participating in communities",
    "social_diffusion": "spread of content and adoption through social ties",
}
# Domain labels written by the loader-schema toolchain, mapped onto DOMAINS.
DOMAIN_ALIASES = {
    "commercial_intervention_response": "advertising_response",
    "software_collaboration_and_adoption": "software_adoption",
}
# How a model's effect estimates are identified (MASTER_SPEC 5.2: keep observational and experimental apart).
IDENTIFICATION = ("observational", "randomized_exposure", "randomized_controlled_trial", "randomized_encouragement_iv")
# A randomised experiment that assigns the intervention (Criteo's incrementality tests) is a randomised controlled
# trial when the estimand is the effect of assignment (intention to treat); randomized_exposure is reserved for
# randomly inserted content, whose effect is not an intervention arm; randomized_encouragement_iv is an effect of the
# treatment actually received (display) identified with the randomised assignment (eligibility) as its instrument: a
# complier effect that additionally rests on the exclusion restriction (assignment acts only through the treatment).
IDENTIFICATION_ALIASES = {"randomized_experiment": "randomized_controlled_trial",
                          "randomised_experiment": "randomized_controlled_trial"}

# Fields every card of each schema carries (card_problems reports the missing ones).
REQUIRED_CARD_FIELDS = {
    SCHEMA: ("schema", "name", "version", "kind", "checkpoint", "files", "domain", "features", "feature_map",
             "target", "metrics", "baseline_metrics", "candidates", "selection", "provenance", "limits", "report"),
    LOADER_SCHEMA: ("schema", "name", "version", "workstream", "domain", "loader", "features", "target", "outputs",
                    "identification", "dataset", "checkpoint", "report", "plan", "fit", "metrics",
                    "baseline_metrics", "candidates", "selection", "limits", "extrapolation_limits",
                    "simulator_mapping", "provenance"),
}


class CalibrationError(RuntimeError):
    """Base class of registry errors."""


class ModelNotFoundError(CalibrationError, KeyError):
    """No card for this name, or the registered version differs."""


class ChecksumError(CalibrationError, OSError):
    """A checkpoint file is missing or fails its SHA-256 check."""


class CardError(CalibrationError, ValueError):
    """A card that the registry cannot read or must refuse (unknown schema, kind or loader)."""


def read_index() -> dict:
    if not os.path.isfile(INDEX_PATH):
        return {}
    with open(INDEX_PATH, encoding="utf-8") as fh:
        return json.load(fh).get("models", {})


def index_entry(name: str, models_dir: str | None = None) -> dict:
    """{card_sha256, files {file: sha256}} of a model as it is on disk now."""
    d = model_dir(name, models_dir)
    card = read_card(name, models_dir)
    return {"version": int(card["version"]), "card_sha256": sha256_file(os.path.join(d, "model_card.json")),
            "files": {fn: sha256_file(os.path.join(d, fn)) for fn in sorted(card_files(card))}}


def write_index() -> str:
    """Pin the current cards and checkpoints of the canonical models directory (an explicit, committed step)."""
    entries = {m.split("@")[0]: index_entry(m.split("@")[0], CANONICAL_MODELS_DIR)
               for m in list_models(CANONICAL_MODELS_DIR)}
    with open(INDEX_PATH + ".tmp", "w", encoding="utf-8") as fh:
        json.dump({"schema": "crucible.v5.calibration.registry_index/1", "models": entries}, fh, indent=1,
                  sort_keys=True)
        fh.write("\n")
    os.replace(INDEX_PATH + ".tmp", INDEX_PATH)
    return INDEX_PATH


def index_problems(name: str, models_dir: str | None = None) -> list[str]:
    """Why a model's card or checkpoint differs from the pinned index ([] when it matches). Models outside the
    canonical directory are checked only when the index knows their name (scratch registries in tests)."""
    idx = read_index()
    root = os.path.abspath(_models_root(models_dir))
    canonical = root == os.path.abspath(CANONICAL_MODELS_DIR)
    if name not in idx:
        return ["model is not in the pinned registry index (run registry.write_index and commit)"] if canonical else []
    want = idx[name]
    try:
        got = index_entry(name, models_dir)
    except (OSError, KeyError, ValueError) as e:
        return [f"cannot hash the model: {e}"]
    probs = []
    if got["card_sha256"] != want.get("card_sha256"):
        probs.append("model_card.json differs from the pinned index")
    if got["files"] != want.get("files"):
        probs.append("checkpoint files differ from the pinned index")
    return probs


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_domain(label: str | None) -> str:
    return DOMAIN_ALIASES.get(label or "", label or "")


def canonical_identification(label) -> str:
    if isinstance(label, dict):
        label = label.get("kind")
    s = str(label or "").strip().lower()
    return IDENTIFICATION_ALIASES.get(s, s)


# ---------------------------------------------------------------- card schemas
def card_schema(card: dict) -> str:
    """The card's schema; cards without one are recognised by their loading field (``kind`` or ``loader``)."""
    s = card.get("schema")
    if s in CARD_SCHEMAS:
        return s
    if s is None and "loader" in card:
        return LOADER_SCHEMA
    if s is None and "kind" in card:
        return SCHEMA
    raise CardError(f"{card.get('name')}: unknown model card schema {s!r} (known: {CARD_SCHEMAS})")


def card_files(card: dict) -> dict[str, str]:
    """{checkpoint file name: sha256} of a card of either schema."""
    if card_schema(card) == LOADER_SCHEMA:
        ck = card.get("checkpoint") or {}
        return {ck["file"]: ck.get("sha256", "")} if ck.get("file") else {}
    return dict(card.get("files") or {})


def checkpoint_file(card: dict) -> str:
    """The checkpoint `.npz` that `load` reads."""
    if card_schema(card) == LOADER_SCHEMA:
        return (card.get("checkpoint") or {}).get("file", "")
    return card.get("checkpoint") or next(iter(card.get("files") or {}), "")


def _feature_map_from_mapping(card: dict) -> dict[str, str]:
    """Per-input simulator mapping of a loader-schema card: an exact key of ``simulator_mapping``, a grouped key
    (``hour_sin/hour_cos``, ``actor_*``), the card's blanket ``features`` statement, or the input's documentation."""
    sm = card.get("simulator_mapping") or {}
    docs = card.get("feature_docs") or {}
    out = {}
    for f in card.get("features", []):
        text = sm.get(f) if isinstance(sm.get(f), str) else None
        if text is None:
            for k, v in sm.items():
                parts = [p.strip() for p in str(k).split("/")]
                if f in parts or any("*" in p and fnmatch.fnmatchcase(f, p) for p in parts):
                    text = v
                    break
        if text is None and isinstance(sm.get("features"), str):
            text = sm["features"]
        if text is None and f in docs:
            text = docs[f]
        if text is not None:
            out[f] = str(text)
    return out


def _loader_primary_metric(card: dict) -> str | None:
    """Primary metric key of a loader-schema card. Its paired-bootstrap comparisons name the metric
    (``"log_loss (test)"``); a composite primary metric defined in words by the plan is stored under a key starting
    with ``primary`` (``primary_pooled_logloss_visit_plus_conversion``)."""
    sel = card.get("selection") or {}
    cands = card.get("candidates") or {}
    kept_val = (cands.get(sel.get("kept")) or {}).get("validation") or {}
    for c in sel.get("comparisons") or []:
        key = str(c.get("metric", "")).split(" (")[0].strip()
        if key in kept_val:
            return key
    prim = [k for k in kept_val if k.startswith("primary")]
    return prim[0] if len(prim) == 1 else None


def _view_kind_card(card: dict) -> dict:
    prov = card.get("provenance") or {}
    sel = card.get("selection") or {}
    ident = prov.get("identification") or card.get("identification") or ""
    plan = prov.get("evaluation_plan")
    fit = prov.get("fit") or {}
    baselines = list(sel.get("baselines") or [])
    kept = sel.get("kept")
    if "kept_is_baseline" in sel:
        kept_is_baseline = bool(sel["kept_is_baseline"])
    else:
        kept_is_baseline = kept in baselines if kept else True
    domain = card.get("domain") or prov.get("domain") or ""
    return {
        "schema": SCHEMA, "name": card.get("name"), "version": card.get("version"),
        "id": f"{card.get('name')}@{card.get('version')}",
        "domain": canonical_domain(domain), "domain_label": domain,
        "identification": canonical_identification(ident),
        "identification_label": ident.get("kind") if isinstance(ident, dict) else ident,
        "datasets": list(prov.get("datasets") or []),
        "files": dict(card.get("files") or {}),
        "loader": {"kind": card.get("kind"), "module": card.get("module")},
        "report": card.get("report"),
        "limits": card.get("limits") or "",
        "mechanisms": list(card.get("mechanisms") or []),
        "feature_map": dict(card.get("feature_map") or {}),
        "plans": [dict(plan, role="plan")] if plan else [],
        "fit": {"commit": fit.get("commit"), "started_at": None, "command": fit.get("command"), "clean": None,
                "dirty_paths": None},
        "selection": {
            "rule": sel.get("rule"), "primary_metric": sel.get("primary_metric"),
            "lower_is_better": bool(sel.get("lower_is_better", True)),
            "validation": dict(sel.get("validation") or {}), "test": dict(sel.get("test") or {}),
            "baselines": baselines, "candidates": list(sel.get("candidates") or []), "kept": kept,
            "kept_is_baseline": kept_is_baseline, "complexity": {}, "comparisons": [], "worse_model_kept": None,
        },
    }


def _view_loader_card(card: dict) -> dict:
    sel = card.get("selection") or {}
    cands = card.get("candidates") or {}
    ds = card.get("dataset") or {}
    ident = card.get("identification") or {}
    fit = card.get("fit") or {}
    primary = _loader_primary_metric(card)
    kept = sel.get("kept")
    baselines = [sel["baseline"]] if sel.get("baseline") else [c for c, v in cands.items() if v.get("complexity") == 0]
    plans = []
    if card.get("plan"):
        plans.append(dict(card["plan"], role="plan"))
    plans += [dict(a, role="amendment") for a in card.get("plan_amendments") or []]
    name = ds.get("name") or (card.get("provenance") or {}).get("dataset")
    return {
        "schema": LOADER_SCHEMA, "name": card.get("name"), "version": card.get("version"),
        "id": f"{card.get('name')}@{card.get('version')}",
        "domain": canonical_domain(card.get("domain")), "domain_label": card.get("domain") or "",
        "identification": canonical_identification(ident),
        "identification_label": ident.get("kind") if isinstance(ident, dict) else ident,
        "datasets": [name] if name else [],
        "files": card_files(card),
        "loader": {"class": card.get("loader")},
        "report": card.get("report"),
        "limits": card.get("limits") or "",
        "mechanisms": list(card.get("fidelity_mechanisms") or card.get("mechanisms") or []),
        "feature_map": _feature_map_from_mapping(card),
        "plans": plans,
        "fit": {"commit": fit.get("commit"), "started_at": fit.get("started_at"), "command": fit.get("command"),
                "clean": fit.get("clean"), "dirty_paths": fit.get("dirty_paths")},
        "selection": {
            "rule": sel.get("rule"), "primary_metric": primary,
            # every loader-schema selection rule keeps the lowest validation primary metric ("lowest validation ...")
            "lower_is_better": "highest" not in str(sel.get("rule") or "").lower(),
            "validation": {c: v["validation"][primary] for c, v in cands.items()
                           if primary in (v.get("validation") or {})},
            "test": {c: v["test"][primary] for c, v in cands.items() if primary in (v.get("test") or {})},
            "baselines": baselines, "candidates": list(cands), "kept": kept,
            "kept_is_baseline": kept in baselines if kept else True,
            "complexity": {c: v.get("complexity") for c, v in cands.items()},
            "comparisons": list(sel.get("comparisons") or []), "worse_model_kept": sel.get("worse_model_kept"),
        },
    }


def card_view(card: dict) -> dict:
    """One normalised description of a card of either schema (see the module docstring)."""
    if card_schema(card) == LOADER_SCHEMA:
        return _view_loader_card(card)
    return _view_kind_card(card)


def card_problems(card: dict) -> list[str]:
    """Why a committed card is not a complete registry entry ([] when it is)."""
    try:
        schema = card_schema(card)
    except CardError as e:
        return [str(e)]
    probs = [f"missing field {f}" for f in REQUIRED_CARD_FIELDS[schema] if f not in card]
    if card.get("schema") != schema:
        probs.append(f"schema {card.get('schema')!r} != {schema}")
    v = card_view(card)
    if v["domain"] not in DOMAINS:
        probs.append(f"domain {v['domain_label']!r} maps to {v['domain']!r}, not one of DOMAINS")
    if v["identification"] not in IDENTIFICATION:
        probs.append(f"identification {v['identification_label']!r} is not one of {IDENTIFICATION}")
    if not v["datasets"]:
        probs.append("no dataset named")
    if not v["files"] or not all(v["files"].values()):
        probs.append("no checkpoint file with a SHA-256")
    if not v["limits"].strip():
        probs.append("no extrapolation limits")
    if not v["plans"]:
        probs.append("no evaluation plan")
    if not v["selection"]["primary_metric"]:
        probs.append("no primary metric in the selection record")
    if schema == LOADER_SCHEMA:
        if (card.get("dataset") or {}).get("synthetic", True):
            probs.append("dataset is synthetic or not declared")
        if not card.get("extrapolation_limits"):
            probs.append("no extrapolation limits list")
    return probs


# ---------------------------------------------------------------- loaded models
def x_and_rng(a, b):
    """(X, rng) from arguments given in either order (`sample(X, rng)`; `sample(rng, X)` is also accepted)."""
    if isinstance(a, np.random.Generator) and not isinstance(b, np.random.Generator):
        return b, a
    return a, b


class FittedModel:
    """A loaded, hash-verified fitted model of either card schema.

    INTERFACE_CONTRACTS section 5: ``name``, ``version``, ``features`` (input order of `predict`), ``target``,
    ``metrics`` (held-out test metrics of the kept model), ``baseline_metrics`` (same metrics of the best simple
    baseline), ``provenance`` (the card's provenance, with ``datasets`` and ``identification`` always present),
    ``limits`` (extrapolation limits in words), ``predict(X)``. Also: ``id`` (``name@version``), ``domain`` and
    ``identification`` (canonical: `DOMAINS`, `IDENTIFICATION`), ``datasets``, ``feature_map`` (how simulator
    quantities map onto each input), ``outputs``, ``card`` (as written) and ``view`` (`card_view`).

    A predictor-kind card wraps one portable predictor (`predictors.KINDS`); a loader-class card is an instance of
    the class it names, which receives the checkpoint's arrays (``arrays``; `submodel(prefix)` selects one portable
    estimator) and may override `predict` and add simulator-facing methods.
    """

    def __init__(self, card: dict, predictor=None, path: str | None = None, *, arrays: dict | None = None):
        view = card_view(card)
        self.card = card
        self.view = view
        self.name: str = card["name"]
        self.version: int = int(card["version"])
        self.id = f"{self.name}@{self.version}"
        self.features: list[str] = list(card["features"])
        self.target: str = card["target"]
        self.outputs: list[str] = list(card.get("outputs") or [])
        self.metrics: dict = dict(card.get("metrics") or {})
        self.baseline_metrics: dict = dict(card.get("baseline_metrics") or {})
        prov = dict(card.get("provenance") or {})
        prov.setdefault("datasets", list(view["datasets"]))
        prov.setdefault("identification", view["identification"])
        self.provenance: dict = prov
        self.limits: str = view["limits"]
        self.domain: str = view["domain"]
        self.identification: str = view["identification"]
        self.datasets: list[str] = list(view["datasets"])
        self.feature_map: dict = dict(view["feature_map"])
        self.simulator_mapping: dict = dict(card.get("simulator_mapping") or view["feature_map"])
        self.path = path
        self.arrays: dict = dict(arrays or {})
        self._predictor = predictor

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.id}, kind={self.kind}, domain={self.domain})"

    @property
    def kind(self) -> str:
        """The predictor kind, or the loader class for loader-class cards."""
        return self.card.get("kind") or self.card.get("loader", "")

    @property
    def predictor(self):
        return self._predictor

    def _check(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1 and X.shape[0] == len(self.features):
            X = X[None, :]
        if X.ndim != 2 or X.shape[1] != len(self.features):
            raise ValueError(f"{self.id}: X must be [n, {len(self.features)}] ordered as {self.features}")
        return X

    def submodel(self, prefix: str) -> dict[str, np.ndarray]:
        p = prefix + "/"
        out = {k[len(p):]: v for k, v in self.arrays.items() if k.startswith(p)}
        if not out:
            raise KeyError(f"{self.id} has no sub-model {prefix!r}")
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._check(X)
        if self._predictor is not None:
            return self._predictor.predict(X)
        from . import portable
        return portable.predict(self.submodel("model"), X)

    def sample(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Draw one target value per row (distributional models only)."""
        X, rng = x_and_rng(X, rng)
        if not hasattr(self._predictor, "sample"):
            raise TypeError(f"{self.name} is not a distributional model")
        X = self._check(X)
        return self._predictor.sample(X, rng.random(len(X)))

    def matrix(self, columns: dict[str, np.ndarray]) -> np.ndarray:
        """Build X from named columns (all of `features` must be present)."""
        missing = [f for f in self.features if f not in columns]
        if missing:
            raise KeyError(f"{self.id}: missing feature columns {missing}")
        return np.column_stack([np.asarray(columns[f], dtype=np.float64) for f in self.features])


# ---------------------------------------------------------------- registry
def _models_root(models_dir: str | None) -> str:
    return models_dir or MODELS_DIR


def _parse(model_name: str) -> tuple[str, int | None]:
    if "@" in model_name:
        n, v = model_name.rsplit("@", 1)
        return n, int(v)
    return model_name, None


def model_dir(name: str, models_dir: str | None = None) -> str:
    return os.path.join(_models_root(models_dir), _parse(name)[0])


def read_card(name: str, models_dir: str | None = None) -> dict:
    p = os.path.join(model_dir(name, models_dir), "model_card.json")
    if not os.path.isfile(p):
        raise ModelNotFoundError(f"no fitted model {_parse(name)[0]!r} in {_models_root(models_dir)}")
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def verify_files(card: dict, directory: str) -> list[str]:
    """Checkpoint files whose SHA-256 differs from the card (or that are missing)."""
    bad = []
    for fn, sha in card_files(card).items():
        p = os.path.join(directory, fn)
        if not os.path.exists(p) or sha256_file(p) != sha:
            bad.append(fn)
    return bad


def verify_checkpoint(model_name: str, models_dir: str | None = None) -> dict:
    """Hash check of every checkpoint file of a registered model, without loading it."""
    card = read_card(model_name, models_dir)
    d = model_dir(model_name, models_dir)
    files = {}
    for fn, sha in card_files(card).items():
        p = os.path.join(d, fn)
        actual = sha256_file(p) if os.path.isfile(p) else None
        files[fn] = {"expected": sha, "actual": actual, "bytes": os.path.getsize(p) if actual else 0,
                     "ok": actual is not None and actual == sha}
    return {"model": f"{card['name']}@{card['version']}", "files": files,
            "bytes": sum(f["bytes"] for f in files.values()),
            "largest_bytes": max((f["bytes"] for f in files.values()), default=0),
            "ok": bool(files) and all(f["ok"] for f in files.values())}


def _load_kind(card: dict, d: str) -> FittedModel:
    mod = card.get("module")
    if mod:
        if not mod.startswith("crucible.calibration."):
            raise CardError(f"{card['name']}: predictor module must live in crucible.calibration, not {mod}")
        importlib.import_module(mod)
    cls = _pred.KINDS.get(card.get("kind"))
    if cls is None:
        raise CardError(f"{card['name']}: unknown predictor kind {card.get('kind')!r}")
    with np.load(os.path.join(d, checkpoint_file(card)), allow_pickle=False) as P:
        params = {k: P[k] for k in P.files}
    return FittedModel(card, cls.from_params(params, card), d)


def _load_class(card: dict, d: str) -> FittedModel:
    loader = str(card.get("loader") or "")
    mod_name, _, cls_name = loader.partition(":")
    if not mod_name.startswith("crucible.calibration.") or not cls_name:
        raise CardError(f"{card['name']}: loader must be crucible.calibration.<module>:<Class>, not {loader!r}")
    cls = getattr(importlib.import_module(mod_name), cls_name, None)
    if not (isinstance(cls, type) and issubclass(cls, FittedModel)):
        raise CardError(f"{card['name']}: loader {loader} is not a registry.FittedModel subclass")
    with np.load(os.path.join(d, checkpoint_file(card)), allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    return cls(card, path=d, arrays=arrays)


def load(model_name: str, models_dir: str | None = None) -> FittedModel:
    """Load `name@version` (or `name`, the registered version) after verifying every checkpoint hash.

    Reads only the card and its checkpoint (never a dataset). Raises `ModelNotFoundError` (a KeyError) for an
    unknown name or version, `ChecksumError` (an IOError) when a checkpoint is missing or altered, and `CardError`
    (a ValueError) for a card the registry refuses; all three are `CalibrationError`s."""
    name, version = _parse(model_name)
    d = model_dir(name, models_dir)
    card = read_card(name, models_dir)
    if version is not None and int(card["version"]) != version:
        raise ModelNotFoundError(f"{name}: registered version is {card['version']}, not {version}")
    schema = card_schema(card)
    if not card_files(card):
        raise CardError(f"{name}: the card lists no checkpoint file with a SHA-256")
    bad = verify_files(card, d)
    if bad:
        raise ChecksumError(f"{name}: checkpoint files fail their SHA-256 check: {bad}")
    ip = index_problems(name, models_dir)
    if ip:
        raise ChecksumError(f"{name}: {ip}")
    if schema == LOADER_SCHEMA:
        return _load_class(card, d)
    return _load_kind(card, d)


def list_models(models_dir: str | None = None) -> list[str]:
    """Registered models (every card under the models directory, either schema) as `name@version`."""
    root = _models_root(models_dir)
    out = []
    if not os.path.isdir(root):
        return out
    for n in sorted(os.listdir(root)):
        p = os.path.join(root, n, "model_card.json")
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as fh:
                out.append(f"{n}@{json.load(fh)['version']}")
    return out


def cards(models_dir: str | None = None) -> dict[str, dict]:
    return {m.split("@")[0]: read_card(m.split("@")[0], models_dir) for m in list_models(models_dir)}


def report(model_name: str, models_dir: str | None = None) -> dict:
    """The held-out evaluation report the card points to (as written)."""
    card = read_card(model_name, models_dir)
    rp = card.get("report")
    if not rp:
        raise CardError(f"{card['name']}: the card names no report")
    with open(rp if os.path.isabs(rp) else os.path.join(CRUCIBLE_ROOT, rp), encoding="utf-8") as fh:
        return json.load(fh)


def save_model(name: str, version: int, predictor, card: dict) -> str:
    """Write the checkpoint (`params.npz`) and a `SCHEMA` card; returns the directory. Refuses files over 5 MB."""
    if card.get("domain") not in DOMAINS:
        raise CardError(f"{name}: domain must be one of {sorted(DOMAINS)}")
    if card.get("provenance", {}).get("identification") not in IDENTIFICATION:
        raise CardError(f"{name}: provenance.identification must be one of {IDENTIFICATION}")
    d = model_dir(name)
    os.makedirs(d, exist_ok=True)
    ck = os.path.join(d, "params.npz")
    np.savez_compressed(ck, **{k: np.asarray(v) for k, v in predictor.params().items()})
    if os.path.getsize(ck) > MAX_CHECKPOINT_BYTES:
        os.remove(ck)
        raise CardError(f"{name}: checkpoint exceeds 5 MB")
    full = dict(card)
    full.update({"schema": SCHEMA, "name": name, "version": int(version), "kind": predictor.kind,
                 "checkpoint": "params.npz", "files": {"params.npz": sha256_file(ck)}})
    tmp = os.path.join(d, "model_card.json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(full, fh, indent=1, sort_keys=False, default=_jsonable)
        fh.write("\n")
    os.replace(tmp, os.path.join(d, "model_card.json"))
    return d


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")
