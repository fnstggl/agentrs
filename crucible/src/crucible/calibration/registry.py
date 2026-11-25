"""The fitted-model registry (INTERFACE_CONTRACTS section 5).

A fitted model is a directory `docs/v5/calibration/models/<name>/` holding `model_card.json` and the checkpoint
files it lists with their SHA-256 (each < 5 MB, NumPy `.npz` loaded without pickle). The registry is the set of those
directories: a calibration workstream registers a model by saving it with `save_model`, never by editing this file.
`load("content_consumption@1")` verifies every checkpoint hash and returns a `FittedModel`; world code depends only on
`load`, never on datasets.

Model card (schema `crucible.v5.model_card/1`), fields used by the gates:

* ``name``, ``version``, ``kind`` (a predictor kind in `predictors.KINDS`; ``module`` names a
  `crucible.calibration.*` module to import first when the kind is defined elsewhere)
* ``domain``: one of `DOMAINS` (behavioural domain of MASTER_SPEC Part 5.3)
* ``features`` (ordered input columns), ``feature_map`` (how simulator state maps onto each input), ``target``
* ``metrics`` / ``baseline_metrics``: held-out test metrics of the kept model and of the best simple baseline
* ``candidates``: every model compared, with validation and test metrics; ``selection``: the prespecified rule,
  the primary metric, the selected candidate and why
* ``provenance``: ``datasets`` (names in `datasets()`), ``identification`` (one of `IDENTIFICATION`), population,
  platform, period, ``evaluation_plan`` {path, sha256, commit}, ``fit`` {commit, command, seconds}
* ``limits``: extrapolation limits in words; ``mechanisms``: simulator mechanisms the model grounds
* ``report``: path of the held-out evaluation report (JSON) under `docs/v5/calibration/reports/`
* ``files``: {file name: sha256}
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os

import numpy as np

from . import predictors as _pred

SCHEMA = "crucible.v5.model_card/1"
CRUCIBLE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CAL_DOCS = os.path.join(CRUCIBLE_ROOT, "docs", "v5", "calibration")
MODELS_DIR = os.path.join(CAL_DOCS, "models")
REPORTS_DIR = os.path.join(CAL_DOCS, "reports")
MAX_CHECKPOINT_BYTES = 5 * 1024 * 1024

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
# How a model's effect estimates are identified (MASTER_SPEC 5.2: keep observational and experimental apart).
IDENTIFICATION = ("observational", "randomized_exposure", "randomized_controlled_trial")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class FittedModel:
    """A loaded, hash-verified fitted model. `predict(X)` takes rows ordered as `features`."""

    def __init__(self, card: dict, predictor, path: str):
        self.card = card
        self.name: str = card["name"]
        self.version: int = int(card["version"])
        self.features: list[str] = list(card["features"])
        self.target: str = card["target"]
        self.metrics: dict = dict(card.get("metrics", {}))
        self.baseline_metrics: dict = dict(card.get("baseline_metrics", {}))
        self.provenance: dict = dict(card.get("provenance", {}))
        self.limits: str = card.get("limits", "")
        self.domain: str = card.get("domain", "")
        self.path = path
        self._predictor = predictor

    def __repr__(self) -> str:
        return f"FittedModel({self.name}@{self.version}, kind={self.card.get('kind')}, domain={self.domain})"

    @property
    def kind(self) -> str:
        return self.card["kind"]

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.features):
            raise ValueError(f"{self.name}: X must be [n, {len(self.features)}] ordered as {self.features}")
        return self._predictor.predict(X)

    def sample(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Draw one target value per row (distributional models only)."""
        if not hasattr(self._predictor, "sample"):
            raise TypeError(f"{self.name} is not a distributional model")
        X = np.asarray(X, dtype=float)
        return self._predictor.sample(X, rng.random(len(X)))

    @property
    def predictor(self):
        return self._predictor


def model_dir(name: str) -> str:
    return os.path.join(MODELS_DIR, name)


def _parse(model_name: str) -> tuple[str, int | None]:
    if "@" in model_name:
        n, v = model_name.split("@", 1)
        return n, int(v)
    return model_name, None


def read_card(name: str) -> dict:
    with open(os.path.join(model_dir(name), "model_card.json"), encoding="utf-8") as fh:
        return json.load(fh)


def verify_files(card: dict, directory: str) -> list[str]:
    """Checkpoint files whose SHA-256 differs from the card (or that are missing)."""
    bad = []
    for fn, sha in card.get("files", {}).items():
        p = os.path.join(directory, fn)
        if not os.path.exists(p) or sha256_file(p) != sha:
            bad.append(fn)
    return bad


def load(model_name: str) -> FittedModel:
    """Load `name@version` (or `name`, the registered version) after verifying every checkpoint hash."""
    name, version = _parse(model_name)
    d = model_dir(name)
    if not os.path.isfile(os.path.join(d, "model_card.json")):
        raise KeyError(f"no fitted model {name!r} in {MODELS_DIR}")
    card = read_card(name)
    if version is not None and int(card["version"]) != version:
        raise KeyError(f"{name}: registered version is {card['version']}, not {version}")
    bad = verify_files(card, d)
    if bad:
        raise IOError(f"{name}: checkpoint files fail their SHA-256 check: {bad}")
    mod = card.get("module")
    if mod:
        if not mod.startswith("crucible.calibration."):
            raise ValueError(f"{name}: predictor module must live in crucible.calibration, not {mod}")
        importlib.import_module(mod)
    cls = _pred.KINDS.get(card["kind"])
    if cls is None:
        raise KeyError(f"{name}: unknown predictor kind {card['kind']!r}")
    with np.load(os.path.join(d, card["checkpoint"]), allow_pickle=False) as P:
        params = {k: P[k] for k in P.files}
    return FittedModel(card, cls.from_params(params, card), d)


def list_models() -> list[str]:
    """Registered models as `name@version`."""
    out = []
    if not os.path.isdir(MODELS_DIR):
        return out
    for n in sorted(os.listdir(MODELS_DIR)):
        p = os.path.join(MODELS_DIR, n, "model_card.json")
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as fh:
                out.append(f"{n}@{json.load(fh)['version']}")
    return out


def cards() -> dict[str, dict]:
    return {m.split("@")[0]: read_card(m.split("@")[0]) for m in list_models()}


def save_model(name: str, version: int, predictor, card: dict) -> str:
    """Write the checkpoint (`params.npz`) and the model card; returns the directory. Refuses files over 5 MB."""
    if card.get("domain") not in DOMAINS:
        raise ValueError(f"{name}: domain must be one of {sorted(DOMAINS)}")
    if card.get("provenance", {}).get("identification") not in IDENTIFICATION:
        raise ValueError(f"{name}: provenance.identification must be one of {IDENTIFICATION}")
    d = model_dir(name)
    os.makedirs(d, exist_ok=True)
    ck = os.path.join(d, "params.npz")
    np.savez_compressed(ck, **{k: np.asarray(v) for k, v in predictor.params().items()})
    if os.path.getsize(ck) > MAX_CHECKPOINT_BYTES:
        os.remove(ck)
        raise ValueError(f"{name}: checkpoint exceeds 5 MB")
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
