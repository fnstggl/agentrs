"""Predictor kinds added for the E4 models (registered into `predictors.KINDS` on import; cards name this module in
their ``module`` field so that `registry.load` imports it first). NumPy only."""
from __future__ import annotations

import numpy as np

from . import predictors as P


def _inv_link(z: np.ndarray, link: str) -> np.ndarray:
    if link == "identity":
        return z
    if link == "log":
        return np.exp(np.clip(z, -40.0, 40.0))
    if link == "logit":
        return P._sigmoid(z)
    raise ValueError(f"unknown link {link}")


class Linear:
    """Generalised linear predictor on standardised features: inverse_link(z), z = ((X - mean) / scale) w + b."""
    kind = "glm"

    def __init__(self, w, b, mean, scale, link: str = "identity"):
        self.w = np.asarray(w, float)
        self.b = float(b)
        self.mean = np.asarray(mean, float)
        self.scale = np.asarray(scale, float)
        self.link = str(link)

    def params(self) -> dict:
        return {"w": self.w, "b": np.array([self.b]), "mean": self.mean, "scale": self.scale,
                "link": np.array([{"identity": 0, "log": 1, "logit": 2}[self.link]])}

    @classmethod
    def from_params(cls, Pm, card=None):
        link = ("identity", "log", "logit")[int(Pm["link"][0])]
        return cls(Pm["w"], float(Pm["b"][0]), Pm["mean"], Pm["scale"], link)

    def raw(self, X) -> np.ndarray:
        return ((np.asarray(X, float) - self.mean) / self.scale) @ self.w + self.b

    def predict(self, X) -> np.ndarray:
        return _inv_link(self.raw(X), self.link)


class TreeRegressor(P.TreeEnsemble):
    """Exported HistGradientBoostingRegressor: inverse_link(sum of trees + baseline)."""
    kind = "gbdt_regression"

    def __init__(self, *a, link: str = "identity", **k):
        super().__init__(*a, **k)
        self.link = link

    @classmethod
    def from_sklearn(cls, model, link: str = "identity") -> "TreeRegressor":
        te = P.TreeEnsemble.from_sklearn(model)
        out = cls(te.baseline, te.roots, te.feature, te.threshold, te.left, te.right, te.is_leaf, te.value,
                  te.missing_left, te.offset, link=link)
        return out

    def params(self) -> dict:
        d = super().params()
        d["link"] = np.array([{"identity": 0, "log": 1, "logit": 2}[self.link]])
        return d

    @classmethod
    def from_params(cls, Pm, card=None):
        link = ("identity", "log", "logit")[int(Pm["link"][0])]
        return cls(float(Pm["baseline"][0]), Pm["roots"], Pm["feature"], Pm["threshold"], Pm["left"], Pm["right"],
                   Pm["is_leaf"], Pm["value"], Pm["missing_left"], float(Pm["offset"][0]), link=link)

    def raw(self, X) -> np.ndarray:
        sk = getattr(self, "_sk", None)
        if sk is not None:
            return np.asarray(sk._raw_predict(np.asarray(X, float))).ravel() + self.offset
        self._sk = None
        return P.TreeEnsemble.logit(self, X)

    def predict(self, X) -> np.ndarray:
        return _inv_link(self.raw(X), self.link)


class Persistence:
    """Rule baseline: prediction = scale * inverse_transform(X[:, col]) (e.g. stars in the last 24 h / 24)."""
    kind = "persistence"

    def __init__(self, col: int, scale: float, transform: str = "expm1"):
        self.col = int(col)
        self.scale = float(scale)
        self.transform = transform

    def params(self) -> dict:
        return {"col": np.array([self.col]), "scale": np.array([self.scale]),
                "transform": np.array([{"identity": 0, "expm1": 1}[self.transform]])}

    @classmethod
    def from_params(cls, Pm, card=None):
        return cls(int(Pm["col"][0]), float(Pm["scale"][0]), ("identity", "expm1")[int(Pm["transform"][0])])

    def predict(self, X) -> np.ndarray:
        v = np.asarray(X, float)[:, self.col]
        return self.scale * (np.expm1(v) if self.transform == "expm1" else v)


class Subset:
    """A predictor applied to a subset of the feature columns (baselines on a few features)."""
    kind = "subset"

    def __init__(self, inner, cols):
        self.inner = inner
        self.cols = np.asarray(cols, np.int64)

    def params(self) -> dict:
        d = {f"inner/{k}": v for k, v in self.inner.params().items()}
        d["cols"] = self.cols
        d["inner_kind"] = np.array([self.inner.kind])
        return d

    @classmethod
    def from_params(cls, Pm, card=None):
        kind = str(Pm["inner_kind"][0])
        inner = P.KINDS[kind].from_params({k[6:]: v for k, v in Pm.items() if k.startswith("inner/")}, card)
        return cls(inner, Pm["cols"])

    def predict(self, X) -> np.ndarray:
        return self.inner.predict(np.asarray(X, float)[:, self.cols])

    def __getattr__(self, name):
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)


for _cls in (Linear, TreeRegressor, Persistence, Subset):
    if _cls.kind not in P.KINDS:
        P.register_kind(_cls.kind, _cls)
