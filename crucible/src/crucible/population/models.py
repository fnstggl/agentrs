"""Class 1: vectorised behavioural decision models (MASTER_SPEC Part 3.2, class 1).

Each `Decision` maps named feature arrays (one row per individual in a cohort) to a probability. The features come
from the individual's own state (preferences, satisfaction, habits, history, money, attention), the opportunity in
front of it (the page it read, the price it saw), its history with that opportunity and its social context, so two
individuals in the same situation decide differently and the same individual decides differently as its state
evolves.

Calibration plug-in (workstream E, contract section 5). `Decision.calibration` lists fitted-model names to try
through `crucible.calibration.load(name)`. A fitted model is used when every name in its `features` list is a
feature this decision documents (`FEATURES[name]`): X is built column by column in the fitted model's order and
`FittedModel.predict(X)` gives the probability. Otherwise the decision's own model is used. `status(world)` says which
is in use. Until E's checkpoints exist, every decision is the clearly labelled plausible model below (fidelity
"plausible, unvalidated"; see docs/v5/FIDELITY_REGISTER.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# documented features (name -> meaning); every Decision's features are drawn from here
FEATURES: dict[str, str] = {
    "attention_ratio": "attention minutes left today / daily budget (0..1)",
    "satisfaction": "overall satisfaction, EMA of experienced session utility (-1..1, starts 0.5)",
    "habit_max": "strongest platform habit (visit propensity)",
    "habit_activity": "habit for the chosen activity",
    "experience_log": "log1p(cumulative browsing experience in sessions)",
    "in_commitment": "1 if the individual is inside its daily commitment (work) window",
    "woken": "1 if the activation is a triggered wake-up (message, notification)",
    "hour_sin": "sin(2 pi local hour / 24)",
    "hour_cos": "cos(2 pi local hour / 24)",
    "novelty_seeking": "preference: novelty seeking (0..1)",
    "error_tolerance": "preference: tolerance of failures (0..1)",
    "quality_sensitivity": "preference: weight on page richness (0..1)",
    "price_sensitivity": "preference: price sensitivity (0..1)",
    "social_conformity": "preference: weight on contacts' opinions (0..1)",
    "loyalty": "preference: loyalty to known resources (0..1)",
    "sociability": "preference: sociability (0..1)",
    "impulsiveness": "preference: impulsiveness (0..1)",
    "page_utility": "utility experienced on the page just read",
    "page_relevance": "cosine of interests and the page's content vector",
    "page_richness": "richness of the page read: sqrt(min(1, chars / 2000)) x distinct-token share",
    "resource_satisfaction": "personal satisfaction with this resource (EMA of visit utilities)",
    "resource_visits_log": "log1p(personal visits to this resource)",
    "price_ratio": "(willingness to pay - price) / willingness to pay, clipped to [-3, 1]",
    "wallet_cover_log": "log(spendable money / price), clipped to [-5, 5]",
    "goal_buy": "1 if a current goal is buying",
    "goal_social": "1 if a current goal is socialising",
    "shopping_skill": "skill: shopping (0..1)",
    "writing_skill": "skill: writing (0..1)",
    "board_relevance": "cosine of interests and the community's content",
    "contacts_in_community": "share of contacts who are members of the community",
    "belief_shops": "belief: shops are trustworthy (0..1)",
    "belief_recommendations": "belief: recommendations are useful (0..1)",
    "tolerance_gap": "abandon threshold minus personal satisfaction with the resource",
}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


@dataclass
class Decision:
    name: str
    doc: str
    coef: dict[str, float]
    bias: float
    calibration: tuple[str, ...] = ()
    fidelity: str = "plausible, unvalidated"
    fitted: object | None = field(default=None, repr=False)

    @property
    def features(self) -> tuple[str, ...]:
        return tuple(self.coef)

    def model_id(self) -> str:
        if self.fitted is not None:
            return f"fitted:{getattr(self.fitted, 'name', '?')}@{getattr(self.fitted, 'version', '?')}"
        return f"plausible:{self.name}@1"

    def logit(self, F: dict[str, np.ndarray]) -> np.ndarray:
        n = len(next(iter(F.values())))
        z = np.full(n, self.bias, np.float64)
        for k, c in self.coef.items():
            z += c * np.asarray(F[k], np.float64)
        return z

    def prob(self, F: dict[str, np.ndarray]) -> np.ndarray:
        if self.fitted is not None:
            X = np.stack([np.asarray(F[f], np.float64) for f in self.fitted.features], axis=1)
            return np.clip(np.asarray(self.fitted.predict(X), np.float64).reshape(-1), 0.0, 1.0)
        return sigmoid(self.logit(F))

    def sample(self, F: dict[str, np.ndarray], rng: np.random.Generator) -> np.ndarray:
        p = self.prob(F)
        return rng.random(p.shape[0]) < p


def _decisions() -> dict[str, Decision]:
    D = Decision
    return {d.name: d for d in [
        D("session_start", "start a session when activated",
          {"attention_ratio": 2.2, "satisfaction": 0.9, "habit_max": 0.35, "in_commitment": -1.6, "woken": 1.5,
           "experience_log": 0.12, "hour_cos": -0.3},
          bias=-0.6, calibration=("browse_session@1",)),
        D("continue", "read another page of the same site",
          {"page_utility": 1.6, "attention_ratio": 1.2, "quality_sensitivity": 0.4, "novelty_seeking": 0.3,
           "page_richness": 0.6},
          bias=-0.9, calibration=("content_consumption@1", "continue_watching@1")),
        D("evaluate_product", "open a product's page when a site lists products",
          {"page_relevance": 1.4, "goal_buy": 1.4, "impulsiveness": 0.8, "shopping_skill": 0.4,
           "attention_ratio": 0.6},
          bias=-2.6),
        D("purchase", "buy the product evaluated",
          {"price_ratio": 1.4, "page_relevance": 1.2, "goal_buy": 1.1, "wallet_cover_log": 0.35,
           "price_sensitivity": -1.1, "resource_satisfaction": 0.8, "belief_shops": 0.9, "impulsiveness": 0.7},
          bias=-4.6, calibration=("commercial_response@1", "purchase@1")),
        D("abandon", "leave a resource that no longer provides value",
          {"tolerance_gap": 6.0, "loyalty": -1.2, "resource_visits_log": -0.25},
          bias=-0.4, calibration=("churn@1",)),
        D("recommend", "recommend a resource to contacts",
          {"page_utility": 2.2, "sociability": 1.5, "resource_satisfaction": 0.8},
          bias=-4.2),
        D("join_community", "join a community after reading it",
          {"board_relevance": 2.5, "sociability": 1.4, "contacts_in_community": 1.5, "goal_social": 0.8},
          bias=-3.6),
        D("follow_creator", "follow the author of a post read",
          {"page_utility": 2.0, "sociability": 1.2, "goal_social": 0.6},
          bias=-3.8),
        D("participate", "reply in a discussion read",
          {"sociability": 2.0, "writing_skill": 1.2, "page_utility": 1.2, "goal_social": 0.8},
          bias=-4.2),
    ]}


def build(cfg: dict | None = None) -> dict[str, Decision]:
    """The decisions of this world, with fitted models plugged in where `crucible.calibration.load` offers one whose
    features are all documented here (config `calibration: false` turns the lookup off)."""
    ds = _decisions()
    if (cfg or {}).get("calibration", True) is False:
        return ds
    try:
        from .. import calibration as cal
        load = getattr(cal, "load", None)
    except ImportError:
        load = None
    if load is None:
        return ds
    for d in ds.values():
        for name in d.calibration:
            try:
                m = load(name)
            except Exception:        # noqa: BLE001 - absent checkpoint: keep the plausible model
                continue
            feats = list(getattr(m, "features", []) or [])
            if feats and all(f in FEATURES for f in feats):
                d.fitted = m
                d.fidelity = "empirically fitted (crucible.calibration)"
                break
    return ds


def status(world) -> dict:
    """{decision: {model, fidelity, features}} in use in this world."""
    ds = world.runtime.get("pop.models") or {}
    return {k: {"model": d.model_id(), "fidelity": d.fidelity, "features": list(d.features)} for k, d in ds.items()}
