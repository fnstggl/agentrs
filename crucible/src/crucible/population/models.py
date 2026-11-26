"""Class 1: vectorised behavioural decision models (MASTER_SPEC Part 3.2, class 1).

Two kinds of decision model drive class 1 (and the class 2 decisions their policy does not take):

* **Fitted runtime decision models** (workstream E, `crucible.calibration.runtime`, contracts in
  docs/v5/calibration/RUNTIME.md), used through the runtime only - never `calibration.load`, never a silent fallback
  (RUNTIME.md rule 4): `daily_return@1` (active today) and `extra_sessions@1` (sessions per active day) in
  `population.activation`; `article_open_semantic@1` (which article of a page's slate is opened, scope
  `content_kind="article"`) and `customer_return@1` (whether a customer buys from the same shop again within 90 days)
  in `population.web`. `FITTED` lists them with the population decision each one takes.
* **Plausible decisions** below (`Decision`): logistic models with hand-set coefficients for the decisions no fitted
  model covers (fidelity "plausible, unvalidated", register rows `pop.decision.<name>`). Each maps named feature
  arrays (one row per individual) to a probability from the individual's own state (preferences, satisfaction,
  habits, history, money, attention), the opportunity in front of it (the page it read, the price it saw), its
  history with that opportunity and its social context, so two individuals in the same situation decide differently
  and the same individual decides differently as its state evolves.

`status(world)` says which model each decision uses in a world.
"""
from __future__ import annotations

from dataclasses import dataclass

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
    fidelity: str = "plausible, unvalidated"

    @property
    def features(self) -> tuple[str, ...]:
        return tuple(self.coef)

    def model_id(self) -> str:
        return f"plausible:{self.name}@1"

    @property
    def fidelity_id(self) -> str:
        return f"pop.decision.{self.name}"

    def logit(self, F: dict[str, np.ndarray]) -> np.ndarray:
        n = len(next(iter(F.values())))
        z = np.full(n, self.bias, np.float64)
        for k, c in self.coef.items():
            z += c * np.asarray(F[k], np.float64)
        return z

    def prob(self, F: dict[str, np.ndarray]) -> np.ndarray:
        return sigmoid(self.logit(F))

    def sample(self, F: dict[str, np.ndarray], rng: np.random.Generator) -> np.ndarray:
        p = self.prob(F)
        return rng.random(p.shape[0]) < p


def _decisions() -> dict[str, Decision]:
    D = Decision
    return {d.name: d for d in [
        D("continue", "read another page of the same site",
          {"page_utility": 1.6, "attention_ratio": 1.2, "quality_sensitivity": 0.4, "novelty_seeking": 0.3,
           "page_richness": 0.6},
          bias=-0.9),
        D("evaluate_product", "open a product's page when a site lists products",
          {"page_relevance": 1.4, "goal_buy": 1.4, "impulsiveness": 0.8, "shopping_skill": 0.4,
           "attention_ratio": 0.6},
          bias=-2.6),
        D("purchase", "buy the product evaluated",
          {"price_ratio": 1.4, "page_relevance": 1.2, "goal_buy": 1.1, "wallet_cover_log": 0.35,
           "price_sensitivity": -1.1, "resource_satisfaction": 0.8, "belief_shops": 0.9, "impulsiveness": 0.7},
          bias=-4.6),
        D("abandon", "leave a resource that no longer provides value",
          {"tolerance_gap": 6.0, "loyalty": -1.2, "resource_visits_log": -0.25},
          bias=-0.4),
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


# the fitted runtime decision models the population uses by default: runtime decision id -> population decision
FITTED = {"daily_return@1": "active today (population.activation, batch pop.day_active)",
          "extra_sessions@1": "sessions on an active day (population.activation, batch pop.day_sessions)",
          "article_open_semantic@1": "open an article of the page's slate (population.web, scope content_kind=article)",
          "customer_return@1": "buy again from the same shop within 90 days (population.web, intention + wake-up)"}


def build(cfg: dict | None = None) -> dict[str, Decision]:
    """The plausible class 1 decisions of this world (the fitted ones are used through crucible.calibration.runtime
    by the code that takes them; see FITTED)."""
    return _decisions()


def status(world) -> dict:
    """{decision: {model, fidelity, features}} in use in this world (plausible decisions and fitted runtime ones)."""
    ds = world.runtime.get("pop.models") or {}
    out = {k: {"model": d.model_id(), "fidelity": d.fidelity, "features": list(d.features)} for k, d in ds.items()}
    for rid, what in FITTED.items():
        out[rid] = {"model": f"runtime:{rid}", "fidelity": "empirically fitted (crucible.calibration.runtime)",
                    "features": [], "decision": what}
    return out
