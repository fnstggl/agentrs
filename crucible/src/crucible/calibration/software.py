"""Software collaboration and adoption models (workstream E2), loaded through the registry.

* `load("software_deep_engagement@1")` -> `DeepEngagementModel`: probability that a developer's first light
  interaction with a repository (a star) is followed, within 24 hours, by deeper engagement by the same developer
  on the same repository (fork, issue or issue comment, pull request or review).
* `load("software_repo_growth@1")` -> `RepoGrowthModel`: expected number of new stars a repository receives on the
  next day, given its activity today (plus a negative-binomial dispersion for sampling).

Both are fitted on GH Archive (public GitHub, 2024-10-07..10 UTC); both are observational. Count features enter as
log1p(count); `matrix_from_counts` applies that transform to raw simulator counts in the order of `features`.
This module never reads a dataset.
"""
from __future__ import annotations

import numpy as np

from . import portable
from .registry_e2 import FittedModel

# Feature definitions shared by the fitting script and the simulator mapping (name -> transform, meaning).
DEEP_FEATURES = [
    ("repo_stars_24h", "log1p", "stars the repository received in the 24 complete hours before the star's hour"),
    ("repo_forks_24h", "log1p", "forks of the repository in that window"),
    ("repo_issue_events_24h", "log1p", "issues opened/closed/commented on the repository in that window"),
    ("repo_pr_events_24h", "log1p", "pull requests and reviews on the repository in that window"),
    ("repo_pushes_24h", "log1p", "pushes (commit batches) to the repository in that window"),
    ("repo_releases_24h", "log1p", "releases published in that window"),
    ("repo_events_24h", "log1p", "all public events on the repository in that window"),
    ("repo_org", "raw", "1 if the repository is owned by an organisation"),
    ("repo_created_in_window", "raw", "1 if the repository was created (observed) before the star within the data window"),
    ("repo_age_hours", "log1p", "hours since creation when repo_created_in_window, else 0"),
    ("actor_events_24h", "log1p", "all public events by the developer in the 24 complete hours before the star's hour"),
    ("actor_stars_24h", "log1p", "stars given by the developer in that window"),
    ("actor_forks_24h", "log1p", "forks made by the developer in that window"),
    ("actor_issue_events_24h", "log1p", "issue events by the developer in that window"),
    ("actor_pr_events_24h", "log1p", "pull-request events by the developer in that window"),
    ("actor_pushes_24h", "log1p", "pushes by the developer in that window"),
    ("actor_repo_events_24h", "log1p", "the developer's events on this repository in that window"),
    ("hour_sin", "raw", "sin(2 pi h / 24), h = UTC hour of the star"),
    ("hour_cos", "raw", "cos(2 pi h / 24)"),
]
GROWTH_FEATURES = [
    ("stars_today", "log1p", "stars received today (non-bot accounts)"),
    ("forks_today", "log1p", "forks today (non-bot accounts)"),
    ("issue_events_today", "log1p", "issue events today"),
    ("pr_events_today", "log1p", "pull-request events today"),
    ("pushes_today", "log1p", "pushes today"),
    ("releases_today", "log1p", "releases today"),
    ("events_today", "log1p", "all public events today"),
    ("stars_yesterday", "log1p", "stars received the previous day"),
    ("created_today", "raw", "1 if the repository was created today"),
    ("created_yesterday", "raw", "1 if the repository was created the previous day"),
    ("repo_org", "raw", "1 if owned by an organisation"),
]


def transform(spec, raw: dict[str, np.ndarray]) -> np.ndarray:
    cols = []
    for name, tf, _ in spec:
        v = np.asarray(raw[name], dtype=np.float64)
        cols.append(np.log1p(v) if tf == "log1p" else v)
    return np.column_stack(cols)


class _Counts(FittedModel):
    SPEC: list = []

    def matrix_from_counts(self, raw: dict[str, np.ndarray]) -> np.ndarray:
        """X from raw simulator quantities (counts, flags, hour terms) named as in `features`."""
        missing = [n for n, _, _ in self.SPEC if n not in raw]
        if missing:
            raise KeyError(f"{self.id}: missing raw quantities {missing}")
        return transform(self.SPEC, raw)

    def candidate_predict(self, candidate: str, X: np.ndarray) -> np.ndarray:
        return portable.predict(self.submodel(f"cand/{candidate}"), self._check(X))


class DeepEngagementModel(_Counts):
    """P(deeper engagement within 24 h | first star), observational."""
    SPEC = DEEP_FEATURES


class RepoGrowthModel(_Counts):
    """Expected next-day stars of a repository; NB2 dispersion for sampling, observational."""
    SPEC = GROWTH_FEATURES

    @property
    def dispersion(self) -> float:
        return float(self.arrays["nb_alpha"][0])

    def sample(self, rng: np.random.Generator, X: np.ndarray) -> np.ndarray:
        """Draw next-day star counts: Poisson-gamma (negative binomial, variance mu + alpha mu^2)."""
        mu = np.maximum(self.predict(X), 1e-12)
        r = 1.0 / self.dispersion
        return rng.poisson(rng.gamma(shape=r, scale=mu / r))
