"""Runtime decision models for the simulator: fitted behaviour models with an explicit feature contract in simulator
terms (MASTER_SPEC Part 5; DECISION_LOG D-15). Owner: workstream E. Consumers: the unified population (wave 2).

A *decision model* is a fitted, hash-verified checkpoint of the registry (`crucible.calibration.load`) wrapped in a
contract that says, input by input, which simulator quantity it reads, in which unit, how that quantity is turned into
the fitted model's feature (a fixed, row-wise formula: the same formula the fitting code used on the data), and which
range of each feature the data covered. It returns calibrated probabilities, quantiles or rates for a batch of
individuals and carries its decision id, the fitted model id and version and its fidelity-register id, so every
recorded decision event can name the model that produced it (decision provenance, D-15).

    from crucible.calibration import runtime
    runtime.specs()                         # {decision id: JSON contract}, without loading any checkpoint
    dm = runtime.get("content_open@1")      # loads content_consumption@1 (SHA-256 verified)
    res = dm.evaluate({"own_views_7d": ..., ...})   # dict of equal-length 1-D arrays, one row per individual
    res.values, res.extrapolated, res.provenance()
    draws = dm.sample(inputs, ctx.rng("pop.decide"))   # draws with the caller's own RNG stream

Rules this module enforces (and `tests/test_v5_calibration_runtime.py` checks):

* **Row independence.** Every transform is elementwise: a row's output depends only on that row's inputs, never on
  the other rows of the batch. There is no rank-within-batch or quantile-within-batch mapping anywhere (the old
  `social.consumption.CalibratedOpen.qmap` made opening probabilities invariant to uniformly better content and
  dependent on who else was in the batch; it is gone). A within-slate statistic (``text_cos_z`` of the news model)
  is computed from the individual's *own* slate, which is part of that individual's decision context, and is passed
  in as an input, never computed across individuals here.
* **Absolute inputs.** Inputs are counts, rates, durations, cosines and flags in the units stated. Where the data
  measured a quantity on a different scale than the simulator (KuaiRand logs only plays of a 7,583-video pool), the
  contract says so and offers a fixed, declared scale factor (``count_scale``), never a data-dependent rescaling.
* **Declared support.** Each result flags rows whose features fall outside the range the data covered
  (``extrapolated``); `evaluate(..., strict=True)` refuses them. Values outside an input's hard domain (negative
  counts, probabilities outside [0, 1]) always raise `ContractError`.
* **No silent fallback.** An unknown decision id, a missing input, a model whose checkpoint fails its hash, or a
  model whose contract cannot be computed in a simulated world (``status: offline_only``) raises; nothing returns a
  plausible default instead.
* **Provenance.** `Result.provenance()` is JSON-native: {decision, model, version, fidelity, rows, extrapolated}.

The extrapolation limits of every decision model are in `spec["limits"]` and, in full, in its model card and
held-out report; `docs/v5/calibration/RUNTIME.md` documents every contract for the population workstream.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .registry import CalibrationError, load as _load

# ---------------------------------------------------------------- contract primitives


class ContractError(CalibrationError, ValueError):
    """Inputs that violate a decision model's contract, or a model that cannot be used at run time."""


@dataclass(frozen=True)
class Input:
    name: str                 # simulator quantity
    meaning: str              # its definition in simulator terms
    unit: str
    kind: str                 # individual | item | pair | context | slate | latent
    lo: float | None = None   # hard domain (inclusive); None = unbounded
    hi: float | None = None
    integer: bool = False

    def as_dict(self) -> dict:
        return {"name": self.name, "meaning": self.meaning, "unit": self.unit, "kind": self.kind,
                "domain": [self.lo, self.hi], "integer": self.integer}


@dataclass
class Spec:
    id: str                   # decision id, e.g. "content_open@1"
    model: str                # registered fitted model, e.g. "content_consumption@1"
    use: str                  # what the decision is, and what it may be used for
    output: str               # probability | quantiles | rate | arm_probabilities
    inputs: tuple
    build: object             # callable(inputs dict, model) -> X [n, d] in the model's feature order (row-wise)
    limits: str
    status: str = "ready"     # ready | offline_only
    status_reason: str = ""
    output_doc: str = ""
    params: dict = field(default_factory=dict)

    @property
    def fidelity_id(self) -> str:
        return f"runtime:{self.id}"

    def as_dict(self) -> dict:
        return {"id": self.id, "model": self.model, "fidelity_id": self.fidelity_id, "use": self.use,
                "output": self.output, "output_doc": self.output_doc, "status": self.status,
                "status_reason": self.status_reason, "inputs": [i.as_dict() for i in self.inputs],
                "params": dict(self.params), "limits": self.limits}


def _f(d: dict, name: str) -> np.ndarray:
    return np.asarray(d[name], dtype=np.float64)


def _hour_terms(hour: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = np.floor(np.mod(hour, 24.0))
    return np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)


# ---------------------------------------------------------------- KuaiRand feed contracts (E1 models)
_KR_PRIOR_N, _KR_TAG_PRIOR_N = 20.0, 5.0          # kuairand.PRIOR_N, kuairand.TAG_PRIOR_N

KR_INPUTS = {
    "own_views_7d": Input("own_views_7d", "items the individual consumed (a view: watched beyond the platform's view "
                          "threshold) on this platform in the previous 7 simulated days", "count", "individual", 0),
    "own_opens_before": Input("own_opens_before", "feed items the individual opened on this platform before today "
                              "(whole history)", "count", "individual", 0),
    "own_impressions_before": Input("own_impressions_before", "feed impressions (items reached in the individual's "
                                    "feed) before today (whole history)", "count", "individual", 0),
    "item_views_7d": Input("item_views_7d", "views of the item in the previous 7 days, all individuals", "count",
                           "item", 0),
    "active_individuals_7d": Input("active_individuals_7d", "distinct individuals with at least one feed impression "
                                   "on the platform in the previous 7 days", "count", "context", 0),
    "item_opens_before": Input("item_opens_before", "opens of the item before today, all individuals", "count",
                               "item", 0),
    "item_impressions_before": Input("item_impressions_before", "impressions of the item before today, all "
                                     "individuals", "count", "item", 0),
    "item_age_days": Input("item_age_days", "days since the item was published (now - published_at) / 86400",
                           "days", "item", 0),
    "item_duration_s": Input("item_duration_s", "the content item's duration (artifact metadata measured from the "
                             "bytes, never declared)", "seconds", "item", 0),
    "own_topic_views": Input("own_topic_views", "the individual's past views (before today) of items whose primary "
                             "topic equals this item's primary topic; primary topic = nearest topic centroid of the "
                             "item's embedding in the shared semantic space (D-16)", "count", "pair", 0),
    "own_views_before": Input("own_views_before", "the individual's past views before today, all topics", "count",
                              "individual", 0),
    "topic_population_share": Input("topic_population_share", "share of all past views on the platform that went to "
                                    "this item's primary topic (smoothed: (topic views + 1) / (views + topics))",
                                    "fraction", "context", 0, 1),
    "own_creator_views": Input("own_creator_views", "the individual's past views (before today) of this item's "
                               "creator", "count", "pair", 0),
    "own_item_exposures": Input("own_item_exposures", "times this item was in the individual's feed before today",
                                "count", "pair", 0),
    "own_item_consumed": Input("own_item_consumed", "1 if the individual viewed this item before today",
                               "flag", "pair", 0, 1),
    "local_hour": Input("local_hour", "local simulated hour of the decision, 0 <= h < 24 (floored)", "hour",
                        "context", 0, 24),
    "consumed": Input("consumed", "1 if the individual opened / kept watching the item (the content_open draw), 0 "
                      "for a skip", "flag", "individual", 0, 1),
}


def _kr_columns(d: dict, prior: float, count_scale: float) -> dict[str, np.ndarray]:
    """The E1 KuaiRand features from simulator counts (kuairand.choice_frame formulas)."""
    s = _f(d, "topic_population_share")
    tag_share = (_f(d, "own_topic_views") + _KR_TAG_PRIOR_N * s) / (_f(d, "own_views_before") + _KR_TAG_PRIOR_N)
    h_sin, h_cos = _hour_terms(_f(d, "local_hour"))
    return {
        "user_log_plays_7d": np.log1p(count_scale * _f(d, "own_views_7d")),
        "user_click_rate": (_f(d, "own_opens_before") + _KR_PRIOR_N * prior) /
                           (_f(d, "own_impressions_before") + _KR_PRIOR_N),
        "item_log_pop_7d": np.log1p(1000.0 * _f(d, "item_views_7d") / np.maximum(_f(d, "active_individuals_7d"), 1)),
        "item_click_rate": (_f(d, "item_opens_before") + _KR_PRIOR_N * prior) /
                           (_f(d, "item_impressions_before") + _KR_PRIOR_N),
        "item_log_age_days": np.log1p(_f(d, "item_age_days")),
        "item_log_duration_s": np.log1p(_f(d, "item_duration_s")),
        "user_tag_share": tag_share,
        "user_tag_lift": np.log(tag_share / np.maximum(s, 1e-12)),
        "creator_familiarity": np.log1p(_f(d, "own_creator_views")),
        "item_seen_before": np.log1p(_f(d, "own_item_exposures")),
        "item_played_before": (_f(d, "own_item_consumed") > 0).astype(np.float64),
        "hour_sin": h_sin, "hour_cos": h_cos,
        **({"consumed": _f(d, "consumed")} if "consumed" in d else {}),
    }


def _kr_build(d: dict, model, spec: Spec) -> np.ndarray:
    cols = _kr_columns(d, float(model.card["prior_rate"]), float(spec.params.get("count_scale", 1.0)))
    return np.column_stack([cols[f] for f in model.features])


_KR_CHOICE_IN = ("own_views_7d", "own_opens_before", "own_impressions_before", "item_views_7d",
                 "active_individuals_7d", "item_opens_before", "item_impressions_before", "item_age_days",
                 "item_duration_s", "own_topic_views", "own_views_before", "topic_population_share",
                 "own_creator_views", "own_item_exposures", "own_item_consumed", "local_hour")
_KR_WATCH_IN = ("consumed", "item_duration_s", "own_topic_views", "own_views_before", "topic_population_share",
                "own_creator_views", "own_views_7d", "own_opens_before", "own_impressions_before",
                "item_opens_before", "item_impressions_before", "item_views_7d", "active_individuals_7d",
                "item_age_days", "own_item_consumed")
_KR_LIMITS = ("KuaiRand-Pure: 27,285 users of the Kuaishou short-video app (China), featured feed, 2022-04-08.."
              "05-08 (one app, one month including the Labour Day holiday). Autoplay short video (6 s - 20 min) only; "
              "not text, articles, long video or other cultures. 'Consumed' in the data means watching beyond about "
              "7 s. Feature effects are associations among the shown items, not causal effects of popularity or "
              "interest. Counts in the data cover only plays of the 7,583 pool videos: own_views_7d is multiplied "
              "by the declared count_scale (default 1.0; a world whose feeds show a wider catalogue should declare "
              "the share it treats as comparable) - never rescaled from the batch. Item ages beyond 30 days are "
              "outside the data.")


# ---------------------------------------------------------------- activity contracts (E1 models)
RET_INPUTS = (
    Input("active_days_7d", "days among the previous 7 on which the individual had at least one feed impression on "
          "the platform", "days", "individual", 0, 7, True),
    Input("days_since_last_active", "days since the last active day, capped at 14 (14 when never active)", "days",
          "individual", 0, 14),
    Input("feed_exposures_7d", "feed impressions in the previous 7 days", "count", "individual", 0),
    Input("sessions_7d", "sessions in the previous 7 days (a session ends after 30 idle minutes)", "count",
          "individual", 0),
    Input("active_yesterday", "1 if active the previous day", "flag", "individual", 0, 1),
    Input("weekend", "1 on Saturdays and Sundays of the simulated calendar", "flag", "context", 0, 1),
)


def _ret_build(d: dict, model, spec: Spec) -> np.ndarray:
    k = float(spec.params.get("count_scale", 1.0))
    cols = {"active_days_7d": _f(d, "active_days_7d"), "days_since_last_active": _f(d, "days_since_last_active"),
            "log_rows_7d": np.log1p(k * _f(d, "feed_exposures_7d")), "log_sessions_7d": np.log1p(_f(d, "sessions_7d")),
            "active_yesterday": _f(d, "active_yesterday"), "weekend": _f(d, "weekend")}
    return np.column_stack([cols[f] for f in model.features])


SESS_INPUTS = (
    Input("extra_sessions_observed", "sum over the individual's observed active days of (sessions that day - 1)",
          "count", "individual", 0),
    Input("active_days_observed", "number of active days over which extra_sessions_observed was counted", "days",
          "individual", 0),
)


def _sess_build(d: dict, model, spec: Spec) -> np.ndarray:
    return np.column_stack([_f(d, "extra_sessions_observed"), _f(d, "active_days_observed")])


# ---------------------------------------------------------------- software contracts (E2 models)
def _software_inputs(spec_list) -> tuple:
    out = []
    for name, tf, doc in spec_list:
        if name in ("hour_sin", "hour_cos"):
            continue
        flag = tf == "raw"
        out.append(Input(name, doc + " (events of the simulated code-hosting / package ecosystem: `code`, `pkg`, "
                         "`issues`; stars = public endorsements, forks, issues, pull requests, pushes, releases)",
                         "flag" if flag else "count", "item" if name.startswith("repo") or name.endswith("today")
                         or name.startswith("created") or name == "stars_yesterday" else "individual", 0,
                         1 if flag else None))
    return tuple(out)


def _software_build(d: dict, model, spec: Spec) -> np.ndarray:
    raw = {k: _f(d, k) for k in d}
    if "utc_hour" in d:
        raw["hour_sin"], raw["hour_cos"] = _hour_terms(_f(d, "utc_hour"))
    return model.matrix_from_counts(raw)


# ---------------------------------------------------------------- spec table
def _base_specs() -> dict[str, Spec]:
    from .software import DEEP_FEATURES, GROWTH_FEATURES
    S = {}
    S["content_open@1"] = Spec(
        "content_open@1", "content_consumption@1",
        use=("individual decision: whether an individual shown an item in a short-video feed keeps watching it "
             "(consumes it beyond ~7 s). Fitted on randomly inserted videos, so it is the individual's response to "
             "an item it did not select, not the recommender's selection."),
        output="probability", inputs=tuple(KR_INPUTS[n] for n in _KR_CHOICE_IN), build=_kr_build,
        limits=_KR_LIMITS, params={"count_scale": 1.0},
        output_doc="P(consume) per row")
    S["content_open_recommender_target@1"] = Spec(
        "content_open_recommender_target@1", "content_consumption_feed@1",
        use=("recommender scoring target only (a learned-engagement recommender ranks candidates by it); never an "
             "individual's choice rule: fitted on recommender-selected exposures, it predicts 0.47 where 0.18 is "
             "observed under random exposure"),
        output="probability", inputs=tuple(KR_INPUTS[n] for n in _KR_CHOICE_IN), build=_kr_build,
        limits=_KR_LIMITS + " Contains the production recommender's selection on unobserved preferences.",
        params={"count_scale": 1.0}, output_doc="P(consume | recommender-selected exposure) per row")
    S["watch_time@1"] = Spec(
        "watch_time@1", "watch_duration@1",
        use=("watch time of an item the individual was shown, given whether it consumed it (draw content_open@1 "
             "first); replaces an exponential abandonment hazard"),
        output="quantiles", inputs=tuple(KR_INPUTS[n] for n in _KR_WATCH_IN), build=_kr_build,
        limits=_KR_LIMITS + " Watch times beyond about 4x the duration (replays) are not reproduced; on "
                            "recommender-selected exposures alone it under-predicts long watches.",
        params={"count_scale": 1.0}, output_doc="quantiles of watch seconds at the card's quantile levels")
    S["daily_return@1"] = Spec(
        "daily_return@1", "return_daily@1",
        use="whether an individual is active on the platform on a simulated day, given its recent activity",
        output="probability", inputs=RET_INPUTS, build=_ret_build, params={"count_scale": 1.0},
        limits=("KuaiRand-Pure app activity of 27,285 users, 2022-04-22..05-08 (17 days, includes a holiday); an "
                "active day is a day with a logged pool-video exposure and the random-insertion rate rose during "
                "the period; no notification, content or social inputs, so it cannot respond to what the platform "
                "shows; observational. See the E3 review (reviews/return_daily.json) for its multi-day forward "
                "simulation drift."),
        output_doc="P(active today) per individual")
    S["extra_sessions@1"] = Spec(
        "extra_sessions@1", "session_rate@1",
        use="number of sessions beyond the first on an active day (gamma-Poisson individual rates)",
        output="rate", inputs=SESS_INPUTS, build=_sess_build,
        limits=("KuaiRand-Pure sessions (30-minute gap) of 27,285 users, 2022-04-22..05-08; only sessions with a "
                "logged exposure are seen, so true counts are higher; held-out rates drift upward (count calibration "
                "error 0.43); assumes a stable individual rate over two weeks."),
        output_doc="posterior mean extra sessions per active day; sample() draws a count for one active day")
    S["article_open@1"] = Spec(
        "article_open@1", "news_click@1",
        use="probability that an individual opens a news article shown in a slate (MIND-small)",
        output="probability",
        inputs=(Input("text_cos_lsa", "cosine of 128-d LSA (TF-IDF of title + abstract) vectors fitted on MIND "
                      "news text", "cosine", "pair", -1, 1),),
        build=lambda d, m, s: None,
        status="offline_only",
        status_reason=("its text_cos feature is a cosine in MIND's own TF-IDF/LSA space, whose projection is not "
                       "saved, and pop_ctr_24h counts whole-slate impressions; neither can be computed in a "
                       "simulated world without a batch-relative mapping. Use article_open@2 (news_click@2, fitted "
                       "on cosines of the shared semantic encoder) when it is registered."),
        limits=("MIND-small, 50,000 Microsoft News users (US, English), 2019-11-09..15, production-recommender "
                "exposure (observational); no dwell time."),
        output_doc="P(open) per slot")
    S["ad_response@1"] = Spec(
        "ad_response@1", "commercial_response@1",
        use=("response to a randomised advertising intervention: an individual carries a persistent latent "
             "response type (drawn once with draw_types) and, in each campaign period, visits / converts with its "
             "type's treated probabilities when it is in the campaign's treated group, else with its control "
             "probabilities"),
        output="arm_probabilities",
        inputs=(Input("response_type", "the individual's persistent response type, an integer drawn once at "
                      "creation with DecisionModel.draw_types(n, rng) and stored in the individual's state",
                      "type index", "latent", 0, 63, True),
                Input("treated", "1 if the individual is in the campaign's treated (eligible) group this period",
                      "flag", "context", 0, 1)),
        build=lambda d, m, s: np.column_stack([_f(d, "response_type"), _f(d, "treated")]),
        limits=("Criteo Uplift v2.1 randomized incrementality tests (13,979,592 users, period and advertisers "
                "undisclosed, released 2018): intention-to-treat effects of campaign eligibility, identified by "
                "randomisation. Absolute rates and effect sizes are not representative (non-uniform sub-sampling); "
                "the response does NOT depend on ad content, price, relevance or frequency (not identified by the "
                "data) - a world that wants those must compose this with another fitted model and register the "
                "composition; negative predicted effects are not confirmed; transport() assumes constant per-type "
                "odds ratios."),
        output_doc="columns p_visit, p_conversion for the row's arm")
    S["repo_deep_engagement@1"] = Spec(
        "repo_deep_engagement@1", "software_deep_engagement@1",
        use=("probability that a developer's first star on a repository is followed within 24 h by a fork, issue "
             "or pull request by the same developer on the same repository"),
        output="probability",
        inputs=_software_inputs(DEEP_FEATURES) + (Input("utc_hour", "hour of the star (world clock), floored",
                                                        "hour", "context", 0, 24),),
        build=_software_build,
        limits=("GH Archive public GitHub events 2024-10-07..10 UTC (Mon-Thu); observational, not the effect of "
                "promotion; stars are not usefulness; 24 h horizon; top decile over-predicts (0.169 vs 0.131)."),
        output_doc="P(deeper engagement within 24 h) per first star")
    S["repo_star_growth@1"] = Spec(
        "repo_star_growth@1", "software_repo_growth@1",
        use="expected stars a repository receives the next day given its activity today (NB2 for sampling)",
        output="rate", inputs=_software_inputs(GROWTH_FEATURES), build=_software_build,
        limits=("GH Archive, repositories with a star, fork or creation that day, 2024-10-08..10 UTC; observational; "
                "weekday only; short horizon; inactive repositories are outside the data."),
        output_doc="expected next-day stars; sample() draws negative-binomial counts")
    return S


_SPECS: dict[str, Spec] | None = None
_LOADED: dict[str, "DecisionModel"] = {}
_USAGE: dict[str, int] = {}
_EXTRA: list = []          # callables returning {id: Spec}; registered by later calibration modules


def register_specs(fn) -> None:
    """Add decision models (fn() -> {decision id: Spec}); used by calibration modules that fit new models."""
    global _SPECS
    _EXTRA.append(fn)
    _SPECS = None


def _all_specs() -> dict[str, Spec]:
    global _SPECS
    if _SPECS is None:
        S = _base_specs()
        for fn in _EXTRA:
            S.update(fn())
        from . import runtime_ext
        S.update(runtime_ext.specs())
        _SPECS = S
    return _SPECS


def specs() -> dict[str, dict]:
    """{decision id: JSON-native contract}; loads no checkpoint."""
    return {k: v.as_dict() for k, v in sorted(_all_specs().items())}


def get(decision_id: str) -> "DecisionModel":
    """The decision model (its fitted checkpoint loaded and hash-verified once per process)."""
    S = _all_specs()
    if decision_id not in S:
        raise ContractError(f"unknown decision model {decision_id!r}; known: {sorted(S)}")
    if decision_id not in _LOADED:
        _LOADED[decision_id] = DecisionModel(S[decision_id], _load(S[decision_id].model))
    return _LOADED[decision_id]


def usage() -> dict[str, int]:
    """Rows evaluated per decision id in this process (for in-world probes)."""
    return dict(_USAGE)


def reset_usage() -> None:
    _USAGE.clear()


# ---------------------------------------------------------------- results and models
@dataclass
class Result:
    decision: str
    model: str
    version: int
    fidelity: str
    values: np.ndarray
    extrapolated: np.ndarray
    columns: tuple = ()

    @property
    def n(self) -> int:
        return int(self.values.shape[0])

    def provenance(self) -> dict:
        return {"decision": self.decision, "model": self.model, "version": int(self.version),
                "fidelity": self.fidelity, "rows": self.n, "extrapolated": int(np.count_nonzero(self.extrapolated))}


class DecisionModel:
    def __init__(self, spec: Spec, fitted):
        self.spec = spec
        self.fitted = fitted
        self.id = spec.id
        self.model_id = fitted.id
        self.version = fitted.version
        self.fidelity_id = spec.fidelity_id
        q = fitted.card.get("feature_quantiles")
        self._support = None
        if q:
            self._support = (np.array([q[f][0] for f in fitted.features], float),
                             np.array([q[f][-1] for f in fitted.features], float))

    def __repr__(self) -> str:
        return f"DecisionModel({self.id} -> {self.model_id}, {self.spec.output})"

    def contract(self) -> dict:
        return self.spec.as_dict()

    # ------------------------------------------------------------ inputs
    def _validate(self, inputs: dict) -> tuple[dict, int]:
        if self.spec.status != "ready":
            raise ContractError(f"{self.id} is {self.spec.status}: {self.spec.status_reason}")
        names = [i.name for i in self.spec.inputs]
        missing = [n for n in names if n not in inputs]
        if missing:
            raise ContractError(f"{self.id}: missing inputs {missing}")
        n = None
        out = {}
        for i in self.spec.inputs:
            v = np.asarray(inputs[i.name], dtype=np.float64)
            if v.ndim != 1:
                raise ContractError(f"{self.id}: input {i.name} must be a 1-D array (one row per individual)")
            if n is None:
                n = v.shape[0]
            elif v.shape[0] != n:
                raise ContractError(f"{self.id}: inputs have different lengths ({i.name}: {v.shape[0]} != {n})")
            if not np.all(np.isfinite(v)):
                raise ContractError(f"{self.id}: input {i.name} has non-finite values")
            if (i.lo is not None and np.any(v < i.lo)) or (i.hi is not None and np.any(v > i.hi)):
                raise ContractError(f"{self.id}: input {i.name} outside its domain [{i.lo}, {i.hi}]")
            if i.integer and np.any(v != np.round(v)):
                raise ContractError(f"{self.id}: input {i.name} must be integer-valued")
            out[i.name] = v
        return out, int(n or 0)

    def features(self, inputs: dict) -> np.ndarray:
        """The fitted model's feature matrix for these simulator inputs (row-wise formulas)."""
        d, n = self._validate(inputs)
        if n == 0:
            return np.zeros((0, len(self.fitted.features)))
        return np.asarray(self.spec.build(d, self.fitted, self.spec), dtype=np.float64)

    def _extrapolated(self, X: np.ndarray) -> np.ndarray:
        if self._support is None or X.size == 0:
            return np.zeros(X.shape[0], bool)
        lo, hi = self._support
        return np.any((X < lo - 1e-9) | (X > hi + 1e-9), axis=1)

    # ------------------------------------------------------------ outputs
    def _predict(self, X: np.ndarray) -> tuple[np.ndarray, tuple]:
        out = self.spec.output
        if out == "arm_probabilities":
            tab = self.fitted.response_types()
            t = X[:, 0].astype(np.int64)
            tr = X[:, 1] > 0
            pv = np.where(tr, tab["p1_visit"][t], tab["p0_visit"][t])
            pc = np.where(tr, tab["p1_conversion"][t], tab["p0_conversion"][t])
            return np.column_stack([pv, pc]), ("p_visit", "p_conversion")
        y = np.asarray(self.fitted.predict(X), dtype=np.float64)
        if out == "probability":
            return np.clip(y.reshape(-1), 0.0, 1.0), ("p",)
        if out == "quantiles":
            return np.maximum(np.expm1(y), 0.0), tuple(f"q{lv:g}" for lv in self.levels)
        if out == "rate":
            return np.maximum(y.reshape(-1), 0.0), ("rate",)
        raise ContractError(f"{self.id}: unknown output kind {out}")

    @property
    def levels(self) -> np.ndarray:
        return np.asarray(self.fitted.card.get("quantile_levels") or [], dtype=np.float64)

    def _result(self, values, X, cols) -> Result:
        _USAGE[self.id] = _USAGE.get(self.id, 0) + int(values.shape[0])
        return Result(self.id, self.model_id, self.version, self.fidelity_id, values, self._extrapolated(X), cols)

    def evaluate(self, inputs: dict, strict: bool = False) -> Result:
        """Calibrated outputs for a batch of individuals from simulator inputs (see the module docstring)."""
        X = self.features(inputs)
        ex = self._extrapolated(X)
        if strict and np.any(ex):
            raise ContractError(f"{self.id}: {int(ex.sum())} rows outside the data's feature support")
        vals, cols = self._predict(X) if X.shape[0] else (np.zeros((0,)), ())
        return self._result(vals, X, cols)

    def evaluate_features(self, X: np.ndarray) -> Result:
        """Outputs for rows already in the fitted model's feature space (held-out evaluation, audits)."""
        X = np.asarray(X, dtype=np.float64)
        if self.spec.output == "arm_probabilities":
            P = self.fitted.predict(X)
            return self._result(P, np.zeros((X.shape[0], 0)), ("p_visit_control", "p_visit_treated",
                                                                "p_conversion_control", "p_conversion_treated"))
        if self.spec.output == "quantiles":
            return self._result(np.asarray(self.fitted.predict(X), np.float64), X,
                                tuple(f"log1p_q{lv:g}" for lv in self.levels))
        vals, cols = self._predict(X)
        return self._result(vals, X, cols)

    def sample(self, inputs: dict, rng: np.random.Generator) -> np.ndarray:
        """One draw per row with the caller's RNG stream (Bernoulli, inverse-CDF from quantiles, or counts)."""
        res = self.evaluate(inputs)
        n = res.n
        out = self.spec.output
        if out == "probability":
            return rng.random(n) < res.values
        if out == "quantiles":
            from .predictors import interpolate_quantiles
            Q = np.log1p(res.values)
            return np.maximum(np.expm1(interpolate_quantiles(Q, self.levels, rng.random(n))), 0.0)
        if out == "rate":
            if self.spec.model.startswith("session_rate"):
                d, _ = self._validate(inputs)
                p = self.fitted.predictor
                lam = rng.gamma(p.shape + d["extra_sessions_observed"], 1.0 / (p.rate + d["active_days_observed"]))
                return rng.poisson(lam)
            if hasattr(self.fitted, "dispersion"):
                r = 1.0 / self.fitted.dispersion
                mu = np.maximum(res.values, 1e-12)
                return rng.poisson(rng.gamma(shape=r, scale=mu / r))
            return rng.poisson(res.values)
        if out == "arm_probabilities":
            d, _ = self._validate(inputs)
            r = self.fitted.respond(rng, d["response_type"].astype(np.int64), d["treated"] > 0)
            return np.column_stack([r["visit"], r["conversion"]])
        raise ContractError(f"{self.id}: cannot sample output kind {out}")

    def draw_types(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Persistent latent response types for n new individuals (ad_response only)."""
        if self.spec.output != "arm_probabilities":
            raise ContractError(f"{self.id} has no latent types")
        return self.fitted.sample_types(rng, n)


def contract_markdown() -> str:
    """The feature contract of every decision model as Markdown (docs/v5/calibration/RUNTIME.md is generated by
    `python -m crucible.calibration.runtime`)."""
    lines = []
    for sid, s in specs().items():
        lines += [f"## `{sid}` -> `{s['model']}`", "", f"* use: {s['use']}", f"* output: {s['output']} "
                  f"({s['output_doc']})", f"* status: {s['status']}" + (f" - {s['status_reason']}"
                                                                         if s['status_reason'] else ""),
                  f"* fidelity record: `{s['fidelity_id']}`", f"* parameters: {s['params'] or 'none'}", "",
                  "| input | kind | unit | domain | meaning |", "|---|---|---|---|---|"]
        for i in s["inputs"]:
            lo, hi = i["domain"]
            dom = f"[{'-inf' if lo is None else lo}, {'inf' if hi is None else hi}]" + (" int" if i["integer"] else "")
            lines.append(f"| `{i['name']}` | {i['kind']} | {i['unit']} | {dom} | {i['meaning']} |")
        lines += ["", f"Extrapolation limits: {s['limits']}", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    sys.stdout.write(contract_markdown())
