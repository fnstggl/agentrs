"""cam_live.harness.modality — adaptive selection among execution modalities (spec 12A.7).

Owned by workstream ES.

For one intended operation there may be several feasible realisations: a validated skill, structured browser
operations, an authorized programmatic interface, sandboxed code, lightweight visual grounding, or a heavier visual
model. :class:`ModalitySelector` keeps, per kind of realisation and per *generic* execution context, Bayesian
estimates of

* the probability that execution succeeds (Beta posterior),
* latency, computation, number of primitive operations per attempt, and the cost of a failure (Normal posteriors
  of the means),

with the specification's default ordering encoded only as a weak prior, and chooses by the expected cost of
reaching success

    C = (latency + w_c * compute + w_o * primitive_ops + (1 - p) * recovery) / p

with Thompson sampling. The probability of every option is recorded with the choice, and every result updates the
estimates, so the ordering is learned from experience rather than fixed.

Context features are generic (operation family, hash of the site host, element role, properties of the
observation). Estimates are shared across contexts by hierarchical back-off: each more specific context uses the
posterior of the more general one as its prior.

**A refusal is never a reason to try another modality.** When a result is DENIED / NEEDS_APPROVAL or its failure
class is in ``FailureClass.HARD``, the *intent* is blocked: :meth:`ModalitySelector.choose` then offers no
realisation at all for it, and the refusal is not counted as evidence about the modality.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from urllib.parse import urlsplit

from ..contracts import (ActionResult, ActionSpec, CandidateEvaluation, Clock, FailureClass, Modality, Observation,
                         OutcomeKind, Quantity, ResultStatus, WallClock, new_id)

#: the specification's default preference order (code execution is placed after programmatic interfaces)
KINDS = ("skill", "structured_browser", "api", "code", "visual_light", "visual_heavy")
METRICS = ("latency_s", "compute_s", "primitive_ops", "recovery_s")


@dataclass(frozen=True)
class KindPrior:
    p_success: float
    latency_s: float
    compute_s: float
    primitive_ops: float
    recovery_s: float


#: weak priors whose expected cost of success increases along the specification's ordering
DEFAULT_PRIORS: Dict[str, KindPrior] = {
    "skill": KindPrior(0.90, 1.0, 0.0, 3.0, 2.0),
    "structured_browser": KindPrior(0.85, 1.5, 0.0, 1.0, 2.0),
    "api": KindPrior(0.80, 2.0, 0.0, 1.0, 2.0),
    "code": KindPrior(0.75, 3.0, 0.5, 1.0, 3.0),
    "visual_light": KindPrior(0.65, 2.5, 0.5, 2.0, 3.0),
    "visual_heavy": KindPrior(0.70, 4.0, 3.0, 2.0, 3.0),
}


def default_kind(action: ActionSpec) -> str:
    """Kind of realisation an action is, from its modality (visual actions name their grounder tier in
    ``arguments["grounder_tier"]``: "light" or "heavy")."""
    m = action.modality
    if m == Modality.SKILL:
        return "skill"
    if m == Modality.STRUCTURED_BROWSER:
        return "structured_browser"
    if m == Modality.API:
        return "api"
    if m == Modality.CODE:
        return "code"
    if m == Modality.VISUAL:
        return "visual_heavy" if str(action.arguments.get("grounder_tier", "light")) == "heavy" else "visual_light"
    raise ValueError(f"modality {m!r} is not an external execution modality")


def _bucket(n: Any) -> str:
    try:
        n = int(n)
    except (TypeError, ValueError):
        return "?"
    return "0" if n <= 0 else "1-9" if n < 10 else "10-49" if n < 50 else "50+"


@dataclass(frozen=True)
class ExecContextFeatures:
    """Generic, task-independent description of where an operation is to be executed."""
    tool: str = ""            # operation family, e.g. "click", "fill", "request", "navigate"
    host_hash: str = ""       # hash of the site host (never the host itself)
    element_role: str = ""    # accessibility role of the target element, if any
    props: Tuple[Tuple[str, str], ...] = ()   # observation properties: has_a11y, has_canvas, interactive bucket, ...

    def keys(self) -> List[str]:
        """Context keys from the most general to the most specific."""
        out = ["*"]
        if self.tool:
            out.append(f"t={self.tool}")
        if self.host_hash:
            out.append(f"h={self.host_hash}")
            if self.tool:
                out.append(f"h={self.host_hash}|t={self.tool}")
        if self.element_role or self.props:
            p = ",".join(f"{k}:{v}" for k, v in self.props)
            out.append(f"h={self.host_hash}|t={self.tool}|r={self.element_role}|p={p}")
        return out

    @classmethod
    def of(cls, action: Optional[ActionSpec] = None, observation: Optional[Observation] = None, *, tool: str = "",
           url: str = "", element_role: str = "") -> "ExecContextFeatures":
        if action is not None:
            tool = tool or action.operation.rsplit(".", 1)[-1]
            url = url or str(action.arguments.get("url", "")) or (action.target if "://" in action.target else "")
            element_role = element_role or str(action.arguments.get("role", ""))
        props: List[Tuple[str, str]] = []
        if observation is not None:
            d = observation.data
            url = url or str(d.get("url", "")) or observation.provenance.uri
            for name in ("has_a11y", "has_canvas", "dynamic"):
                if name in d:
                    props.append((name, "1" if d[name] else "0"))
            if "n_interactive" in d or "interactive" in d:
                n = d.get("n_interactive", len(d.get("interactive") or []))
                props.append(("interactive", _bucket(n)))
        host = urlsplit(url).netloc.lower() if "://" in url else ""
        return cls(tool=tool, host_hash=hashlib.sha256(host.encode()).hexdigest()[:12] if host else "",
                   element_role=element_role, props=tuple(sorted(props)))


@dataclass
class ModalityEstimate:
    kind: str
    p_success: float
    p_sd: float
    latency_s: float
    compute_s: float
    primitive_ops: float
    recovery_s: float
    expected_cost: float
    n: float                   # attempts observed for this kind in any context
    n_context: float           # attempts observed in the most specific matching context

    def to_dict(self) -> Dict[str, float]:
        return dict(self.__dict__)


@dataclass
class Realisation:
    kind: str
    action: ActionSpec


@dataclass
class ModalityChoice:
    choice_id: str
    ts: float
    intent_key: str
    context_keys: List[str]
    options: List[Dict[str, Any]] = field(default_factory=list)     # kind, action_id, and the estimate of each option
    probabilities: List[float] = field(default_factory=list)
    chosen_index: int = -1
    rule: str = ""
    blocked: bool = False
    reason: str = ""

    @property
    def chosen_kind(self) -> str:
        return self.options[self.chosen_index]["kind"] if self.chosen_index >= 0 else ""

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


class _Mean:
    """Running count / mean / sum of squared deviations (Welford)."""
    __slots__ = ("n", "mean", "m2")

    def __init__(self, n: float = 0.0, mean: float = 0.0, m2: float = 0.0):
        self.n, self.mean, self.m2 = n, mean, m2

    def add(self, x: float) -> None:
        self.n += 1.0
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)

    def var(self) -> Optional[float]:
        return self.m2 / (self.n - 1.0) if self.n >= 2 else None


class _Stats:
    def __init__(self) -> None:
        self.success, self.failure = 0.0, 0.0
        self.m = {name: _Mean() for name in METRICS}

    def dump(self) -> List[Any]:
        return [self.success, self.failure, {k: [v.n, v.mean, v.m2] for k, v in self.m.items()}]

    @classmethod
    def load(cls, d: List[Any]) -> "_Stats":
        s = cls()
        s.success, s.failure = float(d[0]), float(d[1])
        for k, v in d[2].items():
            s.m[k] = _Mean(*v)
        return s


def intent_key(intent: Union[str, ActionSpec]) -> str:
    """Identity of an intended operation, independent of how it is realised."""
    if isinstance(intent, str):
        return intent
    basis = intent.parent_decision_id or f"{intent.operation.rsplit('.', 1)[-1]}|{intent.target}|{intent.expected}"
    return hashlib.sha256(basis.encode()).hexdigest()[:20]


class ModalitySelector:
    def __init__(self, *, priors: Optional[Dict[str, KindPrior]] = None, prior_strength: float = 2.0,
                 backoff_strength: float = 4.0, compute_weight: float = 1.0, op_weight: float = 0.05,
                 n_samples: int = 256, seed: int = 0, clock: Optional[Clock] = None, history: int = 1000):
        self.priors = dict(DEFAULT_PRIORS if priors is None else priors)
        self.prior_strength, self.backoff_strength = float(prior_strength), float(backoff_strength)
        self.compute_weight, self.op_weight, self.n_samples = compute_weight, op_weight, n_samples
        self.clock = clock or WallClock()
        self._rng = random.Random(seed)
        self._stats: Dict[Tuple[str, str], _Stats] = {}
        self._blocked: Dict[str, Dict[str, Any]] = {}
        self.history: List[ModalityChoice] = []
        self._history_max = history

    # ---- posteriors ----------------------------------------------------------------------------------------
    def _posterior(self, kind: str, ctx: ExecContextFeatures) -> Dict[str, Any]:
        """Back-off chain general -> specific. Returns Beta(a, b) for success and (mean, sd of the mean) per metric."""
        pr = self.priors[kind]
        a, b = self.prior_strength * pr.p_success, self.prior_strength * (1.0 - pr.p_success)
        means = {m: (getattr(pr, m), None, self.prior_strength) for m in METRICS}   # mean, variance, weight
        n_total = n_ctx = 0.0
        empty = _Stats()
        for i, key in enumerate(ctx.keys()):
            st = self._stats.get((kind, key), empty)     # an unseen context only widens the inherited posterior
            n_here = st.success + st.failure
            if i == 0:
                n_total = n_here
                a, b = a + st.success, b + st.failure
            else:
                m = a / (a + b)
                a, b = self.backoff_strength * m + st.success, self.backoff_strength * (1.0 - m) + st.failure
            n_ctx = n_here
            for name in METRICS:
                acc = st.m[name]
                mu, var, w = means[name]
                w0 = w if i == 0 else self.backoff_strength
                if acc.n > 0:
                    means[name] = ((w0 * mu + acc.n * acc.mean) / (w0 + acc.n), acc.var() if acc.var() is not None else var,
                                   w0 + acc.n)
                else:
                    means[name] = (mu, var, w0)
        out: Dict[str, Any] = {"a": a, "b": b, "n": n_total, "n_context": n_ctx}
        for name in METRICS:
            mu, var, w = means[name]
            var = (0.5 * max(mu, 0.1)) ** 2 if var is None else var
            out[name] = (mu, math.sqrt(max(var, 1e-12) / max(w, 1e-9)))
        return out

    def _cost(self, p: float, lat: float, comp: float, ops: float, rec: float) -> float:
        p = min(max(p, 1e-3), 1.0)
        attempt = max(lat, 0.0) + self.compute_weight * max(comp, 0.0) + self.op_weight * max(ops, 0.0)
        return (attempt + (1.0 - p) * max(rec, 0.0)) / p

    def estimate(self, kind: str, ctx: Optional[ExecContextFeatures] = None) -> ModalityEstimate:
        post = self._posterior(kind, ctx or ExecContextFeatures())
        a, b = post["a"], post["b"]
        p = a / (a + b)
        vals = {m: post[m][0] for m in METRICS}
        return ModalityEstimate(kind=kind, p_success=p, p_sd=math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1.0))),
                                expected_cost=self._cost(p, vals["latency_s"], vals["compute_s"], vals["primitive_ops"],
                                                         vals["recovery_s"]),
                                n=post["n"], n_context=post["n_context"], **vals)

    # ---- choosing ------------------------------------------------------------------------------------------
    def blocked(self, intent: Union[str, ActionSpec]) -> Optional[Dict[str, Any]]:
        return self._blocked.get(intent_key(intent))

    def choose(self, intent: Union[str, ActionSpec], options: Sequence[Realisation],
               ctx: Optional[ExecContextFeatures] = None, *, rule: str = "thompson") -> ModalityChoice:
        """Pick one realisation of ``intent``. If the intent was refused before (denied, awaiting approval, or a
        platform restriction) the returned choice is ``blocked`` and offers nothing: there is no other way round."""
        ctx = ctx or ExecContextFeatures()
        key = intent_key(intent)
        choice = ModalityChoice(choice_id=new_id("mch"), ts=self.clock.now(), intent_key=key, context_keys=ctx.keys(),
                                rule=rule)
        block = self._blocked.get(key)
        if block is not None:
            choice.blocked, choice.reason = True, (f"intent refused ({block['status']}, {block['failure_class']}) "
                                                   f"through {block['kind']}: no alternative modality is offered")
            self._remember(choice)
            return choice
        options = list(options)
        if not options:
            choice.reason = "no feasible realisation"
            self._remember(choice)
            return choice
        ests = [self.estimate(o.kind, ctx) for o in options]
        choice.options = [dict(est.to_dict(), action_id=o.action.action_id) for o, est in zip(options, ests)]
        if rule == "greedy" or len(options) == 1:
            best = min(range(len(options)), key=lambda i: ests[i].expected_cost)
            choice.probabilities = [1.0 if i == best else 0.0 for i in range(len(options))]
            choice.chosen_index = best
        elif rule == "thompson":
            posts = [self._posterior(o.kind, ctx) for o in options]
            wins = [0.5] * len(options)
            for _ in range(self.n_samples):
                costs = []
                for post in posts:
                    p = self._rng.betavariate(post["a"], post["b"])
                    draw = [max(0.0, self._rng.gauss(post[m][0], post[m][1])) for m in METRICS]
                    costs.append(self._cost(p, *draw))
                wins[min(range(len(costs)), key=costs.__getitem__)] += 1.0
            total = sum(wins)
            choice.probabilities = [w / total for w in wins]
            choice.chosen_index = self._rng.choices(range(len(options)), weights=choice.probabilities)[0]
        else:
            raise ValueError(f"unknown selection rule {rule!r}")
        self._remember(choice)
        return choice

    def _remember(self, choice: ModalityChoice) -> None:
        self.history.append(choice)
        del self.history[:-self._history_max]

    # ---- learning from results -----------------------------------------------------------------------------
    def update(self, kind: str, ctx: Optional[ExecContextFeatures], result: ActionResult, *,
               intent: Union[str, ActionSpec, None] = None, recovery_s: float = 0.0) -> str:
        """Update the estimates from one executed attempt. Returns what was done: "success", "failure",
        "blocked" (a refusal: the intent is blocked and nothing is learned about the modality) or "ignored"."""
        ctx = ctx or ExecContextFeatures()
        refused = (result.status in (ResultStatus.DENIED, ResultStatus.NEEDS_APPROVAL)
                   or result.failure_class in FailureClass.HARD)
        if refused:
            if intent is not None:
                self._blocked[intent_key(intent)] = {"status": result.status, "failure_class": result.failure_class,
                                                     "kind": kind, "ts": self.clock.now(),
                                                     "detail": result.failure_detail[:200]}
            return "blocked"
        if result.status == ResultStatus.CANCELLED:
            return "ignored"
        ok = result.status == ResultStatus.OK and result.verified is not False
        latency = result.cost.seconds if result.cost.seconds > 0 else max(0.0, result.ended - result.started)
        ops = float(result.cost.primitive_ops or len(result.ops) or 1)
        for key in ctx.keys():
            st = self._stats.setdefault((kind, key), _Stats())
            if ok:
                st.success += 1.0
            else:
                st.failure += 1.0
                st.m["recovery_s"].add(latency + max(0.0, recovery_s))
            st.m["latency_s"].add(latency)
            st.m["compute_s"].add(result.cost.compute_s)
            st.m["primitive_ops"].add(ops)
        return "success" if ok else "failure"

    def update_choice(self, choice: ModalityChoice, ctx: Optional[ExecContextFeatures], result: ActionResult,
                      recovery_s: float = 0.0) -> str:
        return self.update(choice.chosen_kind, ctx, result, intent=choice.intent_key, recovery_s=recovery_s)

    def authorization_changed(self, intent: Union[str, ActionSpec]) -> None:
        """To be called by the runtime when the authorization service itself reports a new grant for the intent
        (an approval was resolved). The learner has no path to this method's effect other than a real grant."""
        self._blocked.pop(intent_key(intent), None)

    # ---- exposing the estimates ----------------------------------------------------------------------------
    def quantities(self, ctx: Optional[ExecContextFeatures] = None, kinds: Optional[Sequence[str]] = None) -> List[Quantity]:
        """The current estimates as exact quantities (``exec.modality.<kind>.<field>``) for the learner."""
        out: List[Quantity] = []
        for kind in kinds or KINDS:
            e = self.estimate(kind, ctx)
            pre = f"exec.modality.{kind}"
            out += [Quantity(f"{pre}.p_success", e.p_success), Quantity(f"{pre}.p_sd", e.p_sd),
                    Quantity(f"{pre}.latency_s", e.latency_s, unit="s"),
                    Quantity(f"{pre}.compute_s", e.compute_s, unit="s"),
                    Quantity(f"{pre}.primitive_ops", e.primitive_ops),
                    Quantity(f"{pre}.recovery_s", e.recovery_s, unit="s"),
                    Quantity(f"{pre}.expected_cost", e.expected_cost, unit="s"),
                    Quantity(f"{pre}.n", e.n, kind=OutcomeKind.COUNT)]
        return out

    def annotate(self, evaluations: Sequence[CandidateEvaluation], observation: Optional[Observation] = None) -> None:
        """Write the execution estimates of each candidate's modality into ``CandidateEvaluation.extras``."""
        for ev in evaluations:
            try:
                kind = default_kind(ev.action)
            except ValueError:
                continue
            e = self.estimate(kind, ExecContextFeatures.of(ev.action, observation))
            ev.extras.update(exec_p_success=e.p_success, exec_p_sd=e.p_sd, exec_latency_s=e.latency_s,
                             exec_compute_s=e.compute_s, exec_primitive_ops=e.primitive_ops,
                             exec_recovery_s=e.recovery_s, exec_expected_cost=e.expected_cost)

    # ---- persistence ---------------------------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {"version": 1, "stats": {f"{k[0]}||{k[1]}": s.dump() for k, s in self._stats.items()},
                "blocked": self._blocked}

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        self._stats = {tuple(k.split("||", 1)): _Stats.load(v) for k, v in d.get("stats", {}).items()}
        self._blocked = dict(d.get("blocked", {}))

    def save(self, path: str) -> None:
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state_dict(), f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def load(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            self.load_state_dict(json.load(f))
