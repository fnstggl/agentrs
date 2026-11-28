"""cam_live.reward.validity — when is the learned reward useful? Prospective evidence from verified outcomes (spec 7.6).

The bootstrap mix decides how much of the inner policy update follows R_psi. The question it must answer is whether
reinforcing what R_psi rewards moves the policy toward decisions that the independent verifier later confirms as
progress — better than reinforcing what the bootstrap rewards. This module measures that, prospectively, within one
long objective, and decides the mix on anytime-valid evidence.

Why this quantity. A policy-gradient step with a standardised reward r changes the policy by
alpha * E[grad log pi(a|s) (r(s,a) - b(s))]; to first order the true (verified) value changes by
alpha * <grad J, that step>, which for a softmax policy is a within-state covariance between r and the verified
advantage of the actions. So, to first order, the reward that improves the verified value more is the one whose
*within-state ranking* of decisions agrees better with their verified consequences. Two decisions taken one after the
other in the same objective share most of their state; comparing their rewards with their verified progress removes
what the state contributes to both (and any slow drift of either quantity).

    unit      one independent outcome group (``windows.independence_key``: experiment, else parent decision, else the
              decision), once every decision of it registered so far has its final verified progress
              (``inner.verified_progress``: signed verifier readings only). A group is scored once.
    pair      the next two units that mature in the same objective segment (disjoint pairs, in maturity order; the
              pairing never looks at outcomes)
    score     for a mixture weight m of the bootstrap: c_m = sign(Delta r_m) * sign(Delta y), with
              r_m = m * r_bootstrap / s_b + (1 - m) * r_learned / s_l  (s: running scale of each reward over the
              decisions registered so far) and Delta the difference within the pair; c_m in {-1, 0, 1}
    prospective  r_bootstrap and r_learned are the values both rewards gave the decision when it was rewarded
              (before any of its verified outcomes existed, with the psi of that moment); a unit is scored once and
              never re-scored by a later psi. Every unit can afterwards supervise psi: evaluation precedes training
              (prequential), so no group has to be held out of the meta-update for this evidence.

At every point m_k of the grid {1, 1 - step, ..., 0} two increments compare the mixture in force with the two
pure rewards:

    x_k^+ = (c_0 - c_{m_k}) / 2     R_psi alone agrees better (> 0) or worse (< 0) with what was verified
    x_k^- = (c_1 - c_{m_k}) / 2     the bootstrap alone agrees better (> 0) or worse (< 0)

each in [-1, 1]. Each carries a one-sided sequential test of H0: E[x | past] <= 0 (``BettingTest``: a test
supermartingale with predictable aGRAPA bets; by Ville's inequality valid at every time simultaneously, so the
evidence pools over the whole objective, may be looked at after every decision, and needs no fixed sample size).
Decision at the current mix m_k:

    raise  H0 of x_k^- rejected (wealth >= 2 (G - 1) / alpha): the bootstrap alone is shown to agree better
    lower  H0 of x_k^+ rejected (wealth >= 2 / alpha): R_psi alone is shown to agree better than the mixture, AND
           (G3-core D-RW1) R_psi is shown to agree better than each never-trained control reward and its negation
           (``n_placebo`` of them, the first being R_psi's own initialisation; intersection-union test, each
           component at alpha / 2): a random initialisation is a fixed random function of the signals and can agree
           with the outcomes by chance; only agreement beyond what unlearned rewards reach is credited to learning
    hold   otherwise (with the reason; "mixture_and_R_psi_rank_alike_at_reward_time" when every scored pair ranks
           the same under R_psi and under the mixture in force — a bootstrap tied at reward time, D-RW2: what a
           further step would change is the weight of the bootstrap's later-maturing update-time terms, about which
           reward-time evidence says nothing)

Error control: the mix can be at m_{k+1} only after the x^+ tests of m_0 .. m_k were all rejected at some time (a
fixed-sequence procedure), so P(any lowering although R_psi alone does not agree better) <= alpha / 2; raises are
Bonferroni-corrected over the G - 1 points, P(any raise although the bootstrap does not agree better) <= alpha / 2.
``ConfidenceSequence`` (predictable plug-in empirical Bernstein, Waudby-Smith & Ramdas 2023) gives the reported
interval of each increment. The targets are averages of conditional means over the objective so far (psi changes
as it learns; the evidence is about the learned reward as it was when it was used).

No margin, threshold or mix value is set by hand: the only constants are the error level alpha, the grid and the bet
cap. A learned reward that is merely as good as the bootstrap is not promoted (nothing shows it helps); a frozen R_psi
is never promoted, however well its random initialisation happens to agree (it is its own control); a harmful one is
reverted.
"""
from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = ["ConfidenceSequence", "BettingTest", "ValidityDecision", "ProspectiveValidity", "mix_grid",
           "pair_scores"]


# ================================================================================================ confidence sequence
class ConfidenceSequence:
    """Anytime-valid two-sided confidence sequence for the running average of the conditional means of observations
    in [-1, 1] (mapped to [0, 1] internally): predictable plug-in empirical Bernstein (Waudby-Smith & Ramdas 2023,
    Theorem 2). With probability >= 1 - alpha, for every t simultaneously,

        sum_i lambda_i mu_i / sum_i lambda_i  lies in  [lower_t, upper_t]

    where mu_i = E[x_i | past] and lambda_i is the predictable bet of step i (for i.i.d. or constant-mean
    observations this is the mean itself). Valid under optional stopping and continuous monitoring."""

    def __init__(self, alpha: float = 0.05, lam_max: float = 0.5):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha, self.lam_max = float(alpha), float(lam_max)
        self.t = 0
        self.sum_y = 0.0             # sum of y_i (for the plug-in mean)
        self.sum_sq = 0.0            # sum of (y_i - mu_hat_i)^2 (for the plug-in variance)
        self.sum_lam = 0.0
        self.sum_lam_y = 0.0
        self.sum_pen = 0.0           # sum v_i psi_E(lambda_i)
        self.sum_x = 0.0             # plain sum of the observations (diagnostic)

    # plug-in estimates after t observations (with the regularising pseudo-observation of the paper)
    def _mu_hat(self) -> float:
        return (0.5 + self.sum_y) / (self.t + 1)

    def _var_hat(self) -> float:
        return (0.25 + self.sum_sq) / (self.t + 1)

    def update(self, x: float) -> None:
        x = float(x)
        if not math.isfinite(x) or x < -1.0 - 1e-12 or x > 1.0 + 1e-12:
            raise ValueError(f"observation {x!r} outside [-1, 1]")
        y = 0.5 * (min(1.0, max(-1.0, x)) + 1.0)
        t = self.t + 1
        mu_prev, var_prev = self._mu_hat(), self._var_hat()
        lam = min(math.sqrt(2.0 * math.log(2.0 / self.alpha) / (var_prev * t * math.log(1.0 + t))), self.lam_max)
        v = 4.0 * (y - mu_prev) ** 2
        psi = (-math.log(1.0 - lam) - lam) / 4.0
        self.sum_lam += lam
        self.sum_lam_y += lam * y
        self.sum_pen += v * psi
        self.t = t
        self.sum_y += y
        self.sum_sq += (y - self._mu_hat()) ** 2
        self.sum_x += x

    def bounds(self) -> Tuple[float, float]:
        """(lower, upper) on the [-1, 1] scale; (-1, 1) before any observation."""
        if self.t == 0 or self.sum_lam <= 0:
            return -1.0, 1.0
        c = self.sum_lam_y / self.sum_lam
        w = (math.log(2.0 / self.alpha) + self.sum_pen) / self.sum_lam
        lo, hi = max(0.0, c - w), min(1.0, c + w)
        return 2.0 * lo - 1.0, 2.0 * hi - 1.0

    @property
    def center(self) -> float:
        return 2.0 * (self.sum_lam_y / self.sum_lam) - 1.0 if self.sum_lam > 0 else 0.0

    @property
    def mean(self) -> float:
        return self.sum_x / self.t if self.t else 0.0

    def state_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in ("alpha", "lam_max", "t", "sum_y", "sum_sq", "sum_lam", "sum_lam_y",
                                               "sum_pen", "sum_x")}

    def load_state_dict(self, d: Mapping[str, Any]) -> None:
        for k in ("alpha", "lam_max", "sum_y", "sum_sq", "sum_lam", "sum_lam_y", "sum_pen", "sum_x"):
            setattr(self, k, float(d[k]))
        self.t = int(d["t"])


class BettingTest:
    """One-sided sequential test of H0: E[x_t | past] <= 0 (``sign=+1``) or >= 0 (``sign=-1``) for x in [-1, 1]:
    wealth W_t = prod_i (1 + lambda_i * sign * x_i) with a predictable bet lambda_i in [0, cap] (aGRAPA: the plug-in
    mean over mean square of the past, shrunk by one pseudo-observation). Under H0, W is a nonnegative
    supermartingale, so P(sup_t W_t >= 1 / alpha) <= alpha (Ville). ``rejected`` is that event at the current time."""

    def __init__(self, alpha: float, sign: int = 1, cap: float = 0.75):
        if not 0.0 < alpha < 1.0 or sign not in (1, -1) or not 0.0 < cap < 1.0:
            raise ValueError("alpha in (0, 1), sign +1 / -1, cap in (0, 1)")
        self.alpha, self.sign, self.cap = float(alpha), int(sign), float(cap)
        self.t, self.s, self.ss, self.log_w = 0, 0.0, 0.0, 0.0
        self.max_log_w = 0.0

    def update(self, x: float) -> None:
        x = self.sign * max(-1.0, min(1.0, float(x)))
        mu = self.s / (self.t + 1)
        var = (0.25 + self.ss) / (self.t + 1)
        lam = max(0.0, min(self.cap, mu / (var + mu * mu)))
        self.log_w += math.log1p(lam * x)
        self.max_log_w = max(self.max_log_w, self.log_w)
        self.t += 1
        self.s += x
        self.ss += (x - self.s / (self.t + 1)) ** 2

    @property
    def wealth(self) -> float:
        return math.exp(min(self.log_w, 700.0))

    @property
    def rejected(self) -> bool:
        return self.log_w >= math.log(1.0 / self.alpha)

    def state_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in ("alpha", "sign", "cap", "t", "s", "ss", "log_w", "max_log_w")}

    def load_state_dict(self, d: Mapping[str, Any]) -> None:
        self.alpha, self.sign, self.cap = float(d["alpha"]), int(d["sign"]), float(d["cap"])
        self.t, self.s, self.ss = int(d["t"]), float(d["s"]), float(d["ss"])
        self.log_w, self.max_log_w = float(d["log_w"]), float(d["max_log_w"])


# =========================================================================================================== scores
def mix_grid(step: float) -> List[float]:
    """{1, 1 - step, ..., 0} (the last point is 0 even when step does not divide 1)."""
    step = float(step)
    if not 0.0 < step <= 1.0:
        raise ValueError("step must be in (0, 1]")
    n = int(math.ceil(1.0 / step - 1e-9))
    return [round(max(0.0, 1.0 - k * step), 12) for k in range(n)] + [0.0]


def _sign(x: float, tol: float = 0.0) -> int:
    return 1 if x > tol else (-1 if x < -tol else 0)


def pair_scores(rb: Tuple[float, float], rl: Tuple[float, float], y: Tuple[float, float], grid: Sequence[float],
                s_b: float, s_l: float) -> List[int]:
    """c_m for every m of ``grid``: sign(Delta r_m) * sign(Delta y) for one pair of units (see module doc)."""
    dy = _sign(y[0] - y[1], 1e-12 * max(1.0, abs(y[0]), abs(y[1])))
    db = (rb[0] - rb[1]) / s_b if s_b > 0 else 0.0
    dl = (rl[0] - rl[1]) / s_l if s_l > 0 else 0.0
    return [_sign(m * db + (1.0 - m) * dl, 1e-12) * dy for m in grid]


_MIXED = "\x00mixed-versions"           # a unit whose decisions were rewarded by different R_psi versions


class _Running:
    """Welford mean / variance of one reward over the decisions registered so far."""

    def __init__(self) -> None:
        self.n, self.mean, self.m2 = 0, 0.0, 0.0

    def add(self, x: float) -> None:
        if not math.isfinite(x):
            return
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)

    @property
    def std(self) -> float:
        return math.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else 0.0

    def state(self) -> List[float]:
        return [self.n, self.mean, self.m2]

    def load(self, s: Sequence[float]) -> None:
        self.n, self.mean, self.m2 = int(s[0]), float(s[1]), float(s[2])


# ========================================================================================================= evidence
@dataclass
class ValidityDecision:
    action: str                   # "lower" | "raise" | "hold"
    reason: str
    mix_before: float
    mix_after: float
    n_pairs: int                  # scored pairs (independent units / 2) behind the comparison
    n_informative: int            # pairs whose verified outcomes differ (the others score 0 for every mix)
    mean: float = 0.0             # plain mean of the increments of the comparison that decided (or was examined)
    lower: float = -1.0           # its anytime-valid confidence interval (ConfidenceSequence)
    upper: float = 1.0
    comparison: str = ""
    wealth_up: float = 1.0        # e-values of "the step helps" / "the step hurts" for that comparison
    wealth_down: float = 1.0


@dataclass
class _Row:
    segment: int
    group: str
    rb: float
    rl: float
    ts: float
    version: str
    y: Optional[float] = None
    rp: Tuple[float, ...] = ()        # the unlearned (placebo) rewards' values at the same moment


class ProspectiveValidity:
    """Prospective paired evidence on the learned reward vs the bootstrap reward, and the mix decision it supports."""

    def __init__(self, step: float = 0.25, alpha: float = 0.05, max_pending: int = 8192, same_version: bool = True,
                 n_placebo: int = 0):
        self.grid = mix_grid(step)
        # unlearned control rewards (G3-core D-RW1): a random initialisation of R_psi is a fixed random function of
        # the signals and can agree with the verified outcomes by chance. Lowering the mix also requires that R_psi
        # agrees better than each of ``n_placebo`` never-trained rewards drawn like its own initialisation (its own
        # initialisation psi_0 first) and than each of their negations (an equally likely initialisation): an
        # intersection-union test, each component at alpha / 2, so the level stays alpha / 2. A frozen R_psi equals
        # its placebo psi_0 and can never pass; an R_psi that moved without learning is no better than the best of
        # 2 n_placebo exchangeable unlearned rewards with probability >= 1 - 1 / (2 n_placebo + 1).
        self.n_placebo = int(n_placebo)
        self.pl_up = [BettingTest(alpha / 2.0, +1) for _ in range(self.n_placebo)]      # R_psi vs placebo j
        self.pl_upn = [BettingTest(alpha / 2.0, +1) for _ in range(self.n_placebo)]     # R_psi vs -placebo j
        self.pl_sum = [0.0] * self.n_placebo                                             # sum of placebo j's c
        self.nz_up = [0] * len(self.grid)        # increments of "R_psi alone vs mix m_k" that were not 0 (D-RW2)
        # pair only units rewarded by the same R_psi version: a meta-update between the two units would otherwise
        # shift R_psi's output for one of them, and a shift that coincides with a trend in the verified outcomes
        # (the policy improving) would look like agreement although R_psi ranked nothing
        self.same_version = bool(same_version)
        self.n_version_skipped = 0
        self.alpha = float(alpha)
        G = len(self.grid)
        # at grid point k: "R_psi alone agrees better than the mixture" (lowering; fixed sequence) and "the bootstrap
        # alone agrees better than the mixture" (raising; Bonferroni over the G - 1 points that can be raised)
        self.up = [BettingTest(self.alpha / 2.0, +1) for _ in range(G)]
        self.down = [BettingTest(self.alpha / (2.0 * max(1, G - 1)), +1) for _ in range(G)]
        self.cs_up = [ConfidenceSequence(self.alpha) for _ in range(G)]
        self.cs_down = [ConfidenceSequence(self.alpha) for _ in range(G)]
        self.alone = {"bootstrap": ConfidenceSequence(self.alpha), "learned": ConfidenceSequence(self.alpha)}
        self.max_pending = int(max_pending)
        self.rows: "collections.OrderedDict[str, _Row]" = collections.OrderedDict()
        self.groups: Dict[str, List[str]] = {}
        self.scored_groups: Deque[str] = collections.deque(maxlen=65536)
        self._scored: set = set()
        self.slot: Dict[int, Tuple[Any, ...]] = {}     # segment -> unit waiting for its partner
        self.s_b, self.s_l = _Running(), _Running()
        self.n_units = 0
        self.n_pairs = 0
        self.n_informative = 0
        self.recent: Deque[Dict[str, Any]] = collections.deque(maxlen=256)   # last scored pairs (diagnostics)
        self.versions_scored: Dict[str, int] = {}

    # ---------------------------------------------------------------------------------------------- input
    def note(self, exp_id: str, *, segment: int, group: str, r_bootstrap: float, r_learned: float, ts: float = 0.0,
             version: str = "", r_placebo: Sequence[float] = ()) -> None:
        """A decision was rewarded: the two rewards it got at that moment (raw, before mixing), and the placebo
        rewards' values (``n_placebo`` of them; a decision without them is not a unit when placebos are used)."""
        if exp_id in self.rows or group in self._scored:
            return
        rb, rl = float(r_bootstrap), float(r_learned)
        rp = tuple(float(x) for x in r_placebo)
        if not (math.isfinite(rb) and math.isfinite(rl)) or len(rp) != self.n_placebo or \
                not all(math.isfinite(x) for x in rp):
            return
        self.rows[exp_id] = _Row(int(segment), str(group), rb, rl, float(ts), str(version), None, rp)
        self.groups.setdefault(str(group), []).append(exp_id)
        self.s_b.add(rb)
        self.s_l.add(rl)
        while len(self.rows) > self.max_pending:
            old, row = self.rows.popitem(last=False)
            ids = self.groups.get(row.group, [])
            if old in ids:
                ids.remove(old)
            if not ids:
                self.groups.pop(row.group, None)

    def pending(self) -> List[str]:
        return [i for i, r in self.rows.items() if r.y is None]

    def resolve(self, exp_id: str, y: float) -> int:
        """The final verified progress of a noted decision. Returns the number of pairs scored by it (0 or 1)."""
        row = self.rows.get(exp_id)
        if row is None or row.y is not None or not math.isfinite(float(y)):
            return 0
        row.y = float(y)
        ids = self.groups.get(row.group, [])
        if not ids or any(self.rows[i].y is None for i in ids if i in self.rows):
            return 0
        members = [self.rows.pop(i) for i in ids if i in self.rows]
        self.groups.pop(row.group, None)
        self._scored.add(row.group)
        self.scored_groups.append(row.group)
        if len(self._scored) > 65536:
            self._scored = set(self.scored_groups)
        n = len(members)
        vers = {m.version for m in members}
        unit = (row.group, sum(m.rb for m in members) / n, sum(m.rl for m in members) / n,
                sum(m.y for m in members), vers.pop() if len(vers) == 1 else _MIXED,    # type: ignore[misc]
                tuple(sum(m.rp[j] for m in members) / n for j in range(self.n_placebo)))
        for m in members:
            self.versions_scored[m.version] = self.versions_scored.get(m.version, 0) + 1
        self.n_units += 1
        seg = members[0].segment
        other = self.slot.pop(seg, None)
        if other is None:
            self.slot[seg] = unit
            return 0
        if self.same_version and (unit[4] == _MIXED or other[4] != unit[4]):
            self.slot[seg] = unit                 # R_psi changed between the two: the older unit is not scored
            self.n_version_skipped += 1
            return 0
        self._score(other, unit)
        return 1

    def drop(self, exp_id: str) -> int:
        """A noted decision that will never get a verified progress value: no longer a member of its group (the
        group may complete without it). Returns the number of pairs scored by that (0 or 1)."""
        row = self.rows.pop(exp_id, None)
        if row is None:
            return 0
        ids = self.groups.get(row.group, [])
        if exp_id in ids:
            ids.remove(exp_id)
        if not ids:
            self.groups.pop(row.group, None)
            return 0
        if all(self.rows[i].y is not None for i in ids if i in self.rows):
            last = ids[-1]
            y = self.rows[last].y
            self.rows[last].y = None
            return self.resolve(last, float(y))                       # type: ignore[arg-type]
        return 0

    def _score(self, u: Tuple[Any, ...], v: Tuple[Any, ...]) -> None:
        sb, sl = self.s_b.std, self.s_l.std
        c = pair_scores((u[1], v[1]), (u[2], v[2]), (u[3], v[3]), self.grid, sb, sl)
        for k in range(len(self.grid)):
            xu, xd = 0.5 * (c[-1] - c[k]), 0.5 * (c[0] - c[k])
            self.nz_up[k] += int(xu != 0.0)
            self.up[k].update(xu)
            self.cs_up[k].update(xu)
            self.down[k].update(xd)
            self.cs_down[k].update(xd)
        self.alone["bootstrap"].update(float(c[0]))
        self.alone["learned"].update(float(c[-1]))
        if self.n_placebo:
            dy = _sign(u[3] - v[3], 1e-12 * max(1.0, abs(u[3]), abs(v[3])))
            for j in range(self.n_placebo):
                cp = _sign(u[5][j] - v[5][j], 1e-12 * max(1.0, abs(u[5][j]), abs(v[5][j]))) * dy
                self.pl_up[j].update(0.5 * (c[-1] - cp))
                self.pl_upn[j].update(0.5 * (c[-1] + cp))
                self.pl_sum[j] += cp
        self.n_pairs += 1
        informative = abs(u[3] - v[3]) > 1e-12 * max(1.0, abs(u[3]), abs(v[3]))
        self.n_informative += int(informative)
        self.recent.append({"groups": [u[0], v[0]], "y": [u[3], v[3]], "bootstrap": [u[1], v[1]],
                            "learned": [u[2], v[2]], "scores": c, "scale": [sb, sl]})

    # ---------------------------------------------------------------------------------------------- decision
    def _index(self, mix: float) -> int:
        return min(range(len(self.grid)), key=lambda k: abs(self.grid[k] - mix))

    def decide(self, mix: float) -> ValidityDecision:
        """The mix the evidence supports, one grid step at a time (see module doc)."""
        k = self._index(mix)
        G = len(self.grid)

        def dec(action: str, reason: str, after: float, side: str) -> ValidityDecision:
            cs = self.cs_up[k] if side == "+" else self.cs_down[k]
            lo, hi = cs.bounds()
            return ValidityDecision(action, reason, mix, after, self.n_pairs, self.n_informative, cs.mean, lo, hi,
                                    f"{'learned' if side == '+' else 'bootstrap'}_vs_mix_{self.grid[k]:g}",
                                    self.up[k].wealth, self.down[k].wealth)
        if k > 0 and self.down[k].rejected:
            return dec("raise", "prospective_evidence_shows_the_bootstrap_alone_agrees_better_with_verified_outcomes",
                       self.grid[k - 1], "-")
        if k + 1 < G and self.up[k].rejected:
            if self.placebos_beaten():
                return dec("lower", "prospective_evidence_shows_R_psi_alone_agrees_better_with_verified_outcomes_"
                           "than_the_mixture_and_than_every_unlearned_reward", self.grid[k + 1], "+")
            return dec("hold", "R_psi_not_shown_to_agree_better_than_every_unlearned_reward", mix, "+")
        if self.n_pairs == 0:
            why = "no_scored_pairs"
        elif k + 1 < G and self.nz_up[k] == 0:
            # D-RW2: every scored pair ranks the same under R_psi alone and under the mixture in force (the bootstrap
            # tied at reward time), so reward-time evidence cannot tell these mixes apart; what a further step
            # would change is the weight of the bootstrap's later-maturing (outcome-derived) update-time terms
            why = "mixture_and_R_psi_rank_alike_at_reward_time"
        else:
            why = "inconclusive_evidence"
        return dec("hold", why, mix, "+" if k + 1 < G else "-")

    def placebos_beaten(self) -> bool:
        """R_psi agrees better than every unlearned reward and every negated one (vacuous without placebos)."""
        return all(t.rejected for t in self.pl_up) and all(t.rejected for t in self.pl_upn)

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"units": self.n_units, "pairs": self.n_pairs, "informative_pairs": self.n_informative,
                               "units_unpaired_across_versions": self.n_version_skipped,
                               "pending_rows": len(self.pending()), "grid": list(self.grid), "alpha": self.alpha}
        for name, cs in self.alone.items():
            lo, hi = cs.bounds()
            out[f"concordance_{name}"] = {"mean": cs.mean, "lower": lo, "upper": hi}
        out["by_mix"] = {f"{m:g}": {"learned_minus_mix": dict(zip(("lower", "upper"), self.cs_up[k].bounds()),
                                                              mean=self.cs_up[k].mean, e=self.up[k].wealth),
                                    "bootstrap_minus_mix": dict(zip(("lower", "upper"), self.cs_down[k].bounds()),
                                                                mean=self.cs_down[k].mean, e=self.down[k].wealth)}
                         for k, m in enumerate(self.grid)}
        if self.n_placebo:
            pm = [x / self.n_pairs if self.n_pairs else 0.0 for x in self.pl_sum]
            out["placebo"] = {"n": self.n_placebo, "signs": 2,
                              "rejected": sum(t.rejected for t in self.pl_up) + sum(t.rejected for t in self.pl_upn),
                              "of": 2 * self.n_placebo, "best_abs_concordance": max(abs(x) for x in pm),
                              "psi0_concordance": pm[0],
                              "min_e_learned_vs_placebo": min([t.wealth for t in self.pl_up + self.pl_upn])}
        out["nonzero_increments_by_mix"] = {f"{m:g}": self.nz_up[k] for k, m in enumerate(self.grid)}
        return out

    # ---------------------------------------------------------------------------------------------- persistence
    def state_dict(self) -> Dict[str, Any]:
        return {"grid": list(self.grid), "alpha": self.alpha,
                "up": [t.state_dict() for t in self.up], "down": [t.state_dict() for t in self.down],
                "cs_up": [c.state_dict() for c in self.cs_up], "cs_down": [c.state_dict() for c in self.cs_down],
                "alone": {k: v.state_dict() for k, v in self.alone.items()},
                "rows": [[i, r.segment, r.group, r.rb, r.rl, r.ts, r.version, r.y, list(r.rp)]
                         for i, r in self.rows.items()],
                "n_placebo": self.n_placebo, "pl_up": [t.state_dict() for t in self.pl_up],
                "pl_upn": [t.state_dict() for t in self.pl_upn], "pl_sum": list(self.pl_sum), "nz_up": list(self.nz_up),
                "scored_groups": list(self.scored_groups),
                "slot": {str(k): [v[0], v[1], v[2], v[3], v[4], list(v[5])] for k, v in self.slot.items()},
                "s_b": self.s_b.state(), "s_l": self.s_l.state(), "n_units": self.n_units, "n_pairs": self.n_pairs,
                "same_version": self.same_version, "n_version_skipped": self.n_version_skipped,
                "n_informative": self.n_informative, "versions_scored": dict(self.versions_scored)}

    def load_state_dict(self, d: Mapping[str, Any]) -> None:
        if [float(x) for x in d["grid"]] != self.grid:
            raise ValueError(f"validity evidence was collected on the mix grid {d['grid']}, not {self.grid}")
        if int(d.get("n_placebo", 0)) != self.n_placebo:
            raise ValueError(f"validity evidence was collected with {d.get('n_placebo', 0)} unlearned control "
                             f"rewards, not {self.n_placebo}")
        self.alpha = float(d["alpha"])
        for c, s in zip(self.cs_up, d["cs_up"]):
            c.load_state_dict(s)
        for c, s in zip(self.cs_down, d["cs_down"]):
            c.load_state_dict(s)
        for t, s in zip(self.up, d["up"]):
            t.load_state_dict(s)
        for t, s in zip(self.down, d["down"]):
            t.load_state_dict(s)
        for k, s in d["alone"].items():
            self.alone[k].load_state_dict(s)
        self.rows = collections.OrderedDict()
        self.groups = {}
        for row in d["rows"]:
            i, seg, g, rb, rl, ts, ver, y = row[:8]
            rp = tuple(float(x) for x in row[8]) if len(row) > 8 else ()
            self.rows[i] = _Row(int(seg), g, float(rb), float(rl), float(ts), ver, None if y is None else float(y),
                                rp)
            self.groups.setdefault(g, []).append(i)
        self.scored_groups = collections.deque(d["scored_groups"], maxlen=65536)
        self._scored = set(self.scored_groups)
        self.slot = {int(k): (v[0], float(v[1]), float(v[2]), float(v[3]), str(v[4]) if len(v) > 4 else "",
                              tuple(float(x) for x in v[5]) if len(v) > 5 else ())
                     for k, v in d["slot"].items()}
        for t, st in zip(self.pl_up, d.get("pl_up", [])):
            t.load_state_dict(st)
        for t, st in zip(self.pl_upn, d.get("pl_upn", [])):
            t.load_state_dict(st)
        self.pl_sum = [float(x) for x in d.get("pl_sum", [0.0] * self.n_placebo)]
        self.nz_up = [int(x) for x in d.get("nz_up", [0] * len(self.grid))]
        self.same_version = bool(d.get("same_version", self.same_version))
        self.n_version_skipped = int(d.get("n_version_skipped", 0))
        self.s_b.load(d["s_b"])
        self.s_l.load(d["s_l"])
        self.n_units, self.n_pairs = int(d["n_units"]), int(d["n_pairs"])
        self.n_informative = int(d["n_informative"])
        self.versions_scored = {k: int(v) for k, v in d.get("versions_scored", {}).items()}
