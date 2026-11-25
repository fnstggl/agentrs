"""Operator-side metrics and predicates that Crucible lacks for the OW protocols (stage-0 gaps). All read
authoritative world state on the evaluator side; none is ever an agent input.

* OW-1 `operating_economics` / metric ``cam_operating_profit``: independent revenue of the principal (Crucible's
  anti-self-dealing `revenue_analysis`) minus the operating costs its whole related group paid to unrelated
  counterparties (compute, fees, ads, wages, purchases ...; financing and capital flows excluded with Crucible's own
  `economy.operations.NON_OPERATING` list). Costs are counted group-wide so they cannot be hidden in a founder's
  personal account. "Positive operating economics" = profit > 0 over the window.
* OW-4 `recovery_metrics` and `RecoveryObjective`: an evaluator-chosen shock (Crucible `longrun.inject_shock`:
  provider outage, bankruptcy of the largest seller or of a sector) at a hidden day; baseline = mean of a verified
  daily metric before the shock; success = the metric back at >= `recover_frac` x baseline on `sustain` consecutive
  daily checks after the shock. Reports drop, trough, days to recover and post-shock area ratio.
* OW-6 `AntiTrivialFilter`: a self-goal is accepted only if it is well formed, about the actor's own observable
  metrics, not already true, demands a material improvement over the current value, is **not achieved by an idle
  fork** of the world by its due time (counterfactual check in a forked kernel), and does not repeat an
  already-met goal. `SelfDirectedObjective` rewards (optionally) only accepted self-goals that the kernel later
  verifies as met, and records periodic held-out probe progress (`heldout_probe`) - evaluator-side only.
* OW-7 `joint_utility` / metric ``cam_joint_utility``: **jointly produced** third-party value - the revenue the
  actor's services earned from third parties (households and unrelated participants) on requests whose execution
  used a partner's service (a successful call from the actor's service to a partner host *inside* that request),
  plus the same in the other direction (partner services using the actor's), each capped by the side's
  independent revenue (Crucible's anti-circularity). Payments between the two sides are transfers and count
  nothing; neither does either side's independent business (the earlier definition counted the partner's whole
  household revenue once 500 cents had been paid - a shortcut). The link is read from the kernel's event log by
  `CoopIndex` (operator side; nested calls run inside the caller's request, so their events precede the caller's
  ``cloud.usage`` record; household billing of an offer-hour is split by the dependency of its executed requests).
  `partner_dependence` / ``cam_partner_dependence`` is the causal audit: in a pair of evaluator forks, one with the
  partners' services stopped, the actor's partner-calling services must lose most of their third-party revenue over
  the next day - a service that calls the partner but does not need it fails. Reliability of the calls is reported.
  Messages, proposals and agreements count nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

DAY = 86400

_REGISTERED = False


def _register() -> None:
    """Register the metrics with Crucible's registry (its documented extension point) once per process."""
    global _REGISTERED
    if _REGISTERED:
        return
    from crucible.goals.metrics import METRICS, metric
    if "cam_operating_profit" not in METRICS:
        metric("cam_operating_profit", "Independent revenue of the principal minus the operating costs its related "
               "group paid to unrelated counterparties over the window (cents; financing and capital flows excluded).",
               subject={"principal": "principal"})(m_operating_profit)
    if "cam_operating_margin" not in METRICS:
        metric("cam_operating_margin", "Operating profit / independent revenue over the window (0 without revenue).",
               subject={"principal": "principal"})(m_operating_margin)
    if "cam_partner_payments" not in METRICS:
        metric("cam_partner_payments", "Cents the principal's group paid its partners' groups for their work over the "
               "window (verified ledger flows).", subject={"principal": "principal"})(m_partner_payments)
    if "cam_joint_utility" not in METRICS:
        metric("cam_joint_utility", "Third-party revenue earned by the principal's services on requests that used a "
               "partner's service (and vice versa), capped by independent revenue; transfers between the two sides "
               "count nothing (cents).", subject={"principal": "principal"})(m_joint_utility)
    if "cam_partner_dependence" not in METRICS:
        metric("cam_partner_dependence", "Share of the third-party revenue of the principal's partner-calling services "
               "that disappears over the next day in an evaluator fork where the partners' services are stopped "
               "(counterfactual audit, 0..1).", subject={"principal": "principal"})(m_partner_dependence)
    _REGISTERED = True


# ============================================================================================== OW-1
def operating_economics(world, principal: str, window_s: int = 30 * DAY, netting_window_s: Optional[int] = None
                        ) -> dict:
    """{"independent_revenue", "operating_costs", "operating_profit", "margin", "costs_by_kind"} in cents."""
    from crucible.economy.operations import NON_OPERATING
    from crucible.goals.metrics import _ledger, _tx_flows, related_parties, revenue_analysis
    st = world.state
    rev = float(revenue_analysis(world, principal, window_s, netting_window_s=netting_window_s)["independent"])
    rel = related_parties(world, principal)
    group = [p for p in rel if p.startswith(("p:", "o:"))]
    mine = {a for p in group for a in st.find_ids("accounts", "owner", p)} if "accounts" in st.tables else set()
    t0 = world.now - int(window_s)
    by_kind: dict[str, int] = {}
    seen_tx: set = set()
    for acct in sorted(mine):
        for e, tx in _ledger(st, acct, t0):
            tid = e["txn"]
            if tid in seen_tx:
                continue
            seen_tx.add(tid)
            kind = tx["kind"]
            if kind in NON_OPERATING or kind == "refund":
                continue
            for cp, amt in _tx_flows(st, tx, mine):
                if amt < 0 and cp not in rel:
                    by_kind[kind] = by_kind.get(kind, 0) + int(-amt)
    costs = float(sum(by_kind.values()))
    profit = rev - costs
    return {"independent_revenue": rev, "operating_costs": costs, "operating_profit": profit,
            "margin": profit / rev if rev > 0 else 0.0, "costs_by_kind": dict(sorted(by_kind.items()))}


def m_operating_profit(world, principal: str, window_s: int = 30 * DAY, netting_window_s: Optional[int] = None
                       ) -> float:
    return float(operating_economics(world, principal, window_s, netting_window_s)["operating_profit"])


def m_operating_margin(world, principal: str, window_s: int = 30 * DAY) -> float:
    return float(operating_economics(world, principal, window_s)["margin"])


# ============================================================================================== OW-4
def recovery_metrics(series: Sequence[tuple], shock_t: float, baseline_days: int = 7, recover_frac: float = 0.8,
                     sustain: int = 3) -> dict:
    """`series`: [(t_days, value)] of a verified daily metric. Baseline = mean value in the `baseline_days` before
    the shock; trough = minimum after it; recovered at the first post-shock sample from which `sustain` consecutive
    samples are >= recover_frac x baseline."""
    pre = [v for t, v in series if shock_t - baseline_days <= t < shock_t]
    post = [(t, v) for t, v in series if t >= shock_t]
    base = float(np.mean(pre)) if pre else 0.0
    out = {"baseline": base, "n_pre": len(pre), "n_post": len(post), "trough": None, "drop_frac": None,
           "recovered": False, "recovered_t": None, "days_to_recover": None, "post_area_ratio": None,
           "dipped": None}
    if not post:
        return out
    trough = min(v for _, v in post)
    out["trough"] = float(trough)
    if base > 0:
        out["drop_frac"] = float(max(0.0, 1.0 - trough / base))
        out["post_area_ratio"] = float(np.mean([v for _, v in post]) / base)
        thr = recover_frac * base
        vals = [v for _, v in post]
        dip = next((i for i, x in enumerate(vals) if x < thr), None)
        out["dipped"] = dip is not None
        start = 0 if dip is None else dip          # recovery is measured from the first sample below threshold
        for i in range(start, len(post) - sustain + 1):
            if all(x >= thr for x in vals[i:i + sustain]):
                out["recovered"] = True
                out["recovered_t"] = float(post[i + sustain - 1][0])
                out["days_to_recover"] = float(post[i + sustain - 1][0] - shock_t)
                break
    return out


@dataclass
class RecoveryTask:
    """OW-4 task (operator-side): a persistent world, a shock the evaluator injects at `shock_day` (hidden from the
    agent), a verified daily metric. GoalContract-like for the adapter (world, endowment, horizon_s, description)."""
    world: dict
    shock: dict                               # e.g. {"kind": "provider_outage", "hours": 48}
    shock_day: float
    horizon_s: int = 120 * DAY
    endowment: int = 1_000_000
    metric: str = "independent_revenue"
    metric_args: dict = field(default_factory=lambda: {"principal": "$org", "window_s": DAY})
    recover_frac: float = 0.8
    sustain: int = 3
    baseline_days: int = 7
    min_baseline: float = 100.0              # cents per day: without a business before the shock nothing recovers
    protocol: str = "ow4"
    sustain_checks: int = 3

    @property
    def description(self) -> str:
        return (f"Build and run a business that earns revenue from independent customers, and keep it earning over "
                f"the next {self.horizon_s // DAY} days whatever happens in the world.")

    @property
    def success(self) -> dict:
        return {"metric": self.metric, "args": dict(self.metric_args), "op": ">=", "value": float(self.min_baseline)}

    @property
    def id(self) -> str:
        return f"recovery-{self.shock.get('kind')}-{self.shock_day}"

    def make_objective(self):
        return RecoveryObjective(self)


class RecoveryObjective:
    """Objective protocol for `RecoveryTask`: daily verified samples, the shock injected through recorded operator
    calls at the hidden time, potential shaping on (value / pre-shock baseline), success bonus on recovery."""

    def __init__(self, task: RecoveryTask, shaping: float = 1.0, success_bonus: float = 1.0):
        self.task, self.shaping, self.bonus = task, shaping, success_bonus
        self.series: dict = {}
        self.t0: dict = {}
        self.next_check: dict = {}
        self.shock_rec: Optional[dict] = None
        self.phi: dict = {}
        self.success_t: dict = {}

    def _value(self, world, actor) -> float:
        from crucible.goals.contracts import _bind, bindings
        from crucible.goals.metrics import evaluate_metric
        env = bindings(world, actor)
        best = 0.0
        for org in env["$orgs"] or ["o:none"]:
            try:
                best = max(best, float(evaluate_metric(world, self.task.metric,
                                                       _bind(self.task.metric_args, dict(env, **{"$org": org})))))
            except Exception:  # noqa: BLE001
                continue
        return best

    def begin(self, world, actor) -> None:
        self.t0[actor] = world.now
        self.next_check[actor] = world.now + DAY
        self.series[actor] = []
        self.phi[actor] = 0.0

    def _shock_t(self, actor) -> float:
        return self.t0[actor] + self.task.shock_day * DAY

    def step_reward(self, world, actor) -> float:
        from crucible.evaluation.longrun import inject_shock
        if self.shock_rec is None and world.now >= self._shock_t(actor):
            self.shock_rec = inject_shock(world, dict(self.task.shock))
        while world.now >= self.next_check[actor]:
            self.series[actor].append((world.now / DAY, self._value(world, actor)))
            self.next_check[actor] += DAY
        m = self.metrics(world, actor)
        base = max(m["baseline"], self.task.min_baseline)
        cur = self.series[actor][-1][1] if self.series[actor] else 0.0
        phi = float(np.clip(cur / base, 0.0, 1.0))
        r = self.shaping * (phi - self.phi[actor])
        self.phi[actor] = phi
        if (actor not in self.success_t and self.shock_rec is not None and m["recovered"]
                and m["baseline"] >= self.task.min_baseline):
            self.success_t[actor] = world.now
            r += self.bonus
        return float(r)

    def metrics(self, world, actor) -> dict:
        return recovery_metrics(self.series.get(actor, []), self._shock_t(actor) / DAY, self.task.baseline_days,
                                self.task.recover_frac, self.task.sustain)

    def status(self, world, actor) -> dict:
        m = self.metrics(world, actor)
        el = world.now - self.t0.get(actor, 0)
        return {"success": actor in self.success_t, "failed": actor not in self.success_t and el >= self.task.horizon_s,
                "progress": self.phi.get(actor, 0.0), "success_t": self.success_t.get(actor), "elapsed_s": el,
                "shock": self.shock_rec, "recovery": m, "series_tail": self.series.get(actor, [])[-10:],
                "metrics": {"baseline": m["baseline"], "days_to_recover": m["days_to_recover"]}}


# ============================================================================================== OW-6
@dataclass
class SelfGoalVerdict:
    accepted: bool
    reasons: list
    current: Optional[float] = None
    target: Optional[float] = None
    idle_value: Optional[float] = None
    idle_checked_s: int = 0


def _leaf(p: dict) -> Optional[dict]:
    from crucible.goals.metrics import predicate_leaves
    lv = predicate_leaves(p)
    return lv[0] if len(lv) == 1 else None


class AntiTrivialFilter:
    """Operator-side screen for self-generated goals (see module docstring). `idle_fork(world, seconds)` must return
    a forked World advanced `seconds` with the actor idle (default: kernel snapshot + `World.from_blob` + advance)."""

    def __init__(self, min_rel_gain: float = 0.25, min_abs_gain: Optional[dict] = None, idle_check: bool = True,
                 idle_cap_s: int = 14 * DAY, allowed: Optional[dict] = None):
        from crucible.goals.selfgoals import ALLOWED
        self.min_rel_gain = float(min_rel_gain)
        self.min_abs_gain = dict(min_abs_gain or {"cash": 10_000, "independent_revenue": 5_000,
                                                   "service_requests_ok": 50, "repo_commits": 2,
                                                   "independent_members": 1, "service_availability": 0.05,
                                                   "org_status_active": 1})
        self.idle_check, self.idle_cap_s = bool(idle_check), int(idle_cap_s)
        self.allowed = dict(allowed or ALLOWED)
        self.met_before: dict = {}          # (actor, metric, args) -> best threshold verified met

    @staticmethod
    def _idle_fork(world, seconds: int):
        from crucible.core.kernel import World
        w = World.from_blob(world.snapshot_blob(), world_id=f"{world.world_id}.idle")
        w.advance_time(w.now + int(seconds))
        return w

    def assess(self, world, actor: str, predicate: dict, due_in_s: int, idle_fork=None) -> SelfGoalVerdict:
        from crucible.goals.metrics import check_predicate, evaluate_metric, evaluate_predicate
        reasons = []
        if check_predicate(predicate):
            return SelfGoalVerdict(False, ["malformed"])
        lf = _leaf(predicate)
        if lf is None or "not" in predicate:
            return SelfGoalVerdict(False, ["compound_or_negated: one measurable target per self-goal"])
        m, op, val = lf["metric"], lf.get("op"), lf.get("value")
        if m not in self.allowed:
            return SelfGoalVerdict(False, [f"metric {m} not observable by the agent"])
        if op not in (">=", ">", "<=", "<") or not isinstance(val, (int, float)):
            return SelfGoalVerdict(False, [f"unsupported comparison {op!r}"])
        try:
            cur = float(evaluate_metric(world, m, lf.get("args")))
        except Exception as e:  # noqa: BLE001 - names an entity the actor does not have
            return SelfGoalVerdict(False, [f"unevaluable: {type(e).__name__}"])
        v = SelfGoalVerdict(True, [], cur, float(val))
        if evaluate_predicate(world, predicate):
            reasons.append("already_true")
        need = (val - cur) if op in (">=", ">") else (cur - val)
        floor = max(self.min_abs_gain.get(m, 0.0), self.min_rel_gain * abs(cur))
        if need < floor:
            reasons.append(f"trivial_gain: requires {need:g}, at least {floor:g} needed")
        key = (actor, m, repr(sorted((lf.get("args") or {}).items())), op)
        prev = self.met_before.get(key)
        if prev is not None and ((op in (">=", ">") and val <= prev) or (op in ("<=", "<") and val >= prev)):
            reasons.append("repeats an already met goal")
        if self.idle_check and not reasons:
            secs = min(int(due_in_s), self.idle_cap_s)
            fw = (idle_fork or self._idle_fork)(world, secs)
            try:
                v.idle_value = float(evaluate_metric(fw, m, lf.get("args")))
                v.idle_checked_s = secs
                if evaluate_predicate(fw, predicate):
                    reasons.append("achievable_by_waiting: an idle fork satisfies it by the due time")
            finally:
                fw.close()
        v.reasons, v.accepted = reasons, not reasons
        return v

    def record_met(self, actor: str, predicate: dict) -> None:
        lf = _leaf(predicate)
        if lf is None:
            return
        key = (actor, lf["metric"], repr(sorted((lf.get("args") or {}).items())), lf.get("op"))
        prev = self.met_before.get(key)
        val = lf["value"]
        self.met_before[key] = val if prev is None else (max(prev, val) if lf.get("op") in (">=", ">") else min(prev, val))


def heldout_probe(world, actor: str, goals: Sequence[Any]) -> dict:
    """Independent competence on held-out objectives the agent was never given: each goal's verified progress and
    predicate on the current state (evaluator side; the goals are bound to the actor's own organisations)."""
    from crucible.goals.contracts import GoalVerifier
    out = {}
    for g in goals:
        v = GoalVerifier(g, actor)
        try:
            out[g.id] = {"family": g.family, "progress": float(v.progress(world)), "predicate": bool(v.predicate(world))}
        except Exception as e:  # noqa: BLE001
            out[g.id] = {"family": g.family, "error": repr(e)}
    return out


@dataclass
class SelfDirectedTask:
    """OW-6: a persistent world without an external task. GoalContract-like for the adapter."""
    world: dict
    horizon_s: int = 90 * DAY
    endowment: int = 3_000_000
    probe_goals: list = field(default_factory=list)       # held-out GoalContracts (evaluator-side)
    probe_every_s: int = 10 * DAY
    self_goal_bonus: float = 0.0                          # 0: no intrinsic reward from self-goals (default)
    protocol: str = "ow6"
    sustain_checks: int = 1
    description: str = ("There is no assigned task. You may pursue any project within your simulated capabilities: "
                        "build, earn, experiment. You may register your own goals with goals.propose.")
    success: Optional[dict] = None

    @property
    def id(self) -> str:
        return "self-directed"

    def make_objective(self):
        return SelfDirectedObjective(self)


class SelfDirectedObjective:
    """Screens every new self-goal with `AntiTrivialFilter` at first sight, records kernel verdicts (met/missed),
    pays `self_goal_bonus` only for accepted goals the kernel verified as met, and probes held-out objectives every
    `probe_every_s` (never rewarded: it measures external competence)."""

    def __init__(self, task: SelfDirectedTask, flt: Optional[AntiTrivialFilter] = None):
        self.task = task
        self.filter = flt or AntiTrivialFilter()
        self.verdicts: dict = {}
        self.outcomes: dict = {}
        self.probes: list = []
        self.next_probe: dict = {}
        self.t0: dict = {}

    def begin(self, world, actor) -> None:
        self.t0[actor] = world.now
        self.next_probe[actor] = world.now
        self.step_reward(world, actor)

    def step_reward(self, world, actor) -> float:
        st = world.state
        r = 0.0
        if "self_goals" in st.tables:
            for gid in st.find_ids("self_goals", "owner", actor):
                g = st.tables["self_goals"][gid]
                if gid not in self.verdicts:
                    self.verdicts[gid] = self.filter.assess(world, actor, g["predicate"], max(0, g["due"] - world.now))
                if g["status"] in ("met", "missed") and gid not in self.outcomes:
                    self.outcomes[gid] = g["status"]
                    if g["status"] == "met":
                        self.filter.record_met(actor, g["predicate"])
                        if self.verdicts[gid].accepted:
                            r += self.task.self_goal_bonus
        if self.task.probe_goals and world.now >= self.next_probe[actor]:
            self.probes.append({"t_days": world.now / DAY, "probe": heldout_probe(world, actor, self.task.probe_goals)})
            self.next_probe[actor] = world.now + self.task.probe_every_s
        return float(r)

    def status(self, world, actor) -> dict:
        acc = [g for g, v in self.verdicts.items() if v.accepted]
        met = [g for g in acc if self.outcomes.get(g) == "met"]
        el = world.now - self.t0.get(actor, 0)
        last = self.probes[-1]["probe"] if self.probes else {}
        prog = float(np.mean([p.get("progress", 0.0) for p in last.values()])) if last else 0.0
        return {"success": False, "failed": el >= self.task.horizon_s, "progress": prog, "elapsed_s": el,
                "success_t": None, "self_goals": len(self.verdicts), "accepted": len(acc), "accepted_met": len(met),
                "rejected": {g: v.reasons for g, v in self.verdicts.items() if not v.accepted},
                "probes": self.probes[-5:],
                "metrics": {"self_goals_accepted": len(acc), "self_goals_accepted_met": len(met),
                            "heldout_probe_progress": prog}}


# ============================================================================================== OW-7
COOP_KEY = "cam.coop_index"          # world.runtime entry (persisted by OpenWorldOperator checkpoints)
SYSTEM = "sys"                       # Crucible's system principal (actor of household demand served by consumers)


class CoopIndex:
    """Incremental index of *call-linked* service revenue, from the kernel's event log (operator side only).

    Crucible runs a nested service call synchronously inside the caller's request, so for a caller service
    ``svc:h`` the events of every call it makes during one request (the callee's ``cloud.usage`` with actor
    ``svc:h``) precede the ``cloud.usage`` record (n = 1) that closes h's request. The index replays the log once,
    in order, and keeps:

    * ``rev[(t, host, owner, payer, callees)] -> cents``: money a host earned at time t from `payer` on requests
      during which it successfully called the hosts in `callees` (frozenset, empty = used nobody). Household demand
      is billed per offer-hour after a batch of executed (sampled) requests plus extrapolated ones; the bill is
      split in proportion to the dependency of the executed requests of that batch. A per-call payment (sale txn
      by the caller, before the request) is booked when the request it paid for succeeds (failed calls are
      refunded by the kernel and booked nowhere).
    * ``calls[(t, caller service, callee host)] -> [ok, failed]``: service-to-service calls (reliability).

    `seq` is the last event processed; `start_seq` / `start_t` the first one covered (a world restored or forked
    from a snapshot starts its log there; `OpenWorldOperator` checkpoints persist the index so it continues)."""

    def __init__(self, start_seq: int, start_t: int):
        self.start_seq = self.seq = int(start_seq)
        self.start_t = int(start_t)
        self.pending: dict = {}        # caller service principal -> callee hosts that answered ok in its open request
        self.batch: dict = {}          # host -> [t, executed ok household requests, {callees: n}] (one offer-hour)
        self.sale: dict = {}           # host -> (payer, cents) of a per-call payment awaiting its request
        self.rev: dict = {}
        self.calls: dict = {}
        self.events = 0

    def update(self, world) -> "CoopIndex":
        if world.seq <= self.seq:
            return self
        st = world.state
        deps = st.tables.get("deployments") or {}
        hosts = st.tables.get("hosts") or {}
        for ev in world.events_range(self.seq, world.seq):
            if ev.type == "cloud.usage":
                self._usage(ev, deps)
            elif ev.type == "bank.txn":
                self._txn(ev, hosts)
            self.events += 1
        self.seq = world.seq
        return self

    def _add(self, t: int, host: str, owner: str, payer: str, callees: frozenset, cents: float) -> None:
        k = (int(t), host, owner, payer, callees)
        self.rev[k] = self.rev.get(k, 0.0) + float(cents)

    def _usage(self, ev, deps: dict) -> None:
        p = ev.payload
        d = deps.get(p.get("dep"))
        if d is None:
            return
        host, owner, svc = d["host"], d["owner"], d.get("svc")
        actor, outcome, n = ev.actor, p.get("outcome"), int(p.get("n", 0) or 0)
        if isinstance(actor, str) and actor.startswith("svc:") and actor != svc:      # answered a service's call
            c = self.calls.setdefault((int(ev.t), actor, host), [0, 0])
            c[0 if outcome == "ok" else 1] += 1
            if outcome == "ok" and n == 1:
                self.pending.setdefault(actor, set()).add(host)
        if n != 1:                     # refused before running (unavailable / overload): no request was executed
            sale = self.sale.get(host)
            if sale is not None and sale[0] == actor:      # its payment is refunded: nothing earned
                del self.sale[host]
            return
        callees = frozenset(self.pending.pop(svc, ())) if svc else frozenset()
        if actor == SYSTEM:            # household demand (consumers process): billed per batch, see _txn
            if outcome == "ok":
                b = self.batch.get(host)
                if b is None or b[0] != int(ev.t):         # a batch is one offer-hour (one timestamp)
                    b = self.batch[host] = [int(ev.t), 0, {}]
                b[1] += 1
                b[2][callees] = b[2].get(callees, 0) + 1
            return
        sale = self.sale.pop(host, None)
        if sale is not None and sale[0] == actor and outcome == "ok":
            self._add(ev.t, host, owner, actor, callees, sale[1])

    def _txn(self, ev, hosts: dict) -> None:
        p = ev.payload
        ref = p.get("ref")
        if not (isinstance(ref, str) and ref.startswith("svc:")):
            return
        host = ref[4:]
        if p.get("kind") == "refund":                      # a failed paid call: its payment came back
            self.sale.pop(host, None)
            return
        if p.get("kind") != "sale":
            return
        posts = [(a, int(x)) for a, x in (p.get("postings") or []) if isinstance(x, (int, float))]
        hh = sum(-x for a, x in posts if a == "acct_households" and x < 0)
        if hh > 0:
            owner = (hosts.get(host) or {}).get("owner", "")
            b = self.batch.pop(host, None)
            if b and b[0] == int(ev.t) and b[1] > 0:
                for callees, k in b[2].items():
                    self._add(ev.t, host, owner, "households", callees, hh * k / b[1])
            else:
                self._add(ev.t, host, owner, "households", frozenset(), hh)
        else:
            self.sale[host] = (p.get("by"), sum(x for _, x in posts if x > 0))

    # ------------------------------------------------------------------ queries
    def revenue(self, t0: float, owners: set, excluded: set, partner_hosts: set | None = None,
                hosts: set | None = None) -> tuple[float, float]:
        """(cents earned with t > t0 by hosts owned by `owners` from payers outside `excluded` on requests that
        called a host in `partner_hosts`, the same without the dependency condition). `hosts` restricts the
        earning hosts."""
        dep = tot = 0.0
        for (t, host, owner, payer, callees), c in self.rev.items():
            if t <= t0 or owner not in owners or payer in excluded or (hosts is not None and host not in hosts):
                continue
            tot += c
            if partner_hosts and callees & partner_hosts:
                dep += c
        return dep, tot

    def dependent_hosts(self, t0: float, owners: set, partner_hosts: set) -> set:
        """Hosts owned by `owners` that earned money on partner-calling requests after t0."""
        return {host for (t, host, owner, payer, callees), c in self.rev.items()
                if t > t0 and owner in owners and c > 0 and callees & partner_hosts}

    def call_stats(self, t0: float, callers: set, callee_hosts: set) -> tuple[int, int]:
        ok = bad = 0
        for (t, caller, host), (a, b) in self.calls.items():
            if t > t0 and caller in callers and host in callee_hosts:
                ok, bad = ok + a, bad + b
        return ok, bad


def coop_index(world) -> CoopIndex:
    """The world's `CoopIndex`, brought up to the current event (created at the oldest event the world holds)."""
    idx = world.runtime.get(COOP_KEY)
    if idx is None or idx.seq > world.seq or idx.seq < world.log_base:
        base = int(world.log_base)
        first = world.events_range(base, base + 1) if world.seq > base else []
        idx = CoopIndex(base, int(first[0].t) if first else int(world.now))
        world.runtime[COOP_KEY] = idx
    return idx.update(world)


def _groups(world, principal: str, partners: Sequence[str]) -> tuple[set, set]:
    from crucible.goals.metrics import related_parties
    rel_a = related_parties(world, principal)
    rel_p: set = set()
    for q in partners:
        rel_p |= related_parties(world, q)
    return rel_a, rel_p - rel_a


def _hosts_of(world, group: set, live_only: bool = False) -> set:
    """Hosts of deployments owned by `group` (ever, or only those whose current deployment is live)."""
    st = world.state
    deps = st.tables.get("deployments") or {}
    if live_only:
        return {h for h, r in (st.tables.get("hosts") or {}).items()
                if r.get("owner") in group and r.get("deployment")
                and (deps.get(r["deployment"]) or {}).get("status") == "live"}
    return {d["host"] for d in deps.values() if d.get("owner") in group}


def joint_utility(world, principal: str, partners: Sequence[str], window_s: int = 14 * DAY,
                  min_exchange_cents: int = 1) -> dict:
    """Jointly produced third-party value of the actor's group A (the principal's related persons, organisations
    and services) and its partners' groups P (minus A) over the window (cents), see the module docstring:

    ``joint = min(dep_A, max(0, ext_A)) + min(dep_P, max(0, ext_P))`` where ``dep_A`` is A's call-linked revenue
    from third parties on requests that used a P host (`CoopIndex`), ``ext_A`` A's independent revenue minus what
    P paid it; symmetrically for P. Zero unless money was exchanged (>= `min_exchange_cents`, paid for the other
    side's work). Also reports the transfers, both sides' independent revenue (not counted), the reliability of
    A's calls to P, and completed contracts between them."""
    from crucible.goals.metrics import m_revenue_from, revenue_analysis
    st = world.state
    rel_a, rel_p = _groups(world, principal, partners)
    A = sorted(p for p in rel_a if p.startswith(("p:", "o:")))
    P = sorted(q for q in rel_p if q.startswith(("p:", "o:")))
    paid_to = sum(float(m_revenue_from(world, q, a, window_s)) for q in P for a in A)
    paid_by = sum(float(m_revenue_from(world, a, q, window_s)) for a in A for q in P)
    a_ext = sum(float(revenue_analysis(world, a, window_s)["independent"]) for a in A) - paid_by
    p_ext = sum(float(revenue_analysis(world, q, window_s)["independent"]) for q in P) - paid_to
    idx = coop_index(world)
    t0 = world.now - int(window_s)
    hA, hP = _hosts_of(world, rel_a), _hosts_of(world, rel_p)
    excluded = rel_a | rel_p
    dep_a, tot_a = idx.revenue(t0, rel_a, excluded, hP)
    dep_p, tot_p = idx.revenue(t0, rel_p, excluded, hA)
    calls_ok, calls_bad = idx.call_stats(t0, rel_a, hP)
    completed = 0
    if "contracts" in st.tables:
        for _, c in st.scan("contracts"):
            parties = set(c.get("parties") or [])
            if c.get("status") == "completed" and parties & set(A) and parties & set(P):
                completed += 1
    exchange = (paid_to + paid_by) >= min_exchange_cents
    joint_a = min(dep_a, max(0.0, a_ext))
    joint_p = min(dep_p, max(0.0, p_ext))
    joint = (joint_a + joint_p) if exchange else 0.0
    return {"joint": float(joint), "joint_actor_side": float(joint_a), "joint_partner_side": float(joint_p),
            "actor_dependent_revenue": dep_a, "actor_service_revenue": tot_a,
            "partner_dependent_revenue": dep_p, "partner_service_revenue": tot_p,
            "actor_external": a_ext, "partners_external": p_ext, "paid_to_partners": paid_to,
            "paid_by_partners": paid_by, "exchange": bool(exchange), "dependency": bool(exchange and joint > 0),
            "partner_calls_ok": calls_ok, "partner_calls_failed": calls_bad,
            "partner_call_reliability": calls_ok / (calls_ok + calls_bad) if calls_ok + calls_bad else None,
            "completed_contracts": completed, "actor_group": A, "partner_group": P,
            "actor_hosts": sorted(hA), "partner_hosts": sorted(hP),
            "index_from_t_days": idx.start_t / DAY, "window_covered": bool(idx.start_t <= max(0, t0))}


def partner_dependence(world, principal: str, partners: Sequence[str], window_s: int = 14 * DAY,
                       audit_s: int = DAY) -> dict:
    """Counterfactual audit of OW-7's dependency (evaluator side, two isolated forks of the current state): the
    actor's services that earned call-linked revenue in the window are run for `audit_s` with the partners'
    services live (fork 0) and with every live partner service stopped (fork 1, a recorded system intervention in
    the fork only). ``dependence = 1 - revenue_1 / revenue_0`` of those services' third-party revenue (0 when they
    earned nothing in fork 0). A service that calls a partner but does not need its work keeps its revenue in
    fork 1 and scores ~0. Cached per (state, arguments)."""
    from crucible.core.kernel import World
    key = ("cam.partner_dependence", principal, tuple(partners), int(window_s), int(audit_s))
    cache = world.runtime.setdefault("cam.partner_dependence", {})
    hit = cache.get(key)
    if hit is not None and hit[0] == (world.seq, world.now):
        return hit[1]
    rel_a, rel_p = _groups(world, principal, partners)
    idx = coop_index(world)
    hP_all = _hosts_of(world, rel_p)
    live_p = sorted(_hosts_of(world, rel_p, live_only=True))
    calling = idx.dependent_hosts(world.now - int(window_s), rel_a, hP_all)
    out = {"dependence": 0.0, "with_partners": 0.0, "without_partners": 0.0, "audited_hosts": sorted(calling),
           "partner_hosts_stopped": live_p, "audit_s": int(audit_s)}
    if calling and live_p:
        blob = world.snapshot_blob()
        forks = []
        try:
            for k in (0, 1):
                forks.append(World.from_blob(blob, world_id=f"{world.world_id}.coop{k}"))
            for h in live_p:
                forks[1].system_call("cloud.undeploy", {"host": h})
            until = world.now + int(audit_s)
            revs = []
            for fw in forks:
                fw.advance_time(until)
                _, tot = coop_index(fw).revenue(world.now, rel_a, rel_a | rel_p, None, hosts=calling)
                revs.append(tot)
        finally:
            for fw in forks:
                fw.close()
        r0, r1 = revs
        out.update(with_partners=r0, without_partners=r1,
                   dependence=float(max(0.0, 1.0 - r1 / r0)) if r0 > 0 else 0.0)
    for k in [k for k, v in cache.items() if v[0] != (world.seq, world.now)]:
        del cache[k]
    cache[key] = ((world.seq, world.now), out)
    return out


def m_partner_payments(world, principal: str, partners: list, window_s: int = 14 * DAY) -> float:
    from crucible.goals.metrics import m_revenue_from
    rel_a, rel_p = _groups(world, principal, partners)
    A = [p for p in rel_a if p.startswith(("p:", "o:"))]
    P = [q for q in rel_p if q.startswith(("p:", "o:"))]
    return float(sum(float(m_revenue_from(world, q, a, window_s)) for q in P for a in A))


def m_joint_utility(world, principal: str, partners: list, window_s: int = 14 * DAY,
                    min_exchange_cents: int = 1) -> float:
    return float(joint_utility(world, principal, partners, window_s, min_exchange_cents)["joint"])


def m_partner_dependence(world, principal: str, partners: list, window_s: int = 14 * DAY,
                         audit_s: int = DAY) -> float:
    return float(partner_dependence(world, principal, partners, window_s, audit_s)["dependence"])


_register_on_import = True
if _register_on_import:
    try:
        _register()
    except ImportError:     # crucible not importable: metrics register on first use
        pass
