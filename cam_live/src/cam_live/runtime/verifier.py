"""cam_live.runtime.verifier — independent objective verification (spec 13.2).

The verifier is not a model and takes nothing from the learner except the question. It reads the permitted
evidence itself (through a :class:`MembershipSource` or :class:`MetricSource`), applies exclusion rules that are
*data* (from the operator's task file), signs every :class:`~cam_live.contracts.VerifiedOutcome` with a key file
only its own uid can read, and appends every evaluation to a hash-chained log.

Membership objectives: a member counts only when nothing in the evidence speaks against it. Accounts whose
authenticity cannot be established are *uncertain*: they widen the reported interval instead of being counted, and
the objective is reported ``achieved`` only when the lower confidence bound reaches the target.

Secondary measurements (retention, participation, consented feedback, resource use, cost, time) are reported as
separate :class:`~cam_live.contracts.Quantity` values and never enter the objective status.
"""
from __future__ import annotations

import hashlib
import hmac
import importlib
import json
import math
import os
import stat
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

from ..contracts import (Clock, GoalStatus, OutcomeKind, OutcomeObservation, OutcomeQuery, OutcomeStatus, Quantity,
                         RunFiles, TaskSpec, VerifiedOutcome)
from .logs import atomic_write_json, iter_jsonl, read_json

DAY = 86400.0


# --------------------------------------------------------------------------------------------------------------
# evidence sources
# --------------------------------------------------------------------------------------------------------------
@dataclass
class MemberRecord:
    id: str
    is_bot: bool = False
    joined_at: float = 0.0
    pending: bool = False                # has not passed the platform's membership screening / is unverified
    invite_code: str = ""                # invite attribution, when the source can provide it
    inviter_id: str = ""
    left_at: Optional[float] = None      # departure time, when the source keeps departed members
    account_created_at: float = 0.0
    flags: List[str] = field(default_factory=list)   # platform- or operator-supplied markers
    link_key: str = ""                   # shared by accounts known to belong to one person, where identifiable

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MemberRecord":
        known = {"id", "is_bot", "joined_at", "pending", "invite_code", "inviter_id", "left_at",
                 "account_created_at", "flags", "link_key"}
        d = {k: v for k, v in d.items() if k in known}
        d["id"] = str(d.get("id", ""))
        d["flags"] = [str(f) for f in d.get("flags") or []]
        return cls(**d)


@runtime_checkable
class MembershipSource(Protocol):
    """Permitted evidence about membership. ``live`` is True only for a source that reads the real platform.
    Optional methods (used for secondary measurements when present): ``participation() -> {member id: number of
    voluntary interactions}``, ``feedback() -> [{"member_id", "consented"}]``, ``resource_usage() -> {name: uses}``."""
    name: str
    live: bool

    def list_members(self) -> List[MemberRecord]: ...


@runtime_checkable
class MetricSource(Protocol):
    name: str
    live: bool

    def read(self) -> List[Quantity]: ...


class JsonlMembershipSource:
    """Members from a JSONL snapshot file kept by a platform integration or a staging service (one record per
    line). An optional ``<path>.activity.json`` holds ``{"participation": {...}, "feedback": [...],
    "resource_usage": {...}}``. A file is never live evidence."""
    live = False

    def __init__(self, path: str, name: str = ""):
        self.path, self.name = path, name or f"jsonl:{os.path.basename(path)}"

    def list_members(self) -> List[MemberRecord]:
        return [MemberRecord.from_dict(d) for d in iter_jsonl(self.path)]

    def _activity(self) -> Dict[str, Any]:
        return read_json(self.path + ".activity.json", {}) or {}

    def participation(self) -> Dict[str, int]:
        return {str(k): int(v) for k, v in (self._activity().get("participation") or {}).items()}

    def feedback(self) -> List[Dict[str, Any]]:
        return list(self._activity().get("feedback") or [])

    def resource_usage(self) -> Dict[str, float]:
        return {str(k): float(v) for k, v in (self._activity().get("resource_usage") or {}).items()}


class JsonMetricSource:
    """Metrics from a JSON file ``{name: value}`` written by a staging service. Never live evidence."""
    live = False

    def __init__(self, path: str, name: str = ""):
        self.path, self.name = path, name or f"json:{os.path.basename(path)}"

    def read(self) -> List[Quantity]:
        return [Quantity(str(k), float(v)) for k, v in (read_json(self.path, {}) or {}).items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)]


def source_from_spec(spec: str, task: TaskSpec, run_dir: str) -> Any:
    """``jsonl:<path>`` | ``json:<path>`` | ``<module>:<callable>`` (called as ``factory(task, run_dir)``)."""
    kind, _, rest = spec.partition(":")
    if kind == "jsonl":
        return JsonlMembershipSource(rest)
    if kind == "json":
        return JsonMetricSource(rest)
    fn = getattr(importlib.import_module(kind), rest)
    return fn(task, run_dir)


# --------------------------------------------------------------------------------------------------------------
# signing key and hash-chained log
# --------------------------------------------------------------------------------------------------------------
def load_or_create_key(path: str) -> bytes:
    """The signing key. Created 0600; refused if anyone but its owner can read it or another uid owns it."""
    d = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(d):
        os.makedirs(d, mode=0o700, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        st = os.stat(path)
        if stat.S_IMODE(st.st_mode) & 0o077:
            raise PermissionError(f"verifier key {path} is readable by other users (mode "
                                  f"{stat.S_IMODE(st.st_mode):o}); it must be 0600")
        if st.st_uid != os.geteuid():
            raise PermissionError(f"verifier key {path} is owned by uid {st.st_uid}, not by this process "
                                  f"(uid {os.geteuid()})")
        with open(path, "rb") as f:
            key = f.read()
        if len(key) < 32:
            raise ValueError(f"verifier key {path} is too short")
        return key
    key = os.urandom(32)
    with os.fdopen(fd, "wb") as f:
        f.write(key)
        f.flush()
        os.fsync(f.fileno())
    return key


def _canon(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":")).encode()


class HashChainLog:
    """Append-only log where every entry commits to its predecessor. The head is also kept in ``head_path`` (in
    the verifier's private directory), so replacing the whole file is detectable as well as editing it."""
    GENESIS = "0" * 64

    def __init__(self, path: str, head_path: str = ""):
        self.path, self.head_path = path, head_path
        self._lock = threading.Lock()
        self.seq, self.head = -1, self.GENESIS
        for e in iter_jsonl(path):
            self.seq, self.head = int(e["seq"]), e["hash"]
        if head_path:
            kept = read_json(head_path)
            if kept and (kept.get("seq"), kept.get("hash")) != (self.seq, self.head):
                raise RuntimeError(f"verifier log {path} does not match its recorded head (seq {kept.get('seq')}); "
                                   "the log was altered or replaced")

    @staticmethod
    def entry_hash(prev: str, body: Dict[str, Any]) -> str:
        return hashlib.sha256(prev.encode() + _canon(body)).hexdigest()

    def append(self, ts: float, kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            body = {"seq": self.seq + 1, "ts": float(ts), "kind": kind, "payload": payload}
            entry = {**body, "prev": self.head, "hash": self.entry_hash(self.head, body)}
            new = not os.path.exists(self.path)
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            with os.fdopen(fd, "a") as f:
                f.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
                f.flush()
                os.fsync(f.fileno())
            if new:
                os.chmod(self.path, 0o644)
            self.seq, self.head = entry["seq"], entry["hash"]
            if self.head_path:
                atomic_write_json(self.head_path, {"seq": self.seq, "hash": self.head}, 0o600)
            return entry

    @classmethod
    def verify_chain(cls, path: str, expected_head: Optional[Dict[str, Any]] = None) -> Tuple[bool, int, str]:
        """(intact, number of entries, description of the first problem)."""
        prev, n = cls.GENESIS, 0
        for e in iter_jsonl(path):
            body = {k: e.get(k) for k in ("seq", "ts", "kind", "payload")}
            if e.get("prev") != prev or e.get("seq") != n or e.get("hash") != cls.entry_hash(prev, body):
                return False, n, f"entry {n} does not continue the chain"
            prev, n = e["hash"], n + 1
        if expected_head and (expected_head.get("seq"), expected_head.get("hash")) != (n - 1, prev):
            return False, n, "the log does not end at the recorded head"
        return True, n, ""


# --------------------------------------------------------------------------------------------------------------
# base: signing, logging, deadlines
# --------------------------------------------------------------------------------------------------------------
class SignedVerifier:
    """Signing, signature checks, the evaluation log and first-evaluation times shared by the verifiers."""

    def __init__(self, clock: Clock, state_dir: str, *, log_path: str = "", pinned_task: Optional[TaskSpec] = None):
        self.clock = clock
        self.state_dir = os.path.abspath(state_dir)
        os.makedirs(self.state_dir, mode=0o700, exist_ok=True)
        self._key = load_or_create_key(os.path.join(self.state_dir, "verifier.key"))
        self.log = HashChainLog(log_path or os.path.join(self.state_dir, RunFiles.VERIFIER),
                                os.path.join(self.state_dir, "head.json"))
        self.pinned_task = pinned_task
        self._first_seen: Dict[str, float] = {}
        for e in iter_jsonl(self.log.path):
            if e.get("kind") == "first_seen":
                self._first_seen[e["payload"]["objective_id"]] = float(e["payload"]["ts"])

    # ---- signatures ----------------------------------------------------------------------------------------
    def _signature(self, outcome: VerifiedOutcome) -> str:
        d = outcome.to_dict()
        d.pop("signature", None)
        return hmac.new(self._key, _canon(d), hashlib.sha256).hexdigest()

    def check_signature(self, outcome: VerifiedOutcome) -> bool:
        return bool(getattr(outcome, "signature", "")) and hmac.compare_digest(outcome.signature,
                                                                              self._signature(outcome))

    # ---- helpers -------------------------------------------------------------------------------------------
    def _task(self, task: TaskSpec) -> TaskSpec:
        """The task the verifier judges. When the operator pinned one, the caller's copy is only an identifier:
        success criteria never come from the learner's side."""
        if self.pinned_task is None:
            return task
        if task.objective_id != self.pinned_task.objective_id:
            raise PermissionError(f"this verifier serves objective {self.pinned_task.objective_id!r} only")
        return self.pinned_task

    def _started(self, task: TaskSpec, now: float) -> float:
        t = task.measurement.get("window_start_ts")
        if isinstance(t, (int, float)) and t > 0:
            return float(t)
        if task.objective_id not in self._first_seen:
            self._first_seen[task.objective_id] = now
            self.log.append(now, "first_seen", {"objective_id": task.objective_id, "ts": now})
        return self._first_seen[task.objective_id]

    def _deadline(self, task: TaskSpec, now: float) -> Optional[float]:
        t = task.measurement.get("deadline_ts")
        if isinstance(t, (int, float)) and t > 0:
            return float(t)
        start = self._started(task, now)
        return start + task.limits.duration_s if task.limits.duration_s > 0 else None

    def _emit(self, outcome: VerifiedOutcome, detail: Dict[str, Any]) -> VerifiedOutcome:
        outcome.evidence_refs = list(outcome.evidence_refs) + [f"verifier-log:{self.log.seq + 1}"]
        outcome.signature = self._signature(outcome)
        self.log.append(outcome.evaluated_at, "verify", {"outcome": outcome.to_dict(), "detail": detail})
        return outcome

    def _measure(self, queries: Sequence[OutcomeQuery], values: Callable[[], Dict[str, float]],
                 source_name: str) -> List[OutcomeObservation]:
        now, out, vals = self.clock.now(), [], None
        for q in queries:
            if now < q.due_at:
                out.append(OutcomeObservation(q.query_id, OutcomeStatus.PENDING, source="verifier"))
                continue
            vals = values() if vals is None else vals
            if q.metric not in vals:
                out.append(OutcomeObservation(q.query_id, OutcomeStatus.UNOBSERVABLE, observed_at=now,
                                              source="verifier"))
                continue
            out.append(OutcomeObservation(q.query_id, OutcomeStatus.OBSERVED, float(vals[q.metric]), observed_at=now,
                                          verified=True, source=f"verifier:{source_name}",
                                          evidence_refs=[f"verifier-log:{self.log.seq + 1}"],
                                          dedup_key=f"{q.metric}@{q.due_at:.0f}"))
        if any(o.status == OutcomeStatus.OBSERVED for o in out):
            self.log.append(now, "measure", {"observations": [o.to_dict() for o in out
                                                              if o.status == OutcomeStatus.OBSERVED]})
        return out


# --------------------------------------------------------------------------------------------------------------
# membership
# --------------------------------------------------------------------------------------------------------------
DEFAULT_RULES: Dict[str, Any] = {
    "exclude_bots": True,
    "test_account_ids": [],              # operator-declared test accounts (including the operator's own)
    "count_departed": False,             # a member who left is not a current member
    "exclude_flags": ["purchased", "artificial", "fabricated_activity", "prohibited_engagement",
                      "added_without_consent", "duplicate"],
    "uncertain_flags": ["automation_suspected", "authenticity_unknown"],
    "pending_screening": "uncertain",    # "uncertain" | "exclude" | "count"
    "suspect_invite_codes": [],          # operator-flagged indicators of purchased membership
    "suspect_member_ids": [],
    "min_account_age_s": 0.0,            # accounts younger than this at joining cannot be told from automated ones
    "uncertain_p_range": [0.0, 1.0],     # credible range for the share of uncertain accounts that are legitimate
    "confidence": 0.95,
    "retention_window_s": 7 * DAY,
}


def binom_quantile(p_level: float, n: int, p: float) -> int:
    """Smallest k with P(Binomial(n, p) <= k) >= p_level (exact, by summing the pmf in log space)."""
    if n <= 0 or p <= 0.0:
        return 0
    if p >= 1.0:
        return n
    cum, lq, lp = 0.0, math.log1p(-p), math.log(p)
    for k in range(n + 1):
        cum += math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * lp + (n - k) * lq)
        if cum >= p_level - 1e-12:
            return k
    return n


def classify_members(members: Sequence[MemberRecord], rules: Dict[str, Any], now: float) -> Dict[str, Any]:
    """Apply the exclusion rules. Returns counted / excluded / uncertain member lists with reasons."""
    tests = {str(x) for x in rules["test_account_ids"]}
    excl_flags, unc_flags = list(rules["exclude_flags"]), list(rules["uncertain_flags"])
    suspect_inv = {str(x) for x in rules["suspect_invite_codes"]}
    suspect_ids = {str(x) for x in rules["suspect_member_ids"]}
    counted: List[MemberRecord] = []
    excluded: Dict[str, List[MemberRecord]] = {}
    uncertain: Dict[str, List[MemberRecord]] = {}
    seen_ids: set = set()
    seen_links: set = set()

    def out(bucket: Dict[str, List[MemberRecord]], reason: str, m: MemberRecord) -> None:
        bucket.setdefault(reason, []).append(m)

    for m in sorted(members, key=lambda x: (x.joined_at, x.id)):
        if m.id in seen_ids:
            out(excluded, "duplicates", m)
            continue
        seen_ids.add(m.id)
        if rules["exclude_bots"] and m.is_bot:
            out(excluded, "bots", m)
        elif m.id in tests:
            out(excluded, "test_accounts", m)
        elif any(f in excl_flags for f in m.flags):
            out(excluded, next(f for f in m.flags if f in excl_flags), m)
        elif m.left_at is not None and m.left_at <= now and not rules["count_departed"]:
            out(excluded, "departed", m)
        elif m.link_key and m.link_key in seen_links:
            out(excluded, "duplicate_accounts", m)
        else:
            if m.link_key:
                seen_links.add(m.link_key)
            if m.pending and rules["pending_screening"] == "exclude":
                out(excluded, "pending_screening", m)
            elif m.pending and rules["pending_screening"] == "uncertain":
                out(uncertain, "pending_screening", m)
            elif any(f in unc_flags for f in m.flags):
                out(uncertain, next(f for f in m.flags if f in unc_flags), m)
            elif m.id in suspect_ids or (m.invite_code and m.invite_code in suspect_inv):
                out(uncertain, "purchase_indicator", m)
            elif (rules["min_account_age_s"] > 0 and m.account_created_at > 0
                  and m.joined_at - m.account_created_at < rules["min_account_age_s"]):
                out(uncertain, "new_account", m)
            else:
                counted.append(m)
    return {"counted": counted, "excluded": excluded, "uncertain": uncertain}


class MembershipVerifier(SignedVerifier):
    """Implements ``contracts.OutcomeVerifier`` for objectives of the form "N legitimate voluntary human members"."""

    def __init__(self, source: MembershipSource, clock: Clock, state_dir: str, *, log_path: str = "",
                 pinned_task: Optional[TaskSpec] = None,
                 cost_source: Optional[Callable[[], Dict[str, float]]] = None):
        super().__init__(clock, state_dir, log_path=log_path, pinned_task=pinned_task)
        self.source, self.cost_source = source, cost_source

    def _rules(self, task: TaskSpec) -> Dict[str, Any]:
        rules = dict(DEFAULT_RULES)
        rules.update({k: v for k, v in (task.success.get("exclusions") or {}).items() if k in DEFAULT_RULES
                      and v is not None and not (isinstance(v, str) and v.upper().startswith("REQUIRED"))})
        if isinstance(task.success.get("confidence"), (int, float)):
            rules["confidence"] = float(task.success["confidence"])
        return rules

    def _evaluate(self, task: TaskSpec) -> Dict[str, Any]:
        now = self.clock.now()
        rules = self._rules(task)
        members = list(self.source.list_members())
        c = classify_members(members, rules, now)
        n_c = len(c["counted"])
        n_u = sum(len(v) for v in c["uncertain"].values())
        alpha = 1.0 - float(rules["confidence"])
        p_lo, p_hi = (float(x) for x in rules["uncertain_p_range"])
        lower = n_c + binom_quantile(alpha, n_u, p_lo)
        upper = n_c + binom_quantile(1.0 - alpha, n_u, p_hi)
        digest = hashlib.sha256()
        for m in sorted(members, key=lambda x: x.id):
            digest.update(hmac.new(self._key, m.id.encode(), hashlib.sha256).digest())
        return {"now": now, "rules": rules, "members": members, "classes": c, "confirmed": n_c, "uncertain": n_u,
                "lower": lower, "upper": upper, "snapshot": digest.hexdigest()}

    def _secondary(self, task: TaskSpec, ev: Dict[str, Any]) -> List[Quantity]:
        now, rules, c = ev["now"], ev["rules"], ev["classes"]
        out: List[Quantity] = []
        legit = list(c["counted"]) + list(c["excluded"].get("departed", []))
        win = float(rules["retention_window_s"])
        cohort = [m for m in legit if m.joined_at > 0 and m.joined_at <= now - win]
        if cohort:
            kept = sum(1 for m in cohort if m.left_at is None or m.left_at - m.joined_at >= win)
            out.append(Quantity("secondary.retention_7d", kept / len(cohort), "fraction", OutcomeKind.REAL,
                                float(len(cohort))))
        ids = {m.id for m in c["counted"]}
        part = getattr(self.source, "participation", None)
        if callable(part):
            counts = {k: v for k, v in (part() or {}).items() if k in ids and v > 0}
            out.append(Quantity("secondary.participation.members", float(len(counts)), "members", OutcomeKind.COUNT,
                                float(max(len(ids), 1))))
            out.append(Quantity("secondary.participation.interactions", float(sum(counts.values())), "interactions",
                                OutcomeKind.COUNT, float(max(len(ids), 1))))
        fb = getattr(self.source, "feedback", None)
        if callable(fb):
            n = sum(1 for f in (fb() or []) if f.get("consented") is True and str(f.get("member_id", "")) in ids)
            out.append(Quantity("secondary.feedback.consented", float(n), "responses", OutcomeKind.COUNT))
        use = getattr(self.source, "resource_usage", None)
        if callable(use):
            for name, v in sorted((use() or {}).items()):
                out.append(Quantity(f"secondary.resource_use.{name}", float(v), "uses", OutcomeKind.COUNT))
        if self.cost_source is not None:
            cost = self.cost_source() or {}
            if "money" in cost:
                out.append(Quantity("secondary.operating_cost", float(cost["money"]), str(cost.get("currency", ""))))
            if "compute_s" in cost:
                out.append(Quantity("secondary.compute_s", float(cost["compute_s"]), "s"))
        out.append(Quantity("secondary.time_elapsed_s", max(0.0, now - self._started(task, now)), "s"))
        return out

    def verify(self, task: TaskSpec) -> VerifiedOutcome:
        task = self._task(task)
        ev = self._evaluate(task)
        now, c = ev["now"], ev["classes"]
        target = float(task.success.get("target", 0))
        deadline = self._deadline(task, now)
        if target > 0 and ev["lower"] >= target:
            status = GoalStatus.ACHIEVED
        elif deadline is not None and now >= deadline and ev["upper"] < target:
            status = GoalStatus.FAILED
        else:
            status = GoalStatus.PENDING
        unc = {k: len(v) for k, v in c["uncertain"].items()}
        notes = (f"{ev['confirmed']} members confirmed; {ev['uncertain']} could not be established as legitimate "
                 f"({', '.join(f'{k} {v}' for k, v in sorted(unc.items())) or 'none'}); "
                 f"{ev['rules']['confidence']:.0%} interval for legitimate members: {ev['lower']}–{ev['upper']}.")
        if status == GoalStatus.PENDING and ev["confirmed"] + ev["uncertain"] >= target > ev["lower"]:
            notes += " The target is not established with the required confidence; success is not claimed."
        if status == GoalStatus.PENDING and deadline is not None and now >= deadline:
            notes += " The measurement window has ended with the result undetermined; operator review is required."
        outcome = VerifiedOutcome(
            objective_id=task.objective_id, status=status, value=float(ev["confirmed"]), target=target,
            uncertainty=(ev["upper"] - ev["lower"]) / 2.0, evaluated_at=now,
            evidence_refs=[f"source:{self.source.name}", f"snapshot:hmac-sha256:{ev['snapshot']}"],
            excluded={k: len(v) for k, v in sorted(c["excluded"].items())}, notes=notes,
            secondary=self._secondary(task, ev))
        return self._emit(outcome, {"lower": ev["lower"], "upper": ev["upper"], "uncertain": unc,
                                    "members_listed": len(ev["members"]), "source_live": bool(self.source.live),
                                    "rules": ev["rules"], "deadline": deadline})

    def _values(self, task: TaskSpec) -> Dict[str, float]:
        ev = self._evaluate(task)
        vals = {"membership.confirmed": float(ev["confirmed"]), "membership.uncertain": float(ev["uncertain"]),
                "membership.lower_bound": float(ev["lower"]), "membership.upper_bound": float(ev["upper"])}
        metric = task.success.get("metric")
        if isinstance(metric, str) and metric:
            vals[metric] = float(ev["confirmed"])
        vals.update({q.name: q.value for q in self._secondary(task, ev)})
        return vals

    def measure(self, queries: Sequence[OutcomeQuery]) -> List[OutcomeObservation]:
        if self.pinned_task is None:
            raise RuntimeError("measure() needs the operator's task: construct the verifier with pinned_task")
        return self._measure(queries, lambda: self._values(self.pinned_task), self.source.name)


# --------------------------------------------------------------------------------------------------------------
# metric threshold (a different family of objective, e.g. on the staging services)
# --------------------------------------------------------------------------------------------------------------
class MetricThresholdVerifier(SignedVerifier):
    """Objective: a directly measured metric reaches a target. ``success = {verifier: metric_threshold, metric,
    target}``; the metrics named in ``task.secondary_metrics`` are reported separately when the source has them."""

    def __init__(self, source: MetricSource, clock: Clock, state_dir: str, *, log_path: str = "",
                 pinned_task: Optional[TaskSpec] = None):
        super().__init__(clock, state_dir, log_path=log_path, pinned_task=pinned_task)
        self.source = source

    def verify(self, task: TaskSpec) -> VerifiedOutcome:
        task = self._task(task)
        now = self.clock.now()
        qs = {q.name: q for q in self.source.read()}
        metric, target = str(task.success.get("metric", "")), float(task.success.get("target", 0))
        deadline = self._deadline(task, now)
        present = metric in qs
        value = float(qs[metric].value) if present else 0.0
        if present and target > 0 and value >= target:
            status = GoalStatus.ACHIEVED
        elif deadline is not None and now >= deadline and present:
            status = GoalStatus.FAILED
        else:
            status = GoalStatus.PENDING
        notes = (f"{metric} = {value:g} read from {self.source.name}." if present else
                 f"{metric} is not available from {self.source.name}; nothing is claimed.")
        if not present and deadline is not None and now >= deadline:
            notes += " The measurement window has ended with the result undetermined; operator review is required."
        secondary = [qs[n] for n in task.secondary_metrics if n in qs and n != metric]
        secondary.append(Quantity("secondary.time_elapsed_s", max(0.0, now - self._started(task, now)), "s"))
        outcome = VerifiedOutcome(objective_id=task.objective_id, status=status, value=value, target=target,
                                  evaluated_at=now, evidence_refs=[f"source:{self.source.name}"], notes=notes,
                                  secondary=secondary)
        return self._emit(outcome, {"metric_present": present, "source_live": bool(self.source.live),
                                    "deadline": deadline})

    def measure(self, queries: Sequence[OutcomeQuery]) -> List[OutcomeObservation]:
        return self._measure(queries, lambda: {q.name: float(q.value) for q in self.source.read()}, self.source.name)


VERIFIERS = {"membership": MembershipVerifier, "metric_threshold": MetricThresholdVerifier}


def build_verifier(task: TaskSpec, state_dir: str, clock: Clock, source: Any, *, log_path: str = "",
                   pin: bool = True, cost_source: Optional[Callable[[], Dict[str, float]]] = None) -> SignedVerifier:
    """Verifier for a task, chosen by ``task.success['verifier']``."""
    kind = str(task.success.get("verifier", ""))
    if kind not in VERIFIERS:
        raise ValueError(f"task.success.verifier must be one of {sorted(VERIFIERS)}, got {kind!r}")
    kw: Dict[str, Any] = {"log_path": log_path, "pinned_task": task if pin else None}
    if kind == "membership":
        kw["cost_source"] = cost_source
    return VERIFIERS[kind](source, clock, state_dir, **kw)
