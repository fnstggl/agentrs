"""PolicyService — the independent authorization layer (implements ``contracts.PolicyEngine``).

It runs outside the learner. Nothing the learner sends can change its policy file, keys, limits, approvals or
audit log: a decision is a function of the typed action, the operator-pinned task, the policy file and the
service's own persistent state. It fails closed: anything malformed, unknown or internally inconsistent is denied.

Order of every ``authorize``:  stop flag -> policy valid -> input guard -> structure -> pinned task -> mode / live
authorization -> per-node rules (tools, interfaces, destinations incl. the local-address rule, side effects, data
use, platform rules, prohibited classes) -> rejected approval -> circumvention -> limits -> rate limits -> approvals
-> signed grant.

Accounting: an action is charged for every declared step at authorization (each step is one external action and
one hit of every rate rule it falls under; all of them must fit), and each external-effect primitive an executor
reports through ``check_primitive`` is charged as well -- a primitive that realises a declared step uses that
step's share, any other one is checked against the remaining budget and charged anew. So a program or a skill can
never make more external writes than the remaining budget, whatever the number of steps it carries.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import threading
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ... import contracts as C
from . import destinations as D
from . import liveauth
from . import localnet
from .approvals import APPROVED, PENDING, REJECTED, ApprovalQueue
from .audit import AuditLog, AuditVerification, verify_audit_log
from .evaluate import INTENT_PIDS, WRITE_ONLY_PIDS, Evaluator, Finding, NodeEval, TaskCtx, shingles, similarity
from .grants import GrantError, GrantSigner, load_or_create_key
from .limits import RESOURCE_PREFIX, LimitLedger, min_nonzero
from .schema import EFFECT_RANK, MODES, RATE_CAP_KEYS, Policy, PolicyError, RateRule, load_policy, run_policy_path
from .stop import StopSwitch
from .store import Store, j, uj
from .untrusted import contains_untrusted, detect_instructions, policy_context
from .vault import SUFFIX as SECRET_SUFFIX
from .vault import ExecutorSecrets, SecretBroker, SecretError

ALLOW, DENY, NEEDS = C.AuthVerdict.ALLOW, C.AuthVerdict.DENY, C.AuthVerdict.NEEDS_APPROVAL
REJECTED_SIMILARITY = 0.9
_BROWSER_MODALITIES = (C.Modality.STRUCTURED_BROWSER, C.Modality.VISUAL)
_SAFE_IDENT = re.compile(r"^[A-Za-z0-9_.\-]{1,100}$")


def _seg_glob(path: str, pattern: str) -> bool:
    """Glob match where ``*`` never crosses a ``/``."""
    rx = "".join("[^/]*" if c == "*" else re.escape(c) for c in pattern)
    return re.fullmatch(rx, path) is not None


def is_resource_denial(decision: C.AuthorizationDecision) -> bool:
    """True when a denial is about exhausted limits or rate (maps to ``FailureClass.RESOURCE_LIMIT``)."""
    return decision.verdict == DENY and any(p.startswith("L-") for p in decision.policy_ids)


class PolicyService:
    """See module docstring. Thread-safe; one instance per run directory per process."""

    def __init__(self, run_dir: str, policy_path: Optional[str] = None, clock: Optional[C.Clock] = None, *,
                 caller: Optional[Callable[[], str]] = None, fsync: bool = True):
        self.run_dir = os.path.abspath(run_dir)
        self.clock: C.Clock = clock or C.WallClock()
        self.caller = caller or (lambda: "inproc")
        self.auth_dir = os.path.join(self.run_dir, "auth")
        self.vault_dir = os.path.join(self.run_dir, "vault")
        self.keys_dir = os.path.join(self.auth_dir, "keys")
        for d in (self.auth_dir, self.keys_dir, self.vault_dir):
            os.makedirs(d, mode=0o700, exist_ok=True)
        self._lock = threading.RLock()
        self.stop = StopSwitch(self.auth_dir)
        self.store = Store(os.path.join(self.auth_dir, "state.sqlite"))
        self.vault = SecretBroker(self.vault_dir)
        self.signer = GrantSigner(load_or_create_key(os.path.join(self.keys_dir, "grant.key")))
        self.audit = AuditLog(os.path.join(self.run_dir, C.RunFiles.AUDIT),
                              key=load_or_create_key(os.path.join(self.keys_dir, "audit.key")),
                              head_path=os.path.join(self.auth_dir, "audit.head"), scrub=self.vault.scrub_obj,
                              fsync=fsync)
        self.approvals_q = ApprovalQueue(self.store, 86400.0)
        self.ledger = LimitLedger(self.store)
        self.policy_path = policy_path or run_policy_path()
        self.policy: Optional[Policy] = None
        self.policy_error = ""
        self._eval: Optional[Evaluator] = None
        self._load_policy(first=True)
        self._log("service_start", policy=self.policy.policy_id if self.policy else "", policy_sha256=self.policy.sha256
                  if self.policy else "", policy_path=self.policy_path, policy_error=self.policy_error,
                  stopped=self.stopped(), vault_errors=len(self.vault.load_errors))

    # ==========================================================================================================
    # plumbing
    # ==========================================================================================================
    def _log(self, event: str, **data: Any) -> None:
        self.audit.append(event, self.clock.now(), by=self.caller(), **data)

    def _load_policy(self, first: bool = False) -> None:
        try:
            pol = load_policy(self.policy_path, run_dir=self.run_dir)
        except PolicyError as e:
            self.policy, self._eval, self.policy_error = None, None, str(e)
            return
        self.policy, self.policy_error = pol, ""
        self.approvals_q.ttl_s = pol.approvals_ttl_s
        self._eval = Evaluator(pol, self.vault, sensitive_paths=[self.vault_dir, self.auth_dir + os.sep, SECRET_SUFFIX,
                                                                 os.path.join("auth", "keys"), "grant.key",
                                                                 "audit.key", "operator.key"])
        old = self.store.kv_get("policy_sha256")
        if old != pol.sha256:
            # the rules changed: earlier denials were made under other rules and are not carried over
            self.store.x("DELETE FROM denied_intents")
            self.store.kv_set("policy_sha256", pol.sha256)

    def _epoch(self) -> int:
        return self.store.kv_int("epoch", 0)

    def _stop_identity(self) -> Optional[str]:
        try:
            st = os.lstat(self.stop.stop_path)
        except FileNotFoundError:
            return None
        except OSError:
            return "unreadable"
        return f"{st.st_ino}:{st.st_mtime_ns}:{st.st_ctime_ns}:{st.st_size}"

    def _observe_stop(self) -> bool:
        """However the stop flag was set (API, ``touch <run_dir>/auth/STOP``, offline ``opctl stop``), the first
        time this service sees it engaged every outstanding grant is voided -- exactly what the API stop does.
        Called before every decision and check. Returns whether the stop is engaged."""
        ident = self._stop_identity()
        if ident is None:
            return False
        if self.store.kv_get("stop_voided_for") != ident:
            n = self._void_grants("emergency stop")
            self.store.kv_set("stop_voided_for", ident)
            self._log("emergency_stop_observed", grants_voided=n, flag=self.stop.engaged())
        return True

    def _decision(self, verdict: str, action_hash: str, findings: Sequence[Tuple[str, str]], *, grant: str = "",
                  expires_at: float = 0.0, approval_id: str = "") -> C.AuthorizationDecision:
        reasons, pids = [], []
        for pid, reason in findings:
            reasons.append(reason)
            if pid not in pids:
                pids.append(pid)
        return C.AuthorizationDecision(verdict=verdict, action_hash=action_hash, reasons=reasons, policy_ids=pids,
                                       grant=grant, expires_at=expires_at, approval_id=approval_id)

    @staticmethod
    def _safe_hash(action: Any) -> str:
        try:
            return action.content_hash() if isinstance(action, C.Record) else ""
        except Exception:  # noqa: BLE001 — an unhashable action is simply malformed
            return ""

    # ==========================================================================================================
    # tasks and live authorization
    # ==========================================================================================================
    def _validate_task(self, task: Any) -> List[str]:
        errs = []
        if not isinstance(task, C.TaskSpec):
            return ["task is not a TaskSpec"]
        if not isinstance(task.objective_id, str) or not task.objective_id or len(task.objective_id) > 200:
            errs.append("task.objective_id missing")
        if task.mode not in MODES:
            errs.append(f"unknown task mode {task.mode!r}")
        if not isinstance(task.limits, C.Limits):
            errs.append("task.limits is not a Limits record")
        else:
            for f in ("duration_s", "money", "compute_s", "max_external_actions", "max_decisions"):
                v = getattr(task.limits, f)
                if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0 or v != v:
                    errs.append(f"task.limits.{f} must be a number >= 0")
        if not isinstance(task.resources, dict):
            errs.append("task.resources must be a mapping")
        if not isinstance(task.interfaces, list) or any(not isinstance(x, str) for x in task.interfaces):
            errs.append("task.interfaces must be a list of strings")
        if not isinstance(task.data_permissions, dict) or any(not isinstance(v, bool) for v in task.data_permissions.values()):
            errs.append("task.data_permissions must map names to true/false")
        return errs

    def _pinned(self, task_id: str) -> Optional[Any]:
        return self.store.one("SELECT * FROM tasks WHERE task_id=?", (task_id,))

    def _live_record(self, task_id: str, task_hash: str, now: float) -> Tuple[Optional[Dict[str, Any]], str]:
        rows = self.store.q("SELECT record_json, revoked_at FROM live_records WHERE task_id=? ORDER BY registered_at DESC",
                            (task_id,))
        if not rows:
            return None, "no operator authorization record for this live task"
        why = "the operator authorization record for this task is revoked"
        for r in rows:
            if r["revoked_at"] is not None:
                continue
            rec = uj(r["record_json"], {})
            if rec.get("task_hash") != task_hash:
                why = "the operator authorization record names a different version of this task"
                continue
            return rec, ""
        return None, why

    def register_task(self, task: C.TaskSpec, by: str) -> Dict[str, Any]:
        """Operator: pin (or replace) the task with this ``objective_id``. Clears remembered denials for it."""
        with self._lock:
            errs = self._validate_task(task)
            if errs:
                raise ValueError("; ".join(errs))
            now, h = self.clock.now(), task.content_hash()
            row = self._pinned(task.objective_id)
            started = row["started_at"] if row is not None else now
            self.store.x("INSERT INTO tasks(task_id, task_hash, mode, task_json, pinned_at, pinned_by, started_at) "
                         "VALUES(?,?,?,?,?,?,?) ON CONFLICT(task_id) DO UPDATE SET task_hash=excluded.task_hash, "
                         "mode=excluded.mode, task_json=excluded.task_json, pinned_at=excluded.pinned_at, "
                         "pinned_by=excluded.pinned_by", (task.objective_id, h, task.mode, task.to_json(), now, str(by), started))
            self.store.x("DELETE FROM denied_intents WHERE task_id=?", (task.objective_id,))
            self.store.x("DELETE FROM kv WHERE k=?", (f"task_revoked:{task.objective_id}",))
            self._log("task_registered", task_id=task.objective_id, task_hash=h, mode=task.mode, operator=str(by))
            return {"task_id": task.objective_id, "task_hash": h, "mode": task.mode}

    def register_live_authorization(self, record: Dict[str, Any], by: str) -> Dict[str, Any]:
        """Operator: store a signed live authorization record after verifying its signature against
        ``<run_dir>/auth/keys/operator.pub``. Raises ``LiveAuthorizationError`` when it does not verify."""
        with self._lock:
            pub_path = os.path.join(self.keys_dir, "operator.pub")
            try:
                pub = liveauth.load_public_key(pub_path)
            except (OSError, ValueError) as e:
                self._log("live_authorization_rejected", reason="operator public key missing or unreadable")
                raise liveauth.LiveAuthorizationError(f"operator public key not installed at {pub_path}") from e
            try:
                rec = liveauth.verify_live_authorization(record, pub)
                for k in ("accounts", "owned", "read", "preauthorized", "interactive"):
                    D.parse_patterns(rec.get(k, []))
            except (liveauth.LiveAuthorizationError, D.DestinationError) as e:
                self._log("live_authorization_rejected", reason=str(e), record_id=str(record.get("record_id", ""))[:80]
                          if isinstance(record, dict) else "")
                raise liveauth.LiveAuthorizationError(str(e)) from e
            now = self.clock.now()
            self.store.x("INSERT OR REPLACE INTO live_records(record_id, task_id, record_json, registered_at, registered_by, "
                         "revoked_at) VALUES(?,?,?,?,?,NULL)", (rec["record_id"], rec["task_id"], j(rec), now, str(by)))
            self.store.x("DELETE FROM denied_intents WHERE task_id=?", (rec["task_id"],))
            self._log("live_authorization_registered", record_id=rec["record_id"], task_id=rec["task_id"],
                      task_hash=rec["task_hash"], not_before=rec["not_before"], not_after=rec["not_after"],
                      issued_by=rec["issued_by"], operator=str(by))
            return {"record_id": rec["record_id"], "task_id": rec["task_id"]}

    def _task_ctx(self, task: C.TaskSpec, now: float) -> Tuple[Optional[TaskCtx], List[Tuple[str, str]]]:
        """Resolve the caller's task against the pinned one and build the evaluation context."""
        errs = self._validate_task(task)
        if errs:
            return None, [("P-TASK-INVALID", "malformed task: " + "; ".join(errs))]
        pol = self.policy
        assert pol is not None
        tid, h = task.objective_id, task.content_hash()
        if self.store.kv_get(f"task_revoked:{tid}") is not None:
            return None, [("P-REVOKED", "access for this task has been revoked by the operator")]
        mode = pol.modes[task.mode]
        row = self._pinned(tid)
        record: Optional[Dict[str, Any]] = None
        if mode.requires_authorization_record:
            record, why = self._live_record(tid, h, now)
            if record is None:
                return None, [("P-LIVE-AUTH", f"live mode needs an explicit, signed operator authorization: {why}")]
        if row is None:
            self.store.x("INSERT INTO tasks(task_id, task_hash, mode, task_json, pinned_at, pinned_by, started_at) "
                         "VALUES(?,?,?,?,?,?,?)", (tid, h, task.mode, task.to_json(), now,
                                                  f"live-record:{record['record_id']}" if record else "first-use", now))
            self._log("task_pinned", task_id=tid, task_hash=h, mode=task.mode,
                      how="live-record" if record else "first-use")
            started = now
        elif row["task_hash"] != h:
            if record is not None:          # the operator signed this exact version: it replaces the pin
                self.store.x("UPDATE tasks SET task_hash=?, mode=?, task_json=?, pinned_at=?, pinned_by=? WHERE task_id=?",
                             (h, task.mode, task.to_json(), now, f"live-record:{record['record_id']}", tid))
                self._log("task_pinned", task_id=tid, task_hash=h, mode=task.mode, how="live-record")
                started = row["started_at"]
            else:
                return None, [("P-TASK-MISMATCH", "the task differs from the one pinned for this objective; only the "
                                                  "operator can replace a task")]
        else:
            started = row["started_at"]

        res = task.resources

        def pats(*keys: str, src: Optional[Dict[str, Any]] = None) -> List[D.Pattern]:
            out: List[D.Pattern] = []
            for k in keys:
                v = (src if src is not None else res).get(k, [])
                if isinstance(v, str):
                    v = [v]
                if not isinstance(v, list):
                    raise D.DestinationError(f"resources.{k} must be a list of destination patterns")
                out.extend(D.parse_patterns(v))
            return out

        try:
            deny = pats("deny", "denied_destinations")
            if record is not None:
                owned = pats("owned", "accounts", src=record)
                read = pats("read", src=record)
                pre = pats("preauthorized", src=record)
                inter = pats("interactive", src=record)
                envelope: Optional[List[D.Pattern]] = owned + read
                opted = {str(x) for x in record.get("opted_in_recipients", [])}
                budgets = record.get("budgets", {})
                window = liveauth.window(record)
                rate_req = record.get("rate_limits") or {}
            else:
                owned, read, pre, inter = list(mode.owned), list(mode.read), list(mode.preauthorized), []
                if mode.honour_task_resources:
                    owned += pats("owned", "owned_destinations", "accounts")
                    read += pats("read", "allowed", "allowed_destinations")
                    pre += pats("preauthorized")
                envelope = mode.url_envelope
                o = res.get("opted_in_recipients", []) if mode.honour_task_resources else []
                opted = {str(x) for x in o} if isinstance(o, list) else set()
                budgets, window = {}, None
                rate_req = (res.get("rate_limits") or {}) if mode.honour_task_resources else {}
        except D.DestinationError as e:
            return None, [("P-TASK-INVALID", f"malformed destination pattern in the task or record: {e}")]
        rates, rerr = self._rates(mode, rate_req)
        if rerr:
            return None, [("P-TASK-INVALID", rerr)]
        local = set(pol.local_services) | {(r["host"], int(r["port"])) for r in
                                           self.store.q("SELECT host, port FROM local_services")}

        ifaces = list(task.interfaces) or list(mode.default_interfaces)
        modalities = {i for i in ifaces if i in C.Modality.ALL}
        op_patterns = [i for i in ifaces if i not in C.Modality.ALL]
        caps, tl = mode.caps, task.limits
        limits = {
            "money": min_nonzero(tl.money, caps["money"], budgets.get("money")),
            "money_per_action": min_nonzero(pol.spending["per_action"], caps["money_per_action"], budgets.get("money_per_action")),
            "money_per_day": min_nonzero(pol.spending["per_day"], caps["money_per_day"], budgets.get("money_per_day")),
            "compute_s": min_nonzero(tl.compute_s, caps["compute_s"], budgets.get("compute_s")),
            "max_external_actions": min_nonzero(tl.max_external_actions, caps["max_external_actions"],
                                                budgets.get("max_external_actions")),
            "max_decisions": min_nonzero(tl.max_decisions, caps["max_decisions"], budgets.get("max_decisions")),
            "duration_s": min_nonzero(tl.duration_s, caps["duration_s"], budgets.get("duration_s")),
        }
        for pat in self._revoked_destinations():
            deny.append(pat)
        return TaskCtx(task=task, task_hash=h, mode=mode, started_at=started, owned=owned, read=read, deny=deny,
                       preauth=pre, interactive=inter, envelope=envelope, opted_in=opted, modalities=modalities,
                       op_patterns=op_patterns, limits=limits, strict_unset=mode.requires_authorization_record,
                       window=window, record_id=record["record_id"] if record else "",
                       file_base=os.path.join(self.run_dir, "sandbox"), local_services=local,
                       own_process=mode.own_process_services, rates=rates), []

    def _rates(self, mode: Any, req: Any) -> Tuple[Dict[str, Optional[RateRule]], str]:
        """Effective write rules: the policy's, or what the task (staging) / signed record (live) asks for, clamped
        to the mode's ``rate_caps`` (count at most the cap's, window at least the cap's). A mode without a cap for a
        rule keeps the policy's rule whatever is asked."""
        pol = self.policy
        assert pol is not None
        out: Dict[str, Optional[RateRule]] = {"write": pol.rate_write, "write_global": pol.rate_write_global}
        if not req:
            return out, ""
        if not isinstance(req, dict) or set(req) - set(RATE_CAP_KEYS):
            return out, f"rate_limits may only set {list(RATE_CAP_KEYS)}"
        for k, v in req.items():
            if not isinstance(v, dict) or set(v) - {"count", "window_s"}:
                return out, f"rate_limits.{k} must be {{count, window_s}}"
            c, w = v.get("count"), v.get("window_s")
            if isinstance(c, bool) or not isinstance(c, int) or c < 0 or isinstance(w, bool) or \
                    not isinstance(w, (int, float)) or not w > 0 or w != w:
                return out, f"rate_limits.{k}: count must be an integer >= 0 and window_s a number > 0"
            cap = mode.rate_caps.get(k)
            if cap is not None:
                out[k] = RateRule(count=min(int(c), cap.count), window_s=max(float(w), cap.window_s))
        return out, ""

    def _ctx_for_task_id(self, task_id: str, now: float) -> Tuple[Optional[TaskCtx], List[Tuple[str, str]]]:
        row = self._pinned(task_id)
        if row is None:
            return None, [("P-TASK-UNKNOWN", "no task is pinned under this id")]
        return self._task_ctx(C.TaskSpec.from_dict(uj(row["task_json"], {})), now)

    def _revoked_destinations(self) -> List[D.Pattern]:
        try:
            return D.parse_patterns(uj(self.store.kv_get("revoked_destinations"), []))
        except D.DestinationError:
            return [D.Pattern("any", "*")]          # unreadable revocation list: deny every destination

    # ==========================================================================================================
    # evaluation helpers (stateful parts)
    # ==========================================================================================================
    def _location(self, task_id: str, ctx: Dict[str, str]) -> str:
        r = self.store.one("SELECT v FROM session WHERE task_id=? AND k=?", (task_id, "location:" + ctx.get("tab_id", "")))
        if r is None:
            r = self.store.one("SELECT v FROM session WHERE task_id=? AND k=?", (task_id, "location:"))
        return r["v"] if r is not None else ctx.get("page_url", "")

    def _set_location(self, task_id: str, url: str, tab: str = "") -> None:
        try:
            D.normalise_url(url)
        except D.DestinationError:
            return
        for k in {"location:" + tab, "location:"}:
            self.store.x("INSERT INTO session(task_id, k, v) VALUES(?,?,?) ON CONFLICT(task_id, k) DO UPDATE SET v=excluded.v",
                         (task_id, k, url))

    def _reachable(self, d: D.Destination, tc: TaskCtx) -> bool:
        """Would the policy let this task read ``d`` at all?"""
        pol = self.policy
        assert pol is not None
        if D.match_any(d, pol.deny) or D.match_any(d, tc.deny):
            return False
        if self._eval is not None and self._eval.local_denial(d, tc):
            return False
        if d.kind == "url" and tc.envelope is not None and D.match_any(d, tc.envelope) is None:
            return False
        return D.match_any(d, tc.owned) is not None or D.match_any(d, tc.read) is not None

    def _circumvention(self, nodes: List[NodeEval], tc: TaskCtx) -> List[Finding]:
        """Is this a refused target coming back in another form?

        * a typed destination the rules refuse again -> the earlier refusal is cited;
        * a place the task was refused to *write* to, now approached with an operation that could write there
          (form input, a raw request, ...) and that the operator does not own -> denied;
        * an opaque payload (program text, skill arguments, raw request) that names a refused target -> denied,
          unless every URL of that host in the payload is one the task may read and the refusal was not about
          writing.
        """
        rows = self.store.q("SELECT target, effect, needles_json, operation FROM denied_intents WHERE task_id=?",
                            (tc.task_id,))
        if not rows:
            return []
        out: List[Finding] = []
        seen = set()
        write = EFFECT_RANK["external_write"]
        for ne in nodes:
            for r in rows:
                if r["target"] in seen:
                    continue
                hit: Optional[D.Destination] = None
                how = ""
                for d in ne.dests:
                    if d.intent_key != r["target"]:
                        continue
                    if any(f.dest is d for f in ne.deny):
                        hit, how = d, "targets it again"
                    elif (r["effect"] == "external_write" and EFFECT_RANK[ne.potential] >= write
                          and D.match_any(d, tc.owned) is None):
                        hit, how = d, "approaches it with another operation that can write"
                    if hit is not None:
                        break
                if hit is None and ne.blob and any(n and n in ne.blob for n in uj(r["needles_json"], [])):
                    same_host = [u for u in D.urls_in_text(ne.blob) if u.intent_key == r["target"]]
                    harmless = (r["effect"] != "external_write" and same_host
                                and all(self._reachable(u, tc) for u in same_host))
                    if not harmless:
                        how = "names it in its payload"
                if how:
                    seen.add(r["target"])
                    out.append(Finding("X-CIRCUMVENTION",
                                       f"an earlier attempt on {r['target']} ({r['effect']}, via {r['operation']}) was denied; "
                                       f"this action {how} — a denied action may not be re-issued through another "
                                       f"modality or in another form", hit))
        return out

    def _remember_denials(self, nodes: List[NodeEval], tc: TaskCtx, action_hash: str, now: float) -> None:
        """Remember which targets were refused, by normalised target and effect. A refusal to *write* somewhere
        is remembered as a write denial (reading it may still be fine); any other refusal covers every access."""
        for ne in nodes:
            for f in ne.deny:
                if f.dest is None or f.pid not in INTENT_PIDS:
                    continue
                eff = "external_write" if f.pid in WRITE_ONLY_PIDS else ("local" if f.pid == "P-FILE-ROOT" else "external_read")
                key = f.dest.intent_key
                old = self.store.one("SELECT id, effect FROM denied_intents WHERE task_id=? AND target=?", (tc.task_id, key))
                if old is not None:
                    if EFFECT_RANK[eff] < EFFECT_RANK.get(old["effect"], 0):
                        self.store.x("UPDATE denied_intents SET effect=? WHERE id=?", (eff, old["id"]))
                    continue
                self.store.x("INSERT INTO denied_intents(task_id, target, effect, needles_json, operation, modality, "
                             "policy_ids_json, action_hash, ts) VALUES(?,?,?,?,?,?,?,?,?)",
                             (tc.task_id, key, eff, j(f.dest.needles()), ne.node.operation, ne.node.modality, j([f.pid]),
                              action_hash, now))

    def _duplicates(self, nodes: List[NodeEval], tc: TaskCtx, now: float) -> List[Finding]:
        pr = self.policy.prohibited          # type: ignore[union-attr]
        out: List[Finding] = []
        writes: List[Tuple[str, List[int], D.Destination]] = []
        for ne in nodes:
            wd = [d for c, d in ne.effects if c in ("write", "input")]
            if wd and len(ne.content) >= pr["dup_min_chars"]:
                sh = shingles(ne.content)
                writes.extend((d.intent_key, sh, d) for d in wd)
        targets = {d.intent_key for ne in nodes for c, d in ne.effects if c == "write"}
        if len(targets) > pr["max_write_destinations"]:
            out.append(Finding("X-MASS-OUTREACH", f"one action writes to {len(targets)} destinations "
                                                  f"(at most {pr['max_write_destinations']})"))
        rejected = [uj(r["shingles_json"], []) for r in self.store.q(
            "SELECT shingles_json FROM content_log WHERE task_id=? AND kind='rejected'", (tc.task_id,))]
        for ne in nodes:
            if rejected and len(ne.content) >= 8 and (ne.effects or ne.tool is not None and ne.tool.opaque):
                sh = shingles(ne.content)
                if any(similarity(sh, r) >= REJECTED_SIMILARITY for r in rejected):
                    out.append(Finding("X-CIRCUMVENTION", "this content was rejected by the operator; re-submitting it "
                                                          "with cosmetic changes or through another tool is not permitted"))
                    break
        if not writes:
            return out
        hist = self.store.q("SELECT target, shingles_json FROM content_log WHERE task_id=? AND kind='out' AND ts > ?",
                            (tc.task_id, now - pr["dup_window_s"]))
        past = [(r["target"], uj(r["shingles_json"], [])) for r in hist]
        for target, sh, d in writes:
            same = {target}
            for t2, sh2 in past:
                if t2 != target and similarity(sh, sh2) >= pr["dup_similarity"]:
                    same.add(t2)
            for t2, sh2, _d2 in writes:
                if t2 != target and similarity(sh, sh2) >= pr["dup_similarity"]:
                    same.add(t2)
            if len(same) > pr["dup_max_destinations"]:
                out.append(Finding("X-MASS-OUTREACH-DUPLICATE",
                                   f"near-identical content would reach {len(same)} destinations "
                                   f"(at most {pr['dup_max_destinations']})", d))
                break
        return out

    def _rate_rules(self, cls: str, d: D.Destination, tc: TaskCtx, level: str, rate_key: str = "",
                    ) -> List[Tuple[str, RateRule]]:
        """Rate rules one effect falls under: (counter key, rule). ``level`` "action" (a declared step) or
        "primitive" (an executor-reported operation: write rules only; the per-step rule is for declared steps)."""
        pol = self.policy
        assert pol is not None
        write = cls == "write"
        rk = rate_key or d.rate_key
        eff = "external_write" if write else "external_read"
        specific = next((rr for rr in pol.rate_rules if rr.match is not None and rr.match.matches(d)
                         and (not rr.effects or eff in rr.effects)), None)
        out: List[Tuple[str, RateRule]] = []
        if specific is not None:
            if level == "action" or write:
                out.append((f"s:{tc.task_id}:{rk}", specific))
        else:
            if level == "action":
                out.append((f"a:{tc.task_id}:{rk}", pol.rate_default))
            if write and tc.rates.get("write") is not None:
                out.append((f"w:{tc.task_id}:{rk}", tc.rates["write"]))       # type: ignore[arg-type]
        if write and tc.rates.get("write_global") is not None:
            out.append((f"g:{tc.task_id}:*", tc.rates["write_global"]))       # type: ignore[arg-type]
        return out

    def _rate(self, nodes: List[NodeEval], tc: TaskCtx, now: float, level: str = "action",
              credits: Optional[Dict[str, float]] = None, alias: Optional[Dict[str, str]] = None,
              ) -> Tuple[List[Tuple[str, str]], Dict[str, int], Dict[str, int]]:
        """(violations, counter hits to record if the request goes ahead, prepaid credits it uses).

        Every node counts once for each rule it falls under, so N declared write steps to one destination are N
        writes. At primitive level a credit prepaid by the declared step the primitive realises is used first;
        ``alias`` maps a destination key to the rate key of the declared (logical) effect it realises."""
        pol = self.policy
        assert pol is not None
        need: Dict[str, int] = {}
        rules: Dict[str, Tuple[RateRule, str]] = {}
        for ne in nodes:
            seen = set()
            for cls, d in ne.effects:
                rk = (alias or {}).get(d.key, "")
                todo = self._rate_rules(cls, d, tc, level, rk)
                if level == "primitive":
                    todo.append((f"p:{tc.task_id}:{d.rate_key}", pol.rate_primitive))
                for key, rule in todo:
                    if key in seen:
                        continue
                    seen.add(key)
                    need[key] = need.get(key, 0) + 1
                    rules[key] = (rule, rk or d.rate_key)
        use: Dict[str, int] = {}
        for key in need:
            c = int((credits or {}).get("credit:" + key, 0))
            if c > 0:
                use[key] = min(c, need[key])
                need[key] -= use[key]
        viol: List[Tuple[str, str]] = []
        for key, n in need.items():
            if n <= 0:
                continue
            rule, rk = rules[key]
            have = self.ledger.rate_count(key, rule.window_s, now)
            if have + n > rule.count:
                retry = max(0.0, self.ledger.rate_oldest(key, rule.window_s, now) + rule.window_s - now)
                scope = "all destinations of this task" if key.startswith("g:") else rk
                what = "writes" if key[:2] in ("w:", "g:") else "operations"
                viol.append(("L-RATE", f"{RESOURCE_PREFIX}rate limit for {scope} reached ({rule.count} {what} per "
                                       f"{rule.window_s:g} s; {have} used, {n} more requested); retry in {retry:.0f} s"))
        return viol, {k: n for k, n in need.items() if n > 0}, use

    # -- per-grant meter (credits prepaid by declared steps, charges made by primitives) ------------------------
    def _meter(self, nonce: str) -> Dict[str, float]:
        return {r["k"]: float(r["v"]) for r in self.store.q("SELECT k, v FROM grant_meter WHERE nonce=?", (nonce,))}

    def _meter_put(self, nonce: str, values: Dict[str, float]) -> None:
        for k, v in values.items():
            self.store.x("INSERT INTO grant_meter(nonce, k, v) VALUES(?,?,?) ON CONFLICT(nonce, k) DO UPDATE SET v=excluded.v",
                         (nonce, k, float(v)))

    @staticmethod
    def _external(ne: NodeEval) -> bool:
        return EFFECT_RANK[ne.effect] >= EFFECT_RANK["external_read"] and bool(ne.effects or ne.dests)

    def _bind(self, ah: str, nodes: List[NodeEval], extra: str = "") -> str:
        """What an approval is bound to: the action's content hash and, for nodes whose destination is the page the
        browser is on, that page. The same click approved on one page is not approved on another."""
        locs = []
        for ne in nodes:
            if ne.location_used:
                try:
                    locs.append(D.normalise_url(ne.location_used).key)
                except D.DestinationError:
                    locs.append("invalid:" + ne.location_used[:200])
        if not locs and not extra:
            return ah
        return hashlib.sha256((ah + "|" + extra + "|" + json.dumps(sorted(set(locs)))).encode()).hexdigest()

    def _preview(self, action: C.ActionSpec, nodes: List[NodeEval], tc: TaskCtx, money: float,
                 extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        effects, content = [], []
        for ne in nodes:
            for cls, d in ne.effects:
                effects.append({"class": cls, "destination": d.key, "operation": ne.node.operation})
            if ne.content and any(c in ("write", "input") for c, _ in ne.effects):
                keep = 65536 if ne.tool is not None and ne.tool.kind == "publish" else 4000
                content.append({"operation": ne.node.operation,
                                "destinations": [d.key for c, d in ne.effects if c in ("write", "input")],
                                "text": ne.content[:keep], "content_chars": len(ne.content),
                                "content_truncated": len(ne.content) > keep})
        pv = {"render": action.render()[:6000], "operation": action.operation, "modality": action.modality,
              "level": action.level, "target": action.target[:500], "steps": len(nodes) - 1, "effects": effects[:64],
              "content": content[:16], "est_cost": action.est_cost.to_dict(), "money": money,
              "task_id": tc.task_id, "mode": tc.mode.name, "expected": action.expected[:500]}
        if extra:
            pv.update(extra)
        return self.vault.scrub_obj(pv)

    def _declared_effects(self, nodes: List[NodeEval]) -> List[Dict[str, Any]]:
        out, seen = [], set()
        for ne in nodes:
            for c, d in ne.effects:
                e = {"cls": c, "key": d.key, "scheme": d.scheme, "kind": d.kind, "platform": self._platform_name(d),
                     "ident": d.ident, "tool": ne.tool.id if ne.tool is not None else "", "method": ne.method,
                     "op": ne.node.operation, "rate_key": d.rate_key}
                k = j(e)
                if k not in seen:
                    seen.add(k)
                    out.append(e)
        return out[:512]

    def _platform_name(self, d: D.Destination) -> str:
        plat = self.policy.platform_of(d, None) if self.policy is not None else None
        return plat.name if plat is not None else ""

    @staticmethod
    def _money(action: C.ActionSpec, by_id: Dict[int, NodeEval]) -> float:
        def rec(a: C.ActionSpec) -> float:
            own = by_id[id(a)].money if id(a) in by_id else float(a.est_cost.money)
            return max(own, sum(rec(s) for s in a.steps))
        return rec(action)

    # ==========================================================================================================
    # PolicyEngine: authorize
    # ==========================================================================================================
    def authorize(self, action: C.ActionSpec, task: C.TaskSpec, context: Optional[Dict[str, Any]] = None) -> C.AuthorizationDecision:
        with self._lock:
            now = self.clock.now()
            ah = self._safe_hash(action)
            try:
                dec, info = self._authorize(action, task, context, now, ah)
            except Exception as e:  # noqa: BLE001 — fail closed on any internal error
                dec = self._decision(DENY, ah, [("P-INTERNAL-ERROR", f"authorization failed closed ({type(e).__name__})")])
                info = {"error": traceback.format_exc(limit=4)}
            self._log("authorize", action_hash=dec.action_hash, action_id=str(getattr(action, "action_id", ""))[:200],
                      operation=str(getattr(action, "operation", ""))[:128], modality=str(getattr(action, "modality", ""))[:32],
                      task_id=str(getattr(task, "objective_id", ""))[:200], verdict=dec.verdict, reasons=dec.reasons,
                      policy_ids=dec.policy_ids, approval_id=dec.approval_id, expires_at=dec.expires_at, **info)
            return dec

    def _authorize(self, action: Any, task: Any, context: Any, now: float, ah: str,
                   ) -> Tuple[C.AuthorizationDecision, Dict[str, Any]]:
        info: Dict[str, Any] = {}
        self._observe_stop()
        blocked = self.stop.blocked()
        if blocked:
            pid = "P-STOP" if self.stop.engaged() is not None else "P-PAUSE"
            return self._decision(DENY, ah, [(pid, blocked)]), info
        if self.policy is None or self._eval is None:
            return self._decision(DENY, ah, [("P-POLICY-INVALID", f"no valid policy is loaded: {self.policy_error}")]), info
        if contains_untrusted(action) or contains_untrusted(task):
            return self._decision(DENY, ah, [("P-UNTRUSTED-INPUT", "untrusted environment content was passed as a "
                                                                   "policy input")]), info
        ctx, dropped = policy_context(context)
        if dropped:
            info["dropped_context"] = dropped
        errs = self._eval.structure(action)
        if errs or not ah:
            return self._decision(DENY, ah, [("P-MALFORMED", e) for e in (errs or ["action cannot be hashed"])]), info
        tc, terr = self._task_ctx(task, now)
        if tc is None:
            return self._decision(DENY, ah, terr), info

        flat = Evaluator.flatten(action)
        location = self._location(tc.task_id, ctx)
        nodes = [self._eval.node(n, tc, location) for n in flat]
        by_id = {id(ne.node): ne for ne in nodes}
        bind = self._bind(ah, nodes)
        prior = self.approvals_q.find(bind, now)
        if prior is not None and prior["status"] == REJECTED:
            return self._decision(DENY, ah, [("P-APPROVAL-REJECTED", f"the operator rejected this action "
                                                                     f"({prior['resolved_by']}: {prior['note']})")],
                                  approval_id=prior["approval_id"]), info
        info["effects"] = sorted({f"{c}:{d.key}" for ne in nodes for c, d in ne.effects})[:32]
        info["tools"] = sorted({ne.tool.id for ne in nodes if ne.tool is not None})

        deny: List[Finding] = [f for ne in nodes for f in ne.deny]
        circ = self._circumvention(nodes, tc)
        dups = self._duplicates(nodes, tc, now)
        deny_all = deny + circ + dups
        if deny_all:
            self._remember_denials(nodes, tc, ah, now)
            return self._decision(DENY, ah, [(f.pid, f.reason) for f in deny_all]), info

        n_ext = sum(1 for ne in nodes if self._external(ne))       # every declared external step is one action
        money = self._money(action, by_id)
        if n_ext > self.policy.max_external_primitives_per_action:
            return self._decision(DENY, ah, [("L-PRIMITIVES", f"{RESOURCE_PREFIX}the action declares {n_ext} external "
                                                              f"steps; at most {self.policy.max_external_primitives_per_action} "
                                                              f"per action")]), info
        want = {"money": money, "compute_s": float(action.est_cost.compute_s), "external_actions": float(n_ext),
                "decisions": 1.0}
        lim = self.ledger.check(tc.task_id, tc.limits, want, now, tc.started_at, strict_unset=tc.strict_unset,
                                window=tc.window)
        if lim:
            self._limit_event(tc.task_id, lim)
            return self._decision(DENY, ah, lim), info
        rate_viol, rate_hits, _ = self._rate(nodes, tc, now, "action")
        if rate_viol:
            return self._decision(DENY, ah, rate_viol), info
        info["external_steps"] = n_ext

        approve: List[Finding] = [f for ne in nodes for f in ne.approve]
        approval_id = ""
        if approve:
            entry = prior
            if entry is None or entry["status"] not in (PENDING, APPROVED):
                locs = sorted({ne.location_used for ne in nodes if ne.location_used})
                entry = self.approvals_q.request(bind_hash=bind, action_hash=ah, task_id=tc.task_id, kind="action",
                                                 preview=self._preview(action, nodes, tc, money,
                                                                       {"page": locs[0]} if locs else None),
                                                 action=action.to_dict(),
                                                 reasons=[f.reason for f in approve],
                                                 policy_ids=list(dict.fromkeys(f.pid for f in approve)), now=now)
                self._log("approval_requested", approval_id=entry["approval_id"], action_hash=ah, task_id=tc.task_id,
                          policy_ids=entry["policy_ids"])
            if entry["status"] == PENDING:
                return self._decision(NEEDS, ah, [(f.pid, f.reason) for f in approve], approval_id=entry["approval_id"]), info
            approval_id = entry["approval_id"]

        # allow: signed grant, reservation, rate and content bookkeeping -- one transaction
        pol = self.policy
        g = self.signer.issue(ah, tc.task_id, tc.task_hash, now, now + pol.grants["ttl_s"], self._epoch())
        effects = self._declared_effects(nodes)
        reserve = {"money": money, "compute_s": want["compute_s"], "external_actions": want["external_actions"]}
        nonce = g["payload"]["n"]
        with self.store.tx():
            self.store.x("INSERT INTO grants(nonce, action_hash, task_id, issued_at, expires_at, epoch, status, approval_id, "
                         "reserve_json, effects_json) VALUES(?,?,?,?,?,?,?,?,?,?)",
                         (nonce, ah, tc.task_id, now, g["payload"]["e"], g["payload"]["p"], "issued",
                          approval_id, j(reserve), j(effects)))
            self.ledger.add(tc.task_id, {"decisions": 1.0}, now)
            for k, n in rate_hits.items():
                for _ in range(n):
                    self.ledger.rate_hit(k, now)
            # what the declared steps prepaid; primitives that realise them use these first
            self._meter_put(nonce, dict({"ext_credit": float(n_ext), "ext_units": 0.0, "ext_relay": 0.0,
                                         "ext_prims": 0.0, "money_used": 0.0},
                                        **{"credit:" + k: float(n) for k, n in rate_hits.items()}))
            for ne in nodes:
                wd = [d for c, d in ne.effects if c in ("write", "input")]
                if wd and len(ne.content) >= pol.prohibited["dup_min_chars"]:
                    sh = j(shingles(ne.content))
                    for d in wd:
                        self.store.x("INSERT INTO content_log(task_id, kind, target, shingles_json, ts) VALUES(?,?,?,?,?)",
                                     (tc.task_id, "out", d.intent_key, sh, now))
        info["grant_nonce"] = g["payload"]["n"]
        self._gc(now)
        findings = [("P-ALLOW", "permitted by policy " + pol.policy_id)]
        if approval_id:
            findings.append(("P-APPROVED", f"approved by the operator ({approval_id})"))
        return self._decision(ALLOW, ah, findings, grant=g["token"], expires_at=g["payload"]["e"],
                              approval_id=approval_id), info

    def _gc(self, now: float) -> None:
        """Hourly: drop rate and content history that no rule can look at any more (decisions, grants and the
        audit log are kept)."""
        if now - float(self.store.kv_get("last_gc") or 0.0) < 3600.0:
            return
        pol = self.policy
        assert pol is not None
        rules = [pol.rate_default, pol.rate_write, pol.rate_primitive] + pol.rate_rules + \
            ([pol.rate_write_global] if pol.rate_write_global else []) + \
            [r for m in pol.modes.values() for r in m.rate_caps.values()]
        horizon = max(r.window_s for r in rules)
        self.ledger.rate_gc(now, horizon_s=horizon + 60.0)
        self.store.x("DELETE FROM content_log WHERE kind='out' AND ts < ?", (now - pol.prohibited["dup_window_s"] - 60.0,))
        self.store.kv_set("last_gc", str(now))

    def _limit_event(self, task_id: str, lim: Sequence[Tuple[str, str]]) -> None:
        for pid, reason in lim:
            k = f"limit_event:{task_id}:{pid}"
            if self.store.kv_get(k) is None:        # one audit entry per (task, limit) until usage changes state
                self.store.kv_set(k, "1")
                self._log("limit_event", task_id=task_id, policy_id=pid, reason=reason)

    # ==========================================================================================================
    # PolicyEngine: grants
    # ==========================================================================================================
    def check_grant(self, action: C.ActionSpec, decision: C.AuthorizationDecision, consume: bool = True) -> bool:
        """Verify a grant: signature, binding to this exact action, expiry, epoch (stop / revocation), task pin and
        single use. With ``consume=True`` (the executor, immediately before execution) the grant is used up and
        the action's execution window opens; a second call returns False. ``consume=False`` only looks."""
        with self._lock:
            now = self.clock.now()
            ah = self._safe_hash(action)
            ok, why, nonce = False, "", ""
            try:
                ok, why, nonce = self._check_grant(action, decision, consume, now, ah)
            except Exception as e:  # noqa: BLE001 — fail closed
                ok, why = False, f"internal error ({type(e).__name__})"
            if ok and consume:
                self._log("grant_use", action_hash=ah, nonce=nonce)
            elif not ok:
                self._log("grant_reject", action_hash=ah, nonce=nonce, reason=why, consume=bool(consume))
            return ok

    def _check_grant(self, action: Any, decision: Any, consume: bool, now: float, ah: str) -> Tuple[bool, str, str]:
        self._observe_stop()
        blocked = self.stop.blocked()
        if blocked:
            return False, blocked, ""
        if not isinstance(action, C.ActionSpec) or not isinstance(decision, C.AuthorizationDecision) or not ah:
            return False, "not an action / decision", ""
        if decision.verdict != ALLOW:
            return False, "decision is not an allow", ""
        if decision.action_hash != ah:
            return False, "decision is for a different action", ""
        try:
            p = self.signer.verify(decision.grant, ah, now)
        except GrantError as e:
            return False, str(e), ""
        nonce = p["n"]
        row = self.store.one("SELECT * FROM grants WHERE nonce=?", (nonce,))
        if row is None or row["action_hash"] != ah or row["task_id"] != p["t"]:
            return False, "grant was not issued by this service", nonce
        if row["epoch"] != self._epoch() or p["p"] != self._epoch():
            return False, "grant was revoked (stop or revocation since it was issued)", nonce
        if row["status"] != "issued":
            return False, f"grant already {row['status']} (single use)", nonce
        if now > row["expires_at"]:
            return False, "grant has expired", nonce
        pin = self._pinned(p["t"])
        if pin is None or pin["task_hash"] != p["th"] or self.store.kv_get(f"task_revoked:{p['t']}") is not None:
            return False, "the task changed or was revoked since the grant was issued", nonce
        if not consume:
            if row["approval_id"]:
                e = self.approvals_q.get(row["approval_id"])
                if e is None or e["status"] != APPROVED or e["consumed"]:
                    return False, "the approval behind this grant is no longer usable", nonce
            return True, "", nonce
        pol = self.policy
        window = pol.grants["exec_window_s"] if pol else 60.0
        slack = pol.grants["exec_slack_s"] if pol else 0.0
        exec_until = now + min(window, float(action.timeout_s) + slack)
        with self.store.tx():
            if row["approval_id"] and not self.approvals_q.consume(row["approval_id"], now):
                return False, "the approval behind this grant was already used or has expired", nonce
            cur = self.store.x("UPDATE grants SET status='used', used_at=?, exec_until=? WHERE nonce=? AND status='issued'",
                               (now, exec_until, nonce))
            if cur.rowcount != 1:
                return False, "grant already used (single use)", nonce
        return True, "", nonce

    def _active_grant(self, ah: str, now: float) -> Optional[Any]:
        return self.store.one("SELECT * FROM grants WHERE action_hash=? AND status='used' AND exec_until >= ? AND epoch=? "
                              "ORDER BY used_at DESC LIMIT 1", (ah, now, self._epoch()))

    # ==========================================================================================================
    # PolicyEngine: check_primitive
    # ==========================================================================================================
    def check_primitive(self, op: C.PrimitiveOp, action: C.ActionSpec) -> C.AuthorizationDecision:
        with self._lock:
            now = self.clock.now()
            ah = self._safe_hash(action)
            info: Dict[str, Any] = {}
            try:
                dec = self._check_primitive(op, action, now, ah, info)
            except Exception as e:  # noqa: BLE001 — fail closed
                dec = self._decision(DENY, ah, [("P-INTERNAL-ERROR", f"primitive check failed closed ({type(e).__name__})")])
                info["error"] = traceback.format_exc(limit=4)
            self._log("primitive", action_hash=ah, op=str(getattr(op, "op", ""))[:128],
                      modality=str(getattr(op, "modality", ""))[:32], verdict=dec.verdict, reasons=dec.reasons,
                      policy_ids=dec.policy_ids, approval_id=dec.approval_id, **info)
            return dec

    def _check_primitive(self, op: Any, action: Any, now: float, ah: str, info: Dict[str, Any]) -> C.AuthorizationDecision:
        self._observe_stop()
        blocked = self.stop.blocked()
        if blocked:
            return self._decision(DENY, ah, [("P-STOP" if self.stop.engaged() is not None else "P-PAUSE", blocked)])
        if self.policy is None or self._eval is None:
            return self._decision(DENY, ah, [("P-POLICY-INVALID", f"no valid policy is loaded: {self.policy_error}")])
        if not isinstance(op, C.PrimitiveOp) or not isinstance(action, C.ActionSpec) or not ah:
            return self._decision(DENY, ah, [("P-MALFORMED", "not a PrimitiveOp / ActionSpec")])
        if contains_untrusted(op) or contains_untrusted(action):
            return self._decision(DENY, ah, [("P-UNTRUSTED-INPUT", "untrusted environment content was passed as a "
                                                                   "policy input")])
        grant = self._active_grant(ah, now)
        if grant is None:
            return self._decision(DENY, ah, [("P-NO-ACTIVE-GRANT", "no granted execution of this action is in progress")])
        pol = self.policy
        if grant["n_primitives"] >= pol.max_primitives_per_action:
            return self._decision(DENY, ah, [("L-PRIMITIVES", f"{RESOURCE_PREFIX}more than "
                                                              f"{pol.max_primitives_per_action} primitive operations in one action")])
        args = op.arguments
        if not isinstance(args, dict):
            return self._decision(DENY, ah, [("P-MALFORMED", "primitive arguments must be a mapping")])
        target = args.get("target", "") if isinstance(args.get("target", ""), str) else ""
        node = C.ActionSpec(action_id=action.action_id, operation=op.op, modality=op.modality, target=target,
                            arguments=args, timeout_s=action.timeout_s)
        errs = self._eval.structure(node)
        if errs:
            return self._decision(DENY, ah, [("P-MALFORMED", e) for e in errs])
        tc, terr = self._ctx_for_task_id(grant["task_id"], now)
        if tc is None:
            return self._decision(DENY, ah, terr)
        tab = args.get("tab_id", "") if isinstance(args.get("tab_id", ""), str) else ""
        ne = self._eval.node(node, tc, self._location(tc.task_id, {"tab_id": tab}), primitive=True)
        self._peer_check(ne, args, tc)
        info["effects"] = sorted({f"{c}:{d.key}" for c, d in ne.effects})[:16]
        info["tool"] = ne.tool.id if ne.tool is not None else ""
        circ = self._circumvention([ne], tc)
        deny = ne.deny + circ
        if deny:
            self._remember_denials([ne], tc, ah, now)
            return self._decision(DENY, ah, [(f.pid, f.reason) for f in deny])

        nonce = grant["nonce"]
        granted = [e for e in uj(grant["effects_json"], []) if isinstance(e, dict)]
        meter = self._meter(nonce)
        reserve = uj(grant["reserve_json"], {})
        # which declared effect does each effect of this primitive realise? (rate accounting follows it)
        alias: Dict[str, str] = {}
        for cls, d in ne.effects:
            e = self._granted_effect(cls, d, ne, granted)
            if e is not None and e.get("kind") in ("account", "channel"):
                alias[d.key] = str(e.get("rate_key", ""))

        # -- limits: every external-effect primitive is charged (a declared step's share first) -----------------
        # A skill step announced before dispatch (PrimitiveOp.note "skill:...") is a step declared at run time: it is
        # charged like a declared step and prepays the one realisation that its executor reports next.
        declaration = isinstance(op.note, str) and op.note.startswith("skill:")
        external = self._external(ne) and (ne.tool is None or ne.tool.counts_as_action)
        bump: Dict[str, float] = {}
        put: Dict[str, float] = {}
        if external:
            if meter.get("ext_prims", 0.0) + 1 > pol.max_external_primitives_per_action:
                return self._decision(DENY, ah, [("L-PRIMITIVES", f"{RESOURCE_PREFIX}more than "
                                                                  f"{pol.max_external_primitives_per_action} external "
                                                                  f"operations in one action")])
            put["ext_prims"] = meter.get("ext_prims", 0.0) + 1
            if not declaration and meter.get("ext_relay", 0.0) >= 1:
                put["ext_relay"] = meter["ext_relay"] - 1          # realises a step declared at run time
            elif meter.get("ext_credit", 0.0) >= 1:
                put["ext_credit"] = meter["ext_credit"] - 1        # realises a step declared at authorization
                put["ext_units"] = meter.get("ext_units", 0.0) + 1
            else:
                bump["external_actions"] = 1.0                      # not declared: charged anew
                put["ext_units"] = meter.get("ext_units", 0.0) + 1
            if declaration:
                put["ext_relay"] = put.get("ext_relay", meter.get("ext_relay", 0.0)) + 1
        money_used = meter.get("money_used", 0.0) + ne.money
        if ne.money > 0:
            put["money_used"] = money_used
            extra = money_used - float(reserve.get("money", 0.0))
            if extra > 1e-12:
                bump["money"] = extra
            pa = float(tc.limits.get("money_per_action", 0.0))
            if pa > 0 and money_used > pa + 1e-9:
                return self._decision(DENY, ah, [("L-MONEY-ACTION", f"{RESOURCE_PREFIX}money per action would be "
                                                                    f"exceeded (this action has spent "
                                                                    f"{money_used - ne.money:g}, requested {ne.money:g}, "
                                                                    f"limit {pa:g})")])
        if bump:
            lim = self.ledger.check(tc.task_id, tc.limits, bump, now, tc.started_at, strict_unset=tc.strict_unset,
                                    window=tc.window)
            if lim:
                self._limit_event(tc.task_id, lim)
                return self._decision(DENY, ah, lim)
        viol, rate_hits, rate_use = self._rate([ne], tc, now, "action" if declaration else "primitive",
                                               credits=meter, alias=alias)
        if viol:
            return self._decision(DENY, ah, viol)
        relay_keys = sorted(k for k in set(rate_hits) | set(rate_use) if k[:2] in ("w:", "g:", "s:")) if declaration else []

        # -- approvals: covered only by the declared effect of the same class, operation and exact place -------------
        approval_id = ""
        if ne.approve:
            backed = bool(grant["approval_id"])
            open_f = [f for f in ne.approve if not self._covered(f, ne, granted, backed, action, reserve, money_used)]
            if open_f:
                bind = hashlib.sha256((self._bind(ah, [ne]) + "|" + json.dumps([op.op, op.modality, args], sort_keys=True,
                                                                               default=str)).encode()).hexdigest()
                entry = self.approvals_q.find(bind, now)
                if entry is not None and entry["status"] == REJECTED:
                    return self._decision(DENY, ah, [("P-APPROVAL-REJECTED", "the operator rejected this operation")],
                                          approval_id=entry["approval_id"])
                if entry is None or entry["status"] not in (PENDING, APPROVED):
                    entry = self.approvals_q.request(
                        bind_hash=bind, action_hash=ah, task_id=tc.task_id, kind="primitive",
                        preview=self._preview(node, [ne], tc, ne.money,
                                              dict({"parent_action": action.render()[:2000], "primitive": True},
                                                   **({"page": ne.location_used} if ne.location_used else {}))),
                        action=node.to_dict(), reasons=[f.reason for f in open_f],
                        policy_ids=list(dict.fromkeys(f.pid for f in open_f)), now=now)
                    self._log("approval_requested", approval_id=entry["approval_id"], action_hash=ah, task_id=tc.task_id,
                              policy_ids=entry["policy_ids"], primitive=op.op)
                if entry["status"] == PENDING:
                    return self._decision(NEEDS, ah, [(f.pid, f.reason) for f in open_f], approval_id=entry["approval_id"])
                if not self.approvals_q.consume(entry["approval_id"], now):
                    return self._decision(DENY, ah, [("P-APPROVAL-REJECTED", "the approval for this operation is no longer usable")])
                approval_id = entry["approval_id"]

        with self.store.tx():
            self.store.x("UPDATE grants SET n_primitives=n_primitives+1 WHERE nonce=?", (nonce,))
            for k, n in rate_hits.items():
                for _ in range(n):
                    self.ledger.rate_hit(k, now)
            for k, n in rate_use.items():
                put["credit:" + k] = meter.get("credit:" + k, 0.0) - n
            for k in relay_keys:                     # the declared step's realisation uses these
                put["credit:" + k] = put.get("credit:" + k, meter.get("credit:" + k, 0.0)) + 1
            if declaration and ne.effects:           # ... and is covered by what was checked here
                more = [e for e in self._declared_effects([ne]) if e not in granted]
                if more:
                    self.store.x("UPDATE grants SET effects_json=? WHERE nonce=?", (j((granted + more)[:512]), nonce))
            if bump:
                for k, v in bump.items():
                    reserve[k] = float(reserve.get(k, 0.0)) + v
                self.store.x("UPDATE grants SET reserve_json=? WHERE nonce=?", (j(reserve), nonce))
            self._meter_put(nonce, put)
            if ne.tool is not None and ne.tool.sets_location and node.modality in _BROWSER_MODALITIES:
                urls = [d for d in ne.dests if d.kind == "url"]
                if urls:
                    self._set_location(tc.task_id, urls[0].raw, tab)
        if bump or rate_hits:
            info["charged"] = dict(bump, rate=sorted(rate_hits))
        return self._decision(ALLOW, ah, [("P-ALLOW", "permitted by policy " + pol.policy_id)], approval_id=approval_id)

    def _peer_check(self, ne: NodeEval, args: Dict[str, Any], tc: TaskCtx) -> None:
        """An executor may report the address it actually connected to (``peer_ip`` / ``resolved_ip``): a public
        name that resolved to this machine is caught by the local-address rule as well (DNS rebinding)."""
        for k in ("peer_ip", "resolved_ip", "remote_ip"):
            v = args.get(k)
            if not isinstance(v, str) or not v:
                continue
            urls = [d for d in ne.dests if d.kind == "url"]
            port = urls[0].port if urls else 0
            try:
                host = D._canon_host(v)
            except D.DestinationError:
                ne.deny.append(Finding("P-LOCAL-ADDRESS", f"the executor reported an unusable peer address {v[:60]!r}"))
                continue
            scheme = urls[0].scheme if urls else "https"
            peer = D.Destination("url", f"url:{host}" + (f":{port}" if port else "") + "/", host=host, port=port,
                                 path="/", scheme=scheme, raw=v)
            lk = self._eval.local_denial(peer, tc) if self._eval is not None else ""
            if lk and not (urls and localnet.local_kind(urls[0].host)):
                ne.deny.append(Finding("P-LOCAL-ADDRESS", f"{urls[0].key if urls else 'the destination'} resolved to "
                                                          f"{host} ({lk}), an address of this machine or its network",
                                       peer))

    def _granted_effect(self, cls: str, d: D.Destination, ne: NodeEval, granted: List[Dict[str, Any]],
                        ) -> Optional[Dict[str, Any]]:
        """The declared effect of the granted action that this primitive effect realises, or None. Same class of
        effect (a write covers input as well), and either the same tool class, HTTP method and *exact* destination
        (full path, scheme) or -- for a logical destination (channel, account) -- the same operation on the same
        id, or a transport request the platform's ``transport`` table maps that operation and id to."""
        want = ("write",) if cls == "write" else ("write", "input") if cls == "input" else ("write", "input", "read")
        tool = ne.tool.id if ne.tool is not None else ""
        for e in granted:
            if e.get("cls") not in want:
                continue
            if e.get("kind") in ("account", "channel"):
                if d.kind == e["kind"] and d.key == e.get("key") and e.get("op") == ne.node.operation:
                    return e
                if d.kind == "url" and self._transport(e, d, ne):
                    return e
                continue
            if d.kind != e.get("kind") or d.key != e.get("key") or d.scheme != e.get("scheme", ""):
                continue
            if e.get("tool") != tool or e.get("method", "") != ne.method:
                continue
            return e
        return None

    def _transport(self, e: Dict[str, Any], d: D.Destination, ne: NodeEval) -> bool:
        pol = self.policy
        plat = next((p for p in (pol.platforms if pol else []) if p.name == e.get("platform")), None)
        ident = str(e.get("ident", ""))
        if plat is None or not plat.transport or not ne.method or not _SAFE_IDENT.match(ident):
            return False
        if D.match_any(d, plat.match) is None:
            return False
        for tr in plat.transport:
            if tr.method == ne.method and any(fnmatch.fnmatchcase(str(e.get("op", "")), o) for o in tr.ops) \
                    and _seg_glob(d.path, tr.path.replace("{id}", ident)):
                return True
        return False

    def _covered(self, f: Finding, ne: NodeEval, granted: List[Dict[str, Any]], backed: bool, action: C.ActionSpec,
                 reserve: Dict[str, Any], money_used: float) -> bool:
        """The grant already authorizes this finding. A destination-less finding (spending, code network) is covered
        only by an operator-approved grant of the same operation within the money it approved; a content finding only
        by an approved grant whose declared effect this primitive realises; a write / input finding by the declared
        effect this primitive realises."""
        if f.dest is None:
            return backed and ne.node.operation == action.operation and \
                money_used <= float(reserve.get("money", 0.0)) + 1e-9
        e = self._granted_effect("write" if f.cls == "write" else "input", f.dest, ne, granted)
        if e is None:
            return False
        return backed if f.cls == "content" else True

    # ==========================================================================================================
    # PolicyEngine: record (audit + limit accounting)
    # ==========================================================================================================
    def record(self, action: C.ActionSpec, result: C.ActionResult) -> None:
        with self._lock:
            now = self.clock.now()
            ah = self._safe_hash(action)
            if not isinstance(action, C.ActionSpec) or not isinstance(result, C.ActionResult):
                self._log("record_rejected", action_hash=ah, reason="not an ActionSpec / ActionResult")
                return
            row = self.store.one("SELECT * FROM grants WHERE action_hash=? AND status='used' ORDER BY used_at DESC LIMIT 1", (ah,))
            if row is None:
                self._log("record_without_grant", action_hash=ah, status=str(result.status)[:32])
                return
            reserve = uj(row["reserve_json"], {})
            executed = result.status not in (C.ResultStatus.DENIED, C.ResultStatus.NEEDS_APPROVAL)
            cost = result.cost if isinstance(result.cost, C.Cost) else C.Cost()

            def num(v: Any) -> float:
                return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and v >= 0 else 0.0

            seconds = num(cost.seconds) or max(0.0, num(result.ended) - num(result.started))
            meter = self._meter(row["nonce"])
            # external actions: every external step that was carried out or declared at run time (a partly executed
            # program included); an executed external action that reported none is charged one
            ext = meter.get("ext_units", 0.0)
            if executed and float(reserve.get("external_actions", 0.0)) > 0:
                ext = max(ext, 1.0)
            actual = {"money": max(num(cost.money) if executed else 0.0, meter.get("money_used", 0.0)),
                      "compute_s": num(cost.compute_s), "external_actions": ext,
                      "primitive_ops": float(max(len(result.ops) if isinstance(result.ops, list) else 0, int(num(cost.primitive_ops)))),
                      "seconds": seconds}
            with self.store.tx():
                self.ledger.add(row["task_id"], actual, now)
                self.store.x("UPDATE grants SET status='closed', closed_at=? WHERE nonce=?", (now, row["nonce"]))
                self.store.x("DELETE FROM kv WHERE k LIKE ?", (f"limit_event:{row['task_id']}:%",))
            loc, labels = "", {}
            for o in (result.observations if isinstance(result.observations, list) else []):
                if not isinstance(o, C.Observation):
                    continue
                if o.kind == "page":
                    cand = o.provenance.uri if isinstance(o.provenance, C.Provenance) else ""
                    if not cand and isinstance(o.data, dict) and isinstance(o.data.get("url"), str):
                        cand = o.data["url"]
                    loc = cand or loc
                for cat in detect_instructions(o.text if isinstance(o.text, str) else "")["categories"]:
                    labels[cat] = labels.get(cat, 0) + 1
            if loc and action.modality in _BROWSER_MODALITIES:
                tab = action.arguments.get("tab_id", "") if isinstance(action.arguments.get("tab_id", ""), str) else ""
                self._set_location(row["task_id"], loc, tab)
            over = actual["money"] > float(reserve.get("money", 0.0)) + 1e-9
            self._log("record", action_hash=ah, action_id=action.action_id[:200], task_id=row["task_id"],
                      status=str(result.status)[:32], failure_class=str(result.failure_class)[:40], cost=actual,
                      n_ops=len(result.ops) if isinstance(result.ops, list) else 0,
                      n_observations=len(result.observations) if isinstance(result.observations, list) else 0,
                      instruction_like_labels=labels, overspend=over)
            if over:
                self._log("limit_event", task_id=row["task_id"], policy_id="L-MONEY-ACTION",
                          reason=f"executor reported {actual['money']:g}, more than the {reserve.get('money', 0.0):g} authorized")
            tc, _ = self._ctx_for_task_id(row["task_id"], now)
            if tc is not None:
                st = self.ledger.state(tc.task_id, tc.limits, now, tc.started_at)
                if st["exhausted"]:
                    self._limit_event(tc.task_id, [("L-EXHAUSTED:" + n, f"{RESOURCE_PREFIX}{n} exhausted") for n in st["exhausted"]])

    # ==========================================================================================================
    # PolicyEngine: approvals
    # ==========================================================================================================
    def approvals(self) -> List[Dict[str, Any]]:
        with self._lock:
            return self.vault.scrub_obj(self.approvals_q.list(self.clock.now()))

    def resolve_approval(self, approval_id: str, approve: bool, by: str, note: str = "") -> C.AuthorizationDecision:
        """Operator only (enforced by the service's peer-uid check). On approval of an action the action is
        re-evaluated and, if nothing else stands in the way, a grant is returned; an identical later
        ``authorize`` call also succeeds. The approval is used up by the first execution."""
        with self._lock:
            now = self.clock.now()
            try:
                entry = self.approvals_q.resolve(str(approval_id), bool(approve), str(by), str(note), now)
            except (KeyError, ValueError) as e:
                self._log("approval_resolve_failed", approval_id=str(approval_id)[:64], reason=str(e))
                return self._decision(DENY, "", [("P-APPROVAL-INVALID", str(e).strip("'\""))], approval_id=str(approval_id)[:64])
            self._log("approval_resolved", approval_id=entry["approval_id"], action_hash=entry["action_hash"],
                      status=entry["status"], operator=str(by), note=str(note)[:500], kind=entry["kind"])
            ah = entry["action_hash"]
            if entry["status"] == REJECTED:
                for c in entry["preview"].get("content", []):
                    txt = c.get("text", "") if isinstance(c, dict) else ""
                    if len(txt) >= 8:
                        self.store.x("INSERT INTO content_log(task_id, kind, target, shingles_json, ts) VALUES(?,?,?,?,?)",
                                     (entry["task_id"], "rejected", "", j(shingles(txt)), now))
                return self._decision(DENY, ah, [("P-APPROVAL-REJECTED", f"rejected by {by}" + (f": {note}" if note else ""))],
                                      approval_id=entry["approval_id"])
            if entry["kind"] != "action":
                return self._decision(ALLOW, ah, [("P-APPROVED", f"approved by {by}; the executor may repeat the operation")],
                                      approval_id=entry["approval_id"])
            page = entry["preview"].get("page", "") if isinstance(entry.get("preview"), dict) else ""
            stored = self.approvals_q.stored_action(entry["approval_id"])
            row = self._pinned(entry["task_id"])
            try:
                action = C.ActionSpec.from_dict(stored or {})
                task = C.TaskSpec.from_dict(uj(row["task_json"], {})) if row is not None else None
            except Exception:  # noqa: BLE001
                action, task = None, None
            if action is None or task is None or action.content_hash() != ah:
                return self._decision(ALLOW, ah, [("P-APPROVED", f"approved by {by}; call authorize again with the identical action")],
                                      approval_id=entry["approval_id"])
            return self.authorize(action, task, {"page_url": page} if isinstance(page, str) and page else {})

    # ==========================================================================================================
    # PolicyEngine: stop, pause, revocation
    # ==========================================================================================================
    def emergency_stop(self, reason: str, by: str) -> None:
        with self._lock:
            now = self.clock.now()
            first = self.stop.engaged() is None
            self.stop.engage(str(reason), str(by), now)
            self._observe_stop()                 # voids every outstanding grant the first time this stop is seen
            self._log("emergency_stop", reason=str(reason)[:500], operator=str(by), first=first)

    def stopped(self) -> bool:
        with self._lock:
            return self._observe_stop()

    def reset_stop(self, by: str) -> bool:
        """Operator only. Grants issued before the stop stay void: the stop is observed (and grants voided) if it
        was not yet, and the reset itself voids every grant still outstanding -- none can have been issued while
        the stop was engaged, so all of them predate it."""
        with self._lock:
            self._observe_stop()
            was = self.stop.reset()
            n = self._void_grants("emergency stop reset") if was else 0
            self._log("emergency_stop_reset", operator=str(by), was_engaged=was, grants_voided=n)
            return was

    def pause(self, reason: str, by: str) -> None:
        with self._lock:
            self.stop.pause(str(reason), str(by), self.clock.now())
            self._log("pause", reason=str(reason)[:500], operator=str(by))

    def resume(self, by: str) -> bool:
        with self._lock:
            was = self.stop.resume()
            self._log("resume", operator=str(by), was_paused=was)
            return was

    def paused(self) -> bool:
        return self.stop.paused() is not None

    def _void_grants(self, why: str, task_id: Optional[str] = None) -> int:
        with self.store.tx():
            if task_id is None:
                self.store.kv_set("epoch", str(self._epoch() + 1))
                cur = self.store.x("UPDATE grants SET status='revoked' WHERE status IN ('issued', 'used')")
            else:
                cur = self.store.x("UPDATE grants SET status='revoked' WHERE status IN ('issued', 'used') AND task_id=?", (task_id,))
        return cur.rowcount

    def revoke(self, scope: str, by: str) -> None:
        """Revoke access. Scopes: ``all`` | ``grants`` | ``task:<objective_id>`` | ``live:<record_id>`` |
        ``secret:<name>`` | ``scope:<label>`` | ``destination:<pattern>``. Unknown scopes raise ``ValueError``."""
        with self._lock:
            now = self.clock.now()
            scope = str(scope)
            done: Dict[str, Any] = {}
            if scope == "all":
                done["grants"] = self._void_grants("revoke all")
                done["secrets"] = self.vault.revoke("all", now)
                self.store.x("UPDATE live_records SET revoked_at=? WHERE revoked_at IS NULL", (now,))
            elif scope == "grants":
                done["grants"] = self._void_grants("revoke grants")
            elif scope.startswith("task:") and scope[5:]:
                tid = scope[5:]
                done["grants"] = self._void_grants("revoke task", tid)
                self.store.x("UPDATE live_records SET revoked_at=? WHERE task_id=? AND revoked_at IS NULL", (now, tid))
                self.store.kv_set(f"task_revoked:{tid}", str(now))
            elif scope.startswith("live:") and scope[5:]:
                cur = self.store.x("UPDATE live_records SET revoked_at=? WHERE record_id=? AND revoked_at IS NULL", (now, scope[5:]))
                done["records"] = cur.rowcount
                done["grants"] = self._void_grants("revoke live record")
            elif scope.startswith("secret:") or scope.startswith("scope:"):
                done["secrets"] = self.vault.revoke(scope, now)
            elif scope.startswith("destination:") and scope[12:]:
                pat = scope[12:]
                D.parse_pattern(pat)                 # malformed -> DestinationError (a ValueError)
                cur_list = uj(self.store.kv_get("revoked_destinations"), [])
                if pat not in cur_list:
                    cur_list.append(pat)
                    self.store.kv_set("revoked_destinations", j(cur_list))
                done["secrets"] = self.vault.revoke(scope, now)
                done["grants"] = self._void_grants("revoke destination")
            else:
                self._log("revocation_failed", scope=scope[:200], operator=str(by))
                raise ValueError(f"unknown revocation scope {scope!r}")
            self.store.x("INSERT INTO revocations(scope, by, ts) VALUES(?,?,?)", (scope, str(by), now))
            self._log("revocation", scope=scope[:200], operator=str(by), **done)

    # ==========================================================================================================
    # local services (the only addresses of this machine an action may reach)
    # ==========================================================================================================
    @staticmethod
    def _local_address(address: str) -> Tuple[str, int, str]:
        pat = D.parse_pattern(str(address))
        if pat.kind != "url" or not pat.port or pat.path or pat.scheme or any(c in pat.host for c in "*?"):
            raise ValueError(f"expected an exact host:port, got {address!r}")
        kind = localnet.local_kind(pat.host)
        if not kind:
            raise ValueError(f"{address!r} is not an address of this machine or its network")
        return pat.host, pat.port, kind

    def register_local_service(self, address: str, by: str, note: str = "") -> Dict[str, Any]:
        """Operator only (enforced by the service's peer-uid check). Make ``host:port`` -- a staging service the
        harness started, a service the operator vouches for -- reachable by the tasks of this run despite the
        local-address rule. Recorded in the audit log; undone by ``unregister_local_service``."""
        with self._lock:
            host, port, kind = self._local_address(address)
            addr = f"{host}:{port}"
            self.store.x("INSERT OR REPLACE INTO local_services(address, host, port, registered_at, registered_by, note) "
                         "VALUES(?,?,?,?,?,?)", (addr, host, port, self.clock.now(), str(by), str(note)[:500]))
            self._log("local_service_registered", address=addr, kind=kind, operator=str(by), note=str(note)[:500])
            return {"address": addr, "kind": kind}

    def unregister_local_service(self, address: str, by: str) -> bool:
        with self._lock:
            host, port, _kind = self._local_address(address)
            cur = self.store.x("DELETE FROM local_services WHERE host=? AND port=?", (host, port))
            self._log("local_service_unregistered", address=f"{host}:{port}", operator=str(by), found=cur.rowcount > 0)
            return cur.rowcount > 0

    def local_services(self) -> List[Dict[str, Any]]:
        with self._lock:
            pol = [{"address": f"{h}:{p}", "source": "policy"} for h, p in (self.policy.local_services if self.policy else [])]
            reg = [{"address": r["address"], "source": "registered", "by": r["registered_by"], "at": r["registered_at"],
                    "note": r["note"]} for r in self.store.q("SELECT * FROM local_services ORDER BY address")]
            return pol + reg

    # ==========================================================================================================
    # PolicyEngine: limits_state  (+ status for the operator)
    # ==========================================================================================================
    def limits_state(self) -> Dict[str, Any]:
        with self._lock:
            self._observe_stop()
            now = self.clock.now()
            self.ledger.settle_stale(now)
            tasks: Dict[str, Any] = {}
            for row in self.store.q("SELECT task_id FROM tasks"):
                tc, err = self._ctx_for_task_id(row["task_id"], now) if self.policy is not None else (None, [])
                if tc is None:
                    tasks[row["task_id"]] = {"unavailable": [r for _p, r in err]}
                    continue
                st = self.ledger.state(tc.task_id, tc.limits, now, tc.started_at)
                st["mode"] = tc.mode.name
                if tc.window is not None:
                    st["window"] = list(tc.window)
                tasks[row["task_id"]] = st
            s, p = self.stop.engaged(), self.stop.paused()
            return {"stopped": s is not None, "stop": s, "paused": p is not None, "pause": p, "epoch": self._epoch(),
                    "policy": {"id": self.policy.policy_id if self.policy else "", "sha256": self.policy.sha256 if self.policy else "",
                               "valid": self.policy is not None, "error": self.policy_error},
                    "approvals": self.approvals_q.count(), "tasks": tasks, "ts": now}

    def reload_policy(self, by: str, path: Optional[str] = None) -> Dict[str, Any]:
        """Operator only. A policy that does not load leaves the service denying everything."""
        with self._lock:
            if path:
                self.policy_path = path
            self._load_policy()
            out = {"valid": self.policy is not None, "error": self.policy_error,
                   "policy_id": self.policy.policy_id if self.policy else "", "sha256": self.policy.sha256 if self.policy else ""}
            self._log("policy_reload", operator=str(by), **out)
            return out

    # ==========================================================================================================
    # secrets (executor side) and scrubbing
    # ==========================================================================================================
    def secret_get(self, name: str, action: C.ActionSpec, decision: C.AuthorizationDecision, destination: str) -> str:
        """Release a named secret for one destination to the executor of a granted action that is running now.
        Raises ``SecretError`` (a ``PermissionError``) otherwise. Audited by name."""
        with self._lock:
            now = self.clock.now()
            ah = self._safe_hash(action)
            try:
                value = self._secret_get(name, action, decision, destination, now, ah)
            except SecretError as e:
                self._log("secret_denied", name=str(name)[:64], action_hash=ah, destination=str(destination)[:300], reason=str(e))
                raise
            except Exception as e:  # noqa: BLE001 — fail closed
                self._log("secret_denied", name=str(name)[:64], action_hash=ah, reason=f"internal error ({type(e).__name__})")
                raise SecretError("secret not released") from None
            self._log("secret_access", name=str(name)[:64], action_hash=ah, destination=str(destination)[:300])
            return value

    def _secret_get(self, name: Any, action: Any, decision: Any, destination: Any, now: float, ah: str) -> str:
        self._observe_stop()
        blocked = self.stop.blocked()
        if blocked:
            raise SecretError(blocked)
        if self.policy is None or self._eval is None:
            raise SecretError("no valid policy is loaded")
        if not isinstance(name, str) or not isinstance(action, C.ActionSpec) or not isinstance(decision, C.AuthorizationDecision):
            raise SecretError("malformed secret request")
        if decision.action_hash != ah or decision.verdict != ALLOW:
            raise SecretError("the decision does not authorize this action")
        try:
            p = self.signer.parse(decision.grant)
        except GrantError as e:
            raise SecretError(str(e)) from None
        if p["h"] != ah:
            raise SecretError("the grant is bound to a different action")
        grant = self._active_grant(ah, now)
        if grant is None or grant["nonce"] != p["n"]:
            raise SecretError("no granted execution of this action is in progress")
        try:
            dest = D.parse_destination(destination)
        except D.DestinationError as e:
            raise SecretError(f"bad destination: {e}") from None
        tc, terr = self._ctx_for_task_id(grant["task_id"], now)
        if tc is None:
            raise SecretError(terr[0][1])
        if D.match_any(dest, self.policy.deny) or D.match_any(dest, tc.deny):
            raise SecretError(f"{dest.key} is on the deny list")
        if self._eval.local_denial(dest, tc):
            raise SecretError(f"{dest.key} is an address of this machine that is not a registered local service")
        if dest.kind == "url" and tc.envelope is not None and D.match_any(dest, tc.envelope) is None:
            raise SecretError(f"{dest.key} is outside what {tc.mode.name} mode may reach")
        if D.match_any(dest, tc.owned) is None and D.match_any(dest, tc.read) is None:
            raise SecretError(f"{dest.key} is not an allowed destination for this task")
        return self.vault.release(name, dest)

    def put_secret(self, name: str, value: str, destinations: Sequence[str], scopes: Sequence[str] = (),
                   by: str = "operator", note: str = "") -> Dict[str, Any]:
        """Operator only."""
        with self._lock:
            out = self.vault.put(name, value, list(destinations), list(scopes), by=str(by), note=note, ts=self.clock.now())
            self._log("secret_stored", name=name, destinations=list(destinations), scopes=list(scopes), operator=str(by))
            return out

    def secret_names(self) -> List[Dict[str, Any]]:
        return self.vault.names()

    def secret_has(self, name: str) -> bool:
        """Whether a usable (stored, not revoked) secret of this name exists -- no value, no binding information.
        For an executor reporting which capabilities are available (E->EA-4)."""
        with self._lock:
            ok = any(n.get("name") == str(name) and not n.get("revoked") for n in self.vault.names())
            self._log("secret_has", name=str(name)[:64], found=ok)
            return ok

    def scrub(self, text: str) -> str:
        return self.vault.scrub(text)

    def scrub_obj(self, obj: Any) -> Any:
        return self.vault.scrub_obj(obj)

    def secrets_handle(self) -> ExecutorSecrets:
        """The object to put in ``ExecContext.secrets`` (in-process mode)."""
        return ExecutorSecrets(self.secret_get, self.scrub, self.scrub_obj)

    # ==========================================================================================================
    # audit access
    # ==========================================================================================================
    def audit_head(self) -> Dict[str, Any]:
        return self.audit.head()

    def audit_tail(self, n: int = 50) -> List[Dict[str, Any]]:
        return self.audit.tail(min(int(n), 1000))

    def verify_audit(self) -> AuditVerification:
        return verify_audit_log(self.audit.path, key=self.audit.key, head_path=self.audit.head_path)

    def close(self) -> None:
        with self._lock:
            self.audit.close()
            self.store.close()
