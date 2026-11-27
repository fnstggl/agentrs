"""cam_live.runtime.logs — durable structured run logs (spec Part 14).

One append-only JSONL file per record family (names from ``contracts.RunFiles``). Every line is an envelope

    {"rid", "seq", "ts", "type", "run_id", "provenance", "source", "data"}

* ``rid`` is unique inside the run and is what the journal cites; ``seq`` is one counter shared by all files, so
  the relative order of records in different files (for example a prediction and the execution it precedes) can
  be proven from the logs alone.
* ``ts`` comes from the injected :class:`~cam_live.contracts.Clock`.
* ``data`` is redacted before it is serialised: known secret values, secret-looking strings, prohibited personal
  information (configurable patterns) are removed and member identifiers are replaced by keyed hashes.
* Records marked consequential are fsync'ed before :meth:`RunLog.append` returns.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import threading
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from ..contracts import (Clock, Decision, ExperimentRecord, LearningUpdateRecord, MetaUpdateRecord, ObstacleRecord,
                         Observation, PredictionRecord, Record, RunFiles)

#: file name -> rid prefix
LOG_FILES: Dict[str, str] = {
    RunFiles.EVENTS: "ev", RunFiles.DECISIONS: "dec", RunFiles.EXPERIMENTS: "exp", RunFiles.PREDICTIONS: "pred",
    RunFiles.LEARNING: "lu", RunFiles.META: "mu", RunFiles.OBSTACLES: "ob",
}

#: secret-looking strings (label -> regex). Extended / replaced through ``runtime.redaction.patterns`` in the config.
DEFAULT_SECRET_PATTERNS: Dict[str, str] = {
    "bot_token": r"\b[A-Za-z0-9_-]{23,28}\.[A-Za-z0-9_-]{6,7}\.[A-Za-z0-9_-]{27,}\b",
    "bearer": r"(?i)\b(?:bearer|bot)\s+[A-Za-z0-9._~+/=-]{16,}",
    "api_key": r"\b(?:sk|pk|rk|ghp|gho|github_pat|hf|xox[abprs])[-_][A-Za-z0-9_-]{16,}\b",
    "aws_key": r"\bAKIA[0-9A-Z]{16}\b",
    "private_key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----",
    "assignment": r"(?i)\b(password|passwd|secret|token|api[_-]?key|authorization)\b(\s*[=:]\s*)(\"[^\"]+\"|'[^']+'|[^\s,;&]+)",
    "webhook": r"https://(?:discord|discordapp)\.com/api/webhooks/\d+/[A-Za-z0-9_-]+",
}
#: prohibited personal information (label -> regex)
DEFAULT_PII_PATTERNS: Dict[str, str] = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "phone": r"(?<![\w.])\+?\d{1,3}[\s.-]?\(?\d{2,4}\)?[\s.-]\d{3,4}[\s.-]\d{3,4}(?![\w.])",
    "ipv4": r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b",
}
#: mapping keys whose values identify a person -> replaced by a keyed hash (stable inside one run)
DEFAULT_MEMBER_KEYS = ("member_id", "user_id", "author_id", "inviter_id", "account_id", "username", "user_name",
                       "display_name", "global_name", "nick", "member", "user", "author")
#: mapping keys whose values are secrets whatever they look like
DEFAULT_SECRET_KEYS = ("password", "passwd", "secret", "token", "api_key", "apikey", "authorization", "cookie",
                       "cookies", "credential", "credentials", "grant", "session_token", "access_token",
                       "refresh_token", "private_key")
#: mapping keys dropped unless the data permission named in the value is granted
DEFAULT_PERMISSION_KEYS: Dict[str, str] = {"message_content": "store_message_content",
                                           "dm_content": "store_message_content",
                                           "raw_messages": "store_message_content"}
_MENTION = re.compile(r"<@!?(\d{5,})>")


class Redactor:
    """Removes secrets and prohibited personal information from anything about to be written to a run log."""

    def __init__(self, salt: bytes = b"", *, secret_values: Iterable[str] = (),
                 secret_patterns: Optional[Dict[str, str]] = None, pii_patterns: Optional[Dict[str, str]] = None,
                 member_keys: Sequence[str] = DEFAULT_MEMBER_KEYS, secret_keys: Sequence[str] = DEFAULT_SECRET_KEYS,
                 permission_keys: Optional[Dict[str, str]] = None, data_permissions: Optional[Dict[str, bool]] = None):
        self._salt = salt or b"cam-live-redaction"
        self._secrets: List[str] = []
        for v in secret_values:
            self.add_secret(v)
        sp = DEFAULT_SECRET_PATTERNS if secret_patterns is None else secret_patterns
        pp = DEFAULT_PII_PATTERNS if pii_patterns is None else pii_patterns
        self._patterns = [(label, re.compile(rx)) for label, rx in list(sp.items()) + list(pp.items())]
        self._member_keys = {k.lower() for k in member_keys}
        self._secret_keys = {k.lower() for k in secret_keys}
        self._permission_keys = dict(DEFAULT_PERMISSION_KEYS if permission_keys is None else permission_keys)
        self._permissions = dict(data_permissions or {})
        self.counts: Dict[str, int] = {}

    @classmethod
    def from_config(cls, section: Optional[Dict[str, Any]], salt: bytes = b"",
                    data_permissions: Optional[Dict[str, bool]] = None) -> "Redactor":
        s = dict(section or {})
        sp = dict(DEFAULT_SECRET_PATTERNS)
        sp.update(s.get("patterns", {}))
        pp = dict(DEFAULT_PII_PATTERNS)
        pp.update(s.get("pii_patterns", {}))
        return cls(salt, secret_values=s.get("secret_values", ()), secret_patterns=sp, pii_patterns=pp,
                   member_keys=tuple(DEFAULT_MEMBER_KEYS) + tuple(s.get("member_keys", ())),
                   secret_keys=tuple(DEFAULT_SECRET_KEYS) + tuple(s.get("secret_keys", ())),
                   permission_keys={**DEFAULT_PERMISSION_KEYS, **s.get("permission_keys", {})},
                   data_permissions=data_permissions)

    def add_secret(self, value: str) -> None:
        if value and len(value) >= 4 and value not in self._secrets:
            self._secrets.append(value)
            self._secrets.sort(key=len, reverse=True)

    def set_permissions(self, data_permissions: Optional[Dict[str, bool]]) -> None:
        self._permissions = dict(data_permissions or {})

    def _hit(self, label: str) -> None:
        self.counts[label] = self.counts.get(label, 0) + 1

    def hash_member(self, value: Any) -> str:
        return "member:" + hmac.new(self._salt, str(value).encode(), hashlib.sha256).hexdigest()[:16]

    def text(self, s: str) -> str:
        for v in self._secrets:
            if v in s:
                s = s.replace(v, "[REDACTED:secret]")
                self._hit("secret")
        for label, rx in self._patterns:
            if label == "assignment":
                s, n = rx.subn(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED:secret]", s)
            else:
                s, n = rx.subn(f"[REDACTED:{label}]", s)
            if n:
                self.counts[label] = self.counts.get(label, 0) + n
        s, n = _MENTION.subn(lambda m: "<@" + self.hash_member(m.group(1)) + ">", s)
        if n:
            self.counts["member"] = self.counts.get("member", 0) + n
        return s

    def __call__(self, obj: Any, _key: str = "", _person: bool = False) -> Any:
        key = _key.lower()
        if _person and key in ("id", "name", "tag") and obj not in ("", None) and not isinstance(obj, (dict, list, tuple)):
            self._hit("member")
            return self.hash_member(obj)
        if key and obj is not None and not isinstance(obj, (dict, list, tuple)):
            if key in self._secret_keys and obj != "":
                self._hit("secret")
                return "[REDACTED:secret]"
            if key in self._member_keys and obj != "":
                self._hit("member")
                return self.hash_member(obj)
        if key in self._permission_keys and not self._permissions.get(self._permission_keys[key], False):
            self._hit("not_permitted")
            return "[REDACTED:not_permitted]"
        if isinstance(obj, str):
            return self.text(obj)
        if isinstance(obj, dict):
            person = key in self._member_keys
            return {(self.text(k) if isinstance(k, str) else k): self(v, k if isinstance(k, str) else "", person)
                    for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self(v, _key, _person) for v in obj]
        return obj


# --------------------------------------------------------------------------------------------------------------
# file helpers
# --------------------------------------------------------------------------------------------------------------
def fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_text(path: str, text: str, mode: int = 0o644, durable: bool = True) -> None:
    """tmp file in the same directory -> fsync -> rename -> fsync directory. ``durable=False`` keeps the atomic
    rename but skips the fsyncs (safe against a process crash, not against power loss; used by fast tests)."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f".{os.path.basename(path)}.{os.getpid()}.{threading.get_ident()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            if durable:
                os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    if durable:
        fsync_dir(d)


def atomic_write_json(path: str, obj: Any, mode: int = 0o644, durable: bool = True) -> None:
    atomic_write_text(path, json.dumps(obj, sort_keys=True, default=str, indent=1), mode, durable)


def read_json(path: str, default: Any = None) -> Any:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    """Yield the complete records of a JSONL file; a torn final line (crash mid-write) is skipped."""
    try:
        f = open(path, "rb")
    except FileNotFoundError:
        return
    with f:
        for raw in f:
            if not raw.endswith(b"\n"):
                return
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                continue


def read_jsonl(path: str, tail: Optional[int] = None, types: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    out = [r for r in iter_jsonl(path) if types is None or r.get("type") in types]
    return out[-tail:] if tail else out


def _last_seq(path: str) -> int:
    """seq of the last complete line (reads only the tail of the file)."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return -1
    back = 1 << 16
    with open(path, "rb") as f:
        while True:
            back = min(size, back)
            f.seek(size - back)
            lines = f.read(back).split(b"\n")
            if back < size:
                lines = lines[1:]
            for ln in reversed(lines):
                if not ln.strip():
                    continue
                try:
                    rec = json.loads(ln)
                    if "rid" in rec:                       # only lines written by a RunLog count
                        return int(rec.get("seq", -1))
                except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
                    continue
            if back >= size:
                return -1
            back *= 4


def _repair_torn_tail(path: str) -> int:
    """Truncate an incomplete final line left by a crash. Returns the number of bytes removed."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return 0
    if size == 0:
        return 0
    with open(path, "rb+") as f:
        back = min(size, 1 << 20)
        f.seek(size - back)
        chunk = f.read(back)
        if chunk.endswith(b"\n"):
            return 0
        keep = size - back + chunk.rfind(b"\n") + 1
        f.truncate(keep)
        f.flush()
        os.fsync(f.fileno())
        return size - keep


LOG_POINTER = "runtime_logs.json"


def _foreign(path: str) -> bool:
    """True when ``path`` exists but is not a RunLog file this process can append to (another component, or
    another uid, already uses that name in the run directory)."""
    if not os.path.exists(path):
        return False
    if os.stat(path).st_uid != os.geteuid() or not os.access(path, os.W_OK):
        return True
    for rec in iter_jsonl(path):
        return not (isinstance(rec, dict) and "rid" in rec)
    return False


def resolve_log_dir(run_dir: str, create: bool = False) -> str:
    """Directory holding the runtime's JSONL logs: the run directory itself, unless one of the log file names is
    already taken there by another writer — then ``<run_dir>/learner`` (recorded in ``runtime_logs.json`` so every
    reader finds them)."""
    ptr = read_json(os.path.join(run_dir, LOG_POINTER))
    if isinstance(ptr, dict) and ptr.get("log_dir"):
        return os.path.join(run_dir, ptr["log_dir"])
    if create:
        taken = [n for n in LOG_FILES if _foreign(os.path.join(run_dir, n))]
        if taken:
            os.makedirs(os.path.join(run_dir, "learner"), exist_ok=True)
            atomic_write_json(os.path.join(run_dir, LOG_POINTER),
                              {"log_dir": "learner", "reason": f"{', '.join(taken)} in the run directory belong(s) to "
                                                               "another writer"})
            return os.path.join(run_dir, "learner")
    return run_dir


def _plain(x: Any) -> Any:
    return x.to_dict() if isinstance(x, Record) else x


class RunLog:
    """Writer for the structured records of one run directory (single writer: the runtime process)."""

    def __init__(self, run_dir: str, clock: Clock, *, run_id: str = "", provenance: str = "diagnostic",
                 redactor: Optional[Redactor] = None, source: str = "runtime", durable: bool = True):
        self.run_dir, self.clock, self.durable = run_dir, clock, durable
        self.run_id, self.provenance, self.source = run_id or os.path.basename(os.path.normpath(run_dir)), provenance, source
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, RunFiles.CHECKPOINTS), exist_ok=True)
        self.log_dir = resolve_log_dir(run_dir, create=True)
        self.redact = redactor or Redactor(load_salt(run_dir))
        self._lock = threading.Lock()
        self._fh: Dict[str, Any] = {}
        self.repaired: Dict[str, int] = {}
        seq = -1
        for name in LOG_FILES:
            p = os.path.join(self.log_dir, name)
            removed = _repair_torn_tail(p)
            if removed:
                self.repaired[name] = removed
            seq = max(seq, _last_seq(p))
        self._seq = seq + 1

    # ---- core --------------------------------------------------------------------------------------------
    @property
    def next_seq(self) -> int:
        return self._seq

    def append(self, name: str, type_: str, data: Any, *, consequential: bool = False, trust: str = "") -> str:
        if name not in LOG_FILES:
            raise KeyError(f"{name!r} is not a run log file")
        with self._lock:
            seq = self._seq
            self._seq += 1
            rid = f"{LOG_FILES[name]}-{seq:010d}"
            env = {"rid": rid, "seq": seq, "ts": float(self.clock.now()), "type": type_, "run_id": self.run_id,
                   "provenance": self.provenance, "source": self.source, "data": self.redact(_plain(data))}
            if trust:
                env["trust"] = trust
            fh = self._fh.get(name)
            if fh is None:
                fh = self._fh[name] = open(os.path.join(self.log_dir, name), "a", encoding="utf-8")
            fh.write(json.dumps(env, sort_keys=True, default=str) + "\n")
            fh.flush()
            if consequential and self.durable:
                os.fsync(fh.fileno())
            return rid

    def sync(self) -> None:
        with self._lock:
            for fh in self._fh.values():
                fh.flush()
                if self.durable:
                    os.fsync(fh.fileno())

    def close(self) -> None:
        self.sync()
        with self._lock:
            for fh in self._fh.values():
                fh.close()
            self._fh.clear()

    # ---- typed writers -----------------------------------------------------------------------------------
    def event(self, type_: str, data: Any = None, *, consequential: bool = False, trust: str = "") -> str:
        return self.append(RunFiles.EVENTS, type_, data if data is not None else {}, consequential=consequential,
                           trust=trust)

    def observation(self, obs: Observation, via: str = "observe") -> str:
        return self.append(RunFiles.EVENTS, "observation", {"via": via, "observation": obs.to_dict()},
                           trust=obs.provenance.trust)

    def decision(self, decision: Decision, *, objective: str, situation: Dict[str, Any]) -> str:
        return self.append(RunFiles.DECISIONS, "decision",
                           {"objective": objective, "situation": situation, "decision": decision.to_dict()},
                           consequential=True)

    def predictions(self, preds: Sequence[PredictionRecord], *, decision_id: str = "", chosen_action_id: str = "") -> List[str]:
        """Register predictions. Called (and fsync'ed) before the action they refer to is executed."""
        rids = [self.append(RunFiles.PREDICTIONS, "prediction",
                            {"prediction": p.to_dict(), "decision_id": decision_id or p.decision_id,
                             "for_chosen_action": bool(chosen_action_id) and p.action_id == chosen_action_id})
                for p in preds]
        if rids and self.durable:
            with self._lock:
                os.fsync(self._fh[RunFiles.PREDICTIONS].fileno())
        return rids

    def resolution(self, prediction_id: str, query_id: str, outcome: Dict[str, Any], *, registered_rid: str = "",
                   predicted_mean: Optional[float] = None, predicted_std: Optional[float] = None) -> str:
        """Append-only resolution of an earlier prediction (the registered record is never rewritten)."""
        return self.append(RunFiles.PREDICTIONS, "resolution",
                           {"prediction_id": prediction_id, "query_id": query_id, "outcome": outcome,
                            "registered_rid": registered_rid, "predicted_mean": predicted_mean,
                            "predicted_std": predicted_std}, consequential=True)

    def experiment(self, rec: ExperimentRecord, type_: str = "experiment") -> str:
        return self.append(RunFiles.EXPERIMENTS, type_, rec, consequential=True)

    def experiment_outcome(self, experiment_id: str, decision_id: str, outcome: Dict[str, Any]) -> str:
        return self.append(RunFiles.EXPERIMENTS, "experiment_outcome",
                           {"experiment_id": experiment_id, "decision_id": decision_id, "outcome": outcome},
                           consequential=True)

    def learning_update(self, rec: LearningUpdateRecord) -> str:
        return self.append(RunFiles.LEARNING, "learning_update", rec, consequential=True)

    def meta_update(self, rec: MetaUpdateRecord) -> str:
        return self.append(RunFiles.META, "meta_update", rec, consequential=True)

    def obstacle(self, rec: ObstacleRecord, rev: int = 0) -> str:
        return self.append(RunFiles.OBSTACLES, "obstacle", {"rev": rev, "obstacle": rec.to_dict()}, consequential=True)


def load_salt(run_dir: str) -> bytes:
    """Per-run random salt for member-identifier hashing (stable across restarts of the same run)."""
    p = os.path.join(run_dir, ".redaction_salt")
    try:
        with open(p, "rb") as f:
            return f.read()
    except FileNotFoundError:
        os.makedirs(run_dir, exist_ok=True)
        salt = os.urandom(32)
        try:
            fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            with open(p, "rb") as f:
                return f.read()
        with os.fdopen(fd, "wb") as f:
            f.write(salt)
            f.flush()
            os.fsync(f.fileno())
        return salt


class RunReader:
    """Read-only view of a run directory, shared by the journal, the dashboard and the CLI."""

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.log_dir = resolve_log_dir(run_dir)

    def path(self, name: str) -> str:
        return os.path.join(self.log_dir if name in LOG_FILES else self.run_dir, name)

    def records(self, name: str, types: Optional[Sequence[str]] = None, tail: Optional[int] = None) -> List[Dict[str, Any]]:
        out = [r for r in iter_jsonl(self.path(name)) if "rid" in r and (types is None or r.get("type") in types)]
        return out[-tail:] if tail else out

    def state(self) -> Dict[str, Any]:
        return read_json(self.path(RunFiles.STATE), {}) or {}

    def latest(self, name: str, type_: str) -> Optional[Dict[str, Any]]:
        last = None
        for r in self.records(name, types=[type_]):
            last = r
        return last

    def obstacles(self) -> List[Dict[str, Any]]:
        """Latest revision of every obstacle record (the file is append-only; revisions accumulate)."""
        by: Dict[str, Dict[str, Any]] = {}
        for r in self.records(RunFiles.OBSTACLES):
            d = r.get("data", {})
            oid = d.get("obstacle", {}).get("obstacle_id", "")
            if oid and (oid not in by or d.get("rev", 0) >= by[oid]["data"].get("rev", 0)):
                by[oid] = r
        return list(by.values())
