"""Secret vault and broker.

Secrets are files in ``<run_dir>/vault/`` (directory 0700, files 0600, owned by the authorization uid). The learner
never receives a secret: an *executor* asks for a named secret for one granted action and one destination, and the
authorization service releases it only if the grant is valid and in use, the destination is permitted for that
action and the secret is bound to that destination. Every release is audited by secret **name**.

``scrub`` removes every known secret value — and its common encodings — from text, so nothing an executor observed
can carry a secret back into an ``Observation``, a log line or an RPC response.

Secret file (JSON)::

    {"name": "site_token", "value": "...", "destinations": ["https://api.example.org/*"], "scopes": ["example"],
     "created_at": 0.0, "created_by": "operator", "revoked_at": null, "note": ""}
"""
from __future__ import annotations

import base64
import json
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import quote

from . import destinations as D

NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")
MIN_SECRET_LEN = 8
SUFFIX = ".secret.json"


class SecretError(PermissionError):
    """A secret was not released. The message never contains a secret value."""


def _b64_fragments(value: bytes) -> List[str]:
    """Substrings that must appear in base64(prefix + value + suffix) whatever the surrounding bytes are."""
    out = []
    for alphabet in (base64.b64encode, base64.urlsafe_b64encode):
        for off in (0, 1, 2):
            enc = alphabet(b"\x00" * off + value).decode().rstrip("=")
            if (len(value) + off) % 3:
                enc = enc[:-1]                 # last character depends on the byte that follows
            enc = enc[(0, 2, 3)[off]:]         # leading characters depend on the bytes before
            if len(enc) >= 8:
                out.append(enc)
    return out


def squeeze(text: str) -> str:
    """Letters and digits only, case-folded: the form in which a disguised secret is still recognisable."""
    return "".join(c for c in text.casefold() if c.isalnum())


def variants(value: str) -> List[str]:
    """The value itself plus the encodings an executor or a page is likely to echo it in."""
    raw = value.encode()
    v = [value, raw.hex(), raw.hex().upper(), quote(value, safe=""), json.dumps(value)[1:-1], value[::-1]]
    v += _b64_fragments(raw)
    seen, out = set(), []
    for s in sorted(v, key=len, reverse=True):
        if len(s) >= min(MIN_SECRET_LEN, len(value)) and s not in seen:
            seen.add(s)
            out.append(s)
    return out


@dataclass
class Secret:
    name: str
    value: str
    destinations: List[str] = field(default_factory=list)
    scopes: List[str] = field(default_factory=list)
    created_at: float = 0.0
    created_by: str = ""
    revoked_at: Optional[float] = None
    note: str = ""
    patterns: List[D.Pattern] = field(default_factory=list, repr=False)
    forms: List[str] = field(default_factory=list, repr=False)

    def public(self) -> Dict[str, Any]:
        return {"name": self.name, "destinations": list(self.destinations), "scopes": list(self.scopes),
                "created_at": self.created_at, "revoked": self.revoked_at is not None, "note": self.note}


class SecretBroker:
    """Owns the vault directory. Thread-safe. Values leave only through :meth:`release`."""

    def __init__(self, vault_dir: str):
        self.dir = vault_dir
        os.makedirs(vault_dir, mode=0o700, exist_ok=True)
        self._lock = threading.RLock()
        self._secrets: Dict[str, Secret] = {}
        self._stamp: Any = None
        self.load_errors: List[str] = []
        self._reload()

    # -- storage ---------------------------------------------------------------------------------------------
    def _path(self, name: str) -> str:
        return os.path.join(self.dir, name + SUFFIX)

    def _dir_stamp(self) -> Any:
        try:
            st = os.stat(self.dir)
            return (st.st_mtime_ns, st.st_ino, len(os.listdir(self.dir)))
        except OSError:
            return None

    def _reload(self) -> None:
        with self._lock:
            secrets, errors = {}, []
            try:
                names = sorted(os.listdir(self.dir))
            except OSError as e:
                names, errors = [], [f"vault unreadable: {e.strerror}"]
            for fn in names:
                if not fn.endswith(SUFFIX):
                    continue
                p = os.path.join(self.dir, fn)
                try:
                    with open(p, "r") as f:
                        d = json.load(f)
                    name, value = d["name"], d["value"]
                    if fn != name + SUFFIX or not NAME_RE.match(name) or not isinstance(value, str):
                        raise ValueError("name does not match the file or the value is not a string")
                    if len(value) < MIN_SECRET_LEN:
                        raise ValueError("value shorter than %d characters" % MIN_SECRET_LEN)
                    s = Secret(name=name, value=value, destinations=[str(x) for x in d.get("destinations", [])],
                               scopes=[str(x) for x in d.get("scopes", [])], created_at=float(d.get("created_at", 0)),
                               created_by=str(d.get("created_by", "")), revoked_at=d.get("revoked_at"),
                               note=str(d.get("note", "")))
                    s.patterns = D.parse_patterns(s.destinations)
                    s.forms = variants(value)
                    secrets[name] = s
                except (OSError, ValueError, KeyError, TypeError) as e:
                    errors.append(f"{fn}: {type(e).__name__}: {e}")
            self._secrets, self.load_errors, self._stamp = secrets, errors, self._dir_stamp()

    def _fresh(self) -> None:
        if self._dir_stamp() != self._stamp:
            self._reload()

    def _write(self, s: Secret) -> None:
        body = {"name": s.name, "value": s.value, "destinations": s.destinations, "scopes": s.scopes,
                "created_at": s.created_at, "created_by": s.created_by, "revoked_at": s.revoked_at, "note": s.note}
        tmp = self._path(s.name) + f".{os.getpid()}.tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        try:
            os.write(fd, json.dumps(body, sort_keys=True).encode())
            os.fsync(fd)
        finally:
            os.close(fd)
        os.rename(tmp, self._path(s.name))

    # -- operator side ---------------------------------------------------------------------------------------
    def put(self, name: str, value: str, destinations: Sequence[str], scopes: Sequence[str] = (),
            by: str = "operator", note: str = "", ts: float = 0.0) -> Dict[str, Any]:
        if not NAME_RE.match(name or ""):
            raise ValueError("secret names are 1-64 characters of [A-Za-z0-9_.-]")
        if not isinstance(value, str) or len(value) < MIN_SECRET_LEN:
            raise ValueError("a secret value must be a string of at least %d characters" % MIN_SECRET_LEN)
        if not destinations:
            raise ValueError("a secret must be bound to at least one destination pattern")
        D.parse_patterns(destinations)                 # malformed binding -> refuse to store
        s = Secret(name=name, value=value, destinations=[str(x) for x in destinations],
                   scopes=[str(x) for x in scopes], created_at=float(ts), created_by=str(by), note=str(note))
        with self._lock:
            self._write(s)
            self._reload()
        return s.public()

    def revoke(self, scope: str, ts: float) -> List[str]:
        """Revoke by scope: ``all`` | ``secret:<name>`` | ``scope:<label>`` | ``destination:<pattern>`` (every
        secret bound to a destination pattern equal to it). Returns the names revoked. A revoked value is never
        released again but is still scrubbed."""
        with self._lock:
            self._fresh()
            hit = []
            for s in self._secrets.values():
                if s.revoked_at is not None:
                    continue
                if (scope == "all" or scope == f"secret:{s.name}"
                        or (scope.startswith("scope:") and scope[6:] in s.scopes)
                        or (scope.startswith("destination:") and scope[12:] in s.destinations)):
                    s.revoked_at = float(ts)
                    self._write(s)
                    hit.append(s.name)
            if hit:
                self._reload()
            return hit

    def names(self) -> List[Dict[str, Any]]:
        with self._lock:
            self._fresh()
            return [s.public() for s in self._secrets.values()]

    # -- release (called only by the policy service after it validated the grant) ------------------------------
    def release(self, name: str, destination: D.Destination) -> str:
        with self._lock:
            self._fresh()
            s = self._secrets.get(name)
            if s is None:
                raise SecretError(f"no secret named {name!r}")
            if s.revoked_at is not None:
                raise SecretError(f"secret {name!r} is revoked")
            if D.match_any(destination, s.patterns) is None:
                raise SecretError(f"secret {name!r} is not bound to {destination.key}")
            return s.value

    # -- scrubbing -------------------------------------------------------------------------------------------
    def find(self, text: str) -> List[str]:
        """Names of the secrets whose value (in any known encoding) occurs in ``text``."""
        if not text:
            return []
        with self._lock:
            self._fresh()
            return [s.name for s in self._secrets.values() if any(f in text for f in s.forms)]

    def find_loose(self, text: str) -> List[str]:
        """Names of secrets present in ``text`` in a disguised form: with separators, other case or punctuation
        (``t o k - A B C``), or in parts -- any run of at least max(8, 40 %) of the value's letters and digits that
        contains both a letter and a digit."""
        if not isinstance(text, str) or not text:
            return []
        flat = squeeze(text)
        out = []
        with self._lock:
            self._fresh()
            for s in self._secrets.values():
                sq = squeeze(s.value)
                if len(sq) < MIN_SECRET_LEN:
                    continue
                if sq in flat:
                    out.append(s.name)
                    continue
                k = max(MIN_SECRET_LEN, -(-len(sq) * 2 // 5))
                for i in range(0, len(sq) - k + 1):
                    w = sq[i:i + k]
                    if any(c.isdigit() for c in w) and any(c.isalpha() for c in w) and w in flat:
                        out.append(s.name)
                        break
        return out

    def scrub(self, text: str) -> str:
        if not isinstance(text, str) or not text:
            return text
        with self._lock:
            self._fresh()
            for s in self._secrets.values():
                for f in s.forms:
                    if f in text:
                        text = text.replace(f, f"[REDACTED:{s.name}]")
            return text

    def scrub_obj(self, obj: Any) -> Any:
        """Recursively scrub every string (dictionary keys included) of a JSON-like value."""
        if isinstance(obj, str):
            return self.scrub(obj)
        if isinstance(obj, dict):
            return {(self.scrub(k) if isinstance(k, str) else k): self.scrub_obj(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.scrub_obj(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self.scrub_obj(v) for v in obj)
        if isinstance(obj, (bytes, bytearray)):
            try:
                return self.scrub(bytes(obj).decode("utf-8")).encode("utf-8")
            except UnicodeDecodeError:
                return self.scrub(bytes(obj).decode("latin-1")).encode("latin-1", "replace")
        return obj

    def sensitive_paths(self) -> List[str]:
        return [self.dir]


def scrub_record(broker: "SecretBroker", rec: Any) -> Any:
    """Return a scrubbed copy of a contracts ``Record`` (``Observation``, ``ActionResult``, ...)."""
    cls = type(rec)
    return cls.from_dict(broker.scrub_obj(rec.to_dict()))


class ExecutorSecrets:
    """The handle placed in ``ExecContext.secrets``. It can obtain a secret only for a granted action that is
    being executed, and scrub text; it cannot list values, add, or revoke."""

    def __init__(self, getter: Any, scrubber: Any, obj_scrubber: Any = None):
        self._get, self._scrub, self._scrub_obj = getter, scrubber, obj_scrubber

    def get(self, name: str, action: Any, decision: Any, destination: str) -> str:
        return self._get(name, action, decision, destination)

    def scrub(self, text: str) -> str:
        return self._scrub(text)

    def scrub_obj(self, obj: Any) -> Any:
        if self._scrub_obj is not None:
            return self._scrub_obj(obj)
        if isinstance(obj, str):
            return self._scrub(obj)
        if isinstance(obj, dict):
            return {k: self.scrub_obj(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self.scrub_obj(v) for v in obj]
        return obj

    def scrub_record(self, rec: Any) -> Any:
        return type(rec).from_dict(self.scrub_obj(rec.to_dict()))

    def bind(self, action: Any, decision: Any) -> "BoundSecrets":
        """Handle for one granted action: ``get(name, destination)``. The environment, which holds the
        ``AuthorizationDecision``, binds it before handing ``ExecContext.secrets`` to a modality executor."""
        return BoundSecrets(self, action, decision)


class BoundSecrets:
    """``ExecutorSecrets`` fixed to one action and its grant."""

    def __init__(self, handle: ExecutorSecrets, action: Any, decision: Any):
        self._h, self._action, self._decision = handle, action, decision

    def get(self, name: str, destination: str) -> str:
        return self._h.get(name, self._action, self._decision, destination)

    def scrub(self, text: str) -> str:
        return self._h.scrub(text)

    def scrub_obj(self, obj: Any) -> Any:
        return self._h.scrub_obj(obj)

    def scrub_record(self, rec: Any) -> Any:
        return self._h.scrub_record(rec)
