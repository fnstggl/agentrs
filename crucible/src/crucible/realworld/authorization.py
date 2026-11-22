"""The authorisation document: a JSON file the human operator writes before any real-world experiment.

    {
      "acknowledgement": "I own or am explicitly authorised to test every base URL listed in this document.",
      "authorised_by": "Name <contact>",
      "experiment": "short-name",
      "purpose": "what is being validated",
      "base_urls": ["https://staging.example.org/api/"],
      "methods": ["GET", "POST"],
      "limits": {"max_requests": 200, "max_rate_per_s": 2.0, "max_bytes": 5000000, "max_wall_s": 900},
      "expires": "2026-12-01T00:00:00Z",
      "approval_required": true,
      "allow_private_addresses": [],
      "holdouts": {"staging-holdout-v1": 3}
    }

Every field is required except `allow_private_addresses` and `holdouts`; unknown fields are refused (a typo must
not silently widen or narrow the scope). `approval_required` says whether each consequential (non-GET/HEAD)
action needs live approval; it must be stated explicitly.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import stat
from dataclasses import dataclass, field
from urllib.parse import urlsplit

ACK = "I own or am explicitly authorised to test every base URL listed in this document."
REQUIRED = {"acknowledgement", "authorised_by", "experiment", "purpose", "base_urls", "methods", "limits", "expires",
            "approval_required"}
OPTIONAL = {"allow_private_addresses", "holdouts"}
LIMITS = {"max_requests", "max_rate_per_s", "max_bytes", "max_wall_s"}
METHODS = {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"}


class AuthorizationError(PermissionError):
    pass


@dataclass(frozen=True)
class BaseURL:
    scheme: str
    host: str
    port: int
    path: str          # always ends with "/"

    def __str__(self) -> str:
        return f"{self.scheme}://{self.host}:{self.port}{self.path}"


@dataclass(frozen=True)
class Authorization:
    authorised_by: str
    experiment: str
    purpose: str
    base_urls: tuple[BaseURL, ...]
    methods: frozenset
    max_requests: int
    max_rate_per_s: float
    max_bytes: int
    max_wall_s: float
    expires: float                      # epoch seconds (UTC)
    approval_required: bool
    allow_private_addresses: frozenset = field(default_factory=frozenset)   # "ip" or "ip:port", literal addresses only
    holdouts: tuple = ()                # ((name, budget), ...)
    sha256: str = ""
    path: str = ""

    def holdout_budget(self, name: str) -> int | None:
        return dict(self.holdouts).get(name)

    def summary(self) -> dict:
        return {"authorised_by": self.authorised_by, "experiment": self.experiment, "purpose": self.purpose,
                "base_urls": [str(b) for b in self.base_urls], "methods": sorted(self.methods),
                "limits": {"max_requests": self.max_requests, "max_rate_per_s": self.max_rate_per_s,
                           "max_bytes": self.max_bytes, "max_wall_s": self.max_wall_s},
                "expires": _dt.datetime.fromtimestamp(self.expires, _dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "approval_required": self.approval_required,
                "allow_private_addresses": sorted(self.allow_private_addresses),
                "holdouts": dict(self.holdouts), "sha256": self.sha256}


def default_port(scheme: str) -> int:
    return 443 if scheme == "https" else 80


def parse_base_url(u: str) -> BaseURL:
    if not isinstance(u, str) or any(c in u for c in "*\\ \t\r\n"):
        raise AuthorizationError(f"base URL {u!r}: wildcards, backslashes and whitespace are not allowed")
    s = urlsplit(u)
    if s.scheme not in ("http", "https"):
        raise AuthorizationError(f"base URL {u!r}: scheme must be http or https")
    if s.username is not None or s.password is not None:
        raise AuthorizationError(f"base URL {u!r}: credentials in URLs are refused")
    if not s.hostname:
        raise AuthorizationError(f"base URL {u!r}: no host")
    if s.query or s.fragment:
        raise AuthorizationError(f"base URL {u!r}: must not carry a query or fragment")
    try:
        port = s.port or default_port(s.scheme)
    except ValueError as e:
        raise AuthorizationError(f"base URL {u!r}: {e}")
    path = s.path or "/"
    if ".." in path.split("/") or "%" in path:
        raise AuthorizationError(f"base URL {u!r}: path must be literal (no '..', no percent-escapes)")
    if not path.endswith("/"):
        path += "/"
    return BaseURL(s.scheme, s.hostname.lower().rstrip("."), int(port), path)


def _parse_time(x) -> float:
    if not isinstance(x, str) or not x.endswith("Z"):
        raise AuthorizationError("expires must be an ISO-8601 UTC time ending in 'Z', e.g. 2026-12-01T00:00:00Z")
    try:
        d = _dt.datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=_dt.timezone.utc)
    except ValueError as e:
        raise AuthorizationError(f"expires: {e}")
    return d.timestamp()


def parse_authorization(doc: dict, *, now: float, sha256: str = "", path: str = "") -> Authorization:
    if not isinstance(doc, dict):
        raise AuthorizationError("the authorisation document must be a JSON object")
    keys = set(doc)
    if REQUIRED - keys:
        raise AuthorizationError(f"missing fields: {sorted(REQUIRED - keys)}")
    if keys - REQUIRED - OPTIONAL:
        raise AuthorizationError(f"unknown fields: {sorted(keys - REQUIRED - OPTIONAL)}")
    if doc["acknowledgement"] != ACK:
        raise AuthorizationError("the acknowledgement must be exactly: " + ACK)
    for k in ("authorised_by", "experiment", "purpose"):
        if not isinstance(doc[k], str) or not doc[k].strip():
            raise AuthorizationError(f"{k} must be a non-empty string")
    if not isinstance(doc["base_urls"], list) or not doc["base_urls"] or len(doc["base_urls"]) > 20:
        raise AuthorizationError("base_urls must list 1-20 base URLs")
    bases = tuple(parse_base_url(u) for u in doc["base_urls"])
    if not isinstance(doc["methods"], list) or not doc["methods"]:
        raise AuthorizationError("methods must be a non-empty list")
    methods = frozenset(str(m).upper() for m in doc["methods"])
    if methods - METHODS:
        raise AuthorizationError(f"methods not supported: {sorted(methods - METHODS)}")
    lim = doc["limits"]
    if not isinstance(lim, dict) or set(lim) != LIMITS:
        raise AuthorizationError(f"limits must have exactly {sorted(LIMITS)}")
    for k in LIMITS:
        if isinstance(lim[k], bool) or not isinstance(lim[k], (int, float)) or not lim[k] > 0:
            raise AuthorizationError(f"limits.{k} must be a positive number")
    if lim["max_requests"] > 100_000 or lim["max_wall_s"] > 7 * 86400 or lim["max_bytes"] > 10**10:
        raise AuthorizationError("limits are beyond what this bridge accepts (100,000 requests, 7 days, 10 GB)")
    expires = _parse_time(doc["expires"])
    if expires <= now:
        raise AuthorizationError("the authorisation has expired")
    if expires - now > 90 * 86400:
        raise AuthorizationError("an authorisation may be valid for at most 90 days")
    if not isinstance(doc["approval_required"], bool):
        raise AuthorizationError("approval_required must be true or false")
    priv = doc.get("allow_private_addresses", [])
    if not isinstance(priv, list) or not all(isinstance(x, str) for x in priv):
        raise AuthorizationError("allow_private_addresses must be a list of 'ip' or 'ip:port' strings")
    import ipaddress
    for x in priv:
        host = x.rsplit(":", 1)[0] if x.count(":") == 1 else x
        try:
            ipaddress.ip_address(host.strip("[]"))
        except ValueError:
            raise AuthorizationError(f"allow_private_addresses: {x!r} is not a literal IP address (names are not accepted)")
    hold = doc.get("holdouts", {})
    if not isinstance(hold, dict) or not all(isinstance(k, str) and isinstance(v, int) and not isinstance(v, bool) and v >= 0
                                             for k, v in hold.items()):
        raise AuthorizationError("holdouts must map a holdout name to its evaluation budget (integer >= 0)")
    return Authorization(doc["authorised_by"].strip(), doc["experiment"].strip(), doc["purpose"].strip(), bases, methods,
                         int(lim["max_requests"]), float(lim["max_rate_per_s"]), int(lim["max_bytes"]),
                         float(lim["max_wall_s"]), expires, doc["approval_required"], frozenset(priv),
                         tuple(sorted(hold.items())), sha256, path)


def load_authorization(path: str, *, now: float | None = None) -> Authorization:
    """Read the operator's authorisation document. Refused when the file is missing, is not a regular file owned
    by the current user, is writable by group or others, lives inside a sandbox working directory, or does not
    validate."""
    import time
    if not isinstance(path, str) or "://" in path:
        raise AuthorizationError("the authorisation document must be a local file path")
    real = os.path.realpath(path)
    try:
        st = os.stat(real)
    except OSError as e:
        raise AuthorizationError(f"cannot read the authorisation document: {e.strerror}")
    if not stat.S_ISREG(st.st_mode):
        raise AuthorizationError("the authorisation document must be a regular file")
    if st.st_uid != os.getuid():
        raise AuthorizationError("the authorisation document must be owned by the operator running this process")
    if st.st_mode & 0o022:
        raise AuthorizationError("the authorisation document must not be writable by group or others")
    sbx = os.path.realpath(os.environ.get("CRUCIBLE_SANDBOX_DIR", "/var/tmp/crucible-sbx"))
    if real == sbx or real.startswith(sbx + os.sep):
        raise AuthorizationError("an authorisation document inside a sandbox directory is refused")
    with open(real, "rb") as f:
        raw = f.read(1_000_000)
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise AuthorizationError(f"the authorisation document is not valid JSON: {e}")
    return parse_authorization(doc, now=time.time() if now is None else now,
                               sha256=hashlib.sha256(raw).hexdigest(), path=real)
