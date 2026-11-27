"""Destination normalisation and pattern matching for the authorization layer.

A *destination* is whatever an operation acts on outside the learner: a URL, an API endpoint, an account, a
channel, a file. Policy patterns are strings ``[kind:]pattern``:

* ``example.org`` / ``*.example.org`` / ``https://example.org/docs/*`` / ``127.0.0.1:8765`` — URL patterns
  (optional scheme, host glob, optional port, optional path glob). A bare host matches every path and port, and
  ``www.example.org`` is the same site as ``example.org``; other subdomains need a wildcard.
* ``file:/abs/root`` — a file root (the path itself and everything below it).
* ``account:<platform>:<id-glob>`` and ``channel:<platform>:<id-glob>`` — logical identities.

Normalisation removes cosmetic variation (case, default ports, ``www.``, dot segments, percent-encoding of
unreserved characters, integer / hex IPv4 spellings, user-info, fragments) so that the same place always has the
same key. Nothing here interprets page content: inputs are fields of a typed ``ActionSpec`` / ``PrimitiveOp``.
"""
from __future__ import annotations

import fnmatch
import ipaddress
import os
import posixpath
import re
import socket
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import unquote, urlsplit

KINDS = ("url", "file", "account", "channel", "skill", "sim")
_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
URL_IN_TEXT_RE = re.compile(r"(?i)\b(?:https?|wss?|ftp)://[^\s\"'<>\\)\]}`]+")
_DEFAULT_PORTS = {"http": 80, "https": 443, "ws": 80, "wss": 443, "ftp": 21}
_ID_RE = re.compile(r"^[A-Za-z0-9_.\-@#/]{1,200}$")


class DestinationError(ValueError):
    """The value cannot be turned into a destination (treated as a denial by the caller)."""


@dataclass(frozen=True)
class Destination:
    kind: str                # one of KINDS
    key: str                 # canonical identity, e.g. "url:example.org/docs/a", "channel:discord:123"
    host: str = ""           # url: canonical host (no port)
    port: int = 0            # url: 0 = scheme default
    path: str = ""           # url / file: normalised path
    scheme: str = ""
    platform: str = ""       # account / channel
    ident: str = ""          # account / channel / skill id
    raw: str = ""

    @property
    def rate_key(self) -> str:
        """Granularity of rate limiting: the host for URLs, the identity otherwise."""
        return f"url:{bare_host(self.host)}" if self.kind == "url" else self.key

    @property
    def intent_key(self) -> str:
        """Identity used to recognise the same target across modalities and cosmetic changes."""
        if self.kind == "url":
            return f"url:{bare_host(self.host)}"
        return self.key

    def needles(self) -> List[str]:
        """Strings whose presence in an opaque payload (code, skill arguments) shows that it refers to this
        destination."""
        if self.kind == "url":
            return [bare_host(self.host) + (f":{self.port}" if self.port else "")]
        if self.kind in ("account", "channel", "skill") and len(self.ident) >= 6:
            return [self.ident.lower()]
        if self.kind == "file":
            return [self.path.lower()]
        return []


def bare_host(host: str) -> str:
    return host[4:] if host.startswith("www.") else host


def _canon_host(host: str) -> str:
    h = (host or "").strip().rstrip(".").lower()
    if not h:
        raise DestinationError("empty host")
    if h.startswith("[") and h.endswith("]"):
        h = h[1:-1]
    try:                                   # IPv6 / dotted IPv4
        return str(ipaddress.ip_address(h))
    except ValueError:
        pass
    if re.fullmatch(r"(0x[0-9a-f]+|\d+)(\.(0x[0-9a-f]+|\d+)){0,3}", h):   # integer / hex / short IPv4 spellings
        try:
            return str(ipaddress.ip_address(socket.inet_aton(h)))
        except OSError as e:
            raise DestinationError(f"bad numeric host {host!r}") from e
    try:
        h = h.encode("idna").decode("ascii")
    except UnicodeError as e:
        raise DestinationError(f"bad host {host!r}") from e
    if not re.fullmatch(r"[a-z0-9]([a-z0-9\-_.]*[a-z0-9])?", h):
        raise DestinationError(f"bad host {host!r}")
    return h


def _canon_path(path: str) -> str:
    p = unquote(path or "/")
    if "\x00" in p:
        raise DestinationError("NUL in path")
    p = posixpath.normpath("/" + p.lstrip("/"))
    p = re.sub(r"/{2,}", "/", p)
    return p if p == "/" else p.rstrip("/")


def normalise_url(raw: str) -> Destination:
    s = (raw or "").strip()
    if not s or len(s) > 4096 or any(c in s for c in "\r\n\t "):
        raise DestinationError("not a URL")
    if not _SCHEME_RE.match(s):
        s = "https://" + s
    try:
        u = urlsplit(s)
        port = u.port
    except ValueError as e:
        raise DestinationError(str(e)) from e
    scheme = u.scheme.lower()
    if scheme not in _DEFAULT_PORTS:
        raise DestinationError(f"scheme {scheme!r} is not permitted")
    host = _canon_host(u.hostname or "")
    if port is None or port == _DEFAULT_PORTS[scheme]:
        port = 0
    path = _canon_path(u.path)
    hp = host if not port else f"{host}:{port}"
    return Destination("url", f"url:{hp}{path}", host=host, port=port, path=path, scheme=scheme, raw=raw)


def normalise_file(raw: str, base: str = "") -> Destination:
    s = (raw or "").strip()
    if not s or "\x00" in s or len(s) > 4096:
        raise DestinationError("not a path")
    if s.startswith("file://"):
        s = unquote(urlsplit(s).path)
    if not os.path.isabs(s):
        if not base:
            raise DestinationError("relative path without a workspace root")
        joined = os.path.normpath(os.path.join(base, s))
        if joined != base and not joined.startswith(base.rstrip("/") + "/"):
            raise DestinationError("relative path escapes the workspace root")
        s = joined
    p = os.path.normpath(s)
    try:                                   # follow symlinks of the part that exists (lexical otherwise)
        p = os.path.realpath(p)
    except OSError:
        pass
    return Destination("file", f"file:{p}", path=p, raw=raw)


def normalise_ident(kind: str, platform: str, ident: Any, raw: str = "") -> Destination:
    i = str(ident).strip()
    pf = (platform or "").strip().lower()
    if not pf or not _ID_RE.match(i):
        raise DestinationError(f"bad {kind} identity")
    return Destination(kind, f"{kind}:{pf}:{i}", platform=pf, ident=i, raw=raw or i)


def parse_destination(value: Any, kind: str = "url", platform: str = "", file_base: str = "") -> Destination:
    """Turn one typed action field into a destination. ``value`` may carry its own ``kind:`` prefix."""
    if isinstance(value, bool) or value is None:
        raise DestinationError("no value")
    if isinstance(value, (int,)):
        value = str(value)
    if not isinstance(value, str):
        raise DestinationError("destination must be a string")
    v = value.strip()
    for k in ("account", "channel", "skill", "sim"):
        if v.startswith(k + ":"):
            parts = v.split(":", 2)
            if len(parts) != 3:
                raise DestinationError(f"bad {k} destination")
            return normalise_ident(k, parts[1], parts[2], raw=value)
    if v.startswith("file:") or kind == "file":
        return normalise_file(v, file_base)
    if kind in ("account", "channel", "skill", "sim"):
        return normalise_ident(kind, platform, v, raw=value)
    return normalise_url(v)


# ---------------------------------------------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Pattern:
    kind: str
    text: str                # original text (shown in reasons)
    scheme: str = ""
    host: str = ""           # glob
    port: int = 0            # 0 = any
    path: str = ""           # glob; "" = any
    platform: str = ""
    ident: str = ""          # glob

    def matches(self, d: Destination) -> bool:
        if self.kind == "any":
            return True
        if self.kind != d.kind:
            return False
        if d.kind == "url":
            if self.scheme and self.scheme != d.scheme:
                return False
            if "*" in self.host or "?" in self.host:
                if not fnmatch.fnmatchcase(d.host, self.host):
                    return False
            elif bare_host(d.host) != bare_host(self.host):       # "www." is not a different site
                return False
            if self.port and self.port != (d.port or _DEFAULT_PORTS.get(d.scheme, 0)):
                return False
            if self.path and self.path not in ("/", "/*"):
                base = self.path[:-2] if self.path.endswith("/*") else None
                if not (fnmatch.fnmatchcase(d.path, self.path) or (base is not None and d.path == base)):
                    return False
            return True
        if d.kind == "file":
            root = self.path.rstrip("/") or "/"
            return d.path == root or d.path.startswith(root.rstrip("/") + "/")
        return self.platform == d.platform and fnmatch.fnmatchcase(d.ident, self.ident)


def parse_pattern(text: str) -> Pattern:
    """Parse a policy pattern. Raises ``DestinationError`` for anything malformed (the policy then fails closed)."""
    if not isinstance(text, str) or not text.strip():
        raise DestinationError(f"bad destination pattern {text!r}")
    t = text.strip()
    if t == "*":
        return Pattern("any", t)
    for k in ("account", "channel", "skill", "sim"):
        if t.startswith(k + ":"):
            parts = t.split(":", 2)
            if len(parts) != 3 or not parts[1] or not parts[2]:
                raise DestinationError(f"bad {k} pattern {text!r}")
            return Pattern(k, t, platform=parts[1].lower(), ident=parts[2])
    if t.startswith("file:"):
        p = t[5:]
        if not os.path.isabs(p):
            raise DestinationError(f"file pattern must be absolute: {text!r}")
        return Pattern("file", t, path=os.path.normpath(p))
    scheme = ""
    m = _SCHEME_RE.match(t)
    if m:
        scheme = t[:m.end() - 3].lower()
        if scheme not in _DEFAULT_PORTS:
            raise DestinationError(f"scheme not permitted in pattern {text!r}")
        t = t[m.end():]
    hostport, slash, path = t.partition("/")
    path = (slash + path) if slash else ""
    port = 0
    if hostport.startswith("["):                      # [v6]:port
        end = hostport.find("]")
        if end < 0:
            raise DestinationError(f"bad host in pattern {text!r}")
        host, rest = hostport[1:end], hostport[end + 1:]
        if rest.startswith(":"):
            port = int(rest[1:])
    elif hostport.count(":") == 1:
        host, p = hostport.split(":")
        if not p.isdigit():
            raise DestinationError(f"bad port in pattern {text!r}")
        port = int(p)
    else:
        host = hostport
    host = host.strip().lower().rstrip(".")
    if not host or not re.fullmatch(r"[a-z0-9*?\-_.:]+", host):
        raise DestinationError(f"bad host in pattern {text!r}")
    if "*" not in host and "?" not in host:
        host = _canon_host(host)
    return Pattern("url", text.strip(), scheme=scheme, host=host, port=port, path=path)


def parse_patterns(items: Optional[Iterable[Any]]) -> List[Pattern]:
    return [parse_pattern(i) for i in (items or [])]


def match_any(d: Destination, patterns: Sequence[Pattern]) -> Optional[Pattern]:
    for p in patterns:
        if p.matches(d):
            return p
    return None


def urls_in_text(text: str, limit: int = 64) -> List[Destination]:
    """URL destinations literally present in a string (used for opaque payloads such as program text)."""
    out: List[Destination] = []
    for m in URL_IN_TEXT_RE.finditer(text or ""):
        try:
            out.append(normalise_url(m.group(0).rstrip(".,;:")))
        except DestinationError:
            continue
        if len(out) >= limit:
            break
    return out


def dig(obj: Any, dotted: str) -> Tuple[bool, Any]:
    """``dig({"arguments": {"url": "x"}}, "arguments.url") -> (True, "x")``. Works on dataclasses and dicts."""
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, dict):
            if part not in cur:
                return False, None
            cur = cur[part]
        elif hasattr(cur, part) and not part.startswith("_"):
            cur = getattr(cur, part)
        else:
            return False, None
    return True, cur


def string_leaves(obj: Any, limit: int = 4096, _depth: int = 0) -> List[str]:
    """Every string inside a JSON-like value (keys included), bounded."""
    out: List[str] = []
    stack = [(obj, 0)]
    while stack and len(out) < limit:
        cur, depth = stack.pop()
        if isinstance(cur, str):
            out.append(cur)
        elif isinstance(cur, dict) and depth < 32:
            for k, v in cur.items():
                out.append(str(k))
                stack.append((v, depth + 1))
        elif isinstance(cur, (list, tuple)) and depth < 32:
            for v in cur:
                stack.append((v, depth + 1))
        elif isinstance(cur, (int, float)) and not isinstance(cur, bool):
            out.append(str(cur))
    return out
