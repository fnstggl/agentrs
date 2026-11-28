"""Operator authorization records for live operation.

A live task is authorized by a record the operator signs with an Ed25519 key that never leaves the operator; the
authorization service holds only the public key (``<run_dir>/auth/keys/operator.pub``), so nothing running as a
service uid — including the authorization service itself — can mint or alter a record.

Record (JSON object)::

    {"record_id": "...", "task_id": "<TaskSpec.objective_id>", "task_hash": "<TaskSpec.content_hash()>",
     "accounts": ["account:discord:<bot application id>"], "owned": ["channel:discord:*", "https://my.site/*"],
     "read": ["*"], "preauthorized": [], "interactive": [], "opted_in_recipients": [],
     "budgets": {"money": 0, "money_per_action": 0, "money_per_day": 0, "compute_s": 0,
                 "max_external_actions": 0, "max_decisions": 0, "duration_s": 0},
     "not_before": 0.0, "not_after": 0.0, "issued_by": "name", "issued_at": 0.0, "signature": "<base64>",
     "rate_limits": {"write": {"count": 10, "window_s": 3600}},      # optional; clamped to the policy's live rate_caps
     "standing": {"profile": "routine-reversible-1",                  # optional standing authorization (below)
                  "code_network": ["pypi.org", "files.pythonhosted.org"]}}

**Standing authorization** (operator amendment 2026-11-28, "OPERATIONAL AUTONOMY"). Routine, reversible operations
inside CAM's designated infrastructure run without per-item approval when the record pre-authorizes their
destinations (``preauthorized``). ``standing`` names the profile of the policy file under which that happens
(``standing.profiles`` of the policy; an unknown profile makes the service ignore the block, fail-closed) and may
list ``code_network``: hosts that generated code may reach through the sandbox's egress broker without a per-job
approval (e.g. package registries). What the policy's ``standing.retained_approval`` rules name (irreversible
operations, new external permissions, messages to people, ...) and every spending, data-use and hard platform
constraint keep their approval or denial whatever a record says.
"""
from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, Tuple

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from . import destinations as D

REQUIRED = ("record_id", "task_id", "task_hash", "accounts", "owned", "read", "preauthorized", "budgets",
            "not_before", "not_after", "issued_by", "issued_at")
OPTIONAL = ("interactive", "opted_in_recipients", "signature", "note", "rate_limits", "standing")
STANDING_KEYS = ("profile", "code_network", "routine", "note")
#: methods a record may declare routine for a destination; ``INTERACT`` = clicks and input on its pages. A DELETE is
#: never routine (irreversible), and nothing undeclared is (default: approval)
ROUTINE_METHODS = ("GET", "HEAD", "PUT", "PATCH", "POST", "INTERACT")
MAX_ROUTINE = 64
#: most hosts a standing profile may open to generated code (each one is still checked per connection)
MAX_CODE_NETWORK = 16
BUDGET_KEYS = ("money", "money_per_action", "money_per_day", "compute_s", "max_external_actions", "max_decisions",
               "duration_s")
LIST_KEYS = ("accounts", "owned", "read", "preauthorized", "interactive", "opted_in_recipients")


class LiveAuthorizationError(ValueError):
    pass


def _canonical(record: Dict[str, Any]) -> bytes:
    body = {k: v for k, v in record.items() if k != "signature"}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()


def generate_operator_key(private_path: str, public_path: str) -> None:
    """Create an Ed25519 key pair. The private key is written 0600 and is never needed by any service."""
    key = Ed25519PrivateKey.generate()
    priv = key.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption())
    pub = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    for path, data, mode in ((private_path, priv, 0o600), (public_path, pub, 0o644)):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
        try:
            os.write(fd, base64.b64encode(data))
        finally:
            os.close(fd)


def _read_key(path: str) -> bytes:
    with open(path, "rb") as f:
        raw = base64.b64decode(f.read().strip())
    if len(raw) != 32:
        raise LiveAuthorizationError(f"{path} is not an Ed25519 key")
    return raw


def validate_shape(record: Any) -> Dict[str, Any]:
    if not isinstance(record, dict):
        raise LiveAuthorizationError("record must be an object")
    unknown = set(record) - set(REQUIRED) - set(OPTIONAL)
    missing = [k for k in REQUIRED if k not in record]
    if unknown or missing:
        raise LiveAuthorizationError(f"record has unknown keys {sorted(unknown)} / missing keys {missing}")
    for k in ("record_id", "task_id", "task_hash", "issued_by"):
        if not isinstance(record[k], str) or not record[k]:
            raise LiveAuthorizationError(f"record.{k} must be a non-empty string")
    for k in LIST_KEYS:
        v = record.get(k, [])
        if not isinstance(v, list) or any(not isinstance(x, str) or not x for x in v):
            raise LiveAuthorizationError(f"record.{k} must be a list of strings")
    b = record["budgets"]
    if not isinstance(b, dict) or set(b) - set(BUDGET_KEYS):
        raise LiveAuthorizationError("record.budgets has unknown keys")
    for k in BUDGET_KEYS:
        v = b.get(k, 0)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
            raise LiveAuthorizationError(f"record.budgets.{k} must be a number >= 0")
    rl = record.get("rate_limits", {})
    if not isinstance(rl, dict) or set(rl) - {"write", "write_global"} or any(
            not isinstance(v, dict) or set(v) - {"count", "window_s"} for v in rl.values()):
        raise LiveAuthorizationError("record.rate_limits may only set write / write_global as {count, window_s}")
    st = record.get("standing")
    if st is not None:
        if not isinstance(st, dict) or set(st) - set(STANDING_KEYS) or "profile" not in st:
            raise LiveAuthorizationError(f"record.standing must be an object with 'profile' and optionally "
                                         f"{[k for k in STANDING_KEYS if k != 'profile']}")
        if not isinstance(st["profile"], str) or not st["profile"]:
            raise LiveAuthorizationError("record.standing.profile must be a non-empty string")
        cn = st.get("code_network", [])
        if not isinstance(cn, list) or any(not isinstance(x, str) or not x for x in cn) or len(cn) > MAX_CODE_NETWORK:
            raise LiveAuthorizationError(f"record.standing.code_network must be a list of at most {MAX_CODE_NETWORK} "
                                         f"host patterns")
        for x in cn:
            try:
                pat = D.parse_pattern(x)
            except D.DestinationError as e:
                raise LiveAuthorizationError(f"record.standing.code_network: {e}") from e
            if pat.kind != "url" or not _code_host_ok(pat):
                raise LiveAuthorizationError(f"record.standing.code_network names hosts one by one (a DNS name of at "
                                             f"least two labels, or a loopback address with its port for an operator "
                                             f"registered local service), no wildcard and no public address: {x!r}")
        if not isinstance(st.get("note", ""), str):
            raise LiveAuthorizationError("record.standing.note must be a string")
        routine = st.get("routine", [])
        if not isinstance(routine, list) or len(routine) > MAX_ROUTINE:
            raise LiveAuthorizationError(f"record.standing.routine must be a list of at most {MAX_ROUTINE} entries")
        for i, r in enumerate(routine):
            parse_routine(r, f"record.standing.routine[{i}]")
    for k in ("not_before", "not_after", "issued_at"):
        if isinstance(record[k], bool) or not isinstance(record[k], (int, float)):
            raise LiveAuthorizationError(f"record.{k} must be a number")
    if record["not_after"] <= record["not_before"]:
        raise LiveAuthorizationError("record time window is empty: not_after must be later than not_before")
    return record


def sign_live_authorization(record: Dict[str, Any], private_key_path: str) -> Dict[str, Any]:
    validate_shape(record)
    key = Ed25519PrivateKey.from_private_bytes(_read_key(private_key_path))
    out = {k: v for k, v in record.items() if k != "signature"}
    out["signature"] = base64.b64encode(key.sign(_canonical(out))).decode()
    return out


def verify_live_authorization(record: Any, public_key: bytes) -> Dict[str, Any]:
    """Shape and signature. Raises :class:`LiveAuthorizationError`."""
    validate_shape(record)
    sig = record.get("signature")
    if not isinstance(sig, str) or not sig:
        raise LiveAuthorizationError("record is not signed")
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(base64.b64decode(sig), _canonical(record))
    except (InvalidSignature, ValueError) as e:
        raise LiveAuthorizationError("record signature is invalid") from e
    return record


def load_public_key(path: str) -> bytes:
    return _read_key(path)


def window(record: Dict[str, Any]) -> Tuple[float, float]:
    return float(record["not_before"]), float(record["not_after"])


def _code_host_ok(pat: "D.Pattern") -> bool:
    """A code-network host is named exactly: no wildcard (a shared suffix such as *.github.io or *.co.uk is anyone's),
    no public IP literal (it bypasses the name the operator reviewed), no single-label name; a loopback address only
    with its port (an operator-registered local service, which the local-address rule checks as well)."""
    import ipaddress
    host = pat.host.strip("[]")
    if not host or any(c in host for c in "*?") or pat.path not in ("", "/", "/*"):
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        labels = [x for x in host.split(".") if x]
        return len(labels) >= 2 and all(lbl.replace("-", "").isalnum() for lbl in labels) and not labels[-1].isdigit()
    return ip.is_loopback and bool(pat.port)


def parse_routine(entry: Any, where: str = "routine") -> Tuple[Tuple[str, ...], "D.Pattern", str]:
    """``{"methods": [...], "dest": "<scheme>://<host>/<path glob>"}`` -> (methods, host pattern, path glob). The
    host must be named (no wildcard host); in the path ``*`` is one segment, ``**`` any number of segments."""
    if not isinstance(entry, dict) or set(entry) - {"methods", "dest"} or "dest" not in entry or "methods" not in entry:
        raise LiveAuthorizationError(f"{where} must be an object {{methods, dest}}")
    methods = entry["methods"]
    if not isinstance(methods, list) or not methods or any(not isinstance(m, str) or m.upper() not in ROUTINE_METHODS
                                                            for m in methods):
        raise LiveAuthorizationError(f"{where}.methods must be a non-empty list of {', '.join(ROUTINE_METHODS)} "
                                     f"(a DELETE is never routine)")
    dest = entry["dest"]
    if not isinstance(dest, str):
        raise LiveAuthorizationError(f"{where}.dest must be a string")
    try:
        pat = D.parse_pattern(dest)
    except D.DestinationError as e:
        raise LiveAuthorizationError(f"{where}.dest: {e}") from e
    if pat.kind != "url" or not pat.host or any(c in pat.host for c in "*?") or not pat.path or pat.path in ("/", "/*"):
        raise LiveAuthorizationError(f"{where}.dest names one host and a path (e.g. https://api.host.example/sites/**), "
                                     f"not {dest!r}")
    host_only = D.parse_pattern(dest.split("://", 1)[0] + "://" + dest.split("://", 1)[1].split("/", 1)[0]
                                if "://" in dest else dest.split("/", 1)[0])
    return tuple(sorted({m.upper() for m in methods})), host_only, pat.path


def standing(record: Dict[str, Any]) -> Dict[str, Any]:
    """The record's standing block (``{}`` when it has none)."""
    st = record.get("standing") if isinstance(record, dict) else None
    return dict(st) if isinstance(st, dict) else {}
