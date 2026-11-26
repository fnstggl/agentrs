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
     "not_before": 0.0, "not_after": 0.0, "issued_by": "name", "issued_at": 0.0, "signature": "<base64>"}
"""
from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, Tuple

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

REQUIRED = ("record_id", "task_id", "task_hash", "accounts", "owned", "read", "preauthorized", "budgets",
            "not_before", "not_after", "issued_by", "issued_at")
OPTIONAL = ("interactive", "opted_in_recipients", "signature", "note")
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
