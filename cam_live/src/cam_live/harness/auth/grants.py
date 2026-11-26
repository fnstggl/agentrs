"""Signed grants.

A grant is ``base64url(payload) "." base64url(HMAC-SHA256(key, payload))`` where the payload names the action's
content hash, the task, the expiry, a random nonce and the revocation epoch. The key is readable only by the
authorization uid (and, as a separate copy, by the executor uid for offline checks of binding and expiry).
Single use is enforced by the authorization service, which remembers every nonce it issued.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
from typing import Any, Dict, Optional

GRANT_VERSION = 1
KEY_BYTES = 32


class GrantError(ValueError):
    """The token is not a valid grant for the purpose (the message says why; it is safe to log)."""


def _b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def load_or_create_key(path: str) -> bytes:
    """Read the key file, creating it (0600, exclusive) when absent."""
    try:
        with open(path, "rb") as f:
            key = f.read()
    except FileNotFoundError:
        os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
        key = secrets.token_bytes(KEY_BYTES)
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        except FileExistsError:
            with open(path, "rb") as f:
                return _check_key(f.read(), path)
        try:
            os.write(fd, key)
            os.fsync(fd)
        finally:
            os.close(fd)
    return _check_key(key, path)


def _check_key(key: bytes, path: str) -> bytes:
    if len(key) < KEY_BYTES:
        raise GrantError(f"key file {path} is too short")
    return key


class GrantSigner:
    def __init__(self, key: bytes):
        if len(key) < KEY_BYTES:
            raise GrantError("grant key too short")
        self._key = key

    def issue(self, action_hash: str, task_id: str, task_hash: str, issued_at: float, expires_at: float,
              epoch: int) -> Dict[str, Any]:
        payload = {"v": GRANT_VERSION, "h": action_hash, "t": task_id, "th": task_hash, "i": float(issued_at),
                   "e": float(expires_at), "n": secrets.token_hex(16), "p": int(epoch)}
        body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        sig = hmac.new(self._key, body, hashlib.sha256).digest()
        return {"token": _b64e(body) + "." + _b64e(sig), "payload": payload}

    def parse(self, token: Any) -> Dict[str, Any]:
        """Verify the signature and return the payload. Raises :class:`GrantError` otherwise."""
        if not isinstance(token, str) or token.count(".") != 1 or len(token) > 4096:
            raise GrantError("not a grant token")
        b, s = token.split(".")
        try:
            body, sig = _b64d(b), _b64d(s)
        except (ValueError, TypeError) as e:
            raise GrantError("grant is not decodable") from e
        if not hmac.compare_digest(hmac.new(self._key, body, hashlib.sha256).digest(), sig):
            raise GrantError("grant signature is invalid")
        try:
            p = json.loads(body.decode())
            if p["v"] != GRANT_VERSION:
                raise GrantError("unsupported grant version")
            for k, tp in (("h", str), ("t", str), ("th", str), ("n", str), ("e", (int, float)), ("i", (int, float)),
                          ("p", int)):
                if not isinstance(p[k], tp):
                    raise GrantError("malformed grant payload")
        except (ValueError, KeyError, TypeError, UnicodeDecodeError) as e:
            raise GrantError("malformed grant payload") from e
        return p

    def verify(self, token: Any, action_hash: str, now: float, task_id: Optional[str] = None) -> Dict[str, Any]:
        """Signature, binding to the action hash (and task) and expiry. Does **not** know about single use,
        revocation or the stop flag: only the authorization service does."""
        p = self.parse(token)
        if not hmac.compare_digest(p["h"], action_hash):
            raise GrantError("grant is bound to a different action")
        if task_id is not None and p["t"] != task_id:
            raise GrantError("grant is bound to a different task")
        if now > p["e"]:
            raise GrantError("grant has expired")
        return p


class OfflineGrantVerifier(GrantSigner):
    """Executor-side check of signature, binding and expiry from the executor's own copy of the key
    (``<run_dir>/exec/grant.key``). Use the service's ``check_grant`` to consume the grant."""

    @classmethod
    def from_file(cls, path: str) -> "OfflineGrantVerifier":
        with open(path, "rb") as f:
            return cls(f.read())

    def issue(self, *a: Any, **k: Any) -> Dict[str, Any]:       # the executor never mints grants
        raise GrantError("this verifier does not issue grants")
