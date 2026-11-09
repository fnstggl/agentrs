"""Inbound webhook signature verification (constant-time; any malformed input verifies as False)."""

from __future__ import annotations

import hashlib
import hmac
import time
from typing import Optional, Union

from nacl.exceptions import BadSignatureError
from nacl.signing import VerifyKey


def verify_github_signature(secret: Union[str, bytes], body: bytes, header: Optional[str]) -> bool:
    """Check ``X-Hub-Signature-256: sha256=<hex HMAC-SHA256(secret, body)>``."""
    if not secret or not header or not isinstance(body, (bytes, bytearray)):
        return False
    algo, _, sig = header.strip().partition("=")
    if algo != "sha256" or not sig:
        return False
    key = secret.encode() if isinstance(secret, str) else bytes(secret)
    expected = hmac.new(key, bytes(body), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, sig.lower())


def verify_discord_signature(public_key_hex: str, timestamp: Optional[str], body: bytes,
                             signature_hex: Optional[str], max_age_s: Optional[float] = None,
                             now: Optional[float] = None) -> bool:
    """Check Discord's ``X-Signature-Ed25519`` over ``timestamp + body`` with the application's public key.

    ``max_age_s`` optionally rejects stale timestamps (replay protection)."""
    if not public_key_hex or not timestamp or not signature_hex or not isinstance(body, (bytes, bytearray)):
        return False
    try:
        key = VerifyKey(bytes.fromhex(public_key_hex))
        key.verify(timestamp.encode() + bytes(body), bytes.fromhex(signature_hex))
    except (BadSignatureError, ValueError, TypeError):
        return False
    if max_age_s is not None:
        try:
            if abs((time.time() if now is None else now) - float(timestamp)) > max_age_s:
                return False
        except ValueError:
            return False
    return True
