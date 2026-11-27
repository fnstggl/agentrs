"""cam_live.runtime.private — sealing of the runtime's private records (write-ahead journal, raw approval decisions,
exact resume state).

The structured run logs are redacted before they are written (``runtime.logs.Redactor``). Crash recovery and the
execution of an approved action, however, need the *exact* records: the decision whose content hash the operator
approved, the result that is re-delivered to the learner after a restart, the pending outcome queries with their
original metric names. Those are kept only in sealed form: AES-256-GCM with a per-run key in ``<run_dir>/learner/``
(directory 0700, key 0600, owned by the runtime's uid; in separated mode that is the learner uid's private
directory created by the launcher).

What this protects and what it does not: a reader of the run directory without that key — the dashboard, a copy
of the logs, a backup, an export — sees ciphertext. Whoever can read the key file (the runtime's own uid, and root)
can decrypt the uncommitted span of the journal and the pending approvals.
"""
from __future__ import annotations

import base64
import json
import os
import stat
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

KEY_DIR, KEY_FILE = "learner", "runtime.key"
_AAD = b"cam-live-runtime-private-v1"
PREFIX = "v1:"


class SealError(ValueError):
    pass


def _private_dir(run_dir: str) -> str:
    d = os.path.join(run_dir, KEY_DIR)
    try:
        os.mkdir(d, 0o700)
    except FileExistsError:
        pass
    st = os.lstat(d)
    if not stat.S_ISDIR(st.st_mode):
        raise PermissionError(f"{d} is not a directory")
    if st.st_uid != os.geteuid():
        raise PermissionError(f"{d} is owned by uid {st.st_uid}, not by the runtime (uid {os.geteuid()})")
    if stat.S_IMODE(st.st_mode) & 0o077:
        os.chmod(d, 0o700)
    return d


def load_or_create_key(run_dir: str) -> bytes:
    """The per-run sealing key (32 random bytes). Created once with O_EXCL; refused when it is readable by others or
    owned by another uid."""
    path = os.path.join(_private_dir(run_dir), KEY_FILE)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError:
        st = os.lstat(path)
        if not stat.S_ISREG(st.st_mode) or st.st_uid != os.geteuid():
            raise PermissionError(f"runtime key {path} is not a regular file owned by this uid")
        if stat.S_IMODE(st.st_mode) & 0o077:
            raise PermissionError(f"runtime key {path} is readable by other users (mode {stat.S_IMODE(st.st_mode):o})")
        with open(path, "rb") as f:
            key = base64.b64decode(f.read().strip())
        if len(key) != 32:
            raise SealError(f"runtime key {path} is malformed")
        return key
    key = AESGCM.generate_key(bit_length=256)
    with os.fdopen(fd, "wb") as f:
        f.write(base64.b64encode(key))
        f.flush()
        os.fsync(f.fileno())
    return key


class PrivateBox:
    """``seal(obj) -> str`` / ``open(text) -> obj`` for JSON-serialisable objects."""

    def __init__(self, run_dir: str):
        self._aead = AESGCM(load_or_create_key(run_dir))

    def seal(self, obj: Any) -> str:
        nonce = os.urandom(12)
        body = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":")).encode()
        return PREFIX + base64.b64encode(nonce + self._aead.encrypt(nonce, body, _AAD)).decode()

    def open(self, text: str) -> Any:
        if not isinstance(text, str) or not text.startswith(PREFIX):
            raise SealError("not a sealed record")
        raw = base64.b64decode(text[len(PREFIX):])
        try:
            body = self._aead.decrypt(raw[:12], raw[12:], _AAD)
        except InvalidTag as e:
            raise SealError("sealed record does not decrypt with this run's key (altered, or another run's)") from e
        return json.loads(body)
