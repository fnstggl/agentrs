"""Secret store and credential references.

Credentials are referenced, never copied: account records hold ``secret://<backend>/<name>``
references, and only connectors resolve them at execution time. The model, prompts, logs, audit
records and the dashboard never see secret values (``redact`` scrubs them from any text).

Backends:
* ``env``  — ``AGENTRS_SECRET_<NAME>`` environment variables (injected by the deployment's secret
  manager, e.g. Kubernetes secrets / AWS Secrets Manager / Vault agent);
* ``file`` — an encrypted local store (Fernet, key from ``AGENTRS_SECRET_KEY``) for development and
  for OAuth tokens obtained at runtime;
* ``memory`` — tests only.
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Optional

from cryptography.fernet import Fernet

REF = re.compile(r"^secret://(env|file|memory)/([A-Za-z0-9_.\-]+)$")


class SecretError(KeyError):
    pass


class SecretStore:
    def __init__(self, file_path: Optional[str | Path] = None, key: Optional[bytes] = None):
        self._mem: dict[str, str] = {}
        self._lock = threading.RLock()
        self.file_path = Path(file_path) if file_path else None
        k = key or (os.environ.get("AGENTRS_SECRET_KEY") or "").encode() or None
        self._fernet = Fernet(k) if k else None

    @staticmethod
    def generate_key() -> bytes:
        return Fernet.generate_key()

    # ------------------------------------------------------------------ file backend
    def _load_file(self) -> dict[str, str]:
        if not self.file_path or not self.file_path.exists():
            return {}
        if self._fernet is None:
            raise SecretError("encrypted secret file present but AGENTRS_SECRET_KEY is not set")
        return json.loads(self._fernet.decrypt(self.file_path.read_bytes()))

    def _save_file(self, data: dict[str, str]) -> None:
        if self._fernet is None or self.file_path is None:
            raise SecretError("file secret backend needs a path and AGENTRS_SECRET_KEY")
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.file_path.with_suffix(".tmp")
        tmp.write_bytes(self._fernet.encrypt(json.dumps(data).encode()))
        os.chmod(tmp, 0o600)
        tmp.replace(self.file_path)

    # ------------------------------------------------------------------ API
    def put(self, backend: str, name: str, value: str) -> str:
        with self._lock:
            if backend == "memory":
                self._mem[name] = value
            elif backend == "file":
                d = self._load_file()
                d[name] = value
                self._save_file(d)
            else:
                raise SecretError(f"backend {backend} is read-only from the application")
        return f"secret://{backend}/{name}"

    def get(self, ref: str) -> str:
        m = REF.match(ref or "")
        if not m:
            raise SecretError(f"not a secret reference: {ref!r}")
        backend, name = m.groups()
        with self._lock:
            if backend == "env":
                v = os.environ.get("AGENTRS_SECRET_" + name.upper().replace("-", "_").replace(".", "_"))
            elif backend == "file":
                v = self._load_file().get(name)
            else:
                v = self._mem.get(name)
        if v is None:
            raise SecretError(f"secret {ref} is not set")
        return v

    def delete(self, ref: str) -> None:
        m = REF.match(ref or "")
        if not m:
            return
        backend, name = m.groups()
        with self._lock:
            if backend == "memory":
                self._mem.pop(name, None)
            elif backend == "file":
                d = self._load_file()
                d.pop(name, None)
                self._save_file(d)

    def exists(self, ref: str) -> bool:
        try:
            self.get(ref)
            return True
        except SecretError:
            return False

    def known_values(self) -> list[str]:
        """Values currently held in-process (memory/file) — used to scrub logs and to block exfiltration."""
        vals = list(self._mem.values())
        try:
            vals += list(self._load_file().values())
        except SecretError:
            pass
        vals += [v for k, v in os.environ.items() if k.startswith("AGENTRS_SECRET_")]
        return [v for v in vals if v and len(v) >= 6]

    def redact(self, text: str) -> str:
        for v in sorted(self.known_values(), key=len, reverse=True):
            text = text.replace(v, "[REDACTED]")
        return text

    def contains_secret(self, text: str) -> bool:
        return any(v in text for v in self.known_values())
