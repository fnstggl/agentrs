"""Small durable-filesystem helpers: write-to-temp, fsync, atomic rename, directory fsync."""
from __future__ import annotations

import json
import os
import tempfile


def fsync_dir(path: str) -> None:
    """fsync a directory so a completed rename inside it survives power loss (no-op where unsupported)."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def fsync_file(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path: str, data: bytes, fsync: bool = True) -> None:
    """Write ``data`` to ``path`` atomically: temp file in the same directory -> fsync -> rename -> fsync(dir)."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=d)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            if fsync:
                os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    if fsync:
        fsync_dir(d)


def atomic_write_json(path: str, obj, fsync: bool = True) -> None:
    atomic_write_bytes(path, json.dumps(obj, indent=1, sort_keys=True).encode(), fsync=fsync)


def sha256_file(path: str, bufsize: int = 1 << 20) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def dir_bytes(path: str) -> int:
    tot = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                tot += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return tot
