"""cam_live.harness.safefs — symlink- and race-safe file operations for processes that write into directories
another uid can write (G2-harness R3-M1).

In the launched deployment root (the supervisor, the launcher, the operator CLIs, the sandbox host) writes into the
run directory, which the learner's group may write, and into directories owned by service uids. A path-based
``open(path, O_CREAT|O_TRUNC)`` there follows a symlink the other uid planted (also one planted *after* a check),
so root would overwrite, chmod or chown any file of the machine. Everything here instead:

* walks to the parent directory one component at a time with ``O_DIRECTORY|O_NOFOLLOW`` relative to the previous
  directory's fd, so no symlink anywhere in the path is followed (:func:`open_dir`);
* writes through a temporary file with a random name created ``O_CREAT|O_EXCL|O_NOFOLLOW`` relative to that
  directory fd, sets owner and mode on the open fd (``fchown`` / ``fchmod``), and ``os.replace``-s it onto the
  target *within the same directory fd* -- a rename replaces a planted symlink entry instead of following it --
  after refusing a pre-existing target that is not a regular file (:func:`write_atomic`);
* reads with ``O_NOFOLLOW`` (:func:`read_nofollow`), creates directories and checks them after creation
  (:func:`ensure_dir`), and changes owner / mode only through an fd opened without following symlinks.
"""
from __future__ import annotations

import errno
import os
import secrets
import stat
from typing import Optional, Tuple

_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
#: walking needs search permission only (the run directory is 1771: services may not read it): O_PATH
_WALK_FLAGS = getattr(os, "O_PATH", os.O_RDONLY) | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


class UnsafePath(OSError):
    """A path component is a symlink or the target is not what it must be."""


def _trusted_dir(fd: int) -> bool:
    """A directory only root can change: a symlink found in it was put there by root and may be followed."""
    st = os.fstat(fd)
    return st.st_uid == 0 and not (st.st_mode & (stat.S_IWGRP | stat.S_IWOTH))


def open_dir(path: str, _depth: int = 0) -> int:
    """fd of directory ``path``, reached one component at a time with ``O_NOFOLLOW``. A symlink is followed only
    where it lies in a directory nobody but root can write (``/workspace -> /data``-style links of the machine);
    a symlink in a directory another uid can write (the run directory, a service's directory, ``/tmp``) is refused."""
    if _depth > 8:
        raise UnsafePath(errno.ELOOP, f"{path}: too many levels of symbolic links")
    p = os.path.abspath(path)
    parts = [x for x in p.split("/") if x]
    fd = os.open("/", _WALK_FLAGS)
    try:
        for i, part in enumerate(parts):
            try:
                nfd = os.open(part, _WALK_FLAGS, dir_fd=fd)
            except OSError as e:
                if e.errno not in (errno.ELOOP, errno.ENOTDIR):
                    raise
                try:
                    st = os.stat(part, dir_fd=fd, follow_symlinks=False)
                except OSError:
                    raise e from None
                if stat.S_ISLNK(st.st_mode) and st.st_uid == 0 and _trusted_dir(fd):
                    target = os.readlink(part, dir_fd=fd)
                    base = "/" + "/".join(parts[:i])
                    rest = "/".join(parts[i + 1:])
                    resolved = os.path.normpath(os.path.join(base, target, rest))
                    os.close(fd)
                    fd = -1
                    return open_dir(resolved, _depth + 1)
                raise UnsafePath(errno.ELOOP, f"{p}: component {part!r} is a symlink in a directory another uid "
                                              f"can write, or not a directory") from None
            os.close(fd)
            fd = nfd
        return fd
    except BaseException:
        if fd >= 0:
            os.close(fd)
        raise


def split(path: str) -> Tuple[str, str]:
    p = os.path.abspath(path)
    d, name = os.path.dirname(p), os.path.basename(p)
    if not name or name in (".", ".."):
        raise UnsafePath(errno.EINVAL, f"{path}: not a file name")
    return d, name


def _check_target(dfd: int, name: str) -> Optional[os.stat_result]:
    try:
        st = os.stat(name, dir_fd=dfd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(st.st_mode):
        raise UnsafePath(errno.EEXIST, f"refusing to replace {name!r}: it exists and is not a regular file "
                                       f"(a symlink or another kind of entry)")
    return st


def write_atomic(path: str, data: bytes, mode: int = 0o644, uid: Optional[int] = None, gid: Optional[int] = None,
                 durable: bool = True, umask: bool = False) -> None:
    """Replace ``path`` atomically with ``data``; never follows a symlink, never writes through a planted entry.
    ``mode`` is applied exactly (``fchmod``) unless ``umask`` is true (then ``mode & ~umask``, like ``open``)."""
    d, name = split(path)
    dfd = open_dir(d)
    try:
        _check_target(dfd, name)
        tmp = f".{name}.{secrets.token_hex(8)}.tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                     0o600, dir_fd=dfd)
        try:
            try:
                view = memoryview(data)
                while view:
                    n = os.write(fd, view)
                    view = view[n:]
                if uid is not None or gid is not None:
                    os.fchown(fd, -1 if uid is None else uid, -1 if gid is None else gid)
                if umask:
                    cur = os.umask(0)
                    os.umask(cur)
                    os.fchmod(fd, mode & ~cur)
                else:
                    os.fchmod(fd, mode)
                if durable:
                    os.fsync(fd)
            finally:
                os.close(fd)
            _check_target(dfd, name)            # re-checked right before the rename (an entry planted meanwhile)
            os.replace(tmp, name, src_dir_fd=dfd, dst_dir_fd=dfd)
        except BaseException:
            try:
                os.unlink(tmp, dir_fd=dfd)
            except OSError:
                pass
            raise
        if durable:
            try:
                rfd = os.open(".", os.O_RDONLY | os.O_DIRECTORY, dir_fd=dfd)
                try:
                    os.fsync(rfd)
                finally:
                    os.close(rfd)
            except OSError:
                pass                       # no read permission on the directory: the file itself was synced
    finally:
        os.close(dfd)


def read_nofollow(path: str, max_bytes: int = 64 * 1024 * 1024) -> bytes:
    """Contents of the regular file ``path``; refuses a symlink anywhere in the path or a non-regular file."""
    d, name = split(path)
    dfd = open_dir(d)
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0), dir_fd=dfd)
    except OSError as e:
        os.close(dfd)
        if e.errno == errno.ELOOP:
            raise UnsafePath(e.errno, f"{path} is a symlink") from None
        raise
    os.close(dfd)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise UnsafePath(errno.EINVAL, f"{path} is not a regular file")
        out = bytearray()
        while len(out) < max_bytes:
            chunk = os.read(fd, min(1 << 20, max_bytes - len(out)))
            if not chunk:
                break
            out += chunk
        return bytes(out)
    finally:
        os.close(fd)


def ensure_dir(path: str, mode: int = 0o700, uid: Optional[int] = None, gid: Optional[int] = None) -> None:
    """Create directory ``path`` if missing (parents must exist), never through a symlink; then set owner and mode on
    an fd of the directory itself. An existing non-directory or symlink is refused."""
    d, name = split(path)
    dfd = open_dir(d)
    try:
        try:
            os.mkdir(name, 0o700, dir_fd=dfd)
        except FileExistsError:
            pass
        try:
            fd = os.open(name, _DIR_FLAGS, dir_fd=dfd)
        except OSError as e:
            if e.errno in (errno.ELOOP, errno.ENOTDIR):
                raise UnsafePath(e.errno, f"{path} is a symlink or not a directory") from None
            raise
        try:
            if uid is not None or gid is not None:
                os.fchown(fd, -1 if uid is None else uid, -1 if gid is None else gid)
            os.fchmod(fd, mode)
        finally:
            os.close(fd)
    finally:
        os.close(dfd)


def chown_chmod_nofollow(path: str, uid: Optional[int], gid: Optional[int], mode: Optional[int]) -> None:
    """Owner / mode of an existing entry, without following a symlink (a symlink is refused)."""
    d, name = split(path)
    dfd = open_dir(d)
    try:
        st = os.stat(name, dir_fd=dfd, follow_symlinks=False)
        if stat.S_ISLNK(st.st_mode):
            raise UnsafePath(errno.ELOOP, f"{path} is a symlink")
        if uid is not None or gid is not None:
            os.chown(name, -1 if uid is None else uid, -1 if gid is None else gid, dir_fd=dfd, follow_symlinks=False)
        if mode is not None and not stat.S_ISSOCK(st.st_mode):
            fd = os.open(name, (os.O_RDONLY | os.O_NOFOLLOW) | (os.O_DIRECTORY if stat.S_ISDIR(st.st_mode) else 0),
                         dir_fd=dfd)
            try:
                os.fchmod(fd, mode)
            finally:
                os.close(fd)
        elif mode is not None:
            # a socket cannot be opened: re-check right before a path chmod that does not follow (lchmod is not
            # available on Linux), so set its mode at creation instead (umask) -- here only verify it
            if stat.S_IMODE(st.st_mode) != mode:
                raise UnsafePath(errno.EPERM, f"{path}: socket mode {oct(stat.S_IMODE(st.st_mode))}, expected {oct(mode)}")
    finally:
        os.close(dfd)
