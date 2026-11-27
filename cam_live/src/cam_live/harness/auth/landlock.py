"""Landlock self-restriction for the authorization service (Linux >= 5.13, ABI 1; unprivileged, needs no_new_privs).

After ``restrict_writes`` the calling process — and everything it later executes — can still read what its uid may
read, but can create, modify or delete files only beneath the given directories and in the given existing files.
The restriction is enforced by the kernel and cannot be lifted by the process.
"""
from __future__ import annotations

import ctypes
import os
from typing import Sequence

_CREATE, _ADD_RULE, _RESTRICT = 444, 445, 446
_RULE_PATH_BENEATH = 1
EXECUTE, WRITE_FILE, READ_FILE, READ_DIR, REMOVE_DIR, REMOVE_FILE = 1 << 0, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5
MAKE_CHAR, MAKE_DIR, MAKE_REG, MAKE_SOCK, MAKE_FIFO, MAKE_BLOCK, MAKE_SYM = (1 << i for i in range(6, 13))
ALL_V1 = (1 << 13) - 1
READ = EXECUTE | READ_FILE | READ_DIR
WRITE_TREE = WRITE_FILE | REMOVE_DIR | REMOVE_FILE | MAKE_DIR | MAKE_REG | MAKE_SOCK


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathBeneath(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


def abi_version() -> int:
    """Landlock ABI of the running kernel, 0 when unavailable."""
    libc = ctypes.CDLL(None, use_errno=True)
    v = libc.syscall(_CREATE, None, 0, 1)
    return v if v > 0 else 0


def restrict_writes(dirs: Sequence[str], files: Sequence[str] = ()) -> bool:
    """Apply the restriction. Returns False (and changes nothing) when the kernel has no Landlock; raises
    ``OSError`` if it is available but could not be applied."""
    if abi_version() < 1:
        return False
    libc = ctypes.CDLL(None, use_errno=True)

    def check(rc: int, what: str) -> int:
        if rc < 0:
            e = ctypes.get_errno()
            raise OSError(e, f"landlock {what}: {os.strerror(e)}")
        return rc

    attr = _RulesetAttr(ALL_V1)
    ruleset = check(libc.syscall(_CREATE, ctypes.byref(attr), ctypes.sizeof(attr), 0), "create_ruleset")
    try:
        rules = [("/", READ)] + [(d, READ | WRITE_TREE) for d in dirs] + [(f, READ_FILE | WRITE_FILE) for f in files]
        for path, access in rules:
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = _PathBeneath(access, fd)
                check(libc.syscall(_ADD_RULE, ruleset, _RULE_PATH_BENEATH, ctypes.byref(rule), 0), f"add_rule({path})")
            finally:
                os.close(fd)
        check(libc.prctl(38, 1, 0, 0, 0), "no_new_privs")
        check(libc.syscall(_RESTRICT, ruleset, 0), "restrict_self")
    finally:
        os.close(ruleset)
    return True
