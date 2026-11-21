"""Pluggable isolation backends for agent-written code.

Every sandboxed program always runs under the in-process protections of `boot.py` (uid drop, no_new_privs,
Landlock filesystem/network rules, seccomp filter, rlimits). A *backend* adds an outer isolation layer around that
process. Backends:

* ``landlock-seccomp`` - the default: no outer layer (process isolation on the shared host kernel).
* ``gvisor``           - run the worker under gVisor's user-space kernel (``runsc --rootless --network=none do``).
* ``bubblewrap``       - run the worker in fresh user/mount/pid/net namespaces (``bwrap --unshare-all``).
* ``firecracker``      - a KVM micro-VM per worker pool; requires /dev/kvm and a prepared rootfs/kernel image
                         (``CRUCIBLE_FC_KERNEL``, ``CRUCIBLE_FC_ROOTFS``); the pipe protocol runs over vsock.

`detect()` reports what this host can actually provide, with reasons. Requesting an unavailable backend raises
`SandboxUnavailable`; there is never a silent fallback to weaker isolation. On the development host of this
repository (an unprivileged Docker container: no /dev/kvm, no runsc/bwrap binaries, user namespaces refused) only
``landlock-seccomp`` is available, so VM-grade isolation is *not* provided there.
"""
from __future__ import annotations

import os
import shutil
import subprocess

DEFAULT = "landlock-seccomp"
ISOLATION = {"landlock-seccomp": "process (shared kernel)", "bubblewrap": "namespaces (shared kernel)",
             "gvisor": "user-space kernel", "firecracker": "virtual machine (KVM)"}


class SandboxUnavailable(RuntimeError):
    pass


def _userns_ok() -> bool:
    try:
        return subprocess.run(["unshare", "-Ur", "true"], capture_output=True, timeout=5).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def detect() -> dict[str, dict]:
    from .runner import capabilities
    caps = capabilities()
    out = {DEFAULT: {"available": caps["landlock_abi"] >= 1 and bool(caps["seccomp"]),
                     "reason": f"landlock ABI {caps['landlock_abi']}, seccomp {caps['seccomp']}"}}
    runsc = shutil.which("runsc")
    out["gvisor"] = {"available": bool(runsc), "reason": f"runsc at {runsc}" if runsc else "runsc not installed"}
    bwrap = shutil.which("bwrap")
    userns = _userns_ok() if bwrap else False
    out["bubblewrap"] = {"available": bool(bwrap) and userns,
                         "reason": "bwrap not installed" if not bwrap else
                         ("user namespaces available" if userns else "user namespaces refused by the host")}
    kvm = os.path.exists("/dev/kvm")
    fc = shutil.which("firecracker")
    img = bool(os.environ.get("CRUCIBLE_FC_KERNEL")) and bool(os.environ.get("CRUCIBLE_FC_ROOTFS"))
    out["firecracker"] = {"available": kvm and bool(fc) and img,
                          "reason": ", ".join(x for x in ["" if kvm else "no /dev/kvm",
                                                          "" if fc else "firecracker not installed",
                                                          "" if img else "no kernel/rootfs image configured"] if x)
                          or "ready"}
    for k, v in out.items():
        v["isolation"] = ISOLATION[k]
    return out


def wrap(backend: str, argv: list[str], workdir: str) -> list[str]:
    """The command line that runs `argv` inside `backend` (the inner sandbox is unchanged)."""
    if backend == DEFAULT:
        return list(argv)
    if backend == "gvisor":
        return ["runsc", "--rootless", "--network=none", "--ignore-cgroups", "do", "--"] + list(argv)
    if backend == "bubblewrap":
        return (["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--ro-bind", "/usr", "/usr",
                 "--ro-bind", "/lib", "/lib", "--ro-bind-try", "/lib64", "/lib64", "--ro-bind", "/etc", "/etc",
                 "--ro-bind-try", "/venv", "/venv", "--bind", workdir, workdir, "--proc", "/proc", "--dev", "/dev",
                 "--"] + list(argv))
    if backend == "firecracker":
        raise SandboxUnavailable("firecracker workers are launched by a VM manager, not by wrapping a command")
    raise SandboxUnavailable(f"unknown sandbox backend {backend!r}")


def require(backend: str) -> None:
    info = detect().get(backend)
    if info is None:
        raise SandboxUnavailable(f"unknown sandbox backend {backend!r}")
    if not info["available"]:
        raise SandboxUnavailable(f"sandbox backend {backend!r} unavailable on this host: {info['reason']}")
