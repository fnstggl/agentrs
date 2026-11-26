"""cam_live.harness — the general-purpose computer environment and its independent security services.

MASTER-OWNED file: service identities and socket names shared by the harness workstreams and the runtime.
"""
from __future__ import annotations

import os
import pwd
import subprocess
from typing import Dict

#: role -> unix user. Separate uids are the isolation boundary available in this container.
SERVICE_USERS: Dict[str, str] = {
    "learner": "camlearn",       # runtime + neural learner (GPU). Holds no secret, cannot write audit/verifier logs.
    "browser": "cambrowser",     # Chromium processes and the persistent browser profile (auth cookies)
    "sandbox": "camsbx",         # generated code: python / shell / background jobs
    "auth": "camauth",           # authorization service + secret vault
    "verifier": "camverify",     # independent outcome verifier
}
SERVICE_GROUP = "camlive"        # shared group used only for socket access

#: socket file names inside <run_dir>/sock/
SOCK_AUTH, SOCK_EXEC, SOCK_VERIFY = "auth.sock", "exec.sock", "verify.sock"

#: module entry points started by the launcher (``python -m <module> --run-dir R --socket R/sock/<name>``)
SERVICE_MODULES: Dict[str, str] = {
    "auth": "cam_live.harness.auth.service",
    "exec": "cam_live.harness.service",
    "verifier": "cam_live.runtime.verifier_service",
    "learner": "cam_live.runtime.loop_service",
}


def uid_of(role: str) -> int:
    """uid for a role, or the current uid when the service users do not exist (single-process / test mode)."""
    try:
        return pwd.getpwnam(SERVICE_USERS[role]).pw_uid
    except KeyError:
        return os.getuid()


def users_exist() -> bool:
    try:
        for u in SERVICE_USERS.values():
            pwd.getpwnam(u)
        return True
    except KeyError:
        return False


def ensure_service_users() -> Dict[str, int]:
    """Create the service users and group (idempotent; needs root). Returns role -> uid."""
    if os.geteuid() != 0:
        raise PermissionError("creating service users needs root")
    subprocess.run(["groupadd", "-f", SERVICE_GROUP], check=True)
    for user in SERVICE_USERS.values():
        try:
            pwd.getpwnam(user)
        except KeyError:
            subprocess.run(["useradd", "--system", "--no-create-home", "--shell", "/usr/sbin/nologin",
                            "-G", SERVICE_GROUP, user], check=True)
    return {role: pwd.getpwnam(u).pw_uid for role, u in SERVICE_USERS.items()}
