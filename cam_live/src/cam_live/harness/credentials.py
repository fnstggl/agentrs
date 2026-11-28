"""cam_live.harness.credentials — the operator's way of entering a credential. MASTER-OWNED.

A credential is typed (or piped) by the operator **once**, as root, and from then on exists only in two places
that the learner process cannot read:

* ``<run_dir>/vault/<name>.secret.json`` — owned by the authorization uid, mode 0600. The secret broker releases it
  to an executor only for one granted, executing action and only towards a destination the secret is bound to.
* ``<run_dir>/verifier/secrets/<name>`` — owned by the verifier uid, mode 0600 (only with ``--verifier``). The
  independent verifier measures the objective with its own copy, so the measurement never depends on CAM.

The value is never taken from the command line (it would be visible in the process table and the shell history),
never printed, and never written anywhere else. ``list`` prints names and bindings only.

    python -m cam_live.harness.credentials set discord_bot_token --run-dir R            # hidden prompt
    python -m cam_live.harness.credentials set discord_bot_token --run-dir R --from-file /root/.config/camlive/discord_bot_token
    python -m cam_live.harness.credentials list --run-dir R
    python -m cam_live.harness.credentials revoke discord_bot_token --run-dir R

Narrow, revocable credentials for CAM's own infrastructure (operator amendment 2026-11-28; docs/AUTONOMY.md):

    # a hosting / DNS / API token: bound to its API host, to the operations that may use it, for 30 days
    python -m cam_live.harness.credentials set hosting_token --run-dir R --dest 'https://api.host.example/*' \
        --op http.request --op http.get --expires-hours 720
    # a password for an account CAM is permitted to register: generated here, never shown to anyone; the browser
    # types it into the service's password field by reference (browser.fill secret='secret:<name>')
    python -m cam_live.harness.credentials set svc_password --run-dir R --dest 'https://service.example/*' \
        --op browser.fill --generate

``--dest`` binds the value to destinations (required unless the name is a known kind), ``--op`` to operations of the
granted action (globs), ``--expires-hours`` limits its lifetime; ``revoke`` ends it at once. A run that is already
running picks a new or revoked credential up at its next use (the vault directory is re-read when it changes).
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import secrets as _secrets
import stat
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

from . import users_exist
from .auth.vault import SUFFIX, SecretBroker

#: default destination bindings and whether the verifier needs its own copy
KNOWN: Dict[str, Dict[str, Any]] = {
    "discord_bot_token": {"dest": ["discord.com", "gateway.discord.gg", "*.discord.gg"], "scopes": ["discord"],
                          "verifier": True, "min_len": 50},
}
#: longest lifetime a credential may be given at entry (a longer-lived one is re-entered)
MAX_EXPIRES_HOURS = 24 * 366.0


def generate_value(nbytes: int = 32) -> str:
    """A random credential value (URL-safe, no whitespace). Generated in this process and never printed."""
    return _secrets.token_urlsafe(max(16, int(nbytes)))


def _ids(role: str) -> Optional[tuple]:
    if os.geteuid() != 0 or not users_exist():
        return None
    from .auth.launcher import _ids as ids
    return ids(role)


def _own(path: str, role: str, mode: int) -> None:
    """Mode and owner without following a symlink (root runs this in directories service uids own; R3-M1)."""
    from . import safefs
    who = _ids(role)
    safefs.chown_chmod_nofollow(path, who[0] if who else None, who[1] if who else None, mode)


def read_value(from_file: str = "", stream: Any = None) -> str:
    """The credential value, from a root-only file, a pipe, or a hidden prompt. Never from argv."""
    if from_file:
        st = os.stat(from_file)
        if stat.S_IMODE(st.st_mode) & 0o077:
            raise PermissionError(f"{from_file} is readable by other users; chmod 600 it first")
        with open(from_file) as f:
            return f.read().strip()
    stream = stream if stream is not None else sys.stdin
    if stream.isatty():
        return getpass.getpass("credential value (input hidden): ").strip()
    return stream.read().strip()


def store(run_dir: str, name: str, value: str, *, dest: Sequence[str] = (), scopes: Sequence[str] = (),
          verifier: Optional[bool] = None, by: str = "operator", note: str = "",
          expires_hours: Optional[float] = None, operations: Sequence[str] = ()) -> Dict[str, Any]:
    """Store ``value`` in the run's vault (and, where needed, as the verifier's own copy). Returns names only."""
    known = KNOWN.get(name, {})
    now = time.time()
    expires_at = None
    if expires_hours is not None:
        if not (0 < float(expires_hours) <= MAX_EXPIRES_HOURS):
            raise ValueError(f"--expires-hours must be in (0, {MAX_EXPIRES_HOURS:g}]")
        expires_at = now + float(expires_hours) * 3600.0
    dest = list(dest) or list(known.get("dest", []))
    if not dest:
        raise ValueError("a credential must be bound to at least one destination (--dest)")
    if len(value) < int(known.get("min_len", 8)):
        raise ValueError(f"that value is too short to be a {name}")
    if any(c.isspace() for c in value):
        raise ValueError("the value contains whitespace; paste the credential alone")
    from . import safefs
    R = os.path.abspath(run_dir)
    vault_dir = os.path.join(R, "vault")
    if not os.path.isdir(R):
        os.makedirs(R, exist_ok=True)
    safefs.ensure_dir(vault_dir, 0o700)                    # refuses a planted symlink instead of following it
    out = SecretBroker(vault_dir).put(name, value, dest, list(scopes) or list(known.get("scopes", [])), by=by,
                                      note=note, ts=now, expires_at=expires_at, operations=list(operations))
    _own(vault_dir, "auth", 0o700)
    _own(os.path.join(vault_dir, name + SUFFIX), "auth", 0o600)
    out["vault"] = os.path.join(vault_dir, name + SUFFIX)
    if verifier if verifier is not None else bool(known.get("verifier")):
        vdir = os.path.join(R, "verifier")
        sdir = os.path.join(vdir, "secrets")
        prepared = os.path.lexists(os.path.join(R, "auth"))
        safefs.ensure_dir(vdir, 0o700) if not prepared else None
        safefs.ensure_dir(sdir, 0o700)
        if not prepared:                                          # run directory not prepared yet: own what we made
            _own(vdir, "verifier", 0o700)
        _own(sdir, "verifier", 0o700)
        path = os.path.join(sdir, name)
        who = _ids("verifier")
        safefs.write_atomic(path, value.encode(), 0o600, uid=who[0] if who else None, gid=who[1] if who else None)
        out["verifier_copy"] = path
    return out


def verifier_secret(run_dir: str, name: str) -> str:
    """Read the verifier's own copy. Refuses a file other users can read or that another uid owns."""
    path = os.path.join(os.path.abspath(run_dir), "verifier", "secrets", name)
    st = os.stat(path)
    if stat.S_IMODE(st.st_mode) & 0o077:
        raise PermissionError(f"{path} is readable by other users (mode {stat.S_IMODE(st.st_mode):o})")
    if st.st_uid != os.geteuid() and os.geteuid() != 0:
        raise PermissionError(f"{path} belongs to uid {st.st_uid}, not to this process")
    with open(path) as f:
        return f.read().strip()


def listing(run_dir: str) -> List[Dict[str, Any]]:
    R = os.path.abspath(run_dir)
    rows = SecretBroker(os.path.join(R, "vault")).names() if os.path.isdir(os.path.join(R, "vault")) else []
    sdir = os.path.join(R, "verifier", "secrets")
    copies = set(os.listdir(sdir)) if os.path.isdir(sdir) else set()
    for r in rows:
        r["verifier_copy"] = r["name"] in copies
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="cam_live.harness.credentials", description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["set", "list", "revoke"])
    ap.add_argument("name", nargs="?", default="")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dest", action="append", default=[], help="destination pattern the secret may be sent to")
    ap.add_argument("--scope", action="append", default=[])
    ap.add_argument("--from-file", default="", help="read the value from a root-only (0600) file")
    ap.add_argument("--verifier", dest="verifier", action="store_true", default=None)
    ap.add_argument("--no-verifier", dest="verifier", action="store_false")
    ap.add_argument("--note", default="")
    ap.add_argument("--op", action="append", default=[], help="operation (glob) the secret may be released for")
    ap.add_argument("--expires-hours", type=float, default=None, help="lifetime of the credential")
    ap.add_argument("--generate", action="store_true",
                    help="generate a random value here instead of reading one (it is never printed)")
    a = ap.parse_args(argv)
    if a.command == "list":
        print(json.dumps(listing(a.run_dir), indent=1, sort_keys=True))
        return 0
    if not a.name:
        ap.error("a credential name is required")
    if a.command == "revoke":
        hit = SecretBroker(os.path.join(os.path.abspath(a.run_dir), "vault")).revoke(f"secret:{a.name}", time.time())
        copy = os.path.join(os.path.abspath(a.run_dir), "verifier", "secrets", a.name)
        from . import safefs
        try:
            safefs.unlink(copy)          # relative to the directory fd, no symlink followed (G2-harness R4-m2)
        except FileNotFoundError:
            pass
        print(json.dumps({"revoked": hit}))
        return 0
    if a.generate and a.from_file:
        ap.error("--generate and --from-file exclude each other")
    try:
        value = generate_value() if a.generate else read_value(a.from_file)
        out = store(a.run_dir, a.name, value, dest=a.dest, scopes=a.scope, verifier=a.verifier,
                    by=getpass.getuser(), note=a.note, expires_hours=a.expires_hours, operations=a.op)
        if a.generate:
            out["generated"] = True
    except (ValueError, PermissionError, OSError) as e:
        print(f"not stored: {e}", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
