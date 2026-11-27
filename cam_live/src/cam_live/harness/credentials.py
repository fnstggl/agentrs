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
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
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


def _ids(role: str) -> Optional[tuple]:
    if os.geteuid() != 0 or not users_exist():
        return None
    from .auth.launcher import _ids as ids
    return ids(role)


def _own(path: str, role: str, mode: int) -> None:
    os.chmod(path, mode)
    who = _ids(role)
    if who is not None:
        os.chown(path, who[0], who[1])


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
          verifier: Optional[bool] = None, by: str = "operator", note: str = "") -> Dict[str, Any]:
    """Store ``value`` in the run's vault (and, where needed, as the verifier's own copy). Returns names only."""
    known = KNOWN.get(name, {})
    dest = list(dest) or list(known.get("dest", []))
    if not dest:
        raise ValueError("a credential must be bound to at least one destination (--dest)")
    if len(value) < int(known.get("min_len", 8)):
        raise ValueError(f"that value is too short to be a {name}")
    if any(c.isspace() for c in value):
        raise ValueError("the value contains whitespace; paste the credential alone")
    R = os.path.abspath(run_dir)
    vault_dir = os.path.join(R, "vault")
    os.makedirs(vault_dir, mode=0o700, exist_ok=True)
    out = SecretBroker(vault_dir).put(name, value, dest, list(scopes) or list(known.get("scopes", [])), by=by,
                                      note=note, ts=time.time())
    _own(vault_dir, "auth", 0o700)
    _own(os.path.join(vault_dir, name + SUFFIX), "auth", 0o600)
    out["vault"] = os.path.join(vault_dir, name + SUFFIX)
    if verifier if verifier is not None else bool(known.get("verifier")):
        vdir = os.path.join(R, "verifier")
        sdir = os.path.join(vdir, "secrets")
        os.makedirs(sdir, mode=0o700, exist_ok=True)
        if not os.path.exists(os.path.join(R, "auth")):          # run directory not prepared yet: own what we made
            _own(vdir, "verifier", 0o700)
        _own(sdir, "verifier", 0o700)
        path = os.path.join(sdir, name)
        fd = os.open(path + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        try:
            os.write(fd, value.encode())
            os.fsync(fd)
        finally:
            os.close(fd)
        os.rename(path + ".tmp", path)
        _own(path, "verifier", 0o600)
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
    a = ap.parse_args(argv)
    if a.command == "list":
        print(json.dumps(listing(a.run_dir), indent=1, sort_keys=True))
        return 0
    if not a.name:
        ap.error("a credential name is required")
    if a.command == "revoke":
        hit = SecretBroker(os.path.join(os.path.abspath(a.run_dir), "vault")).revoke(f"secret:{a.name}", time.time())
        copy = os.path.join(os.path.abspath(a.run_dir), "verifier", "secrets", a.name)
        if os.path.exists(copy):
            os.unlink(copy)
        print(json.dumps({"revoked": hit}))
        return 0
    try:
        out = store(a.run_dir, a.name, read_value(a.from_file), dest=a.dest, scopes=a.scope, verifier=a.verifier,
                    by=getpass.getuser(), note=a.note)
    except (ValueError, PermissionError, OSError) as e:
        print(f"not stored: {e}", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
