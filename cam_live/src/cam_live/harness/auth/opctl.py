"""Operator command line for the authorization service (run as root in separated mode).

    python -m cam_live.harness.auth.opctl --run-dir R approvals
    python -m cam_live.harness.auth.opctl --run-dir R approve <approval_id> [--note ...]
    python -m cam_live.harness.auth.opctl --run-dir R reject <approval_id> [--note ...]
    python -m cam_live.harness.auth.opctl --run-dir R stop "<reason>"        # works with the service down too
    python -m cam_live.harness.auth.opctl --run-dir R reset-stop | pause "<reason>" | resume
    python -m cam_live.harness.auth.opctl --run-dir R revoke <scope>
    python -m cam_live.harness.auth.opctl --run-dir R reload-policy [path]
    python -m cam_live.harness.auth.opctl --run-dir R limits | secrets | audit-tail [n] | verify-audit
    python -m cam_live.harness.auth.opctl --run-dir R put-secret <name> --dest <pattern> [--scope s]   # value on stdin
    python -m cam_live.harness.auth.opctl init-operator-key --private K --public P
    python -m cam_live.harness.auth.opctl sign-live <record.json> --private K > signed.json
    python -m cam_live.harness.auth.opctl --run-dir R register-live <signed.json>
    python -m cam_live.harness.auth.opctl --run-dir R local-service add 127.0.0.1:18600 [--note ...] | remove <host:port> | list
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import time
from typing import List, Optional

from . import liveauth
from .audit import verify_audit_log
from .client import connect
from .stop import StopSwitch


def _who() -> str:
    try:
        return getpass.getuser()
    except (KeyError, OSError):
        return f"uid{os.getuid()}"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="cam_live.harness.auth.opctl")
    ap.add_argument("--run-dir", default="")
    ap.add_argument("command")
    ap.add_argument("args", nargs="*")
    ap.add_argument("--note", default="")
    ap.add_argument("--dest", action="append", default=[])
    ap.add_argument("--scope", action="append", default=[])
    ap.add_argument("--private", default="")
    ap.add_argument("--public", default="")
    a = ap.parse_args(argv)
    cmd, args, by = a.command, a.args, _who()

    def out(obj: object) -> int:
        print(json.dumps(obj, indent=1, sort_keys=True, default=str))
        return 0

    if cmd == "init-operator-key":
        liveauth.generate_operator_key(a.private, a.public)
        return out({"private": a.private, "public": a.public})
    if cmd == "sign-live":
        with open(args[0]) as f:
            return out(liveauth.sign_live_authorization(json.load(f), a.private))
    if not a.run_dir:
        ap.error("--run-dir is required for this command")
    sock = os.path.join(a.run_dir, "sock", "auth.sock")
    if cmd == "stop" and not os.path.exists(sock):
        rec = StopSwitch(os.path.join(a.run_dir, "auth")).engage(args[0] if args else "operator stop", by, time.time())
        return out({"stopped": True, "offline": True, "flag": rec})
    if cmd == "verify-audit" and not os.path.exists(sock):
        key_path = os.path.join(a.run_dir, "auth", "keys", "audit.key")
        key = open(key_path, "rb").read() if os.path.exists(key_path) else None
        v = verify_audit_log(os.path.join(a.run_dir, "audit.jsonl"), key=key,
                             head_path=os.path.join(a.run_dir, "auth", "audit.head"))
        out({"ok": v.ok, "entries": v.entries, "head_hash": v.head_hash, "errors": v.errors, "offline": True})
        return 0 if v.ok else 2
    c = connect(a.run_dir)
    if cmd == "approvals":
        return out(c.approvals())
    if cmd in ("approve", "reject"):
        return out(c.resolve_approval(args[0], cmd == "approve", by, a.note).to_dict())
    if cmd == "stop":
        c.emergency_stop(args[0] if args else "operator stop", by)
        return out({"stopped": c.stopped()})
    if cmd == "reset-stop":
        return out({"was_engaged": c.reset_stop(by), "stopped": c.stopped()})
    if cmd == "pause":
        c.pause(args[0] if args else "operator pause", by)
        return out({"paused": c.paused()})
    if cmd == "resume":
        return out({"was_paused": c.resume(by)})
    if cmd == "revoke":
        c.revoke(args[0], by)
        return out({"revoked": args[0]})
    if cmd == "reload-policy":
        r = c.reload_policy(by, args[0] if args else None)
        out(r)
        return 0 if r["valid"] else 2
    if cmd == "limits":
        return out(c.limits_state())
    if cmd == "secrets":
        return out(c.secret_names())
    if cmd == "audit-tail":
        return out(c.audit_tail(int(args[0]) if args else 20))
    if cmd == "verify-audit":
        v = c.verify_audit()
        out(v)
        return 0 if v["ok"] else 2
    if cmd == "put-secret":
        value = sys.stdin.read().rstrip("\n")
        return out(c.put_secret(args[0], value, a.dest, a.scope, by, a.note))
    if cmd == "register-live":
        with open(args[0]) as f:
            return out(c.register_live_authorization(json.load(f), by))
    if cmd == "local-service":
        sub = args[0] if args else "list"
        if sub == "add" and len(args) == 2:
            return out(c.register_local_service(args[1], by, a.note))
        if sub == "remove" and len(args) == 2:
            return out({"removed": c.unregister_local_service(args[1], by)})
        if sub == "list":
            return out(c.local_services())
        ap.error("local-service add <host:port> | remove <host:port> | list")
    ap.error(f"unknown command {cmd!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
