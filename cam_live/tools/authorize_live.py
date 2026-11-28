#!/usr/bin/env python
"""Operator tool: write and sign the live authorization record for one task file.

Live operation never starts without this record (``cam-live run --task T --authorization signed.json``). The
record is bound to the exact content of the task file (its hash), to the accounts and destinations listed here, to
the budgets of the task's ``limits`` and to a time window. It is signed with the operator's private key; the
authorization service holds only the public key and re-checks the signature, the hash and the window on every
decision. Signing is the operator's explicit act: the tool prints what is being authorized and asks for the
objective id to be typed, unless ``--yes`` is given.

    python cam_live/tools/authorize_live.py --task cam_live/tasks/discord_100.yaml \\
        --private /root/.config/camlive/operator_ed25519 --hours 168 --out /root/.config/camlive/discord_100.signed.json

Destinations: ``--owned`` (CAM may write there), ``--read`` (CAM may read there; default: the owned ones),
``--preauthorize`` (publication there needs no per-item approval). Discord destinations of the task
(``account:discord:<guild>``, ``channel:discord:*``) are added to ``owned`` automatically.

Standing authorization (operator amendment 2026-11-28, "OPERATIONAL AUTONOMY"; docs/AUTONOMY.md):
``--standing routine-reversible-1`` pre-authorizes routine, reversible operations inside CAM's designated
infrastructure -- the task's Discord server, the task's owned sites, ``--hosting`` (operator-provided hosting or
deployment API) and ``--integration`` (an approved integration's API) -- so they need no per-item approval, and lets
generated code reach ``--code-host`` hosts (e.g. package registries) without a per-job approval. The policy's
retained-approval rules (irreversible operations, new external permissions, messages to people), spending, data
use and the hard platform constraints keep their approval or denial; the summary lists them.

    python cam_live/tools/authorize_live.py --task T --private K --hours 48 --out signed.json \
        --standing routine-reversible-1 --read 'https://*' --hosting 'https://api.host.example/*' \
        --code-host pypi.org --code-host files.pythonhosted.org
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from cam_live.harness.auth import destinations as D  # noqa: E402
from cam_live.harness.auth import liveauth  # noqa: E402
from cam_live.harness.auth.schema import PolicyError, load_policy, run_policy_path  # noqa: E402
from cam_live.runtime.tasks import TaskValidationError, load_task  # noqa: E402


class StandingError(ValueError):
    pass


def _specific(pattern: str) -> bool:
    """A destination pattern that names a place (a host, optionally '*.' + a domain, or a channel / account class of
    one platform), not every site."""
    try:
        p = D.parse_pattern(pattern)
    except D.DestinationError:
        return False
    if p.kind in ("account", "channel"):
        return bool(p.platform) and "*" not in p.platform and "?" not in p.platform
    if p.kind != "url":
        return False
    host = p.host[2:] if p.host.startswith("*.") else p.host
    if not host or "*" in host or "?" in host:
        return False
    return "." in host or host == "localhost" or bool(p.port)


def _add(lst, *items):
    for x in items:
        if x and x not in lst:
            lst.append(x)


def task_sites(task) -> list:
    """Base URLs of the sites the task file says the operator owns (``owned_site``, ``owned_sites[].base_url``)."""
    res = dict(task.resources or {})
    urls = [res.get("owned_site")] + [s.get("base_url") for s in (res.get("owned_sites") or []) if isinstance(s, dict)]
    out = []
    for u in urls:
        if isinstance(u, str) and u and u != "REQUIRED":
            _add(out, u.rstrip("/") + "/*")
    return out


def build_record(task, *, hours: float, owned, read, preauthorized, accounts, issued_by: str, note: str = "",
                 now: float = 0.0, standing: str = "", hosting=(), integrations=(), code_hosts=()) -> dict:
    now = now or time.time()
    res = dict(task.resources or {})
    d = dict(res.get("discord") or {})
    guild = str(d.get("guild_id") or res.get("discord_guild_id") or "")
    owned, accounts, preauthorized = list(owned), list(accounts), list(preauthorized)
    if guild:
        for p in (f"account:discord:{guild}", "channel:discord:*"):
            if p not in owned:
                owned.append(p)
        app = str(d.get("application_id") or "")
        if app and f"account:discord:{app}" not in accounts:
            accounts.append(f"account:discord:{app}")
    if not standing and (hosting or integrations or code_hosts):
        raise StandingError("--hosting / --integration / --code-host belong to a standing profile (--standing)")
    if standing:
        designated = list(hosting) + list(integrations) + task_sites(task)
        broad = [x for x in designated + preauthorized if not _specific(x)]
        if broad:
            raise StandingError(f"a standing profile pre-authorizes named places only, not {broad}")
        _add(owned, *designated)
        if guild:
            _add(preauthorized, f"account:discord:{guild}", "channel:discord:*")
        _add(preauthorized, *designated)
        read = list(read) or list(owned)
        _add(read, *code_hosts)
    lim = task.limits
    budgets = {"money": float(lim.money), "money_per_action": float(getattr(lim, "money_per_action", 0) or 0),
               "money_per_day": float(getattr(lim, "money_per_day", 0) or 0),
               "compute_s": float(getattr(lim, "compute_s", 0) or 0),
               "max_external_actions": int(lim.max_external_actions), "max_decisions": int(lim.max_decisions),
               "duration_s": float(lim.duration_s)}
    rec = {"record_id": f"live-{task.objective_id}-{int(now)}", "task_id": task.objective_id,
           "task_hash": task.content_hash(), "accounts": accounts, "owned": owned,
           "read": list(read) or list(owned), "preauthorized": list(preauthorized), "interactive": [],
           "opted_in_recipients": [], "budgets": budgets, "not_before": float(now),
           "not_after": float(now + hours * 3600.0), "issued_by": issued_by, "issued_at": float(now)}
    if note:
        rec["note"] = note
    if standing:
        rec["standing"] = {"profile": standing, "code_network": list(code_hosts)}
    liveauth.validate_shape(rec)
    return rec


def retained_lines(policy_path: str = "") -> list:
    """What keeps the operator's approval under a standing profile, read from the policy file the run will use."""
    try:
        pol = load_policy(policy_path or run_policy_path())
    except PolicyError as e:
        return [f"  (the policy file cannot be read: {e})"]
    out = [f"  - {r.id} ({r.cls}): {r.reason}" for r in pol.standing.retained]
    out += ["  - P-SPEND-APPROVAL: any spending of real money",
            "  - data_use: member lists, personal data, message content only with the task's data permissions",
            "  - input on sites the operator does not own (P-INTERACT-APPROVAL), e.g. account registration forms",
            "  - never, whatever is approved: X-HUMAN-VERIFICATION, X-USER-ACCOUNT, X-PLATFORM-ADAPTER"]
    return out


def summary(task, rec: dict) -> str:
    t = lambda x: time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(x))  # noqa: E731
    lines = [f"LIVE AUTHORIZATION for objective {task.objective_id!r} (mode {task.mode})",
             f"  objective : {task.objective}", f"  task hash : {rec['task_hash']}",
             f"  window    : {t(rec['not_before'])}  ->  {t(rec['not_after'])}",
             f"  interfaces: {', '.join(task.interfaces)}",
             f"  accounts  : {', '.join(rec['accounts']) or '(none)'}",
             f"  may write : {', '.join(rec['owned']) or '(nowhere)'}",
             f"  may read  : {', '.join(rec['read']) or '(nowhere)'}",
             f"  publish without per-item approval: {', '.join(rec['preauthorized']) or '(nowhere: every publication asks)'}",
             "  budgets   : " + ", ".join(f"{k}={v:g}" for k, v in rec["budgets"].items())]
    st = rec.get("standing")
    if st:
        lines += [f"  STANDING AUTHORIZATION, profile {st['profile']}: routine operations on the pre-authorized "
                  f"destinations above need no per-item approval",
                  f"  generated code may reach without a per-job approval: {', '.join(st.get('code_network') or []) or '(nothing)'}",
                  "  approval is kept for:"] + retained_lines()
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--task", required=True)
    ap.add_argument("--private", required=True, help="operator Ed25519 private key (stays with the operator)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hours", type=float, required=True, help="length of the authorization window from now")
    ap.add_argument("--owned", action="append", default=[])
    ap.add_argument("--read", action="append", default=[])
    ap.add_argument("--preauthorize", action="append", default=[])
    ap.add_argument("--account", action="append", default=[])
    ap.add_argument("--issued-by", default=os.environ.get("USER", "operator"))
    ap.add_argument("--note", default="")
    ap.add_argument("--standing", default="", help="standing authorization profile of the policy "
                                                   "(routine-reversible-1)")
    ap.add_argument("--hosting", action="append", default=[], help="operator-provided hosting / deployment API "
                                                                   "(owned and pre-authorized; with --standing)")
    ap.add_argument("--integration", action="append", default=[], help="an approved integration's API (owned and "
                                                                       "pre-authorized; with --standing)")
    ap.add_argument("--code-host", action="append", default=[], help="host generated code may reach without a "
                                                                     "per-job approval (with --standing)")
    ap.add_argument("--yes", action="store_true", help="do not ask for the typed confirmation")
    a = ap.parse_args(argv)
    try:
        task = load_task(a.task, for_run=True)
    except TaskValidationError as e:
        print("the task file cannot be authorized as it stands:\n  - " + "\n  - ".join(e.problems), file=sys.stderr)
        return 3
    if task.mode != "live":
        print(f"task mode is {task.mode!r}; only a live task needs (and can use) this record", file=sys.stderr)
        return 3
    if a.hours <= 0:
        print("--hours must be positive", file=sys.stderr)
        return 2
    if a.standing:
        try:
            profiles = load_policy(run_policy_path()).standing.profiles
        except PolicyError as e:
            print(f"the run policy cannot be read: {e}", file=sys.stderr)
            return 3
        if a.standing not in profiles:
            print(f"--standing {a.standing!r} is not a profile of the run policy ({', '.join(profiles) or 'none'})",
                  file=sys.stderr)
            return 2
    try:
        rec = build_record(task, hours=a.hours, owned=a.owned, read=a.read, preauthorized=a.preauthorize,
                           accounts=a.account, issued_by=a.issued_by, note=a.note, standing=a.standing,
                           hosting=a.hosting, integrations=a.integration, code_hosts=a.code_host)
    except (StandingError, liveauth.LiveAuthorizationError) as e:
        print(f"not signed: {e}", file=sys.stderr)
        return 2
    print(summary(task, rec))
    if not a.yes:
        typed = input(f"type the objective id ({task.objective_id}) to sign this authorization: ").strip()
        if typed != task.objective_id:
            print("not signed", file=sys.stderr)
            return 4
    signed = liveauth.sign_live_authorization(rec, a.private)
    fd = os.open(a.out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(signed, f, indent=1, sort_keys=True)
    print(f"signed record written to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
