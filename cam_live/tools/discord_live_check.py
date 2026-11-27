#!/usr/bin/env python
"""Integration check of the Discord adapter through the real environment and the real authorization service.

Every step is an ``ActionSpec`` that goes ``env.validate_action -> policy.authorize(action, task, {}) ->
env.execute(action, decision)``: nothing here calls the adapter or Discord directly, and nothing runs without a
grant. The bot token is never read by this script: the executor obtains it from the run's vault at the moment of
each request, and the evidence file is checked against every secret of the vault before it is written.

Live (the operator runs this; it changes the server and undoes its changes)::

    python cam_live/tools/discord_live_check.py --run-dir R --task tasks/discord_check.yaml \\
        [--policy P] [--skip-writes] [--await-reply-s N] [--approve-wait-s N] [--out FILE]

    Prepared by the caller beforehand (see docs/ES/INTEGRATION.md, "Live check"):
      * the secret ``discord_bot_token`` in R's vault, bound to discord.com and gateway.discord.gg;
      * the operator public key in R/auth/keys/operator.pub and a signed live authorization record for the task
        file given by --task (``--print-record-template`` prints the record to sign for it);
      * a policy whose write rate for the server allows this check (``--write-policy FILE`` writes one: the default
        policy plus the ``rate_limits.per_destination`` entries shown by ``--print-policy-rule``), unless --skip-writes.

Staging (no request leaves the machine; starts the staging REST service and the staging Gateway)::

    python cam_live/tools/discord_live_check.py --staging [--live-policy] [--skip-writes] [--await-reply-s N] --out FILE

    ``--live-policy`` runs the same sequence under the policy's *live* mode: a task in mode ``live`` and an
    authorization record signed with an operator key generated for this run, against the staging services.

The sequence: get_identity, get_permissions, get_guild_settings, list_channels, get_guild_stats, list_invites;
then (unless --skip-writes) create_channel ``cam-integration-test``, send_message, get_channel_activity, an
optional wait for a human reply, edit_message, pin_message, unpin_message, create_invite, list_invites,
delete_invite, modify_channel, set_channel_permissions, create_role / modify_role / delete_role, modify_guild
(rename) and modify_guild (restore the original name, in a ``finally``); get_member_events twice, listen_events
for 10 s, get_audit_summary; delete_channel (in a ``finally``: the temporary channel never survives the script).

A cleanup step that meets a rate limit or a network fault is issued again after the stated wait (at most three
more times and ``--cleanup-wait-s`` in total); if it still fails, the script says what must be undone by hand.

Exit codes: 0 every step the operator's scope allows succeeded; 1 a step failed; 2 usage or setup error; 3 the bot
is not a member of the server; 4 the policy's write rate does not allow the sequence (nothing was written);
5 a step is waiting for the operator's approval; 6 a secret was found in the evidence (it is written redacted).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from cam_live import contracts as C  # noqa: E402
from cam_live.harness.auth import PolicyService  # noqa: E402
from cam_live.harness.auth import destinations as D  # noqa: E402
from cam_live.harness.auth import liveauth  # noqa: E402
from cam_live.harness.discord import LIVE_BASE_URL, DiscordExecutor, environment_of  # noqa: E402
from cam_live.harness.env import ComputerEnvironment  # noqa: E402
from cam_live.harness.http_api import ApiModality  # noqa: E402

CHANNEL_NAME = "cam-integration-test"
ROLE_NAME = "cam-integration-test-role"
RENAME_SUFFIX = " (integration test)"
MESSAGE = "CAM Live integration check: this message was posted by the bot and is removed with its channel."
MESSAGE_EDITED = "CAM Live integration check: edited by the bot; removed with its channel."
#: operations of the sequence that change the server (the default policy also counts discord.listen_events as one)
WRITES = ("create_channel", "send_message", "edit_message", "pin_message", "unpin_message", "create_invite",
          "delete_invite", "modify_channel", "set_channel_permissions", "create_role", "modify_role", "delete_role",
          "modify_guild", "modify_guild", "delete_channel", "listen_events")
SCOPE_MARK = "outside the operator's permission scope"


def load_task(path: str) -> C.TaskSpec:
    """A TaskSpec from a YAML task file (the runtime's loader: structure and no-strategy rules) or from the JSON
    of ``TaskSpec.to_dict()``."""
    if path.endswith(".json"):
        with open(path) as f:
            return C.TaskSpec.from_dict(json.load(f))
    from cam_live.runtime.tasks import load_task as load_yaml
    return load_yaml(path, for_run=False)


def record_template(task: C.TaskSpec, guild_id: str) -> Dict[str, Any]:
    """The live authorization record this check needs for ``task`` (unsigned). The operator reviews, signs and
    registers it: ``opctl sign-live`` / ``opctl register-live``."""
    now = time.time()
    return {"record_id": f"discord-live-check-{int(now)}", "task_id": task.objective_id, "task_hash": task.content_hash(),
            "accounts": [f"account:discord:{guild_id}"], "owned": ["channel:discord:*"],
            "read": ["discord.com", "gateway.discord.gg"],
            "preauthorized": [f"account:discord:{guild_id}", "channel:discord:*"], "interactive": [],
            "opted_in_recipients": [],
            "budgets": {"money": 0, "money_per_action": 0, "money_per_day": 0, "compute_s": 0,
                        "max_external_actions": 200, "max_decisions": 200, "duration_s": 3600},
            "not_before": now - 60, "not_after": now + 3600, "issued_by": "operator", "issued_at": now,
            "note": "integration check of the Discord adapter (tools/discord_live_check.py)"}


def policy_rule(guild_id: str) -> List[Dict[str, Any]]:
    """``rate_limits.per_destination`` entries that let the sequence's writes through."""
    return [{"match": f"account:discord:{guild_id}", "count": 60, "window_s": 3600, "effects": ["external_write"]},
            {"match": "channel:discord:*", "count": 60, "window_s": 3600, "effects": ["external_write"]}]


def write_policy(path: str, guild_id: str) -> str:
    """A copy of the default policy with the write rate of this server raised for the check."""
    import yaml
    from cam_live.harness.auth.schema import DEFAULT_POLICY_PATH
    with open(DEFAULT_POLICY_PATH) as f:
        pol = yaml.safe_load(f)
    pol["policy_id"] = "default-1+discord-live-check"
    pol["rate_limits"]["per_destination"] = list(pol["rate_limits"].get("per_destination") or []) + policy_rule(guild_id)
    with open(path, "w") as f:
        yaml.safe_dump(pol, f, sort_keys=False)
    return path


class Check:
    def __init__(self, policy: PolicyService, env: ComputerEnvironment, discord: DiscordExecutor, task: C.TaskSpec,
                 *, approve_wait_s: float = 0.0, cleanup_wait_s: float = 900.0, echo: Callable[[str], None] = print,
                 sleep: Callable[[float], None] = time.sleep):
        self.policy, self.env, self.discord, self.task = policy, env, discord, task
        self.guild_id = discord.guild_id
        self.approve_wait_s, self.cleanup_wait_s, self.echo, self.sleep = approve_wait_s, cleanup_wait_s, echo, sleep
        self.steps: List[Dict[str, Any]] = []

    def cleanup(self, name: str, args: Dict[str, Any], note: str) -> Dict[str, Any]:
        """A step that undoes a change. A rate limit or a network fault does not end it: the identical step is
        issued again after the stated wait (at most ``cleanup_wait_s`` in total, at most three more times)."""
        waited = 0.0
        rec = self.step(name, args, note=note)
        for _ in range(3):
            if rec["outcome"] == "ok" or rec.get("failure_class") not in (C.FailureClass.RESOURCE_LIMIT,
                                                                           C.FailureClass.TRANSIENT_NETWORK):
                break
            pause = float(rec.get("retry_after_s") or 5.0) + 1.0
            if waited + pause > self.cleanup_wait_s:
                self.echo(f"   {note}: the stated wait of {pause:.0f} s exceeds what this check waits for a cleanup")
                break
            self.echo(f"   {note}: not done ({rec['failure_class']}); trying again in {pause:.0f} s")
            self.sleep(pause)
            waited += pause
            rec = self.step(name, args, note=note + " (again)")
        return rec

    # ---- one step -------------------------------------------------------------------------------------------------
    def step(self, name: str, args: Optional[Dict[str, Any]] = None, *, note: str = "", timeout_s: float = 60.0,
             ) -> Dict[str, Any]:
        op = f"discord.{name}"
        action = C.ActionSpec(action_id=C.new_id("lc"), operation=op, modality=C.Modality.API,
                              arguments=dict(args or {}, guild_id=self.guild_id), timeout_s=timeout_s,
                              provenance="builtin", rationale="integration check")
        rec: Dict[str, Any] = {"n": len(self.steps) + 1, "operation": op, "arguments": dict(action.arguments),
                               "note": note, "validation_errors": [], "verdict": "", "policy_ids": [], "reasons": [],
                               "status": "", "failure_class": "", "failure_detail": "", "quantities": {},
                               "rate_limit": {}, "text": "", "artifacts": {}, "data": {}, "duration_s": 0.0}
        t0 = time.time()
        errs = self.env.validate_action(action)
        if errs:
            rec.update(validation_errors=errs, status="not_executed", failure_detail="; ".join(errs),
                       outcome="scope_refusal" if any(SCOPE_MARK in e for e in errs) else "invalid")
            return self._done(rec, t0)
        decision = self.policy.authorize(action, self.task, {})
        waited = 0.0
        while decision.verdict == C.AuthVerdict.NEEDS_APPROVAL and waited < self.approve_wait_s:
            if not waited:
                self.echo(f"   waiting up to {self.approve_wait_s:g} s for approval {decision.approval_id} "
                          f"({'; '.join(decision.reasons)[:160]})")
            time.sleep(3.0)
            waited += 3.0
            decision = self.policy.authorize(action, self.task, {})        # the identical action, as documented
        rec.update(verdict=decision.verdict, policy_ids=list(decision.policy_ids), reasons=list(decision.reasons),
                   approval_id=decision.approval_id)
        res = self.env.execute(action, decision)
        obs = next((o for o in res.observations if o.kind != "error"), None)
        q = {x.name: x.value for o in res.observations for x in o.quantities}
        rec.update(status=res.status, failure_class=res.failure_class, failure_detail=res.failure_detail,
                   quantities={k: v for k, v in q.items() if k.startswith("discord.") and not k.startswith("discord.ratelimit.")},
                   rate_limit={k.rsplit(".", 1)[-1]: v for k, v in q.items() if k.startswith("discord.ratelimit.")},
                   text=(obs.text if obs is not None else "")[:400], artifacts=dict(res.artifacts),
                   data=dict(obs.data) if obs is not None else {}, retries=res.retries,
                   checked_ops=[o.op for o in res.ops], retry_after_s=q.get("exec.failure.retry_after_s"))
        if res.status == C.ResultStatus.OK:
            rec["outcome"] = "ok"
        elif res.status == C.ResultStatus.NEEDS_APPROVAL:
            rec["outcome"] = "needs_approval"
        elif SCOPE_MARK in res.failure_detail:
            rec["outcome"] = "scope_refusal"
        elif decision.verdict == C.AuthVerdict.DENY:
            rec["outcome"] = "authorization_denied"
        elif res.failure_class == C.FailureClass.MISSING_PERMISSION:
            rec["outcome"] = "permission_refusal"
        else:
            rec["outcome"] = "failed"
        return self._done(rec, t0)

    def _done(self, rec: Dict[str, Any], t0: float) -> Dict[str, Any]:
        rec["duration_s"] = round(time.time() - t0, 3)
        self.steps.append(rec)
        tail = rec["failure_detail"] or rec["text"]
        self.echo(f"{rec['n']:02d} {rec['operation']:33s} auth={rec['verdict'] or '-':14s} -> "
                  f"{rec['status']:14s} {rec['outcome']:20s} {tail[:150]}")
        return rec

    def skip(self, name: str, why: str) -> None:
        rec = {"n": len(self.steps) + 1, "operation": f"discord.{name}", "status": "skipped", "outcome": "skipped",
               "failure_detail": why, "verdict": "", "text": "", "duration_s": 0.0}
        self.steps.append(rec)
        self.echo(f"{rec['n']:02d} {rec['operation']:33s} auth={'-':14s} -> {'skipped':14s} {'skipped':20s} {why}")

    # ---- the policy's write rate ----------------------------------------------------------------------------------
    def write_budget(self) -> Dict[str, Any]:
        """How many external writes on the server the policy still allows now, and how many the sequence makes.
        Read from the policy the service loaded; ``known`` is false when that cannot be determined."""
        need = len(WRITES)
        try:
            pol = self.policy.policy
            dest = D.parse_destination(f"account:discord:{self.guild_id}")
            now = self.policy.clock.now()
            rule = next((r for r in pol.rate_rules if r.match is not None and r.match.matches(dest)
                         and (not r.effects or "external_write" in r.effects)), None)
            tag, rule = ("s", rule) if rule is not None else ("w", pol.rate_write)
            used = self.policy.ledger.rate_count(f"{tag}:{self.task.objective_id}:{dest.rate_key}", rule.window_s, now)
            return {"known": True, "needed": need, "allowed": int(rule.count), "window_s": rule.window_s,
                    "used": int(used), "enough": rule.count - used >= need}
        except Exception as e:  # noqa: BLE001 — the check is advisory; the policy itself still enforces its limits
            return {"known": False, "needed": need, "error": type(e).__name__, "enough": True}

    # ---- the sequence ---------------------------------------------------------------------------------------------
    def run(self, *, skip_writes: bool = False, await_reply_s: float = 0.0, listen_s: int = 10,
            during_reply_wait: Optional[Callable[[str], None]] = None,
            between_readings: Optional[Callable[[], None]] = None,
            during_listen: Optional[Callable[[], None]] = None) -> int:
        ident = self.step("get_identity")
        if ident["outcome"] != "ok":
            self.echo("stopped: the identity of the bot could not be read; nothing else was tried")
            return 5 if ident["outcome"] == "needs_approval" else 1
        if not ident["data"].get("in_guild"):
            self.echo(f"stopped: the bot is not a member of server {self.guild_id}. The server owner must add the "
                      f"application to it; nothing else was tried.")
            return 3
        self.step("get_permissions")
        settings = self.step("get_guild_settings")
        original_name = (settings["data"].get("settings") or {}).get("name") if settings["outcome"] == "ok" else None
        self.step("list_channels")
        self.step("get_guild_stats")
        self.step("list_invites")
        channel_id = ""
        if not skip_writes:
            budget = self.write_budget()
            self.budget = budget
            if not budget["enough"]:
                self.echo(f"stopped before any write: the policy allows {budget['allowed']} external writes per "
                          f"{budget['window_s']:g} s on account:discord:{self.guild_id} ({budget['used']} used); this "
                          f"check makes {budget['needed']}. Start the service with a policy that has the "
                          f"rate_limits.per_destination entries printed by --print-policy-rule (or pass --policy), "
                          f"or use --skip-writes.")
                return 4
        try:
            if not skip_writes:
                made = self.step("create_channel", {"name": CHANNEL_NAME, "type": "text"})
                channel_id = made["artifacts"].get("channel_id", "") if made["outcome"] == "ok" else ""
            if channel_id:
                self._channel_steps(channel_id, await_reply_s, during_reply_wait)
            elif not skip_writes:
                self.skip("send_message ... set_channel_permissions", "no temporary channel was created")
            if not skip_writes:
                self._role_steps()
                self._rename_steps(original_name)
            self.step("get_member_events", note="first reading")
            if between_readings is not None:
                between_readings()
            self.step("get_member_events", note="second reading")
            hook = None
            if during_listen is not None:
                hook = threading.Thread(target=during_listen, daemon=True)
                hook.start()
            self.step("listen_events", {"seconds": int(listen_s)}, timeout_s=float(listen_s) + 35.0)
            if hook is not None:
                hook.join(timeout=5.0)
            self.step("get_audit_summary", {"limit": 50})
        finally:
            if channel_id:
                gone = self.cleanup("delete_channel", {"channel_id": channel_id}, "cleanup: the temporary channel")
                if gone["outcome"] != "ok":
                    self.echo(f"!! the temporary channel {channel_id} ({CHANNEL_NAME}) was NOT deleted: delete it by hand")
        return self.exit_code()

    def _channel_steps(self, cid: str, await_reply_s: float, during_reply_wait: Optional[Callable[[str], None]]) -> None:
        sent = self.step("send_message", {"channel_id": cid, "content": MESSAGE})
        mid = sent["artifacts"].get("message_id", "") if sent["outcome"] == "ok" else ""
        act = self.step("get_channel_activity", {"channel_id": cid, "limit": 50}, note="reads back the bot's message")
        if act["outcome"] == "ok":
            act["bot_message_read_back"] = act["quantities"].get("discord.channel.bot_messages", 0) >= 1
        if await_reply_s > 0:
            self._await_reply(cid, await_reply_s, during_reply_wait)
        if mid:
            self.step("edit_message", {"channel_id": cid, "message_id": mid, "content": MESSAGE_EDITED})
            self.step("pin_message", {"channel_id": cid, "message_id": mid})
            self.step("unpin_message", {"channel_id": cid, "message_id": mid})
        else:
            self.skip("edit_message, pin_message, unpin_message", "no message was posted")
        inv = self.step("create_invite", {"channel_id": cid, "max_age_s": 600, "max_uses": 1, "unique": True})
        code = inv["artifacts"].get("invite_code", "") if inv["outcome"] == "ok" else ""
        self.step("list_invites")
        if code:
            gone = self.step("delete_invite", {"code": code})
            if gone["outcome"] != "ok":
                self.echo(f"   the invite {code} was not deleted; it expires in 600 s and goes with its channel")
        self.step("modify_channel", {"channel_id": cid, "topic": "temporary channel of the CAM Live integration check"})
        self.step("set_channel_permissions", {"channel_id": cid, "role_id": "everyone", "deny": ["SEND_MESSAGES"]})

    def _await_reply(self, cid: str, seconds: float, hook: Optional[Callable[[str], None]]) -> None:
        self.echo(f"   waiting up to {seconds:g} s for a reply by a person in channel {cid} ({CHANNEL_NAME}) ...")
        thread = None
        if hook is not None:
            thread = threading.Thread(target=hook, args=(cid,), daemon=True)
            thread.start()
        end = time.time() + seconds
        arrived, polls = False, 0
        while True:
            poll = self.step("get_channel_activity", {"channel_id": cid, "limit": 50}, note="poll for a human reply")
            polls += 1
            if poll["outcome"] != "ok":
                break
            if poll["quantities"].get("discord.channel.human_messages", 0) >= 1:
                arrived = True
                break
            remaining = end - time.time()
            if remaining <= 0:
                break
            time.sleep(min(5.0, remaining))                 # every 5 s, and once more when the time is up
        if thread is not None:
            thread.join(timeout=5.0)
        self.reply = {"waited_for_s": seconds, "polls": polls, "human_reply_arrived": arrived}
        self.echo(f"   human reply arrived: {'yes' if arrived else 'no'} ({polls} polls)")

    def _role_steps(self) -> None:
        made = self.step("create_role", {"name": ROLE_NAME, "permissions": []})
        rid = made["artifacts"].get("role_id", "") if made["outcome"] == "ok" else ""
        if not rid:
            self.skip("modify_role, delete_role", "no temporary role was created")
            return
        try:
            self.step("modify_role", {"role_id": rid, "name": ROLE_NAME + "-renamed", "mentionable": False})
        finally:
            gone = self.cleanup("delete_role", {"role_id": rid}, "cleanup: the temporary role")
            if gone["outcome"] != "ok":
                self.echo(f"!! the temporary role {rid} ({ROLE_NAME}) was NOT deleted: delete it by hand")

    def _rename_steps(self, original: Optional[str]) -> None:
        if not original:
            self.skip("modify_guild", "the current server name is not known; the server is not renamed")
            return
        renamed = None
        try:
            renamed = self.step("modify_guild", {"name": (original + RENAME_SUFFIX)[:100]}, note="rename")
        finally:
            # restore unless it is certain that nothing was changed (refused before any request)
            untouched = renamed is not None and renamed["outcome"] in ("scope_refusal", "authorization_denied", "invalid",
                                                                      "needs_approval", "permission_refusal")
            if not untouched:
                back = self.cleanup("modify_guild", {"name": original}, "restore the original name")
                if back["outcome"] != "ok":
                    self.echo(f"!! the server name was NOT restored: set it back to {original!r} by hand")

    # ---- summary --------------------------------------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for s in self.steps:
            counts[s["outcome"]] = counts.get(s["outcome"], 0) + 1
        return {"steps": len(self.steps), "by_outcome": dict(sorted(counts.items())),
                "scope_refusals": [s["operation"] for s in self.steps if s["outcome"] == "scope_refusal"],
                "permission_refusals": [f"{s['operation']}: {s['failure_detail']}" for s in self.steps
                                        if s["outcome"] == "permission_refusal"],
                "not_ok": [f"{s['n']:02d} {s['operation']}: {s['outcome']}: {s.get('failure_detail', '')[:200]}"
                           for s in self.steps if s["outcome"] not in ("ok", "scope_refusal")]}

    def exit_code(self) -> int:
        # a step skipped because the step it depends on was refused or failed is counted through that step
        outcomes = [s["outcome"] for s in self.steps if s["outcome"] != "skipped"]
        if all(o in ("ok", "scope_refusal") for o in outcomes):
            return 0
        if all(o in ("ok", "scope_refusal", "needs_approval") for o in outcomes):
            return 5
        return 1


def build(run_dir: str, task: C.TaskSpec, *, policy_path: Optional[str], base_url: str, provenance: str,
          fsync: bool = True, prepare: Optional[Callable[[PolicyService], None]] = None):
    """The real stack: authorization service (in-process), computer environment, Discord executor. ``prepare``
    runs on the authorization service before the environment looks at the tools (staging: store the secret)."""
    policy = PolicyService(run_dir, policy_path, fsync=fsync)
    if policy.policy is None:
        raise SystemExit(f"the authorization policy does not load: {policy.policy_error}")
    if prepare is not None:
        prepare(policy)
    env = ComputerEnvironment(run_dir, policy, secrets=policy.secrets_handle(), provenance=provenance)
    names = lambda: {s["name"] for s in policy.secret_names() if not s["revoked"]}          # noqa: E731
    discord = DiscordExecutor.from_task(task, base_url=base_url, secrets=policy.secrets_handle(),
                                        credential_probe=lambda name: name in names(),
                                        state_dir=os.path.join(run_dir, "discord_state"))
    env.register(ApiModality([discord]))
    return policy, env, discord


def evidence(check: Check, policy: PolicyService, discord: DiscordExecutor, *, mode: str, exit_code: int,
             started: float, extra: Dict[str, Any]) -> Dict[str, Any]:
    out = {"tool": "cam_live/tools/discord_live_check.py", "mode": mode, "environment": discord.environment,
           "base_url": discord.base_url, "policy_mode": check.task.mode,
           "policy": {"id": policy.policy.policy_id if policy.policy else "", "sha256": policy.policy.sha256 if policy.policy else ""},
           "task": {"objective_id": check.task.objective_id, "content_hash": check.task.content_hash()},
           "guild_id": discord.guild_id, "scope": list(discord.scope), "started_at": started, "ended_at": time.time(),
           "exit_code": exit_code, "summary": check.summary(), "write_budget": getattr(check, "budget", None),
           "reply": getattr(check, "reply", None), "missing_capabilities": discord.missing_capabilities(),
           "steps": check.steps}
    out.update(extra)
    return out


def write_evidence(path: str, data: Dict[str, Any], policy: PolicyService, echo: Callable[[str], None]) -> bool:
    """Write the evidence after checking it against every secret of the vault. Returns whether it was free of
    secrets; what is written is scrubbed either way."""
    text = json.dumps(data, indent=1, sort_keys=True, default=str)
    found = policy.vault.find(text)
    clean = policy.scrub(text)
    token_free = not found and clean == text
    final = json.loads(clean)
    final["token_free"] = token_free
    final["secrets_checked"] = len(policy.secret_names())
    out = json.dumps(final, indent=1, sort_keys=True, default=str)
    assert not policy.vault.find(out), "a vault secret is still present after scrubbing"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(out + "\n")
    echo(f"evidence: {path} (free of vault secrets: {'yes' if token_free else 'NO - redacted: ' + ', '.join(found)})")
    return token_free


def run_live(a: argparse.Namespace) -> int:
    preparing = a.print_record_template or a.print_policy_rule or a.write_policy
    if not a.task or not (a.run_dir or preparing):
        print("live mode needs --run-dir and --task", file=sys.stderr)
        return 2
    task = load_task(a.task)
    probe = DiscordExecutor.from_task(task)
    guild_id, base_url = probe.guild_id, probe.base_url
    probe.close()
    if a.print_record_template or a.print_policy_rule or a.write_policy:
        if not guild_id:
            print("the task names no server (resources.discord.guild_id)", file=sys.stderr)
            return 2
        if a.write_policy:
            print(write_policy(a.write_policy, guild_id))
            return 0
        print(json.dumps(record_template(task, guild_id) if a.print_record_template
                         else {"rate_limits": {"per_destination": policy_rule(guild_id)}}, indent=1, sort_keys=True))
        return 0
    if task.mode != "live" or environment_of(base_url) != "live":
        print(f"live mode needs a task in mode 'live' against {LIVE_BASE_URL} (this task: mode {task.mode!r}, "
              f"{base_url}); use --staging for the staging services", file=sys.stderr)
        return 2
    if not guild_id:
        print("the task names no server: set resources.discord.guild_id", file=sys.stderr)
        return 2
    policy, env, discord = build(a.run_dir, task, policy_path=a.policy, base_url=base_url, provenance="live")
    echo = lambda line: print(policy.scrub(line), flush=True)                               # noqa: E731
    ok, why = discord.availability()
    if not ok:
        echo(f"cannot start: {why}")
        env.close()
        return 2
    started = time.time()
    check = Check(policy, env, discord, task, approve_wait_s=a.approve_wait_s, cleanup_wait_s=a.cleanup_wait_s, echo=echo)
    echo(f"LIVE check against {base_url}, server {guild_id}, run directory {os.path.abspath(a.run_dir)}")
    try:
        code = check.run(skip_writes=a.skip_writes, await_reply_s=a.await_reply_s, listen_s=a.listen_s)
    finally:
        env.close()
    try:
        out = a.out or os.path.join(a.run_dir, "discord_live_check.json")
        data = evidence(check, policy, discord, mode="live", exit_code=code, started=started, extra={})
        if not write_evidence(out, data, policy, echo):
            code = 6
        echo(f"summary: {json.dumps(check.summary()['by_outcome'])}; exit code {code}")
        return code
    finally:
        policy.close()


def run_staging(a: argparse.Namespace, mutate: Optional[Callable[[Any, Dict[str, Any]], None]] = None) -> int:
    """The same sequence against the staging services. ``mutate(staging_state, task_resources)`` lets a test put
    the staging server or the task into another condition before the run."""
    from cam_live.harness.staging_discord import StagingDiscord, StagingDiscordServer
    run_dir = os.path.abspath(a.run_dir) if a.run_dir else tempfile.mkdtemp(prefix="discord_check_staging_")
    token = "staging-" + os.urandom(24).hex()                    # a secret of this run only; it opens nothing real
    st = StagingDiscord(token=token, guild_name="staging guild", rate_limit=(1000, 1.0))
    server = StagingDiscordServer(st)
    for i in range(3):
        st.add_human(f"resident {i}")
    st.add_bot("another bot")
    tester = st.add_test_account("operator test account")
    rest, gateway = server.base_url.split("//", 1)[1].split("/", 1)[0], server.gateway.url.split("//", 1)[1]
    owned = [f"account:discord:{st.guild_id}", "channel:discord:*"]
    resources: Dict[str, Any] = {"discord": {"guild_id": st.guild_id, "base_url": server.base_url, "test_account_ids": [tester]}}
    limits = C.Limits(duration_s=3600, max_external_actions=500, max_decisions=500)
    if mutate is not None:
        mutate(st, resources)
    if a.live_policy:
        task = C.TaskSpec(objective_id="discord-live-check", objective="integration check of the Discord adapter",
                          mode="live", resources=resources, interfaces=["api"], limits=limits)
    else:
        resources.update(owned=owned, preauthorized=owned)
        task = C.TaskSpec(objective_id="discord-live-check", objective="integration check of the Discord adapter",
                          mode="staging", resources=resources, interfaces=["api"], limits=limits)
    os.makedirs(run_dir, exist_ok=True)
    policy_path = a.policy or write_policy(os.path.join(run_dir, "policy_discord_check.yaml"), st.guild_id)
    assert environment_of(server.base_url) == "staging"
    policy, env, discord = build(
        run_dir, task, policy_path=policy_path, base_url=server.base_url, provenance="staging", fsync=False,
        prepare=lambda p: p.put_secret("discord_bot_token", token, [rest, gateway], ["discord"], by="staging check"))
    assert discord.environment == "staging"
    extra: Dict[str, Any] = {"staging": True, "staging_note": (
        "STAGING ONLY: local imitation of the Discord REST API and Gateway (cam_live.harness.staging_discord). "
        "No request was sent to Discord. The human reply and the events inside the listening window were produced "
        "by staging hooks of this script, not by people.")}
    if a.live_policy:                       # the policy's live mode: a record signed with a key made for this run
        priv, pub = os.path.join(run_dir, "operator.key"), os.path.join(policy.keys_dir, "operator.pub")
        if not os.path.exists(pub):
            liveauth.generate_operator_key(priv, pub)
        record = record_template(task, st.guild_id)
        record["read"] = [rest, gateway]
        reg = policy.register_live_authorization(liveauth.sign_live_authorization(record, priv), by="staging check")
        extra["live_policy"] = {"record_id": reg["record_id"], "signed_with": "an operator key generated for this run",
                                "read": record["read"], "preauthorized": record["preauthorized"]}
    echo = lambda line: print(policy.scrub(line), flush=True)                               # noqa: E731
    started = time.time()
    check = Check(policy, env, discord, task, approve_wait_s=0.0, echo=echo)
    echo(f"STAGING check against {server.base_url} (gateway {server.gateway.url}), policy mode {task.mode}, "
         f"run directory {run_dir}")

    def reply_hook(cid: str) -> None:              # stands in for a person answering in the test channel
        time.sleep(0.3)
        st.post_member_message(cid, st.humans()[0], "a reply typed by a member (staging hook)")

    def between_hook() -> None:                    # membership changes between the two readings
        st.add_human("joined between the readings")
        st.add_bot("a bot added between the readings")

    def listen_hook() -> None:                     # things that happen in the server during the listening window
        if server.gateway.wait_ready(8.0):
            joined = st.add_human("joined during the window")
            st.post_member_message(st.general_id, joined, "hello (staging hook)")
            st.remove_member(joined)

    name_before = st.guild["name"]
    try:
        try:
            code = check.run(skip_writes=a.skip_writes, await_reply_s=a.await_reply_s, listen_s=a.listen_s,
                             during_reply_wait=reply_hook, between_readings=between_hook, during_listen=listen_hook)
        finally:
            env.close()
            server.close()
        leftover = {"channels": sorted(c["name"] for c in st.channels.values() if c["guild_id"]),
                    "roles": sorted(r["name"] for r in st.roles.values()), "guild_name": st.guild["name"],
                    "invites": len(st.invites)}
        extra["staging_state_after"] = leftover
        extra["staging_requests"] = {"total": len(st.request_log),
                                     "writes": sum(1 for r in st.request_log if r["method"] != "GET"),
                                     "gateway_identifies": [i["outcome"] for i in server.gateway.identify_log]}
        if CHANNEL_NAME in leftover["channels"] or leftover["guild_name"] != name_before \
                or any(r.startswith(ROLE_NAME) for r in leftover["roles"]):
            echo("!! the staging server was not left as it was found")
            code = code or 1
        out = a.out or os.path.join(run_dir, "discord_live_check_staging.json")
        data = evidence(check, policy, discord, mode="staging", exit_code=code, started=started, extra=extra)
        if not write_evidence(out, data, policy, echo):
            code = 6
        echo(f"summary: {json.dumps(check.summary()['by_outcome'])}; exit code {code}")
        return code
    finally:
        policy.close()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="discord_live_check", description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", default="", help="run directory of the authorization service (vault, records, audit log)")
    ap.add_argument("--task", default="", help="task file (YAML, or JSON of TaskSpec.to_dict()) the live record was signed for")
    ap.add_argument("--policy", default=None, help="policy file for the authorization service (default: its default policy)")
    ap.add_argument("--skip-writes", action="store_true", help="read-only steps only")
    ap.add_argument("--await-reply-s", type=float, default=0.0, help="wait this long for a human reply in the test channel")
    ap.add_argument("--approve-wait-s", type=float, default=0.0, help="wait this long for the operator to approve a step")
    ap.add_argument("--cleanup-wait-s", type=float, default=900.0,
                    help="longest total wait for a rate limit before a cleanup step is given up (default 900)")
    ap.add_argument("--listen-s", type=int, default=10, choices=range(1, 61), metavar="1..60",
                    help="length of the listening window in seconds (default 10)")
    ap.add_argument("--out", default="", help="evidence file (JSON)")
    ap.add_argument("--staging", action="store_true", help="run against the local staging REST service and staging Gateway")
    ap.add_argument("--live-policy", action="store_true", help="with --staging: use the policy's live mode with a record signed for this run")
    ap.add_argument("--print-record-template", action="store_true", help="print the live authorization record to sign for --task")
    ap.add_argument("--print-policy-rule", action="store_true", help="print the rate_limits.per_destination entries the writes need")
    ap.add_argument("--write-policy", default="", metavar="FILE",
                    help="write a copy of the default policy with those entries for the server of --task, and exit")
    a = ap.parse_args(argv)
    if a.live_policy and not a.staging:
        ap.error("--live-policy is a variant of --staging")
    return run_staging(a) if a.staging else run_live(a)


if __name__ == "__main__":
    sys.exit(main())
