"""A short read-only live run: the integrated agent receives live Discord observations through the harness.

Operator-authorized (coordinator instruction 2026-11-27 ~18:15 UTC). Prepared as `docs/evidence/discord/README.md`
describes: the bot credential goes into the run's vault through `cam_live.harness.credentials` (read from the
operator's 0600 file by that tool; never printed or copied by this script), the operator's public key is placed
where the authorization service looks, and a one-hour live authorization record for exactly this task is signed with
the operator's private key (path only; the key never leaves `liveauth.sign_live_authorization`).

The task: mode live, interfaces [api], authorized tools `discord.get_*` / `discord.list_*` (aggregate reads),
`member_list: false`, max_external_actions small. The environment offered to the agent lists only those read tools
(defence in depth; the authorization service also refuses anything else). The agent is the integrated
`CamLiveAgent` at debug scale on the CPU (an engineering check of the pathway — observations into the experience
store and the learner — not a learning result). Nothing is written to the server.

    cd /workspace/camlive && .venv/bin/python cam_live/docs/evidence/validation/discord_live/run_live_observe.py \\
        --out cam_live/docs/evidence/validation/discord_live/live_observe_<ts>.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
REPO = os.path.dirname(PROJECT)
SRC = os.path.join(PROJECT, "src")
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(PROJECT, "tools"))
CONF = "/root/.config/camlive"
READ_TOOLS = ("discord.get_identity", "discord.get_permissions", "discord.get_guild_stats", "discord.get_guild_settings",
              "discord.list_channels", "discord.list_invites", "discord.list_events", "discord.list_roles",
              "discord.get_channel_activity", "discord.get_audit_summary")


class ReadOnlyView:
    """The computer environment as the agent sees it: only the authorized aggregate-read tools are listed."""

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def discover_tools(self):
        return [t for t in self._env.discover_tools() if t.name in READ_TOOLS]

    def validate_action(self, action):
        if action.operation not in READ_TOOLS and action.operation != "wait":
            return [f"{action.operation} is not offered in this read-only run"]
        return self._env.validate_action(action)


class NoObjectiveVerifier:
    """There is no objective to verify in an observation run (goal-free mode); measurements are not asked for."""

    def verify(self, task):
        raise RuntimeError("no objective in this run")

    def check_signature(self, outcome):
        return False

    def measure(self, queries):
        return []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--decisions", type=int, default=10)
    a = ap.parse_args(argv)
    import yaml
    from authorize_live import build_record
    from cam_live.agent import CamLiveAgent
    from cam_live.contracts import RunFiles, WallClock
    from cam_live.harness.auth import PolicyService, liveauth
    from cam_live.harness.discord import LIVE_BASE_URL, DiscordExecutor
    from cam_live.harness.env import ComputerEnvironment
    from cam_live.harness.http_api import ApiModality
    from cam_live.runtime.loop import LiveRuntime, RuntimeConfig
    from cam_live.runtime.logs import RunReader
    from cam_live.runtime.tasks import load_task
    from cam_live.validation import staging_config
    ids = json.load(open(os.path.join(CONF, "discord_ids.json")))
    guild = str(ids.get("guild_id") or ids.get("server_id") or "")
    if not guild:
        print("no guild id in discord_ids.json", file=sys.stderr)
        return 2
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    R = os.path.join(REPO, "runs", f"int2_discord_live_{stamp}")
    os.makedirs(R)
    report = {"kind": "discord_live_read_only_observation", "started_utc": stamp, "run_dir": R, "executed": False}
    try:
        task_doc = {"version": 1, "objective_id": "discord-live-observe", "mode": "live", "family": "community/observe",
                    "objective": "", "resources": {"discord": {"guild_id": guild},
                                                   "authorized_tools": ["discord.get_*", "discord.list_*"]},
                    "interfaces": ["api"],
                    "limits": {"duration_s": 3600, "money": 0.01, "compute_s": 3600, "max_external_actions": 12,
                               "max_decisions": int(a.decisions)},
                    "stop_conditions": ["duration_exhausted", "decisions_exhausted", "external_actions_exhausted",
                                        "operator_stop", "emergency_stop", "authorization_revoked"],
                    "data_permissions": {"aggregate_metrics": True, "member_list": False,
                                         "store_member_identifiers": False, "store_message_content": False}}
        task_path = os.path.join(R, "task.yaml")
        yaml.safe_dump(task_doc, open(task_path, "w"), sort_keys=False)
        task = load_task(task_path, for_run=True)
        # 1. the bot credential into this run's vault (the tool reads the operator's file; nothing is printed here)
        p = subprocess.run([sys.executable, "-m", "cam_live.harness.credentials", "set", "discord_bot_token",
                            "--run-dir", R, "--no-verifier", "--from-file", os.path.join(CONF, "discord_bot_token")],
                           env=dict(os.environ, PYTHONPATH=SRC), capture_output=True, text=True, timeout=120)
        report["credential_set_exit"] = p.returncode
        if p.returncode != 0:
            report["credential_error"] = "credentials tool failed (output withheld)"
            return 1
        # 2. operator public key where the authorization service looks; 3. sign + register the one-hour record
        os.makedirs(os.path.join(R, "auth", "keys"), exist_ok=True)
        shutil.copyfile(os.path.join(CONF, "operator_ed25519.pub"), os.path.join(R, "auth", "keys", "operator.pub"))
        policy = PolicyService(R)
        rec = build_record(task, hours=1.0, owned=[], read=[], preauthorized=[], accounts=[],
                           issued_by="operator (read-only live observation, INT2, authorized 2026-11-27)")
        signed = liveauth.sign_live_authorization(rec, os.path.join(CONF, "operator_ed25519"))
        reg = policy.register_live_authorization(signed, by="operator")
        report["authorization"] = {k: v for k, v in reg.items() if k in ("record_id", "task_id", "not_after", "ok",
                                                                         "registered", "accepted")} \
            if isinstance(reg, dict) else str(reg)[:200]
        report["record"] = {k: rec[k] for k in ("record_id", "task_id", "task_hash", "owned", "read", "preauthorized",
                                                "budgets", "not_before", "not_after")}
        env = ComputerEnvironment(os.path.join(R, "exec"), policy, secrets=policy.secrets_handle(), provenance="live")
        names = lambda: {s["name"] for s in policy.secret_names() if not s["revoked"]}          # noqa: E731
        discord = DiscordExecutor.from_task(task, base_url=LIVE_BASE_URL, secrets=policy.secrets_handle(),
                                            credential_probe=lambda name: name in names(),
                                            state_dir=os.path.join(R, "exec", "discord_state"))
        env.register(ApiModality([discord]))
        view = ReadOnlyView(env)
        report["tools_offered"] = sorted(t.name for t in view.discover_tools())
        cfg = staging_config(None, False, 3)
        cfg.sections["agent"].update(use_backbone=False, planner=False, durable=True)
        clock = WallClock()
        agent = CamLiveAgent(cfg, os.path.join(R, "agent"), clock=clock)
        rt = LiveRuntime(agent, view, policy, NoObjectiveVerifier(), task, os.path.join(R, "runtime"), clock,
                         RuntimeConfig(fsync=True, candidates_k=8, keep_checkpoints=1, min_idle_sleep_s=0.0))
        rt.start()
        t0, steps = time.time(), 0
        while rt.status == "running" and steps < 4 * a.decisions and time.time() - t0 < 900:
            rt.step()
            steps += 1
        if rt.status == "running":
            rt._finish("stopped", "observation run step budget reached")
        rt.close()
        r = RunReader(rt.run_dir)
        results = [e["data"] for e in r.records(RunFiles.EVENTS, types=["action_result"])]
        obs_items = [i for rec_ in agent.store.all() for i in agent.store.items(rec_.post_item_refs or [])
                     if i is not None and i.kind in ("observation", "document", "event")]
        ups = [e["data"] for e in r.records(RunFiles.LEARNING)]
        report.update(
            executed=True, seconds=time.time() - t0, status=rt.status, reason=rt.status_reason,
            counters=dict(rt.counters),
            chosen=[(x["data"]["decision"]["chosen"]["operation"], x["data"]["decision"]["chosen"]["arguments"])
                    for x in r.records(RunFiles.DECISIONS)],
            results=[{k: d.get(k) for k in ("kind", "status", "failure_class")} for d in results],
            writes_attempted=sum(1 for x in r.records(RunFiles.DECISIONS)
                                 if x["data"]["decision"]["chosen"]["operation"] not in READ_TOOLS + ("wait",)),
            store_records=agent.store.count(),
            observation_items_in_store=len({i.ref for i in obs_items}),
            observation_examples=[{"kind": i.kind, "text": i.text[:160],
                                   "quantities": [[q.name, q.value] for q in i.quantities[:6]]} for i in obs_items[:6]],
            learning_updates={t: sum(1 for u in ups if u.get("timescale") == t) for t in sorted({u.get("timescale")
                                                                                              for u in ups})},
            weight_deltas_nonzero=sum(1 for u in ups if any(v > 0 for v in (u.get("weight_delta") or {}).values())),
            agent_status={k: agent.status().get(k) for k in ("decisions", "store_records", "goal_free", "model_version",
                                                             "counters")})
        agent.close()
        env.close()
        policy.close()
        # the evidence must contain no vault secret
        blob = json.dumps(report, default=str)
        with open(os.path.join(CONF, "discord_bot_token")) as f:        # compared in memory only, never output
            tok = f.read().strip()
        report["secret_free"] = bool(tok) and blob.find(tok) < 0
        del tok
        return 0
    finally:
        with open(a.out, "w") as f:
            json.dump(report, f, indent=1, default=str)


if __name__ == "__main__":
    sys.exit(main())
