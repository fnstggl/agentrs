"""G2-integration probe p06 — Discord observations enter the experience pipeline (OP-05), staging only.

The local STAGING Discord service (harness.staging_discord, 127.0.0.1), the real DiscordExecutor, and the integrated
agent on the tiny CPU configuration. The agent proposes, evaluates and selects among read-only Discord tools; the
executor executes the chosen action against the staging service; the result goes back through record_result; the
agent learns. Checked: the Discord observations become items of the experience store, are in the context of the
agent's next decisions (what the model's input is built from), the records of Discord actions are in the store
(replay), and learning updates run on that store. Nothing contacts discord.com.

Usage (from /workspace/camlive):
    PYTHONPATH=<copy>/cam_live/src:<copy>/cam_live/tests/harness/skills \
        .venv/bin/python cam_live/docs/verification/repro/g2integration/p06_discord_observations.py
"""
from __future__ import annotations

import os
import sys
import tempfile

from cam_live import contracts as C
from cam_live.harness.discord import LIVE_BASE_URL, DiscordExecutor
from cam_live.harness.staging_discord import StagingDiscord, StagingDiscordServer
from es_support import ScopedPolicy, make_ctx

SCRATCH = os.environ.get("G2I_SCRATCH", "/tmp/claude-0/-workspace/8e531f56-63d9-4df0-a584-c2fb645312ce/scratchpad/"
                                        "G2INTEGRATION/p06")
READ = ("discord.list_channels", "discord.get_guild_stats", "discord.get_channel_activity", "discord.list_events")
broken = 0


def check(name, ok, detail=""):
    global broken
    broken += 0 if ok else 1
    print(f"[{'HOLDS ' if ok else 'BROKEN'}] {name}" + (f" :: {detail}" if detail else ""))


def main() -> int:
    os.makedirs(SCRATCH, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="p06_", dir=SCRATCH)
    server = StagingDiscordServer(StagingDiscord(rate_limit=(1000, 1.0)))
    st = server.state
    for i in range(3):
        st.add_human(f"person{i}")
    assert server.base_url != LIVE_BASE_URL and "127.0.0.1" in server.base_url
    ex = DiscordExecutor(server.base_url, secrets={"discord_bot_token": st.token}, guild_id=st.guild_id)
    ctx = make_ctx(ScopedPolicy(scopes=("discord.read",)))
    tools = [t for t in ex.tools() if t.name in READ]
    from cam_live.agent import CamLiveAgent
    from cam_live.config import CamLiveConfig
    cfg = CamLiveConfig.debug(seed=0)
    cfg.optim.batch_size = 4
    cfg.sections["agent"] = {"use_backbone": False, "durable": False, "planner": False}   # speed; not under test
    cfg.sections["credit"] = {"learner": {"min_independent_outcomes": 1, "warmup_steps": 0, "slow_batch_size": 2}}
    task = C.TaskSpec(objective_id="disc", objective="observe the staging server", mode="staging",
                      interfaces=["discord"],
                      limits=C.Limits(duration_s=3600, money=0, max_decisions=20, max_external_actions=20))
    clock = C.WallClock()
    agent = CamLiveAgent(cfg, os.path.join(tmp, "agent"), clock=clock, task=task)
    obs_ids, recs, ctx_refs, updates = [], [], [], []
    for step in range(6):
        cands = agent.propose(tools, 6)
        dec = agent.select(agent.evaluate(cands))
        ctx_refs.append(list(dec.diagnostics.get("context_item_refs") or []))
        res = ex.execute(dec.chosen, ctx)
        res.action_id = dec.chosen.action_id
        agent.record_result(dec, res)
        obs_ids += [o.obs_id for o in res.observations]
        recs.append(dec.decision_id)
        updates += agent.learn(30.0)
    provs = sorted({str(getattr(o.provenance, "source", "")) + "|" + str(getattr(o.provenance, "environment", ""))
                    for o in res.observations})
    print(f"[NOTE  ] chosen operations: {[agent.store.by_decision(d).decision.chosen.operation for d in recs]}; "
          f"last observation provenance: {provs}")
    check("1 Discord actions executed against the staging service returned observations", bool(obs_ids),
          f"{len(obs_ids)} observations, {len(st.request_log)} requests at the staging service")
    items = agent.store.items(obs_ids, missing_ok=True)
    found = [it for it in items if it is not None]
    check("2 every Discord observation is an item of the experience store", len(found) == len(obs_ids),
          f"{len(found)} of {len(obs_ids)}")
    later = set(r for refs in ctx_refs[1:] for r in refs)
    check("3 Discord observations are in the context the next decisions are built from (model input)",
          any(o in later for o in obs_ids), f"{sum(1 for o in obs_ids if o in later)} of {len(obs_ids)} in later contexts")
    stored = [agent.store.by_decision(d) for d in recs]
    check("4 every Discord action is an experience record of the store (replay)",
          all(r is not None and r.result is not None for r in stored))
    post = set(r for rec in stored if rec is not None for r in (rec.post_item_refs or []))
    check("5 the records point at the Discord observations they produced", any(o in post for o in obs_ids))
    scales = sorted({u.timescale for u in updates})
    nz = [u for u in updates if any(v > 0 for v in (u.weight_delta or {}).values())]
    check("6 learning updates ran on this store with non-zero weight change", bool(nz),
          f"timescales {scales}; {len(nz)} of {len(updates)} with non-zero deltas")
    agent.close()
    server.close()
    print(f"== p06_discord_observations: {broken} broken of 6 checks ({tmp})")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
