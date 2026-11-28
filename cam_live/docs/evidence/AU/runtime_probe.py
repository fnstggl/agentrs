"""Diagnostic: the real runtime (LiveRuntime + CamLiveAgent debug core, CPU) on a live task in the composed environment
behind the real authorization service, staging Discord + local hosting as stand-ins; with / without the standing
profile (same seed). Counts the operations the agent chose and what the authorization service answered."""
import collections, json, os, shutil, sys, tempfile, time
T = "/workspace/camlive/cam_live/tests"
for p in (T + "/integration", T + "/harness/autonomy", T + "/harness/auth", T + "/harness/env", "/workspace/camlive/cam_live/tools"):
    sys.path.insert(0, p)
from cam_live import contracts as C
from cam_live.harness.auth import PolicyService, liveauth
from cam_live.harness.env import ComputerEnvironment
from cam_live.harness.factories import all_executors
from cam_live.harness.staging_discord import StagingDiscord, StagingDiscordServer
from cam_live.runtime.loop import LiveRuntime, RuntimeConfig
from int_helpers import agent_config
from au_support import PROFILE, install_operator_key
from e_fix_support import CountingServer
import authorize_live

class NoVerifier:
    def verify(self, task): raise RuntimeError("not used")
    def check_signature(self, outcome): return False
    def measure(self, queries): return []

def run(standing, seed, steps):
    discord = StagingDiscordServer(StagingDiscord(token="staging-" + os.urandom(12).hex(), rate_limit=(1000, 1.0)))
    st = discord.state
    base = tempfile.mkdtemp(prefix="camlive-au-rt-", dir="/tmp")
    rest = discord.base_url.split("//", 1)[1].split("/", 1)[0]
    task = C.TaskSpec(objective_id=f"obj-au-rt-{int(standing)}", objective="Operate the community server you were given.",
                      mode="live", interfaces=["api", "internal"],
                      success={"verifier": "metric_threshold", "metric": "x.count", "target": 999},
                      resources={"discord": {"guild_id": st.guild_id, "base_url": discord.base_url, "token_secret": "discord_bot_token"},
                                 "authorized_tools": ["discord.*", "internal.*", "wait"]},
                      data_permissions={"aggregate_metrics": True},
                      limits=C.Limits(duration_s=3600, money=1.0, max_external_actions=60, max_decisions=steps, compute_s=600),
                      stop_conditions=["verified_success", "budget_exhausted", "operator_stop"])
    clock = C.WallClock()
    svc = PolicyService(os.path.join(base, "auth"), clock=clock, fsync=False)
    svc.register_local_service(rest, by="operator")
    svc.put_secret("discord_bot_token", st.token, [rest], ["discord"], by="operator")
    priv = install_operator_key(svc, base)
    rec = authorize_live.build_record(task, hours=1.0, owned=[], read=[rest], preauthorized=[], accounts=[], issued_by="operator",
                                      standing=PROFILE if standing else "")
    svc.register_live_authorization(liveauth.sign_live_authorization(rec, priv), "operator")
    svc.register_task(task, "operator")
    env = ComputerEnvironment(os.path.join(base, "exec"), svc, clock=clock, secrets=svc.secrets_handle(), provenance="live")
    for ex in all_executors(os.path.join(base, "exec"), clock, env=env, policy=svc, task=task, browser_enabled=False,
                            discord_base_url=discord.base_url, exclude=("visual", "code", "skill", "structured_browser")):
        env.register(ex)
    from cam_live.agent import CamLiveAgent
    agent = CamLiveAgent(agent_config(seed=seed, use_backbone=False, planner=False), os.path.join(base, "agent"), clock=clock)
    out = collections.Counter()
    try:
        rt = LiveRuntime(agent, env, svc, NoVerifier(), task, os.path.join(base, "rt"), clock,
                         RuntimeConfig(fsync=True, candidates_k=8, keep_checkpoints=1))
        rt.start()
        for _ in range(steps):
            if rt.status != "running": break
            rt.step()
        rt.close()
        for r in agent.store.all():
            if r.result is None: continue
            op = r.decision.chosen.operation
            kind = "write" if op.startswith("discord.") and not any(op.startswith(p) for p in ("discord.get", "discord.list")) else "read/internal"
            out[(kind, str(r.result.status))] += 1
        pend = [a for a in svc.approvals() if a.get("status") == "pending"]
        return {"standing": standing, "seed": seed, "results": {f"{k[0]}:{k[1]}": v for k, v in sorted(out.items())},
                "approvals_pending": len(pend), "channels_created": len([c for c in st.channels.values() if c["guild_id"] == st.guild_id])}
    finally:
        agent.close(); env.close(); svc.close(); discord.close(); shutil.rmtree(base, ignore_errors=True)

if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    seeds = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0]
    for seed in seeds:
        for standing in (True, False):
            t0 = time.time(); r = run(standing, seed, steps); r["wall_s"] = round(time.time() - t0, 1)
            print(json.dumps(r), flush=True)
