"""Manual end-to-end check (not part of the ES test-suite, because it depends on other workstreams' code):
the ES executors registered in EB's ComputerEnvironment, authorized by EA's PolicyService, against a local HTTP
application and the STAGING Discord service. Nothing here is a live result.

Run: cd /workspace/camlive && .venv/bin/python cam_live/docs/evidence/ES/integration_smoke.py
"""
import sys, tempfile, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'tests', 'harness', 'skills'))
from cam_live import contracts as C
from cam_live.harness.env import ComputerEnvironment
from cam_live.harness.auth import PolicyService
from cam_live.harness.http_api import HttpApiExecutor, ApiModality
from cam_live.harness.discord import DiscordExecutor
from cam_live.harness.publishing import PublishingExecutor, Destination
from cam_live.harness.skills import SkillExecutor, SkillLibrary
from cam_live.harness.staging_discord import StagingDiscord, StagingDiscordServer
from es_support import NotesApp, AppServer

tmp = tempfile.mkdtemp()
app = NotesApp(); srv = AppServer(app); U = srv.url
sd = StagingDiscordServer(StagingDiscord(rate_limit=(1000, 1.0))); st = sd.state
site = os.path.join(tmp, "site"); os.makedirs(site)
svc = PolicyService(os.path.join(tmp, "run"), fsync=False)
svc.put_secret("discord_bot_token", st.token, [sd.base_url.split("/api")[0].replace("http://", "")])
host = U.replace("http://", "")
task = C.TaskSpec(objective_id="integ", objective="smoke", mode="staging",
                  resources={"owned": [host, f"channel:discord:{st.general_id}", f"account:discord:{st.guild_id}", f"file:{site}"],
                             "preauthorized": [host], "discord": {"guild_id": st.guild_id}},
                  interfaces=["api", "skill", "internal"], limits=C.Limits(duration_s=3600, money=1, max_decisions=500, max_external_actions=500))
env = ComputerEnvironment(os.path.join(tmp, "run"), svc, secrets=svc.secrets_handle())
http = HttpApiExecutor()
disc = DiscordExecutor(sd.base_url, guild_id=st.guild_id, secrets=svc.secrets_handle(),
                       credential_probe=lambda name: name == "discord_bot_token")     # stored just above
pub = PublishingExecutor(os.path.join(tmp, "state"), [Destination("site", "static_site", root=site, base_url=U),
                                                      Destination("ann", "discord_channel", channel_id=st.general_id)], discord=disc)
lib = SkillLibrary(os.path.join(tmp, "state"), env.clock)
skills = SkillExecutor(lib, [])
env.register(ApiModality([http, pub, disc])); env.register(skills)
skills.inducer.set_tools(env.discover_tools())
print("tools:", sorted(t.name for t in env.discover_tools())[:40])

def go(op, args, modality="api", level="primitive", **kw):
    a = C.ActionSpec(action_id=C.new_id("a"), operation=op, modality=modality, arguments=args, level=level, **kw)
    errs = env.validate_action(a)
    if errs: print("  INVALID", op, errs); return a, None
    dec = svc.authorize(a, task, {})
    res = env.execute(a, dec)
    skills.observe_execution(a, res)
    print(f"  {op:28s} auth={dec.verdict:14s} -> {res.status:14s} {res.failure_class:22s} {res.failure_detail[:110]}")
    return a, res

ids = []
for t, b in (("one a", "body a"), ("two b", "body b")):
    a1, r1 = go("http.request", {"method": "POST", "url": U + "/api/items", "json": {"title": t, "body": b}})
    iid = r1.observations[0].data["json"]["id"]
    a2, r2 = go("http.request", {"method": "POST", "url": f"{U}/api/items/{iid}/publish"})
    ids.append([a1.action_id, a2.action_id])
a, r = go("skill.induce", {"traces": ids, "name": "create and publish"}, modality="skill", level="procedure")
if r and r.status == "ok":
    sid = r.artifacts["skill_id"]
    go("skill.run", {"skill_id": sid, "arguments": {"title": "via skill", "body": "zzz"}}, modality="skill", level="procedure")
    print("  item:", app.items[-1])
go("discord.get_guild_stats", {"guild_id": st.guild_id})
go("discord.list_channels", {"guild_id": st.guild_id})
go("discord.send_message", {"guild_id": st.guild_id, "channel_id": st.general_id, "content": "hello from the smoke check"})
print("  discord messages:", [m["content"] for m in st.messages[st.general_id]])
a, r = go("publish.draft", {"destination": "site", "path": "index.html", "content": "<h1>x</h1>"})
if r:
    d = r.artifacts.get("draft_id")
    a, r = go("publish.preview", {"draft_id": d})
    addr = r.observations[0].data["where"]["address"] if r else ""
    go("publish.submit", {"draft_id": d, "destination": addr})
    go("publish.status", {})
    print("  site files:", os.listdir(site))
print("approvals pending:", [(x.get("kind"), x.get("policy_ids")) for x in svc.approvals()][:5])
srv.close(); sd.close()

# ---- approval flow with the real authorization service -------------------------------------------------------
sd = StagingDiscordServer(st); disc.base_url = sd.base_url; disc.close()
svc.put_secret("discord_bot_token", st.token, [sd.base_url.split("/api")[0].replace("http://", "")])
a = C.ActionSpec(action_id=C.new_id("a"), operation="discord.send_message", modality="api",
                 arguments={"guild_id": st.guild_id, "channel_id": st.general_id, "content": "approved words"})
dec = svc.authorize(a, task, {})
res = env.execute(a, dec)
print("first:", dec.verdict, res.status, [m["content"] for m in st.messages[st.general_id]])
dec2 = svc.resolve_approval(dec.approval_id, True, by="operator")
res2 = env.execute(a, dec2)
print("after approval:", dec2.verdict, res2.status, res2.failure_detail[:200], [m["content"] for m in st.messages[st.general_id]])
# publishing to the discord channel destination through publish.submit
a, r = go("publish.draft", {"destination": "ann", "content": "channel text"})
d = r.artifacts["draft_id"]; a, r = go("publish.preview", {"draft_id": d})
sub = C.ActionSpec(action_id=C.new_id("a"), operation="publish.submit", modality="api",
                   arguments={"draft_id": d, "destination": r.observations[0].data["where"]["address"]})
dec = svc.authorize(sub, task, {}); res = env.execute(sub, dec)
print("publish.submit->discord:", dec.verdict, res.status, res.failure_detail[:160])
if dec.verdict == "needs_approval":
    dec2 = svc.resolve_approval(dec.approval_id, True, by="operator"); res2 = env.execute(sub, dec2)
    print("  after approval:", dec2.verdict, res2.status, res2.failure_class, res2.failure_detail[:200], [m["content"] for m in st.messages[st.general_id]])
    print("  draft:", pub.drafts()[-1]["state"])
a, r = go("publish.update", {"draft_id": d, "content": "channel text 2"})                          # see change request ES-3
a, r = go("publish.update", {"draft_id": d, "content": "channel text 2", "destination": f"channel:discord:{st.general_id}"})
sd.close()
