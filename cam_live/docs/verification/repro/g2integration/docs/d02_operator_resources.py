"""G2-integration / docs probe d02: claims of docs/OPERATOR_RESOURCES.md (and the matching KNOWN_FAILURES rows)
against the code of a tree. Imports the code (CPU only, no model is built, no service started, nothing contacted).

usage (from /workspace/camlive):
  PYTHONPATH=<SRC>/cam_live/src CAM_LIVE_HOME=<scratch>/home CAM_LIVE_GPU_LOCK=<scratch>/gpu.lock CUDA_VISIBLE_DEVICES= \
      .venv/bin/python cam_live/docs/verification/repro/g2integration/docs/d02_operator_resources.py <SRC>
Prints HOLDS (the document's claim is true of the code) / BROKEN (it is not) per claim.
"""
import os
import sys

SRC = os.path.abspath(sys.argv[1])
sys.path.insert(0, os.path.join(SRC, "cam_live", "src"))
os.chdir(SRC)
import yaml  # noqa: E402

from cam_live.config import CamLiveConfig  # noqa: E402
from cam_live.harness import SERVICE_USERS, SERVICE_MODULES  # noqa: E402
from cam_live.harness import credentials as C  # noqa: E402
from cam_live.harness import discord_perms as DP  # noqa: E402
from cam_live.runtime import preflight as PF  # noqa: E402
from cam_live.runtime import cli as CLI  # noqa: E402
from cam_live.runtime.tasks import load_task_raw, parse_task  # noqa: E402

n = {"HOLDS": 0, "BROKEN": 0}


def claim(cid, ok, what):
    k = "HOLDS" if ok else "BROKEN"
    n[k] += 1
    print(f"{k:6}  {cid}: {what}")


cfg_raw = yaml.safe_load(open("cam_live/configs/cam_live.yaml"))
rt = cfg_raw.get("runtime") or {}
# --- persistent storage: "The code has no override for it" (OPERATOR_RESOURCES 0, 2.1; KF-D01)
claim("OR-0/KF-D01 no override for persistent_storage", "persistent_storage" not in getattr(PF, "ACCEPTABLE_RISKS", {}),
      f"preflight.ACCEPTABLE_RISKS = {getattr(PF, 'ACCEPTABLE_RISKS', None)}; cam-live run/preflight/resume flag "
      f"{getattr(CLI, 'ACCEPT_EPHEMERAL', None)!r}")
p = CLI.build_parser()
helps = p.format_help()
run_help = [a for a in p._subparsers._group_actions[0].choices["run"]._actions if "--accept-ephemeral-storage" in a.option_strings]
claim("OR-0 item 1 'no operator command fixes it'", not run_help, "cam-live run accepts --accept-ephemeral-storage"
      if run_help else "no such option")
# --- GPU lock keys: "The default config sets neither key" (OPERATOR_RESOURCES 2.5; REPRODUCE 14; KF-D07)
claim("OR-2.5/KF-D07 default config sets neither runtime.gpu_lock nor gpu_hold_s",
      "gpu_lock" not in rt and "gpu_hold_s" not in rt, f"configs/cam_live.yaml runtime = {rt}")
claim("OR-2.5 runtime holds the lock for the whole process lifetime (unbounded hold by default)",
      not rt.get("gpu_hold_s"), f"gpu_hold_s = {rt.get('gpu_hold_s')} (lease cycled per hold, runtime/gpu_cycle.py)")
# --- operator public key
claim("OR-2.2 configs/cam_live.yaml has no launcher.operator_pubkey",
      not (cfg_raw.get("launcher") or {}).get("operator_pubkey"), f"launcher section = {cfg_raw.get('launcher')}")
# --- service users and isolation
claim("OR-2.5 five service users camlearn/cambrowser/camsbx/camauth/camverify",
      sorted(SERVICE_USERS.values()) == sorted(["camlearn", "cambrowser", "camsbx", "camauth", "camverify"]),
      f"SERVICE_USERS = {SERVICE_USERS}")
claim("RE-1.1/OR (info) services = auth, exec, verifier, learner only", set(SERVICE_MODULES) == {"auth", "exec", "verifier", "learner"},
      f"SERVICE_MODULES = {sorted(SERVICE_MODULES)} (the root sandbox host is a fifth service)")
# --- credential copies and bindings
k = C.KNOWN.get("discord_bot_token", {})
claim("OR-2.3 discord_bot_token: verifier copy, bound to discord.com, gateway.discord.gg, *.discord.gg",
      k.get("verifier") is True and sorted(k.get("dest", [])) == sorted(["discord.com", "gateway.discord.gg", "*.discord.gg"]),
      f"KNOWN = {k}")
# --- Discord scope
bits = DP.bits_of(DP.DEFAULT_SCOPE)
claim("OR-2.4 DEFAULT_SCOPE: 18 permissions, integer 2269727275740401", len(DP.DEFAULT_SCOPE) == 18 and bits == 2269727275740401,
      f"{len(DP.DEFAULT_SCOPE)} permissions, integer {bits}")
doc_never = ["ADMINISTRATOR", "KICK_MEMBERS", "BAN_MEMBERS", "MODERATE_MEMBERS", "MANAGE_GUILD", "MANAGE_ROLES",
             "MANAGE_WEBHOOKS", "MENTION_EVERYONE"]
claim("OR-2.4 NEVER_GRANTED as listed", sorted(DP.NEVER_GRANTED) == sorted(doc_never), f"NEVER_GRANTED = {DP.NEVER_GRANTED}")
# --- planning task values (OPERATOR_RESOURCES 1, 2)
raw = load_task_raw("cam_live/tasks/discord_100_planning.yaml")
t = parse_task(raw)
lim = t.limits
claim("OR-2 planning limits 6 h / 120 decisions / 60 external / 7200 compute-s / money 1",
      (lim.duration_s, lim.max_decisions, lim.max_external_actions, lim.compute_s, lim.money) == (21600, 120, 60, 7200, 1),
      f"{(lim.duration_s, lim.max_decisions, lim.max_external_actions, lim.compute_s, lim.money)}")
claim("OR-2 planning interfaces structured_browser, code, api, internal",
      sorted(t.interfaces) == sorted(["structured_browser", "code", "api", "internal"]), f"{t.interfaces}")
d = (raw.get("resources") or {}).get("discord") or {}
claim("OR-1 guild 1575680989060010034 / application 1575681780462129232",
      str(d.get("guild_id")) == "1575680989060010034" and str(d.get("application_id")) == "1575681780462129232",
      f"guild {d.get('guild_id')}, application {d.get('application_id')}")
raw100 = load_task_raw("cam_live/tasks/discord_100.yaml")
claim("OR-3 execution task adds interfaces visual and skill", {"visual", "skill"} <= set(raw100.get("interfaces") or []),
      f"{raw100.get('interfaces')}")
# --- offline preflight item counts (OPERATOR_RESOURCES 6): 9 and 25, without the gpu item when CUDA is hidden
for f, want in (("discord_100_planning", 9), ("discord_100", 25)):
    rep = PF.preflight(f"cam_live/tasks/{f}.yaml", CamLiveConfig.load())
    items = [m["item"] for m in rep["live"]["missing"]]
    got = len([i for i in items if i != "gpu"])
    claim(f"OR-6 offline preflight {f}: {want} live items (excluding the 'gpu' item, present only when CUDA is hidden)",
          got == want, f"{len(items)} items, {got} without 'gpu'; accepted_risks={rep['live'].get('accepted_risks')}")
    rep2 = PF.preflight(f"cam_live/tasks/{f}.yaml", CamLiveConfig.load(), accept_risks=["persistent_storage"])
    left = [m["item"] for m in rep2["live"]["missing"]]
    claim(f"OR-0 {f}: persistent_storage cannot be accepted by the operator", "persistent_storage" in left,
          f"with accept_risks=[persistent_storage]: still listed = {'persistent_storage' in left}; accepted = "
          f"{[m['item'] for m in rep2['live'].get('accepted_risks', [])]}")
# --- storage numbers (OPERATOR_RESOURCES 2.1, 4)
st = cfg_raw.get("storage") or {}
claim("OR-2.1 storage.min_free_gb 20, runtime.keep_checkpoints 2", st.get("min_free_gb") == 20 and rt.get("keep_checkpoints") == 2,
      f"storage {st}, keep_checkpoints {rt.get('keep_checkpoints')}")
# --- 'cam-live approve --run-dir R <approval_id>' (OPERATOR_RESOURCES 2.6)
sub = p._subparsers._group_actions[0].choices
claim("OR-2.6 approve / reject take an approval id", all(any(a.dest in ("approval_id", "id") for a in sub[s]._actions)
                                                       for s in ("approve", "reject")),
      f"approve args {[a.dest for a in sub['approve']._actions]}")
print(f"summary: {n['HOLDS']} HOLD, {n['BROKEN']} BROKEN")
