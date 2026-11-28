"""G2-runtime probe: live preflight — staging components never count; the ephemeral-storage acknowledgement.

    .venv/bin/python cam_live/docs/verification/repro/g2runtime/probe_preflight.py

  1  `cam-live preflight --task tasks/discord_100.yaml` (as shipped): exit 3, and every unset path is named
  2  environment components that are local / staging services but are spelled so that preflight._labels does not
     recognise them (an environment whose top-level label says live): are they named as blocking?
  3  the Discord identity check uses the executor's own answer; the executor derives 'live' from its base URL
     (harness.discord.environment_of): spellings of a staging address never come out as live
  4  --accept-ephemeral-storage on the command line of `cam-live preflight` (live task, real doctor of this machine):
     accepted only for persistent_storage, recorded as accepted by the operator's command line; no effect without
     the flag or in staging mode; no other channel (environment variable, config, task file) enables it
"""
import json
import os
import subprocess
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g2common import PROJECT, SRC, Report, scratch  # noqa: E402

from cam_live.harness.discord import environment_of  # noqa: E402
from cam_live.runtime import preflight as PF  # noqa: E402
from cam_live.runtime import tasks as T  # noqa: E402

R = Report("probe_preflight")
tmp = scratch("pre")
HOME = os.path.join(tmp, "home")
ENV = dict(os.environ, CAM_LIVE_HOME=HOME, PYTHONPATH=SRC)


def cli(*args, env=None):
    p = subprocess.run([sys.executable, "-m", "cam_live.runtime.cli", *args], capture_output=True, text=True,
                       env=env or ENV, timeout=600)
    return p.returncode, p.stdout, p.stderr


# ---- 1 --------------------------------------------------------------------------------------------------------
shipped = os.path.join(PROJECT, "tasks", "discord_100.yaml")
code, out, err = cli("preflight", "--task", shipped, "--json")
rep = json.loads(out) if out.strip().startswith("{") else {}
items = {m["item"] for m in (rep.get("live") or {}).get("missing", [])}
unset = T.unset_paths(T.load_task_raw(shipped))
R.check("1 shipped Discord task: exit 3, LIVE blocked, every REQUIRED path named", code == 3 and rep["live"]["blocked"]
        and all(p in items or any(p.startswith(i) for i in items) for p in unset),
        {"exit": code, "unset_paths": len(unset), "not_named": [p for p in unset if p not in items], "missing": len(items)})

# ---- 2 --------------------------------------------------------------------------------------------------------
spellings = {"environment 'staging-discord'": {"executor": {"environment": "staging-discord"}},
             "environment 'Staging ' (trailing space)": {"executor": {"environment": "Staging "}},
             "base_url http://0.0.0.0:18471": {"executor": {"base_url": "http://0.0.0.0:18471"}},
             "base_url http://localhost.:18471 (trailing dot)": {"executor": {"base_url": "http://localhost.:18471"}},
             "base_url http://[::ffff:127.0.0.1]:18471": {"executor": {"base_url": "http://[::ffff:127.0.0.1]:18471"}},
             "base_url http://10.1.2.3:18471 (private network)": {"executor": {"base_url": "http://10.1.2.3:18471"}},
             "api_base http://127.0.0.1:18471 (key not inspected)": {"executor": {"api_base": "http://127.0.0.1:18471"}},
             "control: base_url http://127.0.0.1:18471": {"executor": {"base_url": "http://127.0.0.1:18471"}}}
missed = []
for label, comp in spellings.items():
    caps = {"live": True, "provenance": "live", "components": comp}
    hits = PF._labels(caps)
    if not hits:
        missed.append(label)
R.check("2 every local / staging component of a live-labelled environment is named as blocking", not missed,
        {"not recognised": missed})

# ---- 3 --------------------------------------------------------------------------------------------------------
bad = [u for u in ("http://127.0.0.1:18471", "http://0.0.0.0:18471", "http://localhost.:18471",
                   "https://discord.com.evil.example", "https://discordapp.com.localhost", "http://[::1]:18471")
       if environment_of(u) == "live"]
R.check("3 no staging / local Discord base URL is classified live by the executor", not bad, bad)

# ---- 4 --------------------------------------------------------------------------------------------------------
raw = T.load_task_raw(shipped)
raw["success"]["exclusions"]["test_account_ids"] = []
raw["resources"].update(discord_guild_id="123456789012345678", discord_bot_credential_name="bot",
                        owned_site="https://example.invalid", operator_authorization_record="rec-1",
                        authorized_tools=["discord.*"], authorized_destinations=["example.invalid"], currency="USD")
raw["resources"]["publishing"].update(authorized=False, destinations=[])
raw["limits"] = {"duration_s": 86400, "money": 10, "compute_s": 3600, "max_external_actions": 100, "max_decisions": 100}
raw["data_permissions"]["collect_feedback_with_consent"] = False
live = os.path.join(tmp, "live.yaml")
yaml.safe_dump(raw, open(live, "w"))
raw_s = dict(raw, mode="staging")
stg = os.path.join(tmp, "staging.yaml")
yaml.safe_dump(raw_s, open(stg, "w"))


def report(*a, env=None):
    c, o, e = cli("preflight", "--json", *a, env=env)
    return c, (json.loads(o) if o.strip().startswith("{") else {"err": e[-300:]})


c0, plain = report("--task", live)
storage_blocks = "persistent_storage" in {m["item"] for m in plain["live"]["missing"]}
R.note("this machine's persistent_storage check blocks live", storage_blocks)
c1, acc = report("--task", live, "--accept-ephemeral-storage")
names = [m["item"] for m in acc["live"]["accepted_risks"]]
R.check("4a with the flag: persistent_storage moves to accepted_risks (recorded with the command-line flag); nothing "
        "else is accepted; live stays blocked by the other items",
        (not storage_blocks or (names == ["persistent_storage"] and "--accept-ephemeral-storage"
                                in acc["live"]["accepted_risks"][0]["accepted_by"])) and acc["live"]["blocked"]
        and c1 == 3, {"accepted": acc["live"]["accepted_risks"], "exit": c1})
c2, envtry = report("--task", live, env=dict(ENV, CAM_LIVE_ACCEPT_EPHEMERAL_STORAGE="1", ACCEPT_EPHEMERAL_STORAGE="1"))
R.check("4b no environment variable enables it", envtry["live"]["accepted_risks"] == [], envtry["live"]["accepted_risks"])
c3, st = report("--task", stg, "--accept-ephemeral-storage")
R.check("4c no effect in staging mode", st["live"]["accepted_risks"] == [], st["live"]["accepted_risks"])
raw_x = dict(raw, accept_ephemeral_storage=True)
xp = os.path.join(tmp, "live_x.yaml")
yaml.safe_dump(raw_x, open(xp, "w"))
c4, x = report("--task", xp)
R.check("4d a task file cannot carry it (unknown key refused)", c4 == 3 and any("accept_ephemeral_storage" in m["detail"]
        for m in x["live"]["missing"]), [m["detail"] for m in x["live"]["missing"]][:2])
sys.exit(R.done())
