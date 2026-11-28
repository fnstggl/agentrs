"""G2-final probe: the NEW statements of docs/REPRODUCE.md and docs/OPERATOR_RESOURCES.md (DOC refresh to fdab7d3)
against the code of a tree. G2-integration's d03 / d02 encode the previous documents; this probe reads the current
ones and checks what they say now.

    .venv/bin/python cam_live/docs/verification/repro/g2final/probe_docs_commands.py <TREE> <SCRATCH>

TREE = a `git archive` extract containing cam_live/ (documents and code); SCRATCH = an empty directory (CAM_LIVE_HOME,
logs, a scratch operator key). Offline only: nothing is started as a service, nothing is signed for a real task (the
typed confirmation is answered wrongly on purpose), no network, no GPU work (the doctor only reads the GPU's presence).
  H   every command the two documents name — each `cam-live <sub>`, each `python -m <module> [<sub>]`, each script under
      cam_live/tools or docs/evidence — answers --help with exit 0 (the documented exceptions: secret_scan.py exit 2,
      launch_live.sh without arguments exit 2 with its header; gen_manifest / merge_verdicts are not started)
  E   documented executions and numbers: per-directory collection counts and the one-process total, `cam-live test
      --suite storage`, doctor checks, the preflight table of every task file (exit code, LIVE item counts, the 8
      planning items, accepted risk), inspect-harness --lab, run --preflight-only, validate --mode live refused,
      launch_live.sh refusing a non-interactive terminal before it touches anything, authorize_live.py exit 3 / 4,
      task values / limits / scope integer / configuration values quoted in OPERATOR_RESOURCES.md
"""
import json
import os
import re
import subprocess
import sys

TREE, SCR = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
CL = os.path.join(TREE, "cam_live")
SRC = os.path.join(CL, "src")
PY = "/workspace/camlive/.venv/bin/python"
os.makedirs(SCR, exist_ok=True)
ENV = dict(os.environ, PYTHONPATH=SRC, CAM_LIVE_HOME=os.path.join(SCR, "home"), PYTHONDONTWRITEBYTECODE="1",
           HF_HUB_OFFLINE="1", CAM_LIVE_GPU_LOCK="/workspace/agentrs/runs/cam_unified/gpu_heavy.lock")
ENV_NOGPU = dict(ENV, CUDA_VISIBLE_DEVICES="")
n = {"HOLDS": 0, "BROKEN": 0}
rows = []


def claim(cid, ok, what):
    k = "HOLDS" if ok else "BROKEN"
    n[k] += 1
    rows.append((k, cid, what))
    print(f"{k:6}  {cid}: {what}", flush=True)


def run(argv, env=ENV, timeout=900, cwd=None, stdin=subprocess.DEVNULL, inp=None):
    try:
        p = subprocess.run(argv, env=env, cwd=cwd or TREE, capture_output=True, text=True, timeout=timeout,
                           stdin=None if inp is not None else stdin, input=inp)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return -9, "", "timeout"


docs = {name: open(os.path.join(CL, "docs", name)).read() for name in ("REPRODUCE.md", "OPERATOR_RESOURCES.md")}
text = "\n".join(docs.values())

# ---- H: --help of every named command -------------------------------------------------------------------------------
subs = sorted(set(re.findall(r"(?<![\w/-])cam-live ([a-z][a-z-]+)", text)) - {"run's", "is"})
mods = set()
for m in re.finditer(r"(?:\$P|python3?|\.venv/bin/python|\$PY) -m\s+(cam_live(?:\.[a-z_0-9]+)+)(?: ([a-z][a-z_-]+))?", text):
    mods.add((m.group(1), None))
    if m.group(2):
        mods.add((m.group(1), m.group(2)))
# the inventory of REPRODUCE.md §13 names modules without "python -m" and eval.cli subcommands joined by " · "
inv = rep_inv = docs["REPRODUCE.md"].split("## 13.")[1].split("## 14.")[0]
for m in re.findall(r"`((?:runtime|harness|eval|nn|predict|reward|credit)\.[a-z_0-9.]+)`", inv):
    mods.add(("cam_live." + m, None))
for sub in ("run", "batch", "transfer-batch", "anchors", "analyse", "integrity", "meta", "scaling", "large"):
    mods.add(("cam_live.eval.cli", sub))
for sub in ("sequence", "transfer"):
    mods.add(("cam_live.eval.oe_runner", sub))
mods.discard(("cam_live.harness.grounding.worker", None))     # internal child, documented as not run
scripts = sorted(set(re.findall(r"(cam_live/(?:tools|docs/evidence)/[\w/]+\.py)", text)))
for s in subs:
    rc, out, err = run([PY, "-m", "cam_live.runtime.cli", s, "--help"], env=ENV_NOGPU, timeout=300)
    claim(f"H cam-live {s} --help", rc == 0, f"exit {rc}" + (f": {err.strip().splitlines()[-1][:120]}" if rc else ""))
for mod, sub in sorted(mods, key=lambda x: (x[0], x[1] or "")):
    argv = [PY, "-m", mod] + ([sub] if sub else []) + ["--help"]
    rc, out, err = run(argv, env=ENV_NOGPU, timeout=600)
    if sub and rc != 0 and ("invalid choice" in err or "unrecognized" in err):
        # the token after the module is a positional value, not a subcommand (e.g. 'set discord_bot_token'): check
        # that the module's own usage names it
        rc2, out2, _ = run([PY, "-m", mod, "--help"], env=ENV_NOGPU, timeout=600)
        claim(f"H {mod} {sub} (positional)", rc2 == 0 and sub in out2, f"module --help exit {rc2}, names {sub!r}: "
                                                                        f"{sub in out2}")
        continue
    claim(f"H {mod}{' ' + sub if sub else ''} --help", rc == 0,
          f"exit {rc}" + (f": {(err or out).strip().splitlines()[-1][:120]}" if rc else ""))
for sc in scripts:
    path = os.path.join(TREE, sc)
    if not os.path.exists(path):
        claim(f"H {sc} exists", False, "named in the documents, not in the tree")
        continue
    if sc.endswith(("gen_manifest.py", "merge_verdicts.py")):
        continue                                    # documented as not started (they write their files)
    rc, out, err = run([PY, path, "--help"], env=ENV_NOGPU, timeout=300)
    if sc.endswith("secret_scan.py"):
        claim(f"H {sc} --help", rc == 2, f"documented: usage line, exit 2; got exit {rc}")
    else:
        claim(f"H {sc} --help", rc == 0, f"exit {rc}")
rc, out, err = run(["bash", os.path.join(CL, "tools", "launch_live.sh")])
claim("H launch_live.sh without arguments: header, exit 2", rc == 2 and "launch_live.sh" in out, f"exit {rc}")
for sub in ("prepare", "serve", "holds", "check"):
    rc, out, err = run([PY, os.path.join(CL, "tools", "planning_rehearsal.py"), sub, "--help"], env=ENV_NOGPU)
    claim(f"H planning_rehearsal.py {sub} --help", rc == 0, f"exit {rc}")

# ---- E: executions and numbers --------------------------------------------------------------------------------------
rep = docs["REPRODUCE.md"]
table = dict(re.findall(r"\| `([a-z_/.]+)` \| (\d+) \|", rep))
for row in re.findall(r"\| `([a-z_./]+)`[^|]* \| (\d+) \| `([a-z_./]+)` \| (\d+) \|", rep):
    table[row[0]], table[row[2]] = row[1], row[3]
sub_dirs = dict(re.findall(r"`harness/(auth|compute|env|skills)` (\d+)", rep))      # the breakdown line
total = int(re.search(r"\*\*(\d+)\*\* \(= the sum", rep).group(1))
got_sum = 0
for d, want in sorted(table.items()):
    p = os.path.join(CL, "tests", d.split(" ")[0])
    if not os.path.exists(p):
        continue
    rc, out, err = run([PY, "-m", "pytest", os.path.join("cam_live", "tests", d), "--collect-only", "-q",
                        "-p", "no:cacheprovider", "--rootdir=cam_live"], env=ENV_NOGPU, timeout=900)
    m = re.search(r"(\d+) tests? collected", out)
    got = int(m.group(1)) if m else -1
    got_sum += max(got, 0)
    claim(f"E collect {d}", rc == 0 and got == int(want), f"documented {want}, collected {got} (exit {rc})")
for sd, want in sorted(sub_dirs.items()):
    rc, out, err = run([PY, "-m", "pytest", os.path.join("cam_live", "tests", "harness", sd), "--collect-only", "-q",
                        "-p", "no:cacheprovider", "--rootdir=cam_live"], env=ENV_NOGPU, timeout=900)
    m = re.search(r"(\d+) tests? collected", out)
    claim(f"E collect harness/{sd}", rc == 0 and m and int(m.group(1)) == int(want),
          f"documented {want}, collected {m.group(1) if m else None} (exit {rc})")
rc, out, err = run([PY, "-m", "pytest", "cam_live/tests", "--collect-only", "-q", "-p", "no:cacheprovider",
                    "--rootdir=cam_live"], env=ENV_NOGPU, timeout=1800)
m = re.search(r"(\d+) tests? collected", out)
claim("E whole suite collected in one process", rc == 0 and m and int(m.group(1)) == total == got_sum,
      f"documented {total} (= sum); collected {m.group(1) if m else None}, sum of directories {got_sum}, exit {rc}")
rc, out, err = run([PY, "-m", "cam_live.runtime.cli", "test", "--suite", "storage", "--results-dir",
                    os.path.join(SCR, "res")], env=ENV_NOGPU, timeout=900)
claim("E cam-live test --suite storage: exit 0, 4 passed", rc == 0 and "4 passed" in out + err, f"exit {rc}")
rc, out, err = run([PY, "-m", "cam_live.runtime.cli", "doctor", "--json"], env=ENV, timeout=900)
try:
    summ = json.loads(out)["summary"]
except Exception:  # noqa: BLE001
    summ = {}
claim("E doctor: 52 checks, 49 ok; failed persistent_storage, kernel_network_filtering, user_namespaces; blocks "
      "live only persistent_storage", summ.get("checks") == 52 and summ.get("ok") == 49 and
      sorted(summ.get("failed", [])) == ["kernel_network_filtering", "persistent_storage", "user_namespaces"] and
      summ.get("blocking_live") == ["persistent_storage"], summ)
PF = {  # task file: (flag, documented exit, documented LIVE count)
    "discord_100_planning.yaml": [("", 3, 9), ("--accept-ephemeral-storage", 3, 8)],
    "discord_launch_check.yaml": [("", 3, 9), ("--accept-ephemeral-storage", 3, 8)],
    "discord_100.yaml": [("", 3, 25), ("--accept-ephemeral-storage", 3, 24)],
    "staging/discord_100_planning_staging.yaml": [("", 0, 10)],
    "lab_staging.yaml": [("", 0, None)],
    "utility_staging.yaml": [("", 3, 8)],
}
PLAN8 = ["authorization service", "discord_bot_token", "computer environment", "verifier credential copy",
         "bot membership in the authorized guild", "privileged intents GUILD_MEMBERS",
         "the bot's permissions against the operator's scope", "independent verifier"]
for tf, cases in PF.items():
    for flag, want_rc, want_n in cases:
        rc, out, err = run([PY, "-m", "cam_live.runtime.cli", "preflight", "--task", f"cam_live/tasks/{tf}", "--json"]
                           + ([flag] if flag else []), env=ENV, timeout=900)
        try:
            d = json.loads(out)
            miss = [x.get("item") for x in d["live"]["missing"]]
            acc = [x.get("item") for x in d["live"]["accepted_risks"]]
            st = d.get("staging") or {}
        except Exception:  # noqa: BLE001
            miss, acc, st = None, None, {}
        ok = rc == want_rc and (want_n is None or (miss is not None and len(miss) == want_n))
        extra = ""
        if tf == "discord_100_planning.yaml" and flag:
            ok = ok and miss == PLAN8 and acc == ["persistent_storage"]
            extra = f"; items {miss}; accepted {acc}"
        if tf == "utility_staging.yaml":
            smiss = [x.get("item") for x in (st.get("missing") or [])]
            ok = ok and sorted(smiss) == ["resources.staging_site", "success.evidence_source"]
            extra = f"; staging missing {smiss}"
        claim(f"E preflight {tf} {flag or '(no flag)'}", ok,
              f"documented exit {want_rc}, LIVE {want_n}; got exit {rc}, LIVE {None if miss is None else len(miss)}"
              + extra)
rc, out, err = run([PY, "-m", "cam_live.runtime.cli", "inspect-harness", "--lab"], env=ENV_NOGPU, timeout=600)
claim("E inspect-harness --lab: exit 0, 5 LabWorld tools, Discord missing", rc == 0 and out.count("lab.") >= 5
      and "Discord" in out, f"exit {rc}, 'lab.' x{out.count('lab.')}")
rc, out, err = run([PY, "-m", "cam_live.runtime.cli", "run", "--task", "cam_live/tasks/lab_staging.yaml",
                    "--preflight-only"], env=ENV_NOGPU, timeout=600)
claim("E run --task lab_staging.yaml --preflight-only: exit 0", rc == 0, f"exit {rc}")
rc, out, err = run([PY, "-m", "cam_live.runtime.cli", "validate", "--mode", "live"], env=ENV_NOGPU, timeout=300)
claim("E validate --mode live: refused, exit 2", rc == 2, f"exit {rc}")
runs_before = sorted(os.listdir(os.path.join(TREE, "runs"))) if os.path.isdir(os.path.join(TREE, "runs")) else []
rc, out, err = run(["bash", os.path.join(CL, "tools", "launch_live.sh"), "--accept-ephemeral-storage",
                    "cam_live/tasks/discord_launch_check.yaml", "1"], stdin=subprocess.DEVNULL)
runs_after = sorted(os.listdir(os.path.join(TREE, "runs"))) if os.path.isdir(os.path.join(TREE, "runs")) else []
claim("E launch_live.sh refuses a non-interactive terminal (exit 2) before it creates or stores anything",
      rc == 2 and "interactive terminal" in err and runs_before == runs_after, f"exit {rc}; runs/ {runs_after}")
key = os.path.join(SCR, "op_ed25519")
rc, out, err = run([PY, "-m", "cam_live.harness.auth.opctl", "init-operator-key", "--private", key, "--public",
                    key + ".pub"], env=ENV_NOGPU)
claim("E opctl init-operator-key (scratch key)", rc == 0 and os.path.exists(key), f"exit {rc}")
rc, out, err = run([PY, os.path.join(CL, "tools", "authorize_live.py"), "--task", "cam_live/tasks/lab_staging.yaml",
                    "--private", key, "--hours", "1", "--out", os.path.join(SCR, "x.signed.json")], env=ENV_NOGPU)
claim("E authorize_live.py refuses a task that is not mode live (exit 3)", rc == 3, f"exit {rc}")
rc, out, err = run([PY, os.path.join(CL, "tools", "authorize_live.py"), "--task",
                    "cam_live/tasks/discord_100_planning.yaml", "--private", key, "--hours", "1", "--out",
                    os.path.join(SCR, "y.signed.json")], env=ENV_NOGPU, inp="not-the-objective-id\n")
claim("E authorize_live.py: a wrong typed objective id exits 4 'not signed' and writes nothing",
      rc == 4 and not os.path.exists(os.path.join(SCR, "y.signed.json")), f"exit {rc}; {(out + err).strip()[-80:]}")

sys.path.insert(0, SRC)
import yaml  # noqa: E402
from cam_live.harness import SERVICE_MODULES  # noqa: E402
from cam_live.harness.auth.launcher import ENV_PASSTHROUGH, START_ORDER  # noqa: E402
from cam_live.harness.discord_perms import DEFAULT_SCOPE, NEVER_GRANTED, bits_of  # noqa: E402
from cam_live.runtime import tasks as T  # noqa: E402
from cam_live.runtime.preflight import ACCEPTABLE_RISKS  # noqa: E402

orr = docs["OPERATOR_RESOURCES.md"]
plan = T.load_task_raw(os.path.join(CL, "tasks", "discord_100_planning.yaml"))
claim("E OR §2 planning limits 21600 s / 120 decisions / 60 external / 7200 compute-s / money 1; interfaces",
      plan["limits"] == {"duration_s": 21600, "money": 1, "compute_s": 7200, "max_external_actions": 60,
                         "max_decisions": 120} and plan["interfaces"] == ["structured_browser", "code", "api", "internal"],
      {"limits": plan["limits"], "interfaces": plan["interfaces"]})
lc = T.load_task_raw(os.path.join(CL, "tasks", "discord_launch_check.yaml"))
claim("E OR §3 launch check: one decision, two external actions, 30 min",
      (lc["limits"]["max_decisions"], lc["limits"]["max_external_actions"], lc["limits"]["duration_s"]) == (1, 2, 1800),
      lc["limits"])
unset = T.unset_paths(T.load_task_raw(os.path.join(CL, "tasks", "discord_100.yaml")))
listed = set(re.findall(r"`((?:limits|resources|success|data_permissions)\.[a-z_.]+)`", orr.split("## 5.")[1].split("## 6.")[0]))
lim5 = {f"limits.{k}" for k in ("duration_s", "money", "compute_s", "max_external_actions", "max_decisions")}
claim("E OR §5: 16 unset REQUIRED values, the paths of the table", len(unset) == 16 and set(unset) <= (listed | lim5),
      {"unset": len(unset), "not in the table": sorted(set(unset) - listed - lim5)})
claim("E OR §6 scope: 18 permissions, integer 2269727275740401; NEVER_GRANTED as listed",
      len(DEFAULT_SCOPE) == 18 and bits_of(DEFAULT_SCOPE) == 2269727275740401 and set(NEVER_GRANTED) ==
      {"ADMINISTRATOR", "KICK_MEMBERS", "BAN_MEMBERS", "MODERATE_MEMBERS", "MANAGE_GUILD", "MANAGE_ROLES",
       "MANAGE_WEBHOOKS", "MENTION_EVERYONE"}, {"n": len(DEFAULT_SCOPE), "bits": bits_of(DEFAULT_SCOPE)})
cfg = yaml.safe_load(open(os.path.join(CL, "configs", "cam_live.yaml")))
claim("E OR §2.2/2.3: gpu_hold_s 840, gpu_yield_s 15, keep_checkpoints 2, min_free_gb 20, warn_free_gb 50",
      (cfg["runtime"]["gpu_hold_s"], cfg["runtime"]["gpu_yield_s"], cfg["runtime"]["keep_checkpoints"],
       cfg["storage"]["min_free_gb"], cfg["storage"]["warn_free_gb"]) == (840, 15, 2, 20, 50),
      {"runtime": cfg["runtime"], "storage": cfg["storage"]})
claim("E OR §2.1: ACCEPTABLE_RISKS is exactly persistent_storage -> --accept-ephemeral-storage",
      ACCEPTABLE_RISKS == {"persistent_storage": "--accept-ephemeral-storage"}, ACCEPTABLE_RISKS)
claim("E OR §2.2: the launcher passes CAM_LIVE_GPU_LOCK to the services", "CAM_LIVE_GPU_LOCK" in ENV_PASSTHROUGH, "")
claim("E OR §2.4 / RE §9.1: five services, start order auth, verifier, sandbox, exec, learner",
      set(SERVICE_MODULES) == {"auth", "exec", "sandbox", "verifier", "learner"} and
      tuple(START_ORDER) == ("auth", "verifier", "sandbox", "exec", "learner"), {"start": START_ORDER})
rc, out, err = run([PY, "-m", "cam_live.runtime.cli", "run", "--help"], env=ENV_NOGPU)
claim("E RE §9.1 / OR §3: cam-live run --start-timeout defaults to 120 s",
      re.search(r"p\.add_argument\(\"--start-timeout\", type=float, default=120\.0\)",
                open(os.path.join(SRC, "cam_live", "runtime", "cli.py")).read()) is not None, "")
sh = open(os.path.join(CL, "tools", "launch_live.sh")).read()
claim("E launch_live.sh: --accept-ephemeral-storage only as the first argument (line 31), refuses an existing "
      "state.json (exit 3), passes no --start-timeout and no --yes",
      sh.splitlines()[30].startswith('if [ "${1:-}" = "--accept-ephemeral-storage" ]') and "exit 3" in sh and
      "--start-timeout" not in sh and "--yes" not in sh, "")
print(f"summary: {n['HOLDS']} HOLD, {n['BROKEN']} BROKEN")
json.dump(rows, open(os.path.join(SCR, "probe_docs_commands.json"), "w"), indent=1)
sys.exit(1 if n["BROKEN"] else 0)
