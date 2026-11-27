"""G-rt probe 05 — task loader: strategy content in disguise, unset REQUIRED values in live mode, degenerate limits.

    .venv/bin/python cam_live/docs/verification/repro/rt/p05_tasks.py
"""
import copy
import math
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from standin import Report, scratch  # noqa: E402

from cam_live.config import PROJECT_ROOT  # noqa: E402
from cam_live.runtime import tasks as T  # noqa: E402

R = Report("p05_tasks")
DISCORD = os.path.join(PROJECT_ROOT, "tasks", "discord_100.yaml")
tmp = scratch("p05")


def filled():
    raw = T.load_task_raw(DISCORD)
    raw["success"]["evidence_source"] = "g_fake_module:live_source"
    raw["success"]["exclusions"]["test_account_ids"] = []
    raw["resources"].update(discord_guild_id="G1", discord_bot_credential_name="bot", owned_site="https://example.invalid",
                            operator_authorization_record="rec-1", authorized_tools=["api"], authorized_destinations=["G1"],
                            currency="USD")
    raw["resources"]["publishing"].update(authorized=True, destinations=["G1"])
    raw["limits"] = {"duration_s": 86400, "money": 10, "compute_s": 3600, "max_external_actions": 100, "max_decisions": 100}
    raw["data_permissions"]["collect_feedback_with_consent"] = False
    return raw


def loads(raw, for_run=True):
    p = os.path.join(tmp, "t.yaml")
    with open(p, "w") as f:
        yaml.safe_dump(raw, f)
    try:
        return T.load_task(p, for_run=for_run), []
    except T.TaskValidationError as e:
        return None, e.problems


def loads_text(text):
    p = os.path.join(tmp, "t_text.yaml")
    with open(p, "w") as f:
        f.write(text)
    try:
        return T.load_task(p, for_run=True), []
    except (T.TaskValidationError, yaml.YAMLError) as e:
        return None, getattr(e, "problems", [str(e)])


# ---- 0. the shipped Discord task -------------------------------------------------------------------------
raw0 = T.load_task_raw(DISCORD)
R.check("0a tasks/discord_100.yaml: mechanical scan finds no strategy", T.scan_file(DISCORD) == [])
spec_obj = ("Create and operate a genuinely useful Discord community that attracts 100 distinct real human members who "
            "voluntarily choose to join, within the authorized experimental constraints.")
R.check("0b its objective is the specification's text verbatim", raw0["objective"] == spec_obj)
comments = [ln for ln in open(DISCORD) if "#" in ln]
R.note("0c comment lines of the Discord task (read by the verifier; none prescribes a niche, channel, schedule or step)",
       len(comments))
t, probs = loads(raw0)
R.check("0d the Discord task as shipped (REQUIRED items unset) cannot be loaded for a run", t is None, len(probs))
t, probs = loads(filled())
R.check("0e baseline: the completely filled task loads (so the rejections below are caused by the planted content)",
        t is not None, probs)

# ---- 1. each REQUIRED item left unset, one at a time ---------------------------------------------------
unset = T.unset_paths(raw0)
R.check("1a number of REQUIRED items in the Discord task equals the status file's 17", len(unset) == 17, len(unset))
leaks = []
for path in unset:
    raw = filled()
    node, parts = raw, path.split(".")
    for k in parts[:-1]:
        node = node[k]
    node[parts[-1]] = "REQUIRED"
    t, probs = loads(raw)
    if t is not None:
        leaks.append(path)
R.check("1b a live task with any single REQUIRED item unset is refused", not leaks, {"accepted_with_unset": leaks})
soft = {}
for marker in ("TODO", "TBD", "<fill in>", "???", "changeme", "required"):
    raw = filled()
    raw["resources"]["discord_guild_id"] = marker
    soft[marker] = loads(raw)[0] is None
R.note("1c other placeholder spellings treated as unset (only REQUIRED/empty is the documented marker)", soft)

# ---- 2. degenerate limits in live mode ----------------------------------------------------------------------
for name, val in (("NaN", float("nan")), ("infinity", float("inf")), ("zero", 0), ("negative", -5), ("string", "3600"),
                  ("boolean", True)):
    raw = filled()
    raw["limits"]["money"] = val
    t, probs = loads(raw)
    enforced = t is not None and t.limits.money > 0 and math.isfinite(t.limits.money)
    R.check(f"2  live task with limits.money = {name}: refused, or loaded as a finite enforceable limit",
            t is None or enforced, "accepted with limit " + repr(t.limits.money) if t is not None else "refused")

# ---- 3. strategy content in disguise ------------------------------------------------------------------------
def plant(fn):
    raw = filled()
    fn(raw)
    return raw


CASES = {
    "3a extra top-level key 'strategy'": lambda r: r.update(strategy="open a server for retro handheld fans"),
    "3b nested key 'growth_plan'": lambda r: r["resources"].update(growth_plan={"a": "b"}),
    "3c forbidden key deep in a list of mappings": lambda r: r["resources"].update(extra=[{"x": [{"niche": "anime"}]}]),
    "3d camelCase key 'growthPlan'": lambda r: r["resources"].update(growthPlan="open a server for retro handheld fans"),
    "3e plural key 'strategies'": lambda r: r["resources"].update(strategies=["retro handheld fans", "cross-promotion"]),
    "3f key 'milestones' holding an ordered list of moves": lambda r: r["resources"].update(milestones=[
        "Open a server about retro handheld consoles",
        "Write two guides per week and link the invite under each one in r/handhelds",
        "Run a giveaway once thirty members have joined"]),
    "3g key 'todo' holding moves": lambda r: r["measurement"].update(todo=["Open a server for anime fans",
                                                                            "Ask partner servers for cross-promotion"]),
    "3h community subject and channel inside the objective text": lambda r: r.update(
        objective=r["objective"] + " The community is for indie game developers and members are recruited through devlog threads on Reddit."),
    "3i recruitment channel inside success.definition": lambda r: r["success"].update(
        definition="A member counts when listed by the platform. Members are recruited through AMA threads in r/gamedev."),
    "3j subject and tactic inside 'family'": lambda r: r.update(family="community/membership: anime fans, cross-promote with partner servers"),
    "3k a move as a list item of secondary_metrics": lambda r: r["secondary_metrics"].append("post memes for teenagers in the evening hours"),
    "3l a move as a list item of interfaces": lambda r: r["interfaces"].append("api: use it to DM server owners for cross-promotion"),
    "3m the whole procedure as a mapping KEY (keys are not text-scanned)": lambda r: r["resources"].update(
        notes={"first open an anime server then post daily on reddit": True}),
    "3n numbered procedure as mapping keys": lambda r: r["resources"].update(
        notes={"1. open an anime server": 1, "2. post in r/anime": 2, "3. run a giveaway": 3}),
    "3o non-English text (Spanish) with schedule": lambda r: r["resources"].update(
        nota="Estrategia: abrir un servidor de anime y publicar cada dia en Reddit"),
    "3p directive phrased without the listed modal verbs": lambda r: r["resources"].update(
        note="CAM is expected to open a server for speedrunners and to recruit in speedrun forums"),
    "3q posting cadence without the listed timing words": lambda r: r["measurement"].update(
        cadence="three announcements per week, mornings, in r/gaming"),
    "3r 'suggested steps' key spelled 'suggested_steps'": lambda r: r["resources"].update(suggested_steps=["a", "b"]),
    "3s list of suggested steps under key 'outline'": lambda r: r["resources"].update(
        outline=["Create the server", "Write a welcome guide", "Share the invite in three subreddits"]),
}
accepted = []
for name, fn in CASES.items():
    t, probs = loads(plant(fn))
    if R.check(f"{name}: rejected", t is None, "" if t is None else "ACCEPTED and handed to agent.set_objective"):
        continue
    accepted.append(name)

# YAML anchors / merge keys
anchor_text = yaml.safe_dump(filled()).replace(
    "resources:\n", "resources:\n  base_notes: &n\n    focus: speedrunners, recruited in speedrun forums\n  more:\n    <<: *n\n", 1)
t, probs = loads_text(anchor_text)
R.check("3t content introduced through a YAML anchor + merge key: rejected", t is None,
        "" if t is None else "ACCEPTED: " + repr(t.resources.get("more")))
anchor2 = yaml.safe_dump(filled()).replace(
    "resources:\n", "resources:\n  a: &s\n  - step 1 open a server\n  b: *s\n", 1)
t, probs = loads_text(anchor2)
R.check("3u anchored list with 'step 1 ...' text: rejected", t is None, probs[:1])
R.note("3  disguises accepted", accepted)

# ---- 4. what an accepted disguise reaches -------------------------------------------------------------------
t, _ = loads(plant(CASES["3f key 'milestones' holding an ordered list of moves"]))
if t is not None:
    seen = {}

    class A:
        def set_objective(self, task):
            seen["resources"] = task.resources
    A().set_objective(t)
    R.check("4  an accepted disguise is at least withheld from the TaskSpec the learner receives",
            "milestones" not in seen["resources"], {"TaskSpec.resources.milestones": seen["resources"].get("milestones")})
sys.exit(R.finish())
