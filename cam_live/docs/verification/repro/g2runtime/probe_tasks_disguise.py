"""G2-runtime probe: can strategy content still reach the learner through a task file that load_task accepts?

    .venv/bin/python cam_live/docs/verification/repro/g2runtime/probe_tasks_disguise.py

Attack: a completely filled copy of tasks/discord_100.yaml (as in G-rt p05) with strategy content written so that the
closed schema accepts it (identifiers and labels joined with '_' / '-', which the schema allows) or so that the
regular expressions of the text scan do not match (zero-width characters inside the words the patterns look for).
For each variant: is it accepted by load_task(for_run=True), and does the content reach the learner
(runtime.tasks.learner_view, which is what agent.set_objective, run_started and task.json carry)?
Controls: the same content written plainly is rejected.
"""
import copy
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g2common import PROJECT, Report, scratch  # noqa: E402

from cam_live.runtime import tasks as T  # noqa: E402

R = Report("probe_tasks_disguise")
tmp = scratch("tasks")
DISCORD = os.path.join(PROJECT, "tasks", "discord_100.yaml")
ZW = "​"     # ZERO WIDTH SPACE: neither \w nor \s for Python's re


def filled():
    raw = T.load_task_raw(DISCORD)
    raw["success"]["exclusions"]["test_account_ids"] = []
    raw["resources"].update(discord_guild_id="123456789012345678", discord_bot_credential_name="bot",
                            owned_site="https://example.invalid", operator_authorization_record="rec-1",
                            authorized_tools=["discord.*", "http.*"], authorized_destinations=["example.invalid"],
                            currency="USD")
    raw["resources"]["publishing"].update(authorized=True, destinations=["example.invalid"])
    raw["limits"] = {"duration_s": 86400, "money": 10, "compute_s": 3600, "max_external_actions": 100,
                     "max_decisions": 100}
    raw["data_permissions"]["collect_feedback_with_consent"] = False
    return raw


def load(raw):
    p = os.path.join(tmp, "t.yaml")
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, allow_unicode=True, sort_keys=False)
    try:
        return T.load_task(p, for_run=True), []
    except T.TaskValidationError as e:
        return None, e.problems


base = filled()
t0, p0 = load(base)
R.check("0 the filled Discord task loads for a run (baseline)", t0 is not None, p0[:3])

PLAN_WORDS = ("meme", "gamer", "reddit", "r3ddit", "r e d d i t", "invite", "giveaway", "g1veaway", "daily", "strategy",
              "marketing", "crosspost", "dm_", "disboard", "top.gg", "bump", "r\u0435ddit")


def learner_text(task):
    return json.dumps(T.learner_view(task).to_dict(), ensure_ascii=False).replace(ZW, "")


variants = {
    "family (lowercase words joined by '-', no length bound)":
        lambda r: r.update(family="community/meme-server-for-pc-gamers/post-three-memes-then-invite-friends-personally/"
                                  "host-game-tournaments-with-prizes"),
    "secondary_metrics item (metric name joined by '_')":
        lambda r: r["secondary_metrics"].append("secondary.post_memes_in_gaming_subreddits_then_dm_new_joiners"),
    "secondary_metrics item with strategy vocabulary (\\b does not split '_')":
        lambda r: r["secondary_metrics"].append("secondary.reddit_marketing_strategy_with_daily_giveaways"),
    "interfaces item (tool pattern joined by '_' / '.')":
        lambda r: r["interfaces"].append("crosspost_memes_on_reddit.then_invite_gamers.*"),
    "resources.authorized_tools item":
        lambda r: r["resources"].update(authorized_tools=["discord.*", "step_one_create_meme_channel.step_two_invite_gamers"]),
    "resources.authorized_destinations item (URL query, '_' joined)":
        lambda r: r["resources"].update(authorized_destinations=[
            "example.invalid", "https://example.invalid/?plan=meme_server_for_gamers_then_crosspost_to_reddit_daily"]),
    "resources.discord.channels labels (one move per key)":
        lambda r: r["resources"].update(discord={"guild_id": "123456789012345678", "channels": {
            "start-meme-contest": "223456789012345678", "invite-gamer-friends": "323456789012345678",
            "host-game-tournaments": "423456789012345678", "dm-every-new-joiner": "523456789012345678"}}),
    "objective_id ('_' joined)":
        lambda r: r.update(objective_id="discord_100_gaming_memes_via_reddit_marketing_and_giveaways"),
    "objective text with zero-width spaces inside the matched words (one sentence)":
        lambda r: r.update(objective=("Create and operate a genuinely useful Discord community of pc g" + ZW + "amers that "
                                      "attracts 100 distinct real human members who voluntarily choose to join by "
                                      "posting memes on r" + ZW + "eddit e" + ZW + "very day and running "
                                      "give" + ZW + "aways, within the authorized experimental constraints")),
}
# ---- round 2 (final freeze): disguises that survive NFKC + invisible-character stripping + separator splitting ----
CY_E, CY_A = "\u0435", "\u0430"      # Cyrillic small ie / a: letters (category Ll), unchanged by NFKC
OBJ = "Create and operate a genuinely useful Discord community that attracts 100 distinct real human members who " \
      "voluntarily choose to join "
variants.update({
    "R2 objective with Cyrillic homoglyphs inside the matched words (one sentence)":
        lambda r: r.update(objective=OBJ + "by posting memes on r" + CY_E + "ddit " + CY_E + "very day and running "
                           "give" + CY_A + "ways, within the authorized experimental constraints"),
    "R2 objective in leetspeak":
        lambda r: r.update(objective=OBJ + "by posting memes on r3ddit 3very d4y and running g1veaways, within the "
                           "authorized experimental constraints"),
    "R2 objective naming Discord listing sites the lists do not contain":
        lambda r: r.update(objective=OBJ + "by listing it on disboard and top.gg and bumping it whenever the cooldown "
                           "allows, within the authorized experimental constraints"),
    "R2 objective with the platform name spelled letter by letter":
        lambda r: r.update(objective=OBJ + "by posting memes on r e d d i t and running g i v e a w a y s, within the "
                           "authorized experimental constraints"),
    "R2 secondary_metrics item, words concatenated without separators":
        lambda r: r["secondary_metrics"].append("secondary.redditmarketingstrategywithdailygiveaways"),
    "R2 family, words concatenated":
        lambda r: r.update(family="community/memeserverforgamers/invitefriendspersonally"),
    "R2 discord.channels labels, words concatenated":
        lambda r: r["resources"].update(discord={"guild_id": "123456789012345678", "channels": {
            "postmemesdaily": "223456789012345678", "invitegamerfriends": "323456789012345678",
            "dmeverynewjoiner": "423456789012345678"}}),
    "R2 objective_id, words concatenated":
        lambda r: r.update(objective_id="discord100viaredditmarketingandgiveaways"),
    "R2 authorized_destinations URL query with Cyrillic homoglyphs":
        lambda r: r["resources"].update(authorized_destinations=[
            "example.invalid", "https://example.invalid/?plan=post_memes_on_r" + CY_E + "ddit_d" + CY_A + "ily"]),
})
controls = {
    "family with spaces": lambda r: r.update(family="community/meme server for gamers"),
    "secondary_metrics item with spaces": lambda r: r["secondary_metrics"].append("secondary.reddit marketing"),
    "objective text without zero-width spaces":
        lambda r: r.update(objective=("Create and operate a genuinely useful Discord community for pc gamers that attracts "
                                      "100 distinct real human members who voluntarily choose to join by posting memes "
                                      "on reddit every day and running weekly giveaways, within the authorized "
                                      "experimental constraints")),
    "channel label 'post-memes-daily' (cadence word)":
        lambda r: r["resources"].update(discord={"guild_id": "123456789012345678",
                                                 "channels": {"post-memes-daily": "223456789012345678"}}),
}

accepted, reaching = [], []
for label, mut in variants.items():
    raw = copy.deepcopy(base)
    mut(raw)
    task, probs = load(raw)
    if task is None:
        R.check(f"V {label}: refused by the loader", True, probs[:2])
        continue
    lv = learner_text(task)
    hits = sorted({w for w in PLAN_WORDS if w in lv.lower()} - {w for w in PLAN_WORDS if w in learner_text(t0).lower()})
    accepted.append(label)
    if hits:
        reaching.append(label)
    R.check(f"V {label}: strategy content is refused, or at least does not reach the learner",
            not hits, {"accepted_by_load_task": True, "strategy_findings": T.strategy_findings(raw),
                       "plan words in learner_view": hits})

for label, mut in controls.items():
    raw = copy.deepcopy(base)
    mut(raw)
    task, probs = load(raw)
    R.check(f"C control — {label}: refused", task is None, probs[:2])

R.note("summary", {"variants": len(variants), "accepted": len(accepted), "reach the learner": len(reaching),
                   "which": reaching})
sys.exit(R.done())
