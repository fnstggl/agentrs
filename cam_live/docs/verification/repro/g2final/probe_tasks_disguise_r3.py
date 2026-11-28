"""G2-final probe: strategy content in a task file after INT3B round 3 (G2R-1b: confusable skeleton, leetspeak,
spelled letters, concatenation split, listing / bump sites, outreach moves).

    G2F_SRC=<archive>/cam_live/src .venv/bin/python cam_live/docs/verification/repro/g2final/probe_tasks_disguise_r3.py

Same harness as G2-runtime's probe_tasks_disguise.py: a fully filled copy of tasks/discord_100.yaml, one change per
variant, ``load_task(for_run=True)``; a variant "passes" when the loader accepts it and the disguised words reach
``learner_view`` (what agent.set_objective receives). Every variant has a control: the same sentence written plainly,
which must be refused (otherwise the variant is not a disguise of something the rules cover).
Variants are new readings of the same classes the round-3 fix claims to cover (other look-alike letters, other
separators, other spellings) plus unlisted words of the covered categories.
"""
import copy
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g2fcommon import PROJECT, Report, scratch  # noqa: E402

from cam_live.runtime import tasks as T  # noqa: E402

R = Report("probe_tasks_disguise_r3")
tmp = scratch("tasks_r3")
DISCORD = os.path.join(PROJECT, "tasks", "discord_100.yaml")


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
OBJ = ("Create and operate a genuinely useful Discord community that attracts 100 distinct real human members who "
       "voluntarily choose to join ")
END = ", within the authorized experimental constraints"


def obj(s):
    return lambda r: r.update(objective=OBJ + s + END)


def dest(u):
    return lambda r: r["resources"].update(authorized_destinations=["example.invalid", u])


# (label, variant mutation, disguised fragment that must not reach the learner, plain control mutation)
CASES = [
    ("Latin letters with diacritics (precomposed, NFKC keeps them): réddit, gíveaways",
     obj("by posting memes on réddit and running gíveaways"), "réddit",
     obj("by posting memes on reddit and running giveaways")),
    ("combining marks on Latin letters (NFKC composes them into diacritics)",
     obj("by posting memes on rẹddit"), "dit",
     obj("by posting memes on reddit")),
    ("Latin small capitals (U+1D00 block, not NFKC-folded, not in the confusable map)",
     obj("by posting memes on ʀᴇᴅᴅɪᴛ"), "ʀᴇᴅᴅɪᴛ",
     obj("by posting memes on reddit")),
    ("letters spelled with '*' / '_' between them (the spelled-letter rule joins only space, '.' and '-')",
     obj("by posting memes on r*e*d*d*i*t and running g_i_v_e_a_w_a_y_s"), "r*e*d*d*i*t",
     obj("by posting memes on reddit and running giveaways")),
    ("a word split once in the middle (objective text: no concatenation / segmentation)",
     obj("by posting memes on red dit and running give aways"), "red dit",
     obj("by posting memes on reddit and running giveaways")),
    ("percent-encoded letters in an authorized_destinations URL",
     dest("https://example.invalid/?plan=post+memes+on+%72%65%64%64%69%74+and+run+%67%69%76%65%61%77%61%79%73"),
     "%72%65%64%64%69%74",
     dest("https://example.invalid/?plan=post+memes+on+reddit+and+run+giveaways")),
    ("leetspeak with substitutions outside the folded set ('(' for c, '+' for t, '8' for b, '9' for g)",
     obj("by cross(-)posting memes on +ik+ok and running 9iveaways"), "9iveaways",
     obj("by crossposting memes on tiktok and running giveaways")),
    ("unlisted platforms of the covered category (imgur, 9gag, lemmy, vk)",
     obj("by posting memes on imgur, 9gag, lemmy and vk"), "imgur",
     obj("by posting memes on reddit, tiktok, youtube and twitter")),
    ("unlisted synonyms of covered strategy words (raffles / prize draws for giveaways, flyers / ads for advertising)",
     obj("by holding raffles and prize draws and handing out flyers"), "raffles",
     obj("by holding giveaways and advertising with flyers")),
]

passed = []
for label, mut, frag, ctrl in CASES:
    raw = copy.deepcopy(base)
    ctrl(raw)
    tc, pc = load(raw)
    control_refused = tc is None
    raw = copy.deepcopy(base)
    mut(raw)
    task, probs = load(raw)
    if task is None:
        R.check(f"V {label}: refused by the loader", True, {"problems": probs[:2], "control_refused": control_refused})
        continue
    lv = json.dumps(T.learner_view(task).to_dict(), ensure_ascii=False)
    reaches = frag in lv
    if reaches and control_refused:
        passed.append(label)
    R.check(f"V {label}: refused, or its content does not reach the learner",
            not (reaches and control_refused),
            {"accepted_by_load_task": True, "reaches_learner_view": reaches,
             "plain_control_refused": control_refused, "control_problems": pc[:1],
             "strategy_findings": T.strategy_findings(raw)[:3]})
R.note("summary", {"variants": len(CASES), "accepted and reaching the learner (control refused)": len(passed),
                   "which": passed})
# the shipped task files still pass the scan (false-refusal control)
for name in sorted(os.listdir(os.path.join(PROJECT, "tasks"))):
    if name.endswith(".yaml"):
        R.check(f"S shipped task file {name}: no strategy finding", not T.scan_file(os.path.join(PROJECT, "tasks", name)),
                T.scan_file(os.path.join(PROJECT, "tasks", name))[:3])
sys.exit(R.done())
