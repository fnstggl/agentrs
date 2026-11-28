"""INT3B round 2: defects found by the independent verifier G2-runtime (docs/verification/G2_runtime.md) at a5ab772.
Each test fails on a5ab772 and passes with the fix; the verifier's probes in repro/g2runtime/ report HOLDS."""
from __future__ import annotations

import copy
import os

import pytest
import yaml

from cam_live.runtime import tasks as T

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DISCORD = os.path.join(PROJECT, "tasks", "discord_100.yaml")
PLANNING = os.path.join(PROJECT, "tasks", "discord_100_planning.yaml")
ZW = "​"


# ---------------------------------------------------------------------------------------------------------- G2R-1
def _filled():
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


def _refused(raw, tmp_path) -> bool:
    p = tmp_path / "t.yaml"
    p.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    try:
        T.load_task(str(p), for_run=True)
    except T.TaskValidationError:
        return True
    return False


# neutral placeholders for content words (the test file itself carries no community strategy): the rules look for
# kinds of words (platforms, cadence, strategy vocabulary, ordered / imperative steps), not for these examples
DISGUISES = {
    "underscore-joined metric": lambda r: r["secondary_metrics"].append("secondary.reddit_widget_strategy_with_daily_thing"),
    "dash-joined family": lambda r: r.update(family="community/widget-thing-then-invite-friends"),
    "tool pattern": lambda r: r["resources"].update(authorized_tools=["discord.*", "step_one_make_x.step_two_invite_y"]),
    "interfaces item": lambda r: r["interfaces"].append("widget_on_reddit.then_invite_people.*"),
    "destination query": lambda r: r["resources"].update(authorized_destinations=[
        "example.invalid", "https://example.invalid/?q=widget_then_share_daily"]),
    "channel labels": lambda r: r["resources"].update(discord={"guild_id": "123456789012345678", "channels": {
        "invite-some-friends": "223456789012345678", "start-a-contest": "323456789012345678"}}),
    "objective_id": lambda r: r.update(objective_id="discord_100_via_reddit_marketing"),
    "camelCase key": lambda r: r["resources"].update(discord={"guild_id": "123456789012345678",
                                                              "channels": {"growthPlan": "223456789012345678"}}),
    "full-width platform word": lambda r: r["secondary_metrics"].append("secondary.ｒｅｄｄｉｔ_x"),
    "zero-width objective": lambda r: r.update(objective="Create and operate a genuinely useful Discord community that "
                                                         "attracts 100 members by posting on r" + ZW + "eddit e" + ZW +
                                                         "very day, within the authorized experimental constraints"),
    "zero-width in an identifier": lambda r: r["resources"].update(authorized_tools=["disc" + ZW + "ord.*"]),
}


@pytest.mark.parametrize("name", sorted(DISGUISES))
def test_disguised_strategy_content_is_refused(tmp_path, name):
    base = _filled()
    assert not _refused(copy.deepcopy(base), tmp_path)             # the filled task itself loads
    raw = copy.deepcopy(base)
    DISGUISES[name](raw)
    assert _refused(raw, tmp_path), name


def test_normalisation_reads_like_a_reader_and_shipped_tasks_stay_clean():
    assert T.normalise_for_scan("secondary.a_b-c:d/e") == "secondary a b c d e"
    assert T.normalise_for_scan("growthPlan") == "growth Plan"
    assert T.normalise_for_scan("ｒｅｄ" + ZW + "dit") == "reddit"
    assert T.invisible_characters("a‮b" + ZW) == ["U+200B", "U+202E"]
    for f in ("discord_100.yaml", "discord_100_planning.yaml", "lab_staging.yaml", "utility_staging.yaml",
              os.path.join("staging", "discord_100_planning_staging.yaml")):
        raw = T.load_task_raw(os.path.join(PROJECT, "tasks", f))
        assert T.strategy_findings(raw) == [] and T.structural_problems(raw) == [], f
    # the operator's confirmed objective still loads verbatim
    t = T.load_task(PLANNING, for_run=False)
    assert T.learner_view(t).objective == T.load_task_raw(PLANNING)["objective"]
