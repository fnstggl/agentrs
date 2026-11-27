"""Regression tests for the task-loader fixes (G_rt.md D5, D6, D25 and the Discord evidence default).

Each test pins a property that the verifier's probe ``docs/verification/repro/rt/p05_tasks.py`` found broken: the
closed schema, YAML hygiene, strategy text in free text and in the objective, degenerate limits, what the learner
receives, and the comments of ``tasks/discord_100.yaml``.
"""
import copy
import math
import os

import pytest
import yaml

from cam_live.contracts import Limits, TaskSpec
from cam_live.runtime import tasks as T

TASKS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tasks")
DISCORD = os.path.join(TASKS, "discord_100.yaml")
SPEC_OBJECTIVE = ("Create and operate a genuinely useful Discord community that attracts 100 distinct real human "
                  "members who voluntarily choose to join, within the authorized experimental constraints.")


def filled():
    """discord_100.yaml with every REQUIRED item filled (flat Discord keys, as the operator would)."""
    raw = T.load_task_raw(DISCORD)
    raw["success"]["exclusions"]["test_account_ids"] = ["123456789012345678"]
    raw["resources"].update(discord_guild_id="987654321098765432", discord_bot_credential_name="discord_bot",
                            owned_site="https://example.invalid", operator_authorization_record="rec-1",
                            authorized_tools=["api", "discord.*"], authorized_destinations=["987654321098765432"],
                            currency="USD")
    raw["resources"]["publishing"].update(authorized=True, destinations=["987654321098765432"])
    raw["limits"] = {"duration_s": 86400, "money": 10, "compute_s": 3600, "max_external_actions": 100,
                     "max_decisions": 100}
    raw["data_permissions"]["collect_feedback_with_consent"] = False
    return raw


def problems(raw, for_run=True):
    return T.validate_task(raw, for_run=for_run)


def write(tmp_path, text, name="t.yaml"):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_baseline_filled_discord_task_loads_and_shipped_one_refuses_only_for_required_items():
    assert problems(filled()) == []
    with pytest.raises(T.TaskValidationError) as e:
        T.load_task(DISCORD, for_run=True)
    unset = T.unset_paths(T.load_task_raw(DISCORD))
    items = [p.split(":", 1)[0] for p in e.value.problems]
    assert all(any(u == i or u.startswith(i + ".") or u.startswith(i + "[") for u in unset) for i in items), items
    assert T.load_task(DISCORD, for_run=False).objective == SPEC_OBJECTIVE


# ---- D6: closed schema ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("where,key,value", [
    ("resources", "growthPlan", "open a server for retro handheld fans"),
    ("resources", "strategies", ["retro handheld fans", "cross-promotion"]),
    ("resources", "milestones", ["Open a server about retro handheld consoles", "Run a giveaway"]),
    ("resources", "outline", ["Create the server", "Write a welcome guide"]),
    ("resources", "notes", {"first open an anime server then post daily on reddit": True}),
    ("resources", "nota", "Estrategia: abrir un servidor de anime y publicar cada dia en Reddit"),
    ("resources", "note", "CAM is expected to open a server for speedrunners"),
    ("measurement", "todo", ["Open a server for anime fans"]),
    ("measurement", "cadence", "three announcements per week, mornings, in r/gaming"),
    ("success", "hint", "anything"),
    ("success.exclusions", "how_to_grow", "x"),
    ("resources.discord", "plan_b", "x"),
    ("resources.publishing", "schedule_notes", "x"),
])
def test_unknown_keys_anywhere_are_rejected_with_their_path(where, key, value):
    raw = filled()
    node = raw
    for part in where.split("."):
        node = node.setdefault(part, {})
    node[key] = value
    found = problems(raw)
    assert any(f"{where}.{key}" in p for p in found), found
    with pytest.raises(T.TaskValidationError):
        T.task_from_dict(raw)


def test_unknown_top_level_key_is_named():
    raw = filled()
    raw["strategy"] = "do this"
    assert any("unknown top-level key 'strategy'" in p for p in problems(raw))


@pytest.mark.parametrize("field,value", [
    ("family", "community/membership: anime fans, cross-promote with partner servers"),
    ("secondary_metrics", ["secondary.retention_7d", "post memes for teenagers in the evening hours"]),
    ("interfaces", ["api", "api: use it to DM server owners for cross-promotion"]),
    ("stop_conditions", ["verified_success", "when it feels done"]),
])
def test_labels_and_list_items_cannot_carry_prose(field, value):
    raw = filled()
    raw[field] = value
    assert problems(raw), (field, value)


def test_free_form_channel_names_are_labels_and_are_scanned():
    raw = filled()
    raw["resources"]["discord"] = {"guild_id": "987654321098765432", "channels": {"announcements": "111111111111111111"}}
    assert problems(raw) == []
    raw["resources"]["discord"]["channels"] = {"post daily on reddit": "1"}
    assert any("resources.discord.channels" in p for p in problems(raw))
    raw["resources"]["discord"]["channels"] = {"cross-promote": "1"}
    assert any("cross-promote" in p for p in problems(raw))


def test_free_text_is_bounded():
    raw = filled()
    raw["success"]["definition"] = "A member counts when listed. " * 100
    assert any("success.definition" in p and "at most 2000" in p for p in problems(raw))
    raw = filled()
    from cam_live.runtime.tasks import TEXT_LIMITS
    n = TEXT_LIMITS["objective"]
    raw["objective"] = SPEC_OBJECTIVE + " " + "x" * n
    assert any(p.startswith(f"objective: must be text of at most {n}") for p in problems(raw))


@pytest.mark.parametrize("marker", ["TODO", "TBD", "changeme", "???"])
def test_placeholder_spellings_are_not_taken_as_values(marker):
    raw = filled()
    raw["resources"]["discord_guild_id"] = marker
    assert any("placeholder" in p for p in problems(raw))


# ---- D6: YAML hygiene ---------------------------------------------------------------------------------------
def test_yaml_anchor_alias_and_merge_key_are_refused(tmp_path):
    text = yaml.safe_dump(filled())
    for planted in ("resources:\n  base_notes: &n\n    focus: speedrunners\n  more:\n    <<: *n\n",
                    "resources:\n  a: &s\n  - step 1 open a server\n  b: *s\n"):
        p = write(tmp_path, text.replace("resources:\n", planted, 1))
        with pytest.raises(T.TaskValidationError) as e:
            T.load_task(p)
        assert any("anchor" in x or "alias" in x for x in e.value.problems), e.value.problems
    # a merge key without an anchor is refused too
    with pytest.raises(T.TaskValidationError) as e:
        T.load_task_text("objective_id: x\nlimits:\n  <<: {duration_s: 1}\n")
    assert "merge" in str(e.value)


def test_duplicate_keys_are_refused(tmp_path):
    text = yaml.safe_dump(filled())
    p = write(tmp_path, text + "objective: \"Reach 100 members by posting in r/gaming every day.\"\n")
    with pytest.raises(T.TaskValidationError) as e:
        T.load_task(p)
    assert "duplicate key 'objective'" in str(e.value)


def test_invalid_yaml_is_a_task_validation_error(tmp_path):
    with pytest.raises(T.TaskValidationError):
        T.load_task(write(tmp_path, "objective_id: [unclosed\n"))


# ---- D6: strategy text ---------------------------------------------------------------------------------------
def test_shipped_task_files_are_not_flagged():
    for name in sorted(os.listdir(TASKS)):
        if name.endswith(".yaml"):
            raw = T.load_task_raw(os.path.join(TASKS, name))
            assert T.scan_file(os.path.join(TASKS, name)) == [], name
            # an objective text the operator gave verbatim may read like strategy vocabulary only when the file
            # confirms exactly that text (discord_100_planning.yaml: "determine the community's niche ...")
            if T.objective_flags(raw.get("objective", "")):
                assert raw.get("objective_confirmation") == T.objective_confirmation(raw["objective"]), name
            assert T.structural_problems(raw) == [], name


@pytest.mark.parametrize("extra", [
    " The community is for indie game developers and members are recruited through devlog threads on Reddit.",
    " Members are recruited through AMA threads in r/gamedev.",
    " Grow it with weekly giveaways.",
])
def test_objective_with_strategy_is_refused_unless_the_operator_confirms_that_exact_text(extra):
    raw = filled()
    raw["objective"] = SPEC_OBJECTIVE + extra
    assert T.objective_flags(raw["objective"])
    found = problems(raw)
    assert any(p.startswith("objective: text reads as a solution strategy") for p in found), found
    raw["objective_confirmation"] = T.objective_confirmation(SPEC_OBJECTIVE)        # confirms another text
    assert any(p.startswith("objective:") for p in problems(raw))
    raw["objective_confirmation"] = T.objective_confirmation(raw["objective"])
    assert problems(raw) == []
    assert T.task_from_dict(raw).objective == raw["objective"]


def test_single_sentence_objectives_with_an_audience_are_flagged():
    assert T.objective_flags("Create a Discord community for indie game developers that attracts 100 members.")
    assert T.objective_flags("Create and operate a Discord server about retro handheld consoles.")
    assert T.objective_flags(SPEC_OBJECTIVE) == []


@pytest.mark.parametrize("field,text", [
    ("success.definition", "A member counts when listed by the platform. Members are recruited through AMA threads "
                           "in r/gamedev."),
    ("success.definition", "A member counts when they joined after we advertise on Twitter."),
    ("measurement.protocol", "Post the invite in three subreddits, then count the members."),
    ("measurement.protocol", "The verifier reads the member list. Open a server for anime fans first."),
])
def test_verifier_texts_with_moves_are_rejected(field, text):
    raw = filled()
    a, b = field.split(".")
    raw[a][b] = text
    assert any(p.startswith(field) for p in problems(raw)), problems(raw)


def test_verifier_texts_of_the_shipped_files_are_accepted():
    raw = T.load_task_raw(DISCORD)
    assert "joined through their own action" in raw["success"]["definition"]
    assert "through the official bot interface" in raw["measurement"]["protocol"]
    assert T.strategy_findings(raw) == []


# ---- D5: limits ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -5, "3600", True])
@pytest.mark.parametrize("mode", ["live", "staging", "diagnostic"])
def test_degenerate_limits_are_rejected_at_load_in_every_mode(value, mode):
    raw = filled()
    raw["mode"] = mode
    raw["limits"]["money"] = value
    found = problems(raw, for_run=False)
    assert any(p.startswith("limits.money") for p in found), found


def test_zero_limit_is_not_set_outside_live_and_refused_in_live():
    raw = filled()
    raw["limits"]["money"] = 0
    assert any("limits.money" in p for p in problems(raw))
    raw["mode"] = "staging"
    assert problems(raw) == []


def test_code_built_task_with_invalid_limits_is_not_runnable():
    for bad in (float("nan"), float("inf"), -1.0, True):
        t = TaskSpec("x", "do it", mode="diagnostic", limits=Limits(duration_s=60, max_decisions=5))
        t.limits.money = bad
        assert "money" in T.limit_problems(t), bad
        assert any("limits.money is invalid" in p for p in T.check_runnable(t)), bad
    t = TaskSpec("x", "do it", mode="diagnostic", limits=Limits(duration_s=float("nan"), max_decisions=float("inf")))
    probs = T.check_runnable(t)
    assert any("duration_s" in p for p in probs) and any("max_decisions" in p for p in probs)
    assert any("unbounded" in p for p in probs)          # NaN / inf never count as a bound
    assert T.limit_problems(TaskSpec("x", "y", limits=Limits(1, 2, 3, 4, 5))) == {}


# ---- D6: what the learner receives -----------------------------------------------------------------------------
def test_learner_view_withholds_verifier_only_content_and_is_pure():
    raw = filled()
    raw["resources"]["discord"] = {"guild_id": "987654321098765432", "token_secret": "discord_bot",
                                   "test_account_ids": ["123456789012345678"], "channels": {"general": "1"},
                                   "base_url": "https://discord.com/api/v10"}
    raw["resources"]["owned_sites"] = [{"name": "site", "root": "/srv/site", "base_url": "https://example.invalid",
                                        "access_log": "/var/log/site.log"}]
    task = T.task_from_dict(raw)
    before = task.to_json()
    v = T.learner_view(task)
    blob = v.to_json()
    for withheld in (raw["success"]["definition"], raw["measurement"]["protocol"], "exclusions", "evidence_source",
                     "discord_bot", "rec-1", "123456789012345678", "/srv/site", "/var/log/site.log", "base_url\": \"https://discord"):
        assert withheld not in blob, withheld
    assert v.objective == SPEC_OBJECTIVE and v.objective_id == task.objective_id and v.mode == task.mode
    assert v.limits == task.limits and v.interfaces == task.interfaces and v.data_permissions == task.data_permissions
    assert v.success == {"metric": "membership.confirmed", "target": 100} and v.measurement == {}
    assert v.secondary_metrics == task.secondary_metrics and v.stop_conditions == task.stop_conditions
    assert v.resources["discord_guild_id"] == "987654321098765432"
    assert v.resources["discord"] == {"guild_id": "987654321098765432", "channels": {"general": "1"}}
    assert v.resources["owned_sites"] == [{"name": "site", "base_url": "https://example.invalid"}]
    assert v.resources["authorized_tools"] == ["api", "discord.*"] and v.resources["publishing"]["authorized"] is True
    v.resources["discord"]["channels"]["x"] = "2"
    v.limits.money = 1e9
    assert task.to_json() == before                      # the runtime's own TaskSpec is untouched


# ---- Discord evidence default --------------------------------------------------------------------------------
def test_discord_tasks_default_to_the_verifier_side_membership_source():
    assert T.DEFAULT_DISCORD_EVIDENCE_SOURCE == "cam_live.harness.discord_membership:membership_source"
    raw = T.load_task_raw(DISCORD)
    assert raw["success"]["evidence_source"] == T.DEFAULT_DISCORD_EVIDENCE_SOURCE
    assert "success.evidence_source" not in T.unset_paths(raw)
    t = TaskSpec("d", "x", success={"verifier": "membership"}, resources={"discord": {"guild_id": "1"}})
    assert T.has_discord(t) and T.evidence_source(t) == T.DEFAULT_DISCORD_EVIDENCE_SOURCE
    t = TaskSpec("d", "x", success={"verifier": "membership", "evidence_source": "REQUIRED"},
                 resources={"discord_guild_id": "1"})
    assert T.evidence_source(t) == T.DEFAULT_DISCORD_EVIDENCE_SOURCE
    t = TaskSpec("d", "x", success={"evidence_source": "json:/tmp/m.json"}, resources={"discord_guild_id": "1"})
    assert T.evidence_source(t) == "json:/tmp/m.json"
    t = TaskSpec("u", "x", success={"verifier": "metric_threshold"})
    assert not T.has_discord(t) and T.evidence_source(t) == ""
    # a live task without Discord resources and without an evidence source names the missing item
    raw = filled()
    del raw["resources"]["discord_guild_id"]
    raw["success"].pop("evidence_source")
    assert any(p.startswith("success.evidence_source") for p in problems(raw))
    raw = filled()
    raw["success"].pop("evidence_source")                 # Discord resources present: the default applies
    assert problems(raw) == []


# ---- D25: the comments of the Discord task tell the truth --------------------------------------------------------
def test_discord_task_comments_state_what_the_code_does():
    text = open(DISCORD).read()
    assert "never by the learner" not in text
    assert "hashed before they reach any log" not in text
    assert "learner_view" in text and "Redactor" in text


# ---- files other components write keep loading -----------------------------------------------------------------
def test_task_files_written_by_other_components_still_load(tmp_path):
    from cam_live.diagnostics.world import LabConfig, LabWorld, lab_task
    doc = lab_task(LabWorld(LabConfig(seed=1)), "obj_second").to_dict()
    doc.update(version=1, mode="staging", stop_conditions=["verified_success", "verified_failure"])
    p = write(tmp_path, yaml.safe_dump(doc), "second.yaml")
    assert T.load_task(p).objective_id == "obj_second"
    p = write(tmp_path, "version: 1\nobjective_id: discord-live-check\nmode: live\n"
                        'objective: "Integration check of the Discord adapter in the authorized server."\n'
                        'resources:\n  discord:\n    guild_id: "123456789012345678"\ninterfaces: [api]\n'
                        "limits:\n  duration_s: 3600\n  money: 0\n  compute_s: 0\n  max_external_actions: 200\n"
                        "  max_decisions: 200\n", "check.yaml")
    assert T.load_task(p, for_run=False).resources["discord"]["guild_id"] == "123456789012345678"
    assert math.isfinite(T.load_task(os.path.join(TASKS, "lab_staging.yaml")).limits.money)
