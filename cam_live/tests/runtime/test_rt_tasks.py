"""Task files: validation, the Discord objective, and the mechanical no-strategy scan."""
import copy
import os

import pytest
import yaml

from cam_live.contracts import Limits, TaskSpec
from cam_live.runtime import tasks as T

TASKS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tasks")
DISCORD = os.path.join(TASKS, "discord_100.yaml")
SPEC_OBJECTIVE = ("Create and operate a genuinely useful Discord community that attracts 100 distinct real human "
                  "members who voluntarily choose to join, within the authorized experimental constraints.")


def _lab():
    return T.load_task_raw(os.path.join(TASKS, "lab_staging.yaml"))


def test_discord_task_is_the_spec_objective_with_independent_criteria():
    raw = T.load_task_raw(DISCORD)
    t = T.load_task(DISCORD, for_run=False)
    assert t.objective == SPEC_OBJECTIVE and t.mode == "live" and t.objective_id == "discord-100"
    s = t.success
    assert s["verifier"] == "membership" and s["target"] == 100 and 0.5 < s["confidence"] < 1
    ex = s["exclusions"]
    assert ex["exclude_bots"] is True and "purchased" in ex["exclude_flags"] and "duplicate" in ex["exclude_flags"]
    assert "fabricated_activity" in ex["exclude_flags"] and "prohibited_engagement" in ex["exclude_flags"]
    assert ex["pending_screening"] == "uncertain" and "test_account_ids" in ex
    for m in ("secondary.retention_7d", "secondary.participation.members", "secondary.feedback.consented",
              "secondary.operating_cost", "secondary.time_elapsed_s"):
        assert m in t.secondary_metrics
    assert t.data_permissions["store_message_content"] is False
    assert t.data_permissions["train_on_message_content"] is False
    assert set(t.interfaces) >= {"structured_browser", "visual", "code", "api", "skill"}
    assert {"verified_success", "duration_exhausted", "money_exhausted", "operator_stop"} <= set(t.stop_conditions)
    # everything the operator must supply is a named, unset item — nothing is assumed
    for path in ("resources.discord_guild_id", "resources.discord_bot_credential_name", "resources.owned_site",
                 "limits.duration_s", "limits.money", "limits.compute_s", "limits.max_external_actions",
                 "limits.max_decisions", "resources.authorized_tools", "resources.publishing.authorized"):
        assert path in T.unset_paths(raw), path
    assert t.limits == Limits()                    # unset limits are 0 = "must be set", never "unlimited"


def test_live_task_with_unset_items_cannot_be_loaded_for_a_run():
    with pytest.raises(T.TaskValidationError) as e:
        T.load_task(DISCORD, for_run=True)
    text = " | ".join(e.value.problems)
    for needle in ("limits.duration_s", "limits.money", "limits.compute_s", "limits.max_external_actions",
                   "limits.max_decisions", "resources.authorized_tools", "resources.publishing.authorized",
                   "data_permissions.collect_feedback_with_consent", "resources.discord_guild_id",
                   "resources.discord_bot_credential_name", "resources.owned_site"):
        assert needle in text, needle


def test_discord_task_becomes_runnable_only_when_everything_is_filled(tmp_path):
    raw = T.load_task_raw(DISCORD)
    raw["limits"] = {"duration_s": 30 * 86400, "money": 50, "compute_s": 1e6, "max_external_actions": 5000,
                     "max_decisions": 100000}
    raw["resources"].update(discord_guild_id="G", discord_bot_credential_name="discord_bot", owned_site="https://x.test",
                            operator_authorization_record="rec-1", authorized_tools=["discord", "browser"],
                            authorized_destinations=["https://x.test/*"], currency="USD",
                            publishing={"authorized": True, "approval_required": True, "destinations": ["https://x.test"]})
    raw["success"]["evidence_source"] = "pkg.mod:factory"
    raw["success"]["exclusions"]["test_account_ids"] = []
    raw["data_permissions"]["collect_feedback_with_consent"] = True
    assert T.validate_task(raw, for_run=True) == []
    for drop in ("duration_s", "money", "compute_s", "max_external_actions", "max_decisions"):
        bad = copy.deepcopy(raw)
        bad["limits"][drop] = 0
        assert any(f"limits.{drop}" in p for p in T.validate_task(bad, for_run=True))
    bad = copy.deepcopy(raw)
    bad["stop_conditions"] = []
    assert any("stop_conditions" in p for p in T.validate_task(bad, for_run=True))
    bad = copy.deepcopy(raw)
    del bad["data_permissions"]["train_on_message_content"]
    assert any("train_on_message_content" in p for p in T.validate_task(bad, for_run=True))


def test_task_files_contain_no_strategy_mechanical_scan():
    for name in sorted(os.listdir(TASKS)):
        if name.endswith(".yaml"):
            assert T.scan_file(os.path.join(TASKS, name)) == [], name
            raw = T.load_task_raw(os.path.join(TASKS, name))
            keys = {k.lower() for _p, k, _v in T._walk(raw) if k}
            assert not keys & set(TaskSpec.FORBIDDEN_KEYS), name


@pytest.mark.parametrize("key", list(TaskSpec.FORBIDDEN_KEYS) + ["tactics", "some_plan", "daily_schedule",
                                                                 "marketing", "recommended_approach"])
def test_forbidden_keys_are_rejected_anywhere(key):
    raw = _lab()
    raw[key] = "anything"
    assert T.strategy_findings(raw)
    raw = _lab()
    raw["resources"]["nested"] = {"deeper": [{key: ["a", "b"]}]}
    f = T.strategy_findings(raw)
    assert f and key in f[0]
    with pytest.raises(T.TaskValidationError):
        T.task_from_dict(raw)


@pytest.mark.parametrize("text", [
    "Step 1: do the first thing. Step 2: do the second thing.",
    "First do one thing, then do another thing.",
    "CAM should prefer option A.",
    "Do the thing every day at 9 am.",
    "Pick a niche for it.",
    "For example, post the thing in place X.",
])
def test_strategy_like_text_is_rejected_anywhere(text):
    raw = _lab()
    raw["measurement"]["protocol"] = text
    assert T.strategy_findings(raw), text
    raw = _lab()
    raw["resources"]["note"] = [text]
    assert T.strategy_findings(raw), text
    raw = _lab()
    raw["objective"] = raw["objective"] + " " + text
    assert T.strategy_findings(raw), text


def test_structure_and_limits_in_every_mode():
    raw = _lab()
    assert T.validate_task(raw) == []
    bad = copy.deepcopy(raw)
    bad["limits"]["duration_s"] = "REQUIRED"
    assert any("limits.duration_s" in p for p in T.validate_task(bad))
    bad = copy.deepcopy(raw)
    del bad["limits"]["max_decisions"]
    assert any("max_decisions" in p for p in T.validate_task(bad))
    bad = copy.deepcopy(raw)
    bad["stop_conditions"].append("when_it_feels_done")
    assert any("unknown stop condition" in p for p in T.validate_task(bad))
    bad = copy.deepcopy(raw)
    bad["surprise"] = 1
    assert any("unknown top-level key" in p for p in T.validate_task(bad))
    bad = copy.deepcopy(raw)
    bad["mode"] = "production"
    assert any("mode must be" in p for p in T.validate_task(bad))
    # the runtime re-checks TaskSpec objects built in code
    assert T.check_runnable(TaskSpec("x", "do it", mode="staging")) != []
    assert T.check_runnable(TaskSpec("x", "do it", mode="live", limits=Limits(1, 1, 1, 1, 1))) != []   # no stop conditions


def test_second_objective_is_a_different_family_with_its_own_verifier():
    a = T.load_task(DISCORD, for_run=False)
    b = T.load_task(os.path.join(TASKS, "utility_staging.yaml"), for_run=False)
    assert b.family != a.family and b.objective != a.objective and b.objective_id != a.objective_id
    assert b.success["verifier"] == "metric_threshold" != a.success["verifier"]
    assert b.success["metric"] and b.success["target"] > 0 and b.mode == "staging"
    with pytest.raises(T.TaskValidationError):          # its evidence source must be filled before a run
        T.load_task(os.path.join(TASKS, "utility_staging.yaml"), for_run=True)


def test_autonomous_learning_needs_a_fixed_budget():
    t = T.autonomous_task("auto-1", Limits(duration_s=3600, max_decisions=50))
    assert t.objective == "" and T.check_runnable(t) == []
    with pytest.raises(T.TaskValidationError):
        T.autonomous_task("auto-2", Limits(duration_s=3600))             # no decision budget
    raw = {"objective_id": "auto", "objective": "", "mode": "staging", "limits": {"duration_s": 10, "max_decisions": 5}}
    assert any("autonomous" in p for p in T.validate_task(raw))          # money / compute / actions not stated


def test_autonomous_run_stops_at_its_budget_without_verification(tmp_path):
    from cam_live.contracts import SimClock
    from cam_live.diagnostics.world import AllowAllPolicy, LabConfig, LabVerifier, LabWorld
    from cam_live.runtime.loop import EXHAUSTED, LiveRuntime, RuntimeConfig
    from rt_lab_agent import LabAgent
    clock = SimClock()
    world = LabWorld(LabConfig(seed=2), clock)
    calls = []

    class Counting(LabVerifier):
        def verify(self, task):
            calls.append(task.objective_id)
            return super().verify(task)

    task = T.autonomous_task("auto-run", Limits(duration_s=30 * 86400, money=100, max_decisions=12), mode="diagnostic")
    agent = LabAgent(clock)
    rt = LiveRuntime(agent, world, AllowAllPolicy(), Counting(world), task, str(tmp_path / "run"), clock,
                     RuntimeConfig(fsync=False))
    assert rt.run(max_iterations=500) == EXHAUSTED
    assert agent.calls["select"] == 12 and calls == [] and agent.verified_seen == []
