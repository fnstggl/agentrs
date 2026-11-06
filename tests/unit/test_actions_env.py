import pytest

from agentrs.actions.schema import Action, ActionParseError, ActionType, parse_action, try_parse_action
from agentrs.agent.runtime import RandomActor, TeacherActor, run_episode
from agentrs.data.diff import StateDiff, compute_diff, content_set
from agentrs.demonstrations.teacher import ScriptedTeacher
from agentrs.environments.miniweb.core import MiniWebEnv
from agentrs.environments.miniweb.families import HELDOUT_FAMILIES, TRAIN_FAMILIES
from agentrs.environments.miniweb.tasks import make_task, task_set


# ---------------------------------------------------------------- actions
@pytest.mark.parametrize("s", ['click(e12)', 'type(e3, "a \\"quoted\\" b")', 'clear(e3)', 'select(e7, "Express")',
                               'scroll(down)', 'navigate("/settings")', 'read(e9)', 'wait()', 'finish()',
                               'finish("$12.50")', 'skill(fill_and_submit)'])
def test_action_roundtrip(s):
    a = parse_action(s)
    assert parse_action(a.to_str()) == a
    assert a.to_str() == s


@pytest.mark.parametrize("bad", ["__import__('os').system('ls')", "click(e1); rm -rf /", "click()", "type(e1)",
                                 "scroll(left)", 'navigate(/x)', "eval(\"1\")", "click(e1) extra", "type(e1, \"x)"])
def test_action_parser_rejects_malformed(bad):
    assert try_parse_action(bad) is None
    with pytest.raises(ActionParseError):
        parse_action(bad)


def test_action_type_validation():
    with pytest.raises(ValueError):
        Action(type=ActionType.CLICK)  # missing target
    with pytest.raises(ValueError):
        Action(type=ActionType.WAIT, target="e1")  # extra arg
    with pytest.raises(ValueError):
        Action(type=ActionType.TYPE, target="e1", text="x" * 1000)


# ---------------------------------------------------------------- environment
def test_all_families_solvable_by_teacher_and_verifier_is_independent():
    env = MiniWebEnv()
    for split, fams in (("train", TRAIN_FAMILIES), ("test_family", HELDOUT_FAMILIES)):
        for t in task_set(split, fams, n_per_family=3):
            ep = run_episode(env, t, TeacherActor(ScriptedTeacher()))
            assert ep.success, (t.task_id, ep.termination_reason)
            assert ep.verification["verifier_id"].startswith("miniweb/")


def test_invalid_and_stale_actions_are_observable_errors():
    env = MiniWebEnv()
    t = make_task("checkout", "train", 1)
    obs = env.reset(t)
    r = env.step(Action.click("e99999"))
    assert not r.action_valid and "not found" in r.error and not r.terminated
    cart = next(e for e in obs.elements if e.name.startswith("Cart"))
    first_add = next(e for e in obs.elements if e.role == "button")
    env.step(Action.click(cart.eid))  # navigate -> new ids
    r = env.step(Action.click(first_add.eid))  # stale reference
    assert not r.action_valid and "stale" in r.error
    r = env.step(Action.type_text(cart.eid, "x"))
    assert not r.action_valid
    r = env.step(Action.navigate("https://evil.example.com/"))
    assert not r.action_valid and "not permitted" in r.error


def test_async_loading_and_wait():
    env = MiniWebEnv()
    t = make_task("search_lookup", "train", 0)
    t.private["delay"] = 2
    obs = env.reset(t)
    box = next(e for e in obs.elements if e.role == "searchbox")
    env.step(Action.type_text(box.eid, t.private["product"]))
    btn = next(e for e in env.observe().elements if e.name == "Search")
    r = env.step(Action.click(btn.eid))
    assert any(e.role == "status" for e in r.observation.elements)  # loading
    r = env.step(Action.wait())
    assert not any(e.role == "status" for e in r.observation.elements)
    assert any(t.private["product"] in e.name for e in r.observation.elements)


def test_scroll_partial_observability():
    env = MiniWebEnv()
    t = make_task("scroll_find", "test_family", 0)
    obs = env.reset(t)
    assert t.private["t"] not in [e.name for e in obs.elements]
    assert obs.scroll_info and "below" in obs.scroll_info
    for _ in range(4):
        obs = env.step(Action.scroll("down")).observation
    assert t.private["t"] in [e.name for e in obs.elements]


def test_truncation_vs_termination():
    env = MiniWebEnv()
    t = make_task("navigate_link", "train", 0)
    ep = run_episode(env, t, RandomActor(3))
    last = ep.transitions[-1]
    assert last.terminated or last.truncated
    if not last.terminated:
        assert ep.termination_reason.startswith("truncated")
        assert ep.num_steps == t.max_steps


def test_deterministic_seeded_replay():
    env1, env2 = MiniWebEnv(), MiniWebEnv()
    t = make_task("form_validation", "train", 5)
    e1 = run_episode(env1, t, RandomActor(7))
    e2 = run_episode(env2, t, RandomActor(7))
    assert [x.action for x in e1.transitions] == [x.action for x in e2.transitions]
    assert [x.next_observation for x in e1.transitions] == [x.next_observation for x in e2.transitions]


def test_splits_are_disjoint_and_guarded():
    with pytest.raises(ValueError):
        make_task("dialog_confirm", "train", 0)  # held-out family
    with pytest.raises(ValueError):
        make_task("click_button", "train", 0, layout="distractors")  # held-out layout
    with pytest.raises(ValueError):
        make_task("click_button", "train", 0, template=2)  # held-out template
    tr = {t.seed for t in task_set("train", n_per_family=50)}
    te = {t.seed for t in task_set("test", n_per_family=50)}
    assert not tr & te
    assert "private" not in make_task("login", "train", 0).public().model_dump()


def test_observation_has_no_hidden_state():
    env = MiniWebEnv()
    t = make_task("login", "train", 2)
    txt = env.reset(t).to_text()
    assert t.private["pw"] not in txt or t.private["pw"] in t.goal  # only via the public goal
    assert "verify" not in txt and "server" not in txt.lower()


# ---------------------------------------------------------------- diffs
def test_diff_apply_reproduces_next_observation_content():
    env = MiniWebEnv()
    for fam in ("checkout", "menu", "form_validation", "checkboxes"):
        t = make_task(fam, "train", 4)
        ep = run_episode(env, t, TeacherActor(ScriptedTeacher(0.3, 1)))
        for tr in ep.transitions:
            d = compute_diff(tr.observation, tr.next_observation)
            pred = d.apply(tr.observation)
            assert content_set(pred) == content_set(tr.next_observation)
            # text round-trip preserves the content as well (when not truncated)
            if len(d.added) <= 14:
                pred2 = StateDiff.from_text(d.to_text()).apply(tr.observation)
                assert content_set(pred2) == content_set(tr.next_observation), d.to_text()


def test_task_generation_independent_of_hash_seed():
    """Tasks and teacher episodes must be identical across processes with different PYTHONHASHSEED."""
    import os
    import subprocess
    import sys

    code = ("import json;from agentrs.environments.miniweb.tasks import task_set;"
            "from agentrs.agent.runtime import run_episode, TeacherActor;"
            "from agentrs.demonstrations.teacher import ScriptedTeacher;"
            "from agentrs.environments.miniweb.core import MiniWebEnv;e=MiniWebEnv();ts=task_set('train', n_per_family=3);"
            "print(json.dumps([(t.goal, t.private.get('product'), [x.action for x in run_episode(e,t,"
            "TeacherActor(ScriptedTeacher(0.3, i))).transitions]) for i,t in enumerate(ts)]))")
    src = os.path.join(os.path.dirname(__file__), "..", "..", "src")
    outs = set()
    for hs in ("1", "2", "3"):
        env = {**os.environ, "PYTHONHASHSEED": hs, "PYTHONPATH": src}
        outs.add(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                                check=True).stdout)
    assert len(outs) == 1


def test_gymnasium_adapter():
    from gymnasium.utils.env_checker import check_env

    from agentrs.environments.gym_adapter import GymnasiumEnv

    genv = GymnasiumEnv(MiniWebEnv(), task_set("train", ["click_button", "fill_field"], n_per_family=3))
    check_env(genv, skip_render_check=True)
    obs, info = genv.reset(seed=0, options={"task_index": 0})
    assert obs.startswith("GOAL:") and info["action_mask"]
    obs, r, term, trunc, info = genv.step("__import__('os')")
    assert info["action_valid"] is False and r == 0.0
    task = genv.tasks[0]
    target = next(e for e in info["observation"].elements if e.name == task.private["target"])
    obs, r, term, trunc, info = genv.step(f"click({target.eid})")
    assert term and r == 1.0 and info["verification"]["success"]
