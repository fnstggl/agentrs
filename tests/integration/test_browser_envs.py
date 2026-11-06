"""Integration tests for the real-browser environments (Playwright + Chromium).

Run: python -m pytest tests/integration/test_browser_envs.py -q -m browser
Every test skips (rather than fails) when Chromium cannot be launched.
"""

from __future__ import annotations

import random
import re

import pytest

pytestmark = pytest.mark.browser

playwright = pytest.importorskip("playwright.sync_api")

from agentrs.actions.schema import Action, ActionType  # noqa: E402
from agentrs.agent.runtime import Decision, TeacherActor, run_episode  # noqa: E402
from agentrs.data.schemas import Source  # noqa: E402
from agentrs.demonstrations.teacher import ScriptedTeacher  # noqa: E402
from agentrs.environments.base import TaskSpec  # noqa: E402
from agentrs.environments.miniweb.core import MiniWebEnv  # noqa: E402
from agentrs.environments.miniweb.families import HELDOUT_FAMILIES, TRAIN_FAMILIES  # noqa: E402
from agentrs.environments.miniweb.tasks import make_task  # noqa: E402


def _task(family: str, index: int = 0) -> TaskSpec:
    return make_task(family, "test_family" if family in HELDOUT_FAMILIES else "train", index)


@pytest.fixture(scope="module")
def env():
    from agentrs.environments.web.playwright_env import LocalWebEnv

    e = LocalWebEnv()
    try:
        e.start()
    except Exception as exc:  # missing browser / system libraries
        pytest.skip(f"Chromium cannot be launched: {exc}")
    yield e
    e.close()


# --------------------------------------------------------------------------- (a) teacher solves tasks
@pytest.mark.parametrize("family", TRAIN_FAMILIES + HELDOUT_FAMILIES)
def test_teacher_solves_family_on_localweb(env, family):
    task = _task(family)
    ep = run_episode(env, task, TeacherActor(ScriptedTeacher()))
    assert ep.env_id == "localweb"
    assert ep.success, (family, ep.termination_reason, [(t.action, t.error) for t in ep.transitions])
    assert ep.verification["verifier_id"] == f"localweb/{family}/v1"
    assert ep.termination_reason == "terminated:success"
    assert ep.num_invalid == 0


def test_localweb_matches_miniweb_dynamics(env):
    """Same task + same random action sequence => identical observations, errors,
    rewards and verification in the browser and in the pure-Python simulator."""
    sim = MiniWebEnv()
    to_sim = lambda s: re.sub(r"(?<=[\[( ])a(\d+)\b", r"e\1", s)  # noqa: E731 - id prefix a<n> -> e<n>
    for k, family in enumerate(["menu", "dialog_confirm", "search_lookup", "scroll_find", "toggle_settings"]):
        task, rng = _task(family, 20 + k), random.Random(k)
        assert to_sim(env.reset(task).to_text()) == sim.reset(task).to_text()
        for _ in range(task.max_steps):
            cands = sim.available_actions()
            assert [to_sim(a.to_str()) for a in env.available_actions()] == [a.to_str() for a in cands]
            a = cands[rng.randrange(len(cands))]
            r_web = env.step(a.model_copy(update={"target": "a" + a.target[1:]}) if a.target else a)
            r_sim = sim.step(a)
            assert to_sim(r_web.observation.to_text()) == r_sim.observation.to_text()
            assert (r_web.reward, r_web.terminated, r_web.truncated, r_web.action_valid) == \
                   (r_sim.reward, r_sim.terminated, r_sim.truncated, r_sim.action_valid)
            if r_sim.terminated or r_sim.truncated:
                break
        assert env.verify_outcome().success == sim.verify_outcome().success


def test_make_env_localweb_guarded_runs_across_threads():
    """make_env('localweb') wraps the env in GuardedEnvironment, which steps from
    worker threads; Playwright work must still happen on its own thread."""
    from agentrs.evaluation.benchmark import make_env

    e = make_env("localweb")
    try:
        try:
            e.inner.start()
        except Exception as exc:
            pytest.skip(f"Chromium cannot be launched: {exc}")
        ep = run_episode(e, _task("login"), TeacherActor(ScriptedTeacher()))
        assert ep.success and ep.env_id == "localweb"
    finally:
        e.close()


# --------------------------------------------------------------------------- (b) stale ids
def test_stale_id_after_navigation_is_observable_error(env):
    task = _task("navigate_link")
    obs = env.reset(task)
    link = next(e for e in obs.elements if e.role == "link")
    home_ids = obs.element_ids()
    r = env.step(Action.click(link.eid))  # real navigation -> new document, new ids
    assert r.action_valid and r.observation.url.startswith("/s/")
    assert not (r.observation.element_ids() & home_ids)
    r2 = env.step(Action.click(link.eid))
    assert r2.action_valid is False
    assert r2.error and "not found" in r2.error and link.eid in r2.error
    assert r2.observation.last_action_error == r2.error
    assert r2.observation.url == r.observation.url  # nothing happened
    # other malformed interactions are observable errors too, never crashes
    heading = next(e for e in r2.observation.elements if e.role == "heading")
    r3 = env.step(Action.type_text(heading.eid, "x"))
    assert not r3.action_valid and "not editable" in r3.error
    r4 = env.step(Action.select(heading.eid, "x"))
    assert not r4.action_valid and "not a select box" in r4.error


# --------------------------------------------------------------------------- (c) network allowlist
def test_external_navigation_and_control_channel_are_blocked(env):
    task = _task("navigate_link", 1)
    obs = env.reset(task)
    r = env.step(Action.navigate("https://example.com"))
    assert r.action_valid is False and "not permitted" in r.error
    assert r.observation.url == obs.url
    # navigation to an app path outside the app's allowlist is refused as in MiniWebEnv
    r = env.step(Action.navigate("/__control/verify"))
    assert r.action_valid is False and "not permitted" in r.error

    # Defense in depth at the network layer: even direct browser requests are aborted.
    def probe(page):
        html = page.content()
        control_url = f"{env._base}/__control/{env._token}/verify"
        fetched = page.evaluate(
            "async (u) => { try { await fetch(u, {method: 'POST', body: '{}'}); return 'reached'; }"
            " catch (e) { return 'blocked'; } }", control_url)
        try:
            page.goto("https://example.com/", timeout=5000)
            external = "reached"
        except playwright.Error as e:
            external = str(e)
        return external, fetched, html

    external, fetched, html = env.run_in_browser(probe)
    assert "ERR_BLOCKED_BY_CLIENT" in external
    assert fetched == "blocked"  # the verifier is unreachable from the page, even with the token
    assert env._token not in html  # the secret never appears in a page
    assert any(u.startswith("https://example.com") for u in env.blocked_requests)
    assert any("/__control/" in u for u in env.blocked_requests)
    # the env notices the browser error page and returns to the app page
    obs = env.observe()
    assert obs.url == "/" and any(e.role == "link" for e in obs.elements)


# --------------------------------------------------------------------------- (d) verifier cannot be fooled
def test_wrong_actions_yield_verified_failure(env):
    # wrong button: the episode terminates and the server-side verifier says failure
    task = _task("click_button", 3)
    obs = env.reset(task)
    target = re.findall(r'"([^"]+)"', task.goal)[0]
    wrong = next(e for e in obs.elements if e.role == "button" and e.name != target)
    r = env.step(Action.click(wrong.eid))
    assert r.terminated and r.reward == 0.0
    assert r.info["verification"]["success"] is False
    assert env.verify_outcome().success is False

    # claiming success with finish() does not make it so
    for family, claim in (("navigate_link", None), ("search_lookup", "$0.01"), ("checkout", "done")):
        env.reset(_task(family, 2))
        r = env.step(Action.finish(claim))
        assert r.terminated and r.reward == 0.0 and r.info["verification"]["success"] is False
        assert env.verify_outcome().success is False

    # a random policy is judged by the verifier too
    env.reset(_task("login", 4))
    rng = random.Random(0)
    while True:
        cands = [a for a in env.available_actions() if a.type != ActionType.FINISH]
        r = env.step(rng.choice(cands))
        if r.terminated or r.truncated:
            break
    assert r.reward == 0.0 and env.verify_outcome().success is False


# --------------------------------------------------------------------------- generic pages
_GENERIC_PAGE = """<!doctype html><html><head><title>Generic page</title></head><body>
<h1>Account</h1>
<p>Welcome <b>back</b>, friend.</p>
<label for="em">Email</label><input id="em" type="email" placeholder="you@x.org">
<label>Password <input id="pw" type="password" value="hunter2"></label>
<input id="q" type="search" aria-label="Search site">
<select id="plan"><option>Free</option><option selected>Pro</option></select>
<input type="submit" value="Save changes">
<button id="go" onclick="document.getElementById('out').textContent = 'Saved ' + document.getElementById('em').value">Apply</button>
<button disabled>Locked</button>
<button aria-expanded="false" aria-haspopup="menu">More</button>
<div role="switch" aria-checked="true" tabindex="0">Wifi</div>
<div contenteditable="true" aria-label="Notes">draft</div>
<p id="out"></p>
<div style="display:none"><button>Hidden</button></div>
<button style="visibility:hidden">Invisible</button>
<div aria-hidden="true"><button>Decorative</button></div>
<a id="ext" href="https://example.com/">External site</a>
<div style="height:2000px"></div><button>Far below</button>
</body></html>"""


@pytest.fixture(scope="module")
def generic_site():
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            body = _GENERIC_PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/"
    srv.shutdown()
    srv.server_close()


def test_generic_extractor_and_dispatcher(generic_site):
    from agentrs.environments.web.playwright_env import PlaywrightEnv

    env = PlaywrightEnv(start_url=generic_site)
    try:
        try:
            env.start()
        except Exception as exc:
            pytest.skip(f"Chromium cannot be launched: {exc}")
        task = TaskSpec(task_id="generic/0", env_id="playwright", family="generic", goal='Enter "a@b.co"',
                        max_steps=50)
        obs = env.reset(task)
        by_name = {e.name: e for e in obs.elements}
        assert obs.title == "Generic page" and obs.scroll_info == "more content below"
        assert by_name["Account"].role == "heading"
        assert by_name["Welcome back, friend."].role == "text"
        assert by_name["Email"].role == "textbox" and by_name["Email"].value == ""
        assert by_name["Password"].value == "•" * 7  # secrets are masked
        assert by_name["Search site"].role == "searchbox"
        plan = next(e for e in obs.elements if e.role == "combobox")
        assert plan.value == "Pro" and plan.options == ["Free", "Pro"]
        assert by_name["Save changes"].role == "button"
        assert by_name["Locked"].disabled
        assert by_name["More"].role == "menubutton" and by_name["More"].expanded is False
        assert by_name["Wifi"].role == "switch" and by_name["Wifi"].checked is True
        assert by_name["Notes"].role == "textbox" and by_name["Notes"].value == "draft"
        for hidden in ("Hidden", "Invisible", "Decorative", "Far below"):  # invisible / off-screen
            assert hidden not in by_name
        assert all(re.fullmatch(r"a\d+", e.eid) for e in obs.elements)

        r = env.step(Action.type_text(by_name["Email"].eid, "a@b.co"))
        assert r.action_valid and r.observation.element(by_name["Email"].eid).value == "a@b.co"
        r = env.step(Action.click(by_name["Apply"].eid))
        assert any(e.role == "text" and e.name == "Saved a@b.co" for e in r.observation.elements)
        r = env.step(Action.select(plan.eid, "free"))
        assert r.action_valid and r.observation.element(plan.eid).value == "Free"
        r = env.step(Action.click(by_name["Locked"].eid))
        assert not r.action_valid and "disabled" in r.error
        for _ in range(8):  # half a viewport per scroll
            r = env.step(Action.scroll("down"))
            if any(e.name == "Far below" for e in r.observation.elements):
                break
        assert any(e.name == "Far below" for e in r.observation.elements)
        assert r.observation.scroll_info == "more content above"
        while r.observation.scroll_info and "above" in r.observation.scroll_info:
            r = env.step(Action.scroll("up"))
        assert by_name["External site"].eid in r.observation.element_ids()  # same ids after scrolling back
        # clicking a link that leaves the allowlist: aborted, reported, page restored
        r = env.step(Action.click(by_name["External site"].eid))
        assert not r.action_valid and "blocked" in r.error
        assert r.observation.url == generic_site
        r = env.step(Action.finish("done"))
        assert r.action_valid and r.terminated and r.reward == 0.0  # no verifier => never a success
        assert env.verify_outcome().verifier_id == "playwright/unverified"
    finally:
        env.close()


# --------------------------------------------------------------------------- (e) BrowserGym
@pytest.fixture(scope="module")
def bg_env():
    pytest.importorskip("browsergym.miniwob")
    from agentrs.environments.browsergym_adapter import DEFAULT_MINIWOB_DIR, BrowserGymEnv

    if not (DEFAULT_MINIWOB_DIR / "click-test.html").exists():
        pytest.skip("MiniWoB++ HTML not available")
    e = BrowserGymEnv()
    try:
        e.reset()
    except Exception as exc:
        pytest.skip(f"BrowserGym cannot start: {exc}")
    yield e
    e.close()


class _ClickTextActor:
    """Minimal actor: click the button whose label is quoted in the goal (or the only button)."""

    name = "test:click-text"

    def begin_episode(self, belief):
        pass

    def decide(self, belief, candidates):
        obs = belief.observation
        want = re.findall(r'"([^"]+)"', belief.task.goal)
        buttons = [e for e in obs.elements if e.role == "button"]
        pick = next((e for e in buttons if want and e.name == want[0]), buttons[0] if buttons else None)
        a = Action.click(pick.eid) if pick else Action.finish()
        return Decision(action=a, candidates=[c.to_str() for c in candidates], selection="policy")


def test_browsergym_click_test_end_to_end(bg_env):
    task = TaskSpec(task_id="browsergym/miniwob.click-test/0", env_id="browsergym", family="miniwob.click-test",
                    goal="", seed=0, max_steps=4)
    ep = run_episode(bg_env, task, _ClickTextActor())
    assert task.goal == "Click the button."  # filled in from the benchmark at reset
    assert ep.success is True and ep.termination_reason == "terminated:success"
    assert ep.verification["verifier_id"] == "browsergym/miniwob.click-test"
    first = ep.transitions[0]
    assert first.env_id == "browsergym" and re.fullmatch(r"click\(b\d+\)", first.action)
    assert any(e.role == "button" and e.name == "Click Me!" for e in first.observation.elements)


def test_browsergym_click_button_success_and_failure(bg_env):
    spec = lambda: TaskSpec(task_id="bg/click-button", env_id="browsergym",  # noqa: E731
                            family="miniwob.click-button", goal="", seed=5, max_steps=4)
    ep = run_episode(bg_env, spec(), _ClickTextActor())
    assert ep.success is True
    # wrong button => the benchmark's own verifier reports failure
    t = spec()
    obs = bg_env.reset(t)
    want = re.findall(r'"([^"]+)"', t.goal)[0]
    wrong = next(e for e in obs.elements if e.role == "button" and e.name != want)
    r = bg_env.step(Action.click(wrong.eid))
    assert r.terminated and r.reward == 0.0 and bg_env.verify_outcome().success is False
    # stale ids / navigation are observable errors
    bg_env.reset(spec())
    r = bg_env.step(Action.click("b99999"))
    assert not r.action_valid and "not found" in r.error
    r = bg_env.step(Action.navigate("https://example.com"))
    assert not r.action_valid and "not permitted" in r.error


# --------------------------------------------------------------------------- (f) human demonstrations
def test_recorder_captures_click_and_fill_with_human_provenance(env):
    from agentrs.demonstrations.browser_recorder import record_episode

    task = _task("fill_field", 5)
    value = re.findall(r'"([^"]+)"', task.goal)[0]
    seen = {}

    def driver(rec):
        box = next(e for e in rec.observation.elements
                   if e.role == "textbox" and re.search(rf"\b{re.escape(e.name)}\b", task.goal.replace(value, "")))
        seen["box"] = box.eid
        rec.type(box.eid, value)  # real keyboard input + blur
        submit = next(e for e in rec.observation.elements if e.role == "button" and e.name == "Submit")
        seen["submit"] = submit.eid
        rec.click(submit.eid)  # real mouse click

    ep = record_episode(task, consent=True, demonstrator_id="demo-user-1", driver=driver, env=env)
    assert [t.action for t in ep.transitions] == [f'type({seen["box"]}, "{value}")', f'click({seen["submit"]})']
    assert [t.action_type for t in ep.transitions] == ["type", "click"]
    assert ep.success is True and ep.termination_reason == "terminated:success"
    assert ep.provenance.source == Source.HUMAN and ep.provenance.consent is True
    assert ep.provenance.source_id == "demo-user-1" and ep.provenance.privacy == "private"
    for tr in ep.transitions:
        assert tr.provenance.source == Source.HUMAN and tr.provenance.consent and tr.selection == "human"
        assert tr.episode_success is True
    t0, t1 = ep.transitions
    assert t0.observation.element(seen["box"]).value == ""  # state before the gesture
    assert t0.next_observation.element(seen["box"]).value == value
    assert t1.terminated and t1.reward == 1.0 and t1.verification["success"] is True
    assert ep.metadata["unmapped_events"] == 0


def test_recorder_select_scroll_wait_and_finish(env):
    from agentrs.demonstrations.browser_recorder import BrowserDemoRecorder

    rec = BrowserDemoRecorder(consent=True, demonstrator_id="demo-user-2", privacy="internal", env=env)
    # dropdown: select + click
    task = _task("dropdown", 6)
    opt = re.findall(r'"([^"]+)"', task.goal)[0]
    obs = rec.start(task)
    box = next(e for e in obs.elements if e.role == "combobox")
    assert rec.select(box.eid, opt).action == f'select({box.eid}, "{opt}")'
    rec.click(next(e for e in rec.observation.elements if e.name == "Submit").eid)
    ep = rec.stop()
    assert ep.success and ep.provenance.privacy == "internal"
    # search: typing, waiting for async results, answering with finish
    task = _task("search_lookup", 6)
    product = re.findall(r'"([^"]+)"', task.goal)[0]
    obs = rec.start(task)
    rec.type(next(e for e in obs.elements if e.role == "searchbox").eid, product)
    rec.click(next(e for e in rec.observation.elements if e.name == "Search").eid)
    for _ in range(3):
        hit = next((e for e in rec.observation.elements if e.name.startswith(product + " - ")), None)
        if hit:
            break
        rec.wait()
    assert hit is not None
    rec.finish(hit.name.split(" - ")[-1])
    ep = rec.stop()
    assert ep.success and ep.transitions[-1].action.startswith("finish(")
    # scrolling with the mouse wheel becomes scroll(down)
    obs = rec.start(_task("scroll_find", 6))
    tr = rec.scroll("down")
    assert tr is not None and tr.action == "scroll(down)"
    assert tr.next_observation.elements[0].name != obs.elements[0].name
    assert rec.stop().termination_reason == "truncated:stopped"


def test_recorder_refuses_without_consent():
    from agentrs.demonstrations.browser_recorder import BrowserDemoRecorder, ConsentRequired, record_episode

    task = _task("click_button")
    with pytest.raises(ConsentRequired):
        record_episode(task, consent=False, demonstrator_id="someone")
    with pytest.raises(ConsentRequired):
        record_episode(task, consent=True, demonstrator_id="   ")
    with pytest.raises(ConsentRequired):
        BrowserDemoRecorder(consent="yes", demonstrator_id="someone")  # only an explicit True counts
    with pytest.raises(ValueError):
        BrowserDemoRecorder(consent=True, demonstrator_id="someone", privacy="public-ish")
