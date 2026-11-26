"""Persistent sessions across an environment restart, CDP attach to an external Chromium, delayed consequences
through the browser (run with ``-m browser``)."""
import json
import os
import re
import stat
import subprocess
import time
import urllib.request

import pytest

from cam_live.contracts import FailureClass, Modality, OutcomeKind, OutcomeStatus, ResultStatus, TaskSpec
from cam_live.harness.browser import BrowserConfig, BrowserExecutor
from cam_live.harness.env import ComputerEnvironment
from cam_live.harness.internal import InternalExecutor

from conftest import Driver, GrantPolicy, make_action

pytestmark = pytest.mark.browser


def start(run_dir, policy=None, clock=None, config=None, task=None):
    policy = policy or GrantPolicy()
    env = ComputerEnvironment(str(run_dir), policy, clock=clock)
    env.register(InternalExecutor(str(run_dir), clock=env.clock))
    env.register(BrowserExecutor(str(run_dir), clock=env.clock, config=config))
    return Driver(env, policy, task or TaskSpec("obj_sessions", "exercise sessions", mode="staging"))


def test_persistent_session_survives_environment_restart(tmp_path, sites):
    s = sites.urls["structured"]
    d = start(tmp_path)
    d.ok("browser.navigate", url=s + "/login")
    d.ok("browser.fill", role="textbox", name="Handle", text="eb-operator")
    assert "Signed in as eb-operator." in d.ok("browser.click", role="button", name="Sign in").observations[0].text
    d.ok("browser.open_tab", url=s + "/items?q=beta")
    obs_before = d.inspect()
    d.env.close()                                                     # the whole environment process goes away

    profile = os.path.join(str(tmp_path), "browser", "profiles", "default")
    assert stat.S_IMODE(os.stat(profile).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(profile + ".storage.json").st_mode) == 0o600
    saved = json.load(open(os.path.join(str(tmp_path), "env", "state.json")))["executors"][Modality.STRUCTURED_BROWSER]
    assert sorted(t["url"] for t in saved["sessions"]["default"]["tabs"]) == [s + "/account", s + "/items?q=beta"]

    d2 = start(tmp_path)
    try:
        caps = d2.env.get_capabilities()["executors"][Modality.STRUCTURED_BROWSER]
        assert sorted(caps["restorable_tabs"]["default"]) == [s + "/account", s + "/items?q=beta"]
        # references from before the restart are outdated
        stale = d2.do("browser.click", ref=obs_before.data["elements"][0]["ref"], obs_id=obs_before.obs_id)
        assert stale.failure_class == FailureClass.STALE_OBSERVATION
        o = d2.inspect()                                              # first use: tabs come back, each URL policy-checked
        assert sorted(t["url"] for t in o.data["tabs"]) == [s + "/account", s + "/items?q=beta"]
        restores = [p for p in d2.policy.primitives if p[1].get("via") == "session_restore"]
        assert sorted(p[1]["url"] for p in restores) == [s + "/account", s + "/items?q=beta"]
        account = d2.ok("browser.navigate", url=s + "/account").observations[0]
        assert "Signed in as eb-operator." in account.text            # the persistent cookie lives in the profile
        assert "Session cookie: present." in account.text             # the session cookie was carried over as well
    finally:
        d2.env.close()

    # a restart under a policy that no longer allows one of the tabs: that tab stays closed
    d3 = start(tmp_path, GrantPolicy(deny_urls=["/items?q=beta"]))
    try:
        o = d3.inspect()
        assert s + "/items?q=beta" not in [t["url"] for t in o.data["tabs"]]
        assert "Signed in as eb-operator." in d3.ok("browser.navigate", url=s + "/account").observations[0].text
    finally:
        d3.env.close()
    # a different identity has its own profile and is not signed in
    d4 = start(tmp_path, config=BrowserConfig(identities=("default", "second")))
    try:
        other = d4.ok("browser.navigate", url=s + "/account", session="identity:second").observations[0]
        assert "Not signed in." in other.text
        assert os.path.isdir(os.path.join(str(tmp_path), "browser", "profiles", "second"))
    finally:
        d4.env.close()


def test_cdp_attach_to_a_separately_launched_chromium(tmp_path, sites):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        exe = p.chromium.executable_path
    proc = subprocess.Popen([exe, "--headless=new", "--no-sandbox", "--disable-dev-shm-usage", "--remote-debugging-port=0",
                             f"--user-data-dir={tmp_path}/external-profile", "about:blank"],
                            stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)
    try:
        endpoint, t0 = None, time.time()
        while endpoint is None and time.time() - t0 < 30:
            m = re.search(r"DevTools listening on (ws://\S+)", proc.stderr.readline())
            endpoint = m.group(1) if m else None
        assert endpoint, "external Chromium did not announce a DevTools endpoint"
        port = re.search(r":(\d+)/", endpoint).group(1)
        d = start(tmp_path / "run", config=BrowserConfig(cdp_endpoints={"ext": f"http://127.0.0.1:{port}"}))
        try:
            assert d.env.executor(Modality.STRUCTURED_BROWSER).attach_cdp("ext2", endpoint) == "cdp:ext2"
            url = sites.urls["structured"] + "/items"
            o = d.ok("browser.navigate", url=url, session="cdp:ext").observations[0]
            assert o.data["session_kind"] == "cdp" and "Beta gasket" in o.text
            d.ok("browser.fill", role="textbox", name="Filter by name", text="gamma", session="cdp:ext")
            done = d.ok("browser.click", role="button", name="Apply filter", session="cdp:ext").observations[0]
            assert done.data["url"] == url + "?q=gamma" and "1 of 3 items shown." in done.text
            # it really is the external browser: its own DevTools endpoint reports the page we drove
            pages = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list"))
            assert url + "?q=gamma" in [x["url"] for x in pages]
            # authorization applies to an attached browser exactly as to a launched one
            d.policy.deny_urls.append("/report")
            assert d.do("browser.navigate", url=sites.urls["structured"] + "/report", session="cdp:ext").status \
                == ResultStatus.DENIED
            assert d.do("browser.navigate", url=url, session="cdp:nope").failure_class == FailureClass.INVALID_ARGUMENTS
        finally:
            d.env.close()
        assert proc.poll() is None                                    # detaching does not stop the operator's browser
        pages = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list"))
        assert any(x["url"].endswith("/items?q=gamma") for x in pages)
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_delayed_consequences_through_the_browser(tmp_path, sites):
    lab, base = sites.labsite.world, sites.urls["labsite"]
    d = start(tmp_path, clock=sites.clock)
    d.env.add_source(sites.outcome_source())
    d.env.annotate("staging", sites.capabilities())
    try:
        caps = d.env.get_capabilities()
        assert caps["staging"]["sites"]["labsite"]["label"] == "staging" and caps["live"] is False
        known = {q.query_id for q in d.env.get_pending_outcomes()}
        o = d.ok("browser.navigate", url=base + "/act").observations[0]
        d.ok("browser.select_option", ref=Driver.ref(o, "combobox", "Channel"), obs_id=o.obs_id, label=lab.names[1])
        d.ok("browser.fill", ref=Driver.ref(o, "spinbutton", "Effort"), obs_id=o.obs_id, text="3")
        started = d.ok("browser.click", ref=Driver.ref(o, "button", "Start"), obs_id=o.obs_id)
        assert "intervention started on " + lab.names[1] in started.observations[0].text
        entry = sites.labsite.history[-1]
        assert entry["operation"] == "lab.act" and entry["arguments"] == {"channel": 1, "effort": 3}
        (q,) = [x for x in d.env.get_pending_outcomes() if x.query_id not in known]
        assert q.kind == OutcomeKind.COUNT and q.metric == f"lab.adopters.from.{entry['action_id']}"
        # nothing has matured yet: the page and the measurement both say so
        total0 = lab.total()
        home = d.ok("browser.navigate", url=base + "/").observations[0]
        assert {x.name: x.value for x in home.quantities}["page.root.current_state.lab_adopters_total"] == float(total0)
        assert d.env.measure([q])[0].status == OutcomeStatus.PENDING
        assert d.env.get_resource_state()["next_outcome_due_at"] <= q.due_at
        # waiting costs no browser activity; then the consequences are visible in the page and measurable
        hits = sum(sites.labsite.hits.values())
        d.ok("wait", seconds=q.due_at - sites.clock.now() + 1, reason="outcome window")
        assert sum(sites.labsite.hits.values()) == hits
        later = d.ok("browser.navigate", url=base + "/").observations[0]
        truth = sum(1 for t, a, _ in lab._arrivals if a == entry["action_id"] and t <= q.due_at)
        assert {x.name: x.value for x in later.quantities}["page.root.current_state.lab_adopters_total"] == float(lab.total())
        (m,) = d.env.measure([q])
        assert m.status == OutcomeStatus.OBSERVED and m.value == float(truth) and m.verified is False
        hist = d.ok("browser.navigate", url=base + "/history").observations[0]
        row = [r for r in hist.data["tables"][0]["rows"] if r[0] == entry["action_id"]][0]
        assert row[-1] == str(sum(1 for t, a, _ in lab._arrivals if a == entry["action_id"] and t <= sites.clock.now()))
    finally:
        d.env.close()
