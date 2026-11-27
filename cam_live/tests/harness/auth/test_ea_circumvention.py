"""A denied action may not be re-issued through another modality or with cosmetic changes."""
from cam_live import contracts as C
from cam_live.harness.auth import DEFAULT_POLICY_PATH
from ea_helpers import ALLOW, DENY, LOOP, NEEDS, act, authorize_live, live_task, prim, reopen, start, task

BLOCKED = "https://blocked.example/data"


def test_denied_browser_action_reissued_as_code_or_api_is_denied(svc):
    t = task()
    code = lambda: act("sandbox.python", arguments={"code": "import urllib.request as u\nprint(u.urlopen('http://blocked.example/data').read())"})  # noqa: E731
    # before any denial the same program is just local code (the sandbox has no network)
    fresh = task(oid="fresh")
    assert svc.authorize(code(), fresh, {}).verdict == ALLOW
    d = svc.authorize(act("browser.navigate", arguments={"url": BLOCKED}), t, {})
    assert d.verdict == DENY and "X-CIRCUMVENTION" not in d.policy_ids
    reissues = [
        code(),                                                                                  # code
        act("sandbox.shell", arguments={"cmd": "curl -s HTTPS://BLOCKED.EXAMPLE/data"}),         # shell, different case
        act("http.get", arguments={"url": BLOCKED}),                                             # api
        act("api.request", arguments={"method": "GET", "url": "http://www.blocked.example:80/data/?x=1#top"}),   # cosmetic
        act("skill.run", arguments={"skill_id": "fetcher", "args": {"site": "blocked.example"}}),  # skill
        act("visual.type", arguments={"text": "blocked.example/data\n", "page_url": BLOCKED}),   # visual
        act("job.start", arguments={"purpose": "fetch", "code": "wget blocked.example/data"}),   # background job
        act("skill.run", steps=[act("wait"), act("browser.navigate", arguments={"url": "blocked.example"})]),   # nested
    ]
    for a in reissues:
        r = svc.authorize(a, t, {})
        assert r.verdict == DENY and "X-CIRCUMVENTION" in r.policy_ids, (a.operation, r.reasons)
        assert not r.grant
    assert any("another modality" in x for x in svc.authorize(code(), t, {}).reasons)
    # unrelated work is untouched, and another task is not affected by this task's denials
    assert svc.authorize(act("sandbox.python", arguments={"code": "print('hello')"}), t, {}).verdict == ALLOW
    assert svc.authorize(code(), fresh, {}).verdict == ALLOW


def test_denied_write_cannot_be_reissued_but_reading_stays_possible(svc, tmp_path, clock):
    t = live_task()
    authorize_live(svc, tmp_path, t, clock, read=["forum.example", "docs.example"])
    post = act("publish.post", arguments={"destination": "https://forum.example/thread/5", "body": "a reply to the thread, long enough"})
    d = svc.authorize(post, t, {})
    assert d.verdict == DENY and "P-WRITE-NON-OWNED" in d.policy_ids
    for a in (act("http.post", arguments={"url": "https://forum.example/api/reply", "json": {"t": 5}}),
              act("browser.fill", arguments={"ref": "e9", "text": "a reply", "page_url": "https://www.forum.example/thread/5"}),
              act("sandbox.python", arguments={"code": "requests.post('https://forum.example/api/reply')"})):
        r = svc.authorize(a, t, {})
        assert r.verdict == DENY and "X-CIRCUMVENTION" in r.policy_ids, a.operation
    assert svc.authorize(act("browser.navigate", arguments={"url": "https://forum.example/thread/5"}), t, {}).verdict == ALLOW
    assert svc.authorize(act("browser.click", arguments={"ref": "e2", "page_url": "https://forum.example/thread/5"}), t, {}).verdict == ALLOW


def test_denied_channel_cannot_be_reached_through_raw_requests_or_skills(svc):
    chan = "900000000000004242"
    t = task(resources={"owned": ["channel:discord:900000000000000011"], "preauthorized": ["channel:discord:*"]})
    d = svc.authorize(act("discord.send_message", arguments={"channel_id": chan, "content": "hello over there, everyone"}), t, {})
    assert d.verdict == DENY and "P-WRITE-NON-OWNED" in d.policy_ids
    raw = act("http.post", arguments={"url": f"{LOOP}/api/v10/channels/{chan}/messages", "json": {"content": "hello"}})
    r = svc.authorize(raw, t, {})
    assert r.verdict == DENY and r.policy_ids == ["X-CIRCUMVENTION"]          # loopback is allowed; the target is not
    sk = act("skill.run", arguments={"skill_id": "poster", "args": {"channel": chan, "text": "hello"}})
    assert svc.authorize(sk, t, {}).policy_ids == ["X-CIRCUMVENTION"]
    # and at primitive level inside an otherwise allowed action
    carrier = act("skill.run", arguments={"skill_id": "notes"})
    start(svc, carrier, t)
    p = svc.check_primitive(prim("net.request", C.Modality.SKILL, method="POST",
                                 url=f"{LOOP}/api/v10/channels/{chan}/messages", json={"content": "x"}), carrier)
    assert p.verdict == DENY and "X-CIRCUMVENTION" in p.policy_ids


def test_denied_file_target_cannot_be_read_by_code(svc):
    t = task()
    d = svc.authorize(act("sandbox.read_file", arguments={"path": "/etc/hostname"}), t, {})
    assert d.verdict == DENY and "P-FILE-ROOT" in d.policy_ids
    r = svc.authorize(act("sandbox.shell", arguments={"cmd": "cat /etc/hostname"}), t, {})
    assert r.verdict == DENY and "X-CIRCUMVENTION" in r.policy_ids


def test_rejected_content_cannot_be_resubmitted_with_cosmetic_changes(svc):
    chan = "900000000000000011"
    t = task(resources={"owned": [f"channel:discord:{chan}", "channel:discord:900000000000000022"]})
    text = "Version two of the handbook is out; it covers setup, the review checklist and the release calendar."
    a = act("discord.send_message", arguments={"channel_id": chan, "content": text})
    d = svc.authorize(a, t, {})
    assert d.verdict == NEEDS
    svc.resolve_approval(d.approval_id, False, "olga", "no")
    variants = [
        act("discord.send_message", arguments={"channel_id": chan, "content": text.upper() + " !!!"}),
        act("discord.send_message", arguments={"channel_id": "900000000000000022", "content": "  " + text.replace(";", " -")}),
        act("http.post", arguments={"url": LOOP + "/api/channels/x/messages", "json": {"content": text + "\n"}}),
    ]
    for v in variants:
        r = svc.authorize(v, t, {})
        assert r.verdict == DENY and "X-CIRCUMVENTION" in r.policy_ids, v.arguments
    rewritten = act("discord.send_message", arguments={"channel_id": chan, "content":
                    "A short notice instead: the handbook has moved to the new documentation site as of today."})
    assert svc.authorize(rewritten, t, {}).verdict == NEEDS                    # genuinely new content goes to the operator


def test_denials_persist_across_restart_and_are_cleared_only_by_the_operator(svc, clock, tmp_path):
    t = task()
    assert svc.authorize(act("browser.navigate", arguments={"url": BLOCKED}), t, {}).verdict == DENY
    code = lambda: act("sandbox.python", arguments={"code": "open_url('blocked.example')"})   # noqa: E731
    s2 = reopen(svc, clock)
    assert "X-CIRCUMVENTION" in s2.authorize(code(), t, {}).policy_ids
    s2.reload_policy("olga")                                   # same rules: nothing is forgotten
    assert "X-CIRCUMVENTION" in s2.authorize(code(), t, {}).policy_ids
    edited = tmp_path / "p.yaml"
    edited.write_text(open(DEFAULT_POLICY_PATH).read() + "\n# reviewed by the operator\n")
    assert s2.reload_policy("olga", str(edited))["valid"]      # the operator changed the rules: decisions start afresh
    assert s2.authorize(code(), t, {}).verdict == ALLOW
    assert s2.authorize(act("browser.navigate", arguments={"url": BLOCKED}), t, {}).verdict == DENY
    s2.register_task(t, "olga")                                # re-registering the task clears its denials as well
    assert s2.authorize(code(), t, {}).verdict == ALLOW
    s2.close()


def test_resource_denials_are_not_remembered_as_forbidden_targets(svc, clock):
    t = task(limits=C.Limits(max_external_actions=1))
    a = act("http.get", arguments={"url": LOOP + "/a"})
    start(svc, a, t)
    svc.record(a, C.ActionResult(a.action_id, C.ResultStatus.OK))
    d = svc.authorize(act("http.get", arguments={"url": LOOP + "/b"}), t, {})
    assert d.verdict == DENY and d.policy_ids == ["L-EXTERNAL-ACTIONS"]
    code = act("sandbox.python", arguments={"code": "print('127.0.0.1')"})
    assert svc.authorize(code, t, {}).verdict == ALLOW          # running out of budget is not a prohibition of the place
