"""Approval queue: persistence, binding to the action hash, expiry, single use, listing, primitive approvals."""
from cam_live import contracts as C
from ea_helpers import ALLOW, DENY, LOOP, NEEDS, act, prim, reopen, start, task

CHAN, CHAN2 = "900000000000000011", "900000000000000022"
TEXT = "The weekly notes are published: the build log, the open questions and the next milestone are listed there."


def dtask(**kw):
    return task(resources={"owned": [f"channel:discord:{CHAN}", f"channel:discord:{CHAN2}"]}, **kw)


def send(channel=CHAN, content=TEXT):
    return act("discord.send_message", arguments={"channel_id": channel, "content": content})


def test_request_approve_execute_once(svc):
    a, t = send(), dtask()
    d = svc.authorize(a, t, {})
    assert d.verdict == NEEDS and d.approval_id and not d.grant
    (entry,) = svc.approvals()
    assert entry["status"] == "pending" and entry["approval_id"] == d.approval_id and entry["action_hash"] == a.content_hash()
    pv = entry["preview"]                       # exactly what would be executed
    assert pv["operation"] == "discord.send_message" and TEXT in pv["render"]
    assert {"class": "write", "destination": f"channel:discord:{CHAN}", "operation": "discord.send_message"} in pv["effects"]
    assert pv["content"][0]["text"].count(TEXT) == 1 and entry["reasons"] and entry["policy_ids"] == ["P-PUBLISH-APPROVAL"]
    assert svc.authorize(a, t, {}).approval_id == d.approval_id          # asking again does not duplicate the request
    assert len(svc.approvals()) == 1
    r = svc.resolve_approval(d.approval_id, True, "olga", "looks fine")
    assert r.verdict == ALLOW and r.grant and r.approval_id == d.approval_id
    (entry,) = svc.approvals()
    assert entry["status"] == "approved" and entry["resolved_by"] == "olga" and entry["note"] == "looks fine" and entry["resolved_at"]
    d2 = svc.authorize(a, t, {})                                         # the runtime may also simply ask again
    assert d2.verdict == ALLOW and d2.approval_id == d.approval_id
    assert svc.check_grant(a, r) is True
    assert svc.check_grant(a, d2) is False                               # one approval, one execution
    d3 = svc.authorize(a, t, {})
    assert d3.verdict == NEEDS and d3.approval_id != d.approval_id       # doing it again needs a new approval


def test_approval_binds_to_the_action_hash(svc):
    a, t = send(), dtask()
    d = svc.authorize(a, t, {})
    svc.resolve_approval(d.approval_id, True, "olga")
    edited = send(content=TEXT + " One more sentence that the operator never saw before approving.")
    d2 = svc.authorize(edited, t, {})
    assert d2.verdict == NEEDS and d2.approval_id != d.approval_id       # an edited action needs its own approval
    other_channel = send(CHAN2)
    assert svc.authorize(other_channel, t, {}).verdict == NEEDS
    assert svc.authorize(a, t, {}).verdict == ALLOW


def test_rejection_is_final_for_that_action(svc):
    a, t = send(), dtask()
    d = svc.authorize(a, t, {})
    r = svc.resolve_approval(d.approval_id, False, "olga", "not this")
    assert r.verdict == DENY and r.policy_ids == ["P-APPROVAL-REJECTED"]
    d2 = svc.authorize(a, t, {})
    assert d2.verdict == DENY and "P-APPROVAL-REJECTED" in d2.policy_ids and not d2.grant
    again = svc.resolve_approval(d.approval_id, True, "someone else")    # a decision is not silently changed
    assert again.verdict == DENY and again.policy_ids == ["P-APPROVAL-INVALID"]
    assert svc.resolve_approval("apr_does_not_exist", True, "olga").policy_ids == ["P-APPROVAL-INVALID"]


def test_pending_and_approved_requests_expire(svc, clock):
    a, t = send(), dtask()
    d = svc.authorize(a, t, {})
    clock.advance(svc.policy.approvals_ttl_s + 1)
    assert svc.approvals()[0]["status"] == "expired"
    assert svc.resolve_approval(d.approval_id, True, "olga").verdict == DENY
    d2 = svc.authorize(a, t, {})
    assert d2.verdict == NEEDS and d2.approval_id != d.approval_id
    svc.resolve_approval(d2.approval_id, True, "olga")
    clock.advance(svc.policy.approvals_ttl_s + 1)                        # approved but never used
    assert svc.authorize(a, t, {}).verdict == NEEDS


def test_queue_survives_restart(svc, clock):
    a, t = send(), dtask()
    d = svc.authorize(a, t, {})
    s2 = reopen(svc, clock)
    assert [e["approval_id"] for e in s2.approvals()] == [d.approval_id]
    s2.resolve_approval(d.approval_id, True, "olga")
    s3 = reopen(s2, clock)
    assert s3.approvals()[0]["status"] == "approved"
    d2 = s3.authorize(a, t, {})
    assert d2.verdict == ALLOW and s3.check_grant(a, d2)
    s3.close()


def test_listing_puts_pending_first_and_hides_secrets(svc):
    svc.put_secret("site_token", "tok_Zx81kQ0pLm4RvT7yWb2N", ["https://api.owned.example/*"])
    t = dtask()
    d1 = svc.authorize(send(content=TEXT + " first"), t, {})
    d2 = svc.authorize(send(content=TEXT + " second"), t, {})
    svc.resolve_approval(d1.approval_id, False, "olga")
    lst = svc.approvals()
    assert [e["status"] for e in lst] == ["pending", "rejected"] and lst[0]["approval_id"] == d2.approval_id
    assert "tok_Zx81kQ0pLm4RvT7yWb2N" not in repr(lst)


def test_an_approved_action_cannot_be_used_for_something_else(svc):
    a, t = send(), dtask()
    d = svc.authorize(a, t, {})
    r = svc.resolve_approval(d.approval_id, True, "olga")
    assert svc.check_grant(a, r)
    # the approved write is covered, a different channel or a different kind of write inside it is not
    assert svc.check_primitive(prim("discord.send_message", C.Modality.API, channel_id=CHAN, content=TEXT), a).verdict == ALLOW
    p2 = svc.check_primitive(prim("discord.send_message", C.Modality.API, channel_id=CHAN2, content=TEXT), a)
    assert p2.verdict == NEEDS and p2.approval_id
    assert [e["kind"] for e in svc.approvals() if e["status"] == "pending"] == ["primitive"]
    p3 = svc.check_primitive(prim("discord.send_message", C.Modality.API, channel_id="900000000000009999", content=TEXT), a)
    assert p3.verdict == DENY and "P-WRITE-NON-OWNED" in p3.policy_ids


def test_a_skill_has_no_more_permission_than_its_primitives(svc):
    t = dtask()
    skill = act("skill.run", arguments={"skill_id": "post-notes"})       # no declared steps: allowed as local
    start(svc, skill, t)
    op = prim("discord.send_message", C.Modality.SKILL, channel_id=CHAN, content=TEXT)
    p = svc.check_primitive(op, skill)
    assert p.verdict == NEEDS and p.approval_id                          # the same operation issued alone needs approval too
    assert svc.check_primitive(op, skill).approval_id == p.approval_id
    ok = svc.resolve_approval(p.approval_id, True, "olga")
    assert ok.verdict == ALLOW and not ok.grant
    assert svc.check_primitive(op, skill).verdict == ALLOW               # used up by this one operation
    assert svc.check_primitive(op, skill).verdict == NEEDS
    assert svc.check_primitive(prim("browser.navigate", C.Modality.SKILL, url="https://public.example/"), skill).verdict == DENY
    assert svc.check_primitive(prim("warp.drive", C.Modality.SKILL), skill).policy_ids == ["P-TOOL-UNKNOWN"]


def test_transport_requests_of_an_approved_logical_write_are_covered(svc, tmp_path, clock):
    from ea_helpers import authorize_live, live_task
    t = live_task()
    authorize_live(svc, tmp_path, t, clock, owned=["channel:discord:*", "https://discord.com/api/*"], read=["discord.com"])
    a = send()
    d = svc.authorize(a, t, {})
    assert d.verdict == NEEDS
    r = svc.resolve_approval(d.approval_id, True, "olga")
    assert svc.check_grant(a, r)
    url = f"https://discord.com/api/v10/channels/{CHAN}/messages"
    assert svc.check_primitive(prim("net.request", C.Modality.API, method="POST", url=url, json={"content": TEXT}), a).verdict == ALLOW
    other = f"https://discord.com/api/v10/channels/{CHAN2}/messages"
    assert svc.check_primitive(prim("net.request", C.Modality.API, method="POST", url=other, json={"content": TEXT}), a).verdict == NEEDS
    # the same raw request issued as an action of its own is not pre-approved by anything
    raw = act("http.post", arguments={"url": url, "json": {"content": TEXT}})
    assert svc.authorize(raw, t, {}).verdict == NEEDS
