"""Approvals: bound to the exact action, single use, rejection final, survive restart — and what an approval or a
pre-authorization actually covers at the primitive level.

DEFECTS probed
 1. Approval / pre-authorization coverage is per HOST, the operator's pattern is per PATH (policy.py
    check_primitive.covered(): `if key == f.dest.intent_key: return True`; destinations.py intent_key =
    "url:<host>"). A grant for a pre-authorized path makes every write primitive on any other path of that host
    "covered": the P-WRITE-APPROVAL finding for the non-pre-authorized path is dropped without asking the operator.
 2. A logical-destination grant (channel id) covers ANY primitive on the same platform whose arguments merely
    contain that id as a substring (`ident.lower() in blob`): a different endpoint, a different HTTP method.
 3. An approval is bound to the action's content hash only. For actions whose meaning depends on the page
    (browser.click by element ref), the same bytes approved for one page are accepted for another page.
"""
import copy

from _common import ALLOW, DENY, NEEDS, C, Report, cleanup, fresh_dir
from ea_helpers import act, authorize_live, live_task, prim, reopen, start, task
from cam_live.harness.auth import PolicyService

R = Report("p04_approvals")
API, MB = C.Modality.API, C.Modality.STRUCTURED_BROWSER

# ---------------------------------------------------------------- A. basic approval properties (live mode)
d = fresh_dir("p04a")
clock = C.SimClock()
svc = PolicyService(d, clock=clock, fsync=False)
t = live_task(oid="obj-appr")
authorize_live(svc, d, t, clock, owned=["https://owned.example/*", "channel:discord:*"],
               preauthorized=["https://owned.example/blog/*"])
w = act("http.post", API, arguments={"url": "https://owned.example/admin/settings", "json": {"title": "new title"}})
v = svc.authorize(w, t)
R.check(v.verdict == NEEDS and v.approval_id and not v.grant, "write to an owned, not pre-authorized path needs approval")
R.check(svc.check_grant(w, v) is False, "a needs_approval decision is not a grant")
w2 = copy.deepcopy(w)
w2.arguments["json"]["title"] = "another title"
svc.resolve_approval(v.approval_id, True, "operator")
v2 = svc.authorize(w2, t)
R.check(v2.verdict == NEEDS and v2.approval_id != v.approval_id, "an edited action does not inherit the approval")
svc = reopen(svc, clock)
va = svc.authorize(w, t)
R.check(va.verdict == ALLOW and va.approval_id == v.approval_id, "approved action is allowed after a service restart")
vb = svc.authorize(w, t)                                    # second grant on the same approval
assert svc.check_grant(w, va)
R.check(svc.check_grant(w, vb) is False, "one approval = one execution (second grant on the same approval refused)")
vc = svc.authorize(w, t)
R.check(vc.verdict == NEEDS and vc.approval_id != v.approval_id, "re-issuing the executed action needs a new approval")
rej = svc.authorize(w2, t)
svc.resolve_approval(rej.approval_id, False, "operator", "no")
R.check(svc.authorize(w2, t).verdict == DENY, "rejection is final for the identical action")
w3 = copy.deepcopy(w2)
w3.action_id = "another-id"
v3 = svc.authorize(w3, t)
R.check(v3.verdict == DENY, "rejected content re-submitted under a new action id is refused", ",".join(v3.policy_ids))
bad = svc.resolve_approval("apr-does-not-exist", True, "operator")
R.check(bad.verdict == DENY, "unknown approval id cannot be resolved")
again = svc.resolve_approval(v.approval_id, False, "operator")
R.check(again.verdict == DENY and "P-APPROVAL-INVALID" in again.policy_ids, "a resolved approval cannot be re-decided")

# ---------------------------------------------------------------- B. host-granular coverage (defect 1)
ok = act("http.post", API, arguments={"url": "https://owned.example/blog/post-1", "json": {"text": "hello"}})
g = svc.authorize(ok, t)
assert g.verdict == ALLOW and not g.approval_id, g.reasons     # pre-authorized path: no approval involved
assert svc.check_grant(ok, g)
direct = svc.authorize(act("http.delete", API, arguments={"url": "https://owned.example/admin/site"}), t)
p = svc.check_primitive(prim("net.request", API, method="DELETE", url="https://owned.example/admin/site"), ok)
R.check(p.verdict != ALLOW,
        "a write primitive on a NOT pre-authorized path still needs the operator when the grant was for another path",
        f"as an action: {direct.verdict}; as a primitive under the /blog grant: {p.verdict}")
svc.record(ok, C.ActionResult(action_id=ok.action_id, status=C.ResultStatus.OK))

# ---------------------------------------------------------------- C. substring coverage of a channel grant (defect 2)
msg = act("discord.send_message", API, arguments={"channel_id": "123456789", "content": "release notes are up"})
vm = svc.authorize(msg, t)
svc.resolve_approval(vm.approval_id, True, "operator")          # the operator approved ONE message in ONE channel
gm = svc.authorize(msg, t)
assert gm.verdict == ALLOW and svc.check_grant(msg, gm)
# the record must own the API host for the bot to work at all; give it and look only at the approval logic
svc.close()
cleanup(d)
d = fresh_dir("p04c")
clock = C.SimClock()
svc = PolicyService(d, clock=clock, fsync=False)
t = live_task(oid="obj-appr2")
authorize_live(svc, d, t, clock, owned=["https://discord.com/api/*", "channel:discord:*"], preauthorized=[])
vm = svc.authorize(msg, t)
svc.resolve_approval(vm.approval_id, True, "operator")
gm = svc.authorize(msg, t)
assert gm.verdict == ALLOW and svc.check_grant(msg, gm), gm.reasons
same = svc.check_primitive(prim("net.request", API, method="POST",
                                url="https://discord.com/api/v10/channels/123456789/messages"), msg)
other = svc.check_primitive(prim("net.request", API, method="DELETE",
                                 url="https://discord.com/api/v10/channels/123456789"), msg)
elsewhere = svc.check_primitive(prim("net.request", API, method="POST",
                                     url="https://discord.com/api/v10/channels/999999999/messages",
                                     json={"content": "x", "nonce": "123456789"}), msg)
R.check(same.verdict == ALLOW, "the transport request of the approved message is covered")
R.check(other.verdict != ALLOW, "approval of one message does not cover DELETE of the channel", other.verdict)
R.check(elsewhere.verdict != ALLOW,
        "approval for channel 123456789 does not cover a post to channel 999999999 that mentions the id in its body",
        elsewhere.verdict)
svc.close()
cleanup(d)

# ---------------------------------------------------------------- D. context-dependent action, context changed (defect 3)
d = fresh_dir("p04d")
clock = C.SimClock()
svc = PolicyService(d, clock=clock, fsync=False)
t = live_task(oid="obj-appr3")
authorize_live(svc, d, t, clock, owned=["https://owned.example/*", "https://shop.example/*"], preauthorized=[])
click = act("browser.click", MB, arguments={"ref": "e5"})
v = svc.authorize(click, t, {"page_url": "https://owned.example/drafts/42"})
preview = [a for a in svc.approvals() if a["approval_id"] == v.approval_id][0]["preview"]
svc.resolve_approval(v.approval_id, True, "operator")           # the operator saw: click e5 on owned.example
v2 = svc.authorize(click, t, {"page_url": "https://shop.example/checkout/confirm"})
R.check(v2.verdict != ALLOW or "shop.example" in str(preview),
        "an approval given for a click on one page is not accepted for the same ref on another site",
        f"preview showed {[e['destination'] for e in preview['effects']]}; second authorize on shop.example -> "
        f"{v2.verdict} {v2.policy_ids}")
svc.close()
cleanup(d)
R.done()
