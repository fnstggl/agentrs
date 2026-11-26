"""Grants: HMAC over (action hash, expiry, nonce, task); binding, expiry, single use, revocation."""
import base64
import json

import pytest

from cam_live import contracts as C
from cam_live.harness import rpc
from cam_live.harness.auth import PolicyService
from cam_live.harness.auth.grants import GrantError, GrantSigner, OfflineGrantVerifier
from ea_helpers import ALLOW, DENY, LOOP, act, reopen, start, task


def nav(path="/a"):
    return act("browser.navigate", arguments={"url": LOOP + path})


def test_valid_grant_is_accepted_once(svc):
    a, t = nav(), task()
    d = svc.authorize(a, t, {})
    assert d.verdict == ALLOW and d.grant and d.action_hash == a.content_hash() and d.expires_at > svc.clock.now()
    assert svc.check_grant(a, d, consume=False) is True          # looking does not use it up
    assert svc.check_grant(a, d) is True
    assert svc.check_grant(a, d) is False                        # replay
    assert svc.check_grant(a, d, consume=False) is False
    ev = [e["event"] for e in svc.audit_tail(10)]
    assert "grant_use" in ev and "grant_reject" in ev


def test_grant_is_bound_to_the_exact_action(svc):
    a, t = nav(), task()
    d = svc.authorize(a, t, {})
    edited = C.ActionSpec.from_dict(a.to_dict())
    edited.arguments["url"] = LOOP + "/other"
    assert svc.check_grant(edited, d) is False                   # decision hash no longer matches
    d2 = C.AuthorizationDecision.from_dict(d.to_dict())
    d2.action_hash = edited.content_hash()
    assert svc.check_grant(edited, d2) is False                  # ... and the signature covers the hash
    cosmetic = C.ActionSpec.from_dict(a.to_dict())
    cosmetic.rationale = "same thing, different note"
    assert svc.check_grant(cosmetic, d) is False
    other = nav("/b")
    d_other = svc.authorize(other, t, {})
    swapped = C.AuthorizationDecision.from_dict(d_other.to_dict())
    swapped.grant = d.grant                                      # a real grant, for another action
    assert svc.check_grant(other, swapped) is False
    assert svc.check_grant(a, d) is True                         # the genuine pair still works


def test_forged_grants_are_rejected(svc, tmp_path, clock):
    a, t = nav(), task()
    d = svc.authorize(a, t, {})
    body, sig = d.grant.split(".")
    payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    payload["e"] += 10_000                                       # extend the expiry, keep the old signature
    forged_body = base64.urlsafe_b64encode(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).decode().rstrip("=")
    wrong_key = GrantSigner(b"k" * 32).issue(a.content_hash(), t.objective_id, t.content_hash(), clock.now(), clock.now() + 60, 0)
    other_service = PolicyService(str(tmp_path / "other"), clock=clock, fsync=False)
    foreign = other_service.authorize(a, t, {})                  # a perfectly valid grant of a different service
    for g in (forged_body + "." + sig, body + "." + sig[:-2] + "AA", wrong_key["token"], foreign.grant, "", "x.y", "a" * 5000,
              body, None, 17):
        fake = C.AuthorizationDecision(verdict=ALLOW, action_hash=a.content_hash(), grant=g)
        assert svc.check_grant(a, fake) is False, g
    assert svc.check_grant(a, C.AuthorizationDecision(verdict=DENY, action_hash=a.content_hash(), grant=d.grant)) is False
    assert svc.check_grant(a, "not a decision") is False
    assert svc.check_grant(a, d) is True
    other_service.close()


def test_expired_grant_is_rejected(svc, clock):
    a = nav()
    d = svc.authorize(a, task(), {})
    clock.advance(svc.policy.grants["ttl_s"] + 1)
    assert svc.check_grant(a, d) is False
    assert any("expired" in e["data"].get("reason", "") for e in svc.audit_tail(3))


def test_grants_survive_restart_and_stay_single_use(svc, clock):
    a, t = nav(), task()
    d1, d2 = svc.authorize(a, t, {}), svc.authorize(a, t, {})
    assert d1.grant != d2.grant                                  # a fresh nonce every time
    assert svc.check_grant(a, d1) is True
    s2 = reopen(svc, clock)
    assert s2.check_grant(a, d1) is False                        # still used after a restart
    assert s2.check_grant(a, d2) is True                         # the unused one is still good
    s2.close()


def test_revocation_and_stop_void_outstanding_grants(svc):
    a, t = nav(), task()
    d = svc.authorize(a, t, {})
    svc.revoke("grants", "operator")
    assert svc.check_grant(a, d) is False
    d = svc.authorize(a, t, {})
    svc.revoke(f"task:{t.objective_id}", "operator")
    assert svc.check_grant(a, d) is False
    assert svc.authorize(a, t, {}).policy_ids == ["P-REVOKED"]
    svc.register_task(t, "operator")                             # only the operator restores access
    d = svc.authorize(a, t, {})
    svc.emergency_stop("test", "operator")
    svc.reset_stop("operator")
    assert svc.check_grant(a, d) is False                        # a grant from before the stop stays void
    with pytest.raises(ValueError):
        svc.revoke("everything-ish", "operator")


def test_grant_requires_the_pinned_task(svc):
    a, t = nav(), task()
    d = svc.authorize(a, t, {})
    svc.register_task(task(objective="replaced by the operator"), "operator")
    assert svc.check_grant(a, d) is False


def test_primitives_need_a_grant_in_use(svc, clock):
    a, t = nav(), task()
    op = C.PrimitiveOp("browser.navigate", C.Modality.STRUCTURED_BROWSER, {"url": LOOP + "/a"})
    assert svc.check_primitive(op, a).policy_ids == ["P-NO-ACTIVE-GRANT"]          # never authorized
    d = svc.authorize(a, t, {})
    assert svc.check_primitive(op, a).policy_ids == ["P-NO-ACTIVE-GRANT"]          # authorized but not started
    assert svc.check_grant(a, d)
    assert svc.check_primitive(op, a).verdict == ALLOW
    clock.advance(a.timeout_s + svc.policy.grants["exec_slack_s"] + 1)
    assert svc.check_primitive(op, a).policy_ids == ["P-NO-ACTIVE-GRANT"]          # the execution window closed
    b = nav("/b")
    start(svc, b, t)
    svc.record(b, C.ActionResult(b.action_id, C.ResultStatus.OK))
    assert svc.check_primitive(C.PrimitiveOp("browser.navigate", C.Modality.STRUCTURED_BROWSER, {"url": LOOP + "/b"}), b).verdict == DENY


def test_action_hash_survives_the_rpc_wire():
    a = act("skill.run", arguments={"skill_id": "s1", "n": 3, "f": 1.5, "nested": {"k": [1, "two", None, True]}},
            est_cost=C.Cost(seconds=2.0, money=0.25, primitive_ops=4), timeout_s=45.0,
            verify=C.VerificationCondition("text shows", "text_present", {"text": "ok"}),
            steps=[act("browser.navigate", arguments={"url": LOOP}), act("wait", arguments={"seconds": 60})])
    t = task(resources={"owned": ["channel:staging:1"]}, limits=C.Limits(money=5.0, max_decisions=3),
             data_permissions={"aggregate_metrics": True})
    wire = json.loads(json.dumps(rpc.to_wire([a, t])))
    a2, t2 = rpc.from_wire(wire)
    assert isinstance(a2, C.ActionSpec) and isinstance(a2.steps[0], C.ActionSpec) and isinstance(t2.limits, C.Limits)
    assert a2.content_hash() == a.content_hash() and t2.content_hash() == t.content_hash()
    assert C.ActionSpec.from_dict(a.to_dict()).content_hash() == a.content_hash()


def test_offline_verifier_checks_binding_and_expiry_but_cannot_issue(svc, clock):
    import os
    a = nav()
    d = svc.authorize(a, task(), {})
    v = OfflineGrantVerifier.from_file(os.path.join(svc.keys_dir, "grant.key"))
    assert v.verify(d.grant, a.content_hash(), clock.now())["h"] == a.content_hash()
    with pytest.raises(GrantError):
        v.verify(d.grant, nav("/z").content_hash(), clock.now())
    with pytest.raises(GrantError):
        v.verify(d.grant, a.content_hash(), clock.now() + 10_000)
    with pytest.raises(GrantError):
        v.issue(a.content_hash(), "t", "h", 0.0, 1.0, 0)
    assert oct(os.stat(os.path.join(svc.keys_dir, "grant.key")).st_mode & 0o777) == "0o600"
