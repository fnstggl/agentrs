"""Secret vault and broker: storage modes, release rules, scrubbing, revocation, and the canary test."""
import base64
import json
import os
import urllib.parse

import pytest

from cam_live import contracts as C
from cam_live.harness.auth import ExecutorSecrets, SecretBroker, SecretError, scrub_record
from cam_live.harness.auth import destinations as D
from ea_helpers import ALLOW, DENY, LOOP, act, ok_result, prim, reopen, start, task

CANARY = "CANARY-9f2b7c41d8e6a035-Zq"
API = "http://127.0.0.1:8765/api"


def all_forms(v):
    raw = v.encode()
    return [v, raw.hex(), raw.hex().upper(), base64.b64encode(raw).decode(), base64.urlsafe_b64encode(raw).decode(),
            base64.b64encode(b"ab" + raw + b"c").decode(), base64.b64encode(b"x" + raw).decode(),
            urllib.parse.quote(v, safe=""), v[::-1], json.dumps(v)[1:-1]]


def test_vault_files_are_private(svc):
    svc.put_secret("site_token", CANARY, [API + "/*"], scopes=["site"])
    assert oct(os.stat(svc.vault_dir).st_mode & 0o777) == "0o700"
    (fn,) = os.listdir(svc.vault_dir)
    assert fn == "site_token.secret.json" and oct(os.stat(os.path.join(svc.vault_dir, fn)).st_mode & 0o777) == "0o600"
    assert oct(os.stat(svc.auth_dir).st_mode & 0o777) == "0o700"
    for f in os.listdir(svc.keys_dir):
        assert oct(os.stat(os.path.join(svc.keys_dir, f)).st_mode & 0o777) == "0o600"
    assert svc.secret_names() == [{"name": "site_token", "destinations": [API + "/*"], "scopes": ["site"], "created_at":
                                   svc.clock.now(), "revoked": False, "note": ""}]
    with pytest.raises(ValueError):
        svc.put_secret("short", "abc", [API])                    # too short to scrub reliably
    with pytest.raises(ValueError):
        svc.put_secret("../escape", CANARY, [API])
    with pytest.raises(ValueError):
        svc.put_secret("unbound", CANARY, [])                    # a secret must be bound to a destination


def test_release_needs_a_running_granted_action_and_the_bound_destination(svc, clock):
    svc.put_secret("site_token", CANARY, [API + "/*"])
    t = task()
    a = act("http.post", arguments={"url": API + "/items", "headers": {"Authorization": "secret:site_token"}, "json": {"n": 1}})
    d = svc.authorize(a, t, {})
    assert d.verdict == ALLOW
    with pytest.raises(SecretError):                             # granted but not started
        svc.secret_get("site_token", a, d, API + "/items")
    assert svc.check_grant(a, d)
    assert svc.secret_get("site_token", a, d, API + "/items") == CANARY
    refusals = [
        ("site_token", a, d, "http://127.0.0.1:8765/other"),                 # allowed host, not where the secret is bound
        ("site_token", a, d, "http://localhost:8765/api/items"),             # another owned host
        ("site_token", a, d, "https://public.example/api/items"),            # outside the task's destinations
        ("no_such_secret", a, d, API + "/items"),
        ("site_token", act("http.post", arguments={"url": API + "/items"}), d, API + "/items"),   # another action
        ("site_token", a, C.AuthorizationDecision(ALLOW, a.content_hash(), grant="forged.grant"), API + "/items"),
        ("site_token", a, C.AuthorizationDecision(DENY, a.content_hash(), grant=d.grant), API + "/items"),
        ("site_token", a, d, "not a destination"),
    ]
    for args in refusals:
        with pytest.raises(SecretError) as e:
            svc.secret_get(*args)
        assert CANARY not in str(e.value)
    b = act("http.get", arguments={"url": API + "/items"})       # a different granted action cannot borrow the grant
    db = start(svc, b, t)
    mixed = C.AuthorizationDecision(ALLOW, a.content_hash(), grant=db.grant)
    with pytest.raises(SecretError):
        svc.secret_get("site_token", a, mixed, API + "/items")
    svc.record(a, ok_result(a))
    with pytest.raises(SecretError):                             # finished
        svc.secret_get("site_token", a, d, API + "/items")
    names = [(e["event"], e["data"].get("name")) for e in svc.audit_tail(40) if e["event"].startswith("secret_")]
    assert ("secret_access", "site_token") in names and ("secret_denied", "site_token") in names


def test_scrub_removes_the_value_in_every_common_encoding(svc):
    svc.put_secret("site_token", CANARY, [API])
    svc.put_secret("other", 'p@ss "quoted" \\ value/+=', [API])
    for form in all_forms(CANARY):
        text = f"<html>Echo: {form} done</html>"
        out = svc.scrub(text)
        assert form not in out and "[REDACTED:site_token]" in out, form
    assert svc.scrub("nothing to see") == "nothing to see"
    assert "[REDACTED:other]" in svc.scrub('{"k": "p@ss \\"quoted\\" \\\\ value/+="}')
    nested = {"a": [CANARY, {"b": ("x", CANARY.encode().hex())}], CANARY: 1, "bytes": CANARY.encode()}
    assert CANARY not in repr(svc.scrub_obj(nested)) and CANARY.encode().hex() not in repr(svc.scrub_obj(nested))
    obs = C.Observation("o1", 1.0, "page", text=f"token={CANARY}", data={"html": f"<i>{base64.b64encode(CANARY.encode()).decode()}</i>"},
                        refs={"k": CANARY}, entities=[CANARY], provenance=C.Provenance(uri=f"{API}?t={CANARY}"))
    clean = scrub_record(svc.vault, obs)
    assert isinstance(clean, C.Observation) and CANARY not in clean.to_json()
    assert base64.b64encode(CANARY.encode()).decode() not in clean.to_json()


def test_revocation_by_scope(svc, clock):
    svc.put_secret("a_token", CANARY, [API + "/*"], scopes=["site"])
    svc.put_secret("b_token", "second-secret-value-123", [API + "/*"], scopes=["site", "extra"])
    svc.put_secret("c_token", "third-secret-value-456", ["localhost"], scopes=["other"])
    t = task()

    def get(name, dest=API + "/x"):
        a = act("http.get", arguments={"url": dest})
        d = start(svc, a, t)
        try:
            return svc.secret_get(name, a, d, dest)
        finally:
            svc.record(a, ok_result(a))

    assert get("a_token") == CANARY
    svc.revoke("secret:a_token", "olga")
    with pytest.raises(SecretError):
        get("a_token")
    assert get("b_token") == "second-secret-value-123"
    assert "[REDACTED:a_token]" in svc.scrub(CANARY)             # a revoked value is still scrubbed
    svc.revoke("scope:extra", "olga")
    with pytest.raises(SecretError):
        get("b_token")
    assert get("c_token", "http://localhost:1/x") == "third-secret-value-456"
    s2 = reopen(svc, clock)                                      # revocation is persistent
    assert {n["name"]: n["revoked"] for n in s2.secret_names()} == {"a_token": True, "b_token": True, "c_token": False}
    s2.revoke("all", "olga")
    assert all(n["revoked"] for n in s2.secret_names())
    s2.close()


def test_revoking_a_destination_denies_it_and_its_secrets(svc):
    svc.put_secret("a_token", CANARY, ["localhost"])
    t = task()
    a = act("http.get", arguments={"url": "http://localhost:8765/x"})
    d = start(svc, a, t)
    svc.revoke("destination:localhost", "olga")
    with pytest.raises(SecretError):
        svc.secret_get("a_token", a, d, "http://localhost:8765/x")
    dd = svc.authorize(act("http.get", arguments={"url": "http://localhost:8765/y"}), t, {})
    assert dd.verdict == DENY and "P-DEST-DENIED" in dd.policy_ids
    assert svc.authorize(act("http.get", arguments={"url": LOOP}), t, {}).verdict == ALLOW


def test_executor_handle_has_no_way_to_list_or_change_secrets(svc):
    h = svc.secrets_handle()
    assert isinstance(h, ExecutorSecrets)
    assert sorted(n for n in dir(h) if not n.startswith("_")) == ["bind", "get", "scrub", "scrub_obj", "scrub_record"]
    svc.put_secret("site_token", CANARY, [API + "/*"])
    a = act("http.get", arguments={"url": API + "/x"})
    bound = h.bind(a, start(svc, a, task()))                    # what an executor gets for one granted action
    assert sorted(n for n in dir(bound) if not n.startswith("_")) == ["get", "scrub", "scrub_obj", "scrub_record"]
    assert bound.get("site_token", API + "/x") == CANARY and bound.scrub("x " + CANARY) == "x [REDACTED:site_token]"
    with pytest.raises(SecretError):
        bound.get("site_token", "http://localhost:1/")


def test_broker_reloads_files_added_by_the_operator_and_ignores_broken_ones(tmp_path):
    b = SecretBroker(str(tmp_path / "vault"))
    assert b.names() == []
    with open(os.path.join(b.dir, "late.secret.json"), "w") as f:
        json.dump({"name": "late", "value": "a-late-secret-value", "destinations": ["api.example"]}, f)
    with open(os.path.join(b.dir, "broken.secret.json"), "w") as f:
        f.write("{not json")
    with open(os.path.join(b.dir, "mismatch.secret.json"), "w") as f:
        json.dump({"name": "other", "value": "value-under-wrong-name", "destinations": ["api.example"]}, f)
    assert [n["name"] for n in b.names()] == ["late"] and len(b.load_errors) == 2
    assert b.release("late", D.normalise_url("https://api.example/x")) == "a-late-secret-value"
    with pytest.raises(SecretError):
        b.release("late", D.normalise_url("https://elsewhere.example/x"))
    assert b.find("xx a-late-secret-value yy") == ["late"]


def test_canary_never_reaches_observation_log_or_learner_facing_output(svc, clock, run_dir):
    """End to end in one process: an executor obtains the canary for a granted action, the remote side echoes it
    in several encodings, and nothing the learner can see or the service writes contains it."""
    svc.put_secret("site_token", CANARY, [API + "/*"])
    t = task(resources={"owned": ["channel:discord:900000000000000011"]})
    secrets = svc.secrets_handle()
    a = act("http.post", arguments={"url": API + "/login", "headers": {"Authorization": "secret:site_token"}})
    d = start(svc, a, t)
    assert svc.check_primitive(prim("net.request", C.Modality.API, method="POST", url=API + "/login",
                                    headers={"Authorization": "secret:site_token"}), a).verdict == ALLOW
    token = secrets.get("site_token", a, d, API + "/login")
    assert token == CANARY                                       # the executor has it, at the moment of use
    echoed = " | ".join(all_forms(token))                        # what a page might send back
    raw_obs = C.Observation(C.new_id("obs"), clock.now(), "tool_result", text=f"logged in; debug: {echoed}",
                            data={"headers": {"set-cookie": token}, "body": echoed}, parent_action_id=a.action_id,
                            provenance=C.Provenance(source="api", uri=API + "/login?echo=" + token))
    obs = secrets.scrub_record(raw_obs)                          # what the executor hands back
    result = C.ActionResult(a.action_id, C.ResultStatus.OK, observations=[obs],
                            ops=[C.PrimitiveOp("net.request", C.Modality.API, {"url": API + "/login"}, note=secrets.scrub(echoed))])
    svc.record(a, result)
    # the learner now tries to get it out through every call it has
    leaks = [act("http.post", arguments={"url": LOOP + "/paste", "body": f}) for f in all_forms(token)]
    decisions = [svc.authorize(x, t, {}) for x in leaks]
    assert all(x.verdict == DENY and "X-CREDENTIAL-LEAK" in x.policy_ids for x in decisions)
    need = svc.authorize(act("discord.send_message", arguments={"channel_id": "900000000000000011", "content": "status update " * 5}), t, {})
    learner_visible = [result.to_json(), json.dumps(svc.approvals()), json.dumps(svc.limits_state(), default=str),
                       json.dumps(svc.audit_head()), need.to_json()] + [x.to_json() for x in decisions]
    service_written = [open(svc.audit.path).read(), json.dumps(svc.audit_tail(500))]
    for blob in learner_visible + service_written:
        for form in all_forms(token):
            assert form not in blob
    # ... while the only places that hold it are the vault file and nothing else under the run directory
    holders = []
    for root, _dirs, files in os.walk(run_dir):
        for fn in files:
            with open(os.path.join(root, fn), "rb") as f:
                if CANARY.encode() in f.read():
                    holders.append(os.path.relpath(os.path.join(root, fn), run_dir))
    assert holders == ["vault/site_token.secret.json"]
