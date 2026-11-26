"""The api executor against a real local HTTP application (spec 12A.1 "authorized HTTP and API access", 12A.14)."""
import json

from cam_live.contracts import FailureClass, Modality, ModalityExecutor, ResultStatus, VerificationCondition
from cam_live.harness.http_api import (ApiModality, HttpApiExecutor, RateLimiter, parse_rate_limit, schema_errors,
                                       scrub, structure_hash)
from es_support import NotesApp, ScopedPolicy, act, make_ctx


def test_executor_implements_the_contract():
    http = HttpApiExecutor()
    assert isinstance(http, ModalityExecutor) and http.modality == Modality.API
    names = {t.name: t for t in http.tools()}
    assert set(names) == {"http.get", "http.request"}
    assert names["http.get"].side_effects == "external_read" and names["http.request"].side_effects == "external_write"
    assert all(t.arg_schema["type"] == "object" for t in names.values())
    http.close()


def test_every_request_is_checked_with_method_and_url(notes):
    http, pol = HttpApiExecutor(), ScopedPolicy()
    ctx = make_ctx(pol)
    res = http.execute(act("http.request", {"method": "POST", "url": notes.url + "/api/items",
                                            "json": {"title": "one", "body": "b"}}), ctx)
    assert res.status == ResultStatus.OK and res.cost.primitive_ops == 1
    assert len(pol.checks) == 1
    checked = pol.checks[0]
    assert checked.op == "http.request" and checked.arguments["method"] == "POST"
    assert checked.arguments["url"] == notes.url + "/api/items" and checked.arguments["body_sha256"]
    obs = res.observations[0]
    assert obs.provenance.trust == "untrusted" and obs.provenance.content_sha256 and obs.data["status"] == 201
    q = {x.name: x.value for x in obs.quantities}
    assert q["http.status"] == 201.0 and q["http.api.items.id"] == 1.0 and q["http.elapsed_s"] > 0
    http.close()


def test_denied_request_never_reaches_the_server(notes):
    http = HttpApiExecutor()
    pol = ScopedPolicy(deny=lambda op, a: "destination not allowed" if "/api/items" in op.arguments.get("url", "") else "")
    res = http.execute(act("http.request", {"method": "POST", "url": notes.url + "/api/items", "json": {"title": "x"}}),
                       make_ctx(pol))
    assert res.status == ResultStatus.DENIED and res.failure_class == FailureClass.MISSING_PERMISSION
    assert res.retries == 0 and notes.requests == [] and notes.items == []
    # scope not granted
    pol2 = ScopedPolicy(scopes=("http.read",))
    res = http.execute(act("http.request", {"method": "POST", "url": notes.url + "/api/items", "json": {"title": "x"}}),
                       make_ctx(pol2))
    assert res.status == ResultStatus.DENIED and notes.requests == []
    # approval required: nothing is sent either
    pol3 = ScopedPolicy(approval_rule=lambda op, a: op.arguments.get("method") == "POST")
    res = http.execute(act("http.request", {"method": "POST", "url": notes.url + "/api/items", "json": {"title": "x"}}),
                       make_ctx(pol3))
    assert res.status == ResultStatus.NEEDS_APPROVAL and res.artifacts["approval_id"] and notes.requests == []
    http.close()


def test_secret_is_injected_by_reference_and_never_exposed(notes):
    http, pol = HttpApiExecutor(), ScopedPolicy()
    secrets = {"app_token": NotesApp.TOKEN}
    action = act("http.get", {"url": notes.url + "/api/private", "auth": {"secret": "app_token"}})
    res = http.execute(action, make_ctx(pol, secrets=secrets))
    assert res.status == ResultStatus.OK
    assert notes.requests[-1]["headers"]["authorization"] == f"Bearer {NotesApp.TOKEN}"      # the server got it
    blob = json.dumps([res.to_dict(), [c.to_dict() for c in pol.checks], action.to_dict()])
    assert NotesApp.TOKEN not in blob                               # ...and nothing the learner can see contains it
    assert "[secret]" in res.observations[0].text                   # the echoed header was scrubbed
    assert pol.checks[0].arguments["auth_secret"] == "app_token"    # the policy sees the name
    q = {x.name: x.value for x in res.observations[0].quantities}
    assert q["http.api.private.visits"] == 12.0
    # header and query schemes
    for auth in ({"secret": "app_token", "scheme": "header", "name": "X-Service-Key"},
                 {"secret": "app_token", "scheme": "query", "name": "key"}):
        res = http.execute(act("http.get", {"url": notes.url + "/api/keyed", "auth": auth}), make_ctx(pol, secrets=secrets))
        assert res.status == ResultStatus.OK and NotesApp.TOKEN not in json.dumps(res.to_dict())
    # unknown secret name: nothing is sent
    n = len(notes.requests)
    res = http.execute(act("http.get", {"url": notes.url + "/api/private", "auth": {"secret": "nope"}}),
                       make_ctx(pol, secrets=secrets))
    assert res.failure_class == FailureClass.MISSING_PERMISSION and len(notes.requests) == n
    # without the credential the application itself refuses, which is a hard failure and not retried
    res = http.execute(act("http.get", {"url": notes.url + "/api/private"}), make_ctx(pol))
    assert res.failure_class == FailureClass.MISSING_PERMISSION and res.retries == 0
    http.close()


def test_literal_credentials_in_arguments_are_rejected(notes):
    http = HttpApiExecutor()
    bad = [
        {"url": notes.url + "/api/private", "headers": {"Authorization": "Bearer abc"}},
        {"url": notes.url + "/api/private", "headers": {"X-Api-Key": "abc"}},
        {"url": notes.url + "/api/private", "headers": {"Cookie": "session=abc"}},
        {"url": notes.url.replace("http://", "http://user:pw@") + "/api/private"},
        {"url": "ftp://example.invalid/file"},
        {"url": notes.url + "/x", "auth": {"secret": "s", "scheme": "header"}},
        {"url": notes.url + "/x", "json": {}, "body": "x"},
        {"url": notes.url + "/x", "method": "TRACE"},
    ]
    pol = ScopedPolicy()
    for args in bad:
        a = act("http.request", args)
        assert http.validate(a), args
        res = http.execute(a, make_ctx(pol))
        assert res.failure_class == FailureClass.INVALID_ARGUMENTS
    assert notes.requests == [] and pol.checks == []
    http.close()


def test_rate_limit_headers_are_honoured(notes):
    waits = []
    http, pol = HttpApiExecutor(sleep=waits.append), ScopedPolicy()
    notes.limited_left, notes.retry_after = 1, "0.25"
    res = http.execute(act("http.get", {"url": notes.url + "/api/limited"}), make_ctx(pol))
    assert res.status == ResultStatus.OK and res.retries == 1
    assert waits and abs(waits[0] - 0.25) < 1e-6                     # waited exactly what the server stated
    q = {x.name: x.value for x in res.observations[0].quantities}
    assert q["http.ratelimit.remaining"] == 4.0 and q["http.ratelimit.limit"] == 5.0
    # a stated wait beyond the bound is not waited for; it is reported with the exact number
    notes.limited_left, notes.retry_after = 5, "900"
    res = http.execute(act("http.get", {"url": notes.url + "/api/limited"}), make_ctx(pol))
    assert res.failure_class == FailureClass.RESOURCE_LIMIT and res.retries == 0
    fq = {x.name: x.value for o in res.observations for x in o.quantities}
    assert fq["exec.failure.retry_after_s"] == 900.0
    # ...and the next request to that host is held back locally instead of being sent into the limit
    n = len(notes.requests)
    res = http.execute(act("http.get", {"url": notes.url + "/api/items"}), make_ctx(pol))
    assert res.failure_class == FailureClass.RESOURCE_LIMIT and len(notes.requests) == n
    assert "request not sent" in res.ops[-1].note
    http.close()


def test_exhausted_window_is_waited_for_before_the_next_request(notes):
    waits = []
    http, pol = HttpApiExecutor(sleep=waits.append), ScopedPolicy()
    notes.retry_after = "0.2"
    assert http.execute(act("http.get", {"url": notes.url + "/api/exhausting"}), make_ctx(pol)).status == ResultStatus.OK
    assert waits == []
    assert http.execute(act("http.get", {"url": notes.url + "/api/items"}), make_ctx(pol)).status == ResultStatus.OK
    assert len(waits) == 1 and 0.0 < waits[0] <= 0.2
    http.close()


def test_rate_limit_parsing_variants():
    now = 1_800_000_000.0
    i = parse_rate_limit({"X-RateLimit-Limit": "10", "X-RateLimit-Remaining": "0", "X-RateLimit-Reset": str(now + 30)}, now)
    assert (i.limit, i.remaining, round(i.reset_after_s)) == (10.0, 0.0, 30)
    i = parse_rate_limit({"RateLimit-Remaining": "3", "RateLimit-Reset": "12"}, now)
    assert (i.remaining, i.reset_after_s) == (3.0, 12.0)
    i = parse_rate_limit({"Retry-After": "Wed, 15 Jan 2027 08:00:30 GMT"}, 1800000000.0)
    assert i.retry_after_s is not None and i.retry_after_s >= 0
    i = parse_rate_limit({"Retry-After": "65", "X-RateLimit-Global": "true"}, now, {"retry_after": 64.57, "global": True})
    assert i.retry_after_s == 65.0 and i.is_global
    lim = RateLimiter()
    lim.update("a", i, 429, now)
    assert lim.wait_needed("b", now) == 65.0                        # a global limit blocks every key


def test_redirects_are_rechecked_and_credentials_do_not_cross_origins(notes, other_origin):
    http, pol = HttpApiExecutor(), ScopedPolicy()
    res = http.execute(act("http.get", {"url": notes.url + "/api/redirect"}), make_ctx(pol))
    assert res.status == ResultStatus.OK and len(res.ops) == 2 and len(pol.checks) == 2
    assert pol.checks[1].arguments["url"] == notes.url + "/api/items"
    # cross-origin redirect: the hop is checked as its own request and the credential stays behind
    notes.other_origin = other_origin.url
    res = http.execute(act("http.get", {"url": notes.url + "/api/redirect_out", "auth": {"secret": "tok"}}),
                       make_ctx(pol, secrets={"tok": NotesApp.TOKEN}))
    assert res.status == ResultStatus.OK
    assert "authorization" in notes.requests[-1]["headers"]
    assert "authorization" not in other_origin.requests[-1]["headers"]
    assert pol.checks[-1].arguments["auth_secret"] == ""
    # a policy that refuses the second origin stops the redirect; the other origin is never contacted
    n = len(other_origin.requests)
    strict = ScopedPolicy(deny=lambda op, a: "destination not allowed" if other_origin.url in op.arguments["url"] else "")
    res = http.execute(act("http.get", {"url": notes.url + "/api/redirect_out"}), make_ctx(strict))
    assert res.status == ResultStatus.DENIED and len(other_origin.requests) == n
    http.close()


def test_timeouts_and_connection_failures_are_transient_and_bounded(notes):
    http, pol = HttpApiExecutor(sleep=lambda s: None), ScopedPolicy()
    res = http.execute(act("http.get", {"url": notes.url + "/api/slow"}, timeout_s=0.1), make_ctx(pol))
    assert res.status == ResultStatus.TIMEOUT and res.failure_class == FailureClass.TRANSIENT_NETWORK and res.retries == 3
    res = http.execute(act("http.get", {"url": "http://127.0.0.1:9/nothing-listens-here"}), make_ctx(pol))
    assert res.failure_class == FailureClass.TRANSIENT_NETWORK and res.retries == 3 and len(res.ops) == 4
    http.close()


def test_verification_condition_is_evaluated_on_the_response(notes):
    http, pol = HttpApiExecutor(), ScopedPolicy()
    ok = act("http.get", {"url": notes.url + "/api/items"},
             verify=VerificationCondition("no items yet", "metric", {"path": "json/total", "op": "eq", "value": 0}))
    assert http.execute(ok, make_ctx(pol)).verified is True
    wrong = act("http.get", {"url": notes.url + "/api/items"},
                verify=VerificationCondition("three items", "metric", {"path": "json/total", "op": "eq", "value": 3}))
    res = http.execute(wrong, make_ctx(pol))
    assert res.verified is False and res.status == ResultStatus.FAILED
    assert res.failure_class == FailureClass.WRONG_SITE_ASSUMPTION and res.retries == 0
    plain = http.execute(act("http.get", {"url": notes.url + "/api/text"}), make_ctx(pol))
    assert plain.verified is None and plain.observations[0].text == "plain words, 3 of them"
    http.close()


def test_helpers():
    schema = {"type": "object", "required": ["a"], "additionalProperties": False,
              "properties": {"a": {"type": "integer", "minimum": 1}, "b": {"type": "string", "enum": ["x", "y"]},
                             "c": {"type": "array", "items": {"type": "number"}, "maxItems": 2}}}
    assert schema_errors(schema, {"a": 2, "b": "x", "c": [1, 2.5]}) == []
    assert len(schema_errors(schema, {"a": 0, "b": "z", "c": [1, "q", 3], "d": 1})) == 5
    assert schema_errors(schema, {"a": True}) and schema_errors(schema, []) and schema_errors(schema, {})
    assert scrub({"k": ["token-abcdef", "x"], "n": 1}, ["token-abcdef"]) == {"k": ["[secret]", "x"], "n": 1}
    assert structure_hash({"a": 1, "b": [{"c": "x"}]}) == structure_hash({"b": [{"c": "other"}, {"c": "z"}], "a": 99})
    assert structure_hash({"a": 1}) != structure_hash({"item_a": 1})


def test_api_modality_routes_by_operation(notes):
    http = HttpApiExecutor()

    class Extra:
        modality = Modality.API

        def tools(self):
            return []

        def validate(self, action):
            return []

        def execute(self, action, ctx):
            raise AssertionError("not routed here")

        def close(self):
            self.closed = True

    extra = Extra()
    api, pol = ApiModality([extra, http]), ScopedPolicy()
    assert isinstance(api, ModalityExecutor) and {t.name for t in api.tools()} == {"http.get", "http.request"}
    assert api.execute(act("http.get", {"url": notes.url + "/api/items"}), make_ctx(pol)).status == ResultStatus.OK
    assert api.validate(act("nothing.here", {})) and \
        api.execute(act("nothing.here", {}), make_ctx(pol)).failure_class == FailureClass.INVALID_ARGUMENTS
    api.close()
    assert extra.closed
