"""Regression tests of the E-fix workstream for the api modality, publishing, skills and the modality selector: no
automatic re-send of a non-idempotent request after an uncertain failure (M4), stored secrets by reference (M5),
a publication approval shows and binds the exact text (M11), the protected-resource guard on bound skill steps
(m4) and the reach of the selector's block (m5)."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from cam_live.contracts import (ActionResult, ActionSpec, AuthorizationDecision, AuthVerdict, ExecContext,
                                FailureClass, Modality, ResultStatus, WallClock)
from cam_live.harness.failures import RetryPolicy
from cam_live.harness.http_api import HttpApiExecutor, idempotency_key, resolve_secret, secret_name
from cam_live.harness.modality import ExecContextFeatures, ModalitySelector, Realisation, block_keys
from cam_live.harness.publishing import STATIC_SITE, Destination, PublishingExecutor
from cam_live.harness.skills import SkillExecutor, SkillLibrary, Skill, protected_reasons

from es_support import ScopedPolicy, act, make_ctx


class Counting:
    """A local HTTP server; ``mode`` decides the answer to POST /create (applied every time it is received)."""

    def __init__(self):
        self.applied, self.mode, self.headers = [], "503", []
        outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _go(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n) if n else b""
                outer.applied.append((self.command, self.path, body))
                outer.headers.append(dict(self.headers))
                if outer.mode == "slow":
                    time.sleep(1.5)
                status = 503 if outer.mode == "503" else 200
                out = json.dumps({"stored": len(outer.applied)}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = _go

            def log_message(self, *a):
                return

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def srv():
    s = Counting()
    yield s
    s.close()


def fast_retry():
    from cam_live.harness.failures import RetryRule
    return RetryPolicy(rules={FailureClass.TRANSIENT_NETWORK: RetryRule(max_retries=3, base_delay_s=0.01, jitter=0.0)})


# ---- M4 ---------------------------------------------------------------------------------------------------------
def test_a_post_that_may_have_been_applied_is_not_re_sent(srv):
    http = HttpApiExecutor(retry=fast_retry(), timeout_s=0.5)
    ctx = make_ctx(ScopedPolicy())
    for mode in ("503", "slow"):
        srv.mode, srv.applied[:] = mode, []
        res = http.execute(act("http.request", {"method": "POST", "url": srv.url + "/create", "json": {"t": 1}}), ctx)
        assert res.status != ResultStatus.OK and res.retries == 0 and len(srv.applied) == 1, (mode, res.failure_detail)
        assert "not retried automatically" in res.failure_detail and "Idempotency-Key" in res.failure_detail
    time.sleep(1.6)                      # let the slow handler of the last attempt finish
    # an idempotent method keeps its bounded retries, and so does a POST that carries an idempotency key
    srv.mode, srv.applied[:] = "503", []
    res = http.execute(act("http.request", {"method": "PUT", "url": srv.url + "/create", "json": {"t": 1}}), ctx)
    assert res.retries == 3 and len(srv.applied) == 4
    srv.applied[:] = []
    res = http.execute(act("http.request", {"method": "POST", "url": srv.url + "/create", "json": {"t": 1},
                                            "headers": {"Idempotency-Key": "k-1"}}), ctx)
    assert res.retries == 3 and len(srv.applied) == 4 and srv.headers[-1].get("Idempotency-Key") == "k-1"
    assert idempotency_key({"x-idempotency-key": " abc "}) == "abc" and idempotency_key({}) == ""
    # a request that never left (nothing listens) may be repeated: connecting failed before anything was sent
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    res = http.execute(act("http.request", {"method": "POST", "url": f"http://127.0.0.1:{port}/x", "json": {}}), ctx)
    assert res.retries == 3 and "not retried automatically" not in res.failure_detail
    http.close()


# ---- M5 ---------------------------------------------------------------------------------------------------------
class VaultLike:
    """Shaped like the authorization service's handle: knows names without the 'secret:' prefix."""

    def __init__(self):
        self.asked = []

    def get(self, name, action, decision, destination):
        self.asked.append(name)
        if name != "api_key":
            raise PermissionError(f"no secret named {name!r}")
        return "v4lue-of-the-api-key"


def test_secret_references_resolve_against_the_vault_name(srv):
    v = VaultLike()
    assert secret_name("secret:api_key") == "api_key" and secret_name("api_key") == "api_key"
    assert resolve_secret(v, "secret:api_key", action=None, decision=None, destination="x") == "v4lue-of-the-api-key"
    assert v.asked == ["api_key"]
    srv.mode = "ok"
    http = HttpApiExecutor(retry=RetryPolicy(rules={}))
    res = http.execute(act("http.get", {"url": srv.url + "/echo", "auth": {"secret": "secret:api_key"}}),
                       make_ctx(ScopedPolicy(), secrets=v))
    assert res.status == ResultStatus.OK and srv.headers[-1]["Authorization"] == "Bearer v4lue-of-the-api-key"
    assert "v4lue-of-the-api-key" not in json.dumps(res.to_dict())
    http.close()


# ---- M11 ----------------------------------------------------------------------------------------------------------
def pub_ctx(policy, decision=None):
    ctx = make_ctx(policy)
    ctx.authorization = decision
    return ctx


def test_a_publication_approval_binds_and_shows_the_exact_text(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    pub = PublishingExecutor(str(tmp_path / "state"), [Destination("site", STATIC_SITE, root=str(site))])
    pol = ScopedPolicy(scopes=("publish.draft", "publish.write", "publish.read"))
    ctx = pub_ctx(pol)
    d = pub.execute(act("publish.draft", {"destination": "site", "path": "p.html", "content": "APPROVED TEXT A"},
                        scope="publish.draft"), ctx).artifacts["draft_id"]
    pv = pub.execute(act("publish.preview", {"draft_id": d}, scope="publish.draft"), ctx).observations[0]
    sub_args = pv.data["submit_arguments"]
    assert sub_args["content"] == "APPROVED TEXT A" and sub_args["content_sha256"] == pv.data["content_sha256"]
    submit = act("publish.submit", sub_args, scope="publish.write")
    # the operator approved exactly this action (its arguments carry the text the approval shows); the learner then
    # changes the draft before the approved submission runs
    approved = AuthorizationDecision(AuthVerdict.ALLOW, submit.content_hash(), ["approved by operator"],
                                     approval_id="apr_1", grant="g")
    pub.execute(act("publish.update", {"draft_id": d, "content": "UNSEEN TEXT B"}, scope="publish.draft"), ctx)
    pub.execute(act("publish.preview", {"draft_id": d}, scope="publish.draft"), ctx)
    res = pub.execute(submit, pub_ctx(pol, approved))
    assert res.status == ResultStatus.FAILED and "no longer has the content" in res.failure_detail
    assert not (site / "p.html").exists()
    # an approval of a submission that does not name its text covers no text at all
    bare = act("publish.submit", {"draft_id": d}, scope="publish.write")
    res = pub.execute(bare, pub_ctx(pol, AuthorizationDecision(AuthVerdict.ALLOW, bare.content_hash(),
                                                               approval_id="apr_2", grant="g")))
    assert res.status == ResultStatus.FAILED and "must name the exact content" in res.failure_detail
    assert not (site / "p.html").exists()
    # the new text needs its own submission (and approval), which then publishes exactly that text
    pv2 = pub.execute(act("publish.preview", {"draft_id": d}, scope="publish.draft"), ctx).observations[0]
    again = act("publish.submit", pv2.data["submit_arguments"], scope="publish.write")
    ok = pub.execute(again, pub_ctx(pol, AuthorizationDecision(AuthVerdict.ALLOW, again.content_hash(),
                                                               approval_id="apr_3", grant="g")))
    assert ok.status == ResultStatus.OK and (site / "p.html").read_text() == "UNSEEN TEXT B"
    # without any approval (not required by the policy) a submission by draft id still works
    d2 = pub.execute(act("publish.draft", {"destination": "site", "path": "q.html", "content": "plain"},
                         scope="publish.draft"), ctx).artifacts["draft_id"]
    pub.execute(act("publish.preview", {"draft_id": d2}, scope="publish.draft"), ctx)
    assert pub.execute(act("publish.submit", {"draft_id": d2}, scope="publish.write"), ctx).status == ResultStatus.OK


def test_the_real_authorization_service_shows_the_text_of_a_publication_awaiting_approval(tmp_path):
    from cam_live.contracts import Limits, TaskSpec
    from cam_live.harness.auth import PolicyService
    import os
    site = tmp_path / "site"
    site.mkdir()
    src = open(os.path.join(os.path.dirname(__import__("cam_live.harness.auth", fromlist=["x"]).__file__),
                            "default_policy.yaml")).read()
    custom = tmp_path / "policy.yaml"
    custom.write_text(src.replace("require_approval: unless_preauthorized", "require_approval: always"))
    t = TaskSpec("obj_pub", "publish", mode="staging", resources={"owned": [f"file:{site}"]},
                 limits=Limits(duration_s=3600, money=1, max_decisions=100, max_external_actions=100))
    svc = PolicyService(str(tmp_path / "run"), str(custom), fsync=False)
    svc.register_task(t, "operator")
    svc.register_local_service("127.0.0.1:8", "operator", "the test's static site address")
    pub = PublishingExecutor(str(tmp_path / "state"), [Destination("site", STATIC_SITE, root=str(site),
                                                                   base_url="http://127.0.0.1:8")])
    ctx = make_ctx(svc)
    d = pub.execute(act("publish.draft", {"destination": "site", "path": "p.html", "content": "APPROVED TEXT A"}),
                    ctx).artifacts["draft_id"]
    pv = pub.execute(act("publish.preview", {"draft_id": d}), ctx).observations[0]
    submit = act("publish.submit", pv.data["submit_arguments"])
    dec = svc.authorize(submit, t, {})
    assert dec.verdict == AuthVerdict.NEEDS_APPROVAL, dec.reasons
    shown = json.dumps(svc.approvals(), default=str)
    assert "APPROVED TEXT A" in shown                     # the operator sees the text that would be published


# ---- m4 -------------------------------------------------------------------------------------------------------------
def test_a_protected_resource_named_by_a_run_time_argument_is_refused(tmp_path):
    lib = SkillLibrary(str(tmp_path / "lib"), WallClock())
    calls = []

    def dispatch(step, ctx):
        calls.append(step.arguments)
        return ActionResult(step.action_id, ResultStatus.OK)

    ex = SkillExecutor(lib, [], dispatch=dispatch)
    step = ActionSpec(action_id="s0", operation="http.get", modality=Modality.API,
                      arguments={"url": {"$param": "url"}}, auth_scope="http.read")
    sk = Skill(skill_id="skill_t", name="fetch", description="", steps=[step],
               arg_schema={"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
               status="validated", validation_history=[{"ok": True, "fresh": True, "version": 1}] * 3)
    lib.register(sk)
    assert protected_reasons(step) == []                  # the template itself names nothing protected
    res = ex.execute(act("skill.run", {"skill_id": "skill_t", "arguments": {"url": "http://127.0.0.1:1/x/audit.jsonl"}},
                         modality=Modality.SKILL), make_ctx(ScopedPolicy()))
    assert res.status == ResultStatus.DENIED and "protected" in res.failure_detail and calls == []


# ---- m5 -------------------------------------------------------------------------------------------------------------
def test_a_refusal_blocks_every_realisation_of_the_same_intent():
    ctx = ExecContextFeatures(tool="request")

    def pair(decision):
        api = act("http.request", {"method": "POST", "url": "http://127.0.0.1:9/x"}, target="http://127.0.0.1:9/x",
                  expected="the thing is sent", parent_decision_id=decision)
        brw = act("browser.click", {"role": "button", "name": "Send"}, modality=Modality.STRUCTURED_BROWSER,
                  target="http://127.0.0.1:9/x", expected="the thing is sent", parent_decision_id=decision)
        return api, brw

    denied = lambda a: ActionResult(action_id=a.action_id, status=ResultStatus.DENIED,  # noqa: E731
                                    failure_class=FailureClass.MISSING_PERMISSION)
    sel = ModalitySelector(seed=0)
    api, brw = pair("dec_1")
    c = sel.choose(api, [Realisation("api", api), Realisation("structured_browser", brw)], ctx, rule="greedy")
    sel.update_choice(c, ctx, denied(api))
    api2, brw2 = pair("dec_2")                            # the same effect proposed again under the next decision
    assert sel.choose(brw2, [Realisation("api", api2), Realisation("structured_browser", brw2)], ctx).blocked
    sel2 = ModalitySelector(seed=0)
    api3, brw3 = pair("")
    sel2.update_choice(sel2.choose(api3, [Realisation("api", api3)], ctx, rule="greedy"), ctx, denied(api3))
    assert sel2.choose(brw3, [Realisation("structured_browser", brw3)], ctx).blocked   # other realisation as intent
    sel3 = ModalitySelector(seed=0)
    sel3.update("api", ctx, denied(api3))                 # no intent given: the refused action stays refused
    assert sel3.choose("anything", [Realisation("api", api3)], ctx).blocked
    # an unrelated intent is not blocked, and a real grant lifts the block (also after save / load)
    other = act("http.get", {"url": "http://127.0.0.1:9/y"}, target="http://127.0.0.1:9/y", expected="read it")
    assert not sel.choose(other, [Realisation("api", other)], ctx).blocked
    restored = ModalitySelector(seed=1)
    restored.load_state_dict(json.loads(json.dumps(sel.state_dict())))
    assert restored.choose(brw2, [Realisation("structured_browser", brw2)], ctx).blocked
    restored.authorization_changed(api)
    assert not restored.choose(brw2, [Realisation("structured_browser", brw2)], ctx).blocked
    assert len(block_keys(api)) == 4
