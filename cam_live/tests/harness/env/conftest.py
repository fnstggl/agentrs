"""Fixtures for the computer-environment tests (workstream EB).

Nothing here stands in for the code under test: the environment, browser, perception, RPC service and staging
sites are the real ones. ``GrantPolicy`` is a small real ``PolicyEngine`` (HMAC grants bound to the action hash, a
destination deny list, an approval queue) and ``HttpExecutor`` a small real ``ModalityExecutor`` doing plain HTTP
against the staging sites, so the environment core can be tested without Chromium.
"""
import hashlib
import hmac
import json
import os
import sys
import threading
import urllib.parse
import urllib.request

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from cam_live.contracts import (ActionResult, ActionSpec, AuthorizationDecision, AuthVerdict, Cost, FailureClass,  # noqa: E402
                                Level, Modality, OutcomeKind, OutcomeQuery, PrimitiveOp, Quantity, ResultStatus,
                                SimClock, TaskSpec, ToolDescriptor, VerificationCondition, new_id)
from cam_live.harness.perception import Perception  # noqa: E402
from cam_live.harness.staging import StagingSites  # noqa: E402


def pytest_collection_modifyitems(config, items):
    """browser / privileged / network tests run only when selected with -m (default suite: CPU, fast, no Chromium)."""
    expr = config.getoption("-m") or ""
    for item in items:
        for mark in ("browser", "privileged", "network"):
            if mark in item.keywords and mark not in expr:
                item.add_marker(pytest.mark.skip(reason=f"run with -m {mark}"))


class GrantPolicy:
    """A real, small ``contracts.PolicyEngine`` for tests."""

    def __init__(self, deny_urls=(), approve_ops=(), key=b"eb-test-key"):
        self.deny_urls, self.approve_ops, self._key = list(deny_urls), set(approve_ops), key
        self.primitives, self.records, self._queue, self._approved = [], [], {}, set()
        self._stopped, self._lock = False, threading.Lock()

    def _grant(self, action):
        return hmac.new(self._key, action.content_hash().encode(), hashlib.sha256).hexdigest()

    def _denied_url(self, text):
        return next((u for u in self.deny_urls if u and u in text), None)

    def authorize(self, action, task, context=None):
        h = action.content_hash()
        if self._stopped:
            return AuthorizationDecision(AuthVerdict.DENY, h, ["emergency stop"], ["P-STOP"])
        bad = self._denied_url(json.dumps(action.to_dict()))
        if bad:
            return AuthorizationDecision(AuthVerdict.DENY, h, [f"destination not allowed: {bad}"], ["P-DEST"])
        if action.operation in self.approve_ops and action.action_id not in self._approved:
            aid = new_id("apr")
            self._queue[aid] = action
            return AuthorizationDecision(AuthVerdict.NEEDS_APPROVAL, h, ["operator approval required"], ["P-APPROVE"],
                                         approval_id=aid)
        return AuthorizationDecision(AuthVerdict.ALLOW, h, ["allowed"], ["P-ALLOW"], grant=self._grant(action))

    def check_grant(self, action, decision):
        return (decision.verdict == AuthVerdict.ALLOW and decision.action_hash == action.content_hash()
                and hmac.compare_digest(decision.grant, self._grant(action)))

    def check_primitive(self, op, action):
        with self._lock:
            self.primitives.append((op.op, dict(op.arguments), action.action_id))
        if self._stopped:
            return AuthorizationDecision(AuthVerdict.DENY, reasons=["emergency stop"], policy_ids=["P-STOP"])
        bad = self._denied_url(json.dumps(op.arguments, default=str))
        if bad:
            return AuthorizationDecision(AuthVerdict.DENY, reasons=[f"destination not allowed: {bad}"],
                                         policy_ids=["P-DEST"])
        return AuthorizationDecision(AuthVerdict.ALLOW)

    def record(self, action, result):
        with self._lock:
            self.records.append((action.action_id, result.status))

    def approvals(self):
        return [{"approval_id": k, "action_id": a.action_id, "operation": a.operation} for k, a in self._queue.items()]

    def resolve_approval(self, approval_id, approve, by, note=""):
        action = self._queue.pop(approval_id)
        if not approve:
            return AuthorizationDecision(AuthVerdict.DENY, action.content_hash(), [f"rejected by {by}"])
        self._approved.add(action.action_id)
        return AuthorizationDecision(AuthVerdict.ALLOW, action.content_hash(), [f"approved by {by}"],
                                     grant=self._grant(action))

    def emergency_stop(self, reason, by):
        self._stopped = True

    def stopped(self):
        return self._stopped

    def revoke(self, scope, by):
        self.deny_urls.append(scope)

    def limits_state(self):
        return {"primitive_checks": len(self.primitives)}


class HttpExecutor:
    """A real ``ModalityExecutor`` (modality ``api``): plain HTTP GET/POST with a policy check per request."""
    modality = Modality.API

    def __init__(self, clock=None):
        self.perception = Perception(clock)
        self.calls = []

    def tools(self):
        url = {"type": "object", "properties": {"url": {"type": "string"}, "form": {"type": "object"},
                                                 "obs_id": {"type": "string"}},
               "required": ["url"], "additionalProperties": False}
        return [ToolDescriptor("http.get", Modality.API, "Fetch a URL.", url, Level.PRIMITIVE, "http.read",
                               "external_read", Cost(seconds=0.1)),
                ToolDescriptor("http.post", Modality.API, "Send a form to a URL.", url, Level.PRIMITIVE, "http.write",
                               "external_write", Cost(seconds=0.1))]

    def validate(self, action):
        return [] if str(action.arguments.get("url", "")).startswith("http://127.0.0.1") else ["only localhost"]

    def execute(self, action, ctx):
        now = ctx.clock.now()
        res = ActionResult(action.action_id, ResultStatus.OK, started=now, ended=now)
        op = PrimitiveOp(action.operation, self.modality, {"url": action.arguments["url"]}, started=now, ended=now)
        res.ops.append(op)
        dec = ctx.check(op, action)
        if dec.verdict != AuthVerdict.ALLOW:
            res.status = op.status = ResultStatus.DENIED
            res.failure_class = op.failure_class = FailureClass.MISSING_PERMISSION
            res.failure_detail = "; ".join(dec.reasons)
            return res
        self.calls.append(action.action_id)
        data = None
        if action.operation == "http.post":
            data = urllib.parse.urlencode(action.arguments.get("form", {})).encode()
        try:
            with urllib.request.urlopen(urllib.request.Request(action.arguments["url"], data=data), timeout=30) as r:
                body, ctype = r.read(), r.headers.get("Content-Type", "")
        except Exception as e:  # noqa: BLE001
            res.status, res.failure_class, res.failure_detail = ResultStatus.FAILED, FailureClass.TRANSIENT_NETWORK, repr(e)
            return res
        res.observations.append(self.perception.document(body, uri=action.arguments["url"], media_type=ctype,
                                                         parent_action_id=action.action_id, source="http"))
        res.cost = Cost(primitive_ops=1)
        return res

    def close(self):
        pass


@pytest.fixture(scope="session")
def sites():
    clock = SimClock()
    with StagingSites(clock=clock) as s:
        s.clock = clock
        yield s


@pytest.fixture
def task():
    return TaskSpec(objective_id="obj_eb_test", objective="exercise the computer environment", mode="staging")


def op_modality(op):
    if op.startswith("browser."):
        return Modality.STRUCTURED_BROWSER
    if op.startswith("http."):
        return Modality.API
    return Modality.INTERNAL


def make_action(op, verify=None, steps=None, level=Level.PRIMITIVE, timeout=60.0, aid=None, **args):
    """``timeout`` / ``aid`` are the ActionSpec's timeout_s / action_id; every other keyword is a tool argument."""
    return ActionSpec(action_id=aid or new_id("act"), operation=op, modality=op_modality(op), level=level,
                      arguments=args, timeout_s=timeout, steps=list(steps or []),
                      verify=verify or VerificationCondition())


class Driver:
    """Authorize-then-execute convenience around an environment and a policy (what the runtime loop does)."""

    def __init__(self, env, policy, task):
        self.env, self.policy, self.task = env, policy, task

    def submit(self, action):
        return self.env.execute(action, self.policy.authorize(action, self.task, {}))

    def do(self, op, **kw):
        return self.submit(make_action(op, **kw))

    def ok(self, op, **kw):
        res = self.do(op, **kw)
        assert res.status == ResultStatus.OK, (op, res.status, res.failure_class, res.failure_detail)
        return res

    def inspect(self, **kw):
        return self.ok("browser.inspect", **kw).observations[0]

    @staticmethod
    def ref(obs, role, name):
        hits = [e for e in obs.data["elements"] if e["role"] == role and e["name"] == name]
        assert len(hits) == 1, (role, name, [(e["role"], e["name"]) for e in obs.data["elements"]])
        return hits[0]["ref"]


@pytest.fixture
def driver_cls():
    return Driver
