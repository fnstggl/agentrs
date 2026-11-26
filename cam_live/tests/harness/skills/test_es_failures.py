"""Failure taxonomy, bounded retries, failure observations (spec 12A.12; requirement CU-32)."""
import errno

import httpx
import pytest

from cam_live.contracts import (ActionResult, AuthorizationDecision, AuthVerdict, FailureClass, Level, Modality,
                                ResultStatus, SimClock)
from cam_live.harness.failures import (CATEGORY, FailureEvidence, RetryPolicy, RetryRule, authorization_result,
                                       classify, failed_result, failure_observation, failure_of, run_with_retries)
from cam_live.harness.http_api import HttpApiExecutor
from es_support import ScopedPolicy, act, make_ctx

F = FailureClass
CASES = [
    # (evidence, expected class, expected category)
    (FailureEvidence(arg_errors=["arguments.url: no host"]), F.INVALID_ARGUMENTS, "technical"),
    (FailureEvidence(http_status=422), F.INVALID_ARGUMENTS, "technical"),
    (FailureEvidence(exception=ValueError("bad value")), F.INVALID_ARGUMENTS, "technical"),
    (FailureEvidence(stale_ref=True), F.STALE_OBSERVATION, "technical"),
    (FailureEvidence(obs_id_expected="obs_1", obs_id_current="obs_2", target_missing=True), F.STALE_OBSERVATION, "technical"),
    (FailureEvidence(modality=Modality.VISUAL, grounding_hit=False, grounding_confidence=0.9), F.VISUAL_GROUNDING, "technical"),
    (FailureEvidence(modality=Modality.VISUAL, verified=False, grounding_confidence=0.2), F.VISUAL_GROUNDING, "technical"),
    (FailureEvidence(structure_hash_expected="aaaa", structure_hash_actual="bbbb", target_missing=True), F.PAGE_CHANGED, "technical"),
    (FailureEvidence(structure_hash_expected="aaaa", structure_hash_actual="bbbb", verified=False, executed_ok=True),
     F.PAGE_CHANGED, "technical"),
    (FailureEvidence(timed_out=True), F.TRANSIENT_NETWORK, "technical"),
    (FailureEvidence(exception=httpx.ConnectError("refused")), F.TRANSIENT_NETWORK, "technical"),
    (FailureEvidence(exception=OSError(errno.ECONNRESET, "reset")), F.TRANSIENT_NETWORK, "technical"),
    (FailureEvidence(http_status=503), F.TRANSIENT_NETWORK, "technical"),
    (FailureEvidence(http_status=404), F.WRONG_SITE_ASSUMPTION, "informational"),
    (FailureEvidence(target_missing=True), F.WRONG_SITE_ASSUMPTION, "informational"),
    (FailureEvidence(verified=False, executed_ok=True, modality=Modality.API), F.WRONG_SITE_ASSUMPTION, "informational"),
    (FailureEvidence(http_status=403), F.MISSING_PERMISSION, "permission"),
    (FailureEvidence(http_status=401), F.MISSING_PERMISSION, "permission"),
    (FailureEvidence(auth_verdict=AuthVerdict.DENY, auth_reasons=["scope not granted"]), F.MISSING_PERMISSION, "permission"),
    (FailureEvidence(exception=PermissionError("no token")), F.MISSING_PERMISSION, "permission"),
    (FailureEvidence(auth_verdict=AuthVerdict.DENY, platform_rule=True), F.PLATFORM_RESTRICTION, "permission"),
    (FailureEvidence(http_status=451), F.PLATFORM_RESTRICTION, "permission"),
    (FailureEvidence(http_status=403, platform_rule=True), F.PLATFORM_RESTRICTION, "permission"),
    (FailureEvidence(http_status=429, retry_after_s=2.0), F.RESOURCE_LIMIT, "resource"),
    (FailureEvidence(limit_exceeded=True), F.RESOURCE_LIMIT, "resource"),
    (FailureEvidence(exception=MemoryError()), F.RESOURCE_LIMIT, "resource"),
    (FailureEvidence(outcome_mismatch=True, executed_ok=True, verified=True, level=Level.INTERVENTION), F.STRATEGIC, "strategic"),
    (FailureEvidence(), F.UNKNOWN, "informational"),
]


@pytest.mark.parametrize("evidence,expected,category", CASES, ids=[f"{i}-{c[1]}" for i, c in enumerate(CASES)])
def test_classification_from_observable_evidence(evidence, expected, category):
    rep = classify(evidence)
    assert rep.failure_class == expected
    assert rep.category == category == CATEGORY[expected]
    assert rep.hard == (expected in F.HARD)
    assert abs(sum(rep.candidates.values()) - 1.0) < 1e-5 and rep.candidates[expected] == max(rep.candidates.values())


def test_every_failure_class_is_reachable_and_categorised():
    reached = {classify(ev).failure_class for ev, _, _ in CASES}
    assert reached == set(F.ALL) - {F.NONE}
    assert set(CATEGORY) == set(F.ALL)


def test_denial_dominates_any_other_evidence():
    # a refusal together with evidence that would otherwise be transient stays a refusal and is not retryable
    rep = classify(FailureEvidence(auth_verdict=AuthVerdict.DENY, timed_out=True, http_status=503, stale_ref=True))
    assert rep.failure_class == F.MISSING_PERMISSION and rep.hard and not rep.retryable
    rep = classify(FailureEvidence(http_status=403, timed_out=True))
    assert rep.failure_class == F.MISSING_PERMISSION and not rep.retryable
    # ambiguity is reported, not hidden
    rep = classify(FailureEvidence(http_status=404, structure_hash_expected="a", structure_hash_actual="b"))
    assert set(rep.candidates) == {F.PAGE_CHANGED, F.WRONG_SITE_ASSUMPTION}


def test_retry_policy_bounds_and_backoff():
    pol = RetryPolicy(seed=1)
    delays = []
    for n in range(10):
        d = pol.decide(F.TRANSIENT_NETWORK, n)
        if not d.retry:
            break
        delays.append(d.delay_s)
    assert len(delays) == 3                                   # bounded
    assert delays[0] < delays[1] < delays[2]                  # exponential backoff
    assert 0.5 <= delays[0] <= 0.55 and 2.0 <= delays[2] <= 2.2
    assert RetryPolicy(seed=1).decide(F.TRANSIENT_NETWORK, 0).delay_s == delays[0]   # seeded jitter
    # a limit is retried only when the other side stated how long to wait, and only within the bound
    assert not pol.decide(F.RESOURCE_LIMIT, 0).retry
    assert pol.decide(F.RESOURCE_LIMIT, 0, retry_after_s=1.5).delay_s == 1.5
    assert not pol.decide(F.RESOURCE_LIMIT, 0, retry_after_s=3600.0).retry
    assert not pol.decide(F.RESOURCE_LIMIT, 2, retry_after_s=1.0).retry
    assert pol.decide(F.STALE_OBSERVATION, 0).retry and not pol.decide(F.STALE_OBSERVATION, 1).retry
    # total waiting is bounded too
    assert not RetryPolicy(max_total_wait_s=1.0).decide(F.TRANSIENT_NETWORK, 0, waited_s=0.9).retry
    # nothing else is retried: what to do next is the learner's decision
    for cls in (F.INVALID_ARGUMENTS, F.PAGE_CHANGED, F.WRONG_SITE_ASSUMPTION, F.VISUAL_GROUNDING, F.STRATEGIC, F.UNKNOWN):
        assert not pol.decide(cls, 0).retry


def test_hard_classes_are_never_retried_and_cannot_be_configured():
    pol = RetryPolicy()
    for cls in F.HARD:
        assert not pol.decide(cls, 0).retry and not pol.decide(cls, 0, retry_after_s=0.0).retry
        with pytest.raises(ValueError):
            RetryPolicy({cls: RetryRule(max_retries=1)})
    assert not pol.decide(F.TRANSIENT_NETWORK, 0, hard_evidence=True).retry
    with pytest.raises(ValueError):
        RetryPolicy({F.TRANSIENT_NETWORK: RetryRule(max_retries=RetryPolicy.MAX_RETRIES_CEILING + 1)})
    with pytest.raises(ValueError):
        RetryPolicy({F.STRATEGIC: RetryRule(max_retries=1)})


def _result(action, clock, cls_evidence, status=""):
    return failed_result(action, classify(cls_evidence), clock, clock.now(), status=status)


def test_run_with_retries_counts_attempts_and_waits_on_the_clock():
    clock, action = SimClock(), act("http.get", {"url": "http://example.invalid/x"})
    calls = []

    def attempt(n):
        calls.append((n, clock.now()))
        if len(calls) < 3:
            return _result(action, clock, FailureEvidence(http_status=503))
        return ActionResult(action_id=action.action_id, status=ResultStatus.OK)

    res = run_with_retries(attempt, RetryPolicy(seed=0), clock)
    assert res.status == ResultStatus.OK and res.retries == 2 and [c[0] for c in calls] == [0, 1, 2]
    assert calls[1][1] - calls[0][1] >= 0.5 and calls[2][1] - calls[1][1] >= 1.0      # backoff happened on the clock

    calls.clear()

    def always_down(n):
        calls.append(n)
        return _result(action, clock, FailureEvidence(timed_out=True))

    res = run_with_retries(always_down, RetryPolicy(seed=0), clock)
    assert res.status == ResultStatus.TIMEOUT and res.failure_class == F.TRANSIENT_NETWORK
    assert len(calls) == 4 and res.retries == 3                                        # 1 attempt + 3 retries, no more


@pytest.mark.parametrize("evidence", [FailureEvidence(http_status=403), FailureEvidence(http_status=451),
                                      FailureEvidence(auth_verdict=AuthVerdict.DENY),
                                      FailureEvidence(http_status=404), FailureEvidence(arg_errors=["x"])])
def test_run_with_retries_does_not_repeat_non_transient_failures(evidence):
    clock, action, calls = SimClock(), act("http.get", {"url": "http://example.invalid/x"}), []

    def attempt(n):
        calls.append(n)
        return _result(action, clock, evidence)

    t0 = clock.now()
    res = run_with_retries(attempt, RetryPolicy(), clock)
    assert calls == [0] and res.retries == 0 and clock.now() == t0


def test_authorization_result_maps_verdicts_without_retry():
    clock, action = SimClock(), act("http.request", {"url": "http://example.invalid/x", "method": "POST"})
    assert authorization_result(action, AuthorizationDecision(AuthVerdict.ALLOW), clock, clock.now()) is None
    denied = authorization_result(action, AuthorizationDecision(AuthVerdict.DENY, reasons=["destination not allowed"]),
                                  clock, clock.now())
    assert denied.status == ResultStatus.DENIED and denied.failure_class == F.MISSING_PERMISSION
    assert "destination not allowed" in denied.failure_detail
    platform = authorization_result(action, AuthorizationDecision(AuthVerdict.DENY, reasons=["rule"],
                                                                  policy_ids=["platform.rule"]), clock, clock.now())
    assert platform.failure_class == F.PLATFORM_RESTRICTION
    waiting = authorization_result(action, AuthorizationDecision(AuthVerdict.NEEDS_APPROVAL, approval_id="apr_1"),
                                   clock, clock.now())
    assert waiting.status == ResultStatus.NEEDS_APPROVAL and waiting.artifacts["approval_id"] == "apr_1"
    for res in (denied, platform, waiting):
        assert not RetryPolicy().decide(res.failure_class, 0).retry
        assert run_with_retries(lambda n, r=res: r, RetryPolicy(), clock).retries == 0


def test_failure_observation_is_structured_for_the_learner():
    clock, action = SimClock(), act("http.get", {"url": "http://example.invalid/x"})
    rep = classify(FailureEvidence(http_status=429, retry_after_s=4.0, limit_exceeded=True, detail="slow down"))
    obs = failure_observation(rep, action, clock, retries=2, elapsed_s=1.25)
    assert obs.kind == "error" and obs.parent_action_id == action.action_id and obs.provenance.trust == "untrusted"
    q = {x.name: x.value for x in obs.quantities}
    assert q == {"exec.failure.retries": 2.0, "exec.failure.elapsed_s": 1.25, "exec.failure.http_status": 429.0,
                 "exec.failure.retry_after_s": 4.0}
    f = obs.data["failure"]
    assert f["failure_class"] == F.RESOURCE_LIMIT and f["category"] == "resource" and not f["hard"] and f["retryable"]
    assert "http_429" in f["signals"] and obs.data["operation"] == "http.get"
    assert not any(k in obs.to_dict() for k in ("reward", "bonus"))            # an observation, never a reward
    res = failed_result(action, rep, clock, clock.now(), retries=2)
    assert failure_of(res)["retry_after_s"] == 4.0 and res.failure_class == F.RESOURCE_LIMIT


def test_real_transient_failure_is_retried_and_a_real_refusal_is_not(notes):
    http, pol = HttpApiExecutor(sleep=lambda s: None), ScopedPolicy()
    notes.flaky_left = 2
    res = http.execute(act("http.get", {"url": notes.url + "/api/flaky"}), make_ctx(pol))
    assert res.status == ResultStatus.OK and res.retries == 2 and len(res.ops) == 3
    assert sum(1 for r in notes.requests if r["path"] == "/api/flaky") == 3
    notes.flaky_left = 99
    res = http.execute(act("http.get", {"url": notes.url + "/api/flaky"}), make_ctx(pol))
    assert res.status == ResultStatus.FAILED and res.failure_class == F.TRANSIENT_NETWORK and res.retries == 3
    for path, cls in (("/api/forbidden", F.MISSING_PERMISSION), ("/api/legal", F.PLATFORM_RESTRICTION)):
        before = len(notes.requests)
        res = http.execute(act("http.get", {"url": notes.url + path}), make_ctx(pol))
        assert res.failure_class == cls and res.retries == 0 and len(notes.requests) == before + 1
    http.close()
