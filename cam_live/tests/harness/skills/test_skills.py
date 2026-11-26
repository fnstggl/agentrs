"""Acquired executable skills: induction, validation, permission, monitoring, repair (spec 12A.8, 12A.9, 12A.17;
requirements CU-15 .. CU-19). Everything runs against a real local HTTP application through the api executor."""
import json
import os
import time

import pytest
from scipy.stats import beta as scipy_beta

from cam_live.contracts import (ActionSpec, FailureClass, Level, Modality, ModalityExecutor, ResultStatus, WallClock)
from cam_live.harness.http_api import HttpApiExecutor
from cam_live.harness.skills import (STATUS_RETIRED, STATUS_STALE, STATUS_VALIDATED, Skill, SkillBindError,
                                     SkillExecutor, SkillInductionError, SkillLibrary, Trace, beta_cdf,
                                     beta_interval, bind, protected_reasons)
from support import ScopedPolicy, act, create_and_publish, listing_probe, make_ctx

EVIDENCE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "docs", "evidence", "ES"))


def setup(tmp_path, **lib_kw):
    http = HttpApiExecutor(sleep=lambda s: None)
    lib = SkillLibrary(str(tmp_path), WallClock(), **lib_kw)
    return http, lib, SkillExecutor(lib, [http], seed=0)


def trace(http, ctx, url, title, body, generation=1, inputs=None):
    pairs, before, after = create_and_publish(http, ctx, url, title, body, generation)
    return Trace.from_pairs(pairs, inputs=inputs, before=before, after=after)


def learned(tmp_path, notes, pol=None, **lib_kw):
    """A validated create-and-publish skill induced from two traces."""
    http, lib, ex = setup(tmp_path, **lib_kw)
    ctx = make_ctx(pol or ScopedPolicy())
    traces = [trace(http, ctx, notes.url, "alpha note", "first body"), trace(http, ctx, notes.url, "beta memo", "second text")]
    skill, report = ex.induce_validate_register(traces, ctx, probe=listing_probe(notes.url), name="create and publish")
    assert skill is not None, report
    return http, lib, ex, skill


def run(ex, skill_id, args, ctx):
    a = act("skill.run", {"skill_id": skill_id, "arguments": args}, modality=Modality.SKILL)
    a.level = Level.PROCEDURE
    return ex.execute(a, ctx)


# ---- induction ---------------------------------------------------------------------------------------------------
def test_two_traces_yield_a_parameterised_procedure_that_works_with_new_arguments(tmp_path, notes):
    http, lib, ex, skill = learned(tmp_path, notes)
    assert isinstance(ex, ModalityExecutor) and ex.modality == Modality.SKILL
    # what varied between the traces became typed parameters; what the application generated became a capture
    assert skill.arg_schema["required"] == ["body", "title"]
    assert skill.arg_schema["properties"]["title"]["type"] == "string"
    s0, s1 = skill.steps
    assert s0.arguments["json"] == {"title": {"$param": "title"}, "body": {"$param": "body"}}
    assert s0.arguments["method"] == "POST" and s0.arguments["url"] == notes.url + "/api/items"      # constants stay
    assert s1.arguments["url"] == {"$tpl": [notes.url + "/api/items/", {"$capture": {"step": 0, "path": ["json", "id"],
                                                                                    "cast": "str"}}, "/publish"]}
    # pre / postconditions came from the observed before / after states
    kinds = [(c.kind, c.spec.get("op"), c.spec.get("value")) for c in skill.postconditions]
    assert ("metric", "delta_eq", 1.0) in kinds and any(c.kind == "text_present" for c in skill.postconditions)
    assert any(c.kind == "program" and c.spec["check"] == "tools" for c in skill.preconditions)
    # provenance, cost, validation history, reliability, version, failure / recovery conditions
    assert skill.provenance["method"] == "aligned_traces" and len(skill.provenance["trace_ids"]) == 2
    assert len(skill.provenance["action_ids"][0]) == 2 and skill.expected_cost.primitive_ops == 2
    assert skill.version == 1 and skill.status == STATUS_VALIDATED
    assert [h["ok"] for h in skill.validation_history] == [True] and skill.validation_history[0]["fresh"]
    assert {c["when"] for c in skill.failure_conditions} >= {"step_failed", "postcondition_false", "interface_changed"}
    assert {r["action"] for r in skill.recovery} == {"repair", "retire"}
    n_items = len(notes.items)
    res = run(ex, skill.skill_id, {"title": "gamma thing", "body": "third words"}, make_ctx(ScopedPolicy()))
    assert res.status == ResultStatus.OK and res.verified is True
    assert notes.items[-1] == {"id": n_items + 1, "title": "gamma thing", "body": "third words", "status": "published"}
    assert [op.arguments["method"] for op in res.ops] == ["GET", "POST", "POST", "GET"]
    http.close()


def test_single_trace_induction_parameterises_the_declared_inputs(tmp_path, notes):
    http, lib, ex = setup(tmp_path)
    ctx = make_ctx(ScopedPolicy())
    t = trace(http, ctx, notes.url, "delta doc", "fourth words", inputs={"title": "delta doc", "body": "fourth words"})
    skill, report = ex.induce_validate_register([t], ctx, probe=listing_probe(notes.url),
                                                validation_arguments=[{"title": "epsilon", "body": "fifth words"}])
    assert skill is not None, report
    assert skill.provenance["method"] == "single_trace_inputs" and skill.arg_schema["required"] == ["body", "title"]
    assert skill.steps[0].arguments["json"] == {"title": {"$param": "title"}, "body": {"$param": "body"}}
    assert "$tpl" in skill.steps[1].arguments["url"]                 # the generated id is captured, not copied
    assert notes.items[-1]["title"] == "epsilon" and notes.items[-1]["status"] == "published"
    res = run(ex, skill.skill_id, {"title": "zeta", "body": "sixth words"}, ctx)
    assert res.status == ResultStatus.OK and notes.items[-1]["title"] == "zeta"
    # validation arguments equal to the source trace's are not "fresh" and do not count
    t2 = trace(http, ctx, notes.url, "eta doc", "seventh", inputs={"title": "eta doc", "body": "seventh"})
    none, report = ex.induce_validate_register([t2], ctx, probe=listing_probe(notes.url),
                                               validation_arguments=[{"title": "eta doc", "body": "seventh"}])
    assert none is None and report["runs"][0]["termination"] == "not_fresh"
    http.close()


def test_a_value_selected_from_an_earlier_listing_becomes_a_lookup(tmp_path, notes):
    """'Publish the item with this title': the id is found in the listing returned by the first step."""
    http, lib, ex = setup(tmp_path)
    ctx = make_ctx(ScopedPolicy())
    for title in ("red", "green", "blue", "white", "black"):
        assert http.execute(act("http.request", {"method": "POST", "url": notes.url + "/api/items",
                                                 "json": {"title": title, "body": "x"}}), ctx).status == ResultStatus.OK

    def publish_by_title(title):
        a1 = listing_probe(notes.url)
        r1 = http.execute(a1, ctx)
        item = next(i["id"] for i in r1.observations[0].data["json"]["items"] if i["title"] == title)
        a2 = act("http.request", {"method": "POST", "url": f"{notes.url}/api/items/{item}/publish"})
        r2 = http.execute(a2, ctx)
        after = http.execute(listing_probe(notes.url), ctx)
        return Trace.from_pairs([(a1, r1), (a2, r2)], inputs={"title": title}, before=r1.observations, after=after.observations)

    skill, report = ex.induce_validate_register([publish_by_title("green"), publish_by_title("white")], ctx,
                                                probe=listing_probe(notes.url), validation_arguments=[{"title": "red"}])
    assert skill is not None, report
    lookup = skill.steps[1].arguments["url"]["$tpl"][1]["$lookup"]
    assert lookup["where"] == {"title": {"$param": "title"}} and lookup["field"] == "id" and lookup["list"] == ["json", "items"]
    assert run(ex, skill.skill_id, {"title": "black"}, ctx).status == ResultStatus.OK
    assert {i["title"]: i["status"] for i in notes.items} == {"red": "published", "green": "published", "blue": "draft",
                                                             "white": "published", "black": "published"}
    http.close()


def test_trace_from_one_action_result_and_template_binding(notes):
    http, ctx = HttpApiExecutor(), make_ctx(ScopedPolicy())
    a = act("http.request", {"method": "POST", "url": notes.url + "/api/items", "json": {"title": "t one", "body": "b"}})
    r = http.execute(a, ctx)
    t = Trace.from_result(a, r)
    assert len(t.steps) == 1 and t.steps[0].action.operation == "http.request"
    assert t.steps[0].result.observations[0].data["json"]["title"] == "t one"
    data = [{"json": {"id": 7, "items": [{"id": 3, "name": "a"}, {"id": 4, "name": "b"}]}}]
    assert bind({"$tpl": ["/x/", {"$capture": {"step": 0, "path": ["json", "id"], "cast": "str"}}, "/", {"$param": "p"}]},
                {"p": "q"}, data) == "/x/7/q"
    assert bind({"$lookup": {"step": 0, "list": ["json", "items"], "where": {"name": {"$param": "n"}}, "field": "id"}},
                {"n": "b"}, data) == 4
    with pytest.raises(SkillBindError):
        bind({"$capture": {"step": 0, "path": ["json", "gone"]}}, {}, data)
    with pytest.raises(SkillBindError):
        bind({"$param": "missing"}, {}, data)
    http.close()


def test_induction_needs_a_common_successful_segment(tmp_path, notes):
    http, lib, ex = setup(tmp_path)
    ctx = make_ctx(ScopedPolicy())
    t1 = trace(http, ctx, notes.url, "one", "b1")
    other = act("http.get", {"url": notes.url + "/api/text"})
    t2 = Trace.from_pairs([(other, http.execute(other, ctx))])
    with pytest.raises(SkillInductionError, match="no reusable segment"):
        ex.inducer.induce([t1, t2])
    with pytest.raises(SkillInductionError):
        ex.inducer.induce([])
    # failed steps are never abstracted: a trace whose second step failed has no reusable two-step segment
    bad = act("http.request", {"method": "POST", "url": notes.url + "/api/items/999/publish"})
    t3 = Trace.from_pairs([t1.steps[0].action and (t1.steps[0].action, t1.steps[0].result), (bad, http.execute(bad, ctx))])
    with pytest.raises(SkillInductionError):
        ex.inducer.induce([t3])
    http.close()


# ---- validation --------------------------------------------------------------------------------------------------
def test_validation_failure_prevents_registration(tmp_path, notes):
    http, lib, ex = setup(tmp_path)
    ctx = make_ctx(ScopedPolicy())
    http.execute(act("http.request", {"method": "POST", "url": notes.url + "/api/items", "json": {"title": "old", "body": "x"}}), ctx)
    # an over-specific trace: it creates an item but publishes item 1, which existed before the trace
    before = http.execute(listing_probe(notes.url), ctx)
    a1 = act("http.request", {"method": "POST", "url": notes.url + "/api/items", "json": {"title": "fresh one", "body": "y"}})
    r1 = http.execute(a1, ctx)
    a2 = act("http.request", {"method": "POST", "url": notes.url + "/api/items/1/publish"})
    r2 = http.execute(a2, ctx)
    after = http.execute(listing_probe(notes.url), ctx)
    t = Trace.from_pairs([(a1, r1), (a2, r2)], inputs={"title": "fresh one", "body": "y"}, before=before.observations,
                         after=after.observations)
    candidate = ex.inducer.induce([t], probe=listing_probe(notes.url))
    assert candidate.steps[1].arguments["url"].endswith("/api/items/1/publish")          # the constant survived
    skill, report = ex.induce_validate_register([t], ctx, probe=listing_probe(notes.url))
    assert skill is None and not report["passed"]
    assert report["runs"][0]["termination"] == "postcondition_failed" and "published" in report["reason"]
    assert lib.list() == [] and not os.path.exists(os.path.join(lib.dir, "index.json"))
    rejected = [json.loads(l) for l in open(os.path.join(lib.dir, "rejected.jsonl"))]
    assert len(rejected) == 1 and rejected[0]["report"]["passed"] is False
    # an unvalidated candidate cannot be registered or run
    with pytest.raises(ValueError):
        lib.register(candidate)
    res = ex.run_skill(candidate, {"title": "t", "body": "b"}, ctx)
    assert res.status == ResultStatus.FAILED and res.artifacts["termination"] == "refused_stale"
    assert ex.validate(act("skill.run", {"skill_id": candidate.skill_id, "arguments": {}}, modality=Modality.SKILL))
    http.close()


def test_validation_fails_when_the_application_silently_ignores_a_step(tmp_path, notes):
    http, lib, ex = setup(tmp_path)
    ctx = make_ctx(ScopedPolicy())
    traces = [trace(http, ctx, notes.url, "a one", "b1"), trace(http, ctx, notes.url, "a two", "b2")]
    notes.publish_is_noop = True                      # every step still answers 200
    skill, report = ex.induce_validate_register(traces, ctx, probe=listing_probe(notes.url))
    assert skill is None and report["runs"][0]["termination"] == "postcondition_failed" and lib.list() == []
    http.close()


# ---- permission --------------------------------------------------------------------------------------------------
def test_a_skill_cannot_run_a_step_its_caller_may_not_run(tmp_path, notes):
    http, lib, ex, skill = learned(tmp_path, notes)
    # required scopes are exactly the union of the steps' scopes (and the read-only probe's)
    assert set(skill.required_scopes) == {s.auth_scope for s in skill.steps} | {skill.probe.auth_scope} \
        == {"http.read", "http.write"}
    before = (skill.alpha, skill.beta, len(notes.items))
    n_posts = sum(1 for r in notes.requests if r["method"] == "POST")

    # 1. the caller lacks the scope of the write steps
    pol = ScopedPolicy(scopes=("http.read",))
    res = run(ex, skill.skill_id, {"title": "nope", "body": "nope"}, make_ctx(pol))
    assert res.status == ResultStatus.DENIED and res.failure_class == FailureClass.MISSING_PERMISSION
    assert res.artifacts["termination"] == "denied" and res.retries == 0
    assert sum(1 for r in notes.requests if r["method"] == "POST") == n_posts          # no write reached the application
    # the same operation issued directly is refused in exactly the same way: the skill added no permission
    direct = http.execute(act("http.request", {"method": "POST", "url": notes.url + "/api/items",
                                               "json": {"title": "nope", "body": "nope"}}), make_ctx(pol))
    assert direct.status == ResultStatus.DENIED

    # 2. one specific step is refused: the earlier steps ran, the refused one did not, nothing after it ran
    pol = ScopedPolicy(deny=lambda op, a: "destination not allowed" if str(op.arguments.get("url", "")).endswith("/publish") else "")
    res = run(ex, skill.skill_id, {"title": "half", "body": "way"}, make_ctx(pol))
    assert res.status == ResultStatus.DENIED and notes.items[-1] == {"id": before[2] + 1, "title": "half", "body": "way",
                                                                     "status": "draft"}
    assert not any(r["path"].endswith("/publish") and r["path"].startswith(f"/api/items/{before[2] + 1}") for r in notes.requests)
    # every step was put to the policy individually, labelled as a skill step
    notes_ = [c.note for c in pol.checks if c.note.startswith("skill:")]
    assert [n.rsplit(":", 1)[1] for n in notes_] == ["probe0", "s0", "s1"]

    # 3. approval required for a step: the skill stops and reports it; it does not look for another way
    pol = ScopedPolicy(approval_rule=lambda op, a: op.arguments.get("method") == "POST")
    res = run(ex, skill.skill_id, {"title": "wait", "body": "for it"}, make_ctx(pol))
    assert res.status == ResultStatus.NEEDS_APPROVAL and res.artifacts["termination"] == "needs_approval"
    assert notes.items[-1]["title"] == "half"

    # refusals say nothing about the skill's reliability and never make it stale
    s = lib.get(skill.skill_id)
    assert (s.alpha, s.beta) == before[:2] and s.status == STATUS_VALIDATED
    assert [h["attributable"] for h in s.run_history] == [False, False, False]
    http.close()


def test_skill_generation_cannot_touch_authorization_or_the_verifier(tmp_path, notes):
    http, lib, ex, skill = learned(tmp_path, notes)
    ctx = make_ctx(ScopedPolicy())
    good = trace(http, ctx, notes.url, "ok", "ok")
    from cam_live.contracts import ActionResult

    def with_step(step):
        ok = ActionResult(action_id=step.action_id, status=ResultStatus.OK)
        return Trace.from_pairs([(s.action, s.result) for s in good.steps] + [(step, ok)])

    protected = [
        ActionSpec(action_id="x1", operation="auth.grant", modality=Modality.API, arguments={"scope": "everything"}),
        ActionSpec(action_id="x2", operation="approval.resolve", modality=Modality.INTERNAL, arguments={"approve": True}),
        ActionSpec(action_id="x3", operation="verifier.set_outcome", modality=Modality.API, arguments={"value": 100}),
        ActionSpec(action_id="x4", operation="sandbox.shell", modality=Modality.CODE, auth_scope="policy.admin",
                   arguments={"cmd": "true"}),
        ActionSpec(action_id="x5", operation="sandbox.python", modality=Modality.CODE,
                   arguments={"code": "open('/run/cam/audit.jsonl', 'a').write('x')"}),
        ActionSpec(action_id="x6", operation="sandbox.python", modality=Modality.CODE,
                   arguments={"path": "/srv/run/verifier.jsonl"}),
        ActionSpec(action_id="x7", operation="fs.write", modality=Modality.CODE, target="/var/cam/vault/discord_bot_token"),
        ActionSpec(action_id="x8", operation="skill.run", modality=Modality.SKILL, arguments={"skill_id": skill.skill_id}),
    ]
    for step in protected:
        assert protected_reasons(step), step.operation
        with pytest.raises(SkillInductionError, match="protected"):
            ex.inducer.induce([with_step(step)])
        none, report = ex.induce_validate_register([with_step(step)], ctx)
        assert none is None and "protected" in report["reason"]
    assert [s.skill_id for s in lib.list()] == [skill.skill_id]
    # a stored skill that was tampered with to contain such a step is refused at run time, before anything runs
    path = os.path.join(lib.dir, skill.skill_id, "v1.json")
    d = json.load(open(path))
    d["steps"].append(protected[0].to_dict())
    json.dump(d, open(path, "w"))
    lib2 = SkillLibrary(str(tmp_path), WallClock())
    ex2 = SkillExecutor(lib2, [http])
    pol, n = ScopedPolicy(), len(notes.requests)
    res = run(ex2, skill.skill_id, {"title": "t", "body": "b"}, make_ctx(pol))
    assert res.status == ResultStatus.DENIED and res.failure_class == FailureClass.MISSING_PERMISSION
    assert res.artifacts["termination"] == "protected" and pol.checks == [] and len(notes.requests) == n
    http.close()


# ---- monitoring --------------------------------------------------------------------------------------------------
def test_reliability_is_a_beta_posterior_with_a_credible_interval(tmp_path, notes):
    http, lib, ex, skill = learned(tmp_path, notes, forgetting=1.0)
    ctx = make_ctx(ScopedPolicy())
    assert (skill.alpha, skill.beta) == (2.0, 1.0)                    # uniform prior + one validation success
    r0 = skill.reliability()
    assert abs(r0["mean"] - 2 / 3) < 1e-12
    for i in range(6):
        assert run(ex, skill.skill_id, {"title": f"item {i} x", "body": f"body {i} y"}, ctx).status == ResultStatus.OK
    s = lib.get(skill.skill_id)
    r1 = s.reliability()
    assert (s.alpha, s.beta, s.runs, s.successes) == (8.0, 1.0, 6, 6)
    assert r1["mean"] > r0["mean"] and (r1["hi"] - r1["lo"]) < (r0["hi"] - r0["lo"])       # more evidence, narrower
    lo, hi = scipy_beta.ppf([0.05, 0.95], 8.0, 1.0)
    assert abs(r1["lo"] - lo) < 1e-9 and abs(r1["hi"] - hi) < 1e-9 and abs(beta_cdf(0.5, 3, 7) - scipy_beta.cdf(0.5, 3, 7)) < 1e-12
    assert beta_interval(1, 1, 0.9) == pytest.approx((0.05, 0.95), abs=1e-9)
    # an attributable failure lowers it; the summary observation reports it as quantities
    notes.publish_is_noop = True
    res = run(ex, skill.skill_id, {"title": "silently ignored", "body": "zzz"}, ctx)
    assert res.status == ResultStatus.FAILED and res.verified is False
    s = lib.get(skill.skill_id)
    assert (s.alpha, s.beta) == (8.0, 2.0) and s.reliability()["mean"] < r1["mean"] and s.status == STATUS_VALIDATED
    q = {x.name: x.value for x in res.observations[0].quantities}
    assert q["skill.reliability.mean"] == pytest.approx(8 / 9) and q["skill.primitive_ops"] == 4.0
    # a transient network fault is logged but is not evidence about the skill
    notes.publish_is_noop = False
    dead = SkillExecutor(lib, [], dispatch=lambda step, c: http.execute(
        act(step.operation, dict(step.arguments, url=step.arguments["url"].replace(notes.url, "http://127.0.0.1:9"))), c))
    res = dead.run_skill(lib.get(skill.skill_id), {"title": "x y z", "body": "b b b"}, ctx)
    assert res.failure_class == FailureClass.TRANSIENT_NETWORK
    s = lib.get(skill.skill_id)
    assert (s.alpha, s.beta) == (8.0, 2.0) and s.run_history[-1]["attributable"] is False
    # with forgetting, old successes count less (the default library discounts)
    lib_f = SkillLibrary(str(tmp_path / "f"), WallClock(), forgetting=0.5)
    k = Skill(skill_id="k", status=STATUS_VALIDATED, alpha=9.0, beta=1.0,
              validation_history=[{"ok": True, "fresh": True, "version": 1}])
    lib_f.register(k)
    assert lib_f.record_run("k", ok=False, failure_class=FailureClass.WRONG_SITE_ASSUMPTION).alpha == 5.0
    http.close()


def test_stale_after_interface_change_then_repaired_to_a_new_version(tmp_path, notes):
    http, lib, ex, skill = learned(tmp_path, notes)
    ctx = make_ctx(ScopedPolicy())
    assert run(ex, skill.skill_id, {"title": "before change", "body": "fine"}, ctx).status == ResultStatus.OK
    notes.generation = 2                                   # the application changes its interface
    r1 = run(ex, skill.skill_id, {"title": "after change 1", "body": "x"}, ctx)
    assert r1.status == ResultStatus.FAILED and r1.failure_class == FailureClass.WRONG_SITE_ASSUMPTION
    assert lib.get(skill.skill_id).status == STATUS_VALIDATED          # one failure is not yet "repeatedly"
    r2 = run(ex, skill.skill_id, {"title": "after change 2", "body": "x"}, ctx)
    stale = lib.get(skill.skill_id)
    assert r2.status == ResultStatus.FAILED and stale.status == STATUS_STALE and "interface" in stale.status_reason
    # a stale skill is never silently reused: it is refused and nothing is sent
    n = len(notes.requests)
    r3 = run(ex, skill.skill_id, {"title": "after change 3", "body": "x"}, ctx)
    assert r3.status == ResultStatus.FAILED and r3.artifacts["termination"] == "refused_stale" and len(notes.requests) == n
    assert "stale" in r3.failure_detail and r3.failure_class == FailureClass.PAGE_CHANGED
    listed = ex.execute(act("skill.list", {"status": "stale"}, modality=Modality.SKILL), ctx)
    assert [s["skill_id"] for s in listed.observations[0].data["skills"]] == [skill.skill_id]
    assert lib.candidate_actions() == []

    # repair: new successful traces on the new interface -> re-induction -> validation -> version 2
    new_traces = [trace(http, ctx, notes.url, "new one", "t1", generation=2), trace(http, ctx, notes.url, "new two", "t2", generation=2)]
    repaired, report = ex.repair(skill.skill_id, new_traces, ctx, probe=listing_probe(notes.url, 2))
    assert repaired is not None, report
    assert repaired.version == 2 and repaired.parent_version == 1 and repaired.status == STATUS_VALIDATED
    assert repaired.skill_id == skill.skill_id and repaired.provenance["repaired_from_version"] == 1
    assert repaired.arg_schema["required"] == ["name", "text"]
    assert repaired.steps[0].arguments["url"] == notes.url + "/api/v2/entries"
    res = run(ex, skill.skill_id, {"name": "works again", "text": "on the new interface"}, ctx)
    assert res.status == ResultStatus.OK and res.verified is True and res.artifacts["skill_version"] == "2"
    assert notes.items[-1]["title"] == "works again" and notes.items[-1]["status"] == "published"
    # versions are kept on disk and survive a restart
    assert lib.versions(skill.skill_id) == [1, 2] and lib.get(skill.skill_id, 1).status == STATUS_STALE
    again = SkillLibrary(str(tmp_path), WallClock())
    assert again.get(skill.skill_id).version == 2 and again.get(skill.skill_id).status == STATUS_VALIDATED
    assert again.get(skill.skill_id).to_dict() == lib.get(skill.skill_id).to_dict()
    assert [a.target for a in again.candidate_actions()] == [skill.skill_id]
    http.close()


def test_silent_change_is_detected_by_postconditions_and_binding(tmp_path, notes):
    http, lib, ex, skill = learned(tmp_path, notes)
    ctx = make_ctx(ScopedPolicy())
    notes.rename_id_field = True                          # the create response now calls the id "item_id"
    n_publish = sum(1 for r in notes.requests if r["path"].endswith("/publish"))
    res = run(ex, skill.skill_id, {"title": "renamed field", "body": "x"}, ctx)
    assert res.status == ResultStatus.FAILED and res.artifacts["termination"] == "binding_failed"
    assert res.failure_class == FailureClass.PAGE_CHANGED and res.observations[0].data["interface_changed"]
    assert sum(1 for r in notes.requests if r["path"].endswith("/publish")) == n_publish     # stopped before step 2
    notes.rename_id_field, notes.publish_is_noop = False, True
    res = run(ex, skill.skill_id, {"title": "ignored publish", "body": "x"}, ctx)
    assert res.artifacts["termination"] == "postcondition_failed" and res.verified is False
    assert [p["result"] for p in res.observations[0].data["postconditions"] if "published" in p["description"]] == [False, False]
    assert lib.get(skill.skill_id).status == STATUS_STALE
    http.close()


def test_unrecoverable_skill_is_retired(tmp_path, notes):
    http, lib, ex, skill = learned(tmp_path, notes, max_repairs=2)
    ctx = make_ctx(ScopedPolicy())
    old_traces = [trace(http, ctx, notes.url, "old a", "t1"), trace(http, ctx, notes.url, "old b", "t2")]
    notes.generation = 2
    for i in range(2):
        run(ex, skill.skill_id, {"title": f"x {i} y", "body": "zzz"}, ctx)
    assert lib.get(skill.skill_id).status == STATUS_STALE
    # the only traces available describe the old interface: repair cannot be validated
    none, report = ex.repair(skill.skill_id, old_traces, ctx)
    assert none is None and not report["passed"]
    s = lib.get(skill.skill_id)
    assert s.status == STATUS_STALE and s.repair_attempts == 1 and s.version == 1
    none, report = ex.repair(skill.skill_id, old_traces, ctx)
    s = lib.get(skill.skill_id)
    assert none is None and s.status == STATUS_RETIRED and "unrecoverable" in s.status_reason
    n = len(notes.requests)
    res = run(ex, skill.skill_id, {"title": "never", "body": "again"}, ctx)
    assert res.status == ResultStatus.FAILED and "retired" in res.failure_detail and len(notes.requests) == n
    assert ex.repair(skill.skill_id, old_traces, ctx)[0] is None                 # a retired skill stays retired
    assert SkillLibrary(str(tmp_path), WallClock()).get(skill.skill_id).status == STATUS_RETIRED
    assert lib.versions(skill.skill_id) == [1]
    http.close()


def test_low_reliability_alone_marks_a_skill_stale(tmp_path):
    lib = SkillLibrary(str(tmp_path), WallClock(), stale_after=99, forgetting=1.0)
    k = Skill(skill_id="k", status=STATUS_VALIDATED, alpha=2.0, beta=1.0,
              validation_history=[{"ok": True, "fresh": True, "version": 1}])
    lib.register(k)
    statuses = []
    for _ in range(5):
        statuses.append(lib.record_run("k", ok=False, failure_class=FailureClass.UNKNOWN, termination="step_failed:0").status)
    # P(reliability < 0.5 | Beta(2, 1 + f)) crosses 0.9 at the fifth failure (0.891 after four, 0.9375 after five)
    assert statuses == [STATUS_VALIDATED] * 4 + [STATUS_STALE]
    assert "P(reliability" in lib.get("k").status_reason


# ---- tools and temporally extended records -----------------------------------------------------------------------
def test_skill_tools_induce_list_describe(tmp_path, notes):
    http, lib, ex = setup(tmp_path)
    ctx = make_ctx(ScopedPolicy())
    assert {t.name: t.level for t in ex.tools()} == {"skill.run": Level.PROCEDURE, "skill.list": Level.PRIMITIVE,
                                                      "skill.describe": Level.PRIMITIVE, "skill.induce": Level.PROCEDURE}
    ids = []
    for title, body in (("first title", "first body"), ("second title", "second body")):
        pairs, _, _ = create_and_publish(http, ctx, notes.url, title, body)
        for a, r in pairs:
            ex.observe_execution(a, r)                 # what the environment does after each execution
        ids.append([a.action_id for a, _ in pairs])
    induce = act("skill.induce", {"traces": ids, "name": "create and publish", "description": "create an item and publish it",
                                  "probe": {"operation": "http.get", "arguments": {"url": notes.url + "/api/items"}}},
                 modality=Modality.SKILL)
    res = ex.execute(induce, ctx)
    assert res.status == ResultStatus.OK, res.failure_detail
    sid = res.artifacts["skill_id"]
    q = {x.name: x.value for x in res.observations[0].quantities}
    assert q == {"skill.validation.runs": 1.0, "skill.validation.ok": 1.0, "skill.registered": 1.0}
    listed = ex.execute(act("skill.list", {}, modality=Modality.SKILL), ctx).observations[0]
    assert listed.data["skills"][0]["skill_id"] == sid and listed.data["skills"][0]["status"] == STATUS_VALIDATED
    assert listed.data["skills"][0]["required_scopes"] == ["http.read", "http.write"]
    d = ex.execute(act("skill.describe", {"skill_id": sid}, modality=Modality.SKILL), ctx).observations[0].data["skills"][0]
    assert [s["operation"] for s in d["steps"]] == ["http.request", "http.request"] and d["versions"] == [1]
    assert d["postconditions"] and d["validation_history"][0]["ok"] and d["provenance"]["action_ids"] == ids
    # unknown action ids, a write operation as probe, and unknown skills are argument errors
    for bad in ({"traces": [["nope"]]}, {"traces": ids, "probe": {"operation": "http.request", "arguments": {}}}):
        r = ex.execute(act("skill.induce", bad, modality=Modality.SKILL), ctx)
        assert r.status == ResultStatus.FAILED and r.failure_class == FailureClass.INVALID_ARGUMENTS
    r = ex.execute(act("skill.run", {"skill_id": "skill_none", "arguments": {}}, modality=Modality.SKILL), ctx)
    assert r.failure_class == FailureClass.INVALID_ARGUMENTS
    r = ex.execute(act("skill.run", {"skill_id": sid, "arguments": {"title": 5}}, modality=Modality.SKILL), ctx)
    assert r.failure_class == FailureClass.INVALID_ARGUMENTS and lib.get(sid).runs == 0
    # skill executions themselves are not material for induction
    ex.observe_execution(induce, res)
    assert induce.action_id not in ex._buffer
    http.close()


def test_a_skill_run_is_one_procedure_level_action_with_its_low_level_trace(tmp_path, notes):
    http, lib, ex, skill = learned(tmp_path, notes)
    ctx = make_ctx(ScopedPolicy())
    res = run(ex, skill.skill_id, {"title": "option record", "body": "kept trace"}, ctx)
    option = res.observations[0].data["option"]
    assert option["level"] == Level.PROCEDURE and option["skill_id"] == skill.skill_id and option["version"] == 1
    assert option["termination"] == "postconditions_met" and option["n_primitive_ops"] == len(res.ops) == 4
    assert option["start_obs"] and option["end_obs"] and option["start_obs"] != option["end_obs"]
    assert [s["operation"] for s in option["steps"]] == ["http.get", "http.request", "http.request", "http.get"]
    assert option["elapsed_s"] > 0 and res.cost.primitive_ops == 4 and res.cost.seconds == option["elapsed_s"]
    assert all(op.started and op.ended >= op.started for op in res.ops)
    assert res.started <= res.ended and res.artifacts["termination"] == "postconditions_met"
    http.close()


# ---- measured comparison -----------------------------------------------------------------------------------------
def test_measured_primitive_only_versus_skill_based_execution(tmp_path, notes):
    """Repeated task: create an item, publish it, and confirm from the listing that it is published."""
    http, lib, ex = setup(tmp_path)
    ctx = make_ctx(ScopedPolicy())
    n_tasks = 20

    def confirmed(title):
        return any(i["title"] == title and i["status"] == "published" for i in notes.items)

    # primitive-only: every operation is a separate action the decision-maker has to issue
    decisions = [0]
    t0 = time.perf_counter()
    ok_p, ops_p, lat_p = 0, 0, []
    for i in range(n_tasks):
        t = time.perf_counter()
        pairs, _, after = create_and_publish(http, ctx, notes.url, f"primitive item {i}", f"body {i}", counter=decisions,
                                             observe_before=False)
        ops_p += len(pairs) + 1
        lat_p.append(time.perf_counter() - t)
        ok_p += confirmed(f"primitive item {i}")
    wall_p = time.perf_counter() - t0
    # acquisition: two of those successful episodes are abstracted and validated once
    traces = [trace(http, ctx, notes.url, "seed one", "s1"), trace(http, ctx, notes.url, "seed two", "s2")]
    t = time.perf_counter()
    skill, report = ex.induce_validate_register(traces, ctx, probe=listing_probe(notes.url))
    acquire_s = time.perf_counter() - t
    assert skill is not None
    # skill-based: one procedure-level action per task
    t0 = time.perf_counter()
    ok_s, ops_s, lat_s, dec_s = 0, 0, [], 0
    for i in range(n_tasks):
        t = time.perf_counter()
        res = run(ex, skill.skill_id, {"title": f"skill item {i}", "body": f"body {i}"}, ctx)
        dec_s += 1
        lat_s.append(time.perf_counter() - t)
        ops_s += len(res.ops)
        ok_s += res.status == ResultStatus.OK and res.verified is True and confirmed(f"skill item {i}")
    wall_s = time.perf_counter() - t0
    out = {
        "task": "create an item, publish it, confirm from the listing that it is published (local HTTP application)",
        "environment": "staging (loopback HTTP application in the test process); harness latency only, "
                       "no model inference is included in either arm",
        "n_tasks": n_tasks,
        "primitive_only": {"success": ok_p, "decisions": decisions[0], "decisions_per_task": decisions[0] / n_tasks,
                           "primitive_ops": ops_p, "primitive_ops_per_task": ops_p / n_tasks,
                           "latency_s_mean": sum(lat_p) / n_tasks, "latency_s_total": wall_p},
        "skill_based": {"success": ok_s, "decisions": dec_s, "decisions_per_task": dec_s / n_tasks,
                        "primitive_ops": ops_s, "primitive_ops_per_task": ops_s / n_tasks,
                        "latency_s_mean": sum(lat_s) / n_tasks, "latency_s_total": wall_s},
        "skill_acquisition": {"source_traces": 2, "validation_runs": len(report["runs"]),
                              "induce_and_validate_s": acquire_s, "skill_reliability": lib.get(skill.skill_id).reliability()},
        "note": "The skill issues one more primitive operation per task than the primitive-only actor (it observes "
                "the state before as well as after, to check its postconditions) and persists its monitoring "
                "record after every run, so harness latency per task is not lower; what falls is the number of "
                "decisions the learner has to make (3 -> 1 per task).",
        "command": "cd /workspace/camlive && .venv/bin/python -m pytest cam_live/tests/harness/skills/test_skills.py"
                   "::test_measured_primitive_only_versus_skill_based_execution -q --rootdir=cam_live",
    }
    os.makedirs(EVIDENCE, exist_ok=True)
    with open(os.path.join(EVIDENCE, "skills_vs_primitives.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    assert ok_p == ok_s == n_tasks
    assert decisions[0] == 3 * n_tasks and dec_s == n_tasks                 # 3 decisions per task -> 1
    assert ops_p == 3 * n_tasks and ops_s == 4 * n_tasks
    http.close()
