"""SG (amendment item 6): skills CAM acquires from its *own* executed traces — re-executable steps, abstraction of what
the procedure puts into the world, validation of single-example hypotheses, learned applicability, composition,
discovery of recurring procedures and refinement by experience.

Unit tests run without a browser on observation records of the format the structured browser produces; the
``browser`` tests run the real structured browser on the local staging sites through the real authorization service
(``validation.StagingComputer``), with actions shaped exactly like the agent's grounded candidates (``ref`` +
``obs_id`` + ``tab`` of the observation they were read from).
"""
from __future__ import annotations

import os
import random
import tempfile

import pytest

from cam_live.contracts import (ActionResult, ActionSpec, Cost, ExecContext, Level, Modality, Observation,
                                PrimitiveOp, Provenance, ResultStatus, ToolDescriptor, Trust, WallClock, new_id)
from cam_live.harness.skills import (ApplicabilityModel, Skill, SkillExecutor, SkillInducer, SkillInductionError,
                                     SkillLibrary, STATUS_VALIDATED, Trace, bind, element_of, observation_ref,
                                     protected_reasons, stable_arguments, state_features, without_step)

URL = "http://127.0.0.1:45001"


# ---------------------------------------------------------------------------------------------------- fixtures
def browser_tools():
    from cam_live.harness.browser import TOOLS
    return {t.name: t for t in TOOLS}


def form_page(obs_id, refs=("e1", "e2", "e3"), rows=0, extra_text="", tab="t1", url=URL + "/a/"):
    els = [{"role": "textbox", "name": "Designation", "tag": "input", "value": "", "ref": refs[0]},
           {"role": "combobox", "name": "Group", "tag": "select", "value": "Tools",
            "options": ["Tools", "Parts", "Misc"], "ref": refs[1]},
           {"role": "button", "name": "Store entry", "tag": "button", "ref": refs[2]}]
    data = {"url": url, "title": "Entry book (A)", "tab": tab, "elements": els,
            "tables": [{"caption": "Stored entries", "n_rows": rows, "rows": [["V"] for _ in range(rows)]}],
            "diff": {"elements_changed": [1] * rows}, "text_len": 100 + 7 * rows}
    return Observation(obs_id=obs_id, ts=float(rows), kind="page", text=f"Entry book {extra_text}", data=data,
                       provenance=Provenance(source="structured_browser", uri=url, trust=Trust.UNTRUSTED))


def step(op, args, obs_after, ok=True):
    t = browser_tools()[op]
    a = ActionSpec(action_id=new_id("act"), operation=op, modality=t.modality, level=Level.PRIMITIVE,
                   arguments=args, auth_scope=t.auth_scope)
    r = ActionResult(action_id=a.action_id, status=ResultStatus.OK if ok else ResultStatus.FAILED,
                     observations=[obs_after], ops=[PrimitiveOp(op=op, modality=t.modality, arguments=args)],
                     cost=Cost(seconds=0.2, primitive_ops=1))
    return a, r


def grounded_trace(name, label, rows, refs):
    """The agent's grounded store-an-entry actions: each names a ref of the observation before it."""
    o0 = form_page(new_id("obs"), refs, rows)
    o1 = form_page(new_id("obs"), refs, rows)
    o2 = form_page(new_id("obs"), refs, rows)
    o3 = form_page(new_id("obs"), tuple(f"x{r}" for r in refs), rows + 1, extra_text=f"Stored: {name}")
    pairs = [step("browser.fill", {"ref": refs[0], "obs_id": o0.obs_id, "tab": "t1", "text": name}, o1),
             step("browser.select_option", {"ref": refs[1], "obs_id": o1.obs_id, "tab": "t1", "label": label}, o2),
             step("browser.click", {"ref": refs[2], "obs_id": o2.obs_id, "tab": "t1"}, o3)]
    return Trace.from_pairs(pairs, before=[o0], after=[o3], context=[o0, o1, o2])


# ---------------------------------------------------------------------------------------------------- unit level
def test_an_element_reference_is_replaced_by_the_elements_role_and_name():
    tools = browser_tools()
    o = form_page("obs_1")
    a = ActionSpec(action_id="a", operation="browser.fill", modality=Modality.STRUCTURED_BROWSER,
                   arguments={"ref": "e1", "obs_id": "obs_1", "tab": "t1", "text": "x"})
    assert observation_ref(a) == "obs_1" and element_of(a, o)["name"] == "Designation"
    assert stable_arguments(a, o, tools["browser.fill"]) == {"tab": "t1", "text": "x", "role": "textbox",
                                                             "name": "Designation"}
    b = ActionSpec(action_id="b", operation="browser.click", modality=Modality.STRUCTURED_BROWSER,
                   arguments={"ref": "e3@obs_1"})
    assert observation_ref(b) == "obs_1" and stable_arguments(b, o, tools["browser.click"])["name"] == "Store entry"
    # duplicates are told apart by their position among equally named elements
    dup = form_page("obs_2")
    dup.data["elements"].append({"role": "button", "name": "Store entry", "ref": "e9"})
    c = ActionSpec(action_id="c", operation="browser.click", modality=Modality.STRUCTURED_BROWSER,
                   arguments={"ref": "e9", "obs_id": "obs_2"})
    assert stable_arguments(c, dup, tools["browser.click"])["nth"] == 1
    # an unknown observation, or a tool without a semantic locator, gives nothing
    assert stable_arguments(a, None, tools["browser.fill"]) is None
    no_locator = ToolDescriptor(name="x.fill", modality=Modality.API, description="",
                                arg_schema={"properties": {"ref": {}}})
    assert stable_arguments(a, o, no_locator) is None


def test_grounded_traces_induce_a_re_executable_parameterised_procedure():
    ind = SkillInducer(WallClock(), list(browser_tools().values()))
    t1 = grounded_trace("Widget one", "Parts", 0, ("e1", "e2", "e3"))
    t2 = grounded_trace("Gadget two", "Misc", 1, ("e5", "e6", "e7"))
    sk = ind.induce([t1, t2])
    fill, select, click = sk.steps
    # no observation handle survives: refs / obs ids / the session's tab are gone, locators are role + name
    for s in sk.steps:
        assert not {"ref", "obs_id", "tab"} & set(s.arguments)
    assert (fill.arguments["role"], fill.arguments["name"]) == ("textbox", "Designation")
    assert click.arguments == {"role": "button", "name": "Store entry"}
    # what varied is a parameter; a choice among the element's options carries the options as its domain
    props = sk.arg_schema["properties"]
    assert fill.arguments["text"] == {"$param": "text"} and select.arguments["label"] == {"$param": "label"}
    assert props["label"]["enum"] == ["Tools", "Parts", "Misc"] and "enum" not in props["text"]
    # observation bookkeeping (diff to the previous observation) is not a postcondition; the table gaining a row is
    post = [(c.kind, c.spec.get("path"), c.spec.get("op"), c.spec.get("value")) for c in sk.postconditions]
    assert ("metric", "tables/[0]/n_rows", "delta_eq", 1.0) in post
    assert not any(p and str(p).startswith("diff") for _, p, _, _ in post)
    # a size change both examples share (equal-length names) is still only a hypothesis for validation to test:
    # with a fresh, longer argument it differs (measured: 'text_len changes by +25' failed every validation)
    size = [c for c in sk.postconditions if c.spec.get("path") == "text_len"]
    assert size and all(c.spec.get("tentative") for c in size)
    assert not any(c.spec.get("tentative") for c in sk.postconditions if c.spec.get("path") == "tables/[0]/n_rows")
    # the precondition is the element the procedure starts on, not one address
    pre = {c.kind: c.spec for c in sk.preconditions}
    assert pre["element_present"] == {"role": "textbox", "name": "Designation"} and "url_matches" not in pre
    assert sk.provenance["stable_locators"] == 3


def test_a_single_trace_parameterises_what_the_procedure_puts_into_the_world():
    ind = SkillInducer(WallClock(), list(browser_tools().values()))
    sk = ind.induce([grounded_trace("Unique designation", "Parts", 2, ("e1", "e2", "e3"))])
    assert sk.steps[0].arguments["text"] == {"$param": "text"}          # absent before, present after: an input
    assert sk.steps[1].arguments["label"] == "Parts"                    # one example: a constant choice
    assert sk.arg_schema["required"] == ["text"]
    tentative = [c for c in sk.postconditions if c.spec.get("tentative")]
    assert [c.spec["path"] for c in tentative] == ["text_len"]          # a size that depends on the arguments
    strict = [c.spec["path"] for c in sk.postconditions if c.spec.get("op") == "delta_eq" and not c.spec.get("tentative")]
    assert "tables/[0]/n_rows" in strict                                # one more record: the procedure's effect
    assert any(c.kind == "text_present" and not c.spec.get("tentative") for c in sk.postconditions)
    # a length that happens to change by exactly one is still a size, not a count (measured on the records wizard:
    # 'text_len changes by -1' failed validation with a fresh name)
    t = grounded_trace("Another designation", "Parts", 2, ("e1", "e2", "e3"))
    t.after[0].data["text_len"] = t.before[0].data["text_len"] - 1
    sk2 = ind.induce([t])
    assert all(c.spec.get("tentative") for c in sk2.postconditions if c.spec.get("path") == "text_len")
    assert any(c.spec.get("path") == "text_len" for c in sk2.postconditions)


def test_a_step_on_an_unknown_observation_cannot_become_a_skill():
    ind = SkillInducer(WallClock(), list(browser_tools().values()))
    t = grounded_trace("Widget one", "Parts", 0, ("e1", "e2", "e3"))
    t.context = [None, None, None]
    with pytest.raises(SkillInductionError, match="cannot be re-executed"):
        ind.induce([t])


def test_learned_applicability_separates_states_and_gates_only_when_confident():
    m = ApplicabilityModel()
    good, bad = state_features([form_page("o1")]), state_features([form_page("o2", url=URL + "/b/")])
    bad = [f for f in bad if not f.startswith("el:")] + ["el:dialog:new entry"]
    assert m.p(good) == pytest.approx(0.5) and m.n == 0
    for _ in range(6):
        m.update(good, True)
    assert m.p(good) > 0.85 and m.p_below(good, 0.15) < 0.01
    assert m.p_below(bad, 0.15) < 0.75                                  # never seen failing there: keeps trying
    failures = 0
    while m.p_below(bad, 0.15) < 0.75 and failures < 30:            # it keeps working where it worked, and fails
        m.update(bad, False)                                        # in the other state
        m.update(good, True)
        failures += 1
        if failures <= 2:
            assert m.p_below(bad, 0.15) < 0.75                          # two failures are not yet conclusive
    assert m.p(bad) < 0.15 and failures <= 15                           # confidently not applicable there
    assert m.p(good) > 0.8 and m.p_below(good, 0.15) < 0.01            # and still applicable where it worked
    m2 = ApplicabilityModel.from_dict(m.to_dict())
    assert m2.p(bad) == pytest.approx(m.p(bad)) and m2.n == m.n == 6 + 2 * failures


def test_removing_a_step_renumbers_captures_and_refuses_a_needed_step():
    sk = Skill(skill_id="s", steps=[
        ActionSpec(action_id="s.s0", operation="http.get", modality=Modality.API, arguments={"url": "u"}),
        ActionSpec(action_id="s.s1", operation="http.get", modality=Modality.API, arguments={"url": "v"}),
        ActionSpec(action_id="s.s2", operation="http.request", modality=Modality.API,
                   arguments={"url": {"$tpl": ["x/", {"$capture": {"step": 0, "path": ["id"]}}]},
                              "q": {"$capture": {"step": 1, "path": ["n"]}}, "t": {"$param": "t"}})],
        arg_schema={"type": "object", "properties": {"t": {"type": "string"}}, "required": ["t"]})
    with pytest.raises(SkillInductionError):
        without_step(sk, 0)                                            # step 2 uses what step 0 returned
    sk.steps[2].arguments.pop("q")
    short = without_step(sk, 1)
    assert [s.action_id for s in short.steps] == ["s.s0", "s.s1"] and short.status == "candidate"
    assert short.steps[1].arguments["url"]["$tpl"][1]["$capture"]["step"] == 0
    assert bind(short.steps[1].arguments, {"t": "a"}, [{"id": 7}]) == {"url": "x/7", "t": "a"}
    assert short.provenance["dropped_steps"] == [{"operation": "http.get", "index": 1}]


def _buffered(ex, pairs):
    for a, r in pairs:
        ex.observe_execution(a, r)


def test_recurring_procedures_are_found_among_the_executed_actions(tmp_path):
    lib = SkillLibrary(str(tmp_path), WallClock())
    ex = SkillExecutor(lib, [])
    ex.inducer.set_tools(list(browser_tools().values()))
    noise_ops = [("browser.scroll", {"to": "bottom"}), ("browser.read", {"max_chars": 50}), ("browser.inspect", {})]
    for i in range(3):
        _buffered(ex, [step(noise_ops[i][0], noise_ops[i][1], form_page(new_id("obs")))])
        t = grounded_trace(f"Name {i}", ["Tools", "Parts", "Misc"][i], i, (f"e{3 * i}", f"e{3 * i + 1}", f"e{3 * i + 2}"))
        for k, s in enumerate(t.steps):
            ex._note_observations([t.context[k]])
        _buffered(ex, [(s.action, s.result) for s in t.steps])
    found = ex.recurring_procedures()
    assert found and found[0]["operations"] == ["browser.fill", "browser.select_option", "browser.click"]
    assert found[0]["occurrences"] == 3 and len(found[0]["traces"]) == 3
    # the buffered contexts make every occurrence a trace the inducer can turn into one parameterised skill
    traces = [ex.trace_of(ids) for ids in found[0]["traces"]]
    sk = ex.inducer.induce(traces)
    assert sorted(sk.arg_schema["properties"]) == ["label", "text"]
    # read-only repetition alone is never a procedure
    for _ in range(3):
        _buffered(ex, [step("browser.scroll", {"to": "top"}, form_page(new_id("obs"))),
                       step("browser.read", {"max_chars": 9}, form_page(new_id("obs")))])
    assert all(any(op not in ("browser.scroll", "browser.read") for op in f["operations"]) for f in ex.recurring_procedures())


def test_composition_rules():
    run = ActionSpec(action_id="r", operation="skill.run", modality=Modality.SKILL, arguments={"skill_id": "k"})
    assert protected_reasons(run) == []
    for op in ("skill.induce", "skill.discover", "skill.list", "skill.describe"):
        assert protected_reasons(ActionSpec(action_id="x", operation=op, modality=Modality.SKILL))
    ind = SkillInducer(WallClock(), [], skill_scopes=lambda sid: ["browser.write"] if sid == "known" else None)

    def tr(sid):
        o = form_page(new_id("obs"))
        nav = ActionSpec(action_id=new_id("a"), operation="browser.navigate", modality=Modality.STRUCTURED_BROWSER,
                         arguments={"url": URL + "/a/"}, auth_scope="browser.read")
        sr = ActionSpec(action_id=new_id("a"), operation="skill.run", modality=Modality.SKILL,
                        arguments={"skill_id": sid, "arguments": {"text": "Abc def"}})
        ok = lambda a: ActionResult(action_id=a.action_id, status=ResultStatus.OK, observations=[o])
        return Trace.from_pairs([(nav, ok(nav)), (sr, ok(sr))], before=[o], after=[o])
    sk = ind.induce([tr("known")])
    assert sk.steps[1].arguments["skill_id"] == "known" and sk.provenance["composed_of"] == ["known"]
    assert "browser.write" in sk.required_scopes and "browser.read" in sk.required_scopes
    with pytest.raises(SkillInductionError, match="not a validated skill"):
        ind.induce([tr("unknown")])
    with pytest.raises(SkillInductionError, match="contain itself"):
        ind.induce([tr("known")], skill_id="known")


# ---------------------------------------------------------------------------------------------------- real browser
class Staging:
    def __init__(self, root):
        from cam_live.validation import StagingComputer
        self.sc = StagingComputer(os.path.join(root, "cu"), WallClock())
        self.task = self.sc.task("store_entry", max_decisions=200)
        self.env, self.policy, _ = self.sc.open(self.task, self.sc.start_url("store_entry"))
        self.tools = {t.name: t for t in self.env.discover_tools()}
        self.skills = next(e for e in self.env._executors.values() if isinstance(e, SkillExecutor))

    def run(self, op, args):
        t = self.tools[op]
        a = ActionSpec(action_id=new_id("act"), operation=op, modality=t.modality, level=t.level, arguments=args,
                       auth_scope=t.auth_scope)
        return a, self.env.execute(a, self.policy.authorize(a, self.task, {}))

    @staticmethod
    def page(r):
        return next((o for o in reversed(r.observations) if o.data.get("elements")), None)

    def store(self, name, label, extra_read=False):
        """The agent's grounded way of storing an entry (refs of the observation each action was read from)."""
        _, r = self.run("browser.inspect", {})
        o = self.page(r)
        els = {e.get("name"): e for e in o.data["elements"]}
        ids = []
        a, r = self.run("browser.fill", {"ref": els["Designation"]["ref"], "obs_id": o.obs_id, "tab": o.data["tab"],
                                         "text": name})
        ids.append(a.action_id)
        o = self.page(r) or o
        if extra_read:
            a, r = self.run("browser.read", {"max_chars": 200})
            ids.append(a.action_id)
        a, r = self.run("browser.select_option", {"ref": els["Group"]["ref"], "obs_id": o.obs_id,
                                                  "tab": o.data["tab"], "label": label})
        ids.append(a.action_id)
        o = self.page(r) or o
        a, r = self.run("browser.click", {"ref": els["Store entry"]["ref"], "obs_id": o.obs_id, "tab": o.data["tab"]})
        ids.append(a.action_id)
        assert r.status == ResultStatus.OK, r.failure_detail
        return ids

    def entries(self):
        return list(self.sc.sites.variants.entries)

    def close(self):
        self.sc.close()


@pytest.fixture
def staging(tmp_path):
    s = Staging(str(tmp_path))
    yield s
    s.close()


@pytest.mark.browser
def test_own_grounded_traces_become_a_validated_skill_that_is_reused(staging):
    """Before SG both induce calls failed validation: 'observation ... is outdated' (one trace) and 'malformed element
    ref' (two traces, the varying ref became a parameter) — the X4 failures of the real-model runs."""
    t1 = staging.store("Widget zero", "Parts")
    t2 = staging.store("Gadget one", "Misc")
    n = len(staging.entries())
    _, r = staging.run("skill.induce", {"traces": [t1, t2]})
    assert r.status == ResultStatus.OK, r.failure_detail
    sk = staging.skills.library.get(r.artifacts["skill_id"])
    assert sk.status == STATUS_VALIDATED and sorted(sk.arg_schema["properties"]) == ["label", "text"]
    assert len(staging.entries()) == n + 1                        # validation executed it with fresh arguments
    assert r.cost.primitive_ops == 3                              # and that is the induce action's cost
    for i, label in enumerate(("Tools", "Misc")):
        _, r = staging.run("skill.run", {"skill_id": sk.skill_id, "arguments": {"text": f"Reused {i}", "label": label}})
        assert r.status == ResultStatus.OK and r.verified is True and len(r.ops) == 3, r.failure_detail
    assert [(e["name"], e["category"]) for e in staging.entries()[-2:]] == [("Reused 0", "tools"), ("Reused 1", "misc")]
    sk = staging.skills.library.get(sk.skill_id)
    assert sk.successes == 2 and sk.applicability_model().n == 3          # validation + 2 runs taught applicability
    # an argument outside the observed options is refused before anything runs
    _, r = staging.run("skill.run", {"skill_id": sk.skill_id, "arguments": {"text": "x", "label": "Nope"}})
    assert r.status != ResultStatus.OK and len(staging.entries()) == n + 3
    # a single own trace also validates (its text is a parameter: it flowed into the stored state)
    t3 = staging.store("Solo procedure", "Tools")
    _, r = staging.run("skill.induce", {"traces": [t3]})
    assert r.status == ResultStatus.OK, r.failure_detail
    solo = staging.skills.library.get(r.artifacts["skill_id"])
    assert solo.arg_schema["required"] == ["text"] and solo.provenance["postconditions_refuted"]


@pytest.mark.browser
def test_reuse_on_an_unfamiliar_path_and_composition_from_another_page(staging):
    sk_id = staging.run("skill.induce", {"traces": [staging.store("Alpha", "Parts"), staging.store("Beta", "Misc")]})[1] \
        .artifacts["skill_id"]
    # the same form served at a path the skill never saw (/app/): role + name locators carry over
    base = staging.sc.sites.urls["variants"].rstrip("/")
    staging.run("browser.navigate", {"url": base + "/app/"})
    _, r = staging.run("skill.run", {"skill_id": sk_id, "arguments": {"text": "On app path", "label": "Tools"}})
    assert r.status == ResultStatus.OK and staging.entries()[-1]["name"] == "On app path", r.failure_detail
    # from another site's page the skill's starting element is absent: refused cheaply by its precondition
    other = staging.sc.sites.urls["structured"].rstrip("/") + "/"
    staging.run("browser.navigate", {"url": other})
    n = len(staging.entries())
    _, r = staging.run("skill.run", {"skill_id": sk_id, "arguments": {"text": "Wrong page", "label": "Tools"}})
    assert r.artifacts["termination"] == "precondition_failed" and len(staging.entries()) == n and len(r.ops) == 0
    sk = staging.skills.library.get(sk_id)
    assert sk.status == STATUS_VALIDATED and sk.runs == 1           # not counted against its reliability ...
    n_app = sk.applicability_model().n                               # ... but against its applicability here
    # composition: "go to the entry book, then run the skill", executed twice from elsewhere, becomes a composite
    ids = []
    for i in range(2):
        staging.run("browser.navigate", {"url": other})
        a1, r1 = staging.run("browser.navigate", {"url": base + "/a/"})
        a2, r2 = staging.run("skill.run", {"skill_id": sk_id, "arguments": {"text": f"Composed {i}", "label": "Misc"}})
        assert r2.status == ResultStatus.OK, r2.failure_detail
        ids.append([a1.action_id, a2.action_id])
    assert staging.skills.library.get(sk_id).applicability_model().n == n_app + 2            # the two composed runs
    staging.run("browser.navigate", {"url": other})
    _, r = staging.run("skill.induce", {"traces": ids})
    assert r.status == ResultStatus.OK, r.failure_detail
    comp = staging.skills.library.get(r.artifacts["skill_id"])
    assert comp.operations() == ["browser.navigate", "skill.run"] and comp.provenance["composed_of"] == [sk_id]
    assert set(staging.skills.library.get(sk_id).required_scopes) <= set(comp.required_scopes)
    staging.run("browser.navigate", {"url": other})                 # from the other site's page
    _, r = staging.run("skill.run", {"skill_id": comp.skill_id, "arguments": {"text": "Composite run"}})
    assert r.status == ResultStatus.OK and staging.entries()[-1]["name"] == "Composite run", r.failure_detail
    assert len(r.ops) == 4                                          # navigate + the inner skill's three primitives


@pytest.mark.browser
def test_discover_registers_a_recurring_procedure_and_refines_it_with_experience(staging):
    for i, label in enumerate(("Tools", "Parts", "Misc")):
        staging.store(f"Recurring {i}", label, extra_read=True)     # each time with a needless read in the middle
    _, r = staging.run("skill.discover", {})
    assert r.status == ResultStatus.OK, r.failure_detail
    sk = staging.skills.library.get(r.artifacts["skill_id"])
    # the recurring procedure as executed: look at the page, fill, a needless read, choose, store
    assert sk.operations() == ["browser.inspect", "browser.fill", "browser.read", "browser.select_option",
                               "browser.click"]
    assert sorted(sk.arg_schema["properties"]) == ["label", "text"]
    ops_per_run = []
    for version in (1, 2, 3):
        for i in range(3):                                          # experience: three successful uses per version
            _, r = staging.run("skill.run", {"skill_id": sk.skill_id,
                                             "arguments": {"text": f"Use {version}.{i}", "label": "Parts"}})
            assert r.status == ResultStatus.OK, r.failure_detail
            ops_per_run.append(len(r.ops))
        if version < 3:
            _, r = staging.run("skill.discover", {})
            assert r.status == ResultStatus.OK and r.observations[0].data["refined"], r.observations[0].text
    v3 = staging.skills.library.get(sk.skill_id)
    assert v3.version == 3 and v3.operations() == ["browser.fill", "browser.select_option", "browser.click"]
    assert ops_per_run == [5, 5, 5, 4, 4, 4, 3, 3, 3]                 # efficiency improved with experience
    assert staging.skills.library.versions(sk.skill_id) == [1, 2, 3]
    assert staging.entries()[-1]["name"] == "Use 3.2"


@pytest.mark.browser
def test_the_reusable_portion_of_a_noisy_own_trace_is_found_by_validation(staging):
    """The agent's verified-progress trace starts with a detour (from the entry book's index it followed a link to the
    form). Replayed from where validation runs (the form itself) the first step's element is absent: the procedure
    is validated from the first step the current state supports — the part that reproduces."""
    base = staging.sc.sites.urls["variants"].rstrip("/")
    _, r = staging.run("browser.navigate", {"url": base + "/"})
    o = staging.page(r)
    link = next(e for e in o.data["elements"] if e.get("name") == "Layout A")
    a0, r0 = staging.run("browser.click", {"ref": link["ref"], "obs_id": o.obs_id, "tab": o.data["tab"]})
    assert r0.status == ResultStatus.OK
    ids = [a0.action_id] + staging.store("After a detour", "Parts")
    n = len(staging.entries())
    _, r = staging.run("skill.induce", {"traces": [ids]})
    assert r.status == ResultStatus.OK, r.failure_detail
    sk = staging.skills.library.get(r.artifacts["skill_id"])
    assert sk.operations() == ["browser.fill", "browser.select_option", "browser.click"]
    assert sk.provenance["restarted_at"] == 1 and len(staging.entries()) == n + 1
    # the element's identity stays a constant (before: the label was 'captured' from the equal table header text
    # of the detour's page, so the shorter procedure could not be formed)
    assert sk.steps[0].arguments["name"] == "Designation" and sk.steps[1].arguments["name"] == "Group"
    assert any(c.kind == "element_present" and c.spec["name"] == "Designation" for c in sk.preconditions)
