"""ST (amendment item 5): what lets research, hypotheses, experiments and plans be written, kept and revised.

Measured in the live planning run (docs/ST.md §1): with one 256-token answer for 8 alternatives the backbone's last
item was cut off in every decision (``not_json`` "Unterminated string"), the content of what CAM wrote into its own
records (notes, requests) never reached a later prompt (the observation digest showed ``internal.note: {"saved":
...}``), the prompt showed no earlier action's arguments, and the schema walker took "https://*" (an authorization
PATTERN) as a URL, so every walker navigation was refused. These tests pin the architectural fixes:

* independent samples per decision, each with its own token budget, parsed separately and de-duplicated, with each
  sample's raw text recorded with the decision;
* the OWN RECORDS section: every executed action of the objective with its arguments, expectation and result, and
  the text of what the agent wrote into its own records (tools the environment describes as internal with local side
  effects), oldest first; absent before anything was executed and with ``proposer_memory_chars`` 0 (ablation);
* the optional ``expected`` field of the action contract in the output format, kept with the action;
* no walker vocabulary from authorization patterns;
* the generator learns from the agent's own executed actions weighted by its own reward
  (``predict.candidates.ProposalLearningProvider``): gradient reaches the LoRA and the bridge, only positive
  advantages are targets, and an update raises the likelihood of the rewarded line in the rewarded context.

Where a test needs a known backbone output it replaces ``model.generate`` with a fixed text: that tests the parser
and the bookkeeping, not what the real backbone writes (that is measured on the real configuration, docs/ST.md).
"""
from __future__ import annotations

import json
import os

import pytest
import torch

from cam_live.agent import PROPOSAL_INSTRUCTIONS, PROPOSAL_INSTRUCTIONS_EXPECTED, CamLiveAgent
from cam_live.contracts import (ActionResult, ActionSpec, Cost, ExperienceItem, ExperienceRecord, Level, Limits,
                                Modality, Observation, SimClock, TaskSpec, ToolDescriptor)
from cam_live.predict import candidates as C
from cam_live.runtime.tasks import strategy_findings
from int_helpers import agent_config

SUCCESS = {"verifier": "membership", "metric": "m.count", "target": 5}


def _tools():
    obj = lambda req, **p: {"type": "object", "properties": p, "required": req,                 # noqa: E731
                            "additionalProperties": False}
    return [
        ToolDescriptor("internal.note", Modality.INTERNAL, "Append a note to the persistent workspace notebook.",
                       obj(["text"], text={"type": "string"}, tags={"type": "array", "items": {"type": "string"}}),
                       Level.PRIMITIVE, "", "local", Cost()),
        ToolDescriptor("internal.inspect", Modality.INTERNAL, "Read own records.",
                       obj(["what"], what={"type": "string", "enum": ["notes", "actions"]}), Level.PRIMITIVE, "",
                       "none", Cost()),
        ToolDescriptor("web.get", Modality.STRUCTURED_BROWSER, "Read a page.", obj(["url"], url={"type": "string"}),
                       Level.PRIMITIVE, "web.read", "external_read", Cost(seconds=1.0)),
        ToolDescriptor("wait", Modality.INTERNAL, "Wait.", obj(["seconds"], seconds={"type": "number", "minimum": 1,
                                                                                      "maximum": 600}),
                       Level.PRIMITIVE, "", "none", Cost()),
    ]


def _agent(tmp_path, **settings):
    task = TaskSpec(objective_id="obj_st", objective="Grow a group to five members.", success=dict(SUCCESS),
                    interfaces=["structured_browser", "internal"], limits=Limits(duration_s=3600.0, max_decisions=50),
                    mode="staging", resources={"authorized_destinations": ["https://*", "channel:x:*"],
                                               "home": "https://owned.example/"})
    s = dict(use_backbone=True, planner=False, proposer_max_new_tokens=8, proposer_memory_chars=2400,
             proposer_expected=True, proposal_learning=True)       # the ST capabilities under test, switched on
    s.update(settings)
    return CamLiveAgent(agent_config(**s), str(tmp_path / "a"), clock=SimClock(start=1000.0), task=task)


def _fixed_generator(agent, texts_per_call):
    """Replace model.generate by fixed continuations (one list per call); record the call's arguments."""
    calls = []

    def gen(prompts, **kw):
        calls.append({"prompts": list(prompts), **kw})
        out = texts_per_call[min(len(calls) - 1, len(texts_per_call) - 1)]
        return [list(out)[:kw.get("num_return_sequences", 1)]]
    agent.model.generate = gen
    return calls


def _section(prompt, header):
    lines = prompt.splitlines()
    return json.loads(lines[lines.index(header) + 1]) if header in lines else None


def _decide(agent, tools):
    return agent.select(agent.evaluate(agent.propose(tools, 10)))


def _result(agent, d, status="ok", detail="", obs_text=""):
    t0 = agent.clock.now()
    obs = [Observation(obs_id=f"obs_r{agent.n_decisions}", ts=t0 + 0.5, kind="tool_result", text=obs_text)] \
        if obs_text else []
    res = ActionResult(action_id=d.chosen.action_id, status=status, started=t0, ended=t0 + 1.0, observations=obs,
                       failure_class="none" if status == "ok" else "missing_permission", failure_detail=detail)
    agent.record_result(d, res)
    agent.clock.advance(2.0)
    return res


def _choose(agent, tools, op):
    """One decision whose chosen candidate is the first one with operation ``op`` (the selection rule is not under
    test here: the chosen index is set on the evaluated set, as the runtime would execute it)."""
    evals = agent.evaluate(agent.propose(tools, 10))
    d = agent.select(evals)
    idx = next(i for i, c in enumerate(d.candidates) if c.action.operation == op)
    d.chosen_index = idx
    d.chosen = d.candidates[idx].action
    return d


# ------------------------------------------------------------------------------------------------------ format
def test_the_output_format_carries_expected_and_the_ablation_does_not(tmp_path):
    for flag, block in ((True, PROPOSAL_INSTRUCTIONS_EXPECTED), (False, PROPOSAL_INSTRUCTIONS)):
        agent = _agent(tmp_path / str(flag), proposer_expected=flag)
        calls = _fixed_generator(agent, [[""]])
        try:
            agent.propose(_tools(), 10)
            assert calls[0]["prompts"][0].startswith(block.format(k=agent.settings.proposer_k))
            assert ('"expected"' in calls[0]["prompts"][0].split(C.H_OBJECTIVE)[0]) is flag
        finally:
            agent.close()


def test_new_constant_text_carries_no_strategy():
    for t in (PROPOSAL_INSTRUCTIONS.format(k=2), PROPOSAL_INSTRUCTIONS_EXPECTED.format(k=2), C.H_MEMORY,
              "[3] internal.note text: x expect: y -> ok (queued for operator approval) | returned: z"):
        assert strategy_findings({"text": t}) == []


# ------------------------------------------------------------------------------------------------------ memory
def test_own_records_reach_later_prompts_with_arguments_expectation_and_result(tmp_path):
    agent = _agent(tmp_path)
    note = '{"operation": "internal.note", "arguments": {"text": "Hypothesis: X. Test: read page A.", "tags": ["h1"]}}'
    get = ('{"operation": "web.get", "arguments": {"url": "https://a.example/p"}, "expected": "a list of groups"}')
    calls = _fixed_generator(agent, [[note, get]])
    tools = _tools()
    try:
        agent.ingest([Observation(obs_id="obs_1", ts=agent.clock.now(), kind="metric", text="nothing yet")])
        d1 = _choose(agent, tools, "internal.note")
        assert _section(calls[-1]["prompts"][0], C.H_MEMORY) is None         # nothing executed yet: no section
        _result(agent, d1, obs_text='internal.note: {"characters": 34, "saved": "note_1"}')
        d2 = _choose(agent, tools, "web.get")
        mem = _section(calls[-1]["prompts"][0], C.H_MEMORY)
        assert "[1] internal.note tags: [h1]; text: Hypothesis: X. Test: read page A. -> ok" in mem
        _result(agent, d2, status="denied", detail="destination not authorized")
        _decide(agent, tools)
        mem = _section(calls[-1]["prompts"][0], C.H_MEMORY)
        lines = mem.splitlines()
        assert lines[0].startswith("[1] internal.note") and lines[1].startswith("[2] web.get")     # oldest first
        assert ('[2] web.get {"url": "https://a.example/p"} expect: a list of groups -> denied '
                "(missing_permission: destination not authorized)") in mem
        p = calls[-1]["prompts"][0].splitlines()
        assert p.index(C.H_MEMORY) < p.index(C.H_OBSERVATION)                 # own history, then the present
        assert d2.diagnostics["generation"]["memory_chars"] > 0
    finally:
        agent.close()


def test_memory_budget_keeps_own_written_records_longest_and_zero_turns_it_off(tmp_path):
    agent = _agent(tmp_path, proposer_memory_chars=300)
    tools = _tools()
    try:
        agent.propose(tools, 10)                                             # learns the tool descriptors
        a = ActionSpec(action_id="a_note", operation="internal.note", modality="internal", level="primitive",
                       arguments={"text": "keep me"})
        agent.objective_ops.append({"action_id": "a_note", "operation": "internal.note", "ok": True, "level": "p",
                                    "step": 1, **agent._memory_entry(a, ActionResult(action_id="a_note", status="ok"))})
        for i in range(2, 20):
            b = ActionSpec(action_id=f"a{i}", operation="web.get", modality="structured_browser", level="primitive",
                           arguments={"url": f"https://a.example/{i}"})
            agent.objective_ops.append({"action_id": b.action_id, "operation": "web.get", "ok": True, "level": "p",
                                        "step": i, **agent._memory_entry(b, ActionResult(action_id=b.action_id,
                                                                                         status="ok"))})
        mem = agent._memory_digest()
        assert len(mem) <= 300 and mem.splitlines()[0].startswith("[1] internal.note text: keep me")
        assert mem.splitlines()[-1].startswith("[19] web.get")              # newest other actions kept
        assert agent._memory_digest(0) == ""
        agent.objective_ops.append({"action_id": "old", "operation": "x", "ok": True, "level": "p"})  # old format
        assert "old" not in agent._memory_digest()                           # checkpoints before ST: skipped
    finally:
        agent.close()


def test_memory_ablation_sends_no_section(tmp_path):
    agent = _agent(tmp_path, proposer_memory_chars=0)
    calls = _fixed_generator(agent, [['{"operation": "internal.note", "arguments": {"text": "n"}}']])
    tools = _tools()
    try:
        d = _choose(agent, tools, "internal.note")
        _result(agent, d)
        agent.propose(tools, 10)
        assert _section(calls[-1]["prompts"][0], C.H_MEMORY) is None
    finally:
        agent.close()


def test_memory_survives_save_and_load(tmp_path):
    agent = _agent(tmp_path)
    _fixed_generator(agent, [['{"operation": "internal.note", "arguments": {"text": "persist this"}}']])
    tools = _tools()
    try:
        d = _choose(agent, tools, "internal.note")
        _result(agent, d)
        mem = agent._memory_digest()
        path = str(tmp_path / "ck")
        agent.save(path)
        agent2 = _agent(tmp_path / "b")
        agent2.load(path)
        agent2.tool_meta = dict(agent.tool_meta)
        assert agent2._memory_digest() == mem and "persist this" in mem
        agent2.close()
    finally:
        agent.close()


# ------------------------------------------------------------------------------------------------------ learning
def _rewarded_records(agent, n=6):
    """Executed records of one objective with given intrinsic rewards (what internal_rewards writes)."""
    recs = []
    for i in range(n):
        a = ActionSpec(action_id=f"act{i}", operation="web.get", modality="structured_browser", level="primitive",
                       arguments={"url": f"https://a.example/{i}"}, expected=f"page {i}")
        it = ExperienceItem(kind="observation", text=f"situation {i}", ts=1000.0 + i, ref=f"it{i}")
        agent.store.put_item(it)
        from cam_live.contracts import CandidateEvaluation, Decision
        d = Decision(decision_id=f"dec_t{i}", objective_id="obj_st", ts=1000.0 + i,
                     candidates=[CandidateEvaluation(action=a)], chosen_index=0, chosen=a, probabilities=[1.0])
        r = ExperienceRecord(exp_id=f"exp_t{i}", objective_id="obj_st", decision=d,
                             result=ActionResult(action_id=a.action_id, status="ok", started=1000.0 + i,
                                                 ended=1000.5 + i),
                             context_item_refs=[it.ref], goal_text=agent.task.objective, provenance="staging",
                             intrinsic_reward=float(i))
        agent.store.append(r)
        recs.append(agent.store.get(r.exp_id))
    return recs


def test_proposal_learning_targets_only_positive_advantages_and_reaches_the_adapters(tmp_path):
    from cam_live.learning_api import encode_decisions
    agent = _agent(tmp_path)
    try:
        prov = next(p for p in agent.learner.providers if p.name == "proposal_learning")
        assert prov.timescale == "slow_adapters"
        recs = _rewarded_records(agent)
        adv = prov._advantages(recs, agent.store)
        assert [a > 0 for a in adv] == [False, False, False, True, True, True]
        assert all(prov.eligible(r) for r in recs)
        failed = recs[0]
        failed.result.status = "denied"
        assert not prov.eligible(failed)
        batch = encode_decisions(agent.model, agent.store, recs[1:], grad=False)
        loss, m = prov.loss(agent.model, batch, agent.store)
        assert m["n_rows"] == 3.0 and loss is not None and torch.isfinite(loss)
        params = [p for p in agent.model.backbone.lora_parameters()] + list(agent.model.backbone.bridge.parameters())
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        assert any(g is not None and float(g.abs().sum()) > 0 for g in grads)
        none_batch = encode_decisions(agent.model, agent.store, recs[1:3], grad=False)
        loss0, m0 = prov.loss(agent.model, none_batch, agent.store)        # only below-average rows: no target
        assert loss0 is None and m0["n_rows"] == 0.0
        # the target is the action in the proposer's format; the prompt has the proposer's own instruction block
        assert C.proposal_line(recs[5].decision.chosen) == json.dumps(
            {"operation": "web.get", "arguments": {"url": "https://a.example/5"}, "expected": "page 5"})
        assert prov.prompt(recs[5], agent.store).startswith(
            PROPOSAL_INSTRUCTIONS_EXPECTED.format(k=agent.settings.proposer_k))
        assert "situation 5" in prov.prompt(recs[5], agent.store)
    finally:
        agent.close()


def test_a_proposal_learning_step_raises_the_rewarded_line_in_its_context(tmp_path):
    """Implementation correctness on the debug model: plain SGD on the provider's loss (no learner, no guard)
    increases log p(rewarded line | its context, b) and leaves an unrelated line's likelihood less changed."""
    from cam_live.learning_api import encode_decisions
    agent = _agent(tmp_path)
    try:
        prov = next(p for p in agent.learner.providers if p.name == "proposal_learning")
        recs = _rewarded_records(agent)
        bb = agent.model.backbone
        params = list(bb.lora_parameters()) + list(bb.bridge.parameters())
        target = recs[5]

        def lp(line):
            b = encode_decisions(agent.model, agent.store, [target], grad=False).b.detach()
            return float(agent.model.generation_logprob([prov.prompt(target, agent.store)], [line], latent=b)[0])
        line = C.proposal_line(target.decision.chosen)
        other = '{"operation": "wait", "arguments": {"seconds": 5}}'
        before, before_other = lp(line), lp(other)
        opt = torch.optim.SGD(params, lr=0.05)
        for _ in range(5):
            batch = encode_decisions(agent.model, agent.store, recs, grad=False)
            loss, _ = prov.loss(agent.model, batch, agent.store)
            opt.zero_grad()
            loss.backward()
            opt.step()
            bb.sync_caches()
        after, after_other = lp(line), lp(other)
        assert after > before + 1e-3
        assert (after - before) > (after_other - before_other)
    finally:
        agent.close()


def test_proposal_learning_is_off_in_the_ablation_and_without_backbone(tmp_path):
    for kw in ({"proposal_learning": False}, {"use_backbone": False}):
        agent = _agent(tmp_path / str(len(kw)) / str(list(kw)[0]), **kw)
        try:
            assert all(p.name != "proposal_learning" for p in agent.learner.providers)
        finally:
            agent.close()


@pytest.mark.parametrize("samples", [1, 4])
def test_one_decision_runs_end_to_end_on_the_debug_model(tmp_path, samples):
    """The real (tiny) backbone path: generate with num_return_sequences, parse, offer, evaluate, select."""
    agent = _agent(tmp_path, proposer_samples=samples, proposer_max_new_tokens=6)
    try:
        agent.ingest([Observation(obs_id="obs_1", ts=agent.clock.now(), kind="metric", text="nothing yet")])
        d = _decide(agent, _tools())
        assert d.diagnostics["generation"]["samples"] == samples
        assert len(d.diagnostics["generation"]["raw"]) == samples
    finally:
        agent.close()



@pytest.mark.slow
def test_proposal_learning_runs_in_the_learners_slow_step_on_a_runtime_lifetime(tmp_path):
    """Integration: the real runtime drives the agent through a LabWorld lifetime with backbone proposals on; the
    continual learner's slow (adapter) steps include the proposal-learning loss (held-out validated like the others),
    and the loss never touches the reward group."""
    import int_helpers as h
    agent = CamLiveAgent(h.agent_config(use_backbone=True, planner=False, proposal_learning=True,
                                        proposer_max_new_tokens=4, proposer_samples=1, promote_every=10_000),
                         str(tmp_path / "pl"), clock=SimClock())
    try:
        assert any(p.name == "proposal_learning" for p in agent.learner.providers)
        h.run_lifetime(h.Recorder(agent), agent.clock, str(tmp_path), 7, max_steps=24, max_decisions=14)
        ups = [u for u in agent.learner.updates if u.timescale == "slow_adapters"] \
            if hasattr(agent.learner, "updates") else []
        if not ups:
            import json as _j
            path = os.path.join(agent.dir, "learner_updates.jsonl")
            ups = [_j.loads(x) for x in open(path) if x.strip()]
            ups = [u.get("data", u) for u in ups]
            ups = [u for u in ups if u.get("timescale") == "slow_adapters"]
        assert ups, "no slow step was attempted"
        seen = [u for u in ups if "proposal_learning" in (u.get("losses") if isinstance(u, dict) else u.losses)]
        assert seen, [(u.get("losses"), str(u.get("notes"))[:300]) if isinstance(u, dict) else u.losses for u in ups]
        held = agent.store.sample(32, loss="val:proposal_learning", seed=0)
        assert held, "the provider has no held-out records"
    finally:
        agent.close()
