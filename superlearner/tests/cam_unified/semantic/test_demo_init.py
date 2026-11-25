"""Declared demo-init condition: non-privileged ExampleInductor transcripts recorded in the real OpenWorld, replayed
through the integrated CamUnifiedModel's acting computation, behaviour cloning of decoder path + latent core."""
from __future__ import annotations

import pytest
import torch

crucible = pytest.importorskip("crucible")
crucible.load_all_modules()
from crucible.research import llm_eval as L                      # noqa: E402

from superlearner.cam_unified.interfaces import Action, ActionSource, Observation, ObsKind   # noqa: E402
from superlearner.cam_unified.semantic import demo_init as DI    # noqa: E402

TRAIN_SEEDS = (100_000, 100_001)     # training-split ow_tiny seeds at the CAM Unified development offset
HELD_SEED = 100_500                  # a development seed never used for training here


def _task(seed, family="ow_tiny"):
    return L.open_economy_goal(world_family=family, seed=seed, overrides={"n_consumers": 800}, days=8)


@pytest.fixture(scope="module")
def demos(tmp_path_factory):
    d = tmp_path_factory.mktemp("demos")
    (d / "train").mkdir()
    (d / "held").mkdir()
    train = DI.record_demo_dataset([_task(s) for s in TRAIN_SEEDS], str(d / "train"), days=1.5)
    held = DI.record_demo_dataset([_task(HELD_SEED)], str(d / "held"), days=1.5)
    return train, held, d


@pytest.fixture(scope="module")
def model(enc, cfg):
    from dataclasses import replace
    from superlearner.cam_unified.model import build_model
    c = replace(cfg, imag_branches=3, imag_samples=2, imag_horizons=(2,), mem_top_k=4, use_imagination=False,
                declared_conditions=(DI.CONDITION,))
    torch.manual_seed(0)
    return build_model(c, components={"semantic": enc}).to(enc.device)


def test_demo_dataset_holds_the_agents_inputs_and_refuses_privileged_or_heldout(demos, tmp_path):
    train, held, d = demos
    assert len(train) == 2 and len(held) == 1
    ep = train[0]
    assert ep.family == "ow_tiny" and ep.seed == TRAIN_SEEDS[0] and len(ep.decisions) > 30
    acts = [x for x in ep.decisions if x.action.tool != "wait"]
    assert {"use.categories", "use.examples", "id.create_org", "code.commit"} <= {x.action.tool for x in acts}
    for x in ep.decisions:
        assert x.action.source is ActionSource.SCRIPTED and x.action.logp is None and not x.action.logp_parts
        names = [t.name for t in ep.tools(x)]
        assert x.action.tool in names and names[-1] == "wait" and len(names) >= 140
    # the previous action's tool result is part of the next decision's observations (as runtime.run_steps merges)
    k = next(i for i, x in enumerate(ep.decisions) if x.action.tool == "bank.accounts")
    res_txt = [o.content for o in ep.decisions[k].result.observations if o.kind is ObsKind.TOOL_RESULT]
    assert res_txt and res_txt[0] in [o.content for o in ep.decisions[k + 1].observations]
    # the goal is the agent-visible one only
    assert ep.goal.text.startswith("Earn sustainable revenue") and ep.goal.family == "" and not ep.goal.meta
    # JSONL round trip
    again = DI.load_episodes(str(d / "train" / "episodes.jsonl"))
    assert [len(e.decisions) for e in again] == [len(e.decisions) for e in train]
    assert again[0].decisions[k].action.to_dict() == ep.decisions[k].action.to_dict()
    # privileged controls and held-out families never become demonstrations
    from crucible.agents.scripted import Builder
    with pytest.raises(RuntimeError):
        DI.ToolCapture(Builder())
    from superlearner.cam_unified.envs.worldgen import HeldOutFamilyError
    with pytest.raises(HeldOutFamilyError):
        DI.record_demo_dataset([_task(1, family="ow_heldout")], None, days=0.2)
    with pytest.raises(HeldOutFamilyError):           # validation seeds (> TRAIN_SEED_MAX) never train either
        DI.record_demo_dataset([_task(1_000_000 + 100_000)], None, days=0.2)


def test_evidence_labels_and_unrepresentable_demos():
    obs = [Observation(ObsKind.TOOL_RESULT, content='[{"account":"acct_000017","balance":3000000}]', t=0.1),
           Observation(ObsKind.TEXT, content="the forum is quiet", t=0.2),
           Observation(ObsKind.EVENT, content='{"type":"id.org","payload":{"id":"o:zovo-15-labs"}}', t=0.3),
           Observation(ObsKind.NUMERIC, value=3000000.0, unit="cents", t=0.3, source="summary:bank")]
    a = Action("biz.register", {"org": "o:zovo-15-labs", "account": "acct_000017", "sector": "labs"})
    assert DI.evidence_labels(obs, a, 4) == [0, 2]
    assert DI.evidence_labels(obs, a, 1) == [2]                     # newest first, capped
    assert DI.evidence_labels(obs, a, 4, candidates=[1, 2, 3]) == [2]
    assert DI.evidence_labels(obs, Action("x", {"amount": 3000000}), 4) == [3]     # the newest covering one
    assert DI.evidence_labels(obs, Action("x", {"name": "ab"}), 4) == []   # too short to be evidence


def test_teacher_forced_decide_is_the_acting_computation(model, demos):
    """With the same inputs, `teacher_forced_decide` builds the latent state, reasoning, prior, predictions, value,
    option and belief updates of `CamUnifiedModel.decide` (only the action is given instead of sampled): equal up to
    the run-to-run jitter of the GPU kernels (atomic adds in attention pooling), measured with a second decide()."""
    from superlearner.cam_unified.model import BudgetState
    train, _, _ = demos
    ep = train[0]
    st0, st1, st2 = (model.init_state(1, seed=3) for _ in range(3))
    for x in ep.decisions[:4]:
        args = ([list(x.observations)], [x.t], [ep.tools(x)], [ep.goal])
        bud = BudgetState.make(1, device=model.device)
        d1 = model.decide(st1, *args, budget=bud, greedy=True)
        d0 = model.decide(st0, *args, budget=bud, greedy=True)           # the same computation again: jitter
        d2 = DI.teacher_forced_decide(model, st2, *args, actions=list(d1.actions), budget=bud, greedy=True)
        assert d0.actions[0].tool == d1.actions[0].tool
        for name in ("Z", "R", "Z_hat", "value", "pred_mean"):
            a, b, c = (getattr(d, name).float() for d in (d1, d2, d0))
            jitter = float((a - c).abs().max())
            assert float((a - b).abs().max()) <= 4 * jitter + 1e-5 * (1 + float(a.abs().max())), (name, jitter)
        assert torch.equal(st1.option, st2.option) and st1.registries[0].slot == st2.registries[0].slot
        res = [x.result]
        nxt = [list(ep.decisions[x.step + 1].observations)]
        for st, d in ((st0, d0), (st1, d1), (st2, d2)):
            model.observe_outcomes(st, d, res, nxt, [0.0])
        jit = float((st1.fast.logw - st0.fast.logw).abs().max())
        assert float((st1.fast.logw - st2.fast.logw).abs().max()) <= 4 * jit + 1e-4
        st0.Z.copy_(st1.Z)                    # keep the three replicas on one trajectory (jitter does not compound)
        st2.Z.copy_(st1.Z)
        for f in ("particles", "logw", "pscale", "bayes_mu", "bayes_P", "bayes_noise", "bayes_n"):
            getattr(st0.fast, f).copy_(getattr(st1.fast, f))
            getattr(st2.fast, f).copy_(getattr(st1.fast, f))


def test_behaviour_cloning_needs_the_declared_condition(enc, cfg):
    from superlearner.cam_unified.model import build_model
    from dataclasses import replace
    m = build_model(replace(cfg, use_imagination=False), components={"semantic": enc}).to(enc.device)
    with pytest.raises(DI.UndeclaredCondition):
        DI.BehaviourCloning(m)


def test_behaviour_cloning_raises_heldout_demo_logp_through_decoder_and_core(model, demos):
    """BC on two training-seed transcripts raises the demo log-probability on a held-out seed's transcript and
    updates the core (backbone), the tokenizer projections, the decoder's LoRA, bridge and typed heads."""
    train, held, _ = demos
    steps = DI.teacher_forced_rollout(model, train, batch=2)
    held_steps = DI.teacher_forced_rollout(model, held, batch=1)
    assert sum(s.unrepresentable is not None for s in steps) <= 4               # code.run's large test payload
    assert any(s.evidence for s in steps) and any(not s.is_wait for s in steps)
    before = DI.demo_logp_table(model, held_steps)
    p0 = {n: p.detach().clone() for n, p in model.named_parameters()}
    bc = DI.BehaviourCloning(model, DI.BCConfig(batch=4, lr=1e-3, lr_sem=3e-4))
    for _ in range(3):
        bc.epoch(steps)
        steps = DI.teacher_forced_rollout(model, train, batch=2)              # refresh Z with the new weights
    held_steps = DI.teacher_forced_rollout(model, held, batch=1)
    after = DI.demo_logp_table(model, held_steps)
    print("held-out demo logp", before, "->", after)
    assert after["logp_act"] > before["logp_act"] + 5.0 and after["logp_wait"] > before["logp_wait"] + 1.0
    now = dict(model.named_parameters())
    changed = lambda pre: any(not torch.equal(p0[n], now[n].detach()) for n in now if n.startswith(pre))
    for pre in ("core.", "tokenizer.", "action_generator.heads.", "action_generator.bridge.",
                "action_generator.length_head.", "action_generator.lora."):
        assert changed(pre), pre
    assert any(r["core_grad"] > 0 for r in bc.log) and any(r["lora_grad"] > 0 for r in bc.log)
