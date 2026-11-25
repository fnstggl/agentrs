"""CamUnifiedModel: factory registry, parameter accounting, the step equation, imagination, belief updates."""
import copy

import pytest
import torch
import torch.nn as nn

from conftest import Stack, make_cfg
from superlearner.cam_unified import integration_doubles as dbl
from superlearner.cam_unified.interfaces import (ActionSource, ExecutionResult, Observation, ObsKind, TokenKind,
                                                 TokenSet)
from superlearner.cam_unified.model import (DEFAULT_COMPONENT_PATHS, CamUnifiedModel, ComponentUnavailable, ShapeError,
                                            available_components, build_model, check_tokens, fit_budget,
                                            required_components)


def first_obs(st: Stack):
    ob = st.env.reset()
    st.agent.reset(ob.env_ids)
    st.agent.set_goals(st.env.goals())
    return ob


def decide(st: Stack, ob, greedy=False, generator=None):
    agent = st.agent
    rows = list(range(agent.B))
    sub = agent.state.select(rows)
    dec = st.model.decide(sub, [list(o) for o in ob.observations], list(ob.t), st.env.available_tools(),
                          agent.goals, greedy=greedy, generator=generator or agent.torch_rng)
    agent.state.scatter(rows, sub)
    return dec


def test_build_model_from_dotted_paths():
    cfg = make_cfg()
    m = build_model(cfg, paths=dbl.DOUBLE_PATHS)
    assert isinstance(m.core, nn.Module) and type(m.core).__name__ == "DoubleCore"
    assert m.world_model.core is m.core                     # constructor injection by parameter name (shared core)
    assert m.world_model.outcome_head is m.outcome_head
    for name in ("tokenizer", "core", "reasoner", "beliefs", "bayes", "world_model", "retriever", "action_generator",
                 "value_head", "option_policy", "exploration_head", "entity_encoder"):
        assert m.build_report[name].startswith("built from superlearner.cam_unified."), (name, m.build_report[name])
    assert "LatentImaginationPlanner" in m.build_report["planner"]


def test_default_paths_name_every_unmerged_workstream_component():
    cfg = make_cfg()
    avail = available_components()
    assert set(avail) == set(DEFAULT_COMPONENT_PATHS)
    missing = sorted(n for n in required_components(cfg) if avail[n] is not True)
    if not missing:
        pytest.skip("all workstream components are importable; default build is covered by the acceptance tests")
    with pytest.raises(ComponentUnavailable) as e:
        build_model(cfg)
    for n in missing:
        assert f"{n}:" in str(e.value) and DEFAULT_COMPONENT_PATHS[n].split(":")[0] in str(e.value)


def test_required_components_follow_ablation_switches():
    cfg = make_cfg()
    comps = dbl.double_components(cfg)
    with pytest.raises(ComponentUnavailable):
        build_model(cfg, components={**comps, "retriever": None})
    with pytest.raises(ComponentUnavailable):
        build_model(cfg, components={**comps, "beliefs": None})
    cfg2 = make_cfg(use_memory=False, use_fast_adaptation=False, use_reasoning=False)
    m = build_model(cfg2, components={**dbl.double_components(cfg2), "retriever": None, "beliefs": None,
                                      "bayes": None, "reasoner": None})
    assert m.retriever is None and m.beliefs is None and m.reasoner is None
    assert required_components(cfg2) == {"tokenizer", "core", "outcome_head", "world_model", "action_generator",
                                         "value_head"}


class FakeSemantic(nn.Module):
    """Has the SemanticEncoder surface (hidden / embed / sem_dim): counted as pretrained."""
    sem_dim = 16
    version = "base"

    def __init__(self):
        super().__init__()
        self.base = nn.Linear(16, 16)
        self.base.requires_grad_(False)
        self.lora = nn.Linear(16, 2, bias=False)

    def hidden(self, texts, max_tokens):
        raise NotImplementedError

    def embed(self, texts):
        raise NotImplementedError


def test_count_parameters_dedupes_shared_modules_and_separates_semantic():
    cfg = make_cfg()
    comps = dbl.double_components(cfg)
    comps["tokenizer"].sem = FakeSemantic()                   # semantic model reachable only inside the tokenizer
    m = build_model(cfg, components=comps)
    c = m.count_parameters()
    uniq = {id(p): p for p in m.parameters()}
    sem = sum(p.numel() for p in comps["tokenizer"].sem.parameters())
    assert c["semantic_pretrained"]["total"] == sem
    assert c["semantic_pretrained"]["trainable"] == 2 * 16
    assert c["cam_total"] + sem == sum(p.numel() for p in uniq.values())
    core_n = sum(p.numel() for p in m.core.parameters())
    assert c["components"]["core"]["total"] == core_n
    own_wm = sum(p.numel() for n, p in m.world_model.named_parameters() if not n.startswith(("core.", "outcome_head.")))
    assert c["components"]["world_model"]["total"] == own_wm  # the shared core / head are not double counted
    assert c["components"]["query_encoder"]["total"] > 0


def test_decide_implements_the_step_equation_with_contract_shapes():
    st = Stack()
    cfg = st.cfg
    ob = first_obs(st)
    B, S, D = 2, cfg.n_state, cfg.d_model
    Z0 = st.agent.state.Z.clone()
    dec = decide(st, ob)
    assert len(dec.actions) == B
    for t in (dec.Z, dec.R, dec.Z_hat, dec.inputs.Z_prev):
        assert t.shape == (B, S, D)
    assert torch.allclose(dec.inputs.Z_prev, Z0) and not torch.allclose(st.agent.state.Z, Z0)
    check_tokens(dec.ctx, B, D, "ctx")
    kinds = set(dec.ctx.kind[dec.ctx.mask].tolist())
    # O_t (semantic + numeric), entities, B_t, G_t present at step 0 (no memory yet, no previous action/option)
    assert {TokenKind.OBS_SEM, TokenKind.OBS_NUM, TokenKind.ENTITY, TokenKind.BELIEF, TokenKind.GOAL} <= kinds
    assert dec.ctx.N <= cfg.max_ctx_tokens
    Q = len(dec.prospective_specs[0])
    assert dec.pred_mean.shape[1] >= Q and dec.value.shape == (B,) and dec.logp.shape == (B,)
    assert (dec.pred_var >= 0).all()
    for b, a in enumerate(dec.actions):
        assert a.logp is not None and abs(a.logp - float(dec.logp[b])) < 1e-5
        assert abs(a.logp - (a.logp_parts["tool"] + a.logp_parts["args"] + a.logp_parts["option"])) < 1e-4
        assert a.option_id is not None and 0 <= a.option_id < cfg.n_options
        assert set(dec.predicted[b]) == {s.key for s in dec.prospective_specs[b]}
        assert {"ok", "error_code", "cost", "duration", "reward"} <= set(dec.predicted[b])
    # registries bound the observed entities; the stored snapshot predates binding (exact recomputation)
    assert len(st.agent.state.registries[0].slot) > 0 and len(dec.inputs.registries[0].slot) == 0
    # second step: previous action and option tokens enter the context, memory once something is indexed
    st.model.retriever.add([type("R", (), {"event_id": "e0", "action": a, "result": ExecutionResult(True),
                                           "reward": 0.0})() for a in dec.actions[:1]])
    dec2 = decide(st, ob)
    kinds2 = set(dec2.ctx.kind[dec2.ctx.mask].tolist())
    assert {TokenKind.ACTION, TokenKind.OPTION, TokenKind.MEMORY} <= kinds2
    assert dec2.retrieved[0] and dec2.retrieved[0][0]["id"] == "e0"


def test_policy_evaluate_recomputes_behaviour_logp_exactly():
    st = Stack()
    ob = first_obs(st)
    dec = decide(st, ob)
    out = st.model.policy_evaluate(dec.inputs, dec.actions)
    assert torch.allclose(out["logp"], dec.logp, atol=1e-4)
    assert torch.allclose(out["value"], dec.value, atol=1e-5)
    assert out["logp"].requires_grad and out["value"].requires_grad
    out["logp"].sum().backward()
    for mod in ("core", "tokenizer", "reasoner", "action_generator", "option_policy", "exploration_head", "beliefs"):
        g = [p.grad for p in getattr(st.model, mod).parameters() if p.grad is not None]
        assert g and sum(float(x.abs().sum()) for x in g) > 0, mod


def test_reasoning_iterations_change_the_decision_inputs():
    st = Stack(cfg=make_cfg(use_imagination=False))
    ob = first_obs(st)
    state0 = copy.deepcopy(st.agent.state)
    st.model.reason_iters = 1
    d1 = decide(st, ob, greedy=True)
    st.agent.state = copy.deepcopy(state0)
    st.model.reason_iters = 3
    d3 = decide(st, ob, greedy=True)
    assert torch.allclose(d1.Z, d3.Z, atol=1e-5)               # same Z_t ...
    assert not torch.allclose(d1.R, d3.R, atol=1e-4)           # ... different R_t
    assert d1.reasoning[0]["iters"] == 1.0 and d3.reasoning[0]["iters"] > 1.0
    # the policy's evaluation of the same action depends on R (intervention on J changes decision probabilities)
    st.model.reason_iters = 1
    lp1 = st.model.policy_evaluate(d1.inputs, d1.actions)["logp"]
    st.model.reason_iters = 3
    lp3 = st.model.policy_evaluate(d1.inputs, d1.actions)["logp"]
    assert torch.allclose(lp1.detach(), d1.logp, atol=1e-4)
    assert not torch.allclose(lp1.detach(), lp3.detach(), atol=1e-4)


def test_imagination_selects_among_cam_candidates_without_touching_real_beliefs():
    st = Stack()
    ob = first_obs(st)
    fast0 = st.agent.state.fast.clone()
    n_upd = st.model.calls.get("imagination.beliefs.update", 0)
    dec = decide(st, ob)
    f = st.agent.state.fast
    assert torch.equal(f.particles, fast0.particles) and torch.equal(f.logw, fast0.logw)
    assert torch.equal(f.bayes_mu, fast0.bayes_mu)
    assert st.model.calls["imagination.beliefs.update"] - n_upd == st.model.imag_horizon   # imagined belief revisions
    n, H = st.model.imag_candidates, st.model.imag_horizon
    traj = dec.trajectory
    assert len(traj.steps) == H and traj.returns.shape == (2 * n * st.model.imag_samples,)
    assert traj.provenance.value == "imagined"
    for b, a in enumerate(dec.actions):
        assert a.source == ActionSource.CAM_PLANNER
        sc = a.proposal["scores"]
        assert len(sc) == n == len(dec.candidates[b]) == len(dec.imagined[b])
        assert a.proposal["chosen"] == max(range(n), key=lambda i: sc[i])
        chosen = dec.candidates[b][a.proposal["chosen"]]["action"]
        assert chosen["tool"] == a.tool and chosen["args"] == a.args
        assert all(item["horizon"] == H and len(item["next_tools"]) == st.model.imag_samples
                   for item in dec.imagined[b])
    # imagined belief ESS is tracked per step; the second step's actions were chosen from revised beliefs
    assert traj.steps[1].belief_ess.shape == traj.returns.shape


def test_imagined_evaluation_changes_the_real_action():
    """Same CAM candidates (same sampling seed); only the imagined evaluation differs (value head negated) -> the
    planner picks a different real action in at least one environment."""
    st = Stack(B=4, cfg=make_cfg(imag_branches=4))
    ob = first_obs(st)
    s0 = copy.deepcopy(st.agent.state)
    torch.manual_seed(123)
    d_a = decide(st, ob, generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        last = st.model.value_head.net[-1]
        last.weight.mul_(-50.0)
        last.bias.mul_(-50.0)
    st.agent.state = copy.deepcopy(s0)
    torch.manual_seed(123)
    d_b = decide(st, ob, generator=torch.Generator().manual_seed(7))
    for b in range(4):
        assert [c["action"]["args"] for c in d_a.candidates[b]] == [c["action"]["args"] for c in d_b.candidates[b]]
    chosen_a = [a.proposal["chosen"] for a in d_a.actions]
    chosen_b = [a.proposal["chosen"] for a in d_b.actions]
    assert chosen_a != chosen_b


def test_observe_outcomes_updates_particles_and_kalman_layer():
    st = Stack()
    ob = first_obs(st)
    dec = decide(st, ob)
    res = st.env.execute(dec.actions)
    adv = st.env.advance_time([r.duration + st.model.think_days for r in res])
    nxt = [list(r.observations) + list(o) for r, o in zip(res, adv.observations)]
    sub = st.agent.state.select([0, 1])
    logw0, n0, mu0 = sub.fast.logw.clone(), sub.fast.bayes_n.clone(), sub.fast.bayes_mu.clone()
    info = st.model.observe_outcomes(sub, dec, res, nxt, [0.0, 0.0])
    ob_ = info["outcomes"]
    assert ob_.y_mask.any() and ob_.q.x.shape[:2] == ob_.y.shape
    assert not torch.allclose(sub.fast.logw, logw0)                                   # particle weights moved
    assert (sub.fast.bayes_n.sum(1) > n0.sum(1)).all()                                # Kalman counts grew
    assert not torch.allclose(sub.fast.bayes_mu, mu0)
    assert info["loglik"].shape == (2,) and torch.isfinite(info["loglik"]).all()
    assert info["lp"] is None                                                         # no previous update yet
    assert sub.fast_prev is not None and torch.allclose(sub.fast_prev.logw, logw0)
    assert {"ess", "entropy", "top_w"} <= set(info["belief_after"][0])
    for b in range(2):                                                                # channels bound per key
        assert set(info["specs"][b][i].key for i in range(5)) <= set(sub.fast.channels[b])


def test_fit_budget_keeps_priority_kinds_and_most_recent():
    D = 8
    x = torch.randn(1, 6, D)
    kinds = torch.tensor([[TokenKind.MEMORY, TokenKind.OBS_SEM, TokenKind.GOAL, TokenKind.OBS_SEM, TokenKind.BELIEF,
                           TokenKind.ENTITY]])
    time = torch.tensor([[0.0, -5.0, 0.0, -0.1, 0.0, 0.0]])
    ts = TokenSet(x, torch.ones(1, 6, dtype=torch.bool), kinds, time, torch.full((1, 6), -1),
                  [[f"r{i}" for i in range(6)]])
    out = fit_budget(ts, 3)
    assert out.refs[0] == ["r2", "r3", "r4"]                  # goal, newest observation, belief (order preserved)
    assert torch.equal(out.x[0], x[0, [2, 3, 4]])
    with pytest.raises(ShapeError):
        check_tokens(out, 2, D, "bad batch")
