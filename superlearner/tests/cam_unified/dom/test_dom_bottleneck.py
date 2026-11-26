"""DOM Mechanism B (W5): the decision bottleneck closed by construction.

CPU unit tests of the decision package (intent head, selection, the cam_eval stamp, the candidate evaluator on the
real WorldModel / ValueHead, the fast set) and GPU tests on the REAL dom_dev stack (Qwen3-0.6B, 8 x 384 core):
the generator in intent mode has no tool prior and refuses (R, goal, belief) conditioning; I_t = 0 gives the same
(uniform) tool distribution for both goals of the two-goal task; DOM decisions are stamped, carry K evaluated
candidates and route gradients into the intent head, the core and the evaluator.
"""
import importlib
import os

import pytest
import torch

from superlearner.cam_unified.config import make_preset
from superlearner.cam_unified.decision.intent import INTENT_KIND, IntentHead, intent_tokens
from superlearner.cam_unified.interfaces import Action, TokenKind, TokenSet
from superlearner.cam_unified.testing import rand_tokens, tiny_config

SEL = importlib.import_module("superlearner.cam_unified.decision.select")
EV = importlib.import_module("superlearner.cam_unified.decision.evaluator")
FAST = importlib.import_module("superlearner.cam_unified.decision.fast")


# ============================================================================ intent head (CPU)

def test_intent_head_shapes_masking_and_gradients():
    cfg = tiny_config(n_intent=6)
    torch.manual_seed(0)
    h = IntentHead(cfg)
    B, S, N, D = 3, cfg.n_state, 10, cfg.d_model
    R = torch.randn(B, S, D, requires_grad=True)
    ctx = torch.randn(B, N, D, requires_grad=True)
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[:, 7:] = False
    kind = torch.full((B, N), int(TokenKind.OBS_SEM))
    I = h(R, ctx, mask, kind)
    assert I.shape == (B, 6, D) and I.dtype == torch.float32
    assert torch.equal(I, h(R, ctx, mask, kind))                       # deterministic
    ctx2 = ctx.detach().clone()
    ctx2[:, 7:] = 100.0                                                # masked positions are not read
    assert torch.allclose(I, h(R, ctx2, mask, kind), atol=1e-6)
    I.sum().backward()
    assert float(R.grad.abs().sum()) > 0 and float(ctx.grad[:, :7].abs().sum()) > 0
    assert float(ctx.grad[:, 7:].abs().sum()) == 0.0
    ts = intent_tokens(I.detach())
    assert ts.kind.unique().tolist() == [INTENT_KIND] and ts.mask.all()
    with pytest.raises(ValueError):
        IntentHead(tiny_config())                                      # n_intent = 0


# ============================================================================ selection + stamp (CPU)

def test_selection_probabilities_veto_greedy_and_exact_log_prob():
    Q = torch.tensor([[1.0, 2.0, 0.5, 2.0], [-1.0, -2.0, -3.0, -4.0]])
    gen = torch.zeros(2, 4)
    g = torch.Generator().manual_seed(0)
    idx, sel, recs = SEL.select(Q, gen, tau=0.5, generator=g)
    p = torch.softmax(Q / 0.5, -1)
    for b in range(2):
        assert recs[b]["probs"] == pytest.approx(p[b].tolist(), abs=1e-6)
        assert float(sel[b]) == pytest.approx(float(p[b, idx[b]].log()), abs=1e-6)
    # veto: Q < 0.8 rejected in row 0; row 1 entirely vetoed -> selects among all, flagged
    idx, sel, recs = SEL.select(Q, gen, tau=1.0, veto=0.8, generator=g)
    assert recs[0]["vetoed"] == [False, False, True, False] and not recs[0]["veto_all"]
    assert recs[0]["probs"][2] == 0.0 and int(idx[0]) != 2
    assert recs[1]["veto_all"] and recs[1]["vetoed"] == [False] * 4
    # greedy: argmax, deterministic (logp 0)
    idx, sel, recs = SEL.select(Q, gen, tau=1.0, greedy=True)
    assert idx.tolist() == [1, 0] and sel.tolist() == [0.0, 0.0] and recs[0]["greedy"]
    # differentiable recomputation equals the recorded value
    Qg = Q.clone().requires_grad_(True)
    idx, sel, recs = SEL.select(Q, gen, tau=0.7, generator=g)
    vet = torch.tensor([r["vetoed"] for r in recs])
    s, e = SEL.selection_log_prob(Qg, idx, 0.7, vet)
    assert torch.equal(s.detach(), sel)
    s.sum().backward()
    assert float(Qg.grad.abs().sum()) > 0
    assert float(e[0].detach()) > 0


def _stamped(tool="order_stock", args=None):
    cands = [Action(tool=tool, args=dict(args or {}), logp=-0.5), Action(tool="noop", args={}, logp=-1.0)]
    Q = torch.tensor([[0.3, 0.1]])
    idx, sel, recs = SEL.select(Q, torch.tensor([[-0.5, -1.0]]), greedy=True)
    a = Action.from_dict(cands[0].to_dict())
    SEL.attach_stamp(a, recs[0], cands, {"policy": 3, "reward_model": None})
    return a


def test_stamp_verifies_and_refuses_unstamped_tampered_and_forged_actions():
    a = _stamped(args={"n": 2})
    assert SEL.verify_stamp(a) == (True, "ok") and SEL.is_stamped(a)
    SEL.authorize([a, None])                                           # None rows = no action
    b = Action.from_dict(a.to_dict())                                  # survives serialisation (archive records)
    assert SEL.is_stamped(b)
    with pytest.raises(SEL.UnauthorizedAction):
        SEL.authorize([Action(tool="order_stock")])
    t = Action.from_dict(a.to_dict())
    t.args["n"] = 3                                                    # changed after CAM evaluated it
    with pytest.raises(SEL.UnauthorizedAction, match="digest"):
        SEL.require_stamp(t)
    f = Action.from_dict(a.to_dict())
    f.logp_parts[SEL.STAMP_KEY]["q"] = [9.0, 0.1]                      # forged evaluation
    assert not SEL.is_stamped(f)
    o = Action.from_dict(a.to_dict())
    o.tool = "noop"                                                    # swapped executable content
    assert not SEL.is_stamped(o)
    with pytest.raises(ValueError):
        SEL.attach_stamp(Action(tool="x"), {"v": 1}, [], {})           # reserved stamp keys cannot be overwritten


# ============================================================================ candidate evaluator (CPU, real WM)

@pytest.fixture(scope="module")
def wm_stack():
    from superlearner.cam_unified.core.backbone import UnifiedCore
    from superlearner.cam_unified.learn.heads import ValueHead
    from superlearner.cam_unified.world.dynamics import WorldModel
    from superlearner.cam_unified.world.outcomes import OutcomeHead
    cfg = tiny_config(n_intent=4)
    torch.manual_seed(0)
    core = UnifiedCore(cfg)
    wm = WorldModel(cfg, core, OutcomeHead(cfg)).eval()
    vh = ValueHead(cfg.d_model, 4).eval()
    ev = EV.CandidateEvaluator(cfg).eval()
    return cfg, wm, vh, ev


def _cands(B, K, D, seed):
    return [rand_tokens(K, 3 + b, D, kind=int(TokenKind.ACTION), seed=seed + b, ragged=True) for b in range(B)]


def test_evaluator_is_row_exact_zero_at_init_and_detaches_phi_and_value(wm_stack):
    cfg, wm, vh, ev = wm_stack
    B, K, D, S = 3, 4, cfg.d_model, cfg.n_state
    torch.manual_seed(1)
    R = torch.randn(B, S, D)
    I = torch.randn(B, 4, D)
    bel = rand_tokens(B, 5, D, kind=int(TokenKind.BELIEF), seed=3)
    goal = rand_tokens(B, 4, D, kind=int(TokenKind.GOAL), seed=4)
    A = _cands(B, K, D, 10)
    dt = torch.rand(B, K)
    with torch.no_grad():
        Q, pr = ev.score(R, bel, goal, A, dt, world_model=wm, value_head=vh, intent=I)
    assert Q.shape == (B, K) and pr["Z_hat"].shape == (B, K, S, D)
    assert float(Q.abs().max()) < 1e-6                                 # r_hat and V start at exactly 0
    # row exactness: row 1 alone (cut to its own padding) gives bit-identical Q
    with torch.no_grad():
        Q1, _ = ev.score(R[1:2], EV.row_cut(bel, 1), EV.row_cut(goal, 1), [A[1]], dt[1:2], world_model=wm,
                         value_head=vh, intent=I[1:2])
    assert torch.equal(Q1[0], Q[1])
    # Phi and V enter stop-gradient; the residual head is trained by the selection / eval losses
    lin = torch.nn.Linear(D, 1)

    def potential(Z, G):
        return torch.sigmoid(lin(Z.mean(1))).squeeze(-1)
    with torch.no_grad():
        ev.head.fc2.weight.normal_(0, 0.1)
    Q, pr = ev.score(R, bel, goal, A, dt, world_model=wm, value_head=vh, intent=I, potential=potential)
    (Q.sum() + EV.CandidateEvaluator.eval_loss(Q[:, 0], torch.ones(B)).sum()).backward()
    assert lin.weight.grad is None                                     # no gradient into Phi (psi side)
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0 for p in vh.parameters())
    assert float(ev.head.fc2.weight.grad.abs().sum()) > 0
    assert float(pr["phi"].abs().sum()) > 0
    with torch.no_grad():
        ev.head.fc2.weight.zero_()
    for p in ev.parameters():
        p.grad = None
    # batched mode agrees up to float noise
    with torch.no_grad():
        Qe, _ = ev.score(R, bel, goal, A, dt, world_model=wm, value_head=vh, intent=I, exact_rows=True)
        Qb, _ = ev.score(R, bel, goal, A, dt, world_model=wm, value_head=vh, intent=I, exact_rows=False)
    assert torch.allclose(Qe, Qb, atol=1e-4)


def test_fast_set_membership_rules():
    assert FAST.is_fast("intent_head.queries") and FAST.is_fast("decision_evaluator.head.fc2.weight")
    assert FAST.is_fast("value_head.mlp.fc1.weight")
    assert FAST.is_fast("action_generator.tool_q.weight") and FAST.is_fast("action_generator.heads.out.int.weight")
    assert FAST.is_fast("action_generator.ev_k.weight") and FAST.is_fast("action_generator.tool_a")
    for slow in ("core.blocks.0.attn.q.weight", "action_generator.bridge.out.weight", "world_model.prior_net.x",
                 "tokenizer.q_tool", "action_generator.lora.0", "reasoner.iter_emb", "beliefs.film.weight"):
        assert not FAST.is_fast(slow), slow


# ============================================================================ the real dom_dev stack (GPU)

HUB = "/workspace/.hf_home"


@pytest.fixture(scope="module")
def dom():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA and the cached Qwen3-0.6B")
    os.environ.setdefault("HF_HOME", HUB)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from superlearner.cam_unified.capabilities.stack import build_stack
    torch.manual_seed(0)
    st = build_stack(make_preset("dom_dev"), device="cuda", include_builtins=True, exact_rows=True)
    return st.model


@pytest.mark.gpu
def test_real_generator_in_intent_mode_has_no_bypass(dom):
    from superlearner.cam_unified.semantic.grammar import GrammarError
    ag = dom.action_generator
    assert ag.n_intent == dom.cfg.n_intent == 16 and ag.tool_prior is None and ag.src_emb is None
    assert ag.tool_q.bias is None and ag.tool_k.bias is None
    assert dom.build_report["semantic"] == "built from QwenSemantic"          # the real pretrained model
    assert dom.build_report["intent_head"].endswith("decision.intent:IntentHead")
    assert dom.build_report["decision_evaluator"].endswith("decision.evaluator:CandidateEvaluator")
    B, D = 2, dom.cfg.d_model
    R = torch.randn(B, dom.cfg.n_state, D, device="cuda")
    with pytest.raises(GrammarError, match="intent mode"):
        ag.sample(R, None, None, TokenSet.empty(B, D, "cuda"), None, [[], []], [[], []])
    with pytest.raises(GrammarError, match="intent mode"):
        ag.log_prob(R, None, None, TokenSet.empty(B, D, "cuda"), None, [[], []], [[], []], [None, None])


@pytest.mark.gpu
def test_real_zero_intent_tool_distribution_is_goal_independent(dom):
    from superlearner.cam_unified.decision import condition_g as cg
    r = cg.zero_intent_check(dom, B=8, seed=11)
    assert r["max_prob_gap_between_goals_I_zero"] == 0.0            # bilinear, bias-free: logits exactly 0
    assert r["uniform_at_I_zero"] < 1e-6
    assert r["n_tools_incl_builtins"] == 6


@pytest.mark.gpu
def test_real_dom_decision_is_stamped_evaluated_and_trains_the_bottleneck(dom):
    from superlearner.cam_unified.decision import condition_g as cg
    env = cg._new_env(4, 21)
    g = torch.Generator(device="cuda").manual_seed(0)
    dec = cg._decide(dom, env, generator=g)
    K = dom.cfg.n_candidates
    assert dec.intent.shape == (4, dom.cfg.n_intent, dom.cfg.d_model)
    for b, a in enumerate(dec.actions):
        st = a.logp_parts[SEL.STAMP_KEY]
        assert SEL.verify_stamp(a)[0] and st["n"] == K and len(st["candidates"]) == K
        assert a.logp == pytest.approx(a.logp_parts["gen_logp"] + a.logp_parts["selection"]
                                       + a.logp_parts.get("option", 0.0), abs=1e-5)
        assert dec.behavior[b]["policy"] == "cam_select" and dec.behavior[b]["logp_kind"] == "exact"
        assert len(dec.candidates[b]) == K and dec.evaluation[b]["chosen"] == a.proposal["chosen"]
        assert st["versions"]["policy"] == dom.version
    SEL.authorize(dec.actions)
    dom.zero_grad(set_to_none=True)
    dom.train()
    out = dom.policy_evaluate(dec.inputs, dec.actions)
    dom.eval()
    (out["logp"].sum() + out["q_exec"].sum()).backward()

    def nz(prefix):
        return sum(1 for n, p in dom.named_parameters() if n.startswith(prefix) and p.grad is not None
                   and float(p.grad.abs().sum()) > 0)
    assert nz("intent_head.") >= 10
    assert nz("core.blocks.0.") > 0 and nz(f"core.blocks.{dom.cfg.n_layers - 1}.") > 0
    assert nz("action_generator.tool_q.") == 1 and nz("action_generator.tool_k.") == 1
    assert nz("decision_evaluator.") >= 1                              # zero-init output layer: its weight first
    assert nz("tokenizer.") > 0                                        # TOOL / obs / goal tokens through the core
    dom.zero_grad(set_to_none=True)


@pytest.mark.gpu
def test_model_refuses_a_legacy_generator_with_n_intent(dom):
    from superlearner.cam_unified.model import ComponentUnavailable, build_model
    from superlearner.cam_unified.semantic.actions import ActionGenerator
    cfg_legacy = make_preset("dom_dev", n_intent=0)
    gen = ActionGenerator(cfg_legacy, dom.semantic, dom.tokenizer)
    with pytest.raises(ComponentUnavailable, match="intent mode"):
        build_model(make_preset("dom_dev"), components={"semantic": dom.semantic, "tokenizer": dom.tokenizer,
                                                        "action_generator": gen})
