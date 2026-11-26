"""DOM Mechanism B (W5): the behaviour log-probability of a DOM decision, generator AND selection, replays exactly.

Real dom_dev stack (Qwen3-0.6B, 8 x 384 core, K = 4 candidates): the recorded logp = log p_gen(a_k | I) +
log softmax(Q / tau)_k (+ option) of sampled decisions equals `policy_evaluate`'s recomputation bit for bit (no-grad
and gradient mode, and after the actions went through to_dict / from_dict as the archive stores them); the probe
agrees; a permuted half batch agrees up to the documented batch-composition tolerance (the backbone recomputes R in
another padding); greedy decisions are deterministic (logp 0); the fast-set functional view reproduces the policy.
"""
import importlib
import os

import pytest
import torch

from superlearner.cam_unified.config import make_preset
from superlearner.cam_unified.interfaces import Action

pytestmark = pytest.mark.gpu
SEL = importlib.import_module("superlearner.cam_unified.decision.select")


@pytest.fixture(scope="module")
def dom():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA and the cached Qwen3-0.6B")
    os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from superlearner.cam_unified.capabilities.stack import build_stack
    torch.manual_seed(0)
    st = build_stack(make_preset("dom_dev"), device="cuda", include_builtins=True, exact_rows=True)
    m = st.model
    # non-trivial selection: give the evaluator head non-zero weights so Q (and sel_logp) differ across candidates
    with torch.no_grad():
        m.decision_evaluator.head.fc2.weight.normal_(0, 0.5)
    return m


@pytest.fixture(scope="module")
def decision(dom):
    from superlearner.cam_unified.decision import condition_g as cg
    env = cg._new_env(8, 77)
    g = torch.Generator(device="cuda").manual_seed(3)
    return cg._decide(dom, env, generator=g)


def test_selection_is_nontrivial(decision):
    sel = [a.logp_parts["selection"] for a in decision.actions]
    assert any(abs(s - torch.log(torch.tensor(0.25)).item()) > 1e-3 for s in sel)     # not the uniform 1/K
    assert all(len(set(e["q"])) > 1 or len(e["q"]) == 1 for e in decision.evaluation)


def test_behaviour_logp_replays_bit_for_bit(dom, decision):
    rec = decision.logp.float().cpu()
    with torch.no_grad():
        out = dom.policy_evaluate(decision.inputs, decision.actions)
    assert torch.equal(out["logp"].float().cpu(), rec)
    gen = torch.tensor([a.logp_parts["gen_logp"] for a in decision.actions])
    sel = torch.tensor([a.logp_parts["selection"] for a in decision.actions])
    assert torch.equal(out["gen_logp"].float().cpu(), gen.float())
    assert torch.equal(out["sel_logp"].float().cpu(), sel.float())
    # the learner's mode: gradients enabled, train() left on the caller side
    dom.train()
    out2 = dom.policy_evaluate(decision.inputs, decision.actions)
    dom.eval()
    assert out2["logp"].requires_grad
    assert torch.equal(out2["logp"].detach().float().cpu(), rec)
    # records as the archive keeps them
    acts = [Action.from_dict(a.to_dict()) for a in decision.actions]
    with torch.no_grad():
        out3 = dom.policy_evaluate(decision.inputs, acts)
    assert torch.equal(out3["logp"].float().cpu(), rec)
    assert all(SEL.is_stamped(a) for a in acts)


def test_probe_agrees_and_half_batch_within_tolerance(dom, decision):
    probe = dom.policy_probe(decision.inputs, decision.actions)
    for b, row in enumerate(probe["rows"]):
        assert row["logp"]["selection"] == decision.actions[b].logp_parts["selection"]
        assert row["logp_total"] + row["option_logp"] == pytest.approx(float(decision.logp[b]), abs=1e-5)
        assert row["q"] == pytest.approx(decision.evaluation[b]["q"], abs=0)
    js = [6, 3, 0]
    with torch.no_grad():
        out = dom.policy_evaluate(decision.inputs.select(js), [decision.actions[j] for j in js])
    diff = (out["logp"].float().cpu() - decision.logp.float().cpu()[js]).abs().max()
    assert float(diff) < 0.02            # backbone recomputation in another batch padding (documented, not exact)


def test_greedy_decision_is_deterministic(dom):
    from superlearner.cam_unified.decision import condition_g as cg
    env = cg._new_env(4, 88)
    dec = cg._decide(dom, env, greedy=True)
    for a in dec.actions:
        st = a.logp_parts[SEL.STAMP_KEY]
        assert st["n"] == 1 and st["greedy"] and a.logp_parts["selection"] == 0.0
        assert a.logp_parts["gen_logp"] == 0.0 and SEL.is_stamped(a)
        assert a.logp == pytest.approx(a.logp_parts.get("option", 0.0), abs=0)
    assert all(b["logp_kind"] == "deterministic" for b in dec.behavior)


def test_fast_set_functional_view_reproduces_the_policy(dom, decision):
    from superlearner.cam_unified.decision import fast as F
    fast = F.fast_params(dom)
    cnt = F.fast_param_count(dom)
    assert 0 < cnt["fast"] < 50_000_000 and cnt["by_module"]["intent_head"] > 0
    feats = dom.policy_features(decision.inputs, decision.actions)
    out = F.functional_policy(dom, {k: v for k, v in fast.items()}, feats, decision.actions)
    assert torch.equal(out["logp"].detach().float().cpu(), decision.logp.float().cpu())
    # gradients reach only the fast set (features are detached)
    out["logp"].sum().backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in fast.values())
    assert all(p.grad is None for n, p in dom.named_parameters() if n.startswith("core."))
    dom.zero_grad(set_to_none=True)
    # a perturbed fast set changes the policy (functional override really used)
    pert = {k: (v.detach() + 0.05 * torch.randn_like(v)) if k.startswith("intent_head.") else v
            for k, v in fast.items()}
    out2 = F.functional_policy(dom, pert, feats, decision.actions)
    assert float((out2["logp"].detach() - out["logp"].detach()).abs().max()) > 1e-4
