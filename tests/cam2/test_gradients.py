"""Gradient-flow tests for the frozen-LLM / CAM integration (task statement section 2).

* frozen LLM parameters never receive gradients;
* CAM parameters (policy, integration, control vectors) receive non-zero gradients through the frozen LLM;
* an optimizer step changes CAM parameters and changes action probabilities and generation log-probabilities.

The fast tests use a randomly initialised Qwen3-architecture model with the real Qwen3 tokenizer (same code paths);
``test_real_qwen_*`` repeats the key checks on the real pretrained Qwen3-0.6B when it is available locally.
"""
import os

import pytest
import torch

from agentrs.cam2.agent import Agent
from agentrs.cam2.backbone import load_llm
from agentrs.cam2.featurize import ToolRegistry, collate, make_step_input
from agentrs.cam2.envs.base import TaskRef
from agentrs.cam2.envs.fixture import LookupFixtureAdapter
from agentrs.cam2.model import CAM, CAMConfig

SMALL = dict(d=64, integ_layers=1, heads=4, core_dim=64, wm_hidden=64, head_hidden=64, prefix_len=4, prefix_rank=16,
             mem_k=0, wm_ensemble=2)


def _setup(llm, device="cpu"):
    cam = CAM(CAMConfig(**SMALL, llm_pooled_dim=llm.pooled_dim, llm_hidden=llm.hidden),
              prefix_init=llm.token_embedding_rows(" control")).to(device)
    reg = ToolRegistry()
    env = LookupFixtureAdapter()
    obs = env.reset(TaskRef("fixture", "type", 3))
    step = make_step_input(obs, env.available_tools(), reg, "", 0.0, 0, 4, False)
    return cam, reg, step


def _text_logp(llm, cam, reg, step):
    """Forward the full stochastic action path for the 'fill' candidate and return (candidate logp, text logp)."""
    b = collate(llm, reg, [step], None, None, cam.pol_q.weight.device)
    st = cam.init_state(1, b["el"].device)
    f, el_ctx, _, _ = cam.integrate(b, st)
    cand = cam.candidates(b, el_ctx)
    logits = cam.policy_logits(f, cand, b["cmask"])
    k = next(i for i, (t, e, _) in enumerate(step.cands) if reg.sets[step.tools_key]["specs"][t].name == "fill")
    agent = Agent(llm, cam, reg, b["el"].device)
    ctx, stub, _ = agent.gen_ids(step, step.cands[k])
    pre = cam.prefix(f, cand[:, k])
    tokens = llm.tok(" alpha", add_special_tokens=False)["input_ids"] + [llm.im_end]
    lpt = llm.logprob_with_prefix(pre, [ctx], [stub], [tokens])
    return torch.log_softmax(logits, -1)[0, k], lpt[0], (ctx, stub, tokens, k)


def _check(llm, device="cpu"):
    torch.manual_seed(0)
    cam, reg, step = _setup(llm, device)
    lpc, lpt, (ctx, stub, tokens, k) = _text_logp(llm, cam, reg, step)
    (-(lpc + lpt)).backward()  # policy-gradient direction for a positive advantage
    # 1. frozen LLM: no gradients, still frozen
    assert all(p.grad is None and not p.requires_grad for p in llm.model.parameters())
    # 2. CAM receives non-zero gradients through the frozen LLM (control vectors) and in its own policy
    g = {n: (p.grad.abs().sum().item() if p.grad is not None else 0.0) for n, p in cam.named_parameters()}
    assert g["pre_a.0.weight"] > 0 and g["pre_base"] > 0 and g["pre_scale"] > 0, "no gradient into the control vectors"
    assert g["pol_q.weight"] > 0 and g["integ.layers.0.linear1.weight"] > 0
    # 3. an optimizer step changes CAM parameters, action probabilities and generation behaviour
    before = {n: p.detach().clone() for n, p in cam.named_parameters()}
    opt = torch.optim.Adam(cam.parameters(), lr=1e-2)
    opt.step()
    changed = [n for n, p in cam.named_parameters() if not torch.equal(p, before[n])]
    assert "pre_a.0.weight" in changed and "pol_q.weight" in changed
    with torch.no_grad():
        lpc2, lpt2, _ = _text_logp(llm, cam, reg, step)
    assert lpc2 > lpc.detach(), "maximising logp did not raise the action probability"
    assert lpt2 > lpt.detach(), "maximising logp did not raise the frozen LLM's probability of the CAM-conditioned text"
    return float(lpt.detach()), float(lpt2)


def test_tiny_llm_gradient_flow():
    llm = load_llm("tiny-random", device="cpu")
    _check(llm)


def test_prefix_changes_sampled_generation_distribution():
    llm = load_llm("tiny-random", device="cpu")
    cam, reg, step = _setup(llm)
    b = collate(llm, reg, [step], None, None, "cpu")
    f, el_ctx, _, _ = cam.integrate(b, cam.init_state(1, "cpu"))
    cand = cam.candidates(b, el_ctx)
    ctx, stub = llm.chat_ids("Type the word alpha.")
    with torch.no_grad():
        p1 = cam.prefix(f, cand[:, 0])
        g1 = llm.sample_with_prefix(p1, [ctx], [stub], max_new=6, greedy=True)
        g0 = llm.sample_with_prefix(p1 * 0 + 3.0, [ctx], [stub], max_new=6, greedy=True)
    assert g1.tokens != g0.tokens or float(g1.logp) != float(g0.logp)


def test_sample_logp_matches_recompute():
    """The behaviour log-probability of sampled text equals the learner's recomputation (exact replay)."""
    llm = load_llm("tiny-random", device="cpu")
    cam, reg, step = _setup(llm)
    b = collate(llm, reg, [step], None, None, "cpu")
    f, el_ctx, _, _ = cam.integrate(b, cam.init_state(1, "cpu"))
    pre = cam.prefix(f, cam.candidates(b, el_ctx)[:, 0]).detach()
    ctx, stub = llm.chat_ids("Type the word alpha.")
    pre = pre.expand(2, -1, -1).contiguous()
    g = llm.sample_with_prefix(pre, [ctx, ctx[:-3]], [stub, stub], max_new=8, temperature=1.0,
                               gen=torch.Generator().manual_seed(1))
    lp = llm.logprob_with_prefix(pre, [ctx, ctx[:-3]], [stub, stub], g.tokens)
    assert torch.allclose(lp, g.logp, atol=1e-4)


def _real_model_available(name="Qwen/Qwen3-0.6B"):
    hf = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    return os.path.isdir(os.path.join(hf, "hub", "models--" + name.replace("/", "--")))


@pytest.mark.skipif(not (torch.cuda.is_available() and _real_model_available()), reason="needs GPU + local Qwen3-0.6B")
def test_real_qwen_gradient_flow():
    llm = load_llm("Qwen/Qwen3-0.6B", device="cuda")
    fp = llm.fingerprint()
    assert llm.trainable_params() == 0 and llm.n_params() > 5e8
    _check(llm, "cuda")
    assert llm.fingerprint() == fp, "pretrained weights changed"


def test_chunked_text_surrogate_gradient_is_exact():
    """Learner.text_surrogate (memory-bounded) gives the same CAM gradient as backpropagating a non-linear loss
    through the full graph of every row's frozen-LLM text log-probability."""
    from agentrs.cam2.learner import LearnConfig, Learner

    llm = load_llm("tiny-random", device="cpu")
    cam, reg, step = _setup(llm)
    b = collate(llm, reg, [step] * 5, None, None, "cpu")
    f, el_ctx, _, _ = cam.integrate(b, cam.init_state(5, "cpu"))
    cand = cam.candidates(b, el_ctx)
    pre = cam.prefix(f, cand[:, 0])
    ctx, stub = [], []
    for i in range(5):
        c, s = llm.chat_ids(f"Type word number {i}.")
        ctx.append(c)
        stub.append(s)
    toks = [llm.tok(f" w{i}" * (i + 1), add_special_tokens=False)["input_ids"] for i in range(5)]
    adv = torch.tensor([1.0, -0.5, 2.0, 0.3, -1.2])

    def objective(lp):  # PPO-like non-linear function of the log-probabilities
        r = torch.exp(lp - lp.detach())
        return -torch.minimum(r * adv, r.clamp(0.8, 1.2) * adv).mean() + 0.1 * (lp ** 2).mean()

    full = objective(llm.logprob_with_prefix(pre, ctx, stub, toks))
    g_full = torch.autograd.grad(full, list(cam.parameters()), retain_graph=True, allow_unused=True)
    learner = Learner(cam, llm, reg, LearnConfig(text_chunk=2), "cpu")
    with torch.no_grad():
        vals = llm.logprob_with_prefix(pre.detach(), ctx, stub, toks)
    leaf = vals.clone().requires_grad_(True)
    sur = learner.text_surrogate(objective(leaf), {"text": {"leaf": leaf, "pre": pre, "ctx": ctx, "stub": stub,
                                                            "tokens": toks}})
    g_sur = torch.autograd.grad(sur, list(cam.parameters()), allow_unused=True)
    n = 0
    for a, b_ in zip(g_full, g_sur):
        if a is None:
            assert b_ is None or float(b_.abs().max()) == 0.0
            continue
        n += 1
        assert torch.allclose(a, b_, atol=1e-6, rtol=1e-4)
    assert n > 0


@pytest.mark.skipif(not (torch.cuda.is_available() and _real_model_available()), reason="needs GPU + local Qwen3-0.6B")
def test_real_qwen_prefix_replay_correction_is_exact():
    """bf16: a control-vector recomputation that differs by fp32 noise (batch composition) flips bf16 roundings and
    changes the text log-prob; shifting it by (stored bf16 - recomputation at the behaviour parameters) restores the
    behaviour log-prob exactly (LearnConfig.prefix_replay_correction)."""
    llm = load_llm("Qwen/Qwen3-0.6B", device="cuda")
    g = torch.Generator(device="cuda").manual_seed(0)
    ctx, stub = llm.chat_ids("Write the JSON arguments for creating a playlist called road trip.")
    pre = torch.randn(1, 8, llm.hidden, device="cuda", generator=g) * 0.05
    out = llm.sample_with_prefix(pre, [ctx], [stub], max_new=48, gen=g)
    lp_old = llm.logprob_with_prefix(pre, [ctx], [stub], out.tokens)
    stored = pre.to(llm.dtype)
    noisy = lambda: pre * (1 + 3e-5 * torch.randn(pre.shape, device="cuda", generator=g))  # noqa: E731
    pre_prepass, pre_minibatch = noisy(), noisy()
    lp_plain = llm.logprob_with_prefix(pre_minibatch, [ctx], [stub], out.tokens)
    lp_fixed = llm.logprob_with_prefix(pre_minibatch + (stored.float() - pre_prepass), [ctx], [stub], out.tokens)
    assert float((lp_plain - lp_old).abs()) > 0          # the problem exists
    assert float((lp_fixed - lp_old).abs()) == 0.0       # and the correction removes it
