"""The real configuration: Qwen/Qwen3-4B + LoRA + bridge + the 50M experiential network (NA-01..04, NA-09, NA-12).

Run with ``-m gpu``. The module takes the shared GPU lock once for all its tests (they share one loaded model) and
releases memory afterwards.
"""
import time

import pytest
import torch

from cam_live.config import CamLiveConfig
from cam_live.contracts import ExperienceItem, ItemKind, Quantity
from cam_live.nn.api import GROUP_ADAPTERS, GROUP_BACKBONE_FROZEN, GROUP_EXPERIENTIAL
from cam_live.nn.gpu import GpuUnavailable, free_gpu_memory, gpu_lock
from cam_live.nn.model import build_model

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def qwen():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    try:
        with gpu_lock(timeout_s=1800.0, min_free_gb=14.0):
            torch.cuda.reset_peak_memory_stats()
            model = build_model(CamLiveConfig(device="cuda"))
            t0 = time.time()
            yield model
            assert time.time() - t0 < 900, "held the shared GPU lock for more than 15 minutes"
            assert torch.cuda.max_memory_allocated() < 24 * 2 ** 30
            del model
            free_gpu_memory()
    except GpuUnavailable as e:
        pytest.skip(str(e))


def _frozen_untouched(model):
    frozen = model.param_groups()[GROUP_BACKBONE_FROZEN]
    assert all(not p.requires_grad for p in frozen) and all(p.grad is None for p in frozen)


def test_parameter_counts_of_the_real_configuration(qwen):
    c = qwen.count_parameters()
    assert qwen.backbone.name == "Qwen/Qwen3-4B" and qwen.backbone.pretrained
    assert qwen.backbone.revision == "1cfa9a7208912126459214e8b04321603b3df60c"
    assert 3.9e9 < c["frozen"] < 4.1e9, c["frozen"]                       # a 4B-8B pretrained backbone
    assert 10e6 <= c["adapters"] <= 20e6, c["adapters"]                   # 10-20M semantic adapters
    assert c["adapters_lora"] == 11_796_480 and len(qwen.backbone.lora_modules) == 72     # q/v of all 36 layers
    assert 45e6 < c["experiential"] < 55e6, c["experiential"]             # ~50M experiential
    assert 60e6 < c["total_persistent_trainable"] < 70e6
    assert c["scale"] == "50m" and qwen.d_sem == 3 * 2560 and qwen.d_model == 768
    groups = qwen.param_groups()
    ids = [id(p) for ps in groups.values() for p in ps]
    assert len(ids) == len(set(ids)) and set(ids) == {id(p) for p in qwen.parameters()}
    assert next(iter(groups[GROUP_BACKBONE_FROZEN])).dtype == torch.bfloat16
    assert all(p.dtype == torch.float32 for p in groups[GROUP_ADAPTERS] + groups[GROUP_EXPERIENTIAL])
    _frozen_untouched(qwen)


def test_semantic_embeddings_are_meaningful_and_cached(qwen):
    texts = ["The page loaded without errors.", "The page loaded with no errors.", "def f(x):\n    return x + 1",
             "HTTP 404 Not Found", "Click the submit button."]
    e = qwen.embed_texts(texts)
    assert e.shape == (5, 7680) and e.dtype == torch.float32 and torch.isfinite(e).all()
    cos = torch.nn.functional.cosine_similarity(e[0], e[1:], -1)
    assert float(cos[0]) > 0.8 and float(cos[1:].max()) < 0.7            # a paraphrase is close, other texts are not
    hits = qwen.backbone.stats["cache_hits"]
    assert torch.equal(qwen.embed_texts(texts), e) and qwen.backbone.stats["cache_hits"] == hits + 5


def test_bridge_is_an_exact_noop_at_init_on_the_real_backbone(qwen):
    prompts = [qwen.backbone.chat_prompt("Name one colour."), "The capital of France is"]
    lat = torch.randn(2, qwen.d_model, device=qwen.device, generator=torch.Generator(qwen.device).manual_seed(0))
    with torch.no_grad():
        base = qwen.backbone.next_token_logits(prompts)
        cond = qwen.backbone.next_token_logits(prompts, latent=lat)
        lp0 = qwen.generation_logprob(prompts, ["Blue", " Paris"])
        lp1 = qwen.generation_logprob(prompts, ["Blue", " Paris"], latent=lat)
    assert torch.equal(base, cond) and torch.equal(lp0, lp1)
    assert float(lp0[1]) > -2.0                                           # the pretrained model knows the answer
    assert qwen.generate(prompts, max_new_tokens=8, temperature=0.0) == \
        qwen.generate(prompts, latent=lat, max_new_tokens=8, temperature=0.0)


def test_generate_returns_text(qwen):
    p = qwen.backbone.chat_prompt("Reply with exactly one word: hello")
    out = qwen.generate([p], max_new_tokens=12, temperature=0.0)
    assert isinstance(out[0][0], str) and "hello" in out[0][0].lower()
    lat = torch.randn(1, qwen.d_model, device=qwen.device)
    samples = qwen.generate([p], latent=lat, max_new_tokens=12, temperature=0.9, num_return_sequences=3, seed=1)
    assert len(samples) == 1 and len(samples[0]) == 3 and all(isinstance(s, str) and s for s in samples[0])
    assert samples == qwen.generate([p], latent=lat, max_new_tokens=12, temperature=0.9, num_return_sequences=3, seed=1)


def test_real_tokens_keep_numbers_and_the_latent_drives_decisions(qwen):
    """The two failures that only showed with a real backbone before: text swamping numbers (cos 0.98) and a
    decision path that ignored CAM's belief (KL 4e-9)."""
    from cam_live.nn.bench import init_diagnostics
    d = init_diagnostics(qwen)
    assert d["token_cos_plus1_minus1"] < 0.35, d
    assert d["token_cos_100_103"] < 0.9999 and d["token_cos_same_value_two_metrics"] < 0.9
    assert d["z_relative_change_under_permuted_belief"] > 0.3, d
    # at least 40 % of a change of the belief arrives in the decision latent (it was 0 % in the failed system)
    assert d["z_relative_change_under_permuted_belief"] > 0.4 * d["belief_relative_change_under_permutation"], d
    assert d["z_cos_between_two_actions"] < 0.9, d
    assert d["state_constant_fraction_after_one_call"] < 0.6 and 0.02 < d["slot_gate_mean"] < 0.3, d
    assert abs(d["phi_norm"] - 1.0) < 1e-4


def _hidden32_and_targets(bb, prompts, comps):
    P = [bb._prompt_ids(p) for p in prompts]
    C = [bb._encode(c) for c in comps]
    ids, attn = bb._pad([p + c for p, c in zip(P, C)])
    with torch.no_grad():
        h = bb._hidden(ids, attn)
    row = torch.tensor([i for i, c in enumerate(C) for _ in c], device=h.device)
    pos = torch.tensor([len(P[i]) - 1 + j for i, c in enumerate(C) for j in range(len(c))], device=h.device)
    tgt = torch.tensor([t for c in C for t in c], device=h.device)
    return h, row, pos, tgt


def test_generation_logprob_has_a_float32_output_layer(qwen):
    """The bf16 LM head quantised logits of size 16-64 to 1/8-1/4 (per-token log-probability errors up to 0.14
    nats on the same hidden states). Now: the final norm in float32, the head in float32."""
    bb = qwen.backbone
    prompts = [bb.chat_prompt(p) for p in ("Name one colour.", "Which tool should be used next? Answer with one word.",
                                           "Write a short Python function that adds two numbers.")]
    comps = ["Green", "browser", "def add(a, b):\n    return a + b"]
    h, row, pos, tgt = _hidden32_and_targets(bb, prompts, comps)
    assert h.dtype == torch.float32
    w = bb.lm.lm_head.weight
    with torch.no_grad():
        ref = torch.log_softmax(h[row, pos].double() @ w.double().T, -1).gather(1, tgt[:, None]).squeeze(1)
        got = bb._token_logprob(h[row, pos], tgt).double()
        old = torch.log_softmax(bb.lm.lm_head(h[row, pos].to(w.dtype)).float(), -1).gather(1, tgt[:, None]).squeeze(1)
        lp = qwen.generation_logprob(prompts, comps).double()
    err, err_old = float((got - ref).abs().max()), float((old.double() - ref).abs().max())
    assert err < 2e-3, err                                     # measured ~1e-4
    assert err_old > 10 * err, (err_old, err)                  # what the bf16 head gave on the same states
    assert torch.allclose(lp, torch.zeros(3, dtype=torch.float64, device=lp.device).index_add(0, row, ref), atol=5e-3)
    # the KV-cache sampler uses the same output layer: greedy tokens scored again by generation_logprob
    with torch.no_grad():
        P = bb._prompt_ids("The capital of France is")
        ids, attn = bb._pad([P], left=True)
        cache, cur, toks, lps = None, ids, [], []
        for _ in range(12):
            logits, cache = bb._step(cur, attn, cache)
            assert logits.dtype == torch.float32
            lpt = torch.log_softmax(logits, -1)
            nxt = lpt.argmax(-1)
            toks.append(int(nxt))
            lps.append(float(lpt[0, nxt]))
            attn = torch.cat([attn, torch.ones(1, 1, dtype=attn.dtype, device=attn.device)], 1)
            cur = nxt.unsqueeze(1)
        h2, row2, pos2, tgt2 = _hidden32_and_targets(bb, ["The capital of France is"], [bb._decode(toks)])
        full = bb._token_logprob(h2[row2, pos2], tgt2)
    if tgt2.tolist() == toks:                                   # decode/encode round trip kept the tokens
        gap = float((full.double().cpu() - torch.tensor(lps, dtype=torch.float64)).abs().max())
        assert gap < 0.1, gap                                   # remaining gap: bf16 trunk, cached vs full forward


def test_one_argument_changes_the_decision_latent_on_the_real_backbone(qwen):
    """cos(z) for browser.click {"ref": "e3"} vs {"ref": "e4"} was 0.9997 on Qwen3-4B (two operations: 0.82)."""
    from cam_live.contracts import ActionSpec
    cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a, b, -1).mean())
    ctx = [[ExperienceItem(kind=ItemKind.OBSERVATION, text="A checkout form with three empty fields and a disabled "
                           "submit button.", ts=1000.0, quantities=[Quantity("fields", 3.0)])],
           [ExperienceItem(kind=ItemKind.FAILURE, text="timeout transient_network: the request did not complete", ts=200.0)]]
    acts = [ActionSpec("a1", "browser.click", arguments={"ref": "e3"}), ActionSpec("a2", "browser.click", arguments={"ref": "e4"}),
            ActionSpec("a3", "internal.wait", arguments={"seconds": 600}), ActionSpec("a4", "internal.wait", arguments={"seconds": 601}),
            ActionSpec("a5", "browser.type", arguments={"ref": "e7", "text": "hello"}),
            ActionSpec("a6", "browser.type", arguments={"ref": "e7", "text": "hellp"}),
            ActionSpec("a7", "sandbox.python", modality="code", arguments={"code": "print(1)"})]
    with torch.no_grad():
        o = qwen.core_forward(qwen.tokenize(ctx))
        e_a, _ = qwen.encode_actions([acts] * 2)
        z = qwen.decision_latent(o.b, e_a, qwen.encode_goals(["Reach the stated target within the time limit."] * 2))
    one_arg = {"ref": cos(z[:, 0], z[:, 1]), "seconds": cos(z[:, 2], z[:, 3]), "text": cos(z[:, 4], z[:, 5])}
    two_ops = cos(z[:, 0], z[:, 6])
    print("cos(z) one-argument changes", one_arg, "two operations", two_ops)
    assert max(one_arg.values()) < 0.95, one_arg
    assert two_ops < 0.95, two_ops


def test_embeddings_and_beliefs_do_not_depend_on_batch_composition(qwen):
    """bf16 kernels change their reduction order with the batch shape: a text embedded alone vs inside a batch of
    longer texts differed by 1.1 % and the cache kept whichever came first (b of the same context: 1.4 %)."""
    from cam_live.nn.bench import _passage
    bb = qwen.backbone
    t = "The export job finished after twelve minutes."
    others = [_passage(300, bb, 1), _passage(40, bb, 2), _passage(9, bb, 3)]
    with torch.no_grad():
        bb.mark_adapters_updated()
        alone = qwen.embed_texts([t])[0].clone()
        for batch in ([others[0], t, others[1]], [t] + others * 5, [others[2], t]):
            bb.mark_adapters_updated()
            e = qwen.embed_texts(batch)
            assert torch.equal(e[batch.index(t)], alone)
        ctx = [ExperienceItem(kind=ItemKind.OBSERVATION, text=t, ts=10.0, quantities=[Quantity("minutes", 12.0)]),
               ExperienceItem(kind=ItemKind.OUTCOME, text="rows exported", ts=800.0, quantities=[Quantity("rows", 18250.0)])]
        bb.mark_adapters_updated()
        b_alone = qwen.core_forward(qwen.tokenize([ctx])).b[0].clone()
        bb.mark_adapters_updated()
        far = [ExperienceItem(kind=ItemKind.DOCUMENT, text=o, ts=1.0) for o in others]
        b_batch = qwen.core_forward(qwen.tokenize([far, ctx, far[:1]])).b[1]
    assert float((b_alone - b_batch).norm() / b_alone.norm()) < 1e-5
    assert torch.equal(qwen.embed_texts([t], with_grad=True).detach()[0], alone)


def test_features_on_realistic_inputs_real_configuration(qwen):
    """Conditioning of phi on realistic pipeline inputs (2304 combinations, 128 features) - measured and bounded
    (the earlier test fed white noise). It is not well-conditioned at initialisation; the fast layer's float64
    Joseph updates stay exact on it (tests/nn/test_api.py)."""
    from nn_realistic_inputs import gram_stats, realistic_decisions
    phi = realistic_decisions(qwen)
    cond, pr = gram_stats(phi)
    per_q = [gram_stats(phi[:, :, q]) for q in range(phi.shape[2])]
    print("phi Gram, real configuration: all", (cond, pr), "per query", per_q)
    assert torch.isfinite(phi).all() and cond < 1e6 and pr > 3.0, (cond, pr)
    assert max(c for c, _ in per_q) < 1e8, per_q


def test_adapter_caches_follow_writes_through_data(qwen):
    bb = qwen.backbone
    texts = ["The page loaded without errors.", "Click the submit button."]
    prompts = ["The capital of France is"]
    with torch.no_grad():
        e0 = qwen.embed_texts(texts).clone()
        l0 = bb.next_token_logits(prompts).clone()
    g = torch.Generator(qwen.device).manual_seed(5)
    for m in bb.lora_modules:
        m.lora_B.data.copy_(torch.randn(m.lora_B.shape, device=qwen.device, generator=g) * 2e-3)   # no version bump
    try:
        with torch.no_grad():
            e1 = qwen.embed_texts(texts)
            l1 = bb.next_token_logits(prompts)
        l1g = bb.next_token_logits(prompts).detach()
        assert not torch.equal(e1, e0) and torch.equal(e1, qwen.embed_texts(texts, with_grad=True).detach())
        assert not torch.equal(l1, l0) and float((l1 - l1g).abs().max()) < 1e-3
    finally:
        for m in bb.lora_modules:
            m.lora_B.data.zero_()
    with torch.no_grad():
        assert torch.equal(qwen.embed_texts(texts), e0) and torch.equal(bb.next_token_logits(prompts), l0)


def test_adapters_get_gradient_through_both_paths_and_the_bridge_conditions(qwen):
    """Runs last: it takes one optimisation step on the adapters."""
    bb = qwen.backbone
    lora = [(m.lora_A, m.lora_B) for m in bb.lora_modules]
    # (1) embedding path through a downstream loss: tokens -> experiential core -> belief
    ctx = [[ExperienceItem(kind=ItemKind.OBSERVATION, text="The form was submitted and a confirmation appeared.",
                           ts=10.0, quantities=[Quantity("visits", 100.0)]),
            ExperienceItem(kind=ItemKind.OUTCOME, text="three sign-ups were recorded", ts=3600.0,
                           quantities=[Quantity("signups", 3.0)])]]
    tb = qwen.tokenize(ctx, with_grad=True)
    assert tb.values[0, :, 0].tolist() == [100.0, 3.0]
    b = qwen.core_forward(tb).b
    qwen.zero_grad()
    b.pow(2).sum().backward()
    n_b = sum(float(B.grad.abs().sum()) > 0 for _, B in lora)
    assert n_b == len(lora), f"{n_b}/{len(lora)} LoRA modules got gradient through the embedding path"
    g_embed = qwen.grad_norms()
    assert g_embed[GROUP_ADAPTERS] > 0 and g_embed[GROUP_EXPERIENTIAL] > 0
    _frozen_untouched(qwen)
    # (2) generation path
    qwen.zero_grad()
    prompts = [bb.chat_prompt("Name one colour.")]
    lat = torch.randn(1, qwen.d_model, device=qwen.device, generator=torch.Generator(qwen.device).manual_seed(1))
    lat.requires_grad_(True)
    with torch.no_grad():
        p_before = torch.softmax(bb.next_token_logits(prompts, latent=lat), -1)
        p_base_before = torch.softmax(bb.next_token_logits(prompts), -1)
    lp = qwen.generation_logprob(prompts, ["Green"], latent=lat)
    (-lp.sum()).backward()
    assert all(float(B.grad.abs().sum()) > 0 for _, B in lora)
    assert all(float(x.out.weight.grad.abs().sum()) > 0 for x in bb.bridge.xattn)
    _frozen_untouched(qwen)
    # (3) one optimisation step: the latent now changes generation probabilities, and gets gradient itself
    opt = torch.optim.Adam(qwen.param_groups()[GROUP_ADAPTERS], lr=1e-3)
    opt.step()
    qwen.zero_grad()
    with torch.no_grad():
        p_after = torch.softmax(bb.next_token_logits(prompts, latent=lat), -1)
        p_base_after = torch.softmax(bb.next_token_logits(prompts), -1)
        other = torch.softmax(bb.next_token_logits(prompts, latent=-lat.detach()), -1)
    assert float((p_after - p_before).abs().max()) > 1e-4
    assert float((p_after - p_base_after).abs().max()) > 1e-5          # conditioned differs from unconditioned
    assert float((p_after - other).abs().max()) > 1e-5                 # and different beliefs differ
    lp2 = qwen.generation_logprob(prompts, ["Green"], latent=lat)
    (-lp2.sum()).backward()
    assert float(lat.grad.abs().sum()) > 0
    assert all(float(A.grad.abs().sum()) > 0 for A, _ in lora)
    assert float(lp2.detach()) > float(lp.detach())
    _frozen_untouched(qwen)
    assert not bb.lm.training
