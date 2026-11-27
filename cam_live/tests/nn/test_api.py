"""CamLiveModel against the tensor contract nn/api.py: shapes, groups, counts, hashes, persistence (NA-04, NA-05)."""
import copy
import json

import pytest
import torch
import torch.nn.functional as Fn
from torch import nn

from cam_live.config import CamLiveConfig
from cam_live.contracts import (ActionSpec, CandidateEvaluation, Decision, ExperienceRecord, OutcomeKind,
                                OutcomeQuery)
from cam_live.learning_api import encode_decisions
from cam_live.nn.api import (GROUP_ADAPTERS, GROUP_BACKBONE_FROZEN, GROUP_EXPERIENTIAL, GROUP_HEADS, GROUP_REWARD,
                             GROUPS, KIND_INDEX, CoreOutput, CoreState, LatentModelAPI, TokenBatch)
from cam_live.nn.model import CamLiveModel, build_model
from nn_realistic_inputs import gram_stats as _gram_stats, realistic_decisions as _realistic_decisions


def _acts():
    return [[ActionSpec("a1", "browser.click", arguments={"ref": 3}), ActionSpec("a2", "internal.wait"),
             ActionSpec("a3", "sandbox.python", modality="code", level="procedure")],
            [ActionSpec("b1", "lab.probe", arguments={"channel": 2})]]


def _queries():
    return [[OutcomeQuery("q1", "members.count", kind=OutcomeKind.COUNT, horizon_s=86400.0, exposure=2.0)],
            [OutcomeQuery("q2", "latency", horizon_s=30.0, description="seconds until done"),
             OutcomeQuery("q3", "choice", kind=OutcomeKind.CATEGORICAL, categories=["a", "b"])]]


def _pipeline(model, items):
    tb = model.tokenize([items(6, tag="x"), items(3, tag="y")])
    out = model.core_forward(tb)
    e_a, a_mask = model.encode_actions(_acts())
    e_g = model.encode_goals(["reach the stated target", "a different objective"])
    e_q, q_mask, kind_ids, horizon = model.encode_queries(_queries())
    z = model.decision_latent(out.b, e_a, e_g)
    phi = model.features(z, e_q)
    return tb, out, e_a, a_mask, e_g, e_q, q_mask, kind_ids, horizon, z, phi


def test_api_shapes_and_dtypes(model, items):
    assert isinstance(model, LatentModelAPI) and isinstance(model, CamLiveModel)
    D, F, dev = model.d_model, model.n_features, model.device
    tb, out, e_a, a_mask, e_g, e_q, q_mask, kind_ids, horizon, z, phi = _pipeline(model, items)
    assert isinstance(tb, TokenBatch) and isinstance(out, CoreOutput) and isinstance(out.state, CoreState)
    assert tb.x.shape == (2, 6, D) and tb.pad_mask.dtype == torch.bool and tb.pad_mask.sum(1).tolist() == [6, 3]
    assert tb.values.shape == (2, 6, model.core_cfg.max_values) and tb.values.dtype == torch.float64
    assert tb.value_mask.dtype == torch.bool and tb.time.dtype == torch.float64 and tb.kind_ids.dtype == torch.long
    assert len(tb.refs) == 2 and tb.refs[0][:2] == ["r0", "r1"] and tb.refs[1][3:] == ["", "", ""]
    assert out.h.shape == (2, 6, D) and out.b.shape == (2, D)
    assert out.state.slots.shape == (2, model.core_cfg.n_slots, D) and out.state.last_time.dtype == torch.float64
    assert e_a.shape == (2, 3, D) and a_mask.tolist() == [[True, True, True], [True, False, False]]
    assert e_g.shape == (2, D) and e_q.shape == (2, 2, D) and q_mask.tolist() == [[True, False], [True, True]]
    assert kind_ids.dtype == torch.long and kind_ids[1].tolist() == [KIND_INDEX["real"], KIND_INDEX["categorical"]]
    assert horizon.dtype == torch.float64 and horizon[0, 0].item() == 86400.0
    assert z.shape == (2, 3, D) and phi.shape == (2, 3, 2, F)
    for t in (tb.x, out.h, out.b, out.state.slots, e_a, e_g, e_q, z, phi):
        assert t.dtype == torch.float32 and t.device == dev and torch.isfinite(t).all()
    # features: unit norm, bounded, constant last coordinate
    assert torch.allclose(phi.norm(dim=-1), torch.ones(2, 3, 2), atol=1e-4) and float(phi.abs().max()) < 1.0
    assert torch.allclose(phi[..., -1], torch.full((2, 3, 2), F ** -0.5))
    st = model.init_state(4)
    assert st.slots.shape == (4, model.core_cfg.n_slots, D) and st.step == 0 and torch.isnan(st.last_time).all()
    assert model.embed_texts(["a", "b c"]).shape == (2, model.d_sem)
    # the fast state consumes phi directly
    fs = model.new_fast_state()
    mean, epi, noise = fs.predict(phi, OutcomeKind.COUNT)
    assert mean.shape == epi.shape == noise.shape == (2, 3, 2) and mean.dtype == torch.float32
    info = fs.update(phi[0, 0, 0], 4.0, OutcomeKind.COUNT, exposure=2.0)
    assert info["kl"] > 0 and fs.n_updates(OutcomeKind.COUNT) == 1


def test_features_on_realistic_inputs_conditioning_and_fast_layer_numerics(model):
    """Measured, not assumed: on realistic pipeline inputs phi is NOT well-conditioned at initialisation (the
    earlier test fed white noise: max / min eigenvalue ~10). Debug model, 2304 combinations: max / min ~70 over all
    queries, ~0.6-2.2e3 for one query (participation ratio 2-3.5 of 16); the real configuration measures ~1e4-1e6
    (docs/status/A.json). What must hold is that the Bayesian last layer stays exact on such features."""
    phi = _realistic_decisions(model)
    cond_all, pr_all = _gram_stats(phi)
    per_q = [_gram_stats(phi[:, :, q]) for q in range(phi.shape[2])]
    assert cond_all < 500 and pr_all > 3.0, (cond_all, pr_all)          # regression guard (measured ~70 / 6.4)
    assert max(c for c, _ in per_q) < 2e4, per_q                       # measured <= 2.2e3
    # the fast layer on these features: 2304 sequential Joseph updates == the closed-form posterior, SPD
    from cam_live.nn import fast
    from cam_live.nn.fast import F64, FastBeliefState
    P = phi.reshape(-1, phi.shape[-1]).double()
    g = torch.Generator().manual_seed(1)
    y = P @ torch.randn(P.shape[1], generator=g, dtype=F64) + 0.3 * torch.randn(P.shape[0], generator=g, dtype=F64)
    prior = model.exp.fast_prior
    st = FastBeliefState(prior, change_h=0.0, forgetting=1.0)
    st.noise_mode = "fixed"                          # known-noise recursion against the closed form
    p0 = {k: v.detach() for k, v in prior.prior(OutcomeKind.REAL).items()}
    st._post[OutcomeKind.REAL].update(mu=p0["mu"].clone(), cov=p0["cov"].clone())
    st.update(P, y, OutcomeKind.REAL)
    mu_ref, cov_ref = fast.gaussian_fit(P, y, None, p0["mu"], p0["cov"], p0["noise"])
    post = st.posterior(OutcomeKind.REAL)
    assert torch.allclose(post["mu"], mu_ref, atol=1e-8) and torch.allclose(post["cov"], cov_ref, atol=1e-10)
    assert float(torch.linalg.eigvalsh(post["cov"]).min()) > 0 and torch.equal(post["cov"], post["cov"].T)


def test_candidates_differing_in_one_argument_are_distinguishable(model, items):
    """An action was the pooled embedding of its rendered text: one changed argument token left z almost unchanged
    (cos 0.9997 on Qwen3-4B for {"ref": "e3"} vs {"ref": "e4"}). Arguments and the target now reach the encoding
    through the exact value channel."""
    b = model.core_forward(model.tokenize([items(4)])).b
    e_g = model.encode_goals(["reach the stated target"])

    def z_of(acts):
        e_a, _ = model.encode_actions([acts])
        return model.decision_latent(b, e_a, e_g)[0]

    cos = lambda u, v: float(Fn.cosine_similarity(u, v, -1))
    pairs = {"ref e3 / e4": (ActionSpec("a", "browser.click", arguments={"ref": "e3"}),
                             ActionSpec("b", "browser.click", arguments={"ref": "e4"})),
             "seconds 600 / 601": (ActionSpec("a", "internal.wait", arguments={"seconds": 600}),
                                   ActionSpec("b", "internal.wait", arguments={"seconds": 601})),
             "text hello / hellp": (ActionSpec("a", "browser.type", arguments={"ref": "e7", "text": "hello"}),
                                    ActionSpec("b", "browser.type", arguments={"ref": "e7", "text": "hellp"})),
             "nested rows 100 / 200": (ActionSpec("a", "api.call", arguments={"q": {"rows": 100, "f": "csv"}}),
                                       ActionSpec("b", "api.call", arguments={"q": {"rows": 200, "f": "csv"}})),
             "target e3 / e4": (ActionSpec("a", "browser.click", target="e3"), ActionSpec("b", "browser.click", target="e4"))}
    got = {}
    for name, (a1, a2) in pairs.items():
        z = z_of([a1, a2])
        got[name] = cos(z[0], z[1])
    # measured on the debug model: 0.69-0.88 for a one-leaf argument / target change; 0.95 when one of three leaves
    # changes (a bag of leaves: one change is a third of the argument vector)
    assert max(v for k, v in got.items() if not k.startswith("nested")) < 0.92, got
    assert got["nested rows 100 / 200"] < 0.98, got
    # identical candidates are identical, key order is irrelevant, other candidates do not matter
    z = z_of([ActionSpec("a", "api.call", arguments={"x": 1, "y": "u"}), ActionSpec("b", "api.call", arguments={"y": "u", "x": 1}),
              ActionSpec("c", "internal.wait", arguments={"seconds": 5})])
    assert torch.allclose(z[0], z[1], atol=1e-6)
    z_alone = z_of([ActionSpec("c", "internal.wait", arguments={"seconds": 5})])
    assert torch.allclose(z[2], z_alone[0], atol=1e-6)
    # what reaches the value channel: exact numbers (smooth in the value) and a hashed identity of every leaf
    from cam_live.nn.tokens import argument_quantities
    q = argument_quantities(ActionSpec("a", "api.call", target="e9",
                                       arguments={"b": {"n": 600, "on": True}, "a": [1.5, None], "s": "x"}))
    assert [(x.name, x.value) for x in q] == [
        ("arg.target=\"e9\"", 1.0), ("arg.a[0]", 1.5), ("arg.a[0]=1.5", 1.0), ("arg.a[1]=null", 1.0),
        ("arg.b.n", 600.0), ("arg.b.n=600.0", 1.0), ("arg.b.on=true", 1.0), ("arg.s=\"x\"", 1.0)]
    # more than MAX_ARGUMENTS leaves are dropped and counted
    from cam_live.nn.tokens import MAX_ARGUMENTS
    before = model.exp.tokenizer.stats["dropped_arguments"]
    model.encode_actions([[ActionSpec("a", "api.call", arguments={f"k{i}": i for i in range(MAX_ARGUMENTS + 3)})]])
    assert model.exp.tokenizer.stats["dropped_arguments"] == before + 3


def test_arch_signature_covers_every_behaviour_relevant_setting():
    """Same-shape configurations that compute different functions must have different signatures (a state saved
    with max_text_tokens 48 loaded under strict=True into a model with 8, signature unchanged, outputs different)."""
    base = build_model(CamLiveConfig.debug()).arch_signature()
    assert json.loads(json.dumps(base)) == base
    for name, mut in (("max_text_tokens", lambda c: setattr(c.backbone, "max_text_tokens", 8)),
                      ("lora_alpha", lambda c: setattr(c.backbone, "lora_alpha", 8.0)),
                      ("tap_layers", lambda c: setattr(c.backbone, "tap_layers", (0.5, 1.0, 1.0))),
                      ("tiny LM seed (frozen weights)", lambda c: setattr(c, "seed", 1)),
                      ("dropout", lambda c: setattr(c.core, "dropout", 0.1)),
                      ("max_prompt_tokens", lambda c: c.sections.setdefault("nn", {}).update(max_prompt_tokens=64)),
                      ("embedding batch shape", lambda c: c.sections.setdefault("nn", {}).update(embed_batch_tokens=64))):
        c = CamLiveConfig.debug()
        mut(c)
        assert build_model(c).arch_signature() != base, name
    for name, mut in (("cache size", lambda c: setattr(c.backbone, "embed_cache_items", 7)),
                      ("checkpointing", lambda c: setattr(c.backbone, "grad_checkpoint", False))):
        c = CamLiveConfig.debug()
        mut(c)
        assert build_model(c).arch_signature() == base, name


def test_permuted_belief_changes_the_decision_latent(model, items):
    """An earlier policy ignored CAM's latent (KL 4e-9 under a permuted belief)."""
    ctxs = [items(4, tag=t, t0=1000.0 * (i + 1)) for i, t in enumerate(("alpha", "beta", "gamma", "delta"))]
    b = model.core_forward(model.tokenize(ctxs)).b
    acts = [_acts()[0]] * 4
    e_a, _ = model.encode_actions(acts)
    e_g = model.encode_goals(["reach the stated target"] * 4)
    e_q = model.encode_queries([_queries()[0]] * 4)[0]
    z = model.decision_latent(b, e_a, e_g)
    perm = torch.tensor([1, 2, 3, 0])
    z_p = model.decision_latent(b[perm], e_a, e_g)
    cos = Fn.cosine_similarity(z, z_p, -1)
    assert float(cos.mean()) < 0.9, float(cos.mean())
    rel = float((z - z_p).norm() / z.norm())
    assert rel > 0.3, rel
    phi, phi_p = model.features(z, e_q), model.features(z_p, e_q)
    assert float((phi - phi_p).norm() / phi.norm()) > 0.2
    # a linear policy head on z therefore changes its action distribution
    head = nn.Linear(model.d_model, 1)
    kl = Fn.kl_div(torch.log_softmax(head(z_p).squeeze(-1), -1), torch.softmax(head(z).squeeze(-1), -1),
                   reduction="batchmean")
    assert float(kl) > 1e-3
    # and z also depends on the action and on the goal
    assert float(Fn.cosine_similarity(z[:, 0], z[:, 2], -1).mean()) < 0.8
    assert not torch.allclose(z[:, 0], z[:, 1], atol=1e-2)
    z_g = model.decision_latent(b, e_a, model.encode_goals(["something unrelated entirely"] * 4))
    assert float((z - z_g).norm() / z.norm()) > 0.1


def test_parameter_groups_are_disjoint_and_cover_everything(model):
    model.register_head("world", nn.Linear(model.d_model, 7))
    model.register_head("reward_fn", nn.Sequential(nn.Linear(model.d_model, 8), nn.Tanh(), nn.Linear(8, 1)),
                        GROUP_REWARD)
    groups = model.param_groups()
    assert tuple(groups) == GROUPS
    ids = [id(p) for ps in groups.values() for p in ps]
    assert len(ids) == len(set(ids))                                           # disjoint
    assert set(ids) == {id(p) for p in model.parameters()}                     # cover every parameter
    assert all(not p.requires_grad for p in groups[GROUP_BACKBONE_FROZEN]) and groups[GROUP_BACKBONE_FROZEN]
    for g in (GROUP_ADAPTERS, GROUP_EXPERIENTIAL, GROUP_REWARD, GROUP_HEADS):
        assert groups[g] and all(p.requires_grad for p in groups[g])
    c = model.count_parameters()
    n = lambda g: sum(p.numel() for p in groups[g])
    assert c["frozen"] == n(GROUP_BACKBONE_FROZEN) and c["experiential"] == n(GROUP_EXPERIENTIAL)
    assert c["adapters"] == n(GROUP_ADAPTERS) == c["adapters_lora"] + c["adapters_bridge"]
    assert c["reward"] == n(GROUP_REWARD) == 8 * model.d_model + 8 + 9
    assert c["heads"] == {"world": 7 * model.d_model + 7, "reward_fn": c["reward"]} and c["heads_total"] == c["heads"]["world"]
    assert c["total_persistent_trainable"] == c["experiential"] + c["adapters"] + c["reward"] + c["heads_total"]
    assert c["frozen"] + c["total_persistent_trainable"] == sum(p.numel() for p in model.parameters())
    assert sum(c["experiential_breakdown"].values()) == c["experiential"]
    assert model.head("world").out_features == 7 and model.head_names(GROUP_REWARD) == ["reward_fn"]
    with pytest.raises(ValueError):
        model.register_head("bad", nn.Linear(2, 2), GROUP_BACKBONE_FROZEN)
    with pytest.raises(ValueError):
        model.register_head("world", nn.Linear(2, 2))
    with pytest.raises(ValueError):
        model.register_head("a.b", nn.Linear(2, 2))
    sig = model.arch_signature()
    assert json.loads(json.dumps(sig)) == sig and sig["scale"] == "debug" and sig["heads"]["reward_fn"]["group"] == "reward"
    assert sig["backbone"]["name"] == "tiny" and sig["core"]["d_model"] == model.d_model


def _step_all(model, lr=0.05):
    """One SGD step with a synthetic gradient on every trainable parameter."""
    g = torch.Generator().manual_seed(42)
    params = [p for name, ps in model.param_groups().items() if name != GROUP_BACKBONE_FROZEN for p in ps]
    opt = torch.optim.SGD(params, lr=lr)
    for p in params:
        p.grad = torch.randn(p.shape, generator=g).to(p.device)
    opt.step()
    model.zero_grad()


def test_group_hashes_change_after_an_optimizer_step(model, items):
    model.register_head("world", nn.Linear(model.d_model, 3))
    model.register_head("reward_fn", nn.Linear(model.d_model, 1), GROUP_REWARD)
    h0 = model.group_hashes()
    assert set(h0) == set(GROUPS) and h0 == model.group_hashes()
    # a real loss that touches only the experiential group and the world head
    out = model.core_forward(model.tokenize([items(4)]))
    loss = model.head("world")(out.b).pow(2).sum()
    params = model.param_groups()[GROUP_EXPERIENTIAL] + model.param_groups()[GROUP_HEADS]
    opt = torch.optim.Adam(params, lr=1e-3)
    opt.zero_grad()
    loss.backward()
    norms = model.clip_gradients({GROUP_EXPERIENTIAL: 1.0, GROUP_HEADS: 0.5})
    assert norms[GROUP_EXPERIENTIAL] > 0 and model.grad_norms()[GROUP_EXPERIENTIAL] <= 1.0 + 1e-4
    assert model.grad_norms()[GROUP_HEADS] <= 0.5 + 1e-4
    opt.step()
    h1 = model.group_hashes()
    assert h1[GROUP_EXPERIENTIAL] != h0[GROUP_EXPERIENTIAL] and h1[GROUP_HEADS] != h0[GROUP_HEADS]
    assert h1[GROUP_ADAPTERS] == h0[GROUP_ADAPTERS] and h1[GROUP_REWARD] == h0[GROUP_REWARD]
    assert h1[GROUP_BACKBONE_FROZEN] == h0[GROUP_BACKBONE_FROZEN]
    _step_all(model)
    h2 = model.group_hashes()
    assert all(h2[g] != h1[g] for g in (GROUP_ADAPTERS, GROUP_EXPERIENTIAL, GROUP_REWARD, GROUP_HEADS))


def test_trainable_state_round_trips_bit_exactly(cfg, items):
    a = build_model(cfg)
    a.register_head("world", nn.Linear(a.d_model, 3))
    _step_all(a)                                                   # move away from the initial weights
    state = a.trainable_state()
    assert not any(k.startswith("backbone.lm.") and "lora_" not in k for k in state)     # no frozen weights
    assert any("lora_A" in k for k in state) and any(k.startswith("backbone.bridge.") for k in state)
    assert any(k.startswith("exp.core.") for k in state) and "heads.world.weight" in state
    assert "backbone.cal_mu" in state and "exp.fast_prior.mu0" in state
    pnames = {n for n, _ in a.named_parameters()}
    assert sum(v.numel() for k, v in state.items() if k in pnames) == a.count_parameters()["total_persistent_trainable"]
    assert {k for k in state if k not in pnames} == {"backbone.cal_mu", "backbone.cal_sd", "backbone.bridge.h_mu",
                                                     "backbone.bridge.h_sd", "backbone.bridge.h_scale"}
    hashes = a.group_hashes()
    ref = _pipeline(a, items)
    lat = torch.randn(1, a.d_model, generator=torch.Generator().manual_seed(3))
    lp_ref = a.generation_logprob(["a prompt"], ["and its completion"], latent=lat)

    b = build_model(copy.deepcopy(cfg))                            # same frozen backbone, other trainable weights
    b.register_head("world", nn.Linear(b.d_model, 3))
    _step_all(b, lr=0.3)
    assert all(b.group_hashes()[g] != hashes[g] for g in (GROUP_EXPERIENTIAL, GROUP_ADAPTERS, GROUP_HEADS))
    rep = b.load_trainable_state(state, strict=True)
    assert rep == {"missing": [], "unexpected": [], "mismatched": []}
    back = b.trainable_state()
    assert back.keys() == state.keys() and all(torch.equal(back[k], state[k]) for k in state)
    assert b.group_hashes() == hashes
    out = _pipeline(b, items)
    assert torch.equal(out[1].b, ref[1].b) and torch.equal(out[9], ref[9]) and torch.equal(out[10], ref[10])
    assert torch.equal(b.generation_logprob(["a prompt"], ["and its completion"], latent=lat), lp_ref)
    # through torch.save / torch.load (weights only)
    import io
    buf = io.BytesIO()
    torch.save(state, buf)
    buf.seek(0)
    again = torch.load(buf, weights_only=True)
    assert all(torch.equal(again[k], state[k]) for k in state)


_FRESH = r"""
import sys, torch
from torch import nn
from cam_live.config import CamLiveConfig
from cam_live.contracts import ActionSpec, ExperienceItem, ItemKind, OutcomeKind, OutcomeQuery, Quantity
from cam_live.nn.model import build_model
torch.manual_seed(int(sys.argv[3]))                  # a different global RNG state in every process
m = build_model(CamLiveConfig.debug())
m.register_head("world", nn.Linear(m.d_model, 3))
if sys.argv[1] == "load":
    m.load_trainable_state(torch.load(sys.argv[2] + "/state.pt", weights_only=True), strict=True)
else:
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    for _ in range(2):
        opt.zero_grad()
        o = m.core_forward(m.tokenize([[ExperienceItem(kind=ItemKind.OBSERVATION, text="the page loaded", ts=1.0)]], with_grad=True))
        (o.b.pow(2).sum() + m.head("world")(o.b).sum()
         - m.generation_logprob(["a prompt"], ["its completion"], latent=o.b).sum()).backward()
        opt.step()
    torch.save(m.trainable_state(), sys.argv[2] + "/state.pt")
ctx = [[ExperienceItem(kind=ItemKind.OBSERVATION, text="the page loaded", ts=10.0, quantities=[Quantity("visits", 100.0)]),
        ExperienceItem(kind=ItemKind.OUTCOME, text="three sign ups", ts=3600.0, quantities=[Quantity("signups", 3.0)])]]
with torch.no_grad():
    o = m.core_forward(m.tokenize(ctx))
    e_a, _ = m.encode_actions([[ActionSpec("a", "browser.click", arguments={"ref": "e3"}), ActionSpec("b", "internal.wait")]])
    z = m.decision_latent(o.b, e_a, m.encode_goals(["goal"]))
    phi = m.features(z, m.encode_queries([[OutcomeQuery("q", "signups", kind=OutcomeKind.COUNT, horizon_s=60.0)]])[0])
    lp = m.generation_logprob(["a prompt"], ["its completion"], latent=o.b)
    txt = m.generate(["a prompt"], latent=o.b, max_new_tokens=5, seed=3)
torch.save({"b": o.b, "z": z, "phi": phi, "lp": lp, "head": m.head("world")(o.b), "text": txt,
            "hashes": m.group_hashes(), "sig": m.arch_signature()}, sys.argv[2] + f"/out_{sys.argv[1]}.pt")
"""


def test_trainable_state_round_trips_into_a_fresh_process(tmp_path):
    """Save in one process, load in another (different global RNG, fresh caches): every output bit-identical."""
    import os
    import subprocess
    import sys
    import cam_live
    script = tmp_path / "fresh.py"
    script.write_text(_FRESH)
    env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(cam_live.__file__)))
    for mode, seed in (("save", "0"), ("load", "123")):
        r = subprocess.run([sys.executable, str(script), mode, str(tmp_path), seed], env=env, capture_output=True,
                           text=True, timeout=600)
        assert r.returncode == 0, r.stderr[-2000:]
    a, b = (torch.load(tmp_path / f"out_{m}.pt", weights_only=False) for m in ("save", "load"))
    assert a["hashes"] == b["hashes"] and a["sig"] == b["sig"] and a["text"] == b["text"]
    for k in ("b", "z", "phi", "lp", "head"):
        assert torch.equal(a[k], b[k]), k


def test_tolerant_loading_reports_new_heads(cfg):
    a = build_model(cfg)
    a.register_head("world", nn.Linear(a.d_model, 3))
    state = a.trainable_state()
    b = build_model(cfg)
    b.register_head("world", nn.Linear(b.d_model, 3))
    b.register_head("credit", nn.Linear(b.d_model, 2))            # a head added after the checkpoint was written
    before = {k: v.clone() for k, v in b.head("credit").state_dict().items()}
    with pytest.raises(RuntimeError):
        b.load_trainable_state(state, strict=True)
    rep = b.load_trainable_state(state, strict=False)
    assert rep["missing"] == ["heads.credit.bias", "heads.credit.weight"] and rep["unexpected"] == []
    assert torch.equal(b.head("world").weight, a.head("world").weight)
    assert all(torch.equal(v, before[k]) for k, v in b.head("credit").state_dict().items())   # left at its init
    # unexpected and mismatching keys are reported, never silently loaded
    state2 = dict(state)
    state2["heads.gone.weight"] = torch.zeros(1)
    state2["heads.world.weight"] = torch.zeros(5, a.d_model)
    rep = b.load_trainable_state(state2, strict=False)
    assert rep["unexpected"] == ["heads.gone.weight"] and rep["mismatched"] == ["heads.world.weight"]
    assert torch.equal(b.head("world").weight, a.head("world").weight)


class _MemoryStore:
    """Item lookup for the shared ``encode_decisions`` (the real store is Workstream D's)."""

    def __init__(self):
        self._items = {}

    def put_item(self, item):
        self._items[item.ref] = item

    def items(self, refs):
        return [self._items[r] for r in refs]


def test_shared_encode_decisions_runs_on_the_model(model, items):
    store = _MemoryStore()
    its = items(6)
    for it in its:
        store.put_item(it)
    acts = _acts()[0]
    dec = Decision("d1", 2000.0, "obj", acts[1], candidates=[CandidateEvaluation(a) for a in acts],
                   probabilities=[0.2, 0.5, 0.3], chosen_index=1)
    rec = ExperienceRecord("x1", "obj", dec, context_item_refs=[i.ref for i in its[:4]],
                           post_item_refs=[i.ref for i in its], goal_text="reach the stated target", duration_s=30.0)
    rec2 = ExperienceRecord("x2", "obj", Decision("d2", 2100.0, "obj", acts[0]),
                            context_item_refs=[i.ref for i in its[:2]], goal_text="reach the stated target")
    batch = encode_decisions(model, store, [rec, rec2], with_next=True)
    D = model.d_model
    assert batch.b.shape == (2, D) and batch.z.shape == (2, 3, D) and batch.z_chosen.shape == (2, D)
    assert batch.a_mask.tolist() == [[True] * 3, [True, False, False]] and batch.chosen.tolist() == [1, 0]
    assert torch.allclose(batch.propensity, torch.tensor([0.5, 1.0])) and batch.has_next.tolist() == [True, False]
    assert batch.b_next.shape == (2, D) and not torch.allclose(batch.b_next[0], batch.b[0])
    assert batch.b.requires_grad
    slow = encode_decisions(model, store, [rec], with_grad_semantic=True)
    slow.z.sum().backward()
    assert model.grad_norms()[GROUP_ADAPTERS] > 0
    assert not encode_decisions(model, store, [rec], grad=False).b.requires_grad


def test_experiential_weights_change_decisions_with_the_backbone_fixed(model, items):
    """Spec 1.4: changing CAM's experiential weights changes its decision latents while the pretrained weights and
    the semantic embeddings stay exactly as they were."""
    texts = ["the page loaded", "click the button"]
    emb0 = model.embed_texts(texts).clone()
    lp0 = model.generation_logprob(["a prompt"], ["a completion"]).detach().clone()
    before = _pipeline(model, items)
    state0 = model.trainable_state()
    norms0, hashes0 = model.group_norms(), model.group_hashes()
    g = torch.Generator().manual_seed(1)
    params = model.param_groups()[GROUP_EXPERIENTIAL]
    opt = torch.optim.SGD(params, lr=0.05)
    for p in params:
        p.grad = torch.randn(p.shape, generator=g)
    opt.step()
    model.zero_grad()
    after = _pipeline(model, items)
    assert float((after[9] - before[9]).norm() / before[9].norm()) > 0.05          # z moved
    assert float((after[10] - before[10]).norm() / before[10].norm()) > 0.05        # phi moved
    assert torch.equal(model.embed_texts(texts), emb0)                              # the semantic side did not
    assert torch.equal(model.generation_logprob(["a prompt"], ["a completion"]).detach(), lp0)
    hashes1 = model.group_hashes()
    assert hashes1[GROUP_BACKBONE_FROZEN] == hashes0[GROUP_BACKBONE_FROZEN]
    assert hashes1[GROUP_ADAPTERS] == hashes0[GROUP_ADAPTERS] and hashes1[GROUP_EXPERIENTIAL] != hashes0[GROUP_EXPERIENTIAL]
    delta = model.group_delta_norms(state0)
    assert delta[GROUP_EXPERIENTIAL] > 0 and delta[GROUP_ADAPTERS] == 0.0 and set(delta) == set(GROUPS) - {GROUP_BACKBONE_FROZEN}
    assert model.group_norms()[GROUP_EXPERIENTIAL] != norms0[GROUP_EXPERIENTIAL]
    assert model.group_norms()[GROUP_ADAPTERS] == norms0[GROUP_ADAPTERS]
