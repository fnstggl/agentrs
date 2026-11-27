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


def test_feature_gram_is_well_conditioned(model, items):
    """phi must be usable by a Bayesian last layer: no collapsed directions at init."""
    g = torch.Generator().manual_seed(0)
    z = torch.randn(64, 8, model.d_model, generator=g)
    e_q = torch.randn(64, 4, model.d_model, generator=g)
    phi = model.features(z, e_q).reshape(-1, model.n_features).double()
    ev = torch.linalg.eigvalsh(phi.T @ phi / phi.shape[0])
    assert float(ev.max() / ev.min()) < 1e4, float(ev.max() / ev.min())


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
