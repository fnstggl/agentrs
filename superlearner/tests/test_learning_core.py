"""Numerical correctness of the learning path: GAE, likelihood recomputation, PPO and BC updates."""
import numpy as np
import pytest
import torch

from superlearner.algo import rule as R
from superlearner.algo.base import make
from superlearner.algo.ppo import bc_update, flat_params
from superlearner.cam.agent import AgentConfig, CAMAgent, ModelChooser
from superlearner.cam.batch import collate, evaluate
from superlearner.cam.model import CAMController, EmbStore, ModelConfig
from superlearner.llm.server import EmbedOnly
from superlearner.world.core import CoreConfig, SessionCore
from superlearner.world.tasks import task

SMALL = ModelConfig(d=64, layers=2, heads=4, ff=128, bag_dim=32, hid=64)


def ref_gae(r, v, done, trunc, gamma, lam):
    T = len(r)
    adv = np.zeros(T)
    last = 0.0
    for t in range(T - 1, -1, -1):
        end = t == T - 1
        nv = 0.0 if (end and done[t]) else (v[t] if (end and trunc[t]) else (v[t + 1] if not end else 0.0))
        delta = r[t] + gamma * nv - v[t]
        last = delta + (0.0 if end else gamma * lam * last)
        adv[t] = last
    return adv


@pytest.mark.parametrize("lam", [0.0, 0.9, 1.0])
def test_gae_matches_a_reference_implementation_with_padding_termination_and_truncation(lam):
    rng = np.random.default_rng(0)
    lens, T = [5, 3, 7], 7
    r, v = rng.normal(size=(3, T)).astype(np.float32), rng.normal(size=(3, T)).astype(np.float32)
    done, trunc, valid = np.zeros((3, T), np.float32), np.zeros((3, T), np.float32), np.zeros((3, T), bool)
    for b, n in enumerate(lens):
        valid[b, :n] = True
    done[0, 4] = 1          # episode 0 terminates
    trunc[1, 2] = 1         # episode 1 is truncated (bootstraps from its own last value)
    done[2, 6] = 1
    cfg = R.RuleConfig(gamma=0.97, lam=lam)
    adv, ret = R.advantages(*(torch.tensor(x) for x in (r, v, done, trunc)), torch.tensor(valid), cfg)
    for b, n in enumerate(lens):
        want = ref_gae(r[b, :n], v[b, :n], done[b, :n], trunc[b, :n], 0.97, lam)
        assert np.allclose(adv[b, :n].numpy(), want, atol=1e-5)
        assert np.allclose(adv[b, n:].numpy(), 0.0)
        assert np.allclose(ret[b, :n].numpy(), want + v[b, :n], atol=1e-5)
    if lam == 1.0:   # lambda = 1: advantage = discounted Monte-Carlo return - value (terminated episode)
        g = sum(0.97 ** k * r[0, k] for k in range(5))
        assert abs(float(adv[0, 0]) - (g - v[0, 0])) < 1e-4


@pytest.fixture(scope="module")
def rollouts():
    torch.manual_seed(0)
    model = CAMController(SMALL)
    model.eval()
    store = EmbStore()
    ag = CAMAgent(model, store, EmbedOnly(), AgentConfig(use_llm=False))
    eps = []
    for i, fam in enumerate(["found", "fund", "apicall", "found"]):
        core = SessionCore(task(fam, "tiny", 20 + i), CoreConfig(max_decisions=6 + 2 * i))
        eps.append(ag.run(core, ModelChooser(seed=i), f"e{i}"))
    return model, store, eps


def test_recomputed_log_probability_equals_the_one_recorded_while_acting(rollouts):
    """Full-episode unroll + teacher-forced arguments must reproduce acting-time likelihoods exactly; every
    policy-gradient quantity depends on this (dynamic candidate sets, masks and recurrent state included)."""
    model, store, eps = rollouts
    b = collate([e["decisions"] for e in eps], store)
    with torch.no_grad():
        out = evaluate(model, b, store)
    v = b.t["valid"]
    assert int(v.sum()) == sum(len(e["decisions"]) for e in eps)
    assert torch.allclose(out["logp"][v], b.t["logp_old"][v], atol=2e-4), (out["logp"][v] - b.t["logp_old"][v]).abs().max()
    assert torch.allclose(out["values"].mean(-1)[v], b.t["value_old"][v], atol=2e-4)
    ent = torch.tensor([d["ent"] for e in eps for d in e["decisions"]])
    assert torch.allclose(out["ent"][v], ent, atol=2e-4)
    assert torch.isfinite(out["logp"]).all() and (out["logp"][v] < 0).all()


def test_ppo_update_moves_the_weights_stays_finite_and_reports_diagnostics(rollouts):
    model, store, eps = rollouts
    import copy
    m = copy.deepcopy(model)
    algo = make("ppo", m, {"lr": 3e-4, "epochs": 2, "minibatch_eps": 2, "target_kl": 0.0}, "cpu")
    before = flat_params(m).clone()
    out = algo.update(eps, store)
    after = flat_params(m)
    assert torch.isfinite(after).all()
    assert out["param_delta"] > 0 and (after != before).any()
    assert out["grad_steps"] == 4 and out["nonfinite"] == 0
    assert all(np.isfinite(out[k]) for k in ("pol_loss", "v_loss", "entropy", "kl", "grad_norm", "explained_variance"))
    n_changed = sum(int((a != b).any()) for a, b in zip([p for p in m.parameters()], [p for p in model.parameters()]))
    assert n_changed > 20           # gradients reach the encoder, the recurrent cells and the heads
    # the acting policy changed: likelihood of the recorded actions differs from the recorded one
    b = collate([e["decisions"] for e in eps], store)
    with torch.no_grad():
        new = evaluate(m, b, store)["logp"]
    assert (new[b.t["valid"]] - b.t["logp_old"][b.t["valid"]]).abs().max() > 1e-4


def test_gradients_flow_to_every_trainable_component(rollouts):
    model, store, eps = rollouts
    import copy
    m = copy.deepcopy(model)
    m.train()
    b = collate([e["decisions"] for e in eps], store)
    out = evaluate(m, b, store)
    adv, ret = R.advantages(b.t["reward"], b.t["value_old"], b.t["done"], b.t["trunc"], b.t["valid"], R.RuleConfig())
    loss, _ = R.total_loss(out, b, adv + 0.1, ret, R.RuleConfig())
    loss.backward()
    groups = {}
    for n, p in m.named_parameters():
        g = n.split(".")[0]
        groups[g] = groups.get(g, 0.0) + (float(p.grad.abs().sum()) if p.grad is not None else 0.0)
    for g in ("tok", "ek_proj", "num_proj", "typ_emb", "obs_proj", "encoder", "fast", "slow", "mix", "head_score",
              "arg_q", "arg_score", "value", "success"):
        assert groups[g] > 0, f"no gradient reached {g}"


def test_behaviour_cloning_raises_the_likelihood_of_the_demonstrated_choices(rollouts):
    model, store, eps = rollouts
    import copy
    m = copy.deepcopy(model)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    first = bc_update(m, opt, eps, store, "cpu", 4)
    for _ in range(50):
        last = bc_update(m, opt, eps, store, "cpu", 4)
    assert last["bc_nll"] < first["bc_nll"] - 1.5 and last["param_delta"] > 0


def test_sil_algorithm_keeps_a_replay_buffer_of_successful_episodes(rollouts):
    model, store, eps = rollouts
    import copy
    m = copy.deepcopy(model)
    algo = make("ppo_sil", m, {"epochs": 1, "minibatch_eps": 4, "sil_steps": 1}, "cpu")
    fake = [dict(e, summary=dict(e["summary"], success=(i == 0))) for i, e in enumerate(eps)]
    out = algo.update(fake, store)
    assert out["replay_episodes"] == 1 and out["replay_steps"] == 1 and np.isfinite(out["replay_loss"])
    plain = make("ppo", copy.deepcopy(model), {"epochs": 1, "minibatch_eps": 4}, "cpu")
    assert plain.update(fake, store)["replay_steps"] == 0          # the baseline rule replays nothing
