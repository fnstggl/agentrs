"""W6 PRED (plan 2026-11-26 §3): the DOM belief path on the real BeliefModule / BayesLastLayer / OutcomeHead /
WorldModel / UnifiedCore (the tokenizer, action generator and small heads are integration doubles; the environment
is the ToyShop double).

* gated fusion of the particle mixture and the Kalman/NIG predictive is a proper density with the stated moments;
* `belief_change`: exact Gaussian information gain, KL over the particle index across resampling, gradients;
* `observe_outcomes`: world-model likelihood features, reward unobserved in observation_only (FastState
  bit-identical when the verified reward stream changes), full-predictive learning progress, belief-change record;
* `adapt_loss_all_k`: query outcomes never reach query states or the support belief, episode resets, every support
  size in one pass, and equality with the reference `learn.meta.SupportQueryLoss`.
"""
import dataclasses
import math

import pytest
import torch

from superlearner.cam_unified import integration_doubles as dbl
from superlearner.cam_unified.config import DOM_SWITCHES
from superlearner.cam_unified.core import belief_runtime as br
from superlearner.cam_unified.core.bayes import BayesLastLayer
from superlearner.cam_unified.interfaces import Likelihood, OutcomeBatch, SequenceBatch, StepBatch, TokenKind, TokenSet
from superlearner.cam_unified.learn.meta import SupportQueryLoss, belief_callables_from_modules
from superlearner.cam_unified.model import build_model
from superlearner.cam_unified.testing import rand_sequence, tiny_config
from superlearner.cam_unified.world import train_seq

INERT = ("sem_xattn_every", "dec_xattn_every", "n_intent", "n_candidates", "value_channel", "sem_lora",
         "sem_encoder_lora", "microbatch")


def dom_tiny(seed: int = 0, **over):
    sw = {k: v for k, v in DOM_SWITCHES.items() if k not in INERT}
    kw = {**sw, "preset": "dom_test", "n_layers": 3, "wm_transition_blocks": 2, "reason_layers": 1, "burn_in": 0,
          "bptt_steps": 4, "train_horizons": (1, 2)}
    kw.update(over)
    cfg = tiny_config(**kw)
    torch.manual_seed(seed)
    comps = {"tokenizer": dbl.DoubleTokenizer(cfg), "action_generator": dbl.DoubleActionGenerator(cfg),
             "value_head": dbl.DoubleValueHead(cfg), "option_policy": dbl.DoubleOptionPolicy(cfg),
             "exploration_head": dbl.DoubleExplorationHead(cfg), "entity_encoder": None}
    return build_model(cfg, components=comps)


def deterministic(m):
    """Remove the belief module's stochastic parts for exact comparisons: fixed prior draws, no jitter, no
    resampling / expansion, a deterministic (zero) surprise statistic (instance-level, this model only)."""
    bm = m.beliefs
    bm.jitter, bm.resample_enabled, bm.misspec_enabled = 0.0, False, False
    init = bm.init_fast

    def fixed_init(B, device=None, Z=None, generator=None):
        return init(B, device=device, Z=Z, generator=torch.Generator().manual_seed(123))
    bm.init_fast = fixed_init
    bm.surprise_z = lambda dist, logw, y_mask, pred_ll, *a, **k: (torch.zeros_like(pred_ll), torch.zeros_like(pred_ll))
    return m


def one_query(B, y, lik=Likelihood.GAUSSIAN, D=64, key="x", observed=True):
    q = TokenSet(torch.ones(B, 1, D), torch.ones(B, 1, dtype=torch.bool), torch.full((B, 1), int(TokenKind.QUERY)),
                 torch.zeros(B, 1), torch.full((B, 1), -1), None)
    return OutcomeBatch(q, torch.full((B, 1), int(lik)), torch.zeros(B, 1, dtype=torch.long),
                        torch.as_tensor(y, dtype=torch.float32).view(B, 1), torch.zeros(B, 1, dtype=torch.bool),
                        torch.full((B, 1), observed, dtype=torch.bool), torch.zeros(B, 1, dtype=torch.long),
                        [[key] for _ in range(B)])


# ------------------------------------------------------------------------------------------------ fusion

def test_dom_presets_switch_fusion_and_film_on_and_legacy_stays_off():
    m = dom_tiny()
    assert br.dom_beliefs(m) and br.fusion_on(m) and m.bayes.gate is not None
    assert float(m.outcome_head.film.weight.abs().sum()) > 0             # symmetric dead start broken
    legacy = BayesLastLayer(tiny_config())
    assert not legacy.fusion and legacy.gate is None


def test_fused_predictive_is_a_density_with_the_stated_moments():
    m = dom_tiny()
    B = 3
    fast = m.beliefs.init_fast(B)
    feat = torch.randn(B, 1, m.cfg.d_model, generator=torch.Generator().manual_seed(0))
    # bind and update the Kalman channel twice so the Kalman expert is informative
    for y in (0.7, 0.9):
        ob = one_query(B, [y] * B)
        fast = m.kalman_update(fast, feat, ob, br.kalman_base(m, feat, ob, fast))
    grid = torch.linspace(-40, 40, 40001)
    obg = one_query(B, [0.0] * B)
    fp = br.fused_predictive(m, fast, feat, obg)
    assert bool(fp.fused.all()) and fp.gate is not None
    # gate initialised as sigmoid(log1p(n) - 1) with n = 2 observations
    torch.testing.assert_close(fp.gate, torch.full((B, 1), torch.sigmoid(torch.tensor(math.log(3.0) - 1.0)).item()),
                               rtol=1e-4, atol=1e-4)
    lps = []
    for chunk in grid.split(4001):
        n = chunk.numel()
        obc = one_query(B * n, chunk.repeat(B))
        fr = fast.select(torch.arange(B).repeat_interleave(n))
        fr2 = br.fused_predictive(m, fr, feat.repeat_interleave(n, 0), obc)
        lps.append(fr2.query_log_prob(obc).view(B, n))
    lp = torch.cat(lps, 1)
    dx = float(grid[1] - grid[0])
    p = lp.exp()
    mass = p.sum(1) * dx
    torch.testing.assert_close(mass, torch.ones(B), rtol=2e-3, atol=2e-3)
    mean = (p * grid).sum(1) * dx
    var = (p * grid ** 2).sum(1) * dx - mean ** 2
    torch.testing.assert_close(mean, fp.mean()[:, 0], rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(var, fp.var()[:, 0], rtol=3e-2, atol=3e-2)
    # joint log-prob of a single fused query equals its marginal
    torch.testing.assert_close(fp.joint_log_prob(one_query(B, [0.3] * B)),
                               fp.query_log_prob(one_query(B, [0.3] * B))[:, 0])


def test_predictive_log_prob_uses_the_fused_predictive_and_kalman_dropout_gives_the_mixture():
    m = dom_tiny()
    B = 2
    fast = m.beliefs.init_fast(B)
    Zh = torch.randn(B, m.cfg.n_state, m.cfg.d_model)
    ob = one_query(B, [0.5, -0.5])
    feat = m.outcome_head.features(Zh, ob)
    fast = m.kalman_update(fast, feat, ob, br.kalman_base(m, feat, ob, fast))
    full = br.predictive_log_prob(m, fast, Zh, ob, per_outcome=False)
    fp = br.fused_predictive(m, fast, feat, ob)
    torch.testing.assert_close(full, fp.joint_log_prob(ob))
    mix = br.predictive_log_prob(m, fast, Zh, ob, per_outcome=False, kalman=False)
    torch.testing.assert_close(mix, m.mixture_ll(fast, feat, ob, ob.y_mask))
    assert float((full - mix).abs().max()) > 1e-4


# ------------------------------------------------------------------------------------------------ belief change

@pytest.mark.parametrize("mode", ["fixed", "nig"])
def test_kalman_ig_is_the_exact_gaussian_information_gain(mode):
    cfg = tiny_config(bayes_feat=3)
    bayes = BayesLastLayer(cfg, noise_mode=mode)
    from superlearner.cam_unified.testing import zero_fast_state
    B = 2
    fast = zero_fast_state(B, cfg, C=1)
    fast.bayes_P = torch.eye(3, dtype=torch.float64).expand(B, 1, 3, 3).clone() * 0.7
    fast.bayes_noise = torch.full((B, 1), 0.3, dtype=torch.float64)
    fast.bayes_n = torch.zeros(B, 1)
    fast.channels = [{"x": 0} for _ in range(B)]
    phi = torch.tensor([[[0.5, -1.0, 1.0]], [[2.0, 0.1, 1.0]]])
    ob = one_query(B, [1.3, -0.4])
    new = bayes.update(fast, phi, ob)
    ig = br.belief_change(fast, new)["kalman_ig"]
    V = fast.bayes_P[:, 0] / fast.bayes_noise[:, 0, None, None]
    ref = 0.5 * torch.log1p(torch.einsum("bi,bij,bj->b", phi[:, 0].double(), V, phi[:, 0].double()))
    torch.testing.assert_close(ig.double(), ref, rtol=1e-6, atol=1e-8)
    # an unobserved step carries no Kalman information
    same = bayes.update(new, phi, one_query(B, [0.0, 0.0], observed=False))
    assert float(br.belief_change(new, same)["kalman_ig"].abs().max()) == 0.0


def test_kl_weights_use_the_pre_resampling_posterior_and_are_differentiable():
    m = dom_tiny()
    B = 4
    m.beliefs.resample_enabled = True
    m.beliefs.cfg = dataclasses.replace(m.cfg, ess_frac=1.01)              # resample at every informative step
    Zh = torch.randn(B, m.cfg.n_state, m.cfg.d_model)
    ob = one_query(B, [3.0, -3.0, 3.0, -3.0])
    f0 = m.beliefs.init_fast(B)
    feat = m.outcome_head.features(Zh, ob)
    f1, info = m.beliefs.update(f0, Zh, None, ob, m.outcome_head, feat=feat)
    assert bool(info["resampled"].all())
    torch.testing.assert_close(f1.logw, torch.full_like(f1.logw, -math.log(m.cfg.n_particles)))
    bc = br.belief_change(f0, f1, info)
    lw = info["logw_post"]
    ref = (lw.exp() * (lw - f0.logw)).sum(-1)
    torch.testing.assert_close(bc["kl_weights"], ref.clamp_min(0))
    assert float(bc["kl_weights"].min()) > 0                      # informative although the weights were reset
    # without the update info the (uniform) resampled weights would claim no information
    assert float(br.belief_change(f0, f1)["kl_weights"].abs().max()) < 1e-6
    torch.testing.assert_close(bc["d_entropy"], math.log(m.cfg.n_particles) - (-(lw.exp() * lw).sum(-1)))
    # gradient of the information gain reaches the particle-conditioned head and the belief prior
    bc["info_gain"].sum().backward()
    assert float(m.outcome_head.film.weight.grad.abs().sum()) > 0
    assert float(m.beliefs.prior_raw_scale.grad.abs().sum()) > 0


# ------------------------------------------------------------------------------------------------ runtime

def _decide_observe(m, env, state, rewards):
    ob = env.observe() if hasattr(env, "observe") else None
    dec = m.decide(state, [list(o) for o in ob.observations], list(ob.t), env.available_tools(), env.goals())
    res = env.execute(dec.actions)
    adv = env.advance_time([r.duration + m.think_days for r in res])
    nxt = [list(r.observations) + list(o) for r, o in zip(res, adv.observations)]
    return dec, m.observe_outcomes(state, dec, res, nxt, rewards)


def test_observe_outcomes_observation_only_and_world_model_likelihood(monkeypatch):
    out = {}
    for name, rew in (("a", [0.0, 0.0]), ("b", [7.5, -3.0])):
        m = dom_tiny()
        env = dbl.ToyShopEnv(B=2, seed=0)
        env.reset([0, 1])
        state = m.init_state(2)

        def boom(*a, **k):
            raise AssertionError("act_cond must not be used on the DOM path")
        monkeypatch.setattr(m.beliefs, "condition_on_action", boom)
        torch.manual_seed(3)
        infos = [_decide_observe(m, env, state, rew)[1] for _ in range(3)]
        out[name] = (state.fast.clone(), infos)
    (fa, ia), (fb, ib) = out["a"], out["b"]
    for n in ("particles", "logw", "pscale", "bayes_mu", "bayes_P", "bayes_noise", "bayes_n"):
        assert torch.equal(getattr(fa, n), getattr(fb, n)), n
    for info in ia:
        for b in range(2):
            rs = [s for s in info["specs"][b] if s.key == "reward"]
            assert rs and not rs[0].observed
            assert "reward" not in fa.channels[b]
        assert set(info["belief_change"]) == {"kl_weights", "d_entropy", "kalman_ig", "info_gain"}
        assert torch.isfinite(info["loglik"]).all()
    assert ia[1]["lp"] is not None and torch.isfinite(ia[1]["lp"]).all()


# ------------------------------------------------------------------------------------------------ adaptation

def _seq(m, B=3, T=7, seed=0):
    seq = rand_sequence(B, T, m.cfg, Q=6, seed=seed)
    for st in seq.steps:
        st.outcomes.censored = torch.zeros_like(st.outcomes.censored)
    return seq


def shift_continuous(ob, d):
    c = (ob.lik == int(Likelihood.GAUSSIAN)) | (ob.lik == int(Likelihood.STUDENT_T))
    return torch.where(c, ob.y + d, ob.y)


def _loss(m, seq, **kw):
    filt = train_seq.filter_sequence(m, seq, None, burn_in=0)
    return br.adapt_loss_all_k(m, seq, filt, **kw)


def test_adapt_query_outcomes_never_reach_query_states_or_the_support():
    m = deterministic(dom_tiny())
    m.eval()
    seq = _seq(m)
    torch.manual_seed(0)
    base, mets = _loss(m, seq, ks=(2,), max_query=1, use_kalman=True, mix_aux=0.0)
    assert mets["adapt_ks"] == [2] and mets["adapt_queries"] > 0
    # outcomes and observations after the query step (t = 2) change nothing
    for t in range(3, 7):
        seq.steps[t].outcomes.y = shift_continuous(seq.steps[t].outcomes, 5.0)
        seq.steps[t].obs.x = seq.steps[t].obs.x + 3.0
    torch.manual_seed(0)
    again, _ = _loss(m, seq, ks=(2,), max_query=1, use_kalman=True, mix_aux=0.0)
    assert float(again) == float(base)
    # the query outcome itself is what is scored
    seq.steps[2].outcomes.y = shift_continuous(seq.steps[2].outcomes, 1.0)
    torch.manual_seed(0)
    moved, _ = _loss(m, seq, ks=(2,), max_query=1, use_kalman=True, mix_aux=0.0)
    assert float(moved) != float(base)


def test_adapt_scores_every_support_size_in_one_pass_and_resets_at_done():
    m = deterministic(dom_tiny())
    m.eval()
    T = 8
    seq = _seq(m, T=T, seed=4)
    loss, mets = _loss(m, seq, use_kalman=True)
    assert mets["adapt_ks"] == [0, 1, 2, 4] and mets["adapt_skipped_k"] == [8, 16, 32, 64]
    loss.backward()
    g = m.bayes.gate.lin.weight.grad
    assert g is not None and float(g.abs().sum()) > 0                    # the fusion gate learns from L_adapt
    # an episode end after step 2: with no observed outcomes in the first episode, the loss equals the loss of
    # the second episode on its own (fresh belief at the episode start)
    seq2 = _seq(m, T=T, seed=4)
    for t in range(3):
        seq2.steps[t].outcomes.y_mask = torch.zeros_like(seq2.steps[t].outcomes.y_mask)
    seq2.steps[2].done = torch.ones_like(seq2.steps[2].done)
    a, ma = _loss(m, seq2, ks=(0, 1, 2), use_kalman=True, mix_aux=0.0)
    alone = SequenceBatch(seq2.steps[3:])
    b, mb = _loss(m, alone, ks=(0, 1, 2), use_kalman=True, mix_aux=0.0)
    assert ma["adapt_queries"] == mb["adapt_queries"]
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)


def test_adapt_all_k_equals_the_reference_support_query_loss():
    """With the Kalman layer off, the belief module's own likelihood and the prior on Z (no block transition), the
    one-pass batched objective equals `learn.meta.SupportQueryLoss` (per support size and in total)."""
    m = deterministic(dom_tiny(wm_transition_blocks=0))
    m.eval()
    seq = _seq(m, T=6, seed=9)
    filt = train_seq.filter_sequence(m, seq, None, burn_in=0)
    assert not train_seq.prior_on_R(m)
    ks = (0, 1, 2, 4)
    loss, mets = br.adapt_loss_all_k(m, seq, filt, ks=ks, max_query=6, use_kalman=False, mix_aux=0.0,
                                     support_feat="module")
    init, upd, lpf = belief_callables_from_modules(m.beliefs, m.outcome_head, filt.Zs, seq.steps, m.world_model,
                                                   query_Z="support_end", sample=False)
    ref = SupportQueryLoss(ks, max_query=6)(seq, init, upd, lpf)
    for k in ks:
        assert abs(mets[f"adapt_nll_k{k}"] - ref["per_k"][k]) < 1e-4, (k, mets[f"adapt_nll_k{k}"], ref["per_k"][k])
    torch.testing.assert_close(loss, ref["loss"], rtol=1e-4, atol=1e-4)
