"""beta-NLL outcome loss and the direct action input of the DOM block transition (plan 2026-11-26 §3 W6).

Both changes come from measurements on the real dom_dev stack (docs/evidence/dom/W6/world_model_outcome_learning.json):
with plain heteroscedastic NLL the action-dependent InteractionEnv payout stayed at climatology, and the executed
action moved the block transition's output by ~1-2 % of its across-row spread. These unit tests pin the mechanics.
"""
import math

import torch

from superlearner.cam_unified.interfaces import Likelihood, TokenKind, TokenSet
from superlearner.cam_unified.testing import rand_sequence
from superlearner.cam_unified.world.dynamics import WorldModel, outcome_beta_weights
from superlearner.cam_unified.world.outcomes import MixedOutcomeDist

from test_consistency import dom_tiny


def _dist(raw_g_sigma: float = 0.3):
    lik = torch.tensor([[int(Likelihood.GAUSSIAN), int(Likelihood.STUDENT_T), int(Likelihood.BERNOULLI),
                         int(Likelihood.CATEGORICAL)]])
    P = 2 + 3 + 1 + 32 + 2 + 9
    raw = torch.zeros(1, 4, P)
    raw[..., 1] = raw_g_sigma
    edges = torch.linspace(0, 8, 9)
    return MixedOutcomeDist(raw, lik, torch.tensor([[0, 0, 0, 4]]), edges, 32)


def test_beta_weights_are_scale_powers_for_continuous_families_and_one_otherwise():
    d = _dist()
    for beta in (0.5, 1.0):
        w = outcome_beta_weights(d, beta)
        assert w.shape == (1, 4) and not w.requires_grad
        torch.testing.assert_close(w[0, 0], d.g_std[0, 0].float() ** (2 * beta))
        torch.testing.assert_close(w[0, 1], d.t_scale[0, 1].float() ** (2 * beta))
        assert float(w[0, 2]) == 1.0 and float(w[0, 3]) == 1.0
    assert bool((outcome_beta_weights(d, 0.0) == 1).all())


def test_dom_world_model_defaults_to_beta_half_and_legacy_to_plain_nll():
    m = dom_tiny()
    assert m.world_model.outcome_nll_beta == 0.5
    from superlearner.cam_unified.testing import tiny_config
    from superlearner.cam_unified.core.backbone import UnifiedCore
    cfg = tiny_config()
    assert WorldModel(cfg, UnifiedCore(cfg)).outcome_nll_beta == 0.0


def test_beta_reweights_only_the_training_term_not_the_reported_nll():
    m = dom_tiny()
    m.eval()
    seq = rand_sequence(3, 5, m.cfg)
    with torch.no_grad():
        filt = m.filter_sequence(seq, None)
        wm = m.world_model
        outs = {}
        for beta in (0.0, 0.5):
            wm.outcome_nll_beta = beta
            torch.manual_seed(0)                            # same latent samples for both
            outs[beta] = m.world_loss(seq, filt)
    a, b = outs[0.0], outs[0.5]
    assert math.isfinite(float(a["outcome"])) and math.isfinite(float(b["outcome"]))
    assert float(a["outcome"]) != float(b["outcome"])
    # the mixture likelihoods (responsibilities, reported NLL) use the unweighted log-likelihoods
    torch.testing.assert_close(a["open_loop_joint_nll"], b["open_loop_joint_nll"])


def test_action_input_moves_every_state_token_and_is_zero_without_action_tokens():
    m = dom_tiny()
    wm = m.world_model
    B, S, D = 3, m.cfg.n_state, m.cfg.d_model
    g = torch.Generator().manual_seed(3)
    act = TokenSet.from_dense(torch.randn(B, 2, D, generator=g), TokenKind.ACTION)
    act.mask[2] = False                                     # row 2 has no action tokens
    assert wm.action_input_on and wm.trans_act_gain == 0.0
    a = wm.action_input(act, B, torch.device("cpu"))
    assert a.shape == (B, 1, D) and float(a.abs().max()) == 0.0   # zero-initialised projection
    wg = WorldModel(m.cfg, m.core, trans_act_gain=1.0)
    a = wg.action_input(act, B, torch.device("cpu"))
    assert float(a[:2].pow(2).mean().sqrt()) > 0.3          # unit gain at initialisation when requested
    assert float(a[2].abs().max()) == 0.0
    # gradient reaches the projection from the transition output
    R = torch.randn(B, S, D, generator=g)
    z = torch.zeros(B, m.cfg.stoch_groups * m.cfg.stoch_classes)
    z[:, ::m.cfg.stoch_classes] = 1.0
    Zh = wm.transition(R, act, None, torch.ones(B), z)
    Zh.pow(2).sum().backward()
    assert wm.trans_act.weight.grad is not None and float(wm.trans_act.weight.grad.abs().sum()) > 0


def test_action_input_running_standardisation_updates_in_training_only_after_use():
    m = dom_tiny()
    wm = WorldModel(m.cfg, m.core, trans_act_gain=1.0)
    assert wm.act_norm_mode == "running"
    B, D = 4, m.cfg.d_model
    g = torch.Generator().manual_seed(5)
    shared = torch.randn(1, 1, D, generator=g) * 3.0          # dominant component shared by all actions
    act = TokenSet.from_dense(shared + 0.1 * torch.randn(B, 2, D, generator=g), TokenKind.ACTION)
    st = wm.trans_act_stats
    wm.eval()
    with torch.no_grad():
        a0 = wm.action_input(act, B, torch.device("cpu"))
    assert int(st.count) == 0                                 # acting / eval: no statistics update
    wm.train()
    a1 = wm.action_input(act, B, torch.device("cpu"))
    torch.testing.assert_close(a1, a0)                        # the batch is never normalised with its own stats
    assert int(st.count) == 1
    a2 = wm.action_input(act, B, torch.device("cpu"))
    # after the update the shared component is removed: the rows differ at unit scale
    spread0 = float((a0 - a0.mean(0, keepdim=True)).pow(2).mean().sqrt())
    spread2 = float((a2 - a2.mean(0, keepdim=True)).pow(2).mean().sqrt())
    assert spread2 > 3 * spread0
