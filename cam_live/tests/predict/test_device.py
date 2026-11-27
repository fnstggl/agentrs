"""Device placement: every module runs forward / loss / backward on CUDA with CUDA inputs (tiny tensors, < 100 MB,
so no shared lock is needed). Run with ``-m gpu``."""
import pytest
import torch

from cam_live.contracts import GoalStatus
from cam_live.predict import policy as P
from cam_live.predict.evaluator import EvaluatorConfig, GoalEvaluator, delta_v, evaluator_loss, prefix_targets
from cam_live.predict.imagination import imagine_candidates, rollout
from cam_live.predict.policy import CandidatePolicy, PolicyConfig, latent_dependence
from cam_live.predict.voi import DownstreamValueEstimator, downstream_loss
from cam_live.predict.world import (K_BINARY, K_CAT, K_COUNT, K_REAL, K_TTE, KindNormaliser, LatentTransition,
                                    WorldConfig, WorldHeads, WorldTargets, bootstrap_weights, transition_nll, world_nll)

DM = 32


@pytest.mark.gpu
def test_all_modules_run_on_cuda():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(0)
    b, n, q = 4, 5, 5
    z = torch.randn(b, n, DM, generator=g).to(dev)
    e_q = torch.randn(b, q, DM, generator=g).to(dev)
    bel = torch.randn(b, DM, generator=g).to(dev)
    e_g = torch.randn(b, DM, generator=g).to(dev)
    kind = torch.tensor([[K_REAL, K_COUNT, K_BINARY, K_TTE, K_CAT]]).expand(b, q).to(dev)
    hor = torch.full((b, q), 3600.0, dtype=torch.float64, device=dev)
    mask = torch.ones(b, n, dtype=torch.bool, device=dev)
    # world
    heads = WorldHeads(WorldConfig(d_model=DM, hidden=48, n_members=3, n_features=8)).to(dev)
    phi = torch.randn(b, n, q, 8, generator=g).to(dev)
    out = heads(z, e_q, kind, hor, phi)
    y = torch.rand(b, q, generator=g).round().to(dev)
    norm = KindNormaliser().to(dev)
    loss, _ = world_nll(out, WorldTargets(y, torch.ones(b, q, dtype=torch.bool, device=dev),
                                          torch.zeros(b, q, dtype=torch.bool, device=dev)),
                        chosen=torch.zeros(b, dtype=torch.long, device=dev),
                        member_weights=bootstrap_weights([f"k{i}" for i in range(b)], 3), normaliser=norm)
    loss.backward()
    pred = out.predictive()
    assert pred["mean"].device.type == "cuda" and out.thompson_means(g).device.type == "cuda"
    assert out.to_distribution(0, 1, 1).family == "neg_binomial"
    # transition + imagination
    tr = LatentTransition(DM, hidden=48, n_members=2).to(dev)
    tout = tr(z[:, 0], hor[:, 0], bel)
    tl, _ = transition_nll(tout, bel + 0.1, mask=torch.ones(b, dtype=torch.bool, device=dev))
    tl.backward()
    with torch.no_grad():
        ro = rollout(tr, lambda bb, ea, eg: ea + 0.1 * bb[:, None], bel, z[:, :3], e_g, hor[:, :3].float(),
                     n_particles=4, generator=g)
        iv = imagine_candidates(tr, z, hor[:, :n].float(), bel, lambda bb: bb.sum(-1), a_mask=mask, generator=g)
    assert ro.steps[-1].b_mean.device.type == "cuda" and iv.value_mean.shape == (b, n)
    # evaluator
    ev = GoalEvaluator(EvaluatorConfig(d_model=DM, hidden=48, d_ctx=2)).to(dev)
    ctx = torch.randn(b, 2, generator=g).to(dev)
    eo = ev(bel, e_g, hor[:, 0], ctx)
    _, tg = prefix_targets([0.0] * b, status=GoalStatus.ACHIEVED, evaluated_at=5.0, deadline=3600.0,
                           signature_checked=True, value=2.0)
    tg = type(tg)(*(None if getattr(tg, f) is None else getattr(tg, f).to(dev) for f in tg._FIELDS))
    el, _ = evaluator_loss(eo, tg)
    el.backward()
    assert ev.forward_candidates(z, e_g, hor[:, 0], ctx).p_success.shape == (b, n)
    assert delta_v(ev, bel, bel + 1, e_g, hor[:, 0], hor[:, 0], ctx)["dv_prob"].device.type == "cuda"
    # policy + selection
    pol = CandidatePolicy(PolicyConfig(d_model=DM, hidden=48)).to(dev)
    po = pol(z, mask, bel)
    (po.log_prob(torch.zeros(b, dtype=torch.long, device=dev)).sum() + po.value.sum()).backward()
    sel = P.select(P.RULE_THOMPSON, a_mask=mask, q_samples=torch.randn(16, b, n, device=dev), generator=g)
    assert sel.index.device.type == "cuda" and float(sel.probs.sum()) == pytest.approx(b)
    assert pol.functional_forward(pol.named_param_dict(), z, mask, bel).probs.device.type == "cuda"
    assert latent_dependence(pol, z, mask, bel)["kl"] >= 0
    # downstream estimator
    est = DownstreamValueEstimator(DM, n_signals=2).to(dev)
    downstream_loss(est(z, torch.randn(b, n, 2, device=dev)), torch.randn(b, n, device=dev)).backward()
