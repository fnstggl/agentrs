"""Imagination mechanics: independent FastState clones per branch, imagined belief updates that change later
imagined actions, real state bit-identical after a rollout, discounting/bootstrap arithmetic, epistemic estimates,
provenance and the absence of any archive access."""
import ast
import copy
import inspect

import torch

from superlearner.cam_unified.interfaces import Action, Provenance
from superlearner.cam_unified.testing import DoubleCore, tiny_config
from superlearner.cam_unified.world import imagination as imag_mod
from superlearner.cam_unified.world.dynamics import WorldModel
from superlearner.cam_unified.world.imagination import (Candidates, ImaginationHooks, Imagination, candidate_scores,
                                                        trace_summaries)

from world_synth import HypothesisBelief, ProbeWorld, probe_candidates, probe_hooks, real_start


def _setup(seed=0, **kw):
    torch.manual_seed(seed)
    cfg = tiny_config()
    w = ProbeWorld(cfg.d_model)
    bel = HypothesisBelief(cfg, w)
    wm = WorldModel(cfg, DoubleCore(cfg), train_horizons=(1, 2)).eval()
    Z, fast = real_start(wm, w, bel, 2, torch.device("cpu"))
    return cfg, w, bel, wm, Z, fast


def _fast_equal(a, b):
    return all(torch.equal(x, y) for x, y in zip(
        (a.particles, a.logw, a.pscale, a.bayes_mu, a.bayes_P, a.bayes_noise, a.bayes_n, a.surprise, a.steps),
        (b.particles, b.logw, b.pscale, b.bayes_mu, b.bayes_P, b.bayes_noise, b.bayes_n, b.surprise, b.steps))) \
        and a.channels == b.channels


def test_rollout_layout_provenance_and_real_state_bit_identical():
    cfg, w, bel, wm, Z, fast = _setup()
    fast.channels[0]["probe_result"] = 0
    hooks, _ = probe_hooks(w, bel, Z.device)

    def destructive_update(f, Zp, a, ob):          # mutates its input in place: must only ever see a clone
        f.logw.add_(1.0)
        f.particles.mul_(2.0)
        f.channels[0]["touched"] = 1
        return bel.update(f, Zp, a, ob)

    hooks.belief_update_fn = destructive_update
    Z0, f0 = Z.clone(), copy.deepcopy(fast)
    params0 = {n: p.detach().clone() for n, p in wm.named_parameters()}
    im = Imagination(wm, horizon=3, n_samples=5)
    cand = probe_candidates(w, 2, Z.device)
    traj = im.rollout(Z, fast, None, cand, hooks, generator=torch.Generator().manual_seed(0))
    assert torch.equal(Z, Z0) and _fast_equal(fast, f0)
    assert all(torch.equal(p, params0[n]) for n, p in wm.named_parameters())
    Bb = 2 * 4 * 5
    assert traj.returns.shape == (Bb,) and len(traj.steps) == 3
    assert traj.provenance == Provenance.IMAGINED and all(s.provenance == Provenance.IMAGINED for s in traj.steps)
    assert traj.branch_of.tolist() == [b for b in range(2) for _ in range(20)]
    assert traj.first_candidate.tolist() == [i for _ in range(2) for i in range(4) for _ in range(5)]
    assert [a.tool for a in traj.steps[0].action[:20:5]] == w.actions
    assert traj.n_model_evals == Bb * 3
    for s in traj.steps:
        assert s.Z.shape == (Bb, cfg.n_state, cfg.d_model) and s.reward.shape == (Bb,)
        assert s.model_error.shape == (Bb,) and torch.isfinite(s.model_error).all()
        assert s.belief_ess.shape == (Bb,)
    sc = candidate_scores(traj, 2, 4)
    assert sc.shape == (2, 4)
    tr = trace_summaries(traj, 0, 4, keys=["probe_result"])
    assert [e["candidate"] for e in tr] == [0, 1, 2, 3] and all(e["provenance"] == "imagined" for e in tr)
    assert set(tr[1]) >= {"candidate", "horizon", "value", "outcomes", "model_error"}


def test_imagined_belief_updates_change_subsequent_imagined_actions():
    cfg, w, bel, wm, Z, fast = _setup(1)
    cand = probe_candidates(w, 2, Z.device)
    im = Imagination(wm, horizon=2, n_samples=32, outcome_sampling="marginal")
    hooks, _ = probe_hooks(w, bel, Z.device)
    traj = im.rollout(Z, fast, None, cand, hooks, generator=torch.Generator().manual_seed(3))
    probe = traj.first_candidate == 1
    y = traj.steps[0].outcomes.y[:, 0]
    nxt = [a.tool for a in traj.steps[1].action]
    # after an imagined probe: positive result -> exploit, negative -> safe; both futures occur
    pos = [nxt[i] for i in torch.nonzero(probe & (y > 0.5)).flatten().tolist()]
    neg = [nxt[i] for i in torch.nonzero(probe & (y < 0.5)).flatten().tolist()]
    assert pos and neg and set(pos) == {"exploit"} and set(neg) == {"safe"}
    # the imagined belief really moved, and the probe's ESS dropped from K
    assert float(traj.steps[0].belief_ess[probe].max()) < cfg.n_particles - 0.5
    assert torch.allclose(traj.steps[0].belief_ess[~probe], torch.full_like(traj.steps[0].belief_ess[~probe],
                                                                              float(cfg.n_particles)))
    # without imagined belief updates every later decision stays at the uninformed choice
    hooks_id, _ = probe_hooks(w, bel, Z.device, update=False)
    traj2 = im.rollout(Z, fast, None, cand, hooks_id, generator=torch.Generator().manual_seed(3))
    assert {a.tool for a in traj2.steps[1].action} == {"wait"}


def test_returns_are_discounted_rewards_plus_bootstrap():
    cfg, w, bel, wm, Z, fast = _setup(2)
    hooks, _ = probe_hooks(w, bel, Z.device, value_fn=lambda R, g, b: torch.full((R.shape[0],), 2.0))
    im = Imagination(wm, horizon=3, n_samples=3, kappa=0.3, uncertainty_coef=0.5)
    cand = probe_candidates(w, 2, Z.device)
    traj = im.rollout(Z, fast, None, cand, hooks, generator=torch.Generator().manual_seed(1))
    disc = torch.ones_like(traj.returns)
    ret = torch.zeros_like(traj.returns)
    pen = torch.zeros_like(traj.returns)
    for s in traj.steps:
        torch.testing.assert_close(s.discount, disc)
        ret = ret + disc * s.reward
        pen = pen + disc * s.model_error
        disc = disc * torch.exp(-0.3 * s.dt) * s.cont
    torch.testing.assert_close(traj.returns_raw, ret + disc * 2.0)
    torch.testing.assert_close(traj.returns, ret + disc * 2.0 - 0.5 * pen)
    torch.testing.assert_close(traj.bootstrap, torch.full_like(ret, 2.0))
    # reproducible under a generator
    traj2 = im.rollout(Z, fast, None, cand, hooks, generator=torch.Generator().manual_seed(1))
    torch.testing.assert_close(traj.returns, traj2.returns)


def test_epistemic_estimate_is_particle_disagreement():
    cfg, w, bel, wm, Z, fast = _setup(3)
    hooks, _ = probe_hooks(w, bel, Z.device)
    cand = probe_candidates(w, 2, Z.device)
    im = Imagination(wm, horizon=1, n_samples=2)
    with torch.no_grad():
        wm.outcome_head.film.weight.normal_(0, 0.1)          # particle-conditioned predictions now differ
        wm.outcome_head.param.weight.normal_(0, 0.02)
    traj = im.rollout(Z, fast, None, cand, hooks, generator=torch.Generator().manual_seed(0))
    probe = traj.first_candidate == 1
    s = traj.steps[0]
    assert bool((s.epistemic[probe] > 1e-6).all()) and bool((s.epistemic[~probe] == 0).all())   # only observed
    same = copy.deepcopy(fast)
    same.particles[:] = same.particles[:, :1]                 # all particles identical -> no disagreement
    traj2 = im.rollout(Z, same, None, cand, hooks, generator=torch.Generator().manual_seed(0))
    assert float(traj2.steps[0].epistemic.abs().max()) < 1e-9
    assert bool((traj2.steps[0].aleatoric[probe] > 0).all())


def test_imagination_has_no_archive_or_environment_access():
    """Imagined steps can never reach real memory: the module imports nothing from memory/envs/dist and its code
    (docstrings excluded) names no archive, environment adapter or execution call."""
    from superlearner.cam_unified.world import planner as plan_mod
    for mod in (imag_mod, plan_mod):
        tree = ast.parse(inspect.getsource(mod))
        mods = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        mods += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        assert not any(m.split(".")[0] in ("memory", "envs", "dist") or ".memory" in m or ".envs" in m for m in mods), mods
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
            {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        assert not any("archive" in x.lower() for x in names), names
        assert not names & {"EnvAdapter", "execute", "advance_time", "collect_events"}
    im = Imagination(WorldModel(tiny_config(), DoubleCore(tiny_config())))
    assert not any("archive" in k or "env" in k for k in vars(im))
