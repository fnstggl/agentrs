"""Multi-seed learning tests of the DOM world model on the real UnifiedCore (spec 6.6 (1)-(4); plan 2026-11-26 §3 W6,
audit "make every 6.6 learning test multi-seed, add LR decay, assert sampled and mode-latent forecasts").

The world model is the DOM configuration: transition through the last 2 of 3 shared core blocks with adapters,
per-group free bits 0.02, consistency loss, mode-latent open-loop future; every test runs twice: "preset" = the dom
presets' cfg.wm_ema_target=False (latent target = detached posterior, self-predictive gradients routed to the
transition), "ema" = with the EMA target encoder (W6's recommendation; measured 3/3 seeds vs 2/3 for "preset" on the
delay world, see docs/evidence/dom/W6). The synthetic worlds of world_synth.py provide well-separated observation tokens, so these tests isolate the
world-model mechanism from the tokenizer (the integrated dom_dev check is capabilities.causal's DOM driver).
Slow (CAM_RUN_SLOW=1); three model seeds each, all must pass.
"""
import pytest
import torch

from superlearner.cam_unified.config import DOM_SWITCHES
from superlearner.cam_unified.core.backbone import UnifiedCore
from superlearner.cam_unified.interfaces import SequenceBatch
from superlearner.cam_unified.testing import tiny_config
from superlearner.cam_unified.world.dynamics import WorldModel

from world_synth import DelayWorld, InteractionWorld, SwitchWorld, batch_to, device, train

DEV = device()
SEEDS = (0, 1, 2)
WM_KEYS = ("wm_transition_blocks", "wm_ema_target", "kl_free_bits_group", "wm_consistency")


VARIANTS = {"preset": {}, "ema": {"wm_ema_target": True}}


def dom_wm_cfg(**over):
    kw = {k: DOM_SWITCHES[k] for k in WM_KEYS}
    kw.update(n_layers=3, wm_transition_blocks=2, state_emb_std=0.5)
    kw.update(over)
    return tiny_config(**kw)


def dom_wm(cfg, seed, **kw):
    torch.manual_seed(seed)
    wm = WorldModel(cfg, UnifiedCore(cfg), **kw).to(DEV)
    assert (wm.target_core is None) == (not cfg.wm_ema_target) and wm.k_trans == 2 and wm.mode_latent
    return wm


def _gen():
    return torch.Generator(device=DEV).manual_seed(0)


def delay_run(seed: int, steps: int = 700, variant: str = "preset"):
    cfg = dom_wm_cfg(**VARIANTS[variant])
    w = DelayWorld(cfg.d_model)
    batches = [batch_to(w.batch(64, seed=i), DEV) for i in range(16)]
    wm = dom_wm(cfg, seed, train_horizons=(1, 2, 3, 4), open_loop_samples=2, open_loop_starts=4)
    hist = train(wm, batches, steps, lr=4e-3, lr_end_frac=0.1, seed=seed)
    wm.eval()
    plans = [[0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]
    tb = batch_to(w.batch(3, seed=999, plans=plans), DEV)
    f = wm.forecast(tb, 2, (1, 2, 3), n_samples=8, generator=_gen())
    fm = wm.forecast(tb, 2, (1, 2, 3), n_samples=1, generator=_gen(), sample=False)      # mode latent
    pred = torch.stack([f[h]["mean"][:, 0] for h in (1, 2, 3)], 1).cpu()
    mode = torch.stack([fm[h]["mean"][:, 0] for h in (1, 2, 3)], 1).cpu()
    return pred, mode, hist


DELAY_TRUTH = torch.tensor([[1.0, 0, 0], [0, 0, 1.0], [0, 0, 0]])


@pytest.mark.slow
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_dom_world_model_learns_delayed_vs_immediate_effects_on_every_seed(variant):
    errs = []
    for s in SEEDS:
        pred, mode, hist = delay_run(s, variant=variant)
        e, em = float((pred - DELAY_TRUTH).abs().max()), float((mode - DELAY_TRUTH).abs().max())
        errs.append((e, em))
        print(f"{variant} seed {s}: loss {hist[0]:.2f}->{hist[-1]:.2f} sampled max err {e:.3f} mode-latent max err {em:.3f}\n"
              f"{pred}\n{mode}")
    assert all(e < 0.3 and em < 0.3 for e, em in errs), errs


@pytest.mark.slow
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_dom_world_model_learns_the_interaction_on_every_seed(variant):
    out = []
    for s in SEEDS:
        cfg = dom_wm_cfg(**VARIANTS[variant])
        w = InteractionWorld(cfg.d_model)
        raw = [w.batch(64, seed=i) for i in range(10)]
        wm = dom_wm(cfg, s, train_horizons=(1, 2), open_loop_samples=2, open_loop_starts=2)
        train(wm, [batch_to(b, DEV) for b in raw], 400, lr=2e-3, lr_end_frac=0.1, seed=s)
        wm.eval()
        tb = batch_to(w.batch(4, seed=77, plans=[[0, a, 0, 0] for a in range(4)]), DEV)
        pred = wm.forecast(tb, 1, (1,), n_samples=8, generator=_gen())[1]["mean"][:, 0].cpu()
        contrast = float(pred[3] - pred[1] - pred[2] + pred[0])
        truth = torch.tensor([w.truth(a) for a in range(4)])
        out.append((contrast, float((pred - truth).abs().max())))
        print(f"{variant} seed {s}: interaction prediction {pred.tolist()} contrast {contrast:.2f}")
    assert all(abs(c - 2.0) < 0.4 and e < 0.3 for c, e in out), out


@pytest.mark.slow
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_dom_world_model_follows_a_mechanism_change_on_every_seed(variant):
    res = []
    for s in SEEDS:
        cfg = dom_wm_cfg(**VARIANTS[variant])
        w = SwitchWorld(cfg.d_model)
        batches = [batch_to(w.batch(64, seed=i), DEV) for i in range(16)]
        wm = dom_wm(cfg, s, train_horizons=(1, 2, 3, 4), open_loop_samples=2, open_loop_starts=4)
        train(wm, batches, 700, lr=4e-3, lr_end_frac=0.1, seed=s)
        wm.eval()
        preds = {}
        for name, force in (("switch", {"s": 1.0, "sw": 4}), ("none", {"s": 1.0, "sw": None})):
            w.force = force
            tb = batch_to(w.batch(1, seed=5, plans=[[1] * w.T]), DEV)
            w.force = None
            preds[name] = torch.tensor([float(wm.forecast(tb, t, (1,), n_samples=16, generator=_gen())[1]["mean"][0, 0])
                                        for t in range(2, 9)])
        sw, no = preds["switch"], preds["none"]
        ok = bool((sw[3:] < -0.7).all()) and bool((no[3:] > 0.7).all()) and float((sw[:3] - no[:3]).abs().max()) < 1e-4
        res.append(ok)
        print(f"{variant} seed {s}: switch {sw.tolist()} none {no.tolist()} ok {ok}")
    assert all(res), res


@pytest.mark.slow
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_dom_world_model_closes_the_consistency_gap_on_delay_world(variant):
    """The mode-latent one-step prior vs the next posterior: MSE / Var(Z_post) < 0.1 after training (the plan's
    consistency criterion on DelayWorld)."""
    vals = []
    for s in SEEDS:
        cfg = dom_wm_cfg(**VARIANTS[variant])
        w = DelayWorld(cfg.d_model)
        batches = [batch_to(w.batch(64, seed=i), DEV) for i in range(16)]
        wm = dom_wm(cfg, s, train_horizons=(1, 2, 3, 4), open_loop_samples=2, open_loop_starts=4)
        train(wm, batches, 700, lr=4e-3, lr_end_frac=0.1, seed=s)
        wm.eval()
        test = batch_to(w.batch(64, seed=4242), DEV)
        with torch.no_grad():
            out = wm.loss(test, generator=_gen())
        vals.append((float(out["cons_mse_var"]), float(out["cons_mse_var_mode1"])))
        print(f"{variant} seed {s}: consistency MSE/Var teacher-forced {vals[-1][0]:.4f} mode-latent prior {vals[-1][1]:.4f}")
    assert all(a < 0.1 and b < 0.1 for a, b in vals), vals
