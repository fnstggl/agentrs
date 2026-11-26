"""W6 `world.voi.decision_voi` on the real world model / belief / Bayes / outcome modules (tokenizer, action generator
and value head are integration doubles).

* the batched estimate equals `selfeval.voi.voi_monte_carlo` run row by row with the same draws;
* the real FastState is untouched; identical information actions get identical VOI (common random numbers);
* known answers: with a continuation value that depends only on the belief over hypotheses, VOI is exactly 0 when
  the outcome is uninformative (every particle predicts the same distribution) and > 0 when it discriminates.
"""
import torch

from superlearner.cam_unified import integration_doubles as dbl
from superlearner.cam_unified.config import DOM_SWITCHES
from superlearner.cam_unified.interfaces import Action
from superlearner.cam_unified.model import build_model
from superlearner.cam_unified.testing import tiny_config
from superlearner.cam_unified.world.voi import VOIState, decision_voi, decision_voi_reference, outcome_q, value_q

INERT = ("sem_xattn_every", "dec_xattn_every", "n_intent", "n_candidates", "value_channel", "sem_lora",
         "sem_encoder_lora", "microbatch")


def dom_tiny(seed: int = 0, **over):
    sw = {k: v for k, v in DOM_SWITCHES.items() if k not in INERT}
    kw = {**sw, "preset": "dom_test", "n_layers": 3, "wm_transition_blocks": 2, "reason_layers": 1,
          "train_horizons": (1, 2)}
    kw.update(over)
    cfg = tiny_config(**kw)
    torch.manual_seed(seed)
    comps = {"tokenizer": dbl.DoubleTokenizer(cfg), "action_generator": dbl.DoubleActionGenerator(cfg),
             "value_head": dbl.DoubleValueHead(cfg), "option_policy": dbl.DoubleOptionPolicy(cfg),
             "exploration_head": dbl.DoubleExplorationHead(cfg), "entity_encoder": None}
    m = build_model(cfg, components=comps)
    m.eval()
    return m


def deterministic(m):
    """No jitter / resampling / expansion and a deterministic surprise statistic, so batched and row-by-row belief
    updates are the same computation (instance-level, this model only)."""
    bm = m.beliefs
    bm.jitter, bm.resample_enabled, bm.misspec_enabled = 0.0, False, False
    bm.surprise_z = lambda dist, logw, y_mask, pred_ll, *a, **k: (torch.zeros_like(pred_ll), torch.zeros_like(pred_ll))
    return m


def voi_state(m, B=2, seed=0):
    env = dbl.ToyShopEnv(B=B, seed=seed)
    ob = env.reset(list(range(B)))
    state = m.init_state(B)
    dec = m.decide(state, [list(o) for o in ob.observations], list(ob.t), env.available_tools(), env.goals())
    res = env.execute(dec.actions)
    adv = env.advance_time([r.duration + m.think_days for r in res])
    nxt = [list(r.observations) + list(o) for r, o in zip(res, adv.observations)]
    m.observe_outcomes(state, dec, res, nxt, None)
    specs, prosp = m.prospective_batch(state)
    tools = env.available_tools()
    names = [t.name for t in tools[0]]
    return state, VOIState(dec.R, state.fast, m.tokenizer.goal(env.goals()), prosp,
                           dt=torch.full((B,), 0.01)), names


def test_batched_voi_equals_voi_monte_carlo_and_never_touches_the_real_belief():
    m = deterministic(dom_tiny())
    state, vs, names = voi_state(m)
    B = vs.R.shape[0]
    info = [[Action(tool=names[0]), Action(tool=names[1])] for _ in range(B)]
    nxt = [[Action(tool=n) for n in names[:3]] for _ in range(B)]
    before = state.fast.clone()
    for q in (value_q, outcome_q({"cost": -1.0, "num:*": 1.0})):
        out = decision_voi(m, vs, info, 4, next_actions=nxt, q_fn=q, seed=7)
        ref = decision_voi_reference(m, vs, info, 4, next_actions=nxt, q_fn=q, seed=7)
        assert out["voi"].shape == (B, 2) and torch.isfinite(out["voi"]).all()
        torch.testing.assert_close(out["voi"], ref, rtol=1e-4, atol=1e-4)
    for n in ("particles", "logw", "pscale", "bayes_mu", "bayes_P", "bayes_noise", "bayes_n"):
        assert torch.equal(getattr(before, n), getattr(state.fast, n)), n
    # identical information actions: identical VOI (common random numbers)
    same = decision_voi(m, vs, [[Action(tool=names[0])] * 2 for _ in range(B)], 4, next_actions=nxt, seed=3)
    torch.testing.assert_close(same["voi"][:, 0], same["voi"][:, 1], rtol=0, atol=0)
    # cost and discount enter as stated
    c = torch.full((B, 2), 0.25)
    g = torch.full((B, 2), 0.5)
    out2 = decision_voi(m, vs, info, 4, next_actions=nxt, cost=c, gamma=g, seed=7)
    base = decision_voi(m, vs, info, 4, next_actions=nxt, seed=7)
    torch.testing.assert_close(out2["voi"], 0.5 * base["q_after"].mean(-1) - base["base"].unsqueeze(1) - 0.25)


def _hypothesis_q(U):
    """Continuation value that depends only on the belief over hypotheses: Q(b, c) = sum_k w_k U[k, c]."""
    def q(model, R, fast, goal, cands, ob, dt):
        return torch.softmax(fast.logw.float(), -1) @ U
    return q


def test_voi_is_zero_for_an_uninformative_outcome_and_positive_for_a_discriminating_one():
    m = deterministic(dom_tiny())
    state, vs0, names = voi_state(m)
    B = vs0.R.shape[0]
    K = m.cfg.n_particles
    f = vs0.fast.clone()
    f.logw = torch.full_like(f.logw, -float(torch.log(torch.tensor(float(K)))))      # undecided between hypotheses
    vs = VOIState(vs0.R, f, vs0.goal, vs0.prospective, vs0.dt)
    U = torch.zeros(K, 2)
    U[: K // 2, 0] = 1.0                    # candidate 0 pays under the first half of the hypotheses
    U[K // 2:, 1] = 1.0                     # candidate 1 under the second half
    q = _hypothesis_q(U)
    info = [[Action(tool=names[0])] for _ in range(B)]
    nxt = [[Action(tool=names[0]), Action(tool=names[1])] for _ in range(B)]
    # uninformative: every particle predicts the same outcome distribution (no particle conditioning)
    with torch.no_grad():
        film = m.outcome_head.film.weight.clone()
        m.outcome_head.film.weight.zero_()
        flat = decision_voi(m, vs, info, 16, next_actions=nxt, q_fn=q, seed=1)
        assert float(flat["voi"].abs().max()) < 1e-5
        # discriminating: strong particle-conditioned predictions (half the particles predict high, half low)
        m.outcome_head.film.weight.copy_(film * 8.0)
        p = vs.fast.particles.clone()
        p[:, : K // 2] = p[:, : K // 2].abs() + 1.0
        p[:, K // 2:] = -(p[:, K // 2:].abs() + 1.0)
        vs2 = VOIState(vs.R, type(vs.fast)(p, vs.fast.logw, vs.fast.pscale, vs.fast.bayes_mu, vs.fast.bayes_P,
                                         vs.fast.bayes_noise, vs.fast.bayes_n, vs.fast.surprise, vs.fast.steps,
                                         [dict(c) for c in vs.fast.channels]), vs.goal, vs.prospective, vs.dt)
        sharp = decision_voi(m, vs2, info, 32, next_actions=nxt, q_fn=q, seed=1)
    print("VOI uninformative", flat["voi"].flatten().tolist(), "discriminating", sharp["voi"].flatten().tolist())
    assert float(sharp["voi"].min()) > 0.2          # base 0.5 (undecided); the outcome resolves the hypothesis
