"""W6 PRED (plan 2026-11-26 §3): the DOM world model on the real modules (UnifiedCore, Reasoner, BeliefModule,
BayesLastLayer, OutcomeHead, WorldModel; only the tokenizer / action generator / value / option / exploration heads
are the integration doubles, which this objective does not use).

* no EMA target: the latent target is the detached posterior of the agent's own filter (single filter);
* the prior runs on R_t through the last k core blocks shared by reference, identity-composed at init;
* the world loss trains those shared blocks, the reasoner and the adapters;
* per-group free bits, the consistency term, the mode-latent open-loop future;
* observation-only: the verified return is never a target;
* burn-in decisions carry no gradient;
* the consistency gap falls when the transition is trained.
"""
import torch

from superlearner.cam_unified.config import DOM_SWITCHES
from superlearner.cam_unified.integration_doubles import (DoubleActionGenerator, DoubleExplorationHead,
                                                          DoubleOptionPolicy, DoubleTokenizer, DoubleValueHead)
from superlearner.cam_unified.interfaces import SequenceBatch, StepBatch, TokenKind, TokenSet
from superlearner.cam_unified.model import build_model
from superlearner.cam_unified.testing import rand_sequence, tiny_config
from superlearner.cam_unified.world import train_seq
from superlearner.cam_unified.world._util import cat_tokens

INERT = ("sem_xattn_every", "dec_xattn_every", "n_intent", "n_candidates", "value_channel", "sem_lora",
         "sem_encoder_lora", "microbatch")


def dom_tiny(seed: int = 0, **over):
    """Tiny model with the DOM world-model / belief switches and the real W3/W6 modules."""
    sw = {k: v for k, v in DOM_SWITCHES.items() if k not in INERT}
    kw = {**sw, "preset": "dom_test", "n_layers": 3, "wm_transition_blocks": 2, "reason_layers": 1, "burn_in": 2,
          "bptt_steps": 4, "train_horizons": (1, 2, 4)}
    kw.update(over)
    cfg = tiny_config(**kw)
    torch.manual_seed(seed)
    comps = {"tokenizer": DoubleTokenizer(cfg), "action_generator": DoubleActionGenerator(cfg),
             "value_head": DoubleValueHead(cfg), "option_policy": DoubleOptionPolicy(cfg),
             "exploration_head": DoubleExplorationHead(cfg), "entity_encoder": None}
    return build_model(cfg, components=comps)


def no_censoring(seq):
    for st in seq.steps:
        st.outcomes.censored = torch.zeros_like(st.outcomes.censored)
    return seq


def test_no_ema_target_and_single_filter():
    m = dom_tiny()
    wm = m.world_model
    assert wm.target_core is None and not wm.use_ema
    assert all(not n.startswith("world_model.target_core") for n, _ in m.named_parameters())
    wm.update_target()            # no-op, no error
    wm.reset_target()
    seq = rand_sequence(3, 7, m.cfg)
    filt = m.filter_sequence(seq, None)
    assert train_seq.single_filter(m) and train_seq.prior_on_R(m)
    out = m.world_loss(seq, filt)
    burn = filt.burn_in
    assert burn == 2 and out["burn_in"] == 2 and out["graded_steps"] == 5
    # the world model trained on the agent filter's own (graded) states, not on a second filter
    torch.testing.assert_close(out["Z_post"], torch.stack(filt.Zs[burn:]).detach(), rtol=0, atol=0)
    # perturbing the agent filter's states changes the world loss (they are its input)
    filt.Zs[-1] = filt.Zs[-1] + 0.5
    assert float(m.world_loss(seq, filt)["world"]) != float(out["world"])


def test_block_transition_is_the_last_k_core_blocks_by_reference():
    m = dom_tiny()
    wm, core = m.world_model, m.core
    k = m.cfg.wm_transition_blocks
    blocks = wm.transition_block_list()
    assert len(blocks) == k and all(a is b for a, b in zip(blocks, list(core.blocks)[-k:]))
    # no transformer of its own: the only new parameters are the two adapters
    own = {id(p) for n, p in wm.named_parameters() if n.startswith("trans_")}
    assert own and not (own & {id(p) for p in core.parameters()})
    B, S, D = 2, m.cfg.n_state, m.cfg.d_model
    g = torch.Generator().manual_seed(1)
    R = torch.randn(B, S, D, generator=g)
    act = TokenSet.from_dense(torch.randn(B, 2, D, generator=g), TokenKind.ACTION)
    dt = torch.ones(B)
    z = torch.zeros(B, m.cfg.stoch_groups * m.cfg.stoch_classes)
    z[:, ::m.cfg.stoch_classes] = 1.0
    with torch.no_grad():
        Zh = wm.transition(R, act, None, dt, z)
        # adapters are zero at init: exactly the last k shared blocks on [embed_state(R); embed_ctx(ctx)]
        ctx = cat_tokens(act, wm.time_tokens(dt), wm.noise_tokens(z, dt))
        h = torch.cat([core.embed_state(R), core.embed_ctx(ctx)], 1)
        for blk in list(core.blocks)[-k:]:
            h = blk(h, None, None)
        ref = core.out_norm(h[:, :S]).float()
    torch.testing.assert_close(Zh, ref, rtol=1e-5, atol=1e-5)
    # the transition depends on the action
    act2 = TokenSet.from_dense(torch.randn(B, 2, D, generator=g), TokenKind.ACTION)
    with torch.no_grad():
        assert float((wm.transition(R, act2, None, dt, z) - Zh).abs().max()) > 1e-4


def test_world_loss_trains_the_shared_blocks_the_reasoner_and_the_adapters():
    m = dom_tiny()
    m.train()
    seq = rand_sequence(3, 7, m.cfg)
    out = m.world_loss(seq, m.filter_sequence(seq, None))
    out["world"].backward()
    k = m.cfg.wm_transition_blocks
    named = dict(m.named_parameters())
    last = f"core.blocks.{m.cfg.n_layers - 1}.ffn.w2.weight"
    for n in (last, "reasoner.iter_emb", "world_model.trans_out.l2.weight", "world_model.prior_net.out.weight",
              "beliefs.tok_m.weight", "core.blocks.0.ffn.w2.weight"):
        g = named[n].grad
        assert g is not None and float(g.abs().sum()) > 0, n
    assert out["consistency"].requires_grad is False or float(out["consistency"]) > 0
    assert float(out["reward"]) == 0.0 and named["world_model.rew_head.l2.weight"].grad is None
    assert "open_loop_mode" in out and "cons_mse_var" in out and "cons_mse_var_mode1" in out
    assert k == 2


def test_burn_in_steps_carry_no_gradient():
    m = dom_tiny()
    m.train()
    seq = rand_sequence(2, 7, m.cfg)
    filt = m.filter_sequence(seq, None)
    assert filt.burn_in == 2
    assert not filt.Zs[0].requires_grad and not filt.Zs[1].requires_grad and filt.Zs[2].requires_grad
    assert train_seq.burn_in_for(m.cfg, 5) == 1 and train_seq.burn_in_for(m.cfg, 3) == 0
    assert train_seq.burn_in_for(m.cfg, 30) == 2
    assert train_seq.burn_in_for(m.cfg, 30, burn_in=0) == 0


def test_observation_only_world_loss_never_reads_the_verified_reward():
    m = dom_tiny()
    m.eval()                       # no running-statistics update between the two evaluations
    seq = rand_sequence(3, 6, m.cfg, seed=3)
    filt = train_seq.filter_window(m, seq, None, burn_in=0)
    g = torch.Generator().manual_seed(5)
    a = m.world_model.loss(SequenceBatch(seq.steps), Z_post=torch.stack(filt.Zs), R_post=torch.stack(filt.Rs),
                           beliefs=filt.belief_tokens, generator=g)
    steps2 = [StepBatch(**{**s.__dict__, "reward": s.reward * 0 + 123.0}) for s in seq.steps]
    g = torch.Generator().manual_seed(5)
    b = m.world_model.loss(SequenceBatch(steps2), Z_post=torch.stack(filt.Zs), R_post=torch.stack(filt.Rs),
                           beliefs=filt.belief_tokens, generator=g)
    assert float(a["world"]) == float(b["world"]) and float(a["reward"]) == 0.0


def test_per_group_free_bits():
    m = dom_tiny()
    wm = m.world_model
    G, C = m.cfg.stoch_groups, m.cfg.stoch_classes
    fb = wm.free_bits_group
    assert fb == 0.02
    prior = torch.zeros(4, G, C, requires_grad=True)
    post = torch.zeros(4, G, C)
    post[:, 0, 0] = 4.0                              # only group 0 carries information (KL >> free bits)
    post[:, 1, 0] = 0.05                             # group 1: KL far below the free bits
    lp_prior = torch.log_softmax(prior, -1)
    lp_post = torch.log_softmax(post, -1)
    out = wm.kl_terms(lp_post, lp_prior, torch.ones(4, dtype=torch.bool))
    assert out["kl_groups_active"] == 1
    out["kl"].backward()
    gp = prior.grad.abs().sum((0, 2))
    assert float(gp[0]) > 0 and float(gp[1]) == 0.0 and float(gp[2:].sum()) == 0.0
    # every inactive group contributes exactly the floor
    kv = float(out["kl_value"])
    assert float(out["kl"]) >= (G - 1) * fb and kv > 0


def test_mode_latent_future_is_excluded_from_the_mixture():
    m = dom_tiny()
    wm = m.world_model
    assert wm.mode_latent
    R, s, Q = 3, 3, 2
    g = torch.Generator().manual_seed(0)
    ones = torch.ones(R, dtype=torch.bool)
    rt = {"R": R, "s": s, "alive": ones, "om": torch.ones(R, Q, dtype=torch.bool), "nxt_ok": ones,
          "out_lp": torch.randn(R, s, Q, generator=g), "r_ll": torch.zeros(R, s), "c_ll": torch.randn(R, s, generator=g),
          "dist": torch.rand(R, s, generator=g), "cons": torch.rand(R, s, generator=g),
          "obs_rows": torch.zeros(0, dtype=torch.long), "cm": torch.zeros(R, Q, dtype=torch.bool),
          "sq": torch.zeros(R, Q), "r_abs": torch.zeros(R), "mode_last": True}
    resp = wm._responsibilities(rt)
    assert float(resp["rew"][:, -1].abs().max()) == 0.0
    torch.testing.assert_close(resp["rew"].sum(1), torch.ones(R))
    assert rt["joint_ll"].shape == (R, s - 1)


def test_consistency_gap_falls_when_the_transition_is_trained():
    """A short world-objective run on a fixed small sequence set: the teacher-forced consistency gap
    MSE(Z^_{t+1}, Z_post_{t+1}) / Var(Z_post) and the consistency term decrease (the full acceptance, < 0.1 on the
    real dom_dev stack, is the integrated causal run's measurement)."""
    m = dom_tiny(seed=1)
    seqs = [rand_sequence(4, 6, m.cfg, seed=11 * i) for i in range(4)]
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=3e-3)
    first, last = [], []
    for it in range(60):
        m.train()
        seq = seqs[it % len(seqs)]
        out = m.world_loss(seq, train_seq.filter_window(m, seq, None, burn_in=0))
        opt.zero_grad()
        out["world"].backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 10.0)
        opt.step()
        (first if it < 4 else last if it >= 56 else []).append((float(out["consistency"]),
                                                                  float(out["cons_mse_var"])))
    c0 = sum(a for a, _ in first) / len(first)
    c1 = sum(a for a, _ in last) / len(last)
    r0 = sum(b for _, b in first) / len(first)
    r1 = sum(b for _, b in last) / len(last)
    print(f"consistency {c0:.3f} -> {c1:.3f}; MSE/Var {r0:.3f} -> {r1:.3f}")
    assert c1 < 0.5 * c0 and r1 < 0.5 * r0


def test_self_predictive_terms_never_shape_the_encoder_without_an_ema_target():
    """Without an EMA target the latent target is the model's own detached posterior; the latent and consistency
    terms then train only the transition's output adapter and the latent read-out (a context-insensitive filter
    would otherwise be a trivial fixed point: measured collapse to climatology on DelayWorld with the real core)."""
    m = dom_tiny()
    m.train()
    wm = m.world_model
    assert wm.latent_grad == "transition"
    seq = rand_sequence(3, 6, m.cfg, seed=2)
    out = m.world_loss(seq, train_seq.filter_window(m, seq, None, burn_in=0))
    named = dict(m.named_parameters())
    last = f"core.blocks.{m.cfg.n_layers - 1}.ffn.w2.weight"
    probe = {n: named[n] for n in ("core.blocks.0.ffn.w2.weight", last, "reasoner.iter_emb", "core.out_norm.weight",
                                   "world_model.trans_in.weight", "world_model.trans_out.l2.weight",
                                   "world_model.latent_pred.l2.weight")}
    g = torch.autograd.grad(out["consistency"] + out["latent"], list(probe.values()), allow_unused=True,
                            retain_graph=True)
    gs = {n: (0.0 if x is None else float(x.abs().sum())) for n, x in zip(probe, g)}
    for n in ("core.blocks.0.ffn.w2.weight", last, "reasoner.iter_emb", "core.out_norm.weight",
              "world_model.trans_in.weight"):
        assert gs[n] == 0.0, (n, gs[n])
    assert gs["world_model.trans_out.l2.weight"] > 0 and gs["world_model.latent_pred.l2.weight"] > 0
    # the predictive terms still train the shared blocks and the reasoner
    g2 = torch.autograd.grad(out["outcome"] + out["open_loop"], [probe["reasoner.iter_emb"], probe[last]])
    assert all(float(x.abs().sum()) > 0 for x in g2)


def test_ema_target_on_the_agent_filter_path():
    """With cfg.wm_ema_target=True the single-filter path still works: the latent targets are the EMA encoder's
    states on the agent filter's own contexts (equal to the online states while the copies agree), and the
    self-predictive gradients reach the shared encoder (latent_grad 'full')."""
    m = dom_tiny(wm_ema_target=True)
    m.eval()
    wm = m.world_model
    assert wm.target_core is not None and wm.latent_grad == "full" and train_seq.single_filter(m)
    seq = rand_sequence(3, 6, m.cfg, seed=4)
    filt = train_seq.filter_window(m, seq, None, burn_in=0)
    tgt = train_seq.ema_targets(m, filt)
    torch.testing.assert_close(tgt, torch.stack(filt.Zs).detach(), rtol=1e-5, atol=1e-5)
    g = torch.Generator().manual_seed(1)
    a = float(m.world_loss(seq, filt)["latent"])
    with torch.no_grad():
        for p in wm.target_core.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    assert float(m.world_loss(seq, filt)["latent"]) != a          # the target encoder defines the latent target
    wm.update_target(0.5)
