"""W6 follow-ups requested by W1 / W3 / W8 (plan 2026-11-26 §3 W6 interfaces):

* `WorldModel.prior / transition / filter / loss / forecast` take the semantic context (``sem`` + ``sem_rows``) and
  the entity slot-code row map (``code_rows``) of batches whose rows are not the episode rows (B*K decision
  candidates, flattened (t, b) and (start, b, sample) rows of the world objective);
* the latent read-out residual is at the state's own scale (cores without unit-RMS states);
* simulated counts stay in the representable range (surprise simulation on an over-dispersed head raised);
* observation-only outcome batches accept reward None.
"""
import pytest
import torch

from superlearner.cam_unified.core.backbone import UnifiedCore
from superlearner.cam_unified.interfaces import Likelihood, TokenKind, TokenSet
from superlearner.cam_unified.testing import rand_sequence, tiny_config
from superlearner.cam_unified.world.dynamics import WorldModel
from superlearner.cam_unified.world.outcomes import MixedOutcomeDist

DOM_WM = dict(n_layers=3, wm_transition_blocks=2, wm_ema_target=False, kl_free_bits_group=0.02, wm_consistency=1.0)


def dom_wm(seed=0, core_cls=UnifiedCore, **over):
    cfg = tiny_config(**{**DOM_WM, **over})
    torch.manual_seed(seed)
    return WorldModel(cfg, core_cls(cfg), train_horizons=(1, 2), open_loop_samples=2), cfg


def _inputs(cfg, B, g, entity=True):
    S, D = cfg.n_state, cfg.d_model
    R = torch.randn(B, S, D, generator=g)
    act = TokenSet.from_dense(torch.randn(B, 2, D, generator=g), TokenKind.ACTION)
    if entity:
        act = TokenSet(act.x, act.mask, act.kind, act.time, torch.tensor([[0, 1]] * B), act.refs, act.entity2)
    z = torch.zeros(B, cfg.stoch_groups * cfg.stoch_classes)
    z[:, ::cfg.stoch_classes] = 1.0
    return R, act, torch.ones(B), z


def test_prior_maps_candidate_rows_to_their_episode_slot_codes():
    wm, cfg = dom_wm()
    B, K = 2, 3
    g = torch.Generator().manual_seed(0)
    wm.core.new_episode(B, generator=torch.Generator().manual_seed(5))
    R, act, dt, z = _inputs(cfg, B, g)
    with torch.no_grad():
        per_env = wm.transition(R, act, None, dt, z)                       # rows == episode rows
        rows = torch.arange(B).repeat_interleave(K)                         # B*K candidates -> environment row
        Rk = R.repeat_interleave(K, 0)
        ak = TokenSet(act.x.repeat_interleave(K, 0), act.mask.repeat_interleave(K, 0),
                      act.kind.repeat_interleave(K, 0), act.time.repeat_interleave(K, 0),
                      act.entity.repeat_interleave(K, 0), None, None)
        with pytest.raises(ValueError):
            wm.transition(Rk, ak, None, dt.repeat_interleave(K), z.repeat_interleave(K, 0))
        Zk, _ = wm.prior(Rk, ak, None, dt.repeat_interleave(K), z=z.repeat_interleave(K, 0), code_rows=rows)
    torch.testing.assert_close(Zk, per_env.repeat_interleave(K, 0), rtol=1e-5, atol=1e-5)


class _SemCore(UnifiedCore):
    """Unit-test double: records the semantic row maps the world model passes (W3's keyword signature)."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.seen = []

    def project_semantic(self, sem, B, code_rows=None, sem_rows=None):
        if sem is None:                     # W3's real core asks for a projection on every pass (None -> None)
            return None
        self.seen.append(("project", B, None if sem_rows is None else sem_rows.tolist()))
        return {"kv": sem, "rows": sem_rows}

    def run_blocks(self, h, key_mask, blocks=None, attn_bias=None, checkpoint_blocks=None, sem=None, **kw):
        self.seen.append(("run_blocks", h.shape[0], sem is not None))
        return super().run_blocks(h, key_mask, blocks)

    def forward(self, Z_prev, ctx, code_rows=None, sem=None, *, sem_rows=None, blocks=None):
        self.seen.append(("forward", Z_prev.shape[0], None if sem_rows is None else sem_rows.tolist()))
        return super().forward(Z_prev, ctx, code_rows)


def test_semantic_context_reaches_every_transition_with_the_row_map():
    wm, cfg = dom_wm(core_cls=_SemCore)
    B, T = 3, 4
    seq = rand_sequence(B, T, cfg)
    sem = object()                                                        # opaque: only routed, never read here
    wm.core.seen.clear()
    with torch.no_grad():
        wm.loss(seq, sem=sem, generator=torch.Generator().manual_seed(0))
    proj = [s for s in wm.core.seen if s[0] == "project"]
    fwd = [s for s in wm.core.seen if s[0] == "forward"]
    assert fwd and all(n == B for _, n, _ in fwd)                          # the world model's own filter: B rows
    assert proj, "the block transition never received the semantic context"
    for _, n, rows in proj:                                               # every flattened row -> its batch row
        assert rows is not None and len(rows) == n and set(rows) <= set(range(B))
    # teacher-forced rows (t, b) are ordered t-major: b = row % B
    tf = proj[0]
    assert tf[2] == [i % B for i in range(tf[1])]
    seen = wm.core.seen                                                   # each projection feeds the next blocks
    assert all(seen[i + 1][0] == "run_blocks" and seen[i + 1][2] for i, s in enumerate(seen) if s[0] == "project")
    # a source-aligned context: rows composed with sem_rows
    wm.core.seen.clear()
    src = torch.tensor([4, 0, 2])
    with torch.no_grad():
        wm.loss(seq, sem=sem, sem_rows=src, generator=torch.Generator().manual_seed(0))
    tf = [s for s in wm.core.seen if s[0] == "project"][0]
    assert tf[2] == [int(src[i % B]) for i in range(tf[1])]


class _SeamCore(UnifiedCore):
    """A core without Mechanism C inputs (the seam-0 signatures): forward and run_blocks take no `sem`."""

    def forward(self, Z_prev, ctx, code_rows=None):
        return super().forward(Z_prev, ctx, code_rows)

    def run_blocks(self, h, key_mask, blocks=None, attn_bias=None, checkpoint_blocks=None):
        return super().run_blocks(h, key_mask, blocks, attn_bias, checkpoint_blocks)


def test_a_core_without_semantic_input_ignores_sem_and_counts_it():
    wm, cfg = dom_wm(core_cls=_SeamCore)        # W3's merged UnifiedCore takes sem; the seam-0 signature does not
    g = torch.Generator().manual_seed(1)
    R, act, dt, z = _inputs(cfg, 2, g, entity=False)
    with torch.no_grad():
        a = wm.transition(R, act, None, dt, z)
        b = wm.transition(R, act, None, dt, z, sem=object(), sem_rows=torch.tensor([0, 1]))
    torch.testing.assert_close(a, b)
    assert wm.sem_ignored == 1


def test_latent_readout_residual_is_at_the_state_scale():
    wm, cfg = dom_wm()
    torch.nn.init.normal_(wm.latent_pred.l2.weight, std=0.5)
    g = torch.Generator().manual_seed(2)
    S, D = cfg.n_state, cfg.d_model
    Zh = torch.randn(4, S, D, generator=g)
    tgt = torch.zeros_like(Zh)
    one = torch.ones(S, D)
    wm.state_norm.eval()
    with torch.no_grad():
        big = wm._latent_dist(Zh, tgt, one)
        small = wm._latent_dist(0.01 * Zh, tgt, one)
    # the prediction (state + residual) scales with the state, so the error scales by ~0.01^2 (StateNorm's running
    # standardisation is affine, not exactly scale-invariant); an O(1) residual would make `small` ~ `big`
    ratio = small / (big * 1e-4)
    assert bool(((ratio > 0.5) & (ratio < 2.0)).all()), ratio


def test_simulated_counts_stay_in_the_representable_range():
    Mx = 4
    edges = torch.linspace(0, 1, 5)
    sizes = 2 + 3 + 1 + Mx + 2 + edges.numel()
    raw = torch.zeros(64, sizes)
    o = 2 + 3 + 1 + Mx
    raw[:, o] = 60.0               # log-mean far above count_log_max (softbound -> ~count_log_max)
    raw[:, o + 1] = -30.0          # very over-dispersed (small total count)
    d = MixedOutcomeDist(raw, torch.full((64,), int(Likelihood.COUNT)), torch.zeros(64, dtype=torch.long), edges, Mx)
    y = d.sample(torch.Generator().manual_seed(0))
    import math
    assert float(y.max()) <= math.exp(d.count_log_max)
    assert torch.isfinite(d.log_prob(y)).all()       # previously: ValueError "observed count ... not an integer in"


def test_outcomes_from_results_accept_unobserved_reward():
    from superlearner.cam_unified.interfaces import Action, ExecutionResult
    from superlearner.cam_unified.outcome_spec import CORE_INDEX, outcomes_from_results
    res = [ExecutionResult(ok=True, cost=0.0, duration=1.0) for _ in range(2)]
    acts = [Action(tool="a") for _ in range(2)]
    enc = lambda texts: torch.zeros(len(texts), 8)                        # noqa: E731
    r = CORE_INDEX["reward"]
    for rew, want in ((None, [False, False]), ([None, 1.5], [False, True]), ([0.0, 2.0], [True, True])):
        ob, specs = outcomes_from_results([[]] * 2, acts, res, [[]] * 2, rew, enc, d_model=8, return_specs=True)
        assert [bool(s[r].observed) for s in specs] == want
        assert [bool(ob.y_mask[b, r]) for b in range(2)] == want


def test_points_probe_env_shows_the_payout_and_keeps_the_probe_dynamics():
    from superlearner.cam_unified.capabilities.causal import PointsProbeEnv
    from superlearner.cam_unified.interfaces import Action
    env = PointsProbeEnv(4, horizon=3)
    ob = env.reset([0, 1, 2, 3])

    def pts(obs):
        return [o.value for o in obs if o.unit == "points"]
    assert all(pts(o) == [0.0] for o in ob.observations)
    r1 = env.execute([Action("run_test")] * 4)
    assert all(pts(r.observations) == [-1.0] for r in r1)
    sig = [[o.value for o in r.observations if o.unit == "signal"][0] for r in r1]
    r2 = env.execute([Action("launch")] * 4)
    for s, r in zip(sig, r2):
        assert pts(r.observations) == [-1.0 + (10.0 if s > 0 else -10.0)]   # informative test: signal == state


def test_filter_sequence_passes_the_semantic_context_to_core_reasoner_and_world_loss():
    from test_consistency import dom_tiny
    from superlearner.cam_unified.world import train_seq
    m = dom_tiny()
    seen = {"core": 0, "reasoner": 0, "loss": None}
    core_fwd, rsn_fwd, loss_fn = m.core.forward, m.reasoner.forward, m.world_model.loss

    def core_forward(Z_prev, ctx, code_rows=None, sem=None, *, sem_rows=None, blocks=None):
        seen["core"] += sem is not None
        return core_fwd(Z_prev, ctx, code_rows)

    def reasoner_forward(Z, ctx=None, n_iters=None, code_rows=None, sem=None, *, sem_rows=None):
        seen["reasoner"] += sem is not None
        return rsn_fwd(Z, ctx, n_iters)

    import functools

    @functools.wraps(loss_fn)                      # keeps WorldModel.loss's signature (single-filter detection)
    def loss(*a, **k):
        seen["loss"] = k.get("sem")
        return loss_fn(*a, **k)
    m.core.forward, m.reasoner.forward, m.world_model.loss = core_forward, reasoner_forward, loss
    seq = rand_sequence(2, 5, m.cfg)
    sem = object()
    filt = train_seq.filter_window(m, seq, None, sem=sem)
    assert seen["core"] == 5 and seen["reasoner"] == 5 and filt.sem is sem
    train_seq.world_loss(m, seq, filt)
    assert seen["loss"] is sem
    # without sem nothing is passed (the seam call shape)
    seen.update(core=0, reasoner=0)
    train_seq.filter_sequence(m, seq, None)
    assert seen["core"] == 0 and seen["reasoner"] == 0
    # the seam signature filter_sequence(model, seq, aux) reads the context from the replay aux

    class Aux:
        option_prev = option_steps_prev = None
        retrieved = [[] for _ in range(5)]
        now = [0.0] * 5
    aux = Aux()
    aux.sem = sem
    m.option_policy = None
    assert train_seq.filter_sequence(m, seq, aux).sem is sem and seen["core"] == 5
