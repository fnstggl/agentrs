"""Latent reasoning R_{j+1} = Reason(R_j, Z, B, M, G) (A03): depth and interventions change decisions; halting."""
import pytest
import torch
import torch.nn as nn

from superlearner.cam_unified.core.backbone import UnifiedCore
from superlearner.cam_unified.core.reasoning import Reasoner
from superlearner.cam_unified.interfaces import TokenKind, TokenSet
from superlearner.cam_unified.testing import rand_tokens, tiny_config


def _setup(seed=0, **over):
    torch.manual_seed(seed)
    cfg = tiny_config(**over)
    core = UnifiedCore(cfg)
    rs = Reasoner(cfg, core)
    B = 64
    Z, _ = core(core.init_state(B), rand_tokens(B, 12, cfg.d_model, seed=seed))
    Z = Z.detach()
    ctx = TokenSet.cat(rand_tokens(B, cfg.n_particles + 1, cfg.d_model, TokenKind.BELIEF, ragged=False, seed=seed + 1),
                       rand_tokens(B, 3, cfg.d_model, TokenKind.GOAL, seed=seed + 2))
    policy = nn.Linear(cfg.d_model, 6)          # a decision head reading R (stand-in for the action generator)
    return cfg, core, rs, Z, ctx, policy


def _decide(policy, R):
    return policy(R.mean(1))


def test_reasoning_depth_changes_state_and_decisions():
    cfg, core, rs, Z, ctx, policy = _setup()
    rs.eval()
    outs = {J: rs(Z, ctx, n_iters=J) for J in cfg.reason_iters}
    for J, (R, info) in outs.items():
        assert R.shape == Z.shape and torch.isfinite(R).all()
        assert torch.all(info["iters"] == J) and info["halt_p"].shape == (Z.shape[0], J)
        assert torch.allclose(info["ponder"], cfg.reason_compute_cost * info["iters"])
    Js = list(outs)
    for a, b in zip(Js, Js[1:]):
        assert (outs[a][0] - outs[b][0]).abs().max() > 1e-3, (a, b)
    d1 = _decide(policy, outs[1][0]).argmax(-1)
    d16 = _decide(policy, outs[16][0]).argmax(-1)
    changed = (d1 != d16).float().mean()
    assert changed > 0.05, f"depth must change some decisions, changed {changed:.2f}"
    assert (outs[1][0] - Z).abs().max() > 1e-3, "one iteration must transform the state"


def test_reasoning_uses_its_context():
    cfg, core, rs, Z, ctx, policy = _setup()
    rs.eval()
    R1, _ = rs(Z, ctx, n_iters=4)
    ctx2 = TokenSet(ctx.x.clone(), ctx.mask, ctx.kind, ctx.time, ctx.entity, ctx.refs)
    ctx2.x[:, :cfg.n_particles + 1] = torch.randn_like(ctx2.x[:, :cfg.n_particles + 1])
    R2, _ = rs(Z, ctx2, n_iters=4)
    assert (R1 - R2).abs().max() > 1e-3, "beliefs in the context must affect reasoning"


def test_intervention_on_R_changes_downstream_decisions():
    cfg, core, rs, Z, ctx, policy = _setup()
    rs.eval()
    R, _ = rs(Z, ctx, n_iters=4)
    R_again, _ = rs(Z, ctx, n_iters=4)
    assert torch.equal(R, R_again)
    logits = _decide(policy, R)
    target = (logits.argmax(-1) + 1) % 6
    # push the state toward another action's readout direction at iteration 2
    direction = (policy.weight[target] - policy.weight[logits.argmax(-1)])          # [B,D]
    direction = direction / direction.norm(dim=-1, keepdim=True)

    def push(j, Rj):
        return Rj + 25.0 * direction.unsqueeze(1) if j == 4 else Rj

    with rs.intervention(push):
        Rp, _ = rs(Z, ctx, n_iters=4)
    assert rs.intervene is None
    dp = _decide(policy, Rp).argmax(-1)
    assert (dp == target).float().mean() > 0.9
    # an early-iteration perturbation propagates through the remaining iterations
    with rs.intervention(lambda j, Rj: Rj + torch.randn_like(Rj) if j == 1 else Rj):
        Rn, _ = rs(Z, ctx, n_iters=4)
    assert (Rn - R).abs().mean() > 1e-2


def test_gradients_reach_reused_backbone_blocks_halting_and_iteration_embedding():
    cfg, core, rs, Z, ctx, policy = _setup(n_layers=3, reason_layers=2)
    rs.train()
    R, info = rs(Z.requires_grad_(), ctx)
    loss = _decide(policy, R).logsumexp(-1).mean() + info["ponder"].mean() + 0.01 * info["halt_kl"].mean()
    loss.backward()
    reused = list(rs.blocks)
    assert len(reused) == 2 and reused[0] is core.blocks[1] and reused[1] is core.blocks[2]
    for blk in reused:
        assert all(p.grad is not None and p.grad.abs().max() > 0 for p in blk.parameters())
    assert all(p.grad is None for p in core.blocks[0].parameters()), "only the last reason_layers blocks are reused"
    assert rs.halt_head.weight.grad.abs().max() > 0 and rs.iter_emb.grad.abs().max() > 0
    assert Z.grad.abs().max() > 0
    own = rs.count_parameters()
    assert own["own"] < own["shared_blocks_reused"], "the reasoner owns little: it reuses backbone weights"


def test_ponder_training_distribution():
    cfg, core, rs, Z, ctx, policy = _setup()
    rs.train()
    R, info = rs(Z, ctx)
    P = info["halt_p"]
    assert torch.allclose(P.sum(1), torch.ones(P.shape[0]), atol=1e-5) and (P >= 0).all()
    steps = torch.arange(1, P.shape[1] + 1, dtype=P.dtype)
    assert torch.allclose(info["iters"], (P * steps).sum(1), atol=1e-5)
    assert torch.allclose(info["ponder"], cfg.reason_compute_cost * info["iters"])
    assert (info["halt_kl"] >= -1e-6).all()
    assert 1.0 < float(info["iters"].detach().mean()) < cfg.max_reason_iters


def test_adaptive_halting_per_row_and_real_compute_savings():
    """Rows whose state signals 'easy' halt after one iteration and leave the batch; the others continue."""
    cfg, core, rs, Z, ctx, policy = _setup()
    rs.eval()
    B = Z.shape[0]
    easy = torch.arange(B) < B // 2
    v = torch.randn(cfg.d_model)
    v = v / v.norm()
    with torch.no_grad():
        rs.halt_head.weight.copy_(5.0 * v.unsqueeze(0))
        rs.halt_head.bias.fill_(-3.0)
    rows_seen = []
    orig_step = rs.step

    def counting_step(R, j, hctx, cmask):
        rows_seen.append(R.shape[0])
        return orig_step(R, j, hctx, cmask)
    rs.step = counting_step

    def mark(j, Rj):                       # make "easy" rows' state point along v (high halting probability)
        if Rj.shape[0] == B:
            return Rj + (easy.float().view(B, 1, 1) * 10.0) * v
        return Rj
    with rs.intervention(mark):
        R, info = rs(Z, ctx)
    it = info["iters"]
    assert torch.all(it[easy] == 1) and torch.all(it[~easy] > 1)
    assert rows_seen[0] == B and rows_seen[1] == B - int(easy.sum()), rows_seen
    assert torch.allclose(info["halt_p"].sum(1), torch.ones(B), atol=1e-5)


def test_compute_penalty_trains_the_halting_policy():
    """The expected-iterations cost is a real gradient signal: penalising compute lowers E[iters], rewarding it
    raises E[iters]."""
    results = {}
    for sign in (+1.0, -1.0):
        cfg, core, rs, Z, ctx, policy = _setup(seed=1, reason_compute_cost=0.05)
        rs.train()
        opt = torch.optim.Adam(rs.parameters(), lr=0.05)
        with torch.no_grad():
            start = float(rs(Z[:16], ctx.select(list(range(16))))[1]["iters"].mean())
        for _ in range(25):
            _, info = rs(Z[:16], ctx.select(list(range(16))))
            loss = sign * info["ponder"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            end = float(rs(Z[:16], ctx.select(list(range(16))))[1]["iters"].mean())
        results[sign] = (start, end)
    assert results[1.0][1] < results[1.0][0] - 0.5, results
    assert results[-1.0][1] > results[-1.0][0] + 0.5, results


def test_reasoning_switch_off_returns_latent_state():
    cfg, core, rs, Z, ctx, policy = _setup(use_reasoning=False)
    R, info = rs(Z, ctx)
    assert torch.equal(R, Z) and torch.all(info["iters"] == 0)


def test_fixed_mode_when_adaptive_disabled():
    cfg, core, rs, Z, ctx, policy = _setup(adaptive_reasoning=False)
    rs.train()
    R, info = rs(Z, ctx)
    assert torch.all(info["iters"] == rs.default_iters)


# ---------------------------------------------------------------------------- depth matters after training (slow)

def _lookup_batch(B, cfg, g, hops=1, M=6, C=6, Dc=16, device="cpu"):
    """Memory tokens: node code c_n, next node's code, one-hot label of the next node; a GOAL token holds the start
    node's code. Answer: the label reached after `hops` next-pointer steps. With hops=1 this needs a query-dependent
    lookup (find the token whose code equals the goal's code), i.e. two dependent attention steps."""
    import torch.nn.functional as F
    codes = F.normalize(torch.randn(B, M, Dc, generator=g), dim=-1)
    perm = torch.stack([torch.randperm(M, generator=g) for _ in range(B)])
    labels = torch.randint(0, C, (B, M), generator=g)
    x = torch.zeros(B, M + 1, cfg.d_model)
    x[:, :M, :Dc] = codes
    x[:, :M, Dc:2 * Dc] = torch.gather(codes, 1, perm.unsqueeze(-1).expand(-1, -1, Dc))
    x[:, :M, 2 * Dc:2 * Dc + C] = F.one_hot(torch.gather(labels, 1, perm), C).float()
    s = torch.randint(0, M, (B,), generator=g)
    ar = torch.arange(B)
    x[:, M, :Dc] = codes[ar, s]
    x[:, M, 3 * Dc] = 1.0
    node = s.clone()
    for _ in range(hops - 1):
        node = perm[ar, node]
    y = torch.gather(labels, 1, perm)[ar, node]
    kind = torch.full((B, M + 1), int(TokenKind.MEMORY))
    kind[:, M] = int(TokenKind.GOAL)
    ts = TokenSet(x, torch.ones(B, M + 1, dtype=torch.bool), kind, torch.zeros(B, M + 1),
                  torch.full((B, M + 1), -1), None)
    return ts.to(device), y.to(device)


def _train_lookup(J, steps, device, seed=0):
    import torch.nn.functional as F
    torch.manual_seed(seed)
    cfg = tiny_config(reason_layers=1, adaptive_reasoning=False)
    core = UnifiedCore(cfg, autocast=False).to(device)
    rs = Reasoner(cfg, core).to(device)
    head = nn.Linear(cfg.d_model, 6).to(device)
    opt = torch.optim.Adam(list(core.parameters()) + list(rs.parameters()) + list(head.parameters()), lr=1e-3)
    g = torch.Generator().manual_seed(seed + 1)
    for _ in range(steps):
        ctx, y = _lookup_batch(128, cfg, g, device=device)
        R, _ = rs(core.init_state(128, device), ctx, n_iters=J)
        loss = F.cross_entropy(head(R.mean(1)), y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    def acc(n_iters, intervene=None, target_shift=0):
        with torch.no_grad():
            ctx, y = _lookup_batch(2048, cfg, torch.Generator().manual_seed(99), device=device)
            if intervene is None:
                R, _ = rs(core.init_state(2048, device), ctx, n_iters=n_iters)
            else:
                with rs.intervention(intervene):
                    R, _ = rs(core.init_state(2048, device), ctx, n_iters=n_iters)
            return float((head(R.mean(1)).argmax(-1) == torch.roll(y, target_shift, 0)).float().mean())
    return acc, rs


@pytest.mark.slow
def test_trained_reasoning_depth_is_necessary_for_decisions():
    """A03 ablation at matched training budget: a decision that needs a query-dependent lookup (two dependent
    attention steps through one reused block) is learned with J=2 latent iterations and not with J=1; on the J=2
    model, forcing J=1 at test time destroys the decisions, and transplanting another row's final reasoning state
    makes the decision follow that row — the reasoning output is what the decision reads.
    (Measured: swapping R after iteration 1 alone does not change decisions — the intermediate result of this task
    is held by the carried context stream, which the intervention hook does not touch.)"""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    acc2, _ = _train_lookup(2, 3000, dev)
    acc1, _ = _train_lookup(1, 3000, dev)
    a22, a21, a1 = acc2(2), acc2(1), acc1(1)
    a2_r1 = acc2(2, lambda j, R: torch.roll(R, 1, 0) if j == 1 else R)
    a2_final = acc2(2, lambda j, R: torch.roll(R, 1, 0) if j == 2 else R)
    a2_final_follow = acc2(2, lambda j, R: torch.roll(R, 1, 0) if j == 2 else R, target_shift=1)
    print(f"lookup decisions: J=2 trained {a22:.3f}; same model forced J=1 {a21:.3f}; J=1 trained {a1:.3f}; "
          f"R_1 swapped {a2_r1:.3f}; final R swapped: own label {a2_final:.3f}, donor's label {a2_final_follow:.3f}")
    assert a22 > 0.9
    assert a1 < 0.6 and a21 < 0.6
    assert a2_final < 0.3 and a2_final_follow > 0.9
