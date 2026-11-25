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
    """Mechanics (untrained): every depth J yields a different state and some different readout decisions. The
    trained evidence that depth is *necessary* is test_trained_reasoning_depth_is_necessary_for_decisions (slow)."""
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
    """Mechanics of the R hook (untrained; the final-iteration push holds by construction of a linear readout). The
    trained mediation evidence is in the slow lookup test (context-stream ablation / patching at j < J)."""
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


def _entity_ctx(cfg, B, seed=0):
    """Belief + goal tokens and entity-bound memory tokens (slot codes matter)."""
    g = torch.Generator().manual_seed(seed)
    mem = rand_tokens(B, 6, cfg.d_model, TokenKind.MEMORY, ragged=False, seed=seed + 3)
    mem.entity = torch.randint(0, cfg.max_entities, (B, 6), generator=g)
    return TokenSet.cat(rand_tokens(B, cfg.n_particles + 1, cfg.d_model, TokenKind.BELIEF, ragged=False, seed=seed),
                        rand_tokens(B, 3, cfg.d_model, TokenKind.GOAL, seed=seed + 2), mem)


@pytest.mark.parametrize("mode", ["fixed", "act_train", "act_eval", "pondernet_eval"])
def test_reasoner_runs_on_imagination_branches_with_code_rows(mode):
    """After core.new_episode(B), branch batches of B' != B rows run through the reasoner with code_rows (as through
    the core), and each branch row equals the reasoner output of its environment row."""
    torch.manual_seed(0)
    cfg = tiny_config()
    core = UnifiedCore(cfg)
    rs = Reasoner(cfg, core, halting="pondernet" if mode.startswith("pondernet") else "act")
    B = 4
    core.new_episode(B, generator=torch.Generator().manual_seed(1))
    ctx = _entity_ctx(cfg, B)
    with torch.no_grad():
        Z, _ = core(core.init_state(B), ctx)
    rows = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    Zb, cb = Z[rows], ctx.select(rows.tolist())
    rs.train(mode == "act_train")
    kw = {"n_iters": 3} if mode == "fixed" else {}
    with pytest.raises(ValueError, match="code_rows"):
        rs(Zb, cb, **kw)
    R_env, i_env = rs(Z, ctx, **kw)
    R_br, i_br = rs(Zb, cb, code_rows=rows, **kw)
    assert R_br.shape == Zb.shape and torch.isfinite(R_br).all()
    assert torch.allclose(R_br, R_env[rows], atol=1e-5), (R_br - R_env[rows]).abs().max()
    assert torch.equal(i_br["n_computed"], i_env["n_computed"][rows])
    # the slot codes really enter the reasoning context: a fresh episode permutation changes the result
    core.new_episode(B, generator=torch.Generator().manual_seed(2))
    R_new, _ = rs(Z, ctx, **kw)
    assert (R_new - R_env).abs().max() > 1e-4


def test_adaptive_training_unroll_is_bounded_per_row_and_checkpointed():
    """Default adaptive training: the unroll stops once each row's unhalted mass is below halt_eps (reachable from
    the initial lambda = 0.25: 11 iterations, not max_reason_iters = 16); rows that halt leave the batch in training
    too; iterations are activation-checkpointed by default."""
    cfg, core, rs, Z, ctx, policy = _setup()
    rs.train()
    calls = []
    orig = rs._step_impl

    def spy(R, hctx, mask, j):
        calls.append((j, R.shape[0], rs._ckpt_now))
        return orig(R, hctx, mask, j)
    rs._step_impl = spy
    R, info = rs(Z, ctx)
    n = info["n_computed"]
    assert int(n.max()) < cfg.max_reason_iters, n
    # exact stopping rule, recomputed from the per-iteration states: stop at the first j with prod(1 - lambda) < eps
    with torch.no_grad():
        lam = torch.stack([rs.halt_lambda(info["R_all"][:, j]) for j in range(info["R_all"].shape[1])], 1)
        unhalted = torch.cumprod(1 - lam, 1)
        first = (unhalted < rs.halt_eps).float().argmax(1) + 1
    assert torch.equal(first.float(), n), (first, n)
    rem = 1 - info["halt_p"][:, :-1].sum(1)                     # mass left before the last computed iteration
    assert (rem >= rs.halt_eps - 1e-6).all()
    assert info["R_all"].shape == (Z.shape[0], info["halt_p"].shape[1], cfg.n_state, cfg.d_model)
    assert all(c[2] for c in calls), "iterations are checkpointed in adaptive training"
    assert torch.allclose(info["halt_p"].sum(1), torch.ones(Z.shape[0]), atol=1e-5)
    # a hard cap
    rs_cap = Reasoner(cfg, core, max_iters=3)
    rs_cap.train()
    _, ic = rs_cap(Z, ctx)
    assert int(ic["n_computed"].max()) == 3 and ic["halt_p"].shape[1] == 3
    # per-row exit in training: rows that make lambda ~ 1 stop after iteration 1 and leave the batch
    B = Z.shape[0]
    easy = torch.arange(B) < B // 2
    v = torch.randn(cfg.d_model)
    v = v / v.norm()
    with torch.no_grad():
        rs.halt_head.weight.copy_(5.0 * v.unsqueeze(0))
        rs.halt_head.bias.fill_(-3.0)
    calls.clear()

    def mark(j, Rj):
        return Rj + (easy.float().view(B, 1, 1) * 10.0) * v if Rj.shape[0] == B else Rj
    with rs.intervention(mark):
        R, info = rs(Z, ctx)
    assert torch.all(info["n_computed"][easy] == 1) and torch.all(info["n_computed"][~easy] > 1)
    assert calls[0][1] == B and calls[1][1] == B - int(easy.sum()), calls[:3]
    (R.pow(2).mean() + info["ponder"].mean()).backward()
    assert rs.halt_head.weight.grad.abs().max() > 0


def test_act_mode_is_the_same_function_in_training_and_evaluation():
    cfg, core, rs, Z, ctx, policy = _setup(seed=3)
    rs.train()
    R_tr, i_tr = rs(Z, ctx)
    rs.eval()
    with torch.no_grad():
        R_ev, i_ev = rs(Z, ctx)
    assert torch.allclose(R_tr, R_ev, atol=1e-5)
    assert torch.equal(i_tr["n_computed"], i_ev["n_computed"])
    assert torch.allclose(i_tr["halt_p"], i_ev["halt_p"], atol=1e-6)
    assert torch.allclose(i_tr["expected_iters"], i_ev["expected_iters"], atol=1e-5)
    R_mix = (i_tr["halt_p"][:, :, None, None] * i_tr["R_all"]).sum(1)
    assert torch.allclose(R_tr, R_mix, atol=1e-5), "ACT output = sum_j p_j R_j"


def test_iteration_checkpointing_gives_identical_values_and_gradients():
    cfg, core, rs, Z, ctx, policy = _setup(seed=4, n_layers=3, reason_layers=2)
    out = {}
    for ck in (False, True):
        rs.checkpoint_iters = ck
        rs.train()
        core.zero_grad()
        rs.zero_grad()
        R, info = rs(Z, ctx)
        (R.pow(2).mean() + info["ponder"].mean()).backward()
        out[ck] = (R.detach(), [p.grad.clone() for p in rs.blocks.parameters()], rs.halt_head.weight.grad.clone())
    assert torch.allclose(out[False][0], out[True][0], atol=1e-6)
    for a, b in zip(out[False][1], out[True][1]):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-4)
    assert torch.allclose(out[False][2], out[True][2], atol=1e-6, rtol=1e-4)


def test_pondernet_expected_loss_trains_halting_toward_the_best_iteration():
    """PonderNet objective sum_j p_j L(R_j): with L_j lowest at j = 3, training the halting head on it moves the
    halting distribution's mode to iteration 3."""
    cfg, core, rs, Z, ctx, policy = _setup(seed=5)
    rs = Reasoner(cfg, core, halting="pondernet")
    rs.train()
    Z, ctx = Z[:16], ctx.select(list(range(16)))
    with torch.no_grad():
        R3, _ = rs(Z, ctx, n_iters=3)

    def loss_fn(R):
        return (R - R3).pow(2).mean((1, 2))
    R, info = rs(Z, ctx)
    manual = sum(info["halt_p"][:, j] * loss_fn(info["R_all"][:, j]) for j in range(info["halt_p"].shape[1]))
    assert torch.allclose(Reasoner.expected_loss(loss_fn, info), manual, atol=1e-6)
    assert torch.allclose(info["R_all"][:, 2], R3, atol=1e-5), "R_all[j] is R_{j+1} of the same computation"
    js = info["sampled_iter"].long()
    assert torch.allclose(R, info["R_all"][torch.arange(16), js - 1]), "training output = R at a sampled step"
    opt = torch.optim.Adam(rs.parameters(), lr=0.05)
    p3_before = float(info["halt_p"][:, 2].detach().mean())
    for _ in range(60):
        _, info = rs(Z, ctx)
        loss = Reasoner.expected_loss(loss_fn, info).mean() * 100
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        _, info = rs(Z, ctx)
    P = info["halt_p"]
    print(f"pondernet: p(j=3) before {p3_before:.3f} after {float(P[:, 2].mean()):.3f}; "
          f"E[iters] {float(info['expected_iters'].mean()):.2f}")
    assert (P.argmax(1) == 2).float().mean() > 0.9 and float(P[:, 2].mean()) > p3_before + 0.3


def test_pondernet_evaluation_halts_by_the_training_distribution():
    """Sampled evaluation halting draws the halting step from the same (truncated) distribution training uses;
    threshold evaluation returns R at the median step."""
    cfg, core, rs, Z, ctx, policy = _setup(seed=6)
    rs = Reasoner(cfg, core, halting="pondernet", sample_halt=True)
    n = 3000
    Zr, cr = Z[:1].expand(n, -1, -1).contiguous(), ctx.select([0] * n)
    rs.train()
    with torch.no_grad():
        _, itr = rs(Z[:1], ctx.select([0]))
    p = itr["halt_p"][0]
    rs.eval()
    torch.manual_seed(0)
    with torch.no_grad():
        R, iev = rs(Zr, cr)
    freq = torch.bincount(iev["n_computed"].long() - 1, minlength=p.shape[0]).float() / n
    tv = 0.5 * (freq[:p.shape[0]] - p).abs().sum()
    assert tv < 0.05, (freq, p)
    rs.sample_halt = False
    with torch.no_grad():
        Rt, it = rs(Z[:1], ctx.select([0]))
    med = int(torch.nonzero(torch.cumsum(p, 0) >= 0.5)[0]) + 1
    assert int(it["n_computed"][0]) == med
    assert torch.allclose(Rt, itr["R_all"][:, med - 1], atol=1e-5)


def test_context_stream_intervention_hook():
    cfg, core, rs, Z, ctx, policy = _setup(seed=7)
    rs.eval()
    R0, _ = rs(Z, ctx, n_iters=3)
    seen = []

    def ident(j, R, C, rows):
        seen.append((j, tuple(C.shape), rows.tolist()))
        return R, C
    with rs.intervention(ident, context=True):
        R1, _ = rs(Z, ctx, n_iters=3)
    assert torch.equal(R0, R1) and [s[0] for s in seen] == [1, 2, 3]
    assert seen[0][1][0] == Z.shape[0] and seen[0][1][1] == cfg.n_state + ctx.N and seen[0][2] == list(range(64))

    def wipe_ctx(j, R, C, rows):                 # destroy what iteration 1 left in the context stream
        return R, (torch.randn_like(C) if j == 1 else C)
    with rs.intervention(wipe_ctx, context=True):
        R2, _ = rs(Z, ctx, n_iters=3)
    assert (R2 - R0).abs().max() > 1e-2
    assert rs.intervene is None


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
    return acc, rs, head


def _lookup_pairs(n, cfg, device, M=6, Dc=16, C=6, seed=123):
    """Counterfactual pairs: the same memory tokens, different start node in the GOAL token, different answers.
    Returns (ctx_a, y_a, ctx_b, y_b); the two contexts differ only in the goal token."""
    g = torch.Generator().manual_seed(seed)
    ctx, y_a = _lookup_batch(n, cfg, g)
    x = ctx.x
    codes, lab = x[:, :M, :Dc], x[:, :M, 2 * Dc:2 * Dc + C].argmax(-1)    # token m holds the label of next(m)
    ar = torch.arange(n)
    s_a = (codes * x[:, M:M + 1, :Dc]).sum(-1).argmax(1)
    assert torch.equal(lab[ar, s_a], y_a)
    s_b = torch.randint(0, M, (n,), generator=g)
    xb = x.clone()
    xb[:, M, :Dc] = codes[ar, s_b]
    y_b = lab[ar, s_b]
    keep = (y_b != y_a).nonzero().flatten()

    def ts(xx):
        return TokenSet(xx[keep], ctx.mask[keep], ctx.kind[keep], ctx.time[keep], ctx.entity[keep], None).to(device)
    return ts(x), y_a[keep].to(device), ts(xb), y_b[keep].to(device)


@pytest.mark.slow
def test_trained_reasoning_depth_is_necessary_for_decisions():
    """A03 ablation at matched training budget: a decision that needs a query-dependent lookup (two dependent
    attention steps through one reused block) is learned with J=2 latent iterations and not with J=1; on the J=2
    model, forcing J=1 at test time destroys the decisions.

    Intermediate reasoning state (j = 1 < J = 2) causally mediates the decision:
    * ablation: resetting the carried context stream after iteration 1 to its input embedding (undoing what
      iteration 1 computed) destroys the decisions;
    * activation patching: in counterfactual pairs whose contexts differ only in the goal token, copying the
      *memory-token* streams after iteration 1 from the other run (their inputs are identical in both runs, so they
      differ only by what iteration 1 wrote into them) makes the decision follow the other run's answer.
    (Measured as well: R_1 alone does not carry the intermediate result — patching R_1, the goal stream or the
    anchor stream leaves the decisions unchanged; the final-R transplant holds by construction of the readout.)"""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    acc2, rs, head = _train_lookup(2, 3000, dev)
    acc1, _, _ = _train_lookup(1, 3000, dev)
    a22, a21, a1 = acc2(2), acc2(1), acc1(1)
    a2_r1 = acc2(2, lambda j, R: torch.roll(R, 1, 0) if j == 1 else R)
    a2_final = acc2(2, lambda j, R: torch.roll(R, 1, 0) if j == 2 else R)
    a2_final_follow = acc2(2, lambda j, R: torch.roll(R, 1, 0) if j == 2 else R, target_shift=1)
    print(f"lookup decisions: J=2 trained {a22:.3f}; same model forced J=1 {a21:.3f}; J=1 trained {a1:.3f}; "
          f"R_1 swapped {a2_r1:.3f}; final R swapped: own label {a2_final:.3f}, donor's label {a2_final_follow:.3f}")
    assert a22 > 0.9
    assert a1 < 0.6 and a21 < 0.6
    assert a2_final < 0.3 and a2_final_follow > 0.9

    core, cfg = rs.core, rs.cfg
    S, M = cfg.n_state, 6
    ca, ya, cb, yb = _lookup_pairs(4096, cfg, dev)
    n = ca.B
    Z0 = core.init_state(n, dev)
    with torch.no_grad():
        donor = {}

        def capture(j, R, C, rows):
            donor[j] = (R.clone(), C.clone())
            return R, C
        with rs.intervention(capture, context=True):
            rs(Z0, cb, n_iters=2)
        C0, _ = rs._context(Z0, ca)                        # the context stream before any iteration

        def run(fn):
            with rs.intervention(fn, context=True):
                R, _ = rs(Z0, ca, n_iters=2)
            d = head(R.mean(1)).argmax(-1)
            return float((d == ya).float().mean()), float((d == yb).float().mean())

        def patch(sl):
            def fn(j, R, C, rows):
                if j == 1:
                    C = C.clone()
                    C[:, sl] = donor[1][1][:, sl]
                return R, C
            return fn
        plain = run(lambda j, R, C, rows: (R, C))
        ablate = run(lambda j, R, C, rows: (R, C0.clone()) if j == 1 else (R, C))
        p_mem = run(patch(slice(S, S + M)))
        p_goal = run(patch(slice(S + M, S + M + 1)))
        p_anchor = run(patch(slice(0, S)))
        p_r1 = run(lambda j, R, C, rows: (donor[1][0], C) if j == 1 else (R, C))
    print(f"mediation (own, donor) n={n}: plain {plain}; context stream reset after j=1 {ablate}; memory streams "
          f"patched {p_mem}; goal stream patched {p_goal}; anchor patched {p_anchor}; R_1 patched {p_r1}")
    assert plain[0] > 0.9
    assert ablate[0] < 0.5, "undoing iteration 1's context update must destroy the decisions"
    assert p_mem[1] > 0.9, "the intermediate result written into the memory streams decides"
