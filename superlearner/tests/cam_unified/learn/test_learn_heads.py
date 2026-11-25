"""Value head (symlog two-hot), option module (exact decision probabilities, termination, OPTION tokens) and the
BUDGET token encoder."""
import math

import pytest
import torch

from superlearner.cam_unified.interfaces import TokenKind, TokenSet
from superlearner.cam_unified.learn.heads import (BudgetEncoder, BudgetState, OptionModule, ValueHead, symexp, symlog,
                                                  twohot_encode, twohot_mean)
from superlearner.cam_unified.testing import rand_tokens, tiny_config


def test_twohot_roundtrip_and_symlog():
    bins = torch.linspace(-20, 20, 255)
    for v in [-12345.0, -3.0, 0.0, 0.25, 1e4]:
        t = twohot_encode(symlog(torch.tensor(v)), bins)
        assert t.sum().item() == pytest.approx(1.0)
        assert twohot_mean(torch.log(t + 1e-12), bins).item() == pytest.approx(v, rel=1e-3, abs=1e-3)
    assert torch.allclose(symexp(symlog(torch.tensor([-1e6, 0.0, 7.0]))), torch.tensor([-1e6, 0.0, 7.0]))


def test_value_head_starts_at_zero_and_fits_returns_of_very_different_scales():
    torch.manual_seed(0)
    cfg = tiny_config()
    D = cfg.d_model
    head = ValueHead(D, n_heads=4)
    B = 6
    R = torch.randn(B, cfg.n_state, D)
    goal = rand_tokens(B, 3, D, TokenKind.GOAL, seed=1)
    assert torch.allclose(head(R, goal).value, torch.zeros(B))
    target = torch.tensor([25_000.0, -800.0, 3.0, 0.05, -0.2, 0.0])     # cents of revenue ... success bonuses
    opt = torch.optim.Adam(head.parameters(), lr=3e-3)
    for _ in range(600):
        out = head(R, goal)
        loss = head.loss(out.logits, target).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    v = head(R, goal).value.detach()
    for got, want in zip(v.tolist(), target.tolist()):
        assert got == pytest.approx(want, rel=0.05, abs=0.05)


def test_value_head_ignores_padded_context_and_uses_real_context():
    torch.manual_seed(1)
    D = 32
    head = ValueHead(D, n_heads=4)
    torch.nn.init.normal_(head.mlp.fc2.weight, std=0.5)
    R = torch.randn(2, 4, D)
    g = rand_tokens(2, 5, D, TokenKind.GOAL, ragged=False, seed=2)
    g.mask[:, 3:] = False
    g2 = TokenSet(g.x.clone(), g.mask, g.kind, g.time, g.entity)
    g2.x[:, 3:] += 100.0                                              # change only padded tokens
    assert torch.allclose(head(R, g).value, head(R, g2).value, atol=1e-5)
    g3 = TokenSet(g.x.clone(), g.mask, g.kind, g.time, g.entity)
    g3.x[:, 0] += 3.0                                                 # change a real goal token
    assert not torch.allclose(head(R, g).value, head(R, g3).value, atol=1e-4)


def _all_decision_probs(mod, R, prev, steps):
    """Sum of probabilities over every possible recorded decision (continue, or terminate + select o)."""
    B = R.shape[0]
    tot = torch.zeros(B)
    for o in range(mod.n):
        lp, _ = mod.log_prob(R, None, prev, steps, torch.full((B,), o), torch.ones(B, dtype=torch.bool))
        tot += lp.exp()
    cont = (prev >= 0) & (steps < mod.max_steps)
    if bool(cont.any()):
        lp, _ = mod.log_prob(R[cont], None, prev[cont], steps[cont], prev[cont], torch.zeros(int(cont.sum()),
                                                                                             dtype=torch.bool))
        tot[cont] += lp.exp()
    return tot


def test_option_decision_probabilities_are_normalised_and_forced_termination():
    torch.manual_seed(2)
    mod = OptionModule(32, n_options=5, max_steps=4, n_heads=4)
    R = torch.randn(4, 3, 32)
    prev = torch.tensor([-1, 2, 3, 0])
    steps = torch.tensor([0, 1, 4, 2])          # row 2 reached max_steps -> forced reselection
    assert torch.allclose(_all_decision_probs(mod, R, prev, steps), torch.ones(4), atol=1e-5)
    out = mod.step(R, None, prev, steps)
    assert bool(out.new[0]) and bool(out.new[2])
    assert out.beta[0].item() == 1.0 and out.beta[2].item() == 1.0
    assert torch.all(out.steps[out.new] == 1)
    cont = ~out.new
    assert torch.all(out.option[cont] == prev[cont]) and torch.all(out.steps[cont] == steps[cont] + 1)
    lp, ent = mod.log_prob(R, None, prev, steps, out.option, out.new)
    assert torch.allclose(lp, out.logp, atol=1e-6)
    assert torch.all(ent >= 0)
    with pytest.raises(ValueError):
        mod.log_prob(R, None, torch.tensor([1, 2, 1, 0]), torch.zeros(4, dtype=torch.long), torch.tensor([1, 3, 1, 0]),
                     torch.zeros(4, dtype=torch.bool))


def test_option_tokens_and_learning_which_option_and_when_to_terminate():
    torch.manual_seed(3)
    mod = OptionModule(32, n_options=4, max_steps=8, n_heads=4)
    tok = mod.token(torch.tensor([2, -1]), torch.tensor([3, 0]))
    assert tok.x.shape == (2, 1, 32) and tok.kind[0, 0].item() == TokenKind.OPTION
    assert tok.mask.tolist() == [[True], [False]]
    R = torch.randn(64, 3, 32)
    opt = torch.optim.Adam(mod.parameters(), lr=1e-2)
    # REINFORCE: option 1 pays; continuing a running option 1 pays, terminating it costs
    for _ in range(150):
        prev = torch.randint(-1, 4, (64,))
        steps = torch.randint(0, 6, (64,))
        out = mod.step(R, None, prev, steps)
        reward = (out.option == 1).float() - 0.5 * (out.terminated & (prev == 1)).float()
        lp, _ = mod.log_prob(R, None, prev, steps, out.option, out.new)
        loss = -((reward - reward.mean()) * lp).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    p = mod.select_logits(R).softmax(-1).mean(0)
    assert p[1] > 0.8
    beta_1 = mod.beta(R, torch.full((64,), 1)).mean()
    beta_0 = mod.beta(R, torch.full((64,), 0)).mean()
    assert beta_1 < 0.1 < beta_0


def test_budget_token_encodes_scale_presence_and_fractions():
    enc = BudgetEncoder(32)
    b = BudgetState.make(3, days_left=torch.tensor([5.0, 5.0, 85.0]), funds_left=torch.tensor([1e6, 1e2, float("nan")]),
                         decisions_left=torch.tensor([3.0, 3.0, 400.0]), days_total=torch.tensor([90.0, 90.0, 90.0]),
                         decisions_total=torch.tensor([6.0, 6.0, float("nan")]))
    f = enc.features(b)
    assert torch.isfinite(f).all()
    assert f[0].ne(f[1]).any()                     # funds differ
    per_q = f.view(3, 3, -1)
    assert per_q[2, 1, 0].item() == 0.0            # funds unknown -> presence flag 0
    assert per_q[0, 2, 2].item() == pytest.approx(0.5)   # 3 of 6 decisions left
    tok = enc(b)
    assert tok.x.shape == (3, 1, 32) and int(tok.kind[0, 0]) == TokenKind.BUDGET and bool(tok.mask.all())
    unk = enc.features(BudgetState.make(2))
    assert torch.all(unk == 0)
