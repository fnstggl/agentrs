"""NUMERIC VALUE channel (cfg.value_channel) and the semantic stream S_t (ObservationTokenizer.semantic_context) for
DOM Mechanism C. Real Qwen3-0.6B encoder (session fixture)."""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from superlearner.cam_unified.config import make_config
from superlearner.cam_unified.interfaces import EntityRegistry, GoalSpec, Observation, ObsKind, ToolSchema
from superlearner.cam_unified.semantic.layers import ValueFeatures
from superlearner.cam_unified.semantic.tokenizer import ObservationTokenizer, obs_ref


def _num(v, entity=None, metric="balance", unit="cents"):
    return Observation(ObsKind.NUMERIC, value=float(v), unit=unit, source="bank.accounts", entity=entity,
                       entity_type="account" if entity else None, t=0.0, meta={"metric": metric})


def _cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1))


@pytest.fixture(scope="module")
def vtok(enc):
    torch.manual_seed(0)
    return ObservationTokenizer(make_config("dev", value_channel=True, sem_xattn_max_tokens=96), enc)


@pytest.fixture(scope="module")
def ltok(enc):
    torch.manual_seed(0)
    return ObservationTokenizer(make_config("dev", value_channel=False), enc)


def _tok_pair(tk, a, b, entity=None, **kw):
    regs = [EntityRegistry(64, seed=0), EntityRegistry(64, seed=0)]
    with torch.no_grad():
        ts = tk.observations([[_num(a, entity, **kw)], [_num(b, entity, **kw)]], regs, [0.0, 0.0])
    return ts.x[0, 0], ts.x[1, 0]


def test_value_features_are_sign_separated_and_finite():
    vf = ValueFeatures()
    f = vf(torch.tensor([1.0, -1.0, 0.0, float("nan"), float("inf"), -float("inf"), 1e30]))
    assert f.shape == (7, vf.dim) and bool(torch.isfinite(f).all())
    half = (vf.dim - 5) // 2
    assert torch.equal(f[0, :half], f[1, :half]) and torch.equal(f[0, half:2 * half], -f[1, half:2 * half])
    assert float(f[3, -3]) == 1.0 and float(f[4, -2]) == 1.0 and float(f[5, -1]) == 1.0


def test_numeric_tokens_carry_the_value_at_init(vtok, ltok):
    """cos(+1, -1) of two NUMERIC tokens of the same metric / unit at init: < 0.5 with the VALUE channel (plan W4
    acceptance), ~0.99 on the legacy path (the value term was swamped by the unit / descriptor embeddings)."""
    c = _cos(*_tok_pair(vtok, 1, -1))
    c_ent = _cos(*_tok_pair(vtok, 1, -1, entity="acct_1"))
    c_leg = _cos(*_tok_pair(ltok, 1, -1))
    print(f"\ncos(+1,-1): value channel {c:.3f} (same entity {c_ent:.3f}), legacy {c_leg:.3f}")
    assert c < 0.5 and c_leg > 0.9
    assert c_ent < c_leg
    # magnitudes are resolved: 100 vs 200 differ more than 100 vs 101, far apart scales more than both
    near, far, scale = (_cos(*_tok_pair(vtok, a, b)) for a, b in ((100, 101), (100, 200), (100, 1e6)))
    assert near > far > scale
    # the metric still matters: the same value of another metric is a different token
    other = _cos(_tok_pair(vtok, 100, 100)[0], _tok_pair(vtok, 100, 100, metric="latency", unit="ms")[1])
    assert other < 0.99


def test_value_channel_gradients_and_goal_metrics(vtok):
    vtok.zero_grad(set_to_none=True)
    regs = [EntityRegistry(64, seed=0)]
    ts = vtok.observations([[_num(3.0), _num(float("nan"))]], regs, [0.0])
    assert bool(torch.isfinite(ts.x).all()) and vtok.last_stats.get("nonfinite_values") == 1
    ts.x.float().pow(2).sum().backward()
    for name in ("val_in", "num_proj"):
        assert float(getattr(vtok, name).weight.grad.abs().sum()) > 0, name
    assert vtok.numctx_gain.grad is not None
    assert not hasattr(vtok, "num_mlp")                     # the legacy branch is not built (no dead parameters)
    with torch.no_grad():                                   # goal metric targets use the same channel
        g = vtok.goal([GoalSpec("g1", "earn", metrics={"balance": 1000.0}), GoalSpec("g2", "earn", metrics={"balance": -1000.0})])
    assert _cos(g.x[0, -1], g.x[1, -1]) < 0.9


# ------------------------------------------------------------------------------------------------ S_t producer
def _obs(t, text, kind=ObsKind.TEXT, seq=None):
    return Observation(kind, content=text, t=t, seq=seq, source="forum")


def test_semantic_context_is_newest_first_budgeted_and_referenced(vtok, enc):
    rows = [[_obs(0.0, "old post about bread " * 40), _obs(1.0, "new reply: the oven is broken", seq=7),
             _num(5.0)],
            [],
            [_obs(0.5, "a message " * 200)]]
    goals = [GoalSpec("g0", "Earn 10000 cents within 8 days by selling bread."), None, None]
    tools = [[ToolSchema("bank.transfer", "Transfer money.", {"type": "object"})], [], []]
    sc = vtok.semantic_context(rows, tools, goals, max_tokens=64)
    B, Ns, H = sc.h.shape
    assert B == 3 and H == enc.sem_dim and Ns <= 64 and sc.mask.shape == (3, Ns)
    assert sc.h.dtype == torch.float32 and bool(torch.isfinite(sc.h).all())
    n0 = int(sc.mask[0].sum())
    assert n0 == 64 and sc.stats["tokens"][0] == 64
    r0 = sc.refs[0][:n0]
    goal_n = sum(1 for r in r0 if r.startswith("goal:"))
    assert 0 < goal_n <= 16 and r0[0].startswith("goal:")             # goal first, at most goal_share of the budget
    newest = obs_ref(1, rows[0][1])
    first_obs = next(r for r in r0 if r.startswith("obs:"))
    assert first_obs == newest                                         # observations newest first
    assert sc.stats["kept_obs"][0] == 2 and sc.stats["cut_tokens"][0] > 0   # the old post is cut to the budget
    assert all(not r.startswith("obs:2") for r in r0)                  # pure NUMERIC observations: VALUE channel
    assert int(sc.mask[1].sum()) == 0                                  # an empty row is one masked padding token
    assert int(sc.mask[2].sum()) == 64 and sc.stats["kept_obs"][2] == 1
    # the states are the encoder's cached token states, standardised
    text = __import__("superlearner.cam_unified.semantic.tokenizer", fromlist=["x"]).render_observation(
        rows[0][1], vtok.max_obs_chars)
    st = enc.standardize_states(enc.token_states([text])[0])
    j = r0.index(newest)
    k = st.shape[0]
    assert torch.allclose(sc.h[0, j: j + k], st, atol=0, rtol=0)
    hits = enc.stats["hits"]
    sc2 = vtok.semantic_context(rows, tools, goals, max_tokens=64)
    assert enc.stats["hits"] > hits and torch.equal(sc.h, sc2.h)      # cached and deterministic


def test_gated_core_reads_the_semantic_context(enc):
    from superlearner.cam_unified.core.semantic_xattn import GatedUnifiedCore
    from superlearner.cam_unified.testing import rand_tokens, tiny_config
    cfg = tiny_config(sem_xattn_every=1, sem_xattn_dim=enc.sem_dim, sem_xattn_max_tokens=48)
    tk = ObservationTokenizer(cfg, enc)
    core = GatedUnifiedCore(cfg).to(enc.device)
    rows = [[_obs(0.0, "The customer asked for a refund of order 17.")], [_obs(0.0, "Deploy the service now.")]]
    sc = tk.semantic_context(rows, max_tokens=48)
    ctx = rand_tokens(2, 6, cfg.d_model, device=enc.device)
    Z0 = core.init_state(2, device=enc.device)
    Za, _ = core(Z0, ctx)                       # W3's final API: S_t is a forward argument (no module state)
    Zb, _ = core(Z0, ctx, sem=sc)
    assert not torch.equal(Za, Zb)
    Zb.float().pow(2).mean().backward()
    assert float(sum(p.grad.abs().sum() for p in core.sem_proj.parameters() if p.grad is not None)) > 0
