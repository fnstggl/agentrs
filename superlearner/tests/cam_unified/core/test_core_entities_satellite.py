"""EntityEncoder at the satellite width cfg.d_satellite (W3): projections in and out, d_model output, fewer
parameters, type tokens at the semantic width, and unchanged behaviour when d_satellite = 0."""
import pytest
import torch

from superlearner.cam_unified.config import make_config
from superlearner.cam_unified.core.entities import EntityEncoder, entity_type_tokens
from superlearner.cam_unified.interfaces import EntityRegistry, TokenKind, TokenSet

torch.set_num_threads(8)


def _obs(B, N, D, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, N, D, generator=g)
    ts = TokenSet.from_dense(x, int(TokenKind.OBS_SEM))
    ts.entity = torch.randint(0, 6, (B, N), generator=g)
    ts.entity2 = torch.where(torch.rand(B, N, generator=g) < 0.3, torch.randint(0, 6, (B, N), generator=g),
                             torch.full((B, N), -1))
    ts.time = -torch.rand(B, N, generator=g)
    return ts


def test_zero_satellite_width_is_the_pre_dom_encoder():
    cfg = make_config("dev")
    torch.manual_seed(0)
    a = EntityEncoder(cfg)
    torch.manual_seed(0)
    b = EntityEncoder(make_config("dev", d_satellite=cfg.d_model))      # W = d_model: no projections either
    assert a.in_proj is None and a.out_proj is None and a.width == cfg.d_model
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def test_satellite_width_projects_in_and_out_and_trains():
    cfg = make_config("dev", d_satellite=192)
    torch.manual_seed(0)
    enc = EntityEncoder(cfg)
    full = EntityEncoder(make_config("dev"))
    assert enc.width == 192 and enc.rounds[0].attn.d == 192
    assert enc.count_parameters() < 0.4 * full.count_parameters()
    obs = _obs(2, 12, cfg.d_model)
    out = enc(obs)
    assert out.x.shape[-1] == cfg.d_model and torch.isfinite(out.x).all()
    assert set(out.kind[out.mask].tolist()) <= {int(TokenKind.ENTITY), int(TokenKind.RELATION)}
    (out.x * torch.randn_like(out.x)).sum().backward()
    for n, p in enc.named_parameters():
        if n.startswith(("type_proj", "pool_query")):
            continue
        assert p.grad is not None, n


def test_type_tokens_at_the_semantic_width_condition_entities():
    cfg = make_config("dev", d_satellite=192, sem_xattn_dim=40)
    torch.manual_seed(1)
    enc = EntityEncoder(cfg).eval()
    assert enc.type_proj.in_features == 40
    reg = EntityRegistry(cfg.max_entities, seed=0)
    for e, t in (("a", "customer"), ("b", "supplier"), ("c", "customer")):
        reg.bind(e, etype=t)
    obs = _obs(1, 10, cfg.d_model, seed=3)
    obs.entity = torch.tensor([[reg.slot["a"], reg.slot["b"], reg.slot["c"]] * 3 + [reg.slot["a"]]])
    obs.entity2 = torch.full((1, 10), -1)

    class Sem:
        version = "v1"
        calls = 0

        def embed(self, texts):
            Sem.calls += 1
            return torch.stack([torch.randn(40, generator=torch.Generator().manual_seed(hash(t) % 997)) for t in texts])
    sem = Sem()
    tt = enc.make_type_tokens([reg], sem)
    assert tt.x.shape == (1, 3, 40) and Sem.calls == 1
    enc.make_type_tokens([reg], sem)
    assert Sem.calls == 1                                                # cached per semantic version
    with torch.no_grad():
        with_t = enc(obs, tt).x
        without = enc(obs).x
    assert float((with_t - without).abs().max()) > 1e-4
    bad = entity_type_tokens([reg], lambda ts: torch.randn(len(ts), 7))
    with pytest.raises(ValueError, match="type_dim"):
        enc(obs, bad)
