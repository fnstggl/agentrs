"""G-nn probe: causality, padding / batch / candidate invariance, time conditioning, persistent state (CPU).

    .venv/bin/python cam_live/docs/verification/repro/nn/probe_invariants.py [--scale debug|50m]

Uses the tiny random backbone for the text side (the properties tested here live in the experiential network, which
is the real code path). Zero-initialised tensors (attention time bias) are randomised first, otherwise the
time-conditioned attention path is dead at initialisation and a causality test cannot fail there.
"""
import io
import sys

import _common  # noqa: F401
from _common import check, finish, info

import torch
import torch.nn.functional as Fn

from cam_live.config import CamLiveConfig, CoreConfig
from cam_live.contracts import ActionSpec, ExperienceItem, ItemKind, OutcomeKind, OutcomeQuery, Quantity
from cam_live.nn.api import CoreState, TokenBatch
from cam_live.nn.model import build_model

scale = sys.argv[sys.argv.index("--scale") + 1] if "--scale" in sys.argv else "debug"
torch.manual_seed(0)
cfg = CamLiveConfig.debug()
if scale != "debug":
    cfg.core = CoreConfig().apply_scale(scale)
    cfg.core.max_context = 48
model = build_model(cfg)
model.eval()
info("scale", scale)
info("experiential_params", model.count_parameters()["experiential"])

# make every zero-initialised path live (time bias in attention), as it will be after training
g = torch.Generator().manual_seed(123)
with torch.no_grad():
    for blk in model.exp.core.blocks:
        blk.attn.time_bias.weight.copy_(torch.randn(blk.attn.time_bias.weight.shape, generator=g) * 0.5)
    model.exp.core.fresh_emb.copy_(torch.randn(model.exp.core.fresh_emb.shape, generator=g) * 0.5)

KINDS = [ItemKind.OBSERVATION, ItemKind.ACTION, ItemKind.OUTCOME, ItemKind.EVENT, ItemKind.TOOL]


def items(n, t0=1000.0, tag="", gaps=None):
    out, t = [], t0
    for i in range(n):
        qs = [Quantity(f"metric_{i % 3}", float(i * 7 - 3))] if i % 2 == 0 else []
        out.append(ExperienceItem(kind=KINDS[i % 5], text=f"{tag} item number {i} alpha beta", ts=t, quantities=qs,
                                  ref=f"r{i}"))
        t += (gaps[i] if gaps is not None else 60.0)
    return out


@torch.no_grad()
def run(ctxs, state=None):
    tb = model.tokenize(ctxs)
    return tb, model.core_forward(tb, state)


# ---------------------------------------------------------------------------------------------------- causality
base = items(10)
tb0, o0 = run([base])
worst_h, worst_x = 0.0, 0.0
for k in range(1, 10):
    ch = list(base)
    for j in range(k, 10):
        ch[j] = ExperienceItem(kind=ItemKind.FAILURE, text=f"totally different text {j} zzz", ts=base[j].ts + 3.3e6 * (j - k + 1),
                               quantities=[Quantity("other", -9e6 * (j + 1))], trust="operator")
    tb1, o1 = run([ch])
    worst_h = max(worst_h, float((o0.h[:, :k] - o1.h[:, :k]).abs().max()))
    worst_x = max(worst_x, float((tb0.x[:, :k] - tb1.x[:, :k]).abs().max()))
    assert float((o0.h[:, k:] - o1.h[:, k:]).abs().max()) > 1e-3
check("causality: h[:, :k] bit-identical when items >= k change (text, values, timestamps, kind, trust)", worst_h == 0.0,
      f"max|dh|={worst_h:.3e}, max|dx|={worst_x:.3e}")

# later timestamps only (the attention time bias is live)
worst = 0.0
for k in range(1, 10):
    ch = [ExperienceItem(kind=it.kind, text=it.text, ts=it.ts + (5e5 * (j - k + 1) if j >= k else 0.0),
                         quantities=it.quantities, ref=it.ref) for j, it in enumerate(base)]
    _, o1 = run([ch])
    worst = max(worst, float((o0.h[:, :k] - o1.h[:, :k]).abs().max()))
    assert float((o0.h[:, k:] - o1.h[:, k:]).abs().max()) > 1e-4, "timestamps of later items have no effect at all"
check("causality: h[:, :k] bit-identical when only timestamps of items >= k change", worst == 0.0, f"max|dh|={worst:.3e}")

# prefix consistency (different T)
worst = 0.0
for k in range(1, 10):
    _, ok = run([base[:k]])
    worst = max(worst, float((ok.h - o0.h[:, :k]).abs().max()))
check("causality: h of a truncated context equals the prefix of the full one (float32 tolerance 2e-5)", worst < 2e-5,
      f"max|dh|={worst:.3e}")

# gradient form: d h[:, i] / d x[:, j] == 0 for j > i
tb = model.tokenize([base])
x = tb.x.detach().clone().requires_grad_(True)
tb.x = x
h = model.core_forward(tb).h
leak = 0.0
for i in range(9):
    (gx,) = torch.autograd.grad(h[0, i].pow(2).sum(), x, retain_graph=True)
    leak = max(leak, float(gx[0, i + 1:].abs().max()))
check("causality: d h[i] / d x[j>i] is exactly zero", leak == 0.0, f"max future gradient={leak:.3e}")

# ---------------------------------------------------------------------------------------------- padding / batch
a, b_, c = items(3, tag="a"), items(9, tag="b", t0=5e4), items(6, tag="c", t0=9e5)
_, alone = run([a])
_, both = run([a, b_, c])
d_b = float((alone.b[0] - both.b[0]).abs().max())
d_h = float((alone.h[0] - both.h[0, :3]).abs().max())
d_s = float((alone.state.slots[0] - both.state.slots[0]).abs().max())
check("padding invariance of b / h / slots (tol 2e-5)", max(d_b, d_h, d_s) < 2e-5, f"db={d_b:.2e} dh={d_h:.2e} dslots={d_s:.2e}")
check("padded positions of h are exactly zero", float(both.h[0, 3:].abs().sum()) == 0.0)
_, perm = run([c, a, b_])
d = max(float((perm.b[1] - both.b[0]).abs().max()), float((perm.b[2] - both.b[1]).abs().max()),
        float((perm.b[0] - both.b[2]).abs().max()))
check("batch-composition / batch-order invariance of b (tol 2e-5)", d < 2e-5, f"max|db|={d:.2e}")
_, other = run([a, items(9, tag="completely other", t0=1.0), c])
d = max(float((other.b[0] - both.b[0]).abs().max()), float((other.state.slots[0] - both.state.slots[0]).abs().max()))
check("row 0 does not depend on the content of row 1 (tol 2e-5)", d < 2e-5, f"max diff={d:.2e}")

# ------------------------------------------------------------------------------------------------ candidate axis
acts = [ActionSpec("a1", "browser.click", arguments={"ref": 3}), ActionSpec("a2", "internal.wait", timeout_s=600.0),
        ActionSpec("a3", "sandbox.python", modality="code", level="procedure", arguments={"code": "print(1)"}),
        ActionSpec("a4", "api.call", modality="api", arguments={"x": 9})]
with torch.no_grad():
    e_g = model.encode_goals(["reach the stated target"])
    bb = both.b[:1]
    e_all, m_all = model.encode_actions([acts])
    z_all = model.decision_latent(bb, e_all, e_g)
    worst_e, worst_z = 0.0, 0.0
    for n, act in enumerate(acts):
        e1, _ = model.encode_actions([[act]])
        z1 = model.decision_latent(bb, e1, e_g)
        worst_e = max(worst_e, float((e1[0, 0] - e_all[0, n]).abs().max()))
        worst_z = max(worst_z, float((z1[0, 0] - z_all[0, n]).abs().max()))
    check("candidate independence: e_a / z of candidate n do not depend on the other candidates (tol 2e-5)",
          max(worst_e, worst_z) < 2e-5, f"de={worst_e:.2e} dz={worst_z:.2e}")
    p = [2, 0, 3, 1]
    e_p, _ = model.encode_actions([[acts[i] for i in p]])
    z_p = model.decision_latent(bb, e_p, e_g)
    d = float((z_p[0] - z_all[0, p]).abs().max())
    check("candidate axis is permutation-equivariant (tol 2e-5)", d < 2e-5, f"max|dz|={d:.2e}")
    # padded candidate rows
    e_2, m_2 = model.encode_actions([acts, acts[:1]])
    z_2 = model.decision_latent(torch.cat([bb, bb]), e_2, torch.cat([e_g, e_g]))
    d = float((z_2[1, 0] - z_all[0, 0]).abs().max())
    check("candidate padding does not change a real candidate (tol 2e-5)", d < 2e-5 and m_2.tolist()[1] == [True, False, False, False],
          f"max|dz|={d:.2e}")
    info("z at padded candidate slots is non-zero (caller must mask with a_mask)", float(z_2[1, 1:].abs().max()))
    qs = [OutcomeQuery("q1", "members.count", kind=OutcomeKind.COUNT, horizon_s=86400.0),
          OutcomeQuery("q2", "latency", horizon_s=30.0, description="seconds until done"),
          OutcomeQuery("q3", "done", kind=OutcomeKind.BINARY, horizon_s=3600.0)]
    e_q, q_mask, _, _ = model.encode_queries([qs])
    phi = model.features(z_all, e_q)
    worst = 0.0
    for j, q in enumerate(qs):
        eq1 = model.encode_queries([[q]])[0]
        worst = max(worst, float((model.features(z_all, eq1)[:, :, 0] - phi[:, :, j]).abs().max()))
    check("query independence of phi (tol 2e-5)", worst < 2e-5, f"max|dphi|={worst:.2e}")
    check("phi unit norm, constant last coordinate", bool(torch.allclose(phi.norm(dim=-1), torch.ones_like(phi[..., 0]), atol=1e-4))
          and bool(torch.allclose(phi[..., -1], torch.full_like(phi[..., -1], model.n_features ** -0.5))))

# ------------------------------------------------------------------------------------------------------------ time
reg = items(8, gaps=[60.0] * 8)
irr = items(8, gaps=[1.0, 2.6e6, 5.0, 3600.0, 0.5, 86400.0, 30.0, 60.0])
_, o_reg = run([reg])
_, o_irr = run([irr])
rel = float((o_reg.b - o_irr.b).norm() / o_reg.b.norm())
check("time: irregular gaps change the belief (time bias randomised)", rel > 1e-2, f"relative change of b = {rel:.4f}")
sh = [ExperienceItem(kind=it.kind, text=it.text, ts=it.ts + 1.7e9, quantities=it.quantities, ref=it.ref) for it in irr]
_, o_sh = run([sh])
d = float((o_sh.b - o_irr.b).abs().max())
dh = float((o_sh.h - o_irr.h).abs().max())
check("time: absolute shift of all timestamps by 1.7e9 s leaves h / b unchanged (tol 2e-5)", max(d, dh) < 2e-5,
      f"db={d:.2e} dh={dh:.2e}")
# fractional timestamps at epoch scale (float64 resolution at 1.7e9 is 2.4e-7 s)
fr = [ExperienceItem(kind=it.kind, text=it.text, ts=it.ts * 1.000001 + 0.123, quantities=it.quantities) for it in irr]
fr_s = [ExperienceItem(kind=it.kind, text=it.text, ts=it.ts + 1.7e9, quantities=it.quantities) for it in fr]
d = float((run([fr])[1].b - run([fr_s])[1].b).abs().max())
check("time: shift invariance with fractional timestamps (tol 1e-4)", d < 1e-4, f"db={d:.2e}")
# the same test with the model as initialised (time bias zero): how much does time matter before any training?
m0 = build_model(cfg)
with torch.no_grad():
    b_reg = m0.core_forward(m0.tokenize([reg])).b
    b_irr = m0.core_forward(m0.tokenize([irr])).b
info("time: relative change of b between regular and irregular gaps AT INITIALISATION (only the token gap feature is live)",
     round(float((b_reg - b_irr).norm() / b_reg.norm()), 5))
# non-monotonic timestamps: documented behaviour? (elapsed clamps at 0)
bad = list(reg)
bad[4] = ExperienceItem(kind=bad[4].kind, text=bad[4].text, ts=bad[4].ts - 1e5, quantities=bad[4].quantities)
_, o_bad = run([bad])
check("time: out-of-order timestamps give finite outputs", bool(torch.isfinite(o_bad.h).all() and torch.isfinite(o_bad.b).all()))

# ------------------------------------------------------------------------------------------------ persistent state
ctx = items(10, tag="s")
_, one = run([ctx])
_, first = run([ctx[:6]])
_, second = run([ctx[6:]], first.state)
rel = float((one.b - second.b).norm() / one.b.norm())
info("state: relative difference of b, streaming (6 + 4 items, state carried) vs one-shot (10 items)", round(rel, 4))
info("state: streaming is NOT equivalent to one-shot processing (slots are a lossy summary; no claim of equality in DESIGN.md)",
     rel > 1e-3)
# information is carried: same second call after two different first calls
_, first_b = run([items(6, tag="another history", t0=1000.0)])
_, second_b = run([ctx[6:]], first_b.state)
rel_c = float((second.b - second_b.b).norm() / second.b.norm())
_, second_fresh = run([ctx[6:]])
rel_f = float((second.b - second_fresh.b).norm() / second.b.norm())
check("state: the second call's belief depends on the first call's content", rel_c > 1e-3, f"relative change={rel_c:.4f}")
info("state: relative change of b, carried state vs fresh state", round(rel_f, 4))
# elapsed time since the previous call
st_late = CoreState(first.state.slots.clone(), first.state.last_time - 2.6e6, first.state.step)
_, second_late = run([ctx[6:]], st_late)
rel_t = float((second.b - second_late.b).norm() / second.b.norm())
check("state: elapsed time since the previous call conditions the read", rel_t > 1e-4, f"relative change={rel_t:.5f}")
# checkpoint / restore of the state is exact
buf = io.BytesIO()
torch.save({"slots": first.state.slots, "last_time": first.state.last_time, "step": first.state.step}, buf)
buf.seek(0)
sd = torch.load(buf)
_, second_r = run([ctx[6:]], CoreState(sd["slots"], sd["last_time"], sd["step"]))
check("state: save -> load -> continue is bit-exact", bool(torch.equal(second_r.b, second.b) and torch.equal(second_r.h, second.h)
                                                           and torch.equal(second_r.state.slots, second.state.slots)))
# the input state is not mutated in place
s_before = first.state.slots.clone()
run([ctx[6:]], first.state)
check("state: core_forward does not mutate the state passed in", bool(torch.equal(s_before, first.state.slots)))
# many calls: bounded
state, rms = None, []
for t in range(300):
    _, o = run([[ctx[t % 10]]], state)
    state = o.state
    rms.append(float(state.slots.pow(2).mean().sqrt()))
check("state: 300 single-token calls keep the slots finite and bounded", bool(torch.isfinite(state.slots).all()) and rms[-1] < 5.0,
      f"slot rms after 1/100/300 calls: {rms[0]:.3f}/{rms[99]:.3f}/{rms[-1]:.3f}")
# h in a later call depends on the state (not only b)
dh = float((second.h - second_fresh.h).norm() / second.h.norm())
info("state: relative change of h (per-token latents) with carried vs fresh state", round(dh, 5))

# double backward
tb = model.tokenize([items(4)])
tb.x = tb.x.detach().requires_grad_(True)
out = model.core_forward(tb)
z = model.decision_latent(out.b, e_all, e_g)
phi = model.features(z, e_q)
(g1,) = torch.autograd.grad(phi.pow(3).sum(), tb.x, create_graph=True)
(g2,) = torch.autograd.grad(g1.pow(2).sum(), tb.x)
check("double backward through core -> trunk -> features is finite and non-zero", bool(torch.isfinite(g2).all()) and float(g2.abs().sum()) > 0)

sys.exit(finish())
