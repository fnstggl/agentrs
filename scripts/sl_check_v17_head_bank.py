"""v17 head bank checks: (1) no bank = v16 (same names, same initial values); (2) bank adds only bank.* and leaves every
other initial value unchanged; (3) gradients: bank rows train only their bank heads, other rows only the shared heads."""
import sys, importlib, torch
def load(src):
    for k in [k for k in sys.modules if k.startswith("agentrs")]:
        del sys.modules[k]
    sys.path.insert(0, src)
    import agentrs.sl.model as M
    sys.path.pop(0)
    return M
M16 = load("/workspace/agentrs-sl-exp-v16/src"); torch.manual_seed(7); a = M16.SLCore(M16.SLConfig(fam_gate_n0=4096)).state_dict()
M17 = load("/workspace/agentrs-sl/src")
torch.manual_seed(7); b = M17.SLCore(M17.SLConfig(fam_gate_n0=4096)).state_dict()
assert list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a), "no-bank v17 differs from v16"
torch.manual_seed(7); m = M17.SLCore(M17.SLConfig(fam_gate_n0=4096, head_bank=["cam/bizdev"])); c = m.state_dict()
extra = [k for k in c if k not in a]
assert all(k.startswith("bank.0.") for k in extra) and all(torch.equal(a[k], c[k]) for k in a), "bank changed other tensors"
print("names/init ok; bank tensors:", len(extra))
hix = m.head_index(["cam/negotiation", "cam/bizdev", "cam/bizdev", "cam/product"])
print("hix", hix.tolist())
ctx = torch.randn(4, m.c.dh + m.c.dv + m.c.d); z = torch.tensor([0, 1, 2, 3])
for rows, want_shared, want_bank in [([0], True, False), ([1], False, True)]:
    m.zero_grad()
    hi, tl = m.skill_dists(ctx, z, hix); q = m.policy_query(ctx, z, hix); q2 = m.slot_step(torch.randn(4, m.c.da), q, hix)
    (hi[rows].sum() + tl[rows].sum() + q2[rows].sum()).backward()
    gs = sum(float(p.grad.abs().sum()) for n, p in m.named_parameters() if n.split(".")[0] in ("hi", "term", "query", "slot") and p.grad is not None)
    gb = sum(float(p.grad.abs().sum()) for n, p in m.named_parameters() if n.startswith("bank.") and p.grad is not None)
    assert (gs > 0) == want_shared and (gb > 0) == want_bank, (rows, gs, gb)
# no-bank call path returns the shared output exactly
m2 = M17.SLCore(M17.SLConfig()); assert m2.head_index(["x"]) is None
o1 = m2.policy_query(ctx, z); o2 = m2.query(torch.cat([ctx, m2.skill[z]], -1)); assert torch.equal(o1, o2)
print("gradient routing ok")
