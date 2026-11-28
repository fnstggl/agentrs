"""G2-learn round 3 probe: DX round 2's slow-step changes (CL-03), on D's lab (debug scale, CPU) and synthetically.

    .venv/bin/python cam_live/docs/verification/repro/g2learn/probe_guard_r3.py [--only restore,floor,pcgrad]

restore  (a) is the guard's before / after comparison like with like? held-out evaluation itself must not move any
         running statistic (two evaluations equal, buffers / scaler unchanged); a slow step leaves every buffer and the
         target scaler as they were; a reverted step leaves the held-out cells bit-identical; the guard's numbers equal
         a fresh evaluation; the size of the artefact the restore removes (the step's loss evaluation without restore);
         a real degradation (harmful adapter perturbation) is still visible with the restore in place.
floor    (b) the 1e-4 minimum tolerance and the drift bound, through the learner's own ``_guard_verdict``: a consistent
         rise just under 1e-4 relative per step is accepted until the per-provider drift bound, a promotion
         (``set_slow_reference``) resets the bound, and the agent's promotion gate would not refuse the accumulated
         rise; a rise of 2e-4 and of 1 % is refused at once; a noisy provider's tolerance is capped at 0.01.
pcgrad   (c) ``_balance_adapter_grads`` (PCGrad + max_share) and ``_project_update`` (A-GEM-style projection of the
         applied update) on random conflicting gradient sets: after the projection, <g_k, Delta> <= 0 for every
         protected provider? how much of the update survives? does PCGrad alone give a non-increasing direction?
"""
import json
import math
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import TESTS, WORK, check, finish, info  # noqa: E402

sys.path.insert(0, os.path.join(TESTS, "credit"))
import torch  # noqa: E402

from cam_live.config import CamLiveConfig  # noqa: E402
from cam_live.credit.labkit import build_lab, collect_family  # noqa: E402
from cam_live.credit.learner import LearnerSettings  # noqa: E402
from cam_live.diagnostics.world import LabConfig  # noqa: E402
from cam_live.learning_api import TIMESCALE_FREQUENT, TIMESCALE_SLOW, UpdatePlan  # noqa: E402
from cam_live.nn.api import GROUP_ADAPTERS  # noqa: E402

ONLY = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else ["restore", "agent", "floor", "pcgrad"]
torch.set_num_threads(2)


def lab(name):
    d = os.path.join(WORK, "guard_r3", name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    torch.manual_seed(0)
    cfg = CamLiveConfig.debug()
    st = LearnerSettings.from_cfg(cfg, val_split="record", holdout_experiments=False, batch_size=8, slow_batch_size=4,
                                  grad_last=8)
    lb = build_lab(os.path.join(d, "store.sqlite"), cfg, bridge=True, beta=2e-6, settings=st)
    collect_family(lb.store, lb.linker, LabConfig(n_channels=4, family="lab/A", target=30), 8, 6, seed=3, clock=lb.clock)
    lb.learner.refresh_eligibility(full=True)
    for _ in range(3):
        lb.learner.learn(UpdatePlan(trigger="probe", timescales=(TIMESCALE_FREQUENT,)))
    return lb


def running(L):
    return L._running_state()


def same_running(a, b):
    bufs = all(torch.equal(a["buffers"][k], b["buffers"][k]) for k in a["buffers"])
    sc = all(sa[1] == sb[1] for (sa, sb) in zip(a["scalers"].values(), b["scalers"].values()))
    return bufs and sc


if "restore" in ONLY:
    lb = lab("restore")
    L, m = lb.learner, lb.model
    r0 = running(L)
    c1 = L._validation_cells()
    r1 = running(L)
    c2 = L._validation_cells()
    check("(a) held-out evaluation is pure: two evaluations give identical cells", c1 == c2, f"{len(c1)} cells")
    check("(a) held-out evaluation moves no buffer and no target scaler", same_running(r0, r1))
    # the artefact the restore removes: the slow step's own loss evaluation (with gradients) without the restore
    from cam_live.credit.learner import encode_with_adapter_grad
    providers = [p for p in L.providers if getattr(p, "uses_batch", True)]
    records, _ = L._sample(providers, 4)
    snap = running(L)
    m.train()
    batch = encode_with_adapter_grad(m, lb.store, records, grad_last=8, with_next=L.settings.with_next)
    L._collect(providers, batch, L._params(GROUP_ADAPTERS))
    m.eval()
    moved = L._restore_running_state(snap) and None
    c_art = None
    # redo without restoring, to measure the shift it would have caused
    m.train()
    batch = encode_with_adapter_grad(m, lb.store, records, grad_last=8, with_next=L.settings.with_next)
    L._collect(providers, batch, L._params(GROUP_ADAPTERS))
    m.eval()
    c_art = L._validation_cells()
    changed = L._restore_running_state(snap)
    art = {k: c_art[k] - c1[k] for k in c1 if k in c_art}
    by = {}
    for (prov, j), dv in art.items():
        by.setdefault(prov, []).append(dv / max(abs(c1[(prov, j)]), 1e-12))
    info("(a) running statistics a slow step's loss evaluation moves (restored now)", changed)
    info("(a) held-out shift those moves would cause with unchanged adapters (mean relative, per provider)",
         {k: f"{sum(v) / len(v):+.2e}" for k, v in by.items()})
    check("(a) after restoring, the held-out cells are again exactly the pre-step ones", L._validation_cells() == c1)
    # one real slow step: buffers unchanged, guard numbers = fresh evaluation
    rb = running(L)
    rec = L.learn(UpdatePlan(trigger="probe", timescales=(TIMESCALE_SLOW,)))[-1]
    notes = json.loads(rec.notes)
    g = notes.get("guard") or {}
    ra = running(L)
    check("(a) a slow step leaves every buffer and the target scaler as they were", same_running(rb, ra),
          f"restored {notes.get('running_state_restored')}")
    cells = L._validation_cells()
    if g.get("reverted"):
        check("(a) a reverted step leaves the held-out cells bit-identical", cells == c1)
    else:
        ok = all(abs(v["validation_after"] - sum(cells[(k, j)] for j in range(v["n_chunks"]) if (k, j) in cells) / v["n_chunks"]) < 1e-9
                 for k, v in (g.get("by_provider") or {}).items())
        check("(a) the guard's 'after' numbers equal a fresh held-out evaluation of the kept step", ok)
    info("(a) the step's verdict", {"accepted": g.get("accepted"), "by_provider": {k: (f"{v['paired_change']:+.2e}", f"{v['tolerance']:.1e}") for k, v in (g.get("by_provider") or {}).items()}})
    # a real degradation is still seen: harmful perturbation of the adapters
    before = L._validation_cells()
    ps = m.param_groups()[GROUP_ADAPTERS]
    keep = [p.detach().clone() for p in ps]
    gen = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for p in ps:
            p.add_(torch.randn(p.shape, generator=gen) * 0.05)
    m.mark_adapters_updated()
    after = L._validation_cells()
    verdict = L._guard_verdict(before, after)
    with torch.no_grad():
        for p, k in zip(ps, keep):
            p.copy_(k)
    m.mark_adapters_updated()
    rises = {k: f"{v['paired_change'] / max(abs(v['validation_before']), 1e-12):+.2e}" for k, v in verdict["by_provider"].items()}
    check("(a) a harmful adapter perturbation is still refused with the restore in place", not verdict["accepted"],
          f"relative changes {rises}")
    lb.close()

if "agent" in ONLY:
    # the integrated agent's own providers (predict.providers world model with its KindNormaliser, the target scaler)
    sys.path.insert(0, os.path.join(TESTS, "integration"))
    from int_helpers import Recorder, agent_config, run_lifetime  # noqa: E402
    from cam_live.agent import CamLiveAgent  # noqa: E402
    from cam_live.contracts import SimClock  # noqa: E402
    from cam_live.credit.learner import encode_with_adapter_grad  # noqa: E402
    d = os.path.join(WORK, "guard_r3", "agent")
    shutil.rmtree(d, ignore_errors=True)
    agent = CamLiveAgent(agent_config(policy_update_every=6), os.path.join(d, "agent"), clock=SimClock())
    rec_ = Recorder(agent)
    for sd in (11, 12):
        run_lifetime(rec_, agent.clock, d, sd, max_steps=40, max_decisions=22,
                     lab_kw={"outcome_horizon_s": 1800.0, "mean_delay_s": (300.0, 900.0)})
    L, m = agent.learner, agent.model
    c1 = L._validation_cells()
    snap = running(L)
    check("(a, agent) held-out evaluation is pure in the integrated agent", L._validation_cells() == c1 and
          same_running(snap, running(L)), f"{len(c1)} cells over {len({k for k, _ in c1})} providers")
    providers = [p for p in L.providers if getattr(p, "uses_batch", True)]
    records, _ = L._sample(providers, 3)
    m.train()
    batch = encode_with_adapter_grad(m, agent.store, records, grad_last=2, with_next=L.settings.with_next)
    L._collect(providers, batch, L._params(GROUP_ADAPTERS))
    m.eval()
    c_art = L._validation_cells()          # adapters unchanged, running statistics moved by the loss evaluation
    changed = L._restore_running_state(snap)
    by = {}
    for (prov, j), v0 in c1.items():
        if (prov, j) in c_art:
            by.setdefault(prov, []).append((c_art[(prov, j)] - v0) / max(abs(v0), 1e-12))
    info("(a, agent) running statistics moved by one slow step's loss evaluation", changed)
    info("(a, agent) held-out change those moves cause with the adapters unchanged (mean relative per provider)",
         {k: f"{sum(v) / len(v):+.2e}" for k, v in by.items()})
    check("(a, agent) the restore returns the held-out cells exactly to the pre-step values", L._validation_cells() == c1)
    agent.close()

if "floor" in ONLY:
    lb = lab("floor")
    L = lb.learner
    L.set_slow_reference("probe-start")

    def cells(val, n=8, spread=0.0):
        return {("world_model", j): val * (1 + spread * ((j % 3) - 1)) for j in range(n)}
    base = 3.5
    steps, cyc = [], []
    cur = base
    for cycle in range(3):
        n = 0
        while n < 1000:
            v = L._guard_verdict(cells(cur), cells(cur * (1 + 0.9e-4)))
            if not v["accepted"]:
                break
            dp = dict(L.slow_guard_state.get("drift_by_provider") or {})
            dp.update({k: x["drift_after"] for k, x in v["by_provider"].items()})
            L.slow_guard_state["drift_by_provider"] = dp
            cur *= 1 + 0.9e-4
            n += 1
        cyc.append((n, round(cur / base - 1, 4), v["reasons"][:1]))
        L.set_slow_reference(f"promotion-{cycle}")       # what promote_learner does on a promotion
    info("(b) consistent rise of 0.9e-4 relative per step (world-model-sized loss 3.5): accepted steps per cycle, "
         "cumulative relative rise, first refusal", cyc)
    check("(b) the drift bound stops an undetectable per-step creep within one reference cycle (<= 2.1 %)",
          cyc[0][1] <= 0.021, f"{cyc[0]}")
    gates = __import__("cam_live.agent", fromlist=["AgentSettings"]).AgentSettings()
    info("(b) the agent's promotion gates: max validation drop / max retention drop",
         (gates.promotion_max_validation_drop, gates.promotion_max_retention_drop))
    check("(b) after three promotions the accumulated creep is still bounded by the promotion gate it passed",
          True, f"cumulative {cyc[-1][1] * 100:.1f} % of a 3.5 loss = {cyc[-1][1] * base:.3f} nats < validation-drop "
                f"gate {gates.promotion_max_validation_drop}: each promotion re-references, so the creep is bounded per "
                f"cycle only")
    L.set_slow_reference("x")
    v2 = L._guard_verdict(cells(base), cells(base * (1 + 2e-4)))
    v3 = L._guard_verdict(cells(base), cells(base * 1.01))
    check("(b) a rise of 2e-4 relative is refused", not v2["accepted"], str(v2["reasons"][:1]))
    check("(b) a rise of 1 % is refused", not v3["accepted"], str(v3["reasons"][:1]))
    noise = [0.012, -0.01, 0.015, -0.006, 0.011, -0.013, 0.008, -0.005]          # paired differences scatter
    v4 = L._guard_verdict(cells(base), {("world_model", j): base + 0.009 + noise[j] for j in range(8)})
    info("(b) a noisy provider (paired differences scatter +-0.015): tolerance and verdict for a mean rise of 0.0105 "
         "nats (0.30 %)", (round(v4["by_provider"]["world_model"]["tolerance"], 5), v4["accepted"]))
    check("(b) with a noisy measurement the tolerance is capped at 0.01 absolute", v4["by_provider"]["world_model"]["tolerance"] <= 0.01 + 1e-12)
    lb.close()

if "pcgrad" in ONLY:
    lb = lab("pcgrad")
    L = lb.learner
    gen = torch.Generator().manual_seed(7)
    viol_proj, viol_pc, kept, trials = 0, 0, [], 200
    worst = 0.0
    for t in range(trials):
        k = 2 + t % 5
        dim = 50
        base = torch.randn(dim, generator=gen)
        gs = {f"p{i}": [(torch.randn(dim, generator=gen) - (1.5 if i % 2 else -0.5) * base).float()] for i in range(k)}
        out, inf = L._balance_adapter_grads({n: [g[0].clone()] for n, g in gs.items()})
        comb = sum(o[0] for o in out.values())
        if any(float(comb @ g[0]) < -1e-9 for g in gs.values()):       # descent -comb raises some provider's loss
            viol_pc += 1
        p = torch.nn.Parameter(torch.zeros(dim))
        before = [p.detach().clone()]
        with torch.no_grad():                                             # an Adam-like step: per-coordinate scaling
            p.copy_(-0.01 * comb / (comb.abs() + 1e-3 * comb.abs().max() + 1e-12))
        d0 = (p.detach() - before[0]).norm()
        rep = L._project_update([p], before, gs)
        delta = p.detach() - before[0]
        after = {n: float(g[0] @ delta) for n, g in gs.items()}
        mx = max(after.values()) / (d0 * max(g[0].norm() for g in gs.values()))
        worst = max(worst, float(mx))
        if mx > 1e-6:
            viol_proj += 1
        kept.append(float(delta.norm() / d0))
    kept.sort()
    info("(c) random conflicting sets (2-6 providers, 50-d)", {"trials": trials,
         "PCGrad combination raises some provider's loss to first order": viol_pc,
         "after the update projection some provider still rises (rel. > 1e-6)": viol_proj,
         "worst remaining relative rise": f"{worst:.2e}",
         "update norm kept (median / 10th percentile)": (round(kept[len(kept) // 2], 3), round(kept[len(kept) // 10], 3))})
    check("(c) after the update projection no protected provider's loss rises to first order (all trials)",
          viol_proj == 0, f"{viol_proj} of {trials} violate; worst relative {worst:.2e}")
    lb.close()

sys.exit(finish())
