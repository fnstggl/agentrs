"""RW diagnosis (debug scale, CPU): which support decisions does the outer meta-gradient ask R_psi to reward?

    cd /workspace/camlive && PYTHONPATH=cam_live/src .venv/bin/python cam_live/docs/evidence/RW/rw_credit_probe.py \\
        --out cam_live/docs/evidence/RW/credit_probe.json

For the newest live update event of a LabWorld objective (``reward.diagnose.make_stack``, pooled first-order
estimator), the first-order outer surrogate S(psi) = <g, theta_K(psi)> is differentiated with respect to a per-row
offset delta_k added to R_psi's output on support row k (delta = 0): dS/d delta_k is the credit the meta-gradient
gives *reward at row k*. It is compared with

    c_k        = < g, grad_theta log pi_theta_0(a_k) >    how much reinforcing action k itself moves the policy along
                                                         the verified policy gradient g (the "right" per-row target)
    C_k        = sum_{i <= k in the same segment} c_i    the same accumulated over the earlier support rows
    y_k        = the row's verified progress

Inside the inner loss the reward enters through the discounted return G_i = r_i + gamma G_{i+1}; with a return
half-life much longer than the spacing of the support decisions (gamma ~ 1) the reward at row k is credited to every
earlier row, so dS/d delta_k should follow C_k rather than c_k. The probe measures this for the inner-loss half-life
of the configuration (1 day) and for a short one.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys

DAY = 86400.0


def _corr(a, b):
    n = len(a)
    if n < 3:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va <= 0 or vb <= 0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)


def probe(seed: int, half_life_s: float, n_decisions: int, update_every: int, utility: str = "return_to_go") -> dict:
    import torch
    from cam_live.diagnostics.world import LabConfig, LabWorld, lab_task
    from cam_live.diagnostics.collect import LabCollector
    from cam_live.reward import diagnose as dg
    from cam_live.reward.assemble import STORED_KEY
    auth = dg.LabMeasurementAuth()
    ec = dg.ExperimentConfig(min_independent=4)
    model, store, clock, ml = dg.make_stack(seed, ec, mix=1.0, mix_step=0.25, check_measurement=auth,
                                            estimator="pooled_first_order", half_life_s=half_life_s,
                                            utility=utility)
    lab = dict(ec.lab)
    lab.update(horizon_s=120 * DAY, budget=5000.0, target=100_000)
    world = LabWorld(LabConfig(seed=41_000 + seed, **lab), clock=clock)
    ver = dg.LabVerifier(world)
    gen = torch.Generator().manual_seed(50_000 + seed)
    act = dg.model_policy(ml, store, gen)
    buf, pending = [], []

    def annotate():
        keep = []
        for i in pending:
            rec = store.get(i)
            if rec.pending_queries:
                keep.append(i)
                continue
            ids = {q.query_id for q in rec.result.expected_outcomes if q.metric.startswith("lab.adopters.from.")}
            y = sum(float(o.value) for o in rec.outcomes if o.verified and o.value is not None and o.query_id in ids)
            rec.decision = dataclasses.replace(rec.decision, diagnostics={**(rec.decision.diagnostics or {}),
                                                                           STORED_KEY: {"verified_progress_delta": y}})
            store.update(rec)
        pending[:] = keep

    def on_step(rec):
        auth.register(rec, ver)
        ml.internal_rewards([rec])
        pending.append(rec.exp_id)
        buf.append(rec)
        if len(buf) >= update_every:
            ml.policy_update(list(buf))
            buf.clear()
        annotate()
    LabCollector(store, max_context=ec.max_context).run(world, act, n_decisions, seed=seed, on_step=on_step,
                                                         task=lab_task(world, objective_id=f"obj_probe_{seed}"),
                                                         finish=False)
    annotate()
    out = {"seed": seed, "half_life_s": half_life_s, "utility": utility, "events": []}
    for ev in ml.events[-4:]:
        rows, U, groups, _ = ml._pool(ev)
        if len(set(groups)) < 4:
            continue
        ib = ev.ib
        ibq = ml.inner.build_batch(store, rows, policy_only=True)
        ibq = ibq.select([ibq.exp_ids.index(r.exp_id) for r in rows])
        th = {n: v.detach().clone().requires_grad_(True) for n, v in ev.theta0.items()}
        lp = ml.inner.logp(th, ibq)
        rho = torch.exp(lp.detach().double() - ibq.behaviour_logp.double()).clamp(max=ml.mc.weight_clip)
        w = (rho / rho.sum()).float()
        n = len(U)
        base = [(sum(U) - sum(u for u, g in zip(U, groups) if g == gg)) / max(1, n - groups.count(gg)) for gg in groups]
        adv = torch.tensor([u - b for u, b in zip(U, base)])
        gs = torch.autograd.grad((w * lp * adv).sum(), list(th.values()), allow_unused=True)
        g = {k: (x if x is not None else torch.zeros_like(th[k])) for k, x in zip(th, gs)}
        # c_k: alignment of reinforcing support action k with g
        c = []
        for k in range(ib.n):
            sub = ib.select([k])
            lpk = ml.inner.logp(th, sub).sum()
            gk = torch.autograd.grad(lpk, list(th.values()), allow_unused=True)
            c.append(float(sum((g[n_] * (x if x is not None else 0.0)).sum() for n_, x in zip(th, gk))))
        # dS / d delta_k: offset added to R_psi's output on each support row
        delta = torch.zeros(ib.n, requires_grad=True)
        orig = ml.inner.learned_reward
        def shifted(psi, b, _o=orig):
            r = _o(psi, b)
            return r + delta.to(r.dtype)
        ml.inner.learned_reward = shifted
        try:
            psi = {k: v.detach() for k, v in ml.reward.psi().items()}
            with ml.reward.meta_window():
                steps, _ = ml._steps_of([ev])
                tr = ml.engine.inner_update({k: v for k, v in ev.theta0.items()}, psi, steps, ev.inner_lr,
                                            len(steps), create_graph=True, mode="unroll")
                S = sum((g[k].detach() * tr.theta_K[k]).sum() for k in g)
                dS = torch.autograd.grad(S, delta)[0].tolist()
        finally:
            ml.inner.learned_reward = orig
        segs = ib.segments
        C = []
        for s_, e_ in segs:
            acc = 0.0
            for k in range(s_, e_):
                acc += c[k]
                C.append(acc)
        recs = [store.get(i) for i in ib.exp_ids]
        from cam_live.reward.inner import verified_progress
        y = [verified_progress(r) if verified_progress(r) is not None else 0.0 for r in recs]
        gam = ib.gamma.tolist()
        out["events"].append({"event_id": ev.event_id, "n_support": ib.n, "pool_rows": n,
                              "mean_gamma_between_support_rows": sum(gam[:-1]) / max(1, len(gam) - 1),
                              "corr_dS_ddelta_with_c": _corr(dS, c), "corr_dS_ddelta_with_C": _corr(dS, C),
                              "corr_dS_ddelta_with_y": _corr(dS, y), "corr_c_with_y": _corr(c, y)})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--decisions", type=int, default=60)
    ap.add_argument("--update-every", type=int, default=6)
    ap.add_argument("--utility", choices=("return_to_go", "direct"), default="return_to_go")
    a = ap.parse_args(argv)
    res = {"kind": "debug_rw_credit_probe", "utility": a.utility, "runs": []}
    for hl in (DAY, 600.0):
        for s in a.seeds:
            r = probe(s, hl, a.decisions, a.update_every, a.utility)
            res["runs"].append(r)
            print(json.dumps(r), flush=True)
            with open(a.out, "w") as f:
                json.dump(res, f, indent=1)
    agg = {}
    for hl in (DAY, 600.0):
        evs = [e for r in res["runs"] if r["half_life_s"] == hl for e in r["events"]]
        agg[str(hl)] = {k: [e[k] for e in evs] for k in ("mean_gamma_between_support_rows", "corr_dS_ddelta_with_c",
                                                          "corr_dS_ddelta_with_C", "corr_dS_ddelta_with_y",
                                                          "corr_c_with_y")}
    res["summary"] = agg
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(agg, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
