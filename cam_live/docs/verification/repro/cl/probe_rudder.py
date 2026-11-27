"""G-cl probe: RUDDER return decomposition (RC-02, RC-03) and duration-aware discounting (CU-20).

Checks, on D's ReturnDecomposer with random (non-zero) weights:
  * conservation  sum_t Gamma_t c_t = G  with Gamma_t = exp(-beta * (t_t - t_first)) recomputed here (float64);
  * prefix causality: perturbing decisions >= j (latents and times) leaves pred[:, :j+1], signal[:, :j] and
    contrib[:, :j-1] bit-identical (G held fixed);
  * the learning signal equals (G - F(H_<t)) / Gamma_t and the return-to-go identity;
  * unmatured rows are NaN everywhere and give no loss; matured rows with padding;
  * nothing beyond G is ever distributed (sum of contributions of every matured row equals G, never more);
  * trajectory_return on stored records: one event per dedup key, duration-aware discount, matured only when closed.

Run from /workspace/camlive:  .venv/bin/python cam_live/docs/verification/repro/cl/probe_rudder.py
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys

import torch

sys.path.insert(0, "cam_live/src")
sys.path.insert(0, "cam_live/tests/credit")
from cam_live.credit.rudder import ReturnDecomposer, duration_discount      # noqa: E402
FAILS, OUT = [], {}
WORK = os.environ.get("CL_WORK", "/tmp/claude-0/-workspace/8e531f56-63d9-4df0-a584-c2fb645312ce/scratchpad/G_cl/work_rudder")


def check(name, ok, detail=""):
    print(("[PASS] " if ok else "[FAIL] ") + name + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def randomise(m: ReturnDecomposer, seed: int):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(0.3 * torch.randn(p.shape, generator=g))
        m.scale.fill_(3.7)


def decomposer_checks():
    torch.manual_seed(0)
    B, T, D = 7, 13, 10
    for beta in (0.0, 1e-4, 2e-3, 0.05):
        m = ReturnDecomposer(D, d_model=32, n_layers=2, n_heads=4, d_goal=6, beta=beta)
        randomise(m, 1)
        g = torch.Generator().manual_seed(2)
        x = torch.randn(B, T, D, generator=g)
        e_g = torch.randn(B, 6, generator=g)
        lens = torch.tensor([13, 13, 9, 5, 1, 13, 7])
        mask = torch.arange(T)[None, :] < lens[:, None]
        gaps = torch.rand(B, T, generator=g, dtype=torch.float64) * 600.0
        times = 1.7e9 + torch.cumsum(gaps, 1)
        G = torch.randn(B, generator=g, dtype=torch.float64) * 5
        matured = torch.tensor([True, True, True, True, True, False, True])
        red = m.redistribute(x, mask, G, matured, e_g, times)
        # own discount
        first = times[:, :1]
        gam = torch.exp(-beta * (times - first))
        tot = torch.where(mask, gam * red.contrib, torch.zeros_like(red.contrib)).nansum(1)
        err = (tot - G).abs()[matured]
        check(f"beta={beta}: sum_t Gamma_t c_t = G (own Gamma)", float(err.max()) < 1e-9 * max(1.0, float(G.abs().max())),
              f"max err {float(err.max()):.2e}")
        OUT.setdefault("conservation_max_err", {})[str(beta)] = float(err.max())
        dg = (red.discount - gam).abs()[mask & matured[:, None]]
        check(f"beta={beta}: D's Gamma_t equals exp(-beta * seconds since first decision)", float(dg.max()) < 1e-12)
        # signal identity
        pred = red.pred
        sig_own = (G[:, None] - pred[:, :-1]) / gam
        ok = mask & matured[:, None]
        check(f"beta={beta}: signal = (G - F(H_<t)) / Gamma_t", float((sig_own - red.signal).abs()[ok].max()) < 1e-9)
        rtg_own = torch.flip(torch.cumsum(torch.flip(torch.where(mask, gam * red.contrib, torch.zeros_like(gam)), [1]), 1), [1]) / gam
        check(f"beta={beta}: return_to_go = sum_{{k>=t}} Gamma_k c_k / Gamma_t",
              float((rtg_own - red.return_to_go).abs()[ok].max()) < 1e-9)
        # unmatured row
        check(f"beta={beta}: unmatured row is NaN in contrib / signal / rtg", bool(torch.isnan(red.contrib[5]).all())
              and bool(torch.isnan(red.signal[5]).all()) and bool(torch.isnan(red.return_to_go[5]).all()))
        # padding of matured rows carries no credit
        pad = (~mask) & matured[:, None]
        check(f"beta={beta}: padded positions of matured rows carry exactly 0", float(red.contrib[pad].abs().max()) == 0.0)
        # no extra return: total distributed (undiscounted to start) equals G for each matured row, never more
        # prefix causality
        for j in (1, 4, 8, 12):
            x2, t2 = x.clone(), times.clone()
            x2[:, j:] = torch.randn(B, T - j, D, generator=g) * 5
            t2[:, j:] = t2[:, j:] + torch.rand(B, T - j, generator=g, dtype=torch.float64) * 1e5
            t2[:, j:] = torch.cummax(t2[:, j:], 1).values
            red2 = m.redistribute(x2, mask, G, matured, e_g, t2)
            same_pred = torch.equal(red.pred[:, :j + 1], red2.pred[:, :j + 1])
            okj = ok[:, :j]
            same_sig = torch.equal(torch.nan_to_num(red.signal[:, :j]), torch.nan_to_num(red2.signal[:, :j]))
            # contributions of decisions < j-1 (j-1 is the step whose 'after' prediction is still the prefix)
            same_c = torch.equal(torch.nan_to_num(red.contrib[:, :max(j - 1, 0)]),
                                 torch.nan_to_num(red2.contrib[:, :max(j - 1, 0)])) if j > 1 else True
            # except the last step of a row (residual uses the final prediction) -> rows with len <= j are excluded
            long_rows = lens > j
            same_c_long = torch.equal(torch.nan_to_num(red.contrib[long_rows, :j]),
                                      torch.nan_to_num(red2.contrib[long_rows, :j])) if j < T else True
            check(f"beta={beta}: perturbing decisions >= {j} leaves pred[:, :{j}+1] and signal[:, :{j}] unchanged",
                  same_pred and same_sig, f"pred {same_pred} signal {same_sig}")
            check(f"beta={beta}: ... and contributions c_t, t < {j}, of rows longer than {j}", same_c_long)
    # loss: nothing matured -> None; matured rows only
    m = ReturnDecomposer(4, d_model=16, n_layers=1, n_heads=2)
    randomise(m, 3)
    x = torch.randn(3, 5, 4)
    mask = torch.ones(3, 5, dtype=torch.bool)
    G = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    l0, met = m.loss(x, mask, G, torch.zeros(3, dtype=torch.bool))
    check("loss is None when nothing matured (no invented target)", l0 is None, str(met))
    l1, _ = m.loss(x, mask, G, torch.tensor([True, False, False]))
    G2 = G.clone(); G2[1:] = 1e6
    l2, _ = m.loss(x, mask, G2, torch.tensor([True, False, False]))
    check("the G of unmatured rows has no influence on the loss", torch.equal(l1, l2))
    # gradient of the loss w.r.t. unmatured rows' inputs is zero
    xg = x.clone().requires_grad_(True)
    l3, _ = m.loss(xg, mask, G, torch.tensor([True, False, False]))
    l3.backward()
    check("no gradient flows from unmatured rows", float(xg.grad[1:].abs().max()) == 0.0)


def store_checks():
    """trajectory_return on real stored records: duration-aware discount, one event per dedup key, matured when closed."""
    from conftest import build_record
    from cam_live.contracts import OutcomeObservation
    from cam_live.credit.replay import SqliteExperienceStore
    from cam_live.credit.providers import trajectory_return
    if os.path.exists(WORK):
        shutil.rmtree(WORK)
    os.makedirs(WORK)
    st = SqliteExperienceStore(os.path.join(WORK, "s.sqlite"))
    beta = 1e-3
    recs = [build_record(i, ts=1000.0 + 150 * i, horizon=1000.0, objective="o1") for i in range(4)]
    for r in recs:
        st.append(r)
    obs = {0: (5.0, 1500.0, "k0"), 1: (2.0, 1700.0, "k1"), 2: (-1.0, 2000.0, "k2")}
    for i, (v, t, k) in obs.items():
        st.add_outcome(OutcomeObservation(query_id=f"q{i}", status="observed", value=v, observed_at=t, dedup_key=k))
    # the same event (dedup key k1) also reported as the answer to q3's query: counted once
    st.add_outcome(OutcomeObservation(query_id="q3", status="observed", value=2.0, observed_at=1700.0, dedup_key="k1"))
    tr = trajectory_return(st, st.trajectory("o1"), beta=beta)
    G_own = sum(v * math.exp(-beta * (t - 1000.0)) for v, t, _ in obs.values())
    check("trajectory_return: open objective is not matured", not tr.matured, tr.reason)
    check("trajectory_return: G = sum of distinct events discounted to the first decision", abs(tr.G - G_own) < 1e-12,
          f"G={tr.G:.6f} own={G_own:.6f}; events {[(e.key, e.value) for e in tr.events]}")
    st.put_objective_outcome("o1", status="achieved", value=1.0, evaluated_at=3000.0, verified=True)
    tr2 = trajectory_return(st, st.trajectory("o1"), beta=beta)
    check("trajectory_return: closed objective with all queries answered is matured", tr2.matured, tr2.reason)
    # a pending goal-relevant query keeps it unmatured
    r = build_record(9, ts=2500.0, horizon=1e5, objective="o2")
    st.append(r)
    st.put_objective_outcome("o2", status="achieved", value=1.0, evaluated_at=3000.0)
    tr3 = trajectory_return(st, st.trajectory("o2"), beta=beta)
    check("trajectory_return: a pending goal-relevant query keeps the return unmatured", not tr3.matured, tr3.reason)
    OUT["trajectory_return"] = {"G": tr.G, "G_own": G_own, "n_events": len(tr.events), "verified_only_default": False}
    st.close()


def main():
    decomposer_checks()
    store_checks()
    print(json.dumps(OUT, indent=1))
    if FAILS:
        print("FAILED:", FAILS)
        sys.exit(1)


if __name__ == "__main__":
    main()
