"""MetaLearner on the real debug model in LabWorld with the CURRENT inner loss switched on in full (MR-02..MR-07,
MR-10, KF-E01): trust-region live step, matured replay, goal term (verified / per-decision credit / calibrated
prediction), all-candidate model term, return-like credit (KF-E01) replacing the goal return.

The LabCollector records carry no predictions or verified annotations of their own, so this probe ADDS them as data
(seeded, deterministic): random q_goal on every candidate, a signed-looking `reward_signals.verified_progress_delta`
on half of the decisions, and a credit hook returning RUDDER-like (return-like) credit for one third and simple
(per-decision) credit for another third of the decisions. They are inputs of the loss, not outcomes; the outer
utilities still come only from the meta ledger (signed LabWorld verdicts). Everything below the data is my own:

  * my L_inner from the documented formula (reward/inner.py module doc, docs/C/LRN_S5.md 2.2, docs/INT3A.md A):
      r = mix n(r_boot) + (1-mix) n(R_psi) + c_g u/rms(u);  G discounted within the lifetime;  credited rows take
      c_g C/rms(C) as their goal return;  A = n(G) - V(theta_sg);  - mean log pi A + c_v value loss - c_e entropy
      - c_m beta_q^+ mean_i sum_j pi_ij z_ij;  per-prefix norm clip;  theta <- theta - alpha g;
  * my goal sources / standardised candidate values recomputed from the records;
  * my outer estimator (rows with utility not held out, weight 1/N, baseline = mean utility outside the row's
    baseline group else its prefix baseline, on-policy iff the logged policy is the recomputed one, else clipped
    importance weights with the ESS rule);
  * central finite differences of MY outer surrogate along a random psi direction in float64.
R_psi is used only as the function psi -> rewards (PolicyImprovementLoss.learned_reward).
"""
import hashlib
import math
import random
import sys

import torch
from torch.func import functional_call

import _g2d as C
from cam_live.diagnostics.collect import LabCollector, verified_return
from cam_live.diagnostics.world import LabConfig, LabWorld, lab_task
from cam_live.predict.policy import selection_calibration
from cam_live.reward import meta as M
from cam_live.reward.diagnose import ExperimentConfig, make_stack, model_policy
from cam_live.reward.inner import verified_progress
from cam_live.reward.integrity import params_hash

torch.manual_seed(0)
SEED = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 5
ec = ExperimentConfig(min_independent=3, n_decisions=16)
model, store, clock, ml = make_stack(SEED, ec, mix=0.0, target_kl=0.05, replay_matured=16, goal_coef=1.0,
                                     model_coef=1.0)
cfg, mc = ml.inner.cfg, ml.mc
C.info("config", {"target_kl": mc.target_kl, "replay_matured": mc.replay_matured, "goal_coef": cfg.goal_coef,
                  "model_coef": cfg.model_coef, "component_norm": cfg.component_norm, "meta_mix": mc.meta_mix,
                  "inner_lr(lo)": mc.inner_lr, "max_inner_lr": mc.max_inner_lr, "K": mc.inner_steps})


def h01(s, salt):
    return int(hashlib.sha256(f"{salt}|{s}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def credit_fn(rec):
    u = h01(rec.exp_id, "bucket")
    if u < 1 / 3:
        return (4.0 * h01(rec.exp_id, "c") - 1.0, "rudder", True)
    if u < 2 / 3:
        return (2.0 * h01(rec.exp_id, "s"), "simple", False)
    return None


ml.inner.credit_fn = credit_fn


def annotate(rec):
    rng = random.Random(rec.exp_id)
    for c in rec.decision.candidates:
        c.extras = dict(c.extras or {})
        c.extras["q_goal"] = rng.gauss(0.0, 1.0)
    if rng.random() < 0.5:
        d = rec.decision.diagnostics if isinstance(rec.decision.diagnostics, dict) else {}
        d["reward_signals"] = {"verified_progress_delta": rng.gauss(0.5, 1.0)}
        rec.decision.diagnostics = d
    store.update(rec)


# ------------------------------------------------------------------------------------------------- my inner loss
def fwd(th, lat):
    tr = {k[6:]: v for k, v in th.items() if k.startswith("trunk.")}
    pol = {k[7:]: v for k, v in th.items() if k.startswith("policy.")}
    z = functional_call(ml.subset.trunk, tr, (lat.b, lat.e_a, lat.e_g))
    out = functional_call(ml.subset.policy, pol, (z, lat.a_mask, lat.b))
    neg = torch.finfo(out.raw_logits.dtype).min
    lpa = torch.log_softmax(torch.where(lat.a_mask, out.raw_logits, torch.full_like(out.raw_logits, neg)), -1)
    return lpa, out.value


def nrm(x):
    d = x - x.mean()
    return d / torch.sqrt((d ** 2).mean() + 1e-12)


def my_goal_inputs(lat):
    """(u, known, credit values, credit mask, z_pred, z_ok, model weight) recomputed from the stored records."""
    cal = selection_calibration(ml.policy)
    u, src, cv, cok, Z, zok = [], [], [], [], [], []
    n_c = int(lat.a_mask.shape[1])
    for i in lat.exp_ids:
        rec = store.get(i)
        c = credit_fn(rec)
        v = verified_progress(rec)
        if v is not None:
            u.append(v); src.append("verified")
        elif c is not None and not c[2]:
            u.append(c[0]); src.append("credit:" + c[1])
        else:
            f = cal.features(rec.decision)
            p = cal.predict(f[0]) if f is not None else None
            if p is None:
                ch = rec.decision.candidates[rec.decision.chosen_index].extras.get("q_goal")
                p = None if ch is None else float(ch)
            u.append(0.0 if p is None else p); src.append("" if p is None else "predicted")
        if c is not None and c[2]:
            cv.append(c[0]); cok.append(True); src[-1] = "credit:" + c[1]
        else:
            cv.append(0.0); cok.append(False)
        # standardised predicted value of every candidate (population sd over the feasible ones, winsorised at 4)
        cands = rec.decision.candidates
        feas = [bool(x.feasible) for x in cands]
        feas[rec.decision.chosen_index] = True
        q = [float(x.extras["q_goal"]) - float(x.extras.get("net_cost", 0.0) or 0.0) for x in cands]
        fq = [qq for qq, f in zip(q, feas) if f]
        mu = sum(fq) / len(fq)
        sd = math.sqrt(sum((qq - mu) ** 2 for qq in fq) / len(fq))
        sd = sd if sd > 1e-12 else 1.0
        z = [max(-4.0, min(4.0, (qq - mu) / sd)) if f else 0.0 for qq, f in zip(q, feas)]
        Z.append(z + [0.0] * (n_c - len(z))); zok.append(len(z) <= n_c)
    w = max(float(cal.posterior()["mean"][1]), 0.0)
    return (torch.tensor(u), src, torch.tensor(cv), torch.tensor(cok), torch.tensor(Z), torch.tensor(zok), w)


def my_loss(th, th_sg, psi, lat, mix, gi):
    u, src, cvals, cok, Z, zok, w = gi
    dt = lat.b.dtype
    lpa, v = fwd(th, lat)
    _, vb = fwd(th_sg, lat)
    rl = ml.inner.learned_reward(psi, lat)
    rb = lat.r_boot.to(rl.dtype)
    r = nrm(rb) if mix >= 1.0 else (nrm(rl) if mix <= 0.0 else mix * nrm(rb) + (1.0 - mix) * nrm(rl))
    known = torch.tensor([bool(s) for s in src]) & ~cok
    g = torch.zeros(lat.n, dtype=dt)
    if bool(known.any()):
        uu = u.to(dt)
        rms = torch.sqrt((uu[known] ** 2).mean())
        if float(rms) > 0:
            g = cfg.goal_coef * torch.where(known, uu / rms, torch.zeros_like(uu))
    cc = cvals.to(dt)
    if bool(cok.any()):
        rmsc = torch.sqrt((cc[cok] ** 2).mean())
        cc = cfg.goal_coef * (cc / rmsc if float(rmsc) > 0 else cc)
    gam = lat.gamma.to(dt)
    Gr, Gg, acc_r, acc_g = [None] * lat.n, [None] * lat.n, torch.zeros((), dtype=dt), torch.zeros((), dtype=dt)
    for i in reversed(range(lat.n)):
        acc_r = r[i] + gam[i] * acc_r
        acc_g = cc[i] if bool(cok[i]) else g[i] + gam[i] * acc_g
        Gr[i], Gg[i] = acc_r, acc_g
    G = torch.stack(Gr) + torch.stack(Gg)
    Gn = nrm(G)
    lp = lpa[torch.arange(lat.n), lat.chosen]
    p = torch.where(lat.a_mask, lpa.exp(), torch.zeros_like(lpa))
    ent = -(p * torch.where(lat.a_mask, lpa, torch.zeros_like(lpa))).sum(-1)
    loss = -(lp * (Gn - vb)).mean() + cfg.value_coef * (0.5 * (v - Gn) ** 2).mean() - cfg.entropy_coef * ent.mean()
    if cfg.model_coef and w > 0 and bool(zok.any()):
        loss = loss - cfg.model_coef * w * (p * Z.to(dt)).sum(-1)[zok].mean()
    return loss


def my_unroll(theta0, psi, lat, mix, K, lr, gi, graph):
    """K SGD steps of MY loss from theta0 (a constant). With ``graph`` the unroll keeps the graph to psi; the value
    baseline of step k uses a SIBLING node of theta_k (same parents, computed separately), so it is held fixed for
    the inner derivative w.r.t. theta_k but still carries the psi-dependence of theta_k (step 0: theta0, a constant)."""
    names = list(theta0)
    th = {k: v.detach().clone().requires_grad_(True) for k, v in theta0.items()}
    sib = {k: v.detach() for k, v in theta0.items()}
    for _ in range(K):
        if not graph:
            th = {k: v.detach().requires_grad_(True) for k, v in th.items()}
            sib = {k: v.detach() for k, v in th.items()}
        L = my_loss(th, sib, psi, lat, mix, gi)
        gs = torch.autograd.grad(L, [th[k] for k in names], create_graph=graph, allow_unused=True)
        g = {k: (torch.zeros_like(th[k]) if x is None else x) for k, x in zip(names, gs)}
        for pre, mx in (("policy.", mc.clip_policy), ("trunk.", mc.clip_trunk)):
            ks = [k for k in names if k.startswith(pre)]
            if not ks or not mx:
                continue
            n_ = torch.sqrt(sum((g[k] ** 2).sum() for k in ks) + 1e-30)
            s = torch.clamp(mx / n_, max=1.0)
            for k in ks:
                g[k] = g[k] * s
        new = {k: th[k] - lr * g[k] for k in names}
        if graph:
            sib = {k: th[k] - lr * g[k] for k in names}     # sibling of the next theta (separate graph node)
        th = new
    return {k: v.detach() for k, v in th.items()} if not graph else th


# ------------------------------------------------------------------------------------------------- the run
ev_checks = []
PSI_AT_EVENT = {}
GI_AT_EVENT = {}           # event id -> my goal / model-term inputs computed when the live update was applied          # event id -> hash of psi's PARAMETERS when the live update was applied (my own record)
captured = {}


def psi_param_hash():
    h = hashlib.sha256()
    for n, p in sorted(ml.reward.named_parameters()):
        h.update(n.encode())
        h.update(p.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
orig_apply = M.MetaGradient.apply


def capture_apply(optimizer, reward_params, result, clip=None):
    captured["result"] = result
    return orig_apply(optimizer, reward_params, result, clip=clip)


M.MetaGradient.apply = staticmethod(capture_apply)
meta_checked = {"n": 0}
CODE_EST = []
_orig_mg = ml.engine.meta_gradient


def _spy_mg(trace, query, estimator="on_policy", **kw):
    res = _orig_mg(trace, query, estimator, **kw)
    CODE_EST.append((estimator, res.supported, res.reason, int(query.utility.shape[0]),
                     float(res.ess) if res.ess is not None else None))
    return res


ml.engine.meta_gradient = _spy_mg


def check_live_update(recs_given):
    n_ev = len(ml.events)
    rec_upd = ml.policy_update(list(recs_given))
    if rec_upd is None or len(ml.events) == n_ev:
        return
    ev = ml.events[-1]
    PSI_AT_EVENT[ev.event_id] = psi_param_hash()
    GI_AT_EVENT[ev.event_id] = my_goal_inputs(ev.ib)      # my goal / model inputs as of the live update
    theta1 = ml.subset.snapshot()
    lat = ev.ib
    psi = {k: v.detach() for k, v in ml.reward.psi().items()}
    gi = my_goal_inputs(lat)
    ok_goal = torch.allclose(gi[0], lat.r_goal.cpu().float(), atol=1e-5, rtol=1e-5) and gi[1] == list(lat.goal_source)
    ok_z = lat.z_pred is not None and torch.allclose(gi[4], lat.z_pred.cpu().float(), atol=1e-5) and \
        abs(gi[6] - lat.model_weight) < 1e-9 and \
        (lat.g_credit_ok is not None) == bool(gi[3].any()) and \
        (lat.g_credit_ok is None or (torch.equal(gi[3], lat.g_credit_ok.cpu()) and
                                     torch.allclose(gi[2], lat.g_credit.cpu().float())))
    mine = my_unroll(ev.theta0, psi, lat, ev.mix, ev.steps, ev.inner_lr, gi, graph=False)
    num = max(float((mine[k] - theta1[k]).abs().max()) for k in theta1)
    den = max(float((theta1[k] - ev.theta0[k]).abs().max()) for k in theta1)
    replay = ml.engine.inner_update(ev.theta0, psi, [lat], ev.inner_lr, ev.steps, create_graph=False).detached()
    kl = float(rec_upd.prediction_change["policy_kl"])
    lo, hi = float(mc.inner_lr), float(mc.max_inner_lr)
    at_bound = abs(ev.inner_lr - lo) <= 1e-9 * lo or abs(ev.inner_lr - hi) <= 1e-9 * hi
    in_region = mc.target_kl / 1.5 <= kl <= 1.5 * mc.target_kl
    srcs = {s: list(lat.goal_source).count(s) for s in set(lat.goal_source)}
    ev_checks.append(dict(event=ev.event_id, n=lat.n, lr=ev.inner_lr, kl=kl, at_bound=at_bound, in_region=in_region,
                          mine_vs_live=num / max(den, 1e-30), update=den, replay_bit_identical=params_hash(replay)
                          == ev.theta1_hash, goal_inputs_equal=bool(ok_goal), model_inputs_equal=bool(ok_z),
                          sources=srcs, replayed=rec_upd.diagnostics.get("n_replayed_matured"),
                          step_search=rec_upd.diagnostics.get("step_search")))


DBG = []


def my_meta_gradient(eps, D=None):
    """My outer surrogate gradient (and the surrogate as a function for finite differences)."""
    rows = [(k, a) for k, e in enumerate(eps) if e.ibq is not None for a in range(len(e.exp_ids))
            if e.has_u[a] and not e.held[a]]
    N = len(rows)
    tot = sum(eps[k].utility[a] for k, a in rows)
    gs, gc = {}, {}
    for k, a in rows:
        g = eps[k].group[a]
        gs[g] = gs.get(g, 0.0) + eps[k].utility[a]
        gc[g] = gc.get(g, 0) + 1
    base = {}
    for k, a in rows:
        g = eps[k].group[a]
        rest = N - gc[g]
        base[(k, a)] = (tot - gs[g]) / rest if rest > 0 else (eps[k].prefix_base[a] if eps[k].prefix_base else 0.0)
    psi0 = {k: v.detach().clone() for k, v in ml.reward.psi().items()}
    plan = []
    for k, e in enumerate(eps):
        idx = [a for a in range(len(e.exp_ids)) if e.has_u[a] and not e.held[a]]
        if e.ibq is None or not idx:
            continue
        if len(e.events) != 1:
            raise RuntimeError("probe assumes unroll_events == 1")
        ev = e.events[0]
        lat = ev.ib.with_mix(mc.meta_mix if mc.meta_mix is not None else ev.mix)
        gi = GI_AT_EVENT[ev.event_id]     # the inputs of the update that was applied (not today's calibration)
        # documented condition (MetaLearner.logged_policy_is_current): the queries were sampled after a live update
        # applied here, recomputed with the CURRENT psi and the differentiated mix -> the same policy. Input
        # statistics are frozen in the cached batch, so: psi parameters unchanged since the event and same mix.
        current = (not ev.external and ev.theta1_hash != "" and PSI_AT_EVENT.get(ev.event_id) == psi_param_hash()
                   and lat.mix == ev.mix)
        plan.append((k, e, ev, lat, gi, idx, current))

    def surrogate(psi, dtype=None, weights=None):
        tot_s, used, ws = 0.0, [], []
        dbg = []
        for j, (k, e, ev, lat, gi, idx, current) in enumerate(plan):
            lt = lat if dtype is None else lat.to(dtype)
            th0 = ev.theta0 if dtype is None else {n: v.to(dtype) for n, v in ev.theta0.items()}
            gi_ = gi
            thK = my_unroll(th0, psi, lt, lt.mix, ev.steps, ev.inner_lr, gi_, graph=True)
            q = e.ibq.select(idx)
            q = q if dtype is None else q.to(dtype)
            lpa, _ = fwd(thK, q)
            lp = lpa[torch.arange(len(idx)), q.chosen]
            blp = q.behaviour_logp.to(lp.dtype)
            if weights is not None:
                w, est = weights[j]
            else:
                gap = float((lp.detach().exp().double() - blp.exp().double()).abs().max())
                if "--debug" in sys.argv and dtype is None:
                    steps_c, _ = ml._steps_of(e.events)
                    tr_c = ml.engine.inner_update(ev.theta0, {n_: v_.detach() for n_, v_ in ml.reward.psi().items()},
                                                  steps_c, ev.inner_lr, len(steps_c), create_graph=True)
                    lpc = ml.inner.logp(tr_c.detached(), q).detach()
                    gap_c = float((lpc.exp().double() - blp.exp().double()).abs().max())
                    d_mine_code = float((lpc - lp.detach()).abs().max())
                    print("[DEBUG] event", ev.event_id, "K", len(steps_c), ev.steps, "lr", ev.inner_lr, "gap mine", gap,
                          "gap code", gap_c, "max|lp_code-lp_mine|", d_mine_code, flush=True)
                on = current and bool(torch.isfinite(blp).all()) and gap <= mc.on_policy_tol
                dbg.append((ev.event_id, current, round(gap, 6), len(idx)))
                w = torch.full((len(idx),), 1.0 / N, dtype=lp.dtype)
                est = "on"
                if not on:
                    rho = torch.exp(lp.detach() - blp)
                    ess = float(rho.sum() ** 2 / (rho ** 2).sum())
                    if ess < max(mc.min_ess, mc.min_ess_fraction * len(idx)):     # code: per lifetime; one here
                        ws.append((None, "ess"))
                        used.append("ess_skip")
                        continue
                    w = w * rho.clamp(max=mc.weight_clip)
                    est = "iw"
            if w is None:
                continue
            ws.append((w.detach(), est))
            used.append(est)
            U = torch.tensor([e.utility[a] for a in idx], dtype=lp.dtype)
            b = torch.tensor([base[(k, a)] for a in idx], dtype=lp.dtype)
            tot_s = tot_s + (w.to(lp.dtype) * lp * (U - b)).sum()
        if weights is None:
            DBG.append(dbg)
        return tot_s, used, ws

    psi_v = {k: v.detach().clone().requires_grad_(True) for k, v in psi0.items()}
    val, used, ws = surrogate(psi_v)
    names = list(psi_v)
    grads = torch.autograd.grad(val, [psi_v[n] for n in names], allow_unused=True) if torch.is_tensor(val) \
        and val.requires_grad else [None] * len(names)
    g = {n: (torch.zeros_like(psi0[n]) if x is None else x) for n, x in zip(names, grads)}
    return g, used, (surrogate, ws, psi0)


def on_meta(L):
    ok, why = ml.ready()
    if not ok:
        rec = ml.meta_update()
        C.info(f"lifetime {L}: meta-update skipped", rec.skipped_reason)
        return
    eps, _, _ = ml._collect()
    mine, used, fd_parts = my_meta_gradient(eps)
    h = params_hash(ml.reward)
    captured.clear()
    CODE_EST.clear()
    rec = ml.meta_update()
    C.info(f"lifetime {L}: code per-episode (estimator, supported, reason, rows, ess)", list(CODE_EST))
    C.info(f"lifetime {L}: mine per-episode", DBG[-1] if DBG else None)
    res = captured.get("result")
    if not rec.applied or res is None:
        C.info(f"lifetime {L}: meta-update not applied", rec.skipped_reason)
        return
    num = math.sqrt(sum(float(((res.grads[n] - mine[n]) ** 2).sum()) for n in mine))
    den = math.sqrt(sum(float((mine[n] ** 2).sum()) for n in mine))
    meta_checked["n"] += 1
    C.check(f"lifetime {L}: applied meta-gradient == my recomputation (current loss incl. goal / credit / model "
            f"terms, trust-region step)", den > 0 and num / den < 1e-3,
            f"rel_err={num / max(den, 1e-30):.2e} |g|={den:.3e} code estimator={rec.estimator} mine={used}")
    C.check(f"lifetime {L}: psi changed by the applied step", params_hash(ml.reward) != h)
    # finite differences of MY surrogate (weights frozen at psi0) in float64 along a random direction
    surrogate, ws, psi0 = fd_parts
    D = torch.float64
    gen = torch.Generator().manual_seed(L)
    d = {n: torch.randn(v.shape, generator=gen, dtype=D) for n, v in psi0.items()}
    dn = math.sqrt(sum(float((x ** 2).sum()) for x in d.values()))
    d = {n: x / dn for n, x in d.items()}
    ws64 = [(None if w is None else w.to(D), e) for w, e in ws]
    base64 = {n: v.to(D) for n, v in psi0.items()}
    with torch.no_grad():
        pass
    eps_ = 1e-5
    fp = surrogate({n: base64[n] + eps_ * d[n] for n in d}, dtype=D, weights=ws64)[0]
    fm = surrogate({n: base64[n] - eps_ * d[n] for n in d}, dtype=D, weights=ws64)[0]
    fd = float((fp - fm) / (2 * eps_))
    psi_v = {n: v.clone().requires_grad_(True) for n, v in base64.items()}
    val = surrogate(psi_v, dtype=D, weights=ws64)[0]
    gg = torch.autograd.grad(val, list(psi_v.values()), allow_unused=True)
    an = sum(float((x * d[n]).sum()) for x, n in zip(gg, psi_v) if x is not None)
    code_dir = sum(float((res.grads[n].to(D) * d[n]).sum()) for n in d)
    C.check(f"lifetime {L}: my analytic directional derivative == central FD (float64)",
            abs(an - fd) <= 1e-5 * max(abs(fd), 1e-12) + 1e-12, f"analytic={an:.6e} fd={fd:.6e}")
    C.check(f"lifetime {L}: code's gradient along the same direction == FD (float32 code vs float64 FD)",
            abs(code_dir - fd) <= 2e-3 * max(abs(fd), 1e-12) + 1e-9, f"code={code_dir:.6e} fd={fd:.6e}")


for L in range(8):
    world = LabWorld(LabConfig(seed=700_000 + 1000 * SEED + L, **ec.lab), clock=clock)
    ml.new_lifetime()
    act = model_policy(ml, store, torch.Generator().manual_seed(100 * SEED + L))
    buf = []

    def on_step(rec):
        annotate(rec)
        ml.internal_rewards([rec])
        buf.append(rec)
        if len(buf) >= ec.update_every:
            check_live_update(buf)
            buf.clear()
    lt = LabCollector(store, max_context=ec.max_context, seq_start=L * 1000 + SEED * 100_000).run(
        world, act, ec.n_decisions, seed=SEED * 7 + L, on_step=on_step,
        task=lab_task(world, objective_id=f"obj_g2d_{SEED}_{L}"))
    if len(buf) >= 2:
        check_live_update(buf)
    r = ml.admit_objective(lt.verified, utility=verified_return(lt))
    if not r.accepted:
        C.info(f"lifetime {L}: objective refused", r.reason)
    on_meta(L)
    if meta_checked["n"] >= 2:
        break

# ---- K sweep (MR-07: K in 1, 2, 4, 8) on the real adaptable subset with the CURRENT loss: the code's engine
# (MetaLearner.engine: its inner_loss + GroupClip) vs my unroll vs central finite differences, float64
ev = next(e for e in reversed(ml.events) if e.ib is not None and e.ib.n >= 4)
D = torch.float64
lat = ev.ib.with_mix(0.0).to(D)
gi = GI_AT_EVENT[ev.event_id]
th0 = {n: v.to(D) for n, v in ev.theta0.items()}
psi0 = {n: v.detach().clone().to(D) for n, v in ml.reward.psi().items()}
qrec = [store.get(i) for i in ev.exp_ids]
ibq = ml.inner.build_batch(store, qrec, policy_only=True).to(D)
gU = torch.Generator().manual_seed(123)
U = torch.randn(ibq.n, generator=gU, dtype=D)
gd = torch.Generator().manual_seed(321)
dirn = {n: torch.randn(v.shape, generator=gd, dtype=D) for n, v in psi0.items()}
dn = math.sqrt(sum(float((x ** 2).sum()) for x in dirn.values()))
dirn = {n: x / dn for n, x in dirn.items()}
LR = 0.3


def J_mine(psi, K):
    thK = my_unroll(th0, psi, lat, 0.0, K, LR, gi, graph=True)
    lpa, _ = fwd(thK, ibq)
    return (lpa[torch.arange(ibq.n), ibq.chosen] * U).sum() / ibq.n


ks = {}
for K in (1, 2, 4, 8):
    pv = {n: v.clone().requires_grad_(True) for n, v in psi0.items()}
    g_m = torch.autograd.grad(J_mine(pv, K), list(pv.values()), allow_unused=True)
    an = sum(float((x * dirn[n]).sum()) for x, n in zip(g_m, pv) if x is not None)
    e_ = 1e-5
    fd = float((J_mine({n: psi0[n] + e_ * dirn[n] for n in psi0}, K)
                - J_mine({n: psi0[n] - e_ * dirn[n] for n in psi0}, K)).detach() / (2 * e_))
    pc = {n: v.clone().requires_grad_(True) for n, v in psi0.items()}
    tr = ml.engine.inner_update(th0, pc, [lat], LR, K, create_graph=True)
    q = M.QueryData(batch=ibq, utility=U, baseline=torch.zeros(ibq.n, dtype=D), behaviour_logp=None,
                    weights=torch.full((ibq.n,), 1.0 / ibq.n, dtype=D), groups=torch.arange(ibq.n),
                    verified=torch.ones(ibq.n, dtype=torch.bool))
    res = ml.engine.meta_gradient(tr, q, "on_policy")
    cd = sum(float((res.grads[n] * dirn[n]).sum()) for n in dirn) if res.supported else float("nan")
    num = math.sqrt(sum(float(((res.grads[n] - (g if g is not None else 0)) ** 2).sum())
                        for n, g in zip(pv, g_m))) if res.supported else float("nan")
    den = math.sqrt(sum(float((g ** 2).sum()) for g in g_m if g is not None))
    ks[K] = dict(fd=fd, mine=an, code=cd, rel_code_vs_mine=num / max(den, 1e-300))
    C.check(f"K={K}: my analytic d/dpsi == central FD (float64, real subset, current loss)",
            abs(an - fd) <= 1e-6 * max(abs(fd), 1e-12) + 1e-12, f"analytic={an:.8e} fd={fd:.8e}")
    C.check(f"K={K}: the code's engine gradient == mine (all psi tensors, float64)",
            res.supported and num / max(den, 1e-300) < 1e-8, f"rel={num / max(den, 1e-300):.2e} code_dir={cd:.8e}")
C.info("K sweep", ks)

C.info("live updates checked", len(ev_checks))
for e in ev_checks:
    C.info(f"event {e['event']}", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in e.items()
                                   if k != "step_search"})
C.check("every live policy update equals MY K-step SGD on the documented loss (float32, rel. to the update)",
        ev_checks and all(e["mine_vs_live"] < 1e-3 for e in ev_checks),
        [round(e["mine_vs_live"], 7) for e in ev_checks])
C.check("every live update is replayed bit-identically by the engine (what the meta-gradient differentiates)",
        ev_checks and all(e["replay_bit_identical"] for e in ev_checks))
C.check("goal-term inputs (u, sources incl. credit:<est>) of every support batch equal my recomputation",
        ev_checks and all(e["goal_inputs_equal"] for e in ev_checks))
C.check("model-term inputs (standardised candidate values, beta_q^+) and credited returns equal my recomputation",
        ev_checks and all(e["model_inputs_equal"] for e in ev_checks))
C.check("trust region: each live step reaches KL in [target/1.5, 1.5 target] or sits at a step bound",
        ev_checks and all(e["in_region"] or e["at_bound"] for e in ev_checks),
        [(round(e["kl"], 4), round(e["lr"], 4), e["at_bound"]) for e in ev_checks])
C.check("the goal term saw all three sources and return-like credit (KF-E01) in the support batches",
        any(any(s.startswith("credit:rudder") for s in e["sources"]) for e in ev_checks)
        and any(any(s.startswith("credit:simple") for s in e["sources"]) for e in ev_checks)
        and any("verified" in e["sources"] for e in ev_checks))
C.check("at least one applied meta-update was checked", meta_checked["n"] >= 1, meta_checked["n"])
sys.exit(C.finish())
