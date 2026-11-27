"""End-to-end MetaLearner path on the real debug model in LabWorld (spec 7.4 - 7.6; MR-02, MR-03, MR-05, MR-06,
MR-08, MR-10).

1. 'learned' arm (mix 0): lifetimes with real policy updates, signed objective outcomes, meta_update(). The psi step
   that MetaLearner applies is captured and compared with MY recomputation from the episodes (own functional unroll
   of the documented inner loss from each event's theta snapshot, own outer surrogate with group-leave-out
   baseline and 1/N row weights). Reward detached -> the applied gradient is zero.
2. Cold start (mix 1): the meta-update is a shadow update and must use the importance estimator; the behaviour
   log-propensities it uses must be log(decision.probabilities[chosen]) of the stored decisions; removing the
   probabilities of the query decisions must drop them (no silent on-policy / uniform fallback); records labelled.
3. The return-to-go baseline (utility="return_to_go", verifier measurements as utilities): the leave-group-out
   baseline of a row contains that row's OWN measured outcome through the return-to-go of earlier rows of the same
   lifetime (a baseline that depends on the row's own action -> biased outer gradient).
"""
import copy
import math
import sys

import torch
from torch.func import functional_call, grad

import _common as C
from cam_live.contracts import Trust
from cam_live.diagnostics.collect import LabCollector, verified_return
from cam_live.diagnostics.world import LabConfig, LabWorld, lab_task
from cam_live.reward import meta as M
from cam_live.reward.diagnose import ExperimentConfig, LabMeasurementAuth, lab_lifetime, make_stack, model_policy
from cam_live.reward.integrity import params_hash

torch.manual_seed(0)


def run_lifetimes(ml, store, clock, ec, n, seed, act, kind="learned", on_meta=None):
    out = []
    for L in range(n):
        world = LabWorld(LabConfig(seed=900_000 + 1000 * seed + L, **ec.lab), clock=clock)
        ml.new_lifetime()
        buf = []

        def on_step(rec):
            ml.internal_rewards([rec])
            buf.append(rec)
            if len(buf) >= ec.update_every:
                ml.policy_update(list(buf))
                buf.clear()
        lt = LabCollector(store, max_context=ec.max_context, seq_start=L * 1000 + seed * 100_000).run(
            world, act, ec.n_decisions, seed=seed * 7 + L, on_step=on_step,
            task=lab_task(world, objective_id=f"obj_probe_{seed}_{L}"))
        if len(buf) >= 2:
            ml.policy_update(list(buf))
        r = ml.admit_objective(lt.verified, utility=verified_return(lt))
        assert r.accepted, r.reason
        if on_meta is not None:
            on_meta(ml, L)
        out.append(lt)
    return out


def forward(ml, th, lat):
    tr = {k[6:]: v for k, v in th.items() if k.startswith("trunk.")}
    pol = {k[7:]: v for k, v in th.items() if k.startswith("policy.")}
    z = functional_call(ml.subset.trunk, tr, (lat.b, lat.e_a, lat.e_g))
    out = functional_call(ml.subset.policy, pol, (z, lat.a_mask, lat.b))
    lg = torch.where(lat.a_mask, out.raw_logits, torch.full_like(out.raw_logits, -1e30))
    return torch.log_softmax(lg, -1), out.value


def my_loss(ml, th, th_sg, psi, lat, mix):
    cfg = ml.inner.cfg
    lpa, v = forward(ml, th, lat)
    _, vb = forward(ml, th_sg, lat)
    r = mix * lat.r_boot + (1.0 - mix) * ml.inner.learned_reward(psi, lat)
    G, acc = [], torch.zeros(())
    for i in reversed(range(lat.n)):
        acc = r[i] + lat.gamma[i] * acc
        G.append(acc)
    G = torch.stack(G[::-1])
    dG = G - G.mean()
    Gn = dG / torch.sqrt((dG ** 2).mean() + 1e-12)
    lp = lpa[torch.arange(lat.n), lat.chosen]
    p = lpa.exp() * lat.a_mask
    ent = -(p * torch.where(lat.a_mask, lpa, torch.zeros_like(lpa))).sum(-1)
    return -(lp * (Gn - vb)).mean() + cfg.value_coef * (0.5 * (v - Gn) ** 2).mean() - cfg.entropy_coef * ent.mean()


def my_unroll(ml, psi, th, steps, lr):
    for lat in steps:
        g = grad(lambda t, s: my_loss(ml, t, s, psi, lat, lat.mix), argnums=0)(th, th)
        for pre, mx in (("policy.", ml.mc.clip_policy), ("trunk.", ml.mc.clip_trunk)):
            ks = [k for k in g if k.startswith(pre)]
            nrm = torch.sqrt(sum((g[k] ** 2).sum() for k in ks) + 1e-30)
            s = torch.clamp(mx / nrm, max=1.0)
            for k in ks:
                g[k] = g[k] * s
        th = {n: th[n] - lr * g[n] for n in th}
    return th


def my_meta_gradient(ml, eps):
    """Independent recomputation of the step meta_update applies (estimator per episode as documented)."""
    rows = [(k, a) for k, e in enumerate(eps) if e.ibq is not None for a in range(len(e.exp_ids))
            if e.has_u[a] and not e.held[a]]
    N = len(rows)
    U_all = [eps[k].utility[a] for k, a in rows]
    G_all = [eps[k].group[a] for k, a in rows]
    psi = {k: v.detach().clone() for k, v in ml.reward.psi().items()}
    fallback = ml._u_ema if ml._u_ema is not None else 0.0

    def base(g):
        rest = [u for u, gg in zip(U_all, G_all) if gg != g]
        return sum(rest) / len(rest) if rest else fallback

    def surrogate(p):
        tot, used = 0.0, []
        for k, e in enumerate(eps):
            if e.ibq is None:
                continue
            idx = [a for a in range(len(e.exp_ids)) if e.has_u[a] and not e.held[a]]
            if not idx:
                continue
            steps, _ = ml._steps_of(e.events)
            theta0 = {n: v for n, v in e.events[0].theta0.items()}
            thK = my_unroll(ml, p, theta0, steps, e.events[-1].inner_lr)
            ibq = e.ibq.select(idx)
            lpa, _ = forward(ml, thK, ibq)
            lp = lpa[torch.arange(len(idx)), ibq.chosen]
            blp = ibq.behaviour_logp.to(lp.dtype)
            on = bool(torch.isfinite(blp).all()) and float((lp.detach().exp().double() - blp.exp().double()).abs().max()) \
                <= ml.mc.on_policy_tol
            w = torch.full((len(idx),), 1.0 / N)
            if not on:
                w = w * torch.exp(lp.detach() - blp).clamp(max=ml.mc.weight_clip)
            Uq = torch.tensor([e.utility[a] for a in idx])
            bq = torch.tensor([base(e.group[a]) for a in idx])
            used.append("on" if on else "iw")
            tot = tot + (w * lp * (Uq - bq)).sum()
        return tot, used
    g = grad(lambda p: surrogate(p)[0])(psi)
    return g, surrogate(psi)[1]


# ------------------------------------------------------------------------------------------- 1. learned arm (mix 0)
ec = ExperimentConfig(min_independent=3, n_decisions=16)
model, store, clock, ml = make_stack(3, ec, mix=0.0)
act = model_policy(ml, store, torch.Generator().manual_seed(9))
captured = {}
orig_apply = M.MetaGradient.apply


def capture_apply(optimizer, reward_params, result, clip=None):
    captured["result"] = result
    return orig_apply(optimizer, reward_params, result, clip=clip)


M.MetaGradient.apply = staticmethod(capture_apply)
checked = {"done": False}


def on_meta(ml_, L):
    if checked["done"]:
        rec = ml_.meta_update()
        return
    ok, why = ml_.ready()
    if not ok:
        rec = ml_.meta_update()
        C.info(f"lifetime {L}: skipped", rec.skipped_reason)
        C.check(f"lifetime {L}: skipped meta-update leaves psi bit-identical and is recorded", not rec.applied
                and rec.skipped_reason != "")
        return
    eps, _, _ = ml_._collect()
    eps_copy = copy.copy(eps)
    mine, used = my_meta_gradient(ml_, eps_copy)
    h = params_hash(ml_.reward)
    captured.clear()
    rec = ml_.meta_update()
    res = captured.get("result")
    C.check(f"lifetime {L}: meta_update applied and psi changed", rec.applied and params_hash(ml_.reward) != h,
            f"estimator={rec.estimator} n_query={rec.n_query} n_indep={rec.n_independent_outcomes}")
    num = math.sqrt(sum(float(((res.grads[n] - mine[n]) ** 2).sum()) for n in mine))
    den = math.sqrt(sum(float((mine[n] ** 2).sum()) for n in mine))
    C.check("applied meta-gradient == my independent recomputation from the episodes", num / den < 1e-4,
            f"rel_err={num / den:.2e} |g|={den:.3e} my estimators={used}")
    checked["done"] = True


lts = run_lifetimes(ml, store, clock, ec, 6, 3, act, on_meta=on_meta)
C.check("an applied meta-update was reached and checked", checked["done"])

# detached reward on the same real path -> applied gradient zero
model2, store2, clock2, ml2 = make_stack(4, ec, mix=0.0)
orig_lr = ml2.inner.learned_reward
ml2.inner.learned_reward = lambda psi, ib: orig_lr({k: v.detach() for k, v in psi.items()}, ib)
act2 = model_policy(ml2, store2, torch.Generator().manual_seed(10))
norms = []


def on_meta2(ml_, L):
    captured.clear()
    rec = ml_.meta_update()
    if rec.applied:
        norms.append(rec.meta_grad_norm)


run_lifetimes(ml2, store2, clock2, ec, 4, 4, act2, on_meta=on_meta2)
C.check("real path with R_psi detached: every applied meta-gradient is exactly 0", norms and all(n == 0.0 for n in norms),
        f"norms={norms}")

# ------------------------------------------------------------------------------------- 2. cold start: importance path
model3, store3, clock3, ml3 = make_stack(5, ec, mix=1.0)
act3 = model_policy(ml3, store3, torch.Generator().manual_seed(11))
seen = {}


def on_meta3(ml_, L):
    ok, _ = ml_.ready()
    if ok and "est" not in seen:
        eps, _, _ = ml_._collect()
        bad = 0
        for e in eps:
            if e.ibq is None:
                continue
            for i, x in enumerate(e.ibq.exp_ids):
                r = store3.get(x)
                want = math.log(r.decision.probabilities[r.decision.chosen_index])
                bad += abs(float(e.ibq.behaviour_logp[i]) - want) > 1e-6
        seen["bad_blp"] = bad
        rec = ml_.meta_update()
        seen["est"] = rec.estimator
        seen["applied"] = rec.applied
        seen["labels"] = {store3.get(x).reward_version[:5] for e in eps for x in e.exp_ids}
    else:
        ml_.meta_update()


lts3 = run_lifetimes(ml3, store3, clock3, ec, 6, 5, act3, on_meta=on_meta3)
C.info("cold start (mix 1): estimator of the shadow meta-update (API.md: 'must use importance'; the code switches to "
       "on-policy when |pi_K - mu| <= on_policy_tol)", seen.get("est"))
C.check("cold start (mix 1): shadow meta-update applied with importance weighting on at least one episode",
        seen.get("applied") and seen.get("est") in (M.ESTIMATOR_IMPORTANCE, "mixed_on_policy_and_importance"), f"{seen}")
C.check("behaviour log-propensities == log(decision.probabilities[chosen]) of the stored decisions (float32 tol 1e-6)",
        seen.get("bad_blp") == 0, f"mismatches={seen.get('bad_blp')}")
rec_last = lts3[-1].records[-1]
C.check("records label the reward mix (bootstrap_reward and intrinsic_reward stored separately)",
        rec_last.bootstrap_reward is not None and rec_last.intrinsic_reward is not None and rec_last.reward_version != "")
C.check("MetaUpdateRecord.bootstrap_mix reports the bootstrap mix (not labelled as learned)",
        all(r.bootstrap_mix == 1.0 for r in ml3.records), f"{sorted({r.bootstrap_mix for r in ml3.records})}")
# remove the probabilities of new query decisions: they must be dropped (not used without support)
model4, store4, clock4, ml4 = make_stack(6, ec, mix=1.0)
act4 = model_policy(ml4, store4, torch.Generator().manual_seed(12))


def strip(ml_, L):
    for e in ml_.events:
        for x in e.query_ids:
            r = store4.get(x)
            if r is not None and r.decision.probabilities:
                r.decision.probabilities = []
                store4.update(r)
    rec = ml_.meta_update()
    seen.setdefault("strip", []).append((rec.applied, rec.skipped_reason, rec.n_query))


run_lifetimes(ml4, store4, clock4, ec, 4, 6, act4, on_meta=strip)
C.check("query decisions without logged propensities never supervise a meta-update",
        all(not a for a, _, _ in seen["strip"]), f"{seen['strip']}")

# ------------------------------------------------------------------- 3. return-to-go baseline contains own outcome
ec5 = ExperimentConfig(min_independent=2)
auth = LabMeasurementAuth()
model5, store5, clock5, ml5 = make_stack(7, ec5, mix=0.0, check_measurement=auth)
ml5.new_lifetime()
lt5, _, _ = lab_lifetime(ml5, store5, clock5, ec5, seed=4242, auth=auth, n_decisions=24, update_every=8)
clock5.advance(5 * 86400.0)
eps5, used5, held5 = ml5._collect()
found = None
for e in eps5:
    if e.ibq is None:
        continue
    mea = [[x for x in ml5.ledger.by_exp(i) if x.kind == "measurement"] for i in e.exp_ids]
    n = len(e.exp_ids)
    for a in range(1, n):
        if mea[a] and any(e.group[j] != e.group[a] and e.has_u[j] for j in range(a)):
            found = (e, mea, a)
            break
    if found:
        break
if found is None:
    C.info("return-to-go check", "no episode with measured rows (nothing to check)")
else:
    e, mea, a = found
    beta = ml5.inner.cfg.beta
    ts = [store5.get(i).decision.ts for i in e.exp_ids]

    def rtg(mvals):
        return [sum(math.exp(-beta * max(0.0, ts[j] - ts[i])) * mvals[j] for j in range(i, len(mvals)))
                for i in range(len(mvals))]
    m = [sum(x.utility for x in row) for row in mea]
    same = max(abs(u - v) for u, v in zip(rtg(m), e.utility))
    C.check("re-derived return-to-go utilities match MetaLearner's episode utilities", same < 1e-9, f"{same:.1e}")

    def baseline_of(Us, row):
        rest = [u for u, g, hu in zip(Us, e.group, e.has_u) if g != e.group[row] and hu]
        return sum(rest) / len(rest)
    m2 = list(m)
    m2[a] += 1.0
    db = baseline_of(rtg(m2), a) - baseline_of(rtg(m), a)
    C.check("the leave-group-out baseline of a row does not depend on that row's own measured outcome",
            abs(db) < 1e-12, f"d baseline / d own outcome = {db:.3f} (row {a} of {len(m)}; d U_own = 1)")
sys.exit(C.finish())
