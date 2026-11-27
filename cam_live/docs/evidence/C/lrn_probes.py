"""Probes of the four learning-mechanism defects of change request S-5 (workstream LRN), written against APIs that
exist before and after the fix, so the same script measures the old and the new code:

    PYTHONPATH=<snapshot>/cam_live/src .venv/bin/python cam_live/docs/evidence/C/lrn_probes.py --out OUT.json

1. delta_v_offset       the bootstrap's delta V on decisions after which nothing changed (b' = b): its spread across
                        decisions whose candidate value Q(b, a) differs (old: V(b') - Q(b, a), action-dependent)
2. foreign_verified     verified progress credited to a decision for a verifier reading of *another* decision that
                        arrived during its step (old: counted; e.g. a wait paid for earlier interventions)
3. reward_scale         relative change of one inner update when R_psi's output is multiplied by 10, at mix 0.5
                        (old: the mixed reward is not normalised per component, so scale buys weight)
   reward_shape         change of the update when R_psi's *shape* is perturbed, at mix 1.0 / 0.5 / 0.0
4. live_step            mean KL between the policy before and after one live policy update with the default
                        configuration (old: fixed step 1e-2: ~1e-5, behaviour unchanged)
5. selection            share of probability the default-built DecisionMaker's rule ``policy`` and (when it exists)
                        ``thompson_policy`` put on the candidate with the highest predicted goal value, on the
                        debug model after two LabWorld lifetimes
Each entry says whether the property the fix asserts HOLDS on the code that ran.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import torch
    import cam_live
    from cam_live.contracts import ExperienceItem, ItemKind, OutcomeQuery, Quantity, Trust
    from cam_live.predict.evaluator import EvaluatorConfig, GoalEvaluator
    from cam_live.reward import diagnose as dg
    from cam_live.reward.assemble import STORED_KEY, SignalAssembler, record_signals
    from cam_live.reward.meta import flat
    torch.manual_seed(0)
    torch.set_num_threads(2)
    out = {"code": cam_live.__file__}

    ec = dg.ExperimentConfig()
    model, store, clock, ml = dg.make_stack(0, ec, mix=1.0)
    lt, world, _ = dg.lab_lifetime(ml, store, clock, ec, seed=4242, update_every=0)
    recs = lt.records[:10]

    def annotate(rs):
        res = []
        for k, r in enumerate(rs):
            d = r.decision
            cands = [dataclasses.replace(c, p_success=0.05 if (k + j) % 2 else 0.95,
                                         extras={**(c.extras or {}), "v_state": 0.5,
                                                 "q_goal": 5.0 * float(c.action.arguments.get("effort", 0))
                                                 if c.action.operation == "lab.act" else 0.0})
                     for j, c in enumerate(d.candidates)]
            y = 5.0 * float(d.chosen.arguments.get("effort", 0)) if d.chosen.operation == "lab.act" else 0.0
            r.decision = dataclasses.replace(d, candidates=cands,
                                             diagnostics={**(d.diagnostics or {}),
                                                          STORED_KEY: {"verified_progress_delta": y}})
            store.update(r)
            res.append(r)
        return res
    recs = annotate(recs)

    # 1. delta V offset
    ev = GoalEvaluator(EvaluatorConfig(d_model=model.d_model, hidden=16))
    asm = SignalAssembler(evaluator=ev)
    b = torch.randn(len(recs), model.d_model)
    e_g = torch.randn(len(recs), model.d_model)
    s = asm.assemble(recs, store, b=b, b_next=b.clone(), e_g=e_g, has_next=torch.ones(len(recs), dtype=torch.bool))
    dv = s.signals.delta_v
    spread = float(dv.abs().max()) if dv is not None else None
    out["delta_v_offset"] = {"max_abs_delta_v_when_nothing_changed": spread,
                             "holds_no_action_dependent_offset": spread is not None and spread < 1e-2}

    # 2. foreign verified reading
    rec = next(r for r in lt.records if r.post_item_refs)
    other = ExperienceItem(kind=ItemKind.OUTCOME, text="another decision's outcome", ts=rec.decision.ts + 1.0,
                           quantities=[Quantity("lab.adopters.from.someone_else", 9.0)], ref="q_not_mine:out",
                           trust=Trust.VERIFIER)
    store.put_item(other)
    r2 = dataclasses.replace(rec, post_item_refs=list(rec.post_item_refs) + [other.ref],
                             decision=dataclasses.replace(rec.decision, diagnostics={}))
    v = record_signals(r2, store)[0].get("verified_progress_delta")
    out["foreign_verified"] = {"credited": v, "holds_not_credited": v == 0.0}

    # 3. scale / shape of R_psi in the inner update
    ib0 = ml.inner.build_batch(store, recs)
    psi = {k: t.detach().clone() for k, t in ml.reward.psi().items()}

    def upd(ib, p):
        th0 = ml.subset.snapshot()
        th = ml.engine.inner_update(th0, p, [ib], 0.1, 1, create_graph=False).detached()
        return flat({n: th[n] - th0[n] for n in th})
    bound = ml.reward.bound
    ib = ib0.with_mix(0.5)
    d1 = upd(ib, psi)
    ml.reward.bound = bound * 10
    d10 = upd(ib, {k: (t * 10 if k.startswith("out.") else t) for k, t in psi.items()})
    ml.reward.bound = bound
    rel = float((d10 - d1).norm() / d1.norm())
    out["reward_scale"] = {"relative_update_change_at_mix_0.5_for_R_psi_x10": rel, "holds_scale_invariant": rel < 1e-4}
    g = torch.Generator().manual_seed(0)
    reshaped = {k: (t + 0.5 * t.std(unbiased=False) * torch.randn(t.shape, generator=g, dtype=t.dtype)
                    if k.startswith("out.") else t) for k, t in psi.items()}
    shape = {}
    for mix in (1.0, 0.5, 0.0):
        ibm = ib0.with_mix(mix)
        shape[str(mix)] = float((upd(ibm, reshaped) - upd(ibm, psi)).norm())
    out["reward_shape"] = {"update_change_by_R_psi_shape": shape,
                           "holds_shape_matters_below_mix_1_and_not_at_1": shape["1.0"] == 0.0 and shape["0.5"] > 0}

    # 4. live policy step with the default configuration
    from cam_live.config import CamLiveConfig
    from cam_live.reward.metalearner import MetaConfig, MetaLearner
    cfg = CamLiveConfig.debug(seed=0)
    mc = MetaConfig.from_cfg(cfg, provenance="diagnostic")
    ml2 = MetaLearner(model, store, cfg, check_signature=lambda v: False, clock=clock, meta_cfg=mc,
                      policy=ml.policy, reward=ml.reward)
    for r in recs:
        ml2.observe([r])
    up = ml2.policy_update(recs)
    kl = float(up.prediction_change["policy_kl"])
    out["live_step"] = {"default_inner_lr": mc.inner_lr, "step_used": ml2.events[-1].inner_lr, "policy_kl": kl,
                        "policy_prob_max_abs_change": float(up.prediction_change["policy_prob_max_abs_change"]),
                        "holds_policy_moves": kl > 1e-3}

    # 5. selection on the debug model's own predictions
    from cam_live.predict import policy as P
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tests", "predict"))
    try:
        from conftest import LabRig
        import tempfile
        import pathlib
        rig = LabRig(pathlib.Path(tempfile.mkdtemp(prefix="lrn_probe_")), n_lifetimes=2, n_decisions=10)
        dm = P.DecisionMaker(rig.model, heads=rig.heads, clock=rig.clock, seed=2)
        sel = {}
        for k in range(5):
            evs = rig.evaluate()
            sc = dm.scores(evs)
            q = sc["q"][0]
            best = int(torch.where(sc["a_mask"][0], q, torch.full_like(q, -1e18)).argmax())
            for rule in [r for r in ("policy", "thompson_policy") if r in P.RULES]:
                p, _ = dm.probabilities(evs, rule)
                rel_v = float((p * (q - q.min())).sum() / (q.max() - q.min()).clamp_min(1e-12))
                sel.setdefault(rule, []).append({"p_predicted_best": float(p[best]), "relative_predicted_value": rel_v})
        out["selection"] = {r: {"mean_p_predicted_best": sum(x["p_predicted_best"] for x in v) / len(v),
                                "mean_relative_predicted_value": sum(x["relative_predicted_value"] for x in v) / len(v),
                                "n": len(v)} for r, v in sel.items()}
        out["selection"]["holds_rule_tracks_predictions"] = "thompson_policy" in sel and \
            out["selection"]["thompson_policy"]["mean_relative_predicted_value"] > 0.6
    except Exception as e:  # noqa: BLE001
        out["selection"] = {"error": f"{type(e).__name__}: {e}"}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
