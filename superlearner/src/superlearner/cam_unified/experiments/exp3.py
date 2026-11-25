"""Experiment 3 (spec Part 17, 13.7): parametric transfer and retention, with the corrected negative-transfer diagnosis.

Sequential multi-domain learning: CAM Unified is trained on domain 1 (``scm``: Crucible's and the catalogue's
structural-causal-model training families; the source runs of `train`), then on domain 2 (``abm``: agent-based
markets, a different simulator implementation) for a matched budget of target decisions under five conditions
(spec 13.7 + full lifetime, each in a fresh process via `stack.load_condition`):

    A  fresh model and optimiser (target-only training: spec 19 baseline 13)
    B  source weights only (`load_weights_only`: no optimiser moments, replay, archive, memory index, fast state,
       counters; `reset_experience_access`)
    C  source weights + optimiser state
    D  source weights + optimiser + historical replay (the source archive / memory index copied into the target run)
    F  the full source lifetime (also fast / latent state, RNG, counters)

Evaluation (weights-only, fresh process, identical support - `predict`; log score area over k = 0..64 is the primary
quantity, decision regret from the same queries the secondary) at the target run's checkpoints (0 = zero-shot,
milestones, end) on:

    familiar      new instances of the source families (Experiment 1's familiar set) -> retention
    compositional new combinations of source motifs (held-out pairs / triples)
    structural    unseen causal structures (the inverted-U motif never occurs in training)
    abm           the target domain's families, new instances -> acquisition
    abm_heldout   the target domain's held-out families
    tech          queueing systems never trained in any domain (a third simulator implementation)
    scm_rescaled  the familiar tasks behind a new semantic interface (`worldgen.Surface(scale_units=True)`: gauges
                  presented at other scales with other unit labels, renamed levers, other wording)

Unfamiliar full OpenWorld tasks are not part of this numeric experiment (they belong to the OpenWorld protocols).

Diagnosis (spec 13.7: "do not infer the cause solely from the negative score"): B - A isolates what the source
weights do (representations / priors), C - B the optimiser state, D - C the historical replay (interference), F - D
fast state / counters; over-confident priors show as low 90% coverage and log score at small k on the target; the
plasticity instruments of E (`memory.plasticity`: dormant units, effective rank of the latent state on a probe set,
weight change from the source) are recorded per checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

from . import exp1 as E1
from . import mechrun as M
from . import predict as P
from . import stack as S
from . import stats as ST

ROOT = os.path.join(S.RUNS_ROOT, "exp3")
SETS = {"dev": {"offset": 120_000, "n": 2, "design_seed": 31}, "confirm": {"offset": 220_000, "n": 3, "design_seed": 37}}
ABM = ("abm_congestion", "abm_utility", "abm_wom", "cw1_abm_congest_wom", "cw1_abm_util_congest", "cw1_abm_util_wom",
       "cw1_abm_wom_congest")
ABM_HELDOUT = ("abm_commons", "abm_complement", "abm_trust_gate", "cw1_abm_congest_gate", "cw1_abm_util_gate")
TECH = ("tech_handoff", "tech_pipeline", "cw1_tech_pipeline_hidequeue")
STRUCTURAL = ("scm_invu", "cw1_scm_invu", "cw1_scm_invu_delay")
CATEGORY_SETS = ("familiar", "compositional", "structural", "abm", "abm_heldout", "tech", "scm_rescaled")


def set_dir(name: str) -> str:
    d = os.path.join(ROOT, name)
    os.makedirs(d, exist_ok=True)
    return d


def rescaled_surface(seed: int):
    from ..envs.worldgen import Surface
    return lambda b, ep: Surface(10_000 * int(seed) + 97 * b + ep, scale_units=True)


def cmd_prepare(a) -> None:
    """Support episodes (condition A design) for every category; familiar / compositional / structural reuse
    Experiment 1's tasks and support of the same set name."""
    sp = SETS[a.set]
    d = set_dir(a.set)
    e1 = E1.load_support(a.set, "A")
    cats = {"familiar": [e for e in e1 if E1.split_of(e.task.family) == "familiar"],
            "compositional": [e for e in e1 if E1.split_of(e.task.family) == "heldout"
                              and e.task.family not in STRUCTURAL],
            "structural": [e for e in e1 if e.task.family in STRUCTURAL]}
    info = {}
    for name, fams in (("abm", ABM), ("abm_heldout", ABM_HELDOUT), ("tech", TECH)):
        tasks = M.eval_task_set(fams, sp["n"], sp["offset"], workers=1)
        info[name] = {"tasks": [t.id for t in tasks], "failures": [str(f)[:160] for f in M.eval_task_set.failures]}
        cats[name] = []
        for i in range(0, len(tasks), 16):
            cats[name] += P.collect_support(tasks[i:i + 16], k_max=64, n_queries=12, design_seed=sp["design_seed"])
    fam_tasks = [e.task for e in cats["familiar"]]
    cats["scm_rescaled"] = []
    for i in range(0, len(fam_tasks), 16):
        cats["scm_rescaled"] += P.collect_support(fam_tasks[i:i + 16], k_max=64, n_queries=12,
                                                  design_seed=sp["design_seed"], surface=rescaled_surface(i + 1))
    units = sorted({o.unit for ep in cats["scm_rescaled"] for s in ep.steps[:1] for o in s["next_obs"]
                    if o.unit and "reading" in o.unit})
    info["scm_rescaled_units"] = units
    for name, eps in cats.items():
        with open(os.path.join(d, f"support_{name}.pkl"), "wb") as fh:
            pickle.dump(eps, fh)
        info.setdefault(name, {})["n"] = len(eps)
    with open(os.path.join(d, "prepare.json"), "w") as fh:
        json.dump(info, fh, indent=1)
    print(json.dumps(info)[:2000])


def load_cat(set_name: str, cat: str) -> list:
    with open(os.path.join(set_dir(set_name), f"support_{cat}.pkl"), "rb") as fh:
        return pickle.load(fh)


# ============================================================================ plasticity instruments (E)

@torch.no_grad()
def plasticity_report(st, probe_eps: list, ref_state: dict = None) -> dict:
    """E's PlasticityMonitor on the integrated model over a probe pass (the filter over identical support episodes):
    dormant units per unit layer (tau = 0.025), activation effective rank, effective rank / srank of the latent state
    Z over the probe steps, and the weight change from a reference state dict (e.g. the source checkpoint)."""
    from ..memory.plasticity import PlasticityMonitor, effective_rank, srank
    model = st.model
    mon = PlasticityMonitor(model)
    for ep in probe_eps:
        model.tokenizer.register_tools(ep.tools)
    with mon:
        recs = [P.cam_records(model, ep, upto=32)[0] for ep in probe_eps]
        seq, aux = st.agent.tokenise_sequences(recs)
        filt = model.filter_sequence(seq, aux)
    Z = torch.cat([z.reshape(-1, z.shape[-1]) for z in filt.Zs], 0).float()
    dormant = mon.dormant(0.025)
    ranks = mon.activation_rank()
    out = {"dormant_frac": float(dormant["fraction"]),
           "dormant_by_layer": {k: float(v) for k, v in dormant["by_layer"].items()},
           "activation_rank_mean": float(np.mean(list(ranks.values()))) if ranks else None,
           "z_effective_rank": float(effective_rank(Z)), "z_srank": int(srank(Z))}
    if ref_state is not None:
        num = den = 0.0
        for n, p in model.state_dict().items():
            if n in ref_state and p.dtype.is_floating_point and p.shape == ref_state[n].shape:
                num += float((p.float().cpu() - ref_state[n].float()).pow(2).sum())
                den += float(ref_state[n].float().pow(2).sum())
        out["weight_change_rel"] = (num / max(den, 1e-12)) ** 0.5
    return out


# ============================================================================ evaluation of checkpoints

def cmd_predict(a) -> None:
    """Prediction rows for every category set at the given checkpoints of one run (weights-only, fresh process)."""
    d = set_dir(a.set)
    cats = a.cats.split(",") if a.cats else list(CATEGORY_SETS)
    rows, prow = [], []
    sem = None
    ref = None
    if a.source:
        from safetensors.torch import load_file
        ref = load_file(os.path.join(a.source, "weights.safetensors"))
    for ck in a.ckpts.split(","):
        ck = ck.strip()
        learner = "cam_untrained" if ck == "fresh" else "cam"
        st, info = E1.cam_stack(learner, None if ck == "fresh" else ck, a.seed, sem=sem)
        sem = st.sem
        tdec = 0 if ck == "fresh" else int(os.path.basename(ck.rstrip("/")))
        meta = {"run": a.run_tag, "condition": a.condition, "seed": a.seed, "ckpt": ck, "target_decisions": tdec,
                "tag": f"{a.condition}@{tdec}"}
        for cat in cats:
            eps = load_cat(a.set, cat)
            t0 = time.time()
            preds = P.cam_predict(st, eps)
            for i, ep in enumerate(eps):
                for k in P.K_SET:
                    m, v = preds[(i, k)]
                    rows.append({**meta, "cat": cat, "task": ep.task.id, "family": ep.task.family, "k": k,
                                 **P.score_queries(ep, k, m, v, None)})
            print(f"[exp3] {a.run_tag} {ck} {cat}: {time.time() - t0:.0f}s", flush=True)
        if a.plasticity:
            prow.append({**meta, **plasticity_report(st, load_cat(a.set, "abm")[:8], ref)})
        del st
        torch.cuda.empty_cache()
    E1._rows_out(os.path.join(d, "predict.jsonl"), rows)
    if prow:
        E1._rows_out(os.path.join(d, "plasticity.jsonl"), prow)


def cmd_predict_baselines(a) -> None:
    """The predictive baselines on the Experiment 3 category sets (same support, same queries)."""
    d = set_dir(a.set)
    rows = []
    for name in ("nig_exact", "nig_meta", "seq_gru"):
        for seed in ((0,) if name == "nig_exact" else (0, 1, 2)):
            pr = E1.load_predictor(name, seed)
            for cat in CATEGORY_SETS:
                for ep in load_cat(a.set, cat):
                    for k in P.K_SET:
                        rows.append({"run": name, "condition": name, "seed": seed, "target_decisions": 0,
                                     "tag": name, "cat": cat, "task": ep.task.id, "family": ep.task.family, "k": k,
                                     **P.score_queries(ep, k, *pr.predict(ep, k))})
    E1._rows_out(os.path.join(d, "predict.jsonl"), rows)
    print(f"[exp3] baselines: {len(rows)} rows")


def cmd_report(a) -> None:
    d = set_dir(a.set)
    rows = [json.loads(l) for l in open(os.path.join(d, "predict.jsonl"))]
    rep = {"set": a.set, "cells": {}, "contrasts": {}}
    by = {}
    for r in rows:
        by.setdefault((r["tag"], r["cat"]), []).append(r)
    mats = {}
    for (tag, cat), rr in sorted(by.items()):
        tasks = sorted({r["task"] for r in rr})
        alltasks = sorted({r["task"] for r in rows if r["cat"] == cat})
        out = {}
        for metric in ("logscore", "regret", "cover90"):
            per_k = {k: E1._matrix([r for r in rr if r["k"] == k], metric, alltasks) for k in P.K_SET}
            mats[(tag, cat, metric)] = per_k
            out[metric] = {"area": ST.two_level(ST.area(per_k)),
                           "k0": ST.two_level(per_k[0]), "k4": ST.two_level(per_k[4]), "k64": ST.two_level(per_k[64])}
        rep["cells"][f"{tag}/{cat}"] = out
    tags = sorted({t for t, _, _ in mats})

    def contrast(t1, t2, cat, metric="logscore"):
        if (t1, cat, metric) in mats and (t2, cat, metric) in mats:
            return ST.paired(ST.area(mats[(t1, cat, metric)]), ST.area(mats[(t2, cat, metric)]))
        return None
    ends = {}
    for t in tags:
        c, _, n = t.partition("@")
        if n.isdigit():
            ends.setdefault(c, []).append(int(n))
    final = {c: f"{c}@{max(v)}" for c, v in ends.items()}
    zero = {c: f"{c}@0" for c in ends}
    for cat in CATEGORY_SETS:
        blk = {}
        for x, y in (("B", "A"), ("C", "B"), ("D", "C"), ("F", "D"), ("F", "A"), ("D", "A")):
            if x in final and y in final:
                blk[f"final:{x}-{y}"] = {"logscore": contrast(final[x], final[y], cat),
                                          "regret": contrast(final[x], final[y], cat, "regret")}
        if "B" in final and "B@0" in tags and "A@0" in tags:
            blk["zero_shot:B-A"] = {"logscore": contrast("B@0", "A@0", cat), "regret": contrast("B@0", "A@0", cat,
                                                                                               "regret")}
        for c in final:
            if zero[c] in tags:
                blk[f"{c}:final-zero"] = {"logscore": contrast(final[c], zero[c], cat)}
        rep["contrasts"][cat] = blk
    p = os.path.join(d, "plasticity.jsonl")
    if os.path.exists(p):
        rep["plasticity"] = [{k: v for k, v in json.loads(l).items() if k != "dormant_by_layer"} for l in open(p)]
    out = os.path.join(d, "report.json")
    with open(out, "w") as fh:
        json.dump(rep, fh, indent=1, default=float)
    print(f"[exp3] report -> {out}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--set", default="dev")
    p = sub.add_parser("predict"); p.add_argument("--set", default="dev"); p.add_argument("--ckpts", required=True)
    p.add_argument("--condition", required=True); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-tag", default=""); p.add_argument("--cats", default="")
    p.add_argument("--plasticity", action="store_true"); p.add_argument("--source", default=None)
    p = sub.add_parser("baselines"); p.add_argument("--set", default="dev")
    p = sub.add_parser("report"); p.add_argument("--set", default="dev")
    a = ap.parse_args(argv)
    {"prepare": cmd_prepare, "predict": cmd_predict, "baselines": cmd_predict_baselines, "report": cmd_report}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
