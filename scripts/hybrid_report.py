"""Compile the CAM + frozen-LLM hybrid experiment into ``runs/cam_hybrid/report.md``.

    PYTHONPATH=src python scripts/hybrid_report.py [--runs /workspace/agentrs/runs]

Only numbers present in run directories are reported (state.json of the CAM runs, result.json of the
frozen-LLM-direct baselines, probe.json of the representation probe). Validation/test tasks are the same
seeds and environment code as every other CAM experiment; success means verified success at the
unchanged full-difficulty goal (greedy policy, 200 episodes per family unless stated).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

FAMS = ("negotiation", "product", "bizdev")


def load(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def cell(v) -> str:
    return f"{v['success']}/{v['n']} = {v['rate']:.3f}" if v else "–"


def arm(name: str) -> str:
    if name.startswith("hy_A"):
        return "A structured (tokens)"
    if name.startswith("hy_B"):
        return "B + frozen " + ("Qwen3-4B" if "q4b" in name else "Qwen3-0.6B")
    return name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/workspace/agentrs/runs")
    a = ap.parse_args()
    R = Path(a.runs)
    H = R / "cam_hybrid"
    runs = {d.name: s for d in sorted(H.glob("hy_[AB]_*")) if (s := load(d / "state.json"))}
    md = ["# CAM + frozen pretrained LLM: experiment report", "",
          "Arms A (structured CAM, tokens) and B (CAM reading natural language through a frozen Qwen encoder + trainable bridge) "
          "share the learner (PPO inside CAM-v2), rewards, curriculum, interaction budget and evaluation tasks. "
          "`train` = training wording, `heldout` = never-seen wording (same information). Success = verified success on the "
          "unchanged full-difficulty goals, greedy policy.", ""]

    # 1. milestone table
    ms_re = re.compile(r"^(untrained|m(\d+))(-heldout)?$")
    for fam in FAMS:
        rows, cols = [], set()
        per = {}
        for name, st in runs.items():
            for e in st.get("evals", []):
                m = ms_re.match(e["tag"])
                if not m or e["split"] != "val" or fam not in e["by_family"]:
                    continue
                k = 0 if m.group(1) == "untrained" else int(m.group(2))
                cols.add(k)
                per.setdefault((name, "heldout" if m.group(3) else "train"), {})[k] = (e["by_family"][fam], e["env_interactions_so_far"])
        if not per:
            continue
        cols = sorted(cols)
        for (name, style), d in sorted(per.items()):
            rows.append(f"| {name} | {arm(name)} | {style} | " + " | ".join(
                (f"{d[k][0]['rate']:.3f}" + (f" (v{d[k][0]['violation_rate']:.2f})" if d[k][0].get('violation_rate') else "")) if k in d else "–"
                for k in cols) + " |")
        md += [f"## {fam}: validation success at interaction milestones", "",
               "| run | arm | wording | " + " | ".join("untrained" if k == 0 else f"{k // 1000}k" for k in cols) + " |",
               "|---|---|---|" + "---|" * len(cols)] + rows + [""]
    md += ["(vX = fraction of episodes ended by a hard-constraint violation: product bankruptcy, negotiation walk-out.) "
           "Each milestone is evaluated at the first iteration at or past that many real interactions.", ""]

    # 2. sealed tests and final
    md += ["## Sealed test and final validation", ""]
    for name, st in runs.items():
        for e in st.get("evals", []):
            if e["tag"].startswith(("final", "best")) and not e["tag"].startswith(("frontend", "reset")):
                md.append(f"- {name} `{e['tag']}` ({e['split']}, {e.get('style', 'train')} wording): "
                          + ", ".join(f"{k} {cell(v)}" for k, v in e["by_family"].items()))
    md.append("")

    # 3. controls
    md += ["## Controls: CAM reset with the frozen LM kept (arm E), and text front-ends for structured CAM (arm C)", ""]
    for name, st in runs.items():
        for e in st.get("evals", []):
            if e["tag"].startswith(("reset_", "frontend-")):
                md.append(f"- {name} `{e['tag']}` ({e['split']}): " + ", ".join(f"{k} {cell(v)}" for k, v in e["by_family"].items()))
        hf = st.get("hybrid_final")
        if hf:
            pc = hf.get("param_change_vs_initial")
            if pc:
                md.append(f"- {name}: relative parameter change vs initial: " + ", ".join(f"{g} {v:.3f}" for g, v in pc.items()))
            dl = hf.get("decision_logprob_change")
            if dl:
                md.append(f"- {name}: decision log-prob change initial→trained on the same real states: mean |Δ| {dl['mean_abs']:.3f} "
                          f"over {dl['decisions']} decisions (mean log-prob {dl['mean_initial']:.3f} → {dl['mean_trained']:.3f})")
            if "semantic_model_unchanged" in hf:
                md.append(f"- {name}: frozen LM bit-identical at end: {hf['semantic_model_unchanged']} "
                          f"(sha256 {hf['semantic_fingerprint_end'][:16]}…); semantic encoding {hf.get('semantic_compute_s', 0) / 3600:.2f} h")
            for k, v in hf.items():
                if k.startswith("frontend_") and k.endswith("_stats"):
                    md.append(f"- {name}: {k}: {v}")
    md.append("")

    # 4. compute and interactions
    md += ["## Interactions and compute (wall-clock seconds on the shared GPU)", ""]
    for name, st in runs.items():
        c, cs = st.get("counters", {}), st.get("compute_s", {})
        sem = st.get("semantic") or {}
        md.append(f"- {name}: phase {st.get('phase')}, iter {st.get('iter')}, {c.get('transitions', 0):,} real interactions, "
                  f"{c.get('episodes', 0):,} episodes, {c.get('grad_updates', 0):,} update rounds; compute h "
                  + ", ".join(f"{k} {v / 3600:.2f}" for k, v in cs.items() if v > 1)
                  + (f"; frozen encoder {sem.get('model')} layers {sem.get('layers')} dim {sem.get('dim')}" if sem else "")
                  + "; first training successes: " + ", ".join(f"{k} at {v['env_interactions']:,}" for k, v in sorted((st.get("first_success") or {}).items())))
    md.append("")

    # 5. frozen LLM acting directly (arm D)
    md += ["## Arm D: the frozen LLM acting directly (grammar-constrained likelihood choice; no learning)", ""]
    for d in sorted(H.glob("hy_D_*")):
        r = load(d / "result.json")
        if not r:
            md.append(f"- {d.name}: not finished")
            continue
        md.append(f"- {d.name} ({r['model']}, {r['split']}, {r['style']} wording, weights unchanged {r.get('weights_unchanged')}, "
                  f"peak GPU {r.get('peak_gpu_gb') or 0:.1f} GB, {r['wall_s'] / 3600:.2f} h): "
                  + ", ".join(f"{k} {v['success']}/{v['n']} = {v['rate']:.3f} (viol {v['violation_rate']:.2f}, progress {v['final_progress']:.2f})"
                              for k, v in r["families"].items()))
    md.append("")

    # 6. representation probe
    pr = load(H / "probe.json")
    if pr:
        md += ["## Representation probe: can a small probe recover observation fields from frozen embeddings?", "",
               "Probe trained on training-wording texts of training states; tested on unseen states.", ""]
        for m, res in pr.get("models", {}).items():
            for fam, lv in res.get("families", {}).items():
                b = lv.get("mid+last") or next(iter(lv.values()))
                md.append(f"- {m} {fam} (mid+last layers): field accuracy {b['field_acc_unseen_states_train_wording']:.3f} (train wording), "
                          f"{b['field_acc_unseen_states_heldout_wording']:.3f} (held-out wording); majority baseline {b['majority_baseline']:.3f}")
            md.append(f"- {m}: {res.get('texts_per_s', 0):.0f} texts/s, peak GPU {res.get('peak_gpu_gb') or 0:.1f} GB, "
                      f"fingerprint unchanged {res.get('fingerprint_unchanged')}")
        if pr.get("parser_heldout_exact"):
            md.append("- rule-based parser, exact recovery on held-out wording: "
                      + ", ".join(f"{k} {v:.2f}" for k, v in pr["parser_heldout_exact"].items()))
        le = pr.get("llm_extraction_heldout")
        if le:
            md.append(f"- {le['model']} few-shot extraction on held-out wording: "
                      + ", ".join(f"{k} exact {v['exact']:.2f}, token acc {v['token_acc']:.2f}" for k, v in le.items() if isinstance(v, dict)))
        md.append("")

    # 7. reference: earlier CAM systems on the same validation tasks
    md += ["## Reference: earlier CAM systems on the same validation tasks (best greedy validation so far)", ""]
    refs = {"CAM v1 PPO (cam)": R / "cam", "CAM v1 PPO + product fix (cam_pfix)": R / "cam_pfix"}
    for root in ("cam2_v21", "cam2_v22"):
        for d in sorted((R / root).glob("c*")):
            refs[d.name] = d
    for name, d in refs.items():
        st = load(d / "state.json")
        if not st:
            continue
        parts = []
        for fam in ("negotiation", "product"):
            ev = [e for e in st.get("evals", []) if e["split"] == "val" and e.get("greedy", True) and fam in e["by_family"]
                  and not e["tag"].endswith("-monitor") and not e.get("planning")]
            if ev:
                b = max(ev, key=lambda e: e["by_family"][fam]["rate"])
                at200 = [e for e in ev if e["env_interactions_so_far"] <= 200_000]
                parts.append(f"{fam} best {b['by_family'][fam]['rate']:.3f} at {b['env_interactions_so_far']:,}"
                             + (f", ≤200k {at200[-1]['by_family'][fam]['rate']:.3f}" if at200 else ""))
        if parts:
            md.append(f"- {name}: " + "; ".join(parts))
    md.append("")
    out = H / "report.md"
    out.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
