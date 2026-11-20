"""Aggregate CAM 2.0 experiment artifacts into tables (markdown on stdout, JSON) and figures.

  PYTHONPATH=src python scripts/cam2/report.py [--runs runs/cam2] [--fig docs/figures]

Reads only recorded artifacts (log.jsonl, results.json, eval JSONs, population.json); computes nothing new except
summaries (means, bootstrap CIs, best-so-far curves).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_log(run: Path) -> tuple[list, list]:
    ups, evs = [], []
    p = run / "log.jsonl"
    if not p.exists():
        return ups, evs
    for ln in p.read_text().splitlines():
        r = json.loads(ln)
        (ups if r["type"] == "update" else evs).append(r)
    return ups, evs


def fmt_ci(r: dict) -> str:
    if not r:
        return "—"
    lo, hi = r.get("ci95", [float("nan")] * 2)
    return f"{r['success']:.3f} [{lo:.3f}, {hi:.3f}] (n={r.get('n', '?')})"


def split_families(per_family: dict) -> tuple[float, float]:
    from agentrs.cam2.envs.miniwob import TRAIN_FAMILIES, VAL_FAMILIES

    tr = [v for k, v in per_family.items() if k in TRAIN_FAMILIES]
    va = [v for k, v in per_family.items() if k in VAL_FAMILIES]
    return (float(np.mean(tr)) if tr else float("nan"), float(np.mean(va)) if va else float("nan"))


def efficiency(ups: list) -> dict:
    """Learning-curve area (mean of the 400-episode train-success curve over environment steps, trapezoid), reward per
    environment step and successes per wall-clock hour (training episodes; reward = 1 per MiniWoB success)."""
    if not ups:
        return {}
    x = np.array([0.0] + [u["env_steps"] for u in ups])
    y = np.array([ups[0]["train_success_recent"]] + [u["train_success_recent"] for u in ups])
    area = float(np.trapezoid(y, x) / max(1.0, x[-1])) if hasattr(np, "trapezoid") else float(np.trapz(y, x) / max(1.0, x[-1]))
    succ = sum(u["train_success"] * (u["episodes"] - (ups[i - 1]["episodes"] if i else 0)) for i, u in enumerate(ups))
    return {"curve_area": area, "successes": succ, "reward_per_step": succ / max(1, ups[-1]["env_steps"]),
            "successes_per_hour": succ / max(1e-9, ups[-1]["wall_s"] / 3600)}


def run_table(runs: dict) -> list:
    rows = ["| run | env steps | updates | initial val | final val | val: train-family instances | val: held-out families | train-family fit | invalid rate | wall (min) | peak GPU (GB) |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    eff_rows = ["", "| run | learning-curve area | reward / env step | successes / wall-clock hour | LLM gen tokens | LLM encoded tokens |",
                "|---|---|---|---|---|---|"]
    for name, d in runs.items():
        res = json.loads((d / "results.json").read_text()) if (d / "results.json").exists() else {}
        ups, _ = load_log(d)
        e = efficiency(ups)
        if e:
            eff_rows.append(f"| {name} | {e['curve_area']:.3f} | {e['reward_per_step']:.4f} | {e['successes_per_hour']:.0f} | "
                            f"{ups[-1]['llm_gen_tokens']:,} | {ups[-1]['llm_encode_tokens']:,} |")
        fv = res.get("final_val", {})
        trf, vaf = split_families(fv.get("per_family", {})) if fv else (float("nan"), float("nan"))
        last = ups[-1] if ups else {}
        rows.append(f"| {name} | {last.get('env_steps', '—')} | {last.get('updates', '—')} | {fmt_ci(res.get('initial_val'))} | "
                    f"{fmt_ci(fv)} | {trf:.3f} | {vaf:.3f} | {fmt_ci(res.get('final_train'))} | "
                    f"{fv.get('invalid_rate', float('nan')):.3f} | {res.get('train_wall_s', 0) / 60:.1f} | "
                    f"{res.get('peak_gpu_gb', float('nan')):.1f} |")
    return rows + eff_rows


def curves_figure(runs: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for name, d in runs.items():
        ups, evs = load_log(d)
        if not ups:
            continue
        x = [u["env_steps"] for u in ups]
        ax[0].plot(x, [u["train_success_recent"] for u in ups], label=name)
        if evs:   # older logs stored the eval's own step count under env_steps: map evals to training steps by update
            by_u = {u["updates"]: u["env_steps"] for u in ups}
            ax[1].plot([by_u.get(e["updates"], e["env_steps"]) for e in evs], [e["success"] for e in evs], marker="o", label=name)
        res = json.loads((d / "results.json").read_text()) if (d / "results.json").exists() else {}
        if "initial_val" in res:
            ax[1].scatter([0], [res["initial_val"]["success"]], marker="x", color="k")
        if "final_val" in res:   # final evaluation of the saved checkpoint, with its 95% bootstrap CI
            fv = res["final_val"]
            ax[1].errorbar([ups[-1]["env_steps"]], [fv["success"]], yerr=[[fv["success"] - fv["ci95"][0]], [fv["ci95"][1] - fv["success"]]],
                           marker="s", capsize=3, color=ax[1].lines[-1].get_color() if ax[1].lines else None)
    ax[0].set(xlabel="environment steps", ylabel="train success (last 400 episodes)", title="Training (MiniWoB TRAIN families)")
    ax[1].set(xlabel="environment steps", ylabel="outer-validation success (greedy)",
              title="Validation, 88 tasks (x untrained, ■ final ±95% CI)")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def search_summary(root: Path) -> dict:
    pop = json.loads((root / "population.json").read_text())
    cands = pop["candidates"]
    seed = cands[0]
    out = {"name": root.name, "controller": pop["meta"]["config"]["controller"], "n_candidates": len(cands),
           "n_failed": sum(c["status"] == "failed" for c in cands), "budget": pop["budget"], "champion": pop.get("champion"),
           "selection": pop["meta"].get("selection"), "candidates": [], "best_so_far": []}
    best, used = -1.0, 0.0
    for c in cands:
        r0 = c["results"].get("0")
        r1 = c["results"].get("1")
        out["candidates"].append({"cid": c["cid"], "parent": c["parent"], "op": c["op"], "mutation": c["mutation"][:140],
                                  "status": c["status"], "failure": (c.get("failure") or "")[:120],
                                  "params": (c.get("validation", {}).get("params", {}) or {}).get("total"),
                                  "val_r0": r0["val"]["success"] if r0 else None, "val_r1": r1["val"]["success"] if r1 else None,
                                  "gpu_h": sum(r["compute"]["gpu_hours"] for r in c["results"].values())})
        for r in c["results"].values():
            used += r["compute"]["gpu_hours"]
            best = max(best, r["val"]["success"])
            out["best_so_far"].append((used, best))
    hyp = pop["hypotheses"]
    ev = [h for h in hyp if h.get("outcome") == "evaluated"]
    out["hypotheses"] = {"n": len(hyp), "evaluated": len(ev), "improved": sum(h["delta"] > 0 for h in ev),
                         "mean_delta": float(np.mean([h["delta"] for h in ev])) if ev else float("nan"),
                         "by_op": {}}
    for h in hyp:
        d = out["hypotheses"]["by_op"].setdefault(h["op"], {"n": 0, "evaluated": 0, "deltas": []})
        d["n"] += 1
        if h.get("outcome") == "evaluated":
            d["evaluated"] += 1
            d["deltas"].append(round(h["delta"], 3))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/cam2")
    ap.add_argument("--fig", default="docs/figures")
    ap.add_argument("--out", default="runs/cam2/report.json")
    a = ap.parse_args(argv)
    R = Path(a.runs)
    fig = Path(a.fig)
    fig.mkdir(parents=True, exist_ok=True)
    rep: dict = {}
    runs = {n: R / n for n in ("main_s0", "ablate_nowm_s0", "appworld_s0") if (R / n).exists()}
    print("\n".join(run_table(runs)))
    mw = {k: v for k, v in runs.items() if k != "appworld_s0"}
    if mw:
        curves_figure(mw, fig / "cam2_learning_curves.png")
    for n, d in runs.items():
        ups, evs = load_log(d)
        rep[n] = {"results": json.loads((d / "results.json").read_text()) if (d / "results.json").exists() else None,
                  "curve": [{k: u.get(k) for k in ("env_steps", "train_success", "train_success_recent", "update_rel_norm",
                                                    "params_changed", "replay_mismatch_max", "wm_r2_next_f", "wm_r2_reward",
                                                    "wm_reliability", "approx_kl", "llm_gen_tokens", "llm_encode_tokens")}
                            for u in ups], "evals": evs}
        for e in (rep[n]["results"] or {}).values():
            if isinstance(e, dict):
                e.pop("episodes", None)
    sroot = R / "search"
    if sroot.exists():
        rep["search"] = {}
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        f, ax = plt.subplots(figsize=(6, 3.8))
        for d in sorted(sroot.iterdir()):
            if (d / "population.json").exists():
                s = search_summary(d)
                rep["search"][d.name] = s
                xs, ys = zip(*s["best_so_far"]) if s["best_so_far"] else ([], [])
                ax.step(xs, ys, where="post", label=d.name)
                print(f"\n### search {d.name} ({s['controller']}): {s['n_candidates']} candidates, {s['n_failed']} failed, "
                      f"champion {s['champion']}")
                print("| cid | parent | op | change | params | val r0 | val r1 | GPU-h | status |")
                print("|---|---|---|---|---|---|---|---|---|")
                for c in s["candidates"]:
                    v0 = "—" if c["val_r0"] is None else f"{c['val_r0']:.3f}"
                    v1 = "—" if c["val_r1"] is None else f"{c['val_r1']:.3f}"
                    pm = "—" if not c["params"] else f"{c['params'] / 1e6:.1f}M"
                    print(f"| {c['cid']} | {c['parent'] or '—'} | {c['op']} | {c['mutation'].replace('|', '/')} | {pm} | {v0} | {v1} | "
                          f"{c['gpu_h']:.3f} | {c['status']}{(': ' + c['failure']) if c['failure'] else ''} |")
                print("hypotheses:", json.dumps(s["hypotheses"]))
                print("selection:", json.dumps(s["selection"], default=str)[:1500])
        ax.set(xlabel="GPU-hours consumed by the search", ylabel="best outer-validation success so far",
               title="Search efficiency by controller")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        f.tight_layout()
        f.savefig(fig / "cam2_search_best_so_far.png", dpi=130)
        plt.close(f)
    ev = R / "eval"
    if ev.exists():
        rep["eval"] = {}
        for p in sorted(ev.glob("*.json")):
            d = json.loads(p.read_text())
            rep["eval"][p.stem] = {k: {kk: vv for kk, vv in v.items() if kk not in ("episodes", "samples")}
                                   for k, v in d.get("results", {}).items()}
            for k, v in d.get("results", {}).items():
                print(f"eval {p.stem} / {k}: {fmt_ci(v)} invalid={v.get('invalid_rate', float('nan')):.3f} "
                      f"tokens gen={v.get('llm_gen_tokens')} prompt={v.get('llm_prompt_tokens', v.get('llm_encode_tokens'))}")
    rt = R / "retention" / "results.json"
    if rt.exists():
        rep["retention"] = json.loads(rt.read_text())
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
