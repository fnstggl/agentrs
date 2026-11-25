"""Compact summary of capability result files (workstream Y).

    python -m superlearner.cam_unified.capabilities.report <results_dir> [--out summary.json]

Reads every `<name>.json` written by `capabilities.run` in the directory and extracts the headline numbers of each
capability into one dict (all values as measured; nothing is thresholded here)."""
from __future__ import annotations

import argparse
import glob
import json
import os


def _get(d, *path, default=None):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def summarise_file(res: dict) -> dict:
    st = res.get("settings", {})
    if "summary" in res and "trained" in res and "untrained" in res and isinstance(res["summary"], dict):
        return {"capability": "beliefs", "kind": st.get("kind", "gaussian"),
                "component_kwargs": st.get("component_kwargs"), "summary": res["summary"],
                "final_losses": _get(res, "train", "final_losses")}
    if "forced_iters" in res:
        return {"capability": "reasoning", "condition": st.get("condition"), "hops": st.get("hops"),
                "n_colours": st.get("n_colours"), "train_rounds": st.get("train_rounds"), "chance": res.get("chance"),
                "train_return_first10": _get(res, "train", "return_first10"),
                "train_return_last10": _get(res, "train", "return_last10"),
                "untrained": res.get("untrained"), "trained": res.get("trained"),
                "forced_iters": {j: {"acc": v.get("acc"), "changed_vs_trained": v.get("changed_vs_trained")}
                                 for j, v in res["forced_iters"].items()},
                "sensitivity_untrained": res.get("sensitivity_untrained"),
                "sensitivity_trained": res.get("sensitivity_trained"),
                "policy_grad_norms_untrained": res.get("policy_grad_norms_untrained"),
                "policy_grad_norms_trained": res.get("policy_grad_norms_trained")}
    if "lp_by_action" in res:
        ev = {cp: {"p_test_first": _get(e, "informative", "p_test_first"),
                   "greedy_test_first": _get(e, "informative", "greedy_test_first"),
                   "return": _get(e, "informative", "return"),
                   "uninformative_p_test_first": _get(e, "uninformative", "p_test_first")}
              for cp, e in res["evals"].items()}
        return {"capability": "exploration", "condition": st.get("condition"), "lp_coef": res.get("lp_coef"),
                "component_kwargs": st.get("component_kwargs"), "lp_by_action": res["lp_by_action"], "evals": ev,
                "train_return_per_row_per_round": [t["return_per_row_per_round"] for t in res.get("train", [])]}
    if "evals" in res and "value_bounds" in res:
        ev = {}
        for cp, e in res["evals"].items():
            ev[cp] = {m: {"frac": _get(e, m, "frac"), "imagined_value": _get(e, m, "imagined_value"),
                          "candidates_include_test": _get(e, m, "candidates_include_test"),
                          "return": _get(e, m, "return"), "changed_vs_policy": _get(e, m, "changed_vs_policy")}
                      for m in ("policy", "planner", "planner_frozen_beliefs", "planner_h1") if m in e}
            ev[cp]["after_forced_test"] = e.get("after_forced_test")
            ev[cp]["a07_action_conditioned"] = _get(e, "policy", "a07")
        return {"capability": "imagination", "exploring_starts": st.get("exploring_starts"),
                "component_kwargs": st.get("component_kwargs"), "value_bounds": res["value_bounds"], "evals": ev}
    if "retriever_has_loss" in res:
        return {"capability": "memory", "condition": st.get("condition"), "delay": st.get("delay"),
                "untrained": res.get("untrained"), "trained": res.get("trained"),
                "return_per_episode_first3": _get(res, "train", "return_per_episode_first3"),
                "return_per_episode_last3": _get(res, "train", "return_per_episode_last3"),
                "retriever_has_loss": res["retriever_has_loss"],
                "retriever_has_memory_tokens": res["retriever_has_memory_tokens"]}
    if "worlds" in res:
        out = {"capability": "causal", "worlds": {}}
        for w, r in res["worlds"].items():
            out["worlds"][w] = {k: v for k, v in r.items() if k != "components"}
        return out
    return {"capability": "unknown"}


def summarise_dir(path: str) -> dict:
    out = {}
    for fn in sorted(glob.glob(os.path.join(path, "*.json"))):
        name = os.path.splitext(os.path.basename(fn))[0]
        if name == "summary":
            continue
        try:
            res = json.load(open(fn))
        except (OSError, json.JSONDecodeError):
            continue
        s = summarise_file(res)
        s["command"] = res.get("command")
        s["wall_s"] = res.get("wall_s_total", res.get("wall_s"))
        out[name] = s
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    s = summarise_dir(a.dir)
    out = a.out or os.path.join(a.dir, "summary.json")
    with open(out, "w") as f:
        json.dump(s, f, indent=1, default=str)
    print(f"wrote {out} ({len(s)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
