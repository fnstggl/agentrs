"""Summaries of the PS evidence files (recall diagnostic, debug ablation, full-size paired runs, causal logs,
live-checkpoint compatibility) as one JSON document.

    python ps_summarize.py --recall DIR[,DIR] --abl DIR --paired DIR --compat FILE --out summary.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics as st
from typing import Any, Dict, List


def _mean(xs: List[float]) -> Any:
    xs = [x for x in xs if x is not None and isinstance(x, (int, float)) and math.isfinite(x)]
    return {"mean": sum(xs) / len(xs), "n": len(xs), "sd": st.pstdev(xs) if len(xs) > 1 else 0.0} if xs else None


def recall(dirs: List[str]) -> Dict[str, Any]:
    by: Dict[str, List[Dict[str, Any]]] = {}
    for d in dirs:
        for f in glob.glob(os.path.join(d, "recall.json")):
            for r in json.load(open(f))["results"]:
                by.setdefault(r["cond"], []).append(r)
    out = {}
    for c, rs in by.items():
        lags = sorted({int(k) for r in rs for k in r["acc_by_lag"]})
        out[c] = {"seeds": [r["seed"] for r in rs],
                  "acc_in_window": _mean([r["acc_in_window"] for r in rs]),
                  "acc_beyond_window": _mean([r["acc_beyond_window"] for r in rs]),
                  "nll_beyond_window": _mean([r["nll_beyond_window"] for r in rs]),
                  "acc_by_lag": {lag: _mean([r["acc_by_lag"].get(str(lag), r["acc_by_lag"].get(lag)) for r in rs])
                                 for lag in lags},
                  "per_seed_beyond": [round(r["acc_beyond_window"], 3) for r in rs],
                  "rho_final": [r.get("rho") for r in rs]}
    return out


def abl(d: str) -> Dict[str, Any]:
    runs = {}
    for f in sorted(glob.glob(os.path.join(d, "*_s*.json"))):
        r = json.load(open(f))
        runs[os.path.basename(f)[:-5]] = r
    out: Dict[str, Any] = {"runs": {}}
    for name, r in runs.items():
        lts = []
        for lt in r["lifetimes"]:
            nll = lt.get("nll_final") or lt.get("nll") or {}
            lts.append({"index": lt["index"], "family": lt["family"], "progress": lt["progress"],
                        "decisions": lt["decisions"], "nll_all": (nll.get("all") or {}).get("mean"),
                        "n_scored": (nll.get("all") or {}).get("n"),
                        "nll_goal": (nll.get("goal_relevant") or {}).get("mean"),
                        "causal": lt.get("causal"), "rho_mean": (lt.get("persistent_state") or {}).get("rho_mean"),
                        "positions": (lt.get("persistent_state") or {}).get("positions")})
        out["runs"][name] = {"partial": r.get("partial", False), "lifetimes": lts,
                             "retention": r.get("retention_first_phase_records")}
    # paired differences on / off per seed and lifetime index
    pairs = []
    for name in out["runs"]:
        if not name.startswith("on_"):
            continue
        off = out["runs"].get("off_" + name[3:])
        if off is None:
            continue
        for a, b in zip(out["runs"][name]["lifetimes"], off["lifetimes"]):
            pairs.append({"seed": name[3:], "index": a["index"], "family": a["family"],
                          "d_progress": a["progress"] - b["progress"],
                          "d_nll": (a["nll_all"] - b["nll_all"]) if a["nll_all"] is not None and b["nll_all"] is not None else None,
                          "on_nll": a["nll_all"], "off_nll": b["nll_all"]})
    out["paired"] = pairs
    out["mean_d_progress"] = _mean([p["d_progress"] for p in pairs])
    out["mean_d_nll"] = _mean([p["d_nll"] for p in pairs])
    return out


def paired(d: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    f = os.path.join(d, "paired.json")
    if os.path.exists(f):
        r = json.load(open(f))
        for arm, obs in r.get("arms", {}).items():
            out[arm] = {i: {k: ob.get(k) for k in ("family", "problem", "verified_status", "verified_value",
                                                    "decisions", "mean_prediction_nll", "n_predictions_resolved",
                                                    "executable_fraction", "backbone_chosen_fraction",
                                                    "distinct_strategies", "arm_wall_s", "persistent_state",
                                                    "restore_report_persistent")}
                        for i, ob in obs.items()}
    c = os.path.join(d, "ps_causal_on.jsonl")
    if os.path.exists(c):
        rows = [json.loads(l) for l in open(c) if l.strip()]
        ok = [x for x in rows if "tv" in x]
        out["causal_on"] = {"n": len(ok), "errors": len(rows) - len(ok),
                            "tv": _mean([x["tv"] for x in ok]), "tv_max": max((x["tv"] for x in ok), default=None),
                            "argmax_changed": sum(1 for x in ok if x["argmax_changed"]),
                            "rel_db": _mean([x["rel_db"] for x in ok]),
                            "d_p_success": _mean([abs(x["p_success_on"] - x["p_success_off"]) for x in ok
                                                  if x.get("p_success_on") is not None]),
                            "by_position": [(x["position"], round(x["tv"], 4), round(x["rel_db"], 4)) for x in ok]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recall", default="")
    ap.add_argument("--abl", default="")
    ap.add_argument("--paired", default="")
    ap.add_argument("--compat", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out: Dict[str, Any] = {}
    if a.recall:
        out["recall"] = recall(a.recall.split(","))
    if a.abl:
        out["ablation_debug"] = abl(a.abl)
    if a.paired:
        out["full_size"] = paired(a.paired)
    if a.compat and os.path.exists(a.compat):
        out["compat_live"] = json.load(open(a.compat))
    json.dump(out, open(a.out, "w"), indent=1, default=str)
    print(json.dumps(out, indent=1, default=str)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
