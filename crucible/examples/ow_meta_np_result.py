"""The verdict on the `meta_np` positive control, from the recorded learner tables and the criteria declared in advance
(`docs/experiments/ow_meta_np_log.md`). Writes `docs/experiments/ow_meta_np_result.json`.

    python examples/ow_meta_np_result.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from crucible.research import stats as S  # noqa: E402

OUT = os.path.join(HERE, "..", "docs", "experiments")


def _load(name: str) -> dict:
    with open(os.path.join(OUT, name)) as fh:
        return json.load(fh)


def _seed_mean(rows: dict, base: str) -> np.ndarray:
    return np.mean([rows[f"{base} trained s{s}"]["scores"] for s in range(3)], axis=0)


def _cmp(a, b) -> dict:
    d = S.paired_diff(list(map(float, a)), list(map(float, b)))
    d["excludes_zero_above"] = bool(d["lo"] > 0)
    d["excludes_zero_below"] = bool(d["hi"] < 0)
    return d


def _analyse(res: dict) -> dict:
    r = res["rows"]
    t = _seed_mean(r, "meta_np")
    refs = {"probe_exploit": np.array(r["probe_exploit"]["scores"]),
            "meta_np untrained": np.array(r["meta_np untrained"]["scores"]),
            "model_based untrained": np.array(r["model_based untrained"]["scores"]),
            "model_based trained": _seed_mean(r, "model_based")}
    fam = np.array(r["probe_exploit"]["families"])
    untrained = refs["meta_np untrained"]
    return {"n_tasks": len(t), "meta_np_trained": float(t.mean()),
            "per_seed": [float(np.mean(r[f"meta_np trained s{s}"]["scores"])) for s in range(3)],
            "reference_means": {k: float(v.mean()) for k, v in refs.items()},
            "trained_minus": {k: _cmp(t, v) for k, v in refs.items()},
            "exploratory_trained_minus_untrained": {
                "note": "chosen after seeing the results; not part of the verdict",
                "by_engine": {e: _cmp(t[np.char.startswith(fam, e)], untrained[np.char.startswith(fam, e)])
                              for e in sorted({f.split("_")[0] for f in fam})},
                "by_family": {f: {"trained": float(t[fam == f].mean()), "untrained": float(untrained[fam == f].mean())}
                              for f in sorted(set(fam))}}}


def main() -> None:
    val, held = _analyse(_load("ow_learners_table.json")), _analyse(_load("ow_learners_heldout.json"))
    m = val["trained_minus"]
    c1, c2 = m["probe_exploit"]["excludes_zero_above"], m["meta_np untrained"]["excludes_zero_above"]
    c3 = m["model_based untrained"]["excludes_zero_above"] and m["model_based trained"]["excludes_zero_above"]
    res = {"criteria": "docs/experiments/ow_meta_np_log.md (declared before any validation score was seen)",
           "validation": val, "held_out_families": held,
           "verdict": {"C1_beats_scripted": c1, "C2_beats_own_untrained": c2, "C3_beats_model_based": c3,
                       "positive_control": "passed validation" if c1 and c2 else "FAILED",
                       "sealed_set_opened": False,
                       "reason": "C1 and C2 must both hold on validation before a sealed set is opened"}}
    with open(os.path.join(OUT, "ow_meta_np_result.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res["verdict"], indent=1))
    for tag, a in (("validation", val), ("held-out", held)):
        print(tag, "trained", round(a["meta_np_trained"], 3), {k: (round(v["mean_diff"], 3), round(v["lo"], 3), round(v["hi"], 3))
                                                              for k, v in a["trained_minus"].items()})
        print("  exploratory by engine", {k: (v["n"], round(v["mean_diff"], 3), round(v["lo"], 3), round(v["hi"], 3))
                                         for k, v in a["exploratory_trained_minus_untrained"]["by_engine"].items()})


if __name__ == "__main__":
    main()
