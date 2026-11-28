"""Summarise SG's real-configuration runs (sg_runner.py result files) into one table.

    python analyse_real.py runs/real_skills.json runs/real_primitive.json ... --out summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys


def selection_mass(root: str):
    """Per decision: probability mass the agent's selection put on skill.run / skill.induce / skill.discover
    candidates, read from the agent's own experience store (decision records)."""
    path = os.path.join(root, "agent", "learner", "experience.sqlite")
    if not os.path.exists(path):
        return None
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    out = []
    try:
        for (raw,) in con.execute("select static_body from experiences order by seq"):
            try:
                d = json.loads(raw)
            except (TypeError, ValueError):
                continue
            dec = d.get("decision") or {}
            cands = dec.get("candidates") or []
            probs = dec.get("probabilities") or []
            if not cands or len(probs) != len(cands):
                continue
            ops = [c.get("action", {}).get("operation", "") for c in cands]
            mass = {k: sum(p for p, o in zip(probs, ops) if o == k) for k in ("skill.run", "skill.induce", "skill.discover")}
            out.append({"objective_id": dec.get("objective_id", d.get("objective_id", "")),
                        "chosen": (dec.get("chosen") or {}).get("operation"), "n_candidates": len(cands),
                        "n_skill_run": ops.count("skill.run"), **{f"p_{k}": v for k, v in mass.items()}})
    finally:
        con.close()
    return out


def summarise(path: str):
    d = json.load(open(path))
    rows = []
    for o in d.get("objectives", []):
        det = o.get("decisions_detail", [])
        lib = o.get("library") or {}
        rows.append({"index": o["index"], "family": o["family"], "problem": o["problem"],
                     "verified": o["verified_status"], "achieved": o["achieved"], "decisions": o["decisions"],
                     "reason": o.get("reason", "")[:80], "wall_s": round(o["wall_s"], 1),
                     "primitive_ops_total": o.get("primitive_ops_total"),
                     "chosen_ops": [x["operation"] for x in det],
                     "chosen_status": [x["status"] for x in det],
                     "skill_runs": o.get("skill_runs"), "skill_runs_ok": o.get("skill_runs_ok"),
                     "induce": [(x["status"], x["detail"][:160]) for x in det if x["operation"] == "skill.induce"],
                     "discover": [(x["status"], x["detail"][:160]) for x in det if x["operation"] == "skill.discover"],
                     "library": [{k: s.get(k) for k in ("skill_id", "version", "status", "operations", "params",
                                                        "runs", "successes", "method", "composed_of")}
                                 for s in lib.get("skills", [])] if isinstance(lib.get("skills"), list) else lib.get("skills"),
                     "rejected": lib.get("rejected"), "rejected_reasons": lib.get("rejected_reasons"),
                     "goal_free": {k: (o.get("goal_free") or {}).get(k) for k in ("goals", "resolved", "achieved")},
                     "transfer_points": ((o.get("goal_free") or {}).get("transfer") or {}).get("points"),
                     "transfer_by_key": ((o.get("goal_free") or {}).get("transfer") or {}).get("by_key"),
                     "counters": {k: v for k, v in (o.get("agent_counters") or {}).items()
                                  if k in ("success_traces", "verified_progress", "transfer_evidence", "decisions")},
                     "backbone_chosen_fraction": o.get("backbone_chosen_fraction"),
                     "executable_fraction": o.get("executable_fraction")})
    return {"file": path, "plan": d.get("plan"), "skills": d.get("skills"), "real": d.get("real"),
            "hold_s": d.get("hold_s"), "lock_wait_s": d.get("lock_wait_s"), "gpu_peak_gb": d.get("gpu_peak_gb"),
            "not_started": d.get("not_started"), "objectives": rows,
            "selection": selection_mass(os.path.join(os.path.dirname(path), os.path.basename(path)[:-5]))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = [summarise(f) for f in a.files if os.path.exists(f)]
    json.dump(out, open(a.out, "w"), indent=1, default=str)
    for s in out:
        print("==", os.path.basename(s["file"]), s["plan"], "skills" if s["skills"] else "primitive",
              "hold", s["hold_s"], "not_started", s["not_started"])
        for r in s["objectives"]:
            print(f"  {r['index']} {r['family']}:{r['problem']} {r['verified']} decisions {r['decisions']} "
                  f"ops {r['primitive_ops_total']} skill_runs {r['skill_runs']}/{r['skill_runs_ok']} wall {r['wall_s']} "
                  f"induce {r['induce']} discover {r['discover']}")
            print(f"     chosen {list(zip(r['chosen_ops'], r['chosen_status']))}")
            print(f"     library {r['library']} rejected {r['rejected']} {r['rejected_reasons']}")
            print(f"     goal_free {r['goal_free']} transfer {r['transfer_points']} counters {r['counters']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
