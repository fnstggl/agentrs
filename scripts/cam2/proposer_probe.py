"""Probe the LLM architecture proposer offline: how often do its proposals parse, stay inside the search space,
pass every admission check, and differ from each other? (No training: whether proposals *improve* candidates is
measured only inside a search, where llm.propose competes with the other operators.)

  PYTHONPATH=src python scripts/cam2/proposer_probe.py --search runs/cam2/search/adaptive --n 24 --out runs/cam2/eval/proposer_probe.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from agentrs.cam2.backbone import load_llm  # noqa: E402
from agentrs.cam2.search.population import Population  # noqa: E402
from agentrs.cam2.search.proposer import LLMProposer  # noqa: E402
from agentrs.cam2.search.spec import SpecError, spec_hash, validate  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--temperature", type=float, nargs="*", default=[0.0, 0.7])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    pop = Population.load(a.search)            # read-only use: never saved
    llm = load_llm(pop.meta["config"]["llm"], device="cuda")
    parents = [c for c in pop.cands.values() if c.rung >= 0]
    res = {"search": a.search, "parents": [c.cid for c in parents], "by_temperature": {}}
    for T in a.temperature:
        prop = LLMProposer(llm, temperature=T)
        rng = random.Random(0)
        rows = []
        for i in range(a.n):
            parent = parents[i % len(parents)]
            row = {"parent": parent.cid}
            try:
                spec, desc, raw = prop.propose(parent.spec, pop, rng)
                row.update(stage="parsed+schema", desc=desc[:200])
                validate(spec, llm.pooled_dim, llm.hidden, device="cuda")
                row.update(stage="admitted", spec=spec_hash(spec))
            except SpecError as e:
                row.update(error=str(e)[:200])
            rows.append(row)
        adm = [r for r in rows if r.get("stage") == "admitted"]
        res["by_temperature"][str(T)] = {
            "n": len(rows), "admitted": len(adm), "admit_rate": len(adm) / len(rows),
            "distinct_admitted_specs": len({r["spec"] for r in adm}), "tokens": prop.tokens, "rows": rows}
        print(f"T={T}: admitted {len(adm)}/{len(rows)}, distinct {len({r['spec'] for r in adm})}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
