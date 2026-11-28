"""PS full-size comparison: ``eval.oe_runner sequence`` (the substantive protocol: Qwen3-4B + 50m core, open-ended
generation on, one objective per GPU-lock hold, the agent rebuilt and restored from its checkpoint between holds)
with the persistent core state on or off. Nothing else differs between the two arms.

Two additions, both observational (they change no decision):
* the ``persistent_state`` configuration section (``enabled``, ``unroll``) is added to the substantive configuration;
* with the chain on, every ``select`` first measures the causal effect of the carried state on the same decision:
  total variation between the trainable policy's distributions over the offered candidates with the carried and with
  the fresh state, the relative change of the belief b and of the evaluator's success probability (same weights,
  same context; the chain is switched off only for the counterfactual computation).

    python ps_oe.py --persist on|off --unroll 2 -- sequence --condition E --seed 0 --plan SW:rle,MECH:7 \
        --decisions 16 --out R.json --root ROOT --hold-s 840
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, List

import torch


def main(argv: List[str] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cut = argv.index("--") if "--" in argv else len(argv)
    ap = argparse.ArgumentParser()
    ap.add_argument("--persist", choices=("on", "off"), required=True)
    ap.add_argument("--unroll", type=int, default=2)
    a = ap.parse_args(argv[:cut])
    rest = argv[cut + 1:]
    from cam_live.eval import oe_runner
    from cam_live.agent import CamLiveAgent

    base_cfg = oe_runner.substantive_config

    def substantive_config(condition: str, seed: int, *, real: bool = True) -> Any:
        cfg = base_cfg(condition, seed, real=real)
        cfg.sections["persistent_state"] = {"enabled": a.persist == "on", "unroll": int(a.unroll)}
        return cfg

    base_build = oe_runner._build
    log_path = os.path.abspath(os.path.join(os.path.dirname(_out(rest)) or ".", f"ps_causal_{a.persist}.jsonl"))

    def build(condition: str, cfg: Any, run_dir: str, clock: Any) -> Any:
        agent = base_build(condition, cfg, run_dir, clock)
        if agent.persist is not None:
            orig = agent.select

            def select(evaluations: Any) -> Any:
                try:
                    rec = measure(agent, evaluations)
                    if rec is not None:
                        with open(log_path, "a") as fh:
                            fh.write(json.dumps(rec) + "\n")
                except Exception as e:  # noqa: BLE001 — a measurement; never changes the decision
                    with open(log_path, "a") as fh:
                        fh.write(json.dumps({"error": f"{type(e).__name__}: {e}"}) + "\n")
                return orig(evaluations)
            agent.select = select
        return agent

    oe_runner.substantive_config = substantive_config
    oe_runner._build = build
    return oe_runner.main(rest)


def _out(rest: List[str]) -> str:
    return rest[rest.index("--out") + 1] if "--out" in rest else "."


def measure(agent: Any, evaluations: Any) -> Any:
    cands = [e.action for e in evaluations]
    if len(cands) < 2:
        return None
    refs = agent.pending_eval.get("refs") or agent._window()
    items = agent._items(refs)
    with torch.no_grad():
        p_on = agent.policy_distribution(items, cands)
        b_on = agent._belief(items)
        with agent._live():                                 # decision-time resolution: the carried state
            bv_on = agent.predictor.belief_value(items, agent._goal_text()) if agent.task is not None else None
        agent.persist.settings.enabled = False
        try:
            p_off = agent.policy_distribution(items, cands)
            b_off = agent._belief(items)
            bv_off = agent.predictor.belief_value(items, agent._goal_text()) if agent.task is not None else None
        finally:
            agent.persist.settings.enabled = True
    st = agent.persist.status()
    return {"decision": agent.n_decisions, "objective": agent.task.objective_id if agent.task else "",
            "position": st["positions"], "n_candidates": len(cands),
            "tv": float(0.5 * (p_on - p_off).abs().sum()),
            "argmax_changed": bool(int(p_on.argmax()) != int(p_off.argmax())),
            "rel_db": float((b_on - b_off).norm() / b_off.norm().clamp_min(1e-12)),
            "p_success_on": None if bv_on is None else bv_on["p_success"],
            "p_success_off": None if bv_off is None else bv_off["p_success"],
            "rho_mean": st["rho_mean"]}


if __name__ == "__main__":
    sys.exit(main())
