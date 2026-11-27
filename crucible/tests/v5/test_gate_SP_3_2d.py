"""SP-3.2d: class-4 semantic agents are not language-model wrappers around predetermined scripts (MASTER_SPEC 3.2
class 4): they have their own needs and goals and meaningful opportunities to act; the smallest model meeting each
capability is used (routing); checkpoint availability, licences and tool-use capability are verified.

Measured on two runs of the same civilisation (`crucible.civ`, development population with eight semantic
participants), same seed and days: with the shared models (`brain="lm"`) and with every model decision replaced by
a uniform random one (`brain="ablated"`, `crucible.semantic.generalists.brain.AblatedBrain`):

* scripted paths - every event acted as a participant must lie inside a call its brain chose (`verify.provenance`);
  any other is a scripted path (must be 0, in both runs);
* own goals - participants that wrote at least one goal of their own (no goal is assigned anywhere);
* ablation - with the models replaced, behaviour degrades measurably: the share of calls that are consequential
  (succeeded, changed the world, and produced an event another principal can see, directly or through a job the
  call scheduled; content judged by the evaluator not to be junk; a repeated call with the same content once:
  `generalists.verify`; Fisher exact test, p <= 0.01) and
  the number of independently verified productive results (publications with at least three independent viewers)
  are both lower. The share of successful calls and of goals the participant itself judged achieved are reported,
  not asserted: uniform random choices spread over argument-free read tools (the third run: 31 distinct tools
  ablated vs 5 with the models), whose calls succeed whatever the caller understands, and a random "finish" ends
  goals as readily as a judged one;
* routing - every capability is routed to the smallest model effective on held-out templates in the committed
  SP-3.2b-derived decision (`crucible/semantic/data/routing.json`), re-derived here from its own measurements;
* licences and tool use - for each benchmarked checkpoint: present locally, licence from its model card, tool use
  verified by running it (`routing.tool_use`).

    PYTHONPATH=src python -m crucible.scale.slots --cpu 1 --gpu-mb 6000 --ram-mb 16000 --timeout 10800 \
        --label "D: SP-3.2d" -- python -m pytest -q tests/v5/test_gate_SP_3_2d.py
"""
import json
import os

import pytest

from crucible.acceptance.v5.evidence import record

DAYS = float(os.environ.get("CRUCIBLE_SP32D_DAYS", "2"))
MODELS = ("HuggingFaceTB/SmolLM2-135M-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct", "Qwen/Qwen3-0.6B",
          "Qwen/Qwen3-1.7B")

pytestmark = [pytest.mark.slow, pytest.mark.timeout(10800)]


def _run(brain: str):
    from crucible.semantic import generalists as G
    from crucible.semantic.generalists import verify
    w = G.civ_world("balanced", 1, population="development", config={"brain": brain},
                    overrides={"module_config": {"pop": {"classes": {"semantic": 8}}},
                               "params": {"log_retain": 10_000_000}})
    w.advance_time(w.now + int(DAYS * 86400))
    out = {"summary": verify.summary(w), "actions": verify.actions(w), "rows": G.members(w)}
    w.close()
    return out


def test_SP_3_2d_not_scripted_own_goals_routing_and_checkpoints():
    import crucible
    from scipy.stats import fisher_exact
    from crucible.semantic import routing
    crucible.load_all_modules()
    lm, ab = _run("lm"), _run("ablated")
    s, a = lm["summary"], ab["summary"]

    def share(x):
        return round(x["ok_calls"] / max(1, x["calls"]), 4)
    p_ok = fisher_exact([[s["ok_calls"], s["calls"] - s["ok_calls"]], [a["ok_calls"], a["calls"] - a["ok_calls"]]],
                        alternative="greater")[1]
    p_cons = fisher_exact([[s["consequential"], s["calls"] - s["consequential"]],
                           [a["consequential"], a["calls"] - a["consequential"]]], alternative="greater")[1]
    dec = routing.decision()
    assert dec is not None, "no committed routing decision (crucible/semantic/data/routing.json)"
    rederived = {cap: (r["model"], int(r.get("think_tokens", 0))) for cap, r in
                 routing.derive(dec["report"], dec.get("in_situ"), dec.get("in_situ_confirm")).items()}
    routed = {cap: (routing.route(cap), routing.think_tokens(cap)) for cap in routing.capabilities()}
    checks = [routing.tool_use(mdl) for mdl in MODELS]
    cards = [routing.model_card(mdl) for mdl in MODELS]
    verified = [c for c, t in zip(cards, checks) if c["license"] and t["verified"]]
    m = {"scripted_semantic_agent_paths": s["provenance"]["outside_model_chosen_calls"] +
         a["provenance"]["outside_model_chosen_calls"],
         "events_as_participants": s["provenance"]["events_as_participants"],
         "lm_outside_model_chosen_calls": s["provenance"]["outside_model_chosen_calls"],
         "ablated_outside_model_chosen_calls": a["provenance"]["outside_model_chosen_calls"],
         "outside_by_type": json.dumps({"lm": s["provenance"]["outside_by_type"],
                                        "ablated": a["provenance"]["outside_by_type"]}),
         "scheduled_consequences_of_chosen_calls": s["provenance"]["scheduled_consequences_of_chosen_calls"] +
         a["provenance"]["scheduled_consequences_of_chosen_calls"],
         "semantic_agents_with_own_goals": s["participants_with_own_goals"],
         "capabilities_routed_to_smallest_model": routed == rederived,
         "models_with_licence_and_tool_use_verified": len(verified),
         "lm_ok_call_share": share(s), "ablated_ok_call_share": share(a), "ok_share_pvalue": float(f"{p_ok:.3g}"),
         "lm_consequential": s["consequential"], "ablated_consequential": a["consequential"],
         "lm_consequential_raw": s["consequential_raw"], "ablated_consequential_raw": a["consequential_raw"],
         "lm_consequential_junk": s["consequential_junk"], "ablated_consequential_junk": a["consequential_junk"],
         "consequential_pvalue": float(f"{p_cons:.3g}"),
         "lm_consequential_share": round(s["consequential"] / max(1, s["calls"]), 4),
         "ablated_consequential_share": round(a["consequential"] / max(1, a["calls"]), 4),
         "lm_calls": s["calls"], "ablated_calls": a["calls"],
         "lm_goals_done": s["goals_done"], "ablated_goals_done": a["goals_done"],
         "lm_tools_used": s["tools_used"], "ablated_tools_used": a["tools_used"],
         "lm_productive": s["productive_verified"], "ablated_productive": a["productive_verified"],
         "routes": json.dumps(routed), "licences": json.dumps({c["model"]: c["license"] for c in cards}),
         "tool_use": json.dumps({t["model"]: [t["native_tool_template"], t["native_calls_correct"],
                                              t["choice_correct"]] for t in checks})}
    record("SP-3.2d", m, command="PYTHONPATH=src python -m pytest -q tests/v5/test_gate_SP_3_2d.py",
           notes="Same civilisation and seed with the shared models and with every model decision replaced by a "
                 "uniform random one; scripted paths are events acted as a participant outside model-chosen calls.")
    assert m["scripted_semantic_agent_paths"] == 0 and m["events_as_participants"] > 0
    assert m["semantic_agents_with_own_goals"] >= 8
    assert m["capabilities_routed_to_smallest_model"] is True
    assert m["models_with_licence_and_tool_use_verified"] >= 2
    assert m["lm_consequential_share"] > m["ablated_consequential_share"] and m["consequential_pvalue"] <= 0.01
    assert m["lm_productive"] > m["ablated_productive"]
