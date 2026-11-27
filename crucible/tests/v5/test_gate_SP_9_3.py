"""SP-9.3 - Tier 2 domain Content ecosystem (MASTER_SPEC Part 9.3; workstream I-1).

Integrated runs through the learner interface (`base.section_metrics` / `base.run` -> `crucible.experience.make(2,
"content:<config>", seed)`, default reward mode) of `crucible.domains.content_domain`: the reference creator
(LABELLED SCRIPTED, tool-only) on seeds 0 and 1 of the first training configuration over the 10-day horizon, while
the population browses its feeds (consumption from each individual's own interests in the shared space, the items'
intrinsic features, attention and social signals), likes, shares, follows and creates (competing creators). Every
required mechanic is counted by the framework's recorded-event probes; distinct viewers are the exact bitsets of
`social.audience`. Difficulty levels: the objective's targets per level and the population each level's world is
built with (`content_domain.POPULATION_FOR`), measured on the built spec of the highest level (internet profile,
10,000,000 individuals). Controls: the null baseline and keyword-stuffed spam (meaningless text must not reach the
audience that original pieces reach).
"""
import pytest

import crucible

crucible.load_all_modules()
from crucible.acceptance.v5 import checker, evidence  # noqa: E402
from crucible.domains import _t2kit as K  # noqa: E402
from crucible.domains import base  # noqa: E402
from crucible.domains import content_domain as D  # noqa: E402
from crucible.population import profiles as PP  # noqa: E402

CMD = "PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/v5/test_gate_SP_9_3.py"
pytestmark = [pytest.mark.slow, pytest.mark.timeout(5 * 3600)]


def _viewers(run) -> int:
    return int((run["outcome"].get("details") or {}).get("distinct_viewers", 0))


def test_SP_9_3_content_ecosystem_mechanics():
    d = base.part9_domains()["content"]
    assert d is D.DOMAIN
    m, runs = base.section_metrics("content")
    cfg = d.reference_configs[0] if d.reference_configs else d.split("train")[0].name
    null = base.cached_run("content", cfg, 0, "null", None)
    spam = base.cached_run("content", cfg, 0, "spam", None)
    # difficulty levels: public targets and the population of each level's world (built spec, no world created)
    env = base.make_env("content", cfg, 0)
    seat = env.agents()[0]
    levels = {lv: d.make_objective(env.world, seat, lv)["params"]["target_unique_viewers"] for lv in d.difficulties}
    env.close()
    pops = {}
    for lv in d.difficulties:
        sp = base.spec("content", cfg, 0, overrides={"population": D.POPULATION_FOR[lv]})
        pcfg = next(x["config"] for x in sp.modules if x["name"] == "pop")
        pops[lv] = int(PP.resolve(pcfg)["n"])
    top = max(d.difficulties)
    m.update({"difficulty_levels": len(levels), "max_difficulty_unique_viewers": int(levels[top]),
              "max_difficulty_population": pops[top],
              "levels_feasible_by_population": sum(1 for lv in levels if pops[lv] >= levels[lv]),
              "mechanics_exercised_without_seat": null["mechanics"]["exercised"],
              "autonomous_actors_active": null["actors"]["active"],
              "population_activity_decisions_without_seat": null["population"]["activity_decisions"],
              "reference_distinct_viewers_min": min(_viewers(r) for r in runs), "spam_distinct_viewers": _viewers(spam),
              "null_success": bool(null["success"]),
              "invariant_violations": sum(r["invariant_violations"] for r in runs + [null, spam]),
              "reference_script_world_reads": len(base.reads_world(D.reference_script)),
              "process_failures_all_runs": sum(r["proc_failed"] for r in runs + [null, spam])})
    evidence.record("SP-9.3", m, command=CMD, labels=dict(K.SCRIPTED),
                    notes=f"content:{cfg} seeds {[r['seed'] for r in runs]}, {runs[0]['days']} days; LABELLED SCRIPTED "
                          f"reference creator (tool-only). Levels {levels}, population per level {pops}. Mechanic "
                          f"counts {[r['mechanics']['counts'] for r in runs]}; null {null['mechanics']['counts']}; "
                          f"outcomes {[r['outcome'].get('details') for r in runs]}; spam {spam['outcome'].get('details')}")
    assert m["required_mechanics_missing"] == 0, [r["mechanics"]["missing"] for r in runs]
    # domains-framework/3: every evidence function returned recorded events, no two mechanics share evidence
    assert m["unverified_evidence"] == "none" and m["indistinct_mechanics"] == "none", m
    assert m["levels_feasible_by_population"] == len(levels) and m["max_difficulty_population"] >= 10_000_000
    assert m["autonomous_actors_active"] == len(d.actors) and m["population_activity_decisions_without_seat"] > 0
    assert m["process_failures_all_runs"] == 0 and m["invariant_violations"] == 0
    assert m["reference_distinct_viewers_min"] > m["spam_distinct_viewers"] and not m["null_success"]
    assert m["reference_script_world_reads"] == 0          # the generator itself is tool-only too
    assert not checker.unmet("SP-9.3", m), checker.unmet("SP-9.3", m)
